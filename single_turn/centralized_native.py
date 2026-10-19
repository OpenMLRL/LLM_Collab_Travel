"""Native centralized Travel prompt: one planner, two subtask answers.

The concatenated centralized adapter embeds both decentralized role prompts and
then overrides them. This adapter instead writes one planner prompt that states
the trip, scaffold, checklists, duties, budget and both catalogs once. The
planner answers the logistics subtask first and the daily-experience subtask
second, as two separate answers inside one JSON object. A deterministic splitter
turns each answer into the unchanged decentralized role action, so the merger,
reward and evaluator are the same as for the two-agent team.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from comlrl.trainers.centralized import CentralizedComparatorParseError

from single_turn.aggregation import PLAN_FIELDS, ordered_owned_slots, slot_owner
from single_turn.formatting import (
    _format_cents,
    _render_experience_catalog,
    _render_logistics_catalog,
    _render_role_hard_constraint_duty,
    _render_role_slot_checklist,
    _route_scaffold,
    build_role_budget_contract,
)
from single_turn.rewards.reference_evaluator import (
    derive_route_scaffold,
    parse_reference_catalog,
)
from single_turn.structured_generation import (
    FixedSlotJSONLogitsProcessor,
    wrap_formatter_with_chat_template,
)

# One answer per role, in generation order: logistics first, then experience.
NATIVE_SECTION_KEYS: Tuple[str, str] = ("logistics_answer", "experience_answer")
NATIVE_SYSTEM_PROMPT = (
    "You are one travel planner who completes both subtasks of the itinerary "
    "yourself. Follow the user's output contract exactly. Return only the "
    "requested JSON object with one answer per subtask."
)


def native_slots(days: int) -> List[Tuple[int, str]]:
    """All logistics slots, then all experience slots, each in role order."""

    return [*ordered_owned_slots(0, days), *ordered_owned_slots(1, days)]


def _assignment_opening(day: int, field: str) -> str:
    return (
        f'{{"day": {int(day)}, "field": '
        f'{json.dumps(field, ensure_ascii=False)}, "value": "'
    )


def build_native_prefill(days: int) -> str:
    """Fixed assistant prefix through the first logistics value's opening quote."""

    day, field = ordered_owned_slots(0, days)[0]
    return (
        f"{{{json.dumps(NATIVE_SECTION_KEYS[0])}: ["
        f"{_assignment_opening(day, field)}"
    )


class SectionedFixedSlotJSONLogitsProcessor(FixedSlotJSONLogitsProcessor):
    """Force one named answer array per role inside a single JSON object.

    Only the skeleton text between values differs from the role grammar: when
    the next slot starts a new section, the forced suffix closes the current
    answer array and opens the next one. Value tokens, limits and loss masks are
    unchanged.
    """

    def __init__(self, tokenizer: Any, *, slots: Sequence[Tuple[int, str]], **kwargs):
        owners = [
            slot_owner(day, field, max(d for d, _ in slots)) for day, field in slots
        ]
        starts = [0] + [i for i in range(1, len(owners)) if owners[i] != owners[i - 1]]
        if [owners[i] for i in starts] != [0, 1]:
            raise ValueError(
                "Native centralized slots must be all logistics, then all experience."
            )
        self._section_key_at = {
            index: NATIVE_SECTION_KEYS[section]
            for section, index in enumerate(starts)
            if section
        }
        super().__init__(tokenizer, slots=slots, **kwargs)

    def _suffix_after(self, slot_index: int) -> str:
        if slot_index == len(self.slots) - 1:
            return "}]}"
        day, field = self.slots[slot_index + 1]
        key = self._section_key_at.get(slot_index + 1)
        if key is None:
            return "}, " + _assignment_opening(day, field)
        return f"}}], {json.dumps(key)}: [" + _assignment_opening(day, field)


def _without_header(text: str) -> str:
    return text.split("\n", 1)[1]


def _slot_list(slots: Sequence[Tuple[int, str]]) -> str:
    return ", ".join(f"(day {day}, {field})" for day, field in slots)


def _render_native_budget(example: Mapping[str, Any]) -> str:
    contract = build_role_budget_contract(example)
    if contract.feasible:
        status = (
            "feasible=yes. Each cap includes that subtask's reference-derived "
            "minimum plus half of the remaining slack. If both subtasks stay within "
            "their caps, the combined known cost cannot exceed the team budget."
        )
    else:
        status = (
            f"feasible=no ({contract.reason}). The caps still sum to the team "
            "budget but cannot guarantee a feasible plan; minimize cost while "
            "satisfying every grounded required slot."
        )
    return f"""DETERMINISTIC TEAM BUDGET CONTRACT
- team_budget={_format_cents(contract.team_budget_cents)}
- logistics_floor={_format_cents(contract.logistics_floor_cents)}
- experience_floor={_format_cents(contract.experience_floor_cents)}
- logistics_cap={_format_cents(contract.logistics_cap_cents)} (SUBTASK A)
- experience_cap={_format_cents(contract.experience_cap_cents)} (SUBTASK B)
- {status}
Sum the displayed full-party cost once for every selected occurrence. Transportation
and each accommodation night count toward SUBTASK A; every selected meal counts
toward SUBTASK B; current_city, attractions, and "-" cost zero. Keep each subtask at
or below its cap, so the whole itinerary stays within team_budget. Budget compliance
never excuses a missing required value or an ungrounded choice."""


def build_native_centralized_prompt(example: Mapping[str, Any]) -> str:
    """Render one planner prompt with every role's information exactly once."""

    reference = str(example.get("reference_information", ""))
    catalog = parse_reference_catalog(reference)
    if not catalog.parse_success:
        raise ValueError(
            f"Reference catalog parsing failed for Travel sample {example.get('id')!r}."
        )
    days = int(example.get("days", 0))
    scaffold_values = derive_route_scaffold(example, catalog)
    scaffold = "\n".join(_route_scaffold(dict(example), catalog))
    logistics_slots = ordered_owned_slots(0, days)
    experience_slots = ordered_owned_slots(1, days)
    logistics_catalog = _render_logistics_catalog(
        catalog, example=example, scaffold=scaffold_values
    )
    experience_catalog = _render_experience_catalog(
        catalog, example=example, scaffold=scaffold_values
    )
    local_constraint = json.dumps(
        example.get("local_constraint", {}), ensure_ascii=False, sort_keys=True
    )
    first_b = experience_slots[0]
    return f"""You are one travel planner. The itinerary is decomposed into two subtasks,
and you complete BOTH subtasks yourself in one response. Give a separate answer for
each subtask: first the SUBTASK A answer, then the SUBTASK B answer. A deterministic
downstream merger combines the two answers into one itinerary, which is scored as a
whole.

SUBTASK A (LOGISTICS AND FEASIBILITY):
current_city, transportation, and accommodation on every day.
Slots in answer order: {_slot_list(logistics_slots)}

SUBTASK B (DAILY EXPERIENCE):
breakfast, attraction, lunch, and dinner on every day.
Slots in answer order: {_slot_list(experience_slots)}

Because you write both answers, keep them consistent: SUBTASK B must use the cities
fixed by SUBTASK A on the same days, and the combined cost of both answers must stay
within the team budget.

DAY-SLOT CHECKLIST (derived only from the shared scaffold)
SUBTASK A:
{_without_header(_render_role_slot_checklist(example, 0))}
SUBTASK B:
{_without_header(_render_role_slot_checklist(example, 1))}

HARD-CONSTRAINT DUTIES
SUBTASK A:
{_without_header(_render_role_hard_constraint_duty(example, 0))}
SUBTASK B:
{_without_header(_render_role_hard_constraint_duty(example, 1))}

{_render_native_budget(example)}

ITINERARY RULES:
- The itinerary has {len(PLAN_FIELDS) * days} slots: {days} days x {len(PLAN_FIELDS)} fields. The two
  answers together assign every slot exactly once.
- If a slot should intentionally be empty, explicitly assign the string "-".
- Write current_city as "from A to B" on a travel day and as the single current city on
  a stay day. The trip starts at the origin, visits the requested number of cities, and
  returns to the origin on the final day.
- A travel day requires matching transportation; a stay day uses "-" for transportation.
- A stay day requires breakfast, attraction, lunch, and dinner. These experience fields
  may be "-" on a travel day. Accommodation is required on every day except the final
  return day.
- Never use "-" merely because you are uncertain. Use it only in the cases above.
- Use only entities and facts from the TRIP REQUEST, STRUCTURED CONSTRAINTS, and
  REFERENCE-DERIVED PLANNING CONTEXT.
- Do not invent prices, flight numbers, restaurants, attractions, or accommodations.
- Write restaurants, attractions, and accommodations as "Name, City". Never join
  multiple attractions: choose exactly one grounded attraction on a required stay day.
  Copy a complete transportation candidate, including its mode or flight number and the
  matching route, rather than returning an abbreviation.

VALID FIELDS:
{", ".join(PLAN_FIELDS)}

TRIP REQUEST:
{example.get('query') or example.get('prompt') or ''}

STRUCTURED CONSTRAINTS:
origin={example.get('org', '')}
destination={example.get('dest', '')}
days={days}
dates={example.get('date', '')}
people={example.get('people_number', '')}
budget={example.get('budget', '')}
local_constraints={local_constraint}

REFERENCE-DERIVED PLANNING CONTEXT:
SHARED REFERENCE-DERIVED ROUTE SCAFFOLD
This deterministic scaffold comes only from dated reference routes. It coordinates
movement versus stay days; it does not select a flight, lodging, meal, attraction,
or any target itinerary.
{scaffold}

SUBTASK A CATALOG
{logistics_catalog}

SUBTASK B CATALOG
{experience_catalog}

STRICT OUTPUT CONTRACT:
- Your entire response must be exactly one JSON object: no Markdown fence, prefix,
  explanation, note, suffix, or second object.
- The object has exactly two keys, in this order:
  "{NATIVE_SECTION_KEYS[0]}": the SUBTASK A answer, {len(logistics_slots)} assignment objects;
  "{NATIVE_SECTION_KEYS[1]}": the SUBTASK B answer, {len(experience_slots)} assignment objects.
- Every assignment object has exactly three keys: "day" (integer), "field" (one valid
  field), and "value" (string copied from the reference, or "-"), in the slot order
  listed for its subtask.
- Do not copy schema wording or placeholder text as a value.
- The generation system has already written the ASSISTANT PREFILL below. It is
  part of your response, so do not repeat any portion of it.
- The prefill ends immediately after the opening quote of the first SUBTASK A value.
  Generate only that value's contents, then close the value with a double quote.
- After each closing value quote, the generation system supplies the next fixed
  assignment object. After the last SUBTASK A value it closes that answer and opens
  "{NATIVE_SECTION_KEYS[1]}" at (day {first_b[0]}, {first_b[1]}). It fixes every key, day,
  field, comma, bracket, and brace; you choose only each value.
- A per-value token limit prevents one slot from consuming the whole response budget.
  Keep every value concise and copy catalog spelling exactly.

ASSISTANT PREFILL (already supplied; do not repeat it):
{build_native_prefill(days)}

Continue the prefilled JSON now. Your first generated character is the first
character inside the already-open value string.
"""


class TravelNativeCentralizedAdapter:
    """Build the native planner prompt and split its two answers into roles."""

    def __init__(self, tokenizer: Any = None):
        self.tokenizer = tokenizer

    def build_prompt(self, batch_item: Dict[str, Any], agent_prompts: Any) -> str:
        del agent_prompts  # The native prompt never embeds role prompts.
        if self.tokenizer is None:
            # Data/contract dry runs do not need model downloads.
            return build_native_centralized_prompt(batch_item)
        formatter = wrap_formatter_with_chat_template(
            build_native_centralized_prompt,
            self.tokenizer,
            system_prompt=NATIVE_SYSTEM_PROMPT,
        )
        return formatter(batch_item)

    def parse_completion(
        self, completion: str, batch_item: Dict[str, Any], num_agents: int
    ) -> List[str]:
        if num_agents != 2:
            raise ValueError("Centralized Travel requires two logical roles.")
        try:
            return self._split(completion, int(batch_item["days"]))
        except (ValueError, TypeError, KeyError) as exc:
            # A malformed sample is a failed task action, not a training crash.
            # CoMLRL maps this exception to two empty roles.
            raise CentralizedComparatorParseError(str(exc)) from exc

    @staticmethod
    def _split(completion: str, days: int) -> List[str]:
        data = json.loads(completion)
        if not isinstance(data, dict) or list(data) != list(NATIVE_SECTION_KEYS):
            raise ValueError(
                f"Native centralized output must have keys {list(NATIVE_SECTION_KEYS)}."
            )
        outputs = []
        for agent_idx, key in enumerate(NATIVE_SECTION_KEYS):
            answer = data[key]
            expected = ordered_owned_slots(agent_idx, days)
            if not isinstance(answer, list) or len(answer) != len(expected):
                raise ValueError(f"{key} must contain every subtask slot exactly once.")
            assignments = []
            for assignment, slot in zip(answer, expected):
                if not isinstance(assignment, dict) or set(assignment) != {
                    "day",
                    "field",
                    "value",
                }:
                    raise ValueError(f"Invalid {key} assignment schema.")
                if (
                    type(assignment["day"]) is not int
                    or (assignment["day"], assignment["field"]) != slot
                ):
                    raise ValueError(f"{key} slots must follow the subtask order.")
                if not isinstance(assignment["value"], str):
                    raise ValueError(f"{key} values must be strings.")
                assignments.append(
                    {"day": slot[0], "field": slot[1], "value": assignment["value"]}
                )
            outputs.append(
                json.dumps(
                    {"agent_id": agent_idx, "assignments": assignments},
                    ensure_ascii=False,
                )
            )
        return outputs
