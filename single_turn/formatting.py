"""Role-guided prompts for simultaneous TravelPlanner agents."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from single_turn.aggregation import PLAN_FIELDS, assignment_capacity, owned_slots


def _role_instruction(agent_idx: int, role_mode: str, days: int) -> str:
    if role_mode not in {"partitioned_roles", "soft_roles", "same_prompt"}:
        raise ValueError(
            "role_mode must be 'partitioned_roles', 'soft_roles', or 'same_prompt'."
        )
    if role_mode == "same_prompt":
        return (
            "Select a complementary subset of itinerary slots. You have no assigned "
            "specialty, so infer which contribution will most improve the merged plan."
        )

    if role_mode == "partitioned_roles" and agent_idx == 0:
        return (
            "You own LOGISTICS AND FEASIBILITY: current_city, transportation, and "
            "accommodation on every day, plus dinner on even-numbered days. Submit every owned "
            "slot, including an explicit '-' when the correct value is empty. Do not "
            "submit breakfast, attraction, lunch, or odd-day dinner slots owned by Agent 1."
        )
    if role_mode == "partitioned_roles" and agent_idx == 1:
        return (
            "You own DAILY EXPERIENCE: breakfast, attraction, and lunch on every day, "
            "plus dinner on odd-numbered days. Submit every "
            "owned slot, including an explicit '-' when the correct value is empty. "
            "Do not submit current_city, transportation, accommodation, or even-day "
            "dinner slots owned by Agent 0."
        )

    if agent_idx == 0:
        core = (
            "Your primary role is LOGISTICS AND FEASIBILITY. Prioritize current_city, "
            "transportation, accommodation, route consistency, arrival/departure timing, "
            "and preserving budget for the other agent."
        )
        allowed = (
            "You may fill meal or attraction slots when useful; the role is guidance, "
            "not a hard action restriction."
        )
    elif agent_idx == 1:
        core = (
            "Your primary role is DAILY EXPERIENCE. Prioritize breakfast, attraction, "
            "lunch, dinner, user preferences, daily feasibility, and respecting the "
            "budget likely consumed by transport and accommodation."
        )
        allowed = (
            "You may fill city, transportation, or accommodation slots when useful; "
            "the role is guidance, not a hard action restriction."
        )
    else:
        core = f"You are agent {agent_idx + 1}; select useful uncovered itinerary slots."
        allowed = "The role is guidance, not a hard action restriction."

    return f"{core}\n{allowed}"


def build_single_turn_formatter(
    agent_idx: int,
    *,
    num_agents: int = 2,
    role_mode: str = "partitioned_roles",
) -> Callable[[Dict[str, Any]], str]:
    def formatter(example: Dict[str, Any], external_prompts: Any = None) -> str:
        del external_prompts
        days = int(example.get("days", 0))
        total_slots = len(PLAN_FIELDS) * days
        capacity = assignment_capacity(days, num_agents)
        reference = str(example.get("reference_information", ""))
        role = _role_instruction(agent_idx, role_mode, days)
        owned = (
            sorted(
                owned_slots(agent_idx, days),
                key=lambda slot: (slot[0], PLAN_FIELDS.index(slot[1])),
            )
            if role_mode == "partitioned_roles"
            else []
        )
        target_count = len(owned) if owned else capacity
        owned_slot_text = (
            ", ".join(f"(day {day}, {field})" for day, field in owned)
            if owned
            else "Infer a complementary subset from the shared task."
        )
        target_rule = (
            f"Submit exactly {target_count} assignments: one for every slot assigned "
            "to your role."
            if role_mode == "partitioned_roles"
            else f"Submit a useful complementary subset of no more than {capacity} slots."
        )
        output_count_rule = (
            f'"assignments" must contain exactly {target_count} objects.'
            if owned
            else f'"assignments" must contain between 1 and {capacity} objects.'
        )
        local_constraint = json.dumps(
            example.get("local_constraint", {}), ensure_ascii=False, sort_keys=True
        )
        fields = ", ".join(PLAN_FIELDS)
        return f"""You are Agent {agent_idx} in a decentralized travel-planning team.

There are {num_agents} agents. Every agent receives the same trip request and acts
simultaneously. You cannot communicate with the other agent or see its output. A
deterministic downstream merger combines all slot assignments into one itinerary.
Every agent receives exactly the same reward for that merged itinerary.

ROLE GUIDANCE:
{role}

YOUR OWNED SLOTS:
{owned_slot_text}

COLLABORATION RULES:
- The itinerary has {total_slots} slots: {days} days x {len(PLAN_FIELDS)} fields.
- You may submit at most {capacity} assignments.
- {target_rule}
- A slot is identified by (day, field) and should be assigned exactly once by the team.
- Same-slot duplication wastes capacity; different values for one slot create a conflict.
- If a slot should intentionally be empty, explicitly assign the string "-". An omitted
  slot is considered missing, which is different from explicitly assigning "-".
- Never use "-" merely because you are uncertain; use it only when that itinerary slot
  is genuinely empty under the TravelPlanner convention.
- Use only entities and facts from the TRIP REQUEST, STRUCTURED CONSTRAINTS, and
  REFERENCE INFORMATION.
- Do not invent prices, flight numbers, restaurants, attractions, or accommodations.
- Copy complete candidate values rather than abbreviating them. For example, preserve
  the full flight route and times instead of returning only a flight number.
- Your own section is not scored separately; optimize the final team itinerary.

VALID FIELDS:
{fields}

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

REFERENCE INFORMATION:
{reference}

STRICT OUTPUT CONTRACT:
- Your entire response must be exactly one JSON object: no Markdown fence, prefix,
  explanation, note, suffix, or second object.
- The object must have exactly two keys: "agent_id" and "assignments".
- "agent_id" must be the integer {agent_idx}.
- {output_count_rule} It must never exceed {capacity} objects.
- Every assignment object must have exactly three keys: "day" (integer), "field"
  (one valid field), and "value" (string copied from the reference, or "-").
- Do not copy schema wording or placeholder text as a value.
- Your response must begin exactly with this character sequence:
  {{"agent_id": {agent_idx}, "assignments": [
- Your response must end exactly with this character sequence:
  ]}}
"""

    return formatter


def get_single_turn_formatters(
    *,
    num_agents: int,
    role_mode: str = "partitioned_roles",
) -> List[Callable[[Dict[str, Any]], str]]:
    return [
        build_single_turn_formatter(
            agent_idx,
            num_agents=num_agents,
            role_mode=role_mode,
        )
        for agent_idx in range(num_agents)
    ]
