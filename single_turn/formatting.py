"""Role-guided prompts for simultaneous TravelPlanner agents."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from single_turn.aggregation import PLAN_FIELDS, assignment_capacity, owned_slots
from single_turn.rewards.reference_evaluator import (
    CatalogEntry,
    ReferenceCatalog,
    parse_reference_catalog,
)
from single_turn.structured_generation import (
    DEFAULT_SYSTEM_PROMPT,
    wrap_formatter_with_chat_template,
)


def _first_owned_slot(agent_idx: int, days: int) -> Tuple[int, str]:
    slots = sorted(
        owned_slots(agent_idx, days),
        key=lambda slot: (slot[0], PLAN_FIELDS.index(slot[1])),
    )
    if not slots:
        raise ValueError("A Travel agent must own at least one itinerary slot.")
    return slots[0]


def build_agent_json_prefill(agent_idx: int, days: int) -> str:
    """Return the fixed assistant prefix through its first value's opening quote.

    The prefix contains only schema and the deterministic role partition. It does
    not contain a target value or any annotated itinerary information.
    """

    first_day, first_field = _first_owned_slot(int(agent_idx), int(days))
    return (
        f'{{"agent_id": {int(agent_idx)}, "assignments": '
        f'[{{"day": {first_day}, "field": '
        f'{json.dumps(first_field, ensure_ascii=False)}, "value": "'
    )


def _number(value: float | None) -> str:
    if value is None:
        return "unknown"
    return format(float(value), ".15g")


def _canonical_transport_value(entry: CatalogEntry) -> str:
    if entry.mode == "flight":
        return (
            f"Flight Number: {entry.name}, from {entry.origin} "
            f"to {entry.destination}"
        )
    mode = "Self-driving" if entry.mode == "self-driving" else "Taxi"
    return f"{mode}, from {entry.origin} to {entry.destination}"


def _unique_sorted(
    entries: Iterable[CatalogEntry],
    *,
    key: Any,
) -> List[CatalogEntry]:
    unique: Dict[Tuple[Any, ...], CatalogEntry] = {}
    for entry in entries:
        identity = (
            entry.category,
            entry.name,
            entry.city,
            entry.cost,
            entry.cuisines,
            entry.room_type,
            entry.house_rules,
            entry.minimum_nights,
            entry.maximum_occupancy,
            entry.mode,
            entry.origin,
            entry.destination,
            entry.date,
        )
        unique.setdefault(identity, entry)
    return sorted(unique.values(), key=key)


def _coerce_dates(example: Dict[str, Any]) -> List[str]:
    dates = example.get("dates")
    if not isinstance(dates, (list, tuple)):
        dates = example.get("date", [])
    if isinstance(dates, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(dates)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(parsed, (list, tuple)):
                dates = parsed
                break
        else:
            dates = []
    if not isinstance(dates, (list, tuple)):
        return []
    return [str(value).strip() for value in dates]


def _reference_records(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = example.get("reference_records")
    if isinstance(records, list):
        return [dict(record) for record in records if isinstance(record, dict)]
    raw = example.get("reference_information", "")
    if isinstance(raw, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(raw)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(parsed, list):
                return [
                    dict(record) for record in parsed if isinstance(record, dict)
                ]
    return []


def _route_scaffold(
    example: Dict[str, Any], catalog: ReferenceCatalog
) -> List[str]:
    """Derive one shared movement/stay scaffold from dated route records."""

    days = int(example.get("days", 0))
    dates = _coerce_dates(example)
    routes_by_date: Dict[str, set[Tuple[str, str]]] = {}
    for entry in catalog.transportation:
        if entry.date and entry.origin and entry.destination:
            routes_by_date.setdefault(entry.date, set()).add(
                (entry.origin, entry.destination)
            )
    # The catalog intentionally omits unavailable flights, but their dated
    # reference descriptions still define the trip's route. Keep that route
    # signal even when the usable choice must be taxi or self-driving.
    for record in _reference_records(example):
        description = str(record.get("Description", "")).strip()
        match = re.fullmatch(
            r"Flight\s+from\s+(.+?)\s+to\s+(.+?)\s+on\s+"
            r"(\d{4}-\d{2}-\d{2})",
            description,
            flags=re.I,
        )
        if match:
            origin, destination, date = match.groups()
            routes_by_date.setdefault(date, set()).add((origin, destination))

    current_city = str(example.get("org", "")).strip()
    lines: List[str] = []
    for day in range(1, days + 1):
        date = dates[day - 1] if day <= len(dates) else ""
        candidates = sorted(
            routes_by_date.get(date, ()),
            key=lambda route: (route[0].casefold(), route[1].casefold()),
        )
        matching = [
            route
            for route in candidates
            if current_city and route[0].casefold() == current_city.casefold()
        ]
        route = (matching or candidates)[0] if candidates else None
        date_text = f" date={date}" if date else ""
        if route is not None:
            origin, destination = route
            current_city = destination
            city_value = f"from {origin} to {destination}"
            lines.append(
                f"- day={day}{date_text} kind=move "
                f"current_city={json.dumps(city_value, ensure_ascii=False)} "
                "transportation=choose_exact_catalog_option_for_this_route"
            )
        else:
            city_value = current_city or "unknown"
            lines.append(
                f"- day={day}{date_text} kind=stay "
                f"current_city={json.dumps(city_value, ensure_ascii=False)} "
                'transportation="-"'
            )
    return lines


def _render_restaurant_catalog(catalog: ReferenceCatalog) -> str:
    restaurants = _unique_sorted(
        catalog.restaurants,
        key=lambda entry: (entry.city.casefold(), entry.name.casefold()),
    )
    lines = [
        "RESTAURANT OPTIONS",
        "Copy the quoted JSON value exactly; cost/cuisines are metadata.",
    ]
    for entry in restaurants:
        value = f"{entry.name}, {entry.city}"
        cuisines = ", ".join(entry.cuisines) or "unknown"
        lines.append(
            f"- {json.dumps(value, ensure_ascii=False)}"
            f" | average_cost={_number(entry.cost)} | cuisines={cuisines}"
        )
    return "\n".join(lines)


def _render_logistics_catalog(catalog: ReferenceCatalog) -> str:
    transport = _unique_sorted(
        catalog.transportation,
        key=lambda entry: (
            entry.date or "9999-99-99",
            entry.origin.casefold(),
            entry.destination.casefold(),
            entry.mode,
            entry.name.casefold(),
            float(entry.cost or 0.0),
        ),
    )
    accommodations = _unique_sorted(
        catalog.accommodations,
        key=lambda entry: (entry.city.casefold(), entry.name.casefold()),
    )
    lines = [
        "TRANSPORTATION OPTIONS",
        "Copy the quoted JSON value exactly; cost is metadata, not part of the value.",
    ]
    for entry in transport:
        value = _canonical_transport_value(entry)
        date = f" | date={entry.date}" if entry.date else ""
        lines.append(
            f"- {json.dumps(value, ensure_ascii=False)} | cost={_number(entry.cost)}{date}"
        )
    lines.extend(
        [
            "",
            "ACCOMMODATION OPTIONS",
            "Copy the quoted JSON value exactly; remaining attributes are metadata.",
        ]
    )
    for entry in accommodations:
        value = f"{entry.name}, {entry.city}"
        lines.append(
            f"- {json.dumps(value, ensure_ascii=False)}"
            f" | cost_per_night={_number(entry.cost)}"
            f" | room_type={entry.room_type}"
            f" | house_rules={entry.house_rules or 'none'}"
            f" | minimum_nights={_number(entry.minimum_nights)}"
            f" | maximum_occupancy={_number(entry.maximum_occupancy)}"
        )
    # Agent 0 owns dinner on even-numbered days, so logistics still needs the
    # restaurant catalog. Without it, a strict role-following policy cannot
    # ground every slot it owns.
    lines.extend(["", _render_restaurant_catalog(catalog)])
    return "\n".join(lines)


def _render_experience_catalog(catalog: ReferenceCatalog) -> str:
    attractions = _unique_sorted(
        catalog.attractions,
        key=lambda entry: (entry.city.casefold(), entry.name.casefold()),
    )
    lines = [_render_restaurant_catalog(catalog)]
    lines.extend(
        [
            "",
            "ATTRACTION OPTIONS",
            "Copy one quoted JSON value, or join multiple copied values with ';'.",
        ]
    )
    for entry in attractions:
        value = f"{entry.name}, {entry.city}"
        lines.append(f"- {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(lines)


def build_compact_reference_context(
    example: Dict[str, Any], agent_idx: int
) -> str:
    """Render shared routing plus only the catalog needed by one role.

    Fail closed for malformed upstream rows rather than silently reintroducing
    the raw tables, which would restore both excessive context and copy failure
    modes. Normal TravelPlanner rows omit addresses, phone numbers, URLs,
    coordinates, and other irrelevant table columns.
    """

    reference = str(example.get("reference_information", ""))
    catalog = parse_reference_catalog(reference)
    if not catalog.parse_success:
        sample_id = str(example.get("id", "unknown"))
        raise ValueError(
            f"Reference catalog parsing failed for Travel sample {sample_id!r}."
        )

    scaffold = "\n".join(_route_scaffold(example, catalog))
    if int(agent_idx) == 0:
        role_catalog = _render_logistics_catalog(catalog)
    elif int(agent_idx) == 1:
        role_catalog = _render_experience_catalog(catalog)
    else:
        raise ValueError("The initial TravelPlanner task has exactly two agents.")
    return f"""SHARED REFERENCE-DERIVED ROUTE SCAFFOLD
This deterministic scaffold comes only from dated reference routes. It coordinates
movement versus stay days; it does not select a flight, lodging, meal, attraction,
or any target itinerary.
{scaffold}

ROLE-SPECIFIC COMPACT REFERENCE CATALOG
{role_catalog}"""


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
            "slot, including an explicit '-' when the itinerary convention permits it to be empty. Do not "
            "submit breakfast, attraction, lunch, or odd-day dinner slots owned by Agent 1."
        )
    if role_mode == "partitioned_roles" and agent_idx == 1:
        return (
            "You own DAILY EXPERIENCE: breakfast, attraction, and lunch on every day, "
            "plus dinner on odd-numbered days. Submit every "
            "owned slot, including an explicit '-' when the itinerary convention permits it to be empty. "
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
    force_json_prefix: bool = True,
) -> Callable[[Dict[str, Any]], str]:
    def formatter(example: Dict[str, Any], external_prompts: Any = None) -> str:
        del external_prompts
        days = int(example.get("days", 0))
        total_slots = len(PLAN_FIELDS) * days
        capacity = assignment_capacity(days, num_agents)
        reference = build_compact_reference_context(example, agent_idx)
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
        assistant_prefill = build_agent_json_prefill(agent_idx, days)
        if force_json_prefix:
            generation_contract = f"""- The generation system has already written the ASSISTANT PREFILL below. It is
  part of your response, so do not repeat any portion of it.
- The prefill ends immediately after the opening quote of your first owned value.
  Continue with that value's contents, close its quote and assignment object, then
  emit all remaining owned assignments in the listed order.
- The complete reconstructed response must begin exactly with the prefill and obey
  the same JSON schema and assignment count.
- Your response must end exactly with this character sequence:
  ]}}

ASSISTANT PREFILL (already supplied; do not repeat it):
{assistant_prefill}

Continue the prefilled JSON now. Your first generated character is the first
character inside the already-open value string. Stop immediately after the
matching top-level closing "}}"."""
        else:
            generation_contract = f"""- Your response must begin exactly with this character sequence:
  {{"agent_id": {agent_idx}, "assignments": [
- Your response must end exactly with this character sequence:
  ]}}

Output the complete JSON object now. The first generated character must be "{{".
Stop immediately after its matching top-level closing "}}"."""
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
- Write restaurants, attractions, and accommodations as "Name, City". Separate multiple
  attractions with semicolons. Copy a complete transportation candidate, including its
  mode or flight number and the matching route, rather than returning an abbreviation.
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

REFERENCE-DERIVED PLANNING CONTEXT:
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
{generation_contract}
"""

    return formatter


def get_single_turn_formatters(
    *,
    num_agents: int,
    role_mode: str = "partitioned_roles",
    force_json_prefix: bool = True,
    tokenizers: Optional[Sequence[Any]] = None,
    use_chat_template: bool = False,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> List[Callable[[Dict[str, Any]], str]]:
    formatters = [
        build_single_turn_formatter(
            agent_idx,
            num_agents=num_agents,
            role_mode=role_mode,
            force_json_prefix=force_json_prefix,
        )
        for agent_idx in range(num_agents)
    ]
    if not use_chat_template:
        return formatters
    if tokenizers is None or len(tokenizers) != num_agents:
        raise ValueError(
            "use_chat_template=true requires exactly one tokenizer per agent."
        )
    return [
        wrap_formatter_with_chat_template(
            formatter,
            tokenizers[agent_idx],
            system_prompt=system_prompt,
        )
        for agent_idx, formatter in enumerate(formatters)
    ]
