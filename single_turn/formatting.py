"""Role-guided prompts for simultaneous TravelPlanner agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from single_turn.aggregation import (
    PLAN_FIELDS,
    assignment_capacity,
    canonical_value,
    ordered_owned_slots,
)
from single_turn.rewards.reference_evaluator import (
    CatalogEntry,
    ReferenceCatalog,
    accommodation_satisfies_house_rule,
    accommodation_satisfies_room_type,
    coerce_trip_dates,
    derive_route_scaffold,
    parse_reference_catalog,
    scaled_party_cost,
    transportation_satisfies_rule,
)
from single_turn.structured_generation import (
    DEFAULT_SYSTEM_PROMPT,
    wrap_formatter_with_chat_template,
)


def _first_owned_slot(agent_idx: int, days: int) -> Tuple[int, str]:
    slots = ordered_owned_slots(agent_idx, days)
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


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _money_cents(value: Any, *, budget: bool = False) -> int | None:
    """Convert reference money to conservative integer cents.

    Candidate costs are rounded up while the available budget is rounded down,
    so following a rendered cap cannot exceed the float-based evaluator budget
    because of display rounding.
    """

    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not amount.is_finite() or amount < 0:
        return None
    rounding = ROUND_FLOOR if budget else ROUND_CEILING
    return int((amount * 100).to_integral_value(rounding=rounding))


def _format_cents(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / 100:.2f}"


def _constraints(example: Mapping[str, Any]) -> Dict[str, Any]:
    value = example.get("local_constraint", {}) or {}
    return dict(value) if isinstance(value, Mapping) else {}


def _requested_cuisines(constraints: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = constraints.get("cuisine")
    if not raw:
        return ()
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    unique: Dict[str, None] = {}
    for value in values:
        normalized = canonical_value(value)
        if normalized and normalized != "-":
            unique.setdefault(normalized, None)
    return tuple(unique)


def _entry_cost_cents(entry: CatalogEntry, people: int) -> int | None:
    return _money_cents(scaled_party_cost(entry, people))


def _cost_sort_key(value: int | None) -> int:
    return value if value is not None else 10**18


def _route_matches_transport(
    entry: CatalogEntry,
    *,
    day: int,
    route: Tuple[str, str, str],
    dates: Sequence[str],
) -> bool:
    if route[0] != "move":
        return False
    if canonical_value(entry.origin) != canonical_value(route[1]):
        return False
    if canonical_value(entry.destination) != canonical_value(route[2]):
        return False
    return bool(
        entry.mode != "flight"
        or not entry.date
        or day - 1 >= len(dates)
        or entry.date == str(dates[day - 1])
    )


def _transport_days(
    entry: CatalogEntry,
    scaffold: Sequence[Tuple[str, str, str]],
    dates: Sequence[str],
) -> Tuple[int, ...]:
    return tuple(
        day
        for day, route in enumerate(scaffold, start=1)
        if _route_matches_transport(entry, day=day, route=route, dates=dates)
    )


def _city_days(
    city: str,
    scaffold: Sequence[Tuple[str, str, str]],
    *,
    stays_only: bool = False,
    exclude_final: bool = False,
) -> Tuple[int, ...]:
    last_day = len(scaffold)
    return tuple(
        day
        for day, route in enumerate(scaffold, start=1)
        if (not stays_only or route[0] == "stay")
        and (not exclude_final or day < last_day)
        and canonical_value(route[2]) == canonical_value(city)
    )


def _accommodation_days(
    entry: CatalogEntry,
    scaffold: Sequence[Tuple[str, str, str]],
) -> Tuple[int, ...]:
    """Return only days belonging to a long-enough consecutive city run."""

    runs: List[Tuple[str, List[int]]] = []
    for day, route in enumerate(scaffold, start=1):
        if day >= len(scaffold):
            continue
        city = route[2]
        if runs and canonical_value(runs[-1][0]) == canonical_value(city):
            runs[-1][1].append(day)
        else:
            runs.append((city, [day]))
    return tuple(
        day
        for city, days in runs
        if canonical_value(city) == canonical_value(entry.city)
        and len(days) >= entry.minimum_nights
        for day in days
    )


@dataclass(frozen=True)
class RoleBudgetContract:
    """A target-free, reference-derived budget partition for the two roles."""

    team_budget_cents: int
    logistics_floor_cents: int | None
    experience_floor_cents: int | None
    logistics_cap_cents: int
    experience_cap_cents: int
    feasible: bool
    reason: str

    def floor_for_agent(self, agent_idx: int) -> int | None:
        if int(agent_idx) == 0:
            return self.logistics_floor_cents
        if int(agent_idx) == 1:
            return self.experience_floor_cents
        raise ValueError("The initial TravelPlanner task has exactly two agents.")

    def cap_for_agent(self, agent_idx: int) -> int:
        if int(agent_idx) == 0:
            return self.logistics_cap_cents
        if int(agent_idx) == 1:
            return self.experience_cap_cents
        raise ValueError("The initial TravelPlanner task has exactly two agents.")


def _minimum_transport_cost(
    catalog: ReferenceCatalog,
    scaffold: Sequence[Tuple[str, str, str]],
    dates: Sequence[str],
    constraints: Mapping[str, Any],
    people: int,
) -> int | None:
    legs: List[List[Tuple[str, int]]] = []
    for day, route in enumerate(scaffold, start=1):
        if route[0] != "move":
            continue
        candidates: List[Tuple[str, int]] = []
        for entry in catalog.transportation:
            cost = _entry_cost_cents(entry, people)
            if (
                cost is not None
                and transportation_satisfies_rule(
                    entry, constraints.get("transportation")
                )
                and _route_matches_transport(
                    entry, day=day, route=route, dates=dates
                )
            ):
                candidates.append((entry.mode, cost))
        if not candidates:
            return None
        legs.append(candidates)

    if not legs:
        return 0

    # The evaluator permits flight/taxi mixtures, or an all-self-driving plan,
    # but never self-driving mixed with either other mode.
    family_totals: List[int] = []
    for self_driving in (False, True):
        total = 0
        for candidates in legs:
            costs = [
                cost
                for mode, cost in candidates
                if (mode == "self-driving") == self_driving
            ]
            if not costs:
                break
            total += min(costs)
        else:
            family_totals.append(total)
    return min(family_totals) if family_totals else None


def _minimum_accommodation_cost(
    catalog: ReferenceCatalog,
    scaffold: Sequence[Tuple[str, str, str]],
    constraints: Mapping[str, Any],
    people: int,
) -> int | None:
    runs: List[Tuple[str, int]] = []
    for day, route in enumerate(scaffold, start=1):
        if day >= len(scaffold):
            continue
        city = route[2]
        if runs and canonical_value(runs[-1][0]) == canonical_value(city):
            previous_city, nights = runs[-1]
            runs[-1] = (previous_city, nights + 1)
        else:
            runs.append((city, 1))

    total = 0
    for city, nights in runs:
        candidates: List[int] = []
        for entry in catalog.accommodations:
            cost = _entry_cost_cents(entry, people)
            if (
                canonical_value(entry.city) == canonical_value(city)
                and accommodation_satisfies_room_type(
                    entry, constraints.get("room type")
                )
                and accommodation_satisfies_house_rule(
                    entry, constraints.get("house rule")
                )
                and entry.minimum_nights <= nights
                and cost is not None
            ):
                candidates.append(cost * nights)
        if not candidates:
            return None
        total += min(candidates)
    return total


def _minimum_experience_cost(
    catalog: ReferenceCatalog,
    scaffold: Sequence[Tuple[str, str, str]],
    constraints: Mapping[str, Any],
    people: int,
) -> int | None:
    meal_counts: Dict[str, Tuple[str, int]] = {}
    for route in scaffold:
        if route[0] != "stay":
            continue
        key = canonical_value(route[2])
        raw_city, count = meal_counts.get(key, (route[2], 0))
        meal_counts[key] = (raw_city, count + 3)

    requested = _requested_cuisines(constraints)
    requested_index = {name: idx for idx, name in enumerate(requested)}
    if not meal_counts:
        return 0 if not requested else None

    global_costs: Dict[int, int] = {0: 0}
    for city, required_count in meal_counts.values():
        by_identity: Dict[Tuple[str, str, str], CatalogEntry] = {}
        for entry in catalog.restaurants:
            if canonical_value(entry.city) != canonical_value(city):
                continue
            previous = by_identity.get(entry.identity)
            previous_cost = (
                _entry_cost_cents(previous, people)
                if previous is not None
                else None
            )
            entry_cost = _entry_cost_cents(entry, people)
            if previous is None or (
                entry_cost is not None
                and (previous_cost is None or entry_cost < previous_cost)
            ):
                by_identity[entry.identity] = entry
        entries = sorted(
            by_identity.values(), key=lambda entry: entry.name.casefold()
        )
        city_costs: Dict[Tuple[int, int], int] = {(0, 0): 0}
        for entry in entries:
            cost = _entry_cost_cents(entry, people)
            if cost is None:
                continue
            cuisines = {canonical_value(value) for value in entry.cuisines}
            coverage = sum(
                1 << idx
                for name, idx in requested_index.items()
                if name in cuisines
            )
            updated = dict(city_costs)
            for (count, mask), current_cost in city_costs.items():
                if count >= required_count:
                    continue
                key = (count + 1, mask | coverage)
                updated[key] = min(
                    updated.get(key, current_cost + cost), current_cost + cost
                )
            city_costs = updated
        complete_city = {
            mask: cost
            for (count, mask), cost in city_costs.items()
            if count == required_count
        }
        if not complete_city:
            return None
        combined: Dict[int, int] = {}
        for old_mask, old_cost in global_costs.items():
            for city_mask, city_cost in complete_city.items():
                mask = old_mask | city_mask
                combined[mask] = min(
                    combined.get(mask, old_cost + city_cost),
                    old_cost + city_cost,
                )
        global_costs = combined

    full_mask = (1 << len(requested)) - 1
    return global_costs.get(full_mask)


@lru_cache(maxsize=512)
def _cached_role_budget_contract(
    reference: str,
    days: int,
    dates: Tuple[str, ...],
    origin: str,
    people: int,
    raw_budget: str,
    raw_constraints: str,
) -> RoleBudgetContract:
    constraints = json.loads(raw_constraints)
    catalog = parse_reference_catalog(reference)
    scaffold = derive_route_scaffold(
        {
            "days": days,
            "dates": list(dates),
            "org": origin,
            "reference_information": reference,
        },
        catalog,
    )
    team_budget = _money_cents(raw_budget, budget=True) or 0
    transport = _minimum_transport_cost(
        catalog, scaffold, dates, constraints, people
    )
    accommodation = _minimum_accommodation_cost(
        catalog, scaffold, constraints, people
    )
    logistics = (
        transport + accommodation
        if transport is not None and accommodation is not None
        else None
    )
    experience = _minimum_experience_cost(
        catalog, scaffold, constraints, people
    )

    if team_budget <= 0:
        feasible = False
        reason = "non_positive_budget"
    elif logistics is None:
        feasible = False
        reason = "no_constraint_eligible_logistics_plan"
    elif experience is None:
        feasible = False
        reason = "no_grounded_diverse_meal_plan"
    elif logistics + experience > team_budget:
        feasible = False
        reason = "minimum_role_costs_exceed_budget"
    else:
        feasible = True
        reason = "feasible"

    if feasible:
        assert logistics is not None and experience is not None
        slack = team_budget - logistics - experience
        logistics_cap = logistics + (slack + 1) // 2
    elif logistics is not None and experience is not None and logistics + experience:
        # This fallback only communicates scarcity; it makes no feasibility
        # promise. Preserve the floor ratio and keep the exact team sum.
        logistics_cap = team_budget * logistics // (logistics + experience)
    else:
        logistics_cap = (team_budget + 1) // 2
    experience_cap = team_budget - logistics_cap
    return RoleBudgetContract(
        team_budget_cents=team_budget,
        logistics_floor_cents=logistics,
        experience_floor_cents=experience,
        logistics_cap_cents=logistics_cap,
        experience_cap_cents=experience_cap,
        feasible=feasible,
        reason=reason,
    )


def build_role_budget_contract(example: Mapping[str, Any]) -> RoleBudgetContract:
    """Build the shared budget signal without consulting any target plan."""

    constraints = _constraints(example)
    return _cached_role_budget_contract(
        str(example.get("reference_information", "")),
        int(example.get("days", 0)),
        tuple(coerce_trip_dates(example)),
        str(example.get("org", "")),
        max(1, int(example.get("people_number", 1) or 1)),
        str(example.get("budget", 0) or 0),
        json.dumps(constraints, ensure_ascii=False, sort_keys=True, default=str),
    )


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


def _route_scaffold(
    example: Dict[str, Any], catalog: ReferenceCatalog
) -> List[str]:
    """Derive one shared movement/stay scaffold from dated route records."""

    dates = coerce_trip_dates(example)
    lines: List[str] = []
    for day, (kind, origin, destination) in enumerate(
        derive_route_scaffold(example, catalog), start=1
    ):
        date = dates[day - 1] if day <= len(dates) else ""
        date_text = f" date={date}" if date else ""
        if kind == "move":
            city_value = f"from {origin} to {destination}"
            lines.append(
                f"- day={day}{date_text} kind=move "
                f"current_city={json.dumps(city_value, ensure_ascii=False)} "
                "transportation=choose_exact_catalog_option_for_this_route"
            )
        else:
            city_value = destination or "unknown"
            lines.append(
                f"- day={day}{date_text} kind=stay "
                f"current_city={json.dumps(city_value, ensure_ascii=False)} "
                'transportation="-"'
            )
    return lines


def _format_days(days: Sequence[int]) -> str:
    return ",".join(str(day) for day in days) if days else "none"


def _render_restaurant_catalog(
    catalog: ReferenceCatalog,
    *,
    example: Mapping[str, Any],
    scaffold: Sequence[Tuple[str, str, str]],
) -> str:
    constraints = _constraints(example)
    requested = set(_requested_cuisines(constraints))
    people = max(1, int(example.get("people_number", 1) or 1))

    def cuisine_hits(entry: CatalogEntry) -> Tuple[str, ...]:
        available = {canonical_value(value) for value in entry.cuisines}
        return tuple(sorted(requested & available))

    restaurants = _unique_sorted(
        catalog.restaurants,
        key=lambda entry: (
            entry.city.casefold(),
            not bool(
                _entry_cost_cents(entry, people) is not None
                and _city_days(entry.city, scaffold, stays_only=True)
            ),
            not bool(cuisine_hits(entry)),
            _cost_sort_key(_entry_cost_cents(entry, people)),
            entry.name.casefold(),
        ),
    )
    lines = [
        "RESTAURANT OPTIONS",
        "Copy the quoted JSON value exactly. team_cost_per_meal already "
        "includes the full party.",
    ]
    for entry in restaurants:
        value = f"{entry.name}, {entry.city}"
        cuisines = ", ".join(entry.cuisines) or "unknown"
        cost = _entry_cost_cents(entry, people)
        stay_days = _city_days(entry.city, scaffold, stays_only=True)
        eligible = cost is not None and bool(stay_days)
        lines.append(
            f"- {json.dumps(value, ensure_ascii=False)}"
            f" | team_cost_per_meal={_format_cents(cost)}"
            f" | days={_format_days(stay_days)} | cuisines={cuisines}"
            f" | eligible={_yes_no(eligible)}"
        )
    return "\n".join(lines)


def _render_logistics_catalog(
    catalog: ReferenceCatalog,
    *,
    example: Mapping[str, Any],
    scaffold: Sequence[Tuple[str, str, str]],
) -> str:
    constraints = _constraints(example)
    people = max(1, int(example.get("people_number", 1) or 1))
    dates = coerce_trip_dates(example)

    def transport_hard_ok(entry: CatalogEntry) -> bool:
        return transportation_satisfies_rule(
            entry, constraints.get("transportation")
        )

    def transport_eligible(entry: CatalogEntry) -> bool:
        return bool(
            transport_hard_ok(entry)
            and _entry_cost_cents(entry, people) is not None
            and _transport_days(entry, scaffold, dates)
        )

    def accommodation_room_ok(entry: CatalogEntry) -> bool:
        return accommodation_satisfies_room_type(
            entry, constraints.get("room type")
        )

    def accommodation_rule_ok(entry: CatalogEntry) -> bool:
        return accommodation_satisfies_house_rule(
            entry, constraints.get("house rule")
        )

    def accommodation_eligible(entry: CatalogEntry) -> bool:
        return bool(
            accommodation_room_ok(entry)
            and accommodation_rule_ok(entry)
            and _entry_cost_cents(entry, people) is not None
            and _accommodation_days(entry, scaffold)
        )

    transport = _unique_sorted(
        catalog.transportation,
        key=lambda entry: (
            not transport_eligible(entry),
            entry.date or "9999-99-99",
            entry.origin.casefold(),
            entry.destination.casefold(),
            _cost_sort_key(_entry_cost_cents(entry, people)),
            entry.mode,
            entry.name.casefold(),
        ),
    )
    accommodations = _unique_sorted(
        catalog.accommodations,
        key=lambda entry: (
            entry.city.casefold(),
            not accommodation_eligible(entry),
            _cost_sort_key(_entry_cost_cents(entry, people)),
            entry.name.casefold(),
        ),
    )
    lines = [
        "TRANSPORTATION OPTIONS",
        "Copy the quoted JSON value exactly. team_cost already includes "
        "the full party.",
    ]
    for entry in transport:
        value = _canonical_transport_value(entry)
        cost = _entry_cost_cents(entry, people)
        valid_days = _transport_days(entry, scaffold, dates)
        eligible = transport_eligible(entry)
        lines.append(
            f"- {json.dumps(value, ensure_ascii=False)}"
            f" | team_cost={_format_cents(cost)}"
            f" | days={_format_days(valid_days)}"
            f" | eligible={_yes_no(eligible)}"
        )
    lines.extend(
        [
            "",
            "ACCOMMODATION OPTIONS",
            "Copy the quoted JSON value exactly. team_cost_per_night "
            "includes all required rooms.",
        ]
    )
    for entry in accommodations:
        value = f"{entry.name}, {entry.city}"
        cost = _entry_cost_cents(entry, people)
        usable_days = _accommodation_days(entry, scaffold)
        eligible = accommodation_eligible(entry)
        lines.append(
            f"- {json.dumps(value, ensure_ascii=False)}"
            f" | team_cost_per_night={_format_cents(cost)}"
            f" | days={_format_days(usable_days)}"
            f" | room={entry.room_type}"
            f" | rules={entry.house_rules or 'none'}"
            f" | min_nights={_number(entry.minimum_nights)}"
            f" | eligible={_yes_no(eligible)}"
        )
    return "\n".join(lines)


def _render_experience_catalog(
    catalog: ReferenceCatalog,
    *,
    example: Mapping[str, Any],
    scaffold: Sequence[Tuple[str, str, str]],
) -> str:
    attractions = _unique_sorted(
        catalog.attractions,
        key=lambda entry: (entry.city.casefold(), entry.name.casefold()),
    )
    lines = [
        _render_restaurant_catalog(
            catalog, example=example, scaffold=scaffold
        )
    ]
    lines.extend(
        [
            "",
            "ATTRACTION OPTIONS",
            "On each stay day copy exactly one quoted JSON value; never join "
            "multiple attractions.",
        ]
    )
    for entry in attractions:
        value = f"{entry.name}, {entry.city}"
        stay_days = _city_days(entry.city, scaffold, stays_only=True)
        lines.append(
            f"- {json.dumps(value, ensure_ascii=False)}"
            f" | days={_format_days(stay_days)}"
            f" | eligible={_yes_no(bool(stay_days))}"
        )
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

    scaffold_values = derive_route_scaffold(example, catalog)
    scaffold = "\n".join(_route_scaffold(example, catalog))
    if int(agent_idx) == 0:
        role_catalog = _render_logistics_catalog(
            catalog, example=example, scaffold=scaffold_values
        )
    elif int(agent_idx) == 1:
        role_catalog = _render_experience_catalog(
            catalog, example=example, scaffold=scaffold_values
        )
    else:
        raise ValueError("The initial TravelPlanner task has exactly two agents.")
    return f"""SHARED REFERENCE-DERIVED ROUTE SCAFFOLD
This deterministic scaffold comes only from dated reference routes. It coordinates
movement versus stay days; it does not select a flight, lodging, meal, attraction,
or any target itinerary.
{scaffold}

ROLE-SPECIFIC COMPACT REFERENCE CATALOG
{role_catalog}"""


def _render_role_slot_checklist(
    example: Mapping[str, Any], agent_idx: int
) -> str:
    catalog = parse_reference_catalog(
        str(example.get("reference_information", ""))
    )
    scaffold = derive_route_scaffold(example, catalog)
    lines = ["ROLE DAY-SLOT CHECKLIST (derived only from the shared scaffold)"]
    for day, (kind, origin, destination) in enumerate(scaffold, start=1):
        if int(agent_idx) == 0:
            city_value = (
                f"from {origin} to {destination}"
                if kind == "move"
                else destination
            )
            transport = (
                f"REQUIRED eligible=yes catalog option whose days include {day}"
                if kind == "move"
                else 'EMPTY: use "-"'
            )
            accommodation = (
                'EMPTY: use "-"'
                if day == len(scaffold)
                else (
                    "REQUIRED eligible=yes option whose days include "
                    f"{day}"
                )
            )
            lines.append(
                f"- day={day}: current_city=EXACT "
                f"{json.dumps(city_value, ensure_ascii=False)}; "
                f"transportation={transport}; accommodation={accommodation}."
            )
        elif int(agent_idx) == 1:
            if kind == "move":
                lines.append(
                    f"- day={day}: breakfast, attraction, lunch, dinner are "
                    'EMPTY; use "-" for all four.'
                )
            else:
                lines.append(
                    f"- day={day}: breakfast, lunch, and dinner are REQUIRED distinct "
                    f"eligible restaurant values whose days include {day}; attraction "
                    f"is REQUIRED as exactly one eligible value whose days include {day}."
                )
        else:
            raise ValueError("The initial TravelPlanner task has exactly two agents.")
    return "\n".join(lines)


def _render_budget_contract(example: Mapping[str, Any], agent_idx: int) -> str:
    contract = build_role_budget_contract(example)
    own_name = "LOGISTICS" if int(agent_idx) == 0 else "EXPERIENCE"
    own_cap = contract.cap_for_agent(agent_idx)
    own_floor = contract.floor_for_agent(agent_idx)
    if contract.feasible:
        status = (
            "feasible=yes. These caps include each role's reference-derived minimum "
            "and divide the remaining slack. If both agents respect their own cap, "
            "the merged known cost cannot exceed the team budget."
        )
    else:
        status = (
            f"feasible=no ({contract.reason}). The displayed caps still sum to the "
            "team budget but cannot guarantee a feasible merged plan; minimize cost "
            "while satisfying every grounded required slot."
        )
    return f"""DETERMINISTIC TEAM BUDGET CONTRACT
- team_budget={_format_cents(contract.team_budget_cents)}
- logistics_floor={_format_cents(contract.logistics_floor_cents)}
- experience_floor={_format_cents(contract.experience_floor_cents)}
- logistics_cap={_format_cents(contract.logistics_cap_cents)}
- experience_cap={_format_cents(contract.experience_cap_cents)}
- YOUR_ROLE={own_name}; YOUR_CAP={_format_cents(own_cap)};
  YOUR_REFERENCE_FLOOR={_format_cents(own_floor)}
- {status}
Sum the displayed full-party cost once for every selected occurrence. Transportation
and each accommodation night count toward LOGISTICS; every selected meal counts
toward EXPERIENCE; current_city, attractions, and "-" cost zero. Stay at or below
YOUR_CAP. Budget compliance never excuses a missing required value or an ungrounded
choice."""


def _render_role_hard_constraint_duty(
    example: Mapping[str, Any], agent_idx: int
) -> str:
    constraints = _constraints(example)
    if int(agent_idx) == 0:
        return """ROLE-SPECIFIC HARD-CONSTRAINT DUTY
- Choose only transportation and accommodations marked eligible=yes, on a day
  listed for that option. For lodging, eligible also guarantees that its minimum
  stay can be met by the complete consecutive city run.
- A room-type or house-rule requirement applies to every selected accommodation.
- Reuse one accommodation throughout each consecutive city stay when needed to meet
  its minimum_nights. Never mix self-driving with flight or taxi in one itinerary."""
    if int(agent_idx) == 1:
        requested = _requested_cuisines(constraints)
        coverage = (
            json.dumps(requested, ensure_ascii=False)
            if requested
            else "none"
        )
        return f"""ROLE-SPECIFIC HARD-CONSTRAINT DUTY
- Requested cuisine coverage={coverage}. Across all REQUIRED meal slots, the union
  of selected restaurants' cuisines must include every requested cuisine at least once.
- A restaurant serving no requested cuisine is still eligible for another meal.
- Use a different grounded restaurant for every meal and a different single grounded
  attraction on each stay day; never output multiple attractions in one slot."""
    raise ValueError("The initial TravelPlanner task has exactly two agents.")


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
            "accommodation on every day. Submit every owned "
            "slot, including an explicit '-' when the itinerary convention permits it to be empty. Do not "
            "submit breakfast, attraction, lunch, or dinner slots owned by Agent 1."
        )
    if role_mode == "partitioned_roles" and agent_idx == 1:
        return (
            "You own DAILY EXPERIENCE: breakfast, attraction, lunch, and dinner on every day. "
            "Submit every "
            "owned slot, including an explicit '-' when the itinerary convention permits it to be empty. "
            "Do not submit current_city, transportation, or accommodation slots owned by Agent 0."
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
        reference = build_compact_reference_context(example, agent_idx)
        role = _role_instruction(agent_idx, role_mode, days)
        owned = (
            ordered_owned_slots(agent_idx, days)
            if role_mode == "partitioned_roles"
            else []
        )
        capacity = assignment_capacity(
            days,
            num_agents,
            agent_idx=agent_idx if role_mode == "partitioned_roles" else None,
        )
        target_count = len(owned) if owned else capacity
        owned_slot_text = (
            ", ".join(f"(day {day}, {field})" for day, field in owned)
            if owned
            else "Infer a complementary subset from the shared task."
        )
        if role_mode == "partitioned_roles" and int(num_agents) == 2:
            role_coordination = "\n\n".join(
                (
                    _render_role_slot_checklist(example, agent_idx),
                    _render_role_hard_constraint_duty(example, agent_idx),
                    _render_budget_contract(example, agent_idx),
                )
            )
        else:
            role_coordination = (
                "ROLE-SPECIFIC DETERMINISTIC CHECKLISTS AND BUDGET CAPS ARE DISABLED "
                "OUTSIDE THE TWO-AGENT PARTITIONED ROLE MODE."
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
  Generate only that value's contents, then close the value with a double quote.
- After each closing value quote, the generation system supplies the next fixed
  assignment object in YOUR OWNED SLOTS order. It fixes every day, field, key,
  comma, bracket, and brace; you choose only each value.
- Do not try to generate the next assignment's schema yourself. Immediately continue
  inside the next already-open value string. The system closes the JSON object after
  the final owned value.
- A per-value token limit prevents one slot from consuming the whole response budget.
  Keep every value concise and copy catalog spelling exactly.
- The reconstructed response always has the fixed schema and assignment count shown
  above. Semantic omissions and invalid values still receive low reward.

ASSISTANT PREFILL (already supplied; do not repeat it):
{assistant_prefill}

Continue the prefilled JSON now. Your first generated character is the first
character inside the already-open value string."""
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

{role_coordination}

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
- Write restaurants, attractions, and accommodations as "Name, City". Never join
  multiple attractions: choose exactly one grounded attraction on a required stay day.
  Copy a complete transportation candidate, including its mode or flight number and the
  matching route, rather than returning an abbreviation.
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
