"""Reference-backed TravelPlanner checks with no annotated-plan dependency.

The official sole-planning rows already contain the relevant database slice in
``reference_information``.  This module turns that slice into a small catalog
and evaluates any valid itinerary, rather than comparing it with one human
itinerary.  The checks intentionally mirror the public TravelPlanner metric
families, but are named as local/reference-backed metrics because they do not
import the separate official database evaluator.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from single_turn.aggregation import (
    PLAN_FIELDS,
    MergeResult,
    canonical_value,
    owned_slots,
)


MEAL_FIELDS = frozenset({"breakfast", "lunch", "dinner"})
ENTITY_FIELDS = frozenset(
    {"breakfast", "lunch", "dinner", "attraction", "accommodation"}
)
COST_FIELDS = frozenset((*MEAL_FIELDS, "transportation", "accommodation"))


@dataclass(frozen=True)
class CatalogEntry:
    category: str
    name: str
    city: str = ""
    cost: float | None = None
    cuisines: Tuple[str, ...] = ()
    room_type: str = ""
    house_rules: str = ""
    minimum_nights: float = 1.0
    maximum_occupancy: float = 1.0
    mode: str = ""
    origin: str = ""
    destination: str = ""
    date: str = ""

    @property
    def identity(self) -> Tuple[str, str, str]:
        return (
            self.category,
            canonical_value(self.name),
            canonical_value(self.city),
        )


@dataclass(frozen=True)
class ReferenceCatalog:
    restaurants: Tuple[CatalogEntry, ...]
    attractions: Tuple[CatalogEntry, ...]
    accommodations: Tuple[CatalogEntry, ...]
    transportation: Tuple[CatalogEntry, ...]
    expected_record_count: int
    parsed_record_count: int

    @property
    def parse_success(self) -> bool:
        return bool(
            self.restaurants
            and self.attractions
            and self.accommodations
            and self.transportation
        )

    @property
    def allowed_cities(self) -> frozenset[str]:
        cities = {
            canonical_value(entry.city)
            for entries in (
                self.restaurants,
                self.attractions,
                self.accommodations,
            )
            for entry in entries
            if entry.city
        }
        for entry in self.transportation:
            if entry.origin:
                cities.add(canonical_value(entry.origin))
            if entry.destination:
                cities.add(canonical_value(entry.destination))
        cities.discard("-")
        return frozenset(cities)


def scaled_party_cost(entry: CatalogEntry, people: int) -> float | None:
    """Return the evaluator cost of one selected catalog occurrence.

    This public helper is shared by prompt construction and scoring so the
    decentralized budget contract cannot drift away from the hard evaluator.
    """

    if entry.cost is None:
        return None
    party_size = max(1, int(people or 1))
    if entry.category == "transportation":
        if entry.mode == "flight":
            multiplier = party_size
        elif entry.mode == "taxi":
            multiplier = math.ceil(party_size / 4)
        else:
            multiplier = math.ceil(party_size / 5)
        return float(entry.cost * multiplier)
    if entry.category == "restaurant":
        return float(entry.cost * party_size)
    if entry.category == "accommodation":
        return float(
            entry.cost * math.ceil(party_size / max(1.0, entry.maximum_occupancy))
        )
    return float(entry.cost)


def accommodation_satisfies_room_type(
    entry: CatalogEntry, requested: Any
) -> bool:
    if not requested:
        return True
    actual = canonical_value(entry.room_type)
    required = canonical_value(requested)
    if required == "not shared room":
        return actual != "shared room"
    if required == "entire room":
        return actual == "entire home/apt"
    return actual == required


def accommodation_satisfies_house_rule(
    entry: CatalogEntry, requested: Any
) -> bool:
    if not requested:
        return True
    forbidden = f"no {canonical_value(requested)}"
    return forbidden not in canonical_value(entry.house_rules)


def transportation_satisfies_rule(
    entry: CatalogEntry, requested: Any
) -> bool:
    if not requested:
        return True
    forbidden = {
        "no flight": "flight",
        "no self-driving": "self-driving",
    }.get(canonical_value(requested))
    return forbidden is not None and entry.mode != forbidden


def _safe_records(raw: str) -> List[Dict[str, Any]]:
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(raw)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, list):
            return [dict(record) for record in value if isinstance(record, dict)]
    return []


def _data_lines(content: str) -> List[str]:
    lines = [line.rstrip() for line in str(content or "").splitlines() if line.strip()]
    if not lines or canonical_value(lines[0]).startswith("no valid information"):
        return []
    return lines[1:]


def _strip_city_suffix(line: str, city: str) -> str | None:
    match = re.match(rf"^(.*?)\s+{re.escape(city)}\s*$", line, flags=re.I)
    return match.group(1).strip() if match else None


def _parse_restaurants(city: str, content: str) -> List[CatalogEntry]:
    entries: List[CatalogEntry] = []
    for line in _data_lines(content):
        body = _strip_city_suffix(line, city)
        if body is None:
            continue
        match = re.match(
            r"^(.*?)\s+(\d+(?:\.\d+)?)\s+(.+?)\s+(-?\d+(?:\.\d+)?)$",
            body,
        )
        if not match:
            continue
        name, cost, cuisines, _rating = match.groups()
        entries.append(
            CatalogEntry(
                category="restaurant",
                name=name.strip(),
                city=city,
                cost=float(cost),
                cuisines=tuple(
                    part.strip() for part in cuisines.split(",") if part.strip()
                ),
            )
        )
    return entries


def _parse_attractions(city: str, content: str) -> List[CatalogEntry]:
    entries: List[CatalogEntry] = []
    for line in _data_lines(content):
        body = _strip_city_suffix(line, city)
        if body is None:
            continue
        match = re.match(r"^(.*?)\s+-?\d+\.\d+\s+-?\d+\.\d+\s+.+$", body)
        if match:
            entries.append(
                CatalogEntry(
                    category="attraction",
                    name=match.group(1).strip(),
                    city=city,
                    cost=0.0,
                )
            )
    return entries


def _parse_accommodations(city: str, content: str) -> List[CatalogEntry]:
    entries: List[CatalogEntry] = []
    room_pattern = r"Shared room|Private room|Entire home/apt"
    for line in _data_lines(content):
        body = _strip_city_suffix(line, city)
        if body is None:
            continue
        match = re.match(
            rf"^(.*?)\s+(\d+(?:\.\d+)?)\s+({room_pattern})\s+(.*?)\s+"
            r"(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)$",
            body,
            flags=re.I,
        )
        if not match:
            continue
        name, price, room_type, rules, minimum, occupancy, _review = match.groups()
        entries.append(
            CatalogEntry(
                category="accommodation",
                name=name.strip(),
                city=city,
                cost=float(price),
                room_type=room_type.strip(),
                house_rules=rules.strip(),
                minimum_nights=float(minimum),
                maximum_occupancy=max(1.0, float(occupancy)),
            )
        )
    return entries


def _parse_transport(description: str, content: str) -> List[CatalogEntry]:
    match = re.match(
        r"^(Flight|Self-driving|Taxi)\s+from\s+(.+?)\s+to\s+(.+?)"
        r"(?:\s+on\s+(\d{4}-\d{2}-\d{2}))?$",
        description.strip(),
        flags=re.I,
    )
    if not match or canonical_value(content).startswith("no valid information"):
        return []
    mode, origin, destination, date = match.groups()
    mode = mode.casefold()
    if mode == "flight":
        entries = []
        for line in _data_lines(content):
            row = re.match(r"^\s*(F\w+)\s+(\d+(?:\.\d+)?)\s+", line, flags=re.I)
            if row:
                entries.append(
                    CatalogEntry(
                        category="transportation",
                        name=row.group(1),
                        cost=float(row.group(2)),
                        mode=mode,
                        origin=origin,
                        destination=destination,
                        date=date or "",
                    )
                )
        return entries

    cost_match = re.search(r"\bcost\s*:\s*\$?(\d+(?:\.\d+)?)", content, flags=re.I)
    if not cost_match:
        return []
    return [
        CatalogEntry(
            category="transportation",
            name=mode,
            cost=float(cost_match.group(1)),
            mode=mode,
            origin=origin,
            destination=destination,
            date=date or "",
        )
    ]


@lru_cache(maxsize=512)
def parse_reference_catalog(reference_information: str) -> ReferenceCatalog:
    restaurants: List[CatalogEntry] = []
    attractions: List[CatalogEntry] = []
    accommodations: List[CatalogEntry] = []
    transportation: List[CatalogEntry] = []
    records = _safe_records(reference_information)
    parsed_records = 0
    for record in records:
        description = str(record.get("Description", "")).strip()
        content = str(record.get("Content", ""))
        lowered = description.casefold()
        parsed: List[CatalogEntry] = []
        if lowered.startswith("restaurants in "):
            parsed = _parse_restaurants(description[len("Restaurants in ") :], content)
            restaurants.extend(parsed)
        elif lowered.startswith("attractions in "):
            parsed = _parse_attractions(description[len("Attractions in ") :], content)
            attractions.extend(parsed)
        elif lowered.startswith("accommodations in "):
            parsed = _parse_accommodations(
                description[len("Accommodations in ") :], content
            )
            accommodations.extend(parsed)
        elif re.match(r"^(Flight|Self-driving|Taxi)\s+from\s+", description, re.I):
            parsed = _parse_transport(description, content)
            transportation.extend(parsed)
            # A "No valid information" route is intentionally empty, not a
            # catalog parse failure.
            if canonical_value(content).startswith("no valid information"):
                parsed_records += 1
                continue
        if parsed:
            parsed_records += 1
    return ReferenceCatalog(
        restaurants=tuple(restaurants),
        attractions=tuple(attractions),
        accommodations=tuple(accommodations),
        transportation=tuple(transportation),
        expected_record_count=len(records),
        parsed_record_count=parsed_records,
    )


def coerce_trip_dates(batch_item: Mapping[str, Any]) -> List[str]:
    """Return normalized trip dates without consulting any target itinerary."""

    dates = batch_item.get("dates")
    if not isinstance(dates, (list, tuple)):
        dates = batch_item.get("date", [])
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


def derive_route_scaffold(
    batch_item: Mapping[str, Any],
    catalog: ReferenceCatalog | None = None,
) -> List[Tuple[str, str, str]]:
    """Derive the shared move/stay structure shown to both agents.

    This uses only query metadata and dated reference routes, never an
    annotated or target plan.  The same immutable structure drives both the
    prompt and the reward's required-slot mask, so one agent cannot redefine
    its teammate's workload through its generated ``current_city`` values.
    """

    days = int(batch_item.get("days", 0))
    dates = coerce_trip_dates(batch_item)
    if catalog is None:
        catalog = parse_reference_catalog(
            str(batch_item.get("reference_information", ""))
        )

    routes_by_date: Dict[str, set[Tuple[str, str]]] = {}
    for entry in catalog.transportation:
        if entry.date and entry.origin and entry.destination:
            routes_by_date.setdefault(entry.date, set()).add(
                (entry.origin, entry.destination)
            )

    records = batch_item.get("reference_records")
    if not isinstance(records, list):
        records = _safe_records(str(batch_item.get("reference_information", "")))
    for record in records:
        if not isinstance(record, Mapping):
            continue
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

    current_city = str(batch_item.get("org", "")).strip()
    scaffold: List[Tuple[str, str, str]] = []
    for day in range(1, days + 1):
        date = dates[day - 1] if day <= len(dates) else ""
        candidates = sorted(
            routes_by_date.get(date, ()),
            key=lambda route: (route[0].casefold(), route[1].casefold()),
        )
        matching = [
            route
            for route in candidates
            if current_city
            and canonical_value(route[0]) == canonical_value(current_city)
        ]
        route = (matching or candidates)[0] if candidates else None
        if route is None:
            scaffold.append(("stay", current_city, current_city))
            continue
        origin, destination = route
        scaffold.append(("move", origin, destination))
        current_city = destination
    return scaffold


def parse_current_city(value: Any) -> Tuple[str, str, str] | None:
    text = str(value or "").strip()
    if not text or canonical_value(text) == "-":
        return None
    match = re.fullmatch(r"from\s+(.+?)\s+to\s+(.+?)", text, flags=re.I)
    if match:
        origin = match.group(1).strip()
        destination = match.group(2).strip()
        # A stay day must use the single-city form.  Treating ``from X to X``
        # as a move lets the logistics role erase every experience slot that
        # would otherwise be required on that day.
        if canonical_value(origin) == canonical_value(destination):
            return None
        return (
            "move",
            origin,
            destination,
        )
    return ("stay", text, text)


def match_entity(value: str, entries: Sequence[CatalogEntry]) -> CatalogEntry | None:
    cleaned = str(value or "").strip()
    # Keep the public TravelPlanner output convention explicit. Bare entity
    # names are ambiguous across cities and must not receive grounding credit.
    if "," not in cleaned:
        return None
    raw_name, raw_city = cleaned.rsplit(",", 1)
    name = canonical_value(raw_name)
    city = canonical_value(raw_city)
    if not name or name == "-" or not city or city == "-":
        return None
    exact = [
        entry
        for entry in entries
        if canonical_value(entry.name) == name and canonical_value(entry.city) == city
    ]
    if not exact:
        return None
    identities = {entry.identity for entry in exact}
    return exact[0] if len(identities) == 1 else None


def split_attractions(value: str) -> List[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def match_transport(value: str, catalog: ReferenceCatalog) -> CatalogEntry | None:
    text = str(value or "")
    flight_match = re.fullmatch(
        r"\s*Flight\s+Number\s*:\s*(F\d+)\s*,\s*" r"from\s+(.+?)\s+to\s+(.+?)\s*",
        text,
        flags=re.I,
    )
    if flight_match:
        flight_number, start, end = flight_match.groups()
        matches = [
            entry
            for entry in catalog.transportation
            if entry.mode == "flight"
            and canonical_value(entry.name) == canonical_value(flight_number)
            and canonical_value(entry.origin) == canonical_value(start)
            and canonical_value(entry.destination) == canonical_value(end)
        ]
        return matches[0] if len(matches) == 1 else None

    ground_match = re.fullmatch(
        r"\s*(Self-driving|Taxi)\s*,\s*from\s+(.+?)\s+to\s+(.+?)\s*",
        text,
        flags=re.I,
    )
    if not ground_match:
        return None
    mode, start, end = ground_match.groups()
    candidates = [
        entry
        for entry in catalog.transportation
        if entry.mode == canonical_value(mode)
        and canonical_value(entry.origin) == canonical_value(start)
        and canonical_value(entry.destination) == canonical_value(end)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _harmonic(values: Sequence[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _route_row_matches(
    actual: Tuple[str, str, str] | None,
    expected: Tuple[str, str, str] | None,
) -> bool:
    return bool(
        actual is not None
        and expected is not None
        and actual[0] == expected[0]
        and canonical_value(actual[1]) == canonical_value(expected[1])
        and canonical_value(actual[2]) == canonical_value(expected[2])
    )


def _route_evaluation(
    plan: Sequence[Mapping[str, Any]],
    batch_item: Mapping[str, Any],
    catalog: ReferenceCatalog,
    expected_scaffold: Sequence[Tuple[str, str, str]],
) -> Dict[str, Any]:
    days = int(batch_item.get("days", 0))
    origin = canonical_value(batch_item.get("org", ""))
    parsed = [parse_current_city(row.get("current_city", "-")) for row in plan[:days]]
    current = origin
    day_validity: List[float] = []
    day_cities: List[Tuple[str, ...]] = []
    compressed_route: List[str] = [origin]
    for route in parsed:
        if route is None:
            day_validity.append(0.0)
            day_cities.append(())
            continue
        kind, raw_start, raw_end = route
        start = canonical_value(raw_start)
        end = canonical_value(raw_end)
        continuity = start == current
        day_validity.append(float(continuity))
        day_cities.append((start, end) if kind == "move" else (start,))
        if continuity:
            current = end
            if not compressed_route or compressed_route[-1] != end:
                compressed_route.append(end)

    parse_rate = _mean([float(route is not None) for route in parsed])
    continuity_rate = _mean(day_validity)
    starts_at_origin = float(
        bool(parsed)
        and parsed[0] is not None
        and canonical_value(parsed[0][1]) == origin
    )
    closed_loop = float(bool(parsed) and current == origin and continuity_rate == 1.0)
    visited = {city for city in compressed_route if city and city != origin}
    expected_city_count = int(batch_item.get("visiting_city_number", 0) or 0)
    visiting_city_count = float(len(visited) == expected_city_count)
    allowed = set(catalog.allowed_cities)
    allowed.add(origin)
    all_route_cities = {
        canonical_value(city)
        for route in parsed
        if route is not None
        for city in route[1:]
    }
    allowed_city_rate = (
        _mean([float(city in allowed) for city in all_route_cities], default=0.0)
        if all_route_cities
        else 0.0
    )
    middle = compressed_route[1:-1] if len(compressed_route) >= 2 else []
    no_revisit = float(len(middle) == len(set(middle)))
    destination = canonical_value(batch_item.get("dest", ""))
    destination_applicable = destination in allowed
    destination_reached = float(not destination_applicable or destination in visited)
    scaffold_match_bits = []
    for day in range(days):
        actual = parsed[day] if day < len(parsed) else None
        expected = (
            expected_scaffold[day] if day < len(expected_scaffold) else None
        )
        scaffold_match_bits.append(float(_route_row_matches(actual, expected)))
    scaffold_match_rate = _mean(scaffold_match_bits)
    components = [
        parse_rate,
        continuity_rate,
        starts_at_origin,
        closed_loop,
        visiting_city_count,
        allowed_city_rate,
        no_revisit,
        destination_reached,
        scaffold_match_rate,
    ]
    return {
        "soft": _mean(components),
        "pass": float(all(value >= 1.0 for value in components)),
        "parsed": parsed,
        "day_validity": day_validity,
        "day_cities": day_cities,
        "parse_rate": parse_rate,
        "continuity_rate": continuity_rate,
        "starts_at_origin": starts_at_origin,
        "closed_loop": closed_loop,
        "visiting_city_count": visiting_city_count,
        "allowed_city_rate": allowed_city_rate,
        "no_revisit": no_revisit,
        "destination_reached": destination_reached,
        "scaffold_match_rate": scaffold_match_rate,
    }


def _required_slots(route: Mapping[str, Any], days: int) -> set[Tuple[int, str]]:
    required: set[Tuple[int, str]] = set()
    parsed = route["parsed"]
    for day in range(1, days + 1):
        required.add((day, "current_city"))
        day_route = parsed[day - 1] if day - 1 < len(parsed) else None
        if day_route is None:
            required.update(
                (day, field) for field in PLAN_FIELDS if field != "current_city"
            )
        elif day_route[0] == "move":
            required.add((day, "transportation"))
        else:
            required.update((day, field) for field in (*MEAL_FIELDS, "attraction"))
        if day < days:
            required.add((day, "accommodation"))
    return required


def _field_entries(catalog: ReferenceCatalog, field: str) -> Sequence[CatalogEntry]:
    if field in MEAL_FIELDS:
        return catalog.restaurants
    if field == "attraction":
        return catalog.attractions
    if field == "accommodation":
        return catalog.accommodations
    return ()


def _city_consistent(entry: CatalogEntry, cities: Iterable[str]) -> bool:
    allowed = {canonical_value(city) for city in cities if city}
    return bool(entry.city) and canonical_value(entry.city) in allowed


def _transport_validity(
    value: str,
    *,
    day: int,
    expected_scaffold: Sequence[Tuple[str, str, str]],
    dates: Sequence[str],
    catalog: ReferenceCatalog,
) -> Tuple[float, CatalogEntry | None]:
    day_route = (
        expected_scaffold[day - 1]
        if day - 1 < len(expected_scaffold)
        else None
    )
    if canonical_value(value) == "-":
        return (float(day_route is not None and day_route[0] == "stay"), None)
    entry = match_transport(value, catalog)
    if entry is None or day_route is None or day_route[0] != "move":
        return 0.0, entry
    _, start, end = day_route
    route_matches = canonical_value(entry.origin) == canonical_value(
        start
    ) and canonical_value(entry.destination) == canonical_value(end)
    date_matches = bool(
        entry.mode != "flight"
        or not entry.date
        or day - 1 >= len(dates)
        or entry.date == str(dates[day - 1])
    )
    return float(route_matches and date_matches), entry


def evaluate_reference_plan(
    result: MergeResult,
    *,
    batch_item: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return dense constraint scores and binary end metrics for one team plan."""

    days = int(batch_item.get("days", 0))
    total_slots = max(1, len(PLAN_FIELDS) * days)
    catalog = parse_reference_catalog(str(batch_item.get("reference_information", "")))
    plan = result.plan[:days]
    expected_scaffold = derive_route_scaffold(batch_item, catalog)
    route = _route_evaluation(plan, batch_item, catalog, expected_scaffold)
    # The required work surface is fixed before either policy acts.  It is the
    # same reference-derived scaffold shown in both prompts, not a mask inferred
    # from agent 0's generated route.
    required = _required_slots({"parsed": expected_scaffold}, days)
    dates = coerce_trip_dates(batch_item)

    slot_validity: Dict[Tuple[int, str], float] = {}
    selected_entries: Dict[Tuple[int, str], List[CatalogEntry]] = {}
    grounded_entity_bits: List[float] = []
    city_bits: List[float] = []
    transportation_bits: List[float] = []
    required_fill_bits: List[float] = []
    required_valid_bits: List[float] = []
    nonempty_cost_slots: set[Tuple[int, str]] = set()
    cost_complete = True
    estimated_cost = 0.0
    people = max(1, int(batch_item.get("people_number", 1) or 1))

    for day, row in enumerate(plan, start=1):
        expected_day_route = (
            expected_scaffold[day - 1]
            if day - 1 < len(expected_scaffold)
            else None
        )
        expected_day_cities = (
            (
                expected_day_route[1],
                expected_day_route[2],
            )
            if expected_day_route is not None and expected_day_route[0] == "move"
            else (
                (expected_day_route[1],)
                if expected_day_route is not None
                else ()
            )
        )
        for field in PLAN_FIELDS:
            slot = (day, field)
            value = str(row.get(field, "-") or "-").strip() or "-"
            nonempty = canonical_value(value) != "-"
            if nonempty and field in COST_FIELDS:
                nonempty_cost_slots.add(slot)
            if slot in required:
                required_fill_bits.append(float(nonempty))

            if field == "current_city":
                valid = float(
                    _route_row_matches(
                        route["parsed"][day - 1], expected_day_route
                    )
                    and all(
                        canonical_value(city) in catalog.allowed_cities
                        or canonical_value(city)
                        == canonical_value(batch_item.get("org", ""))
                        for city in expected_day_cities
                    )
                )
                slot_validity[slot] = valid
            elif field == "transportation":
                valid, entry = _transport_validity(
                    value,
                    day=day,
                    expected_scaffold=expected_scaffold,
                    dates=dates,
                    catalog=catalog,
                )
                slot_validity[slot] = valid
                transportation_bits.append(valid)
                if nonempty:
                    grounded_entity_bits.append(float(entry is not None))
                    city_bits.append(valid)
                    if entry is not None:
                        selected_entries[slot] = [entry]
                        party_cost = scaled_party_cost(entry, people)
                        if party_cost is None:
                            cost_complete = False
                        else:
                            estimated_cost += party_cost
                    else:
                        cost_complete = False
            else:
                values = split_attractions(value) if field == "attraction" else [value]
                values = [] if not nonempty else values
                entries = [
                    match_entity(candidate, _field_entries(catalog, field))
                    for candidate in values
                ]
                grounded = bool(entries) and all(entry is not None for entry in entries)
                # Accommodation belongs to the city reached at the end of the
                # day. Other entity fields retain the public evaluator's
                # looser travel-day start/end-city convention.
                valid_cities = (
                    expected_day_cities[-1:]
                    if field == "accommodation"
                    else expected_day_cities
                )
                city_valid = grounded and all(
                    _city_consistent(entry, valid_cities)
                    for entry in entries
                    if entry is not None
                )
                if nonempty:
                    grounded_entity_bits.extend(
                        float(entry is not None) for entry in entries
                    )
                    city_bits.extend(
                        float(
                            entry is not None and _city_consistent(entry, valid_cities)
                        )
                        for entry in entries
                    )
                valid = float(
                    (not nonempty and slot not in required) or (grounded and city_valid)
                )
                slot_validity[slot] = valid
                matched = [entry for entry in entries if entry is not None]
                if matched:
                    selected_entries[slot] = matched
                if nonempty and not grounded:
                    cost_complete = False
                for entry in matched:
                    if field in (*MEAL_FIELDS, "accommodation"):
                        party_cost = scaled_party_cost(entry, people)
                        if party_cost is None:
                            cost_complete = False
                        else:
                            estimated_cost += party_cost

            if slot in required:
                required_valid_bits.append(slot_validity[slot])

    assignment_coverage = len(result.merged_assignments) / total_slots
    required_fill_rate = _mean(required_fill_bits)
    required_grounded_recall = _mean(required_valid_bits)
    grounding_precision = _mean(grounded_entity_bits)
    grounding_f1 = _harmonic([grounding_precision, required_grounded_recall])
    within_city_soft = _mean(city_bits)

    restaurant_ids = [
        entry.identity
        for (day, field), entries in selected_entries.items()
        if field in MEAL_FIELDS
        for entry in entries
    ]
    attraction_ids = [
        entry.identity
        for (day, field), entries in selected_entries.items()
        if field == "attraction"
        for entry in entries
    ]
    restaurant_diversity = (
        len(set(restaurant_ids)) / len(restaurant_ids) if restaurant_ids else 1.0
    )
    attraction_diversity = (
        len(set(attraction_ids)) / len(attraction_ids) if attraction_ids else 1.0
    )

    accommodation_runs: List[Tuple[CatalogEntry, int]] = []
    active_accommodation: CatalogEntry | None = None
    for day in range(1, days + 1):
        entries = selected_entries.get((day, "accommodation"), [])
        entry = entries[0] if len(entries) == 1 else None
        if entry is None:
            active_accommodation = None
            continue
        if (
            active_accommodation is not None
            and accommodation_runs
            and active_accommodation.identity == entry.identity
        ):
            previous, count = accommodation_runs[-1]
            accommodation_runs[-1] = (previous, count + 1)
        else:
            accommodation_runs.append((entry, 1))
        active_accommodation = entry
    minimum_night_bits = [
        float(nights >= entry.minimum_nights) for entry, nights in accommodation_runs
    ]
    minimum_nights_soft = _mean(minimum_night_bits, default=1.0)

    modes = {
        entry.mode
        for (day, field), entries in selected_entries.items()
        if field == "transportation"
        for entry in entries
    }
    mode_compatible = not (
        {"self-driving", "flight"}.issubset(modes)
        or {"self-driving", "taxi"}.issubset(modes)
    )
    transport_soft = _mean(transportation_bits) * float(mode_compatible)
    reference_parse_success = float(catalog.parse_success)
    sandbox_soft = _mean(
        [grounding_precision, route["allowed_city_rate"], reference_parse_success]
    )
    complete_soft = _mean([assignment_coverage, required_fill_rate])

    commonsense_soft_values = {
        "reasonable_route": float(route["soft"]),
        "restaurant_diversity": float(restaurant_diversity),
        "attraction_diversity": float(attraction_diversity),
        "minimum_nights": float(minimum_nights_soft),
        "transport_consistency": float(transport_soft),
        "within_current_city": float(within_city_soft),
        "within_reference": float(sandbox_soft),
        "complete_information": float(complete_soft),
    }
    commonsense_pass_values = {
        key: float(value >= 1.0 - 1e-9)
        for key, value in commonsense_soft_values.items()
    }

    required_cost_slots = {
        slot for slot in required if slot[1] in COST_FIELDS
    }
    required_cost_known_bits = [
        float(
            bool(selected_entries.get(slot))
            and all(entry.cost is not None for entry in selected_entries[slot])
        )
        for slot in required_cost_slots
    ]
    required_cost_completeness = _mean(
        required_cost_known_bits, default=0.0
    )
    budget_cost_slots = required_cost_slots | nonempty_cost_slots
    budget_cost_known_bits = [
        float(
            bool(selected_entries.get(slot))
            and all(entry.cost is not None for entry in selected_entries[slot])
        )
        for slot in budget_cost_slots
    ]
    budget_cost_completeness = _mean(
        budget_cost_known_bits, default=0.0
    )

    budget = float(batch_item.get("budget", 0) or 0)
    if budget <= 0:
        budget_margin_score = 0.0
    elif estimated_cost <= budget:
        budget_margin_score = 1.0
    else:
        budget_margin_score = max(
            0.0, 1.0 - (estimated_cost - budget) / budget
        )
    # Dense training surface: retain grounding and known-price progress even
    # when one malformed entity keeps the strict terminal budget check at zero.
    # A fully grounded, fully costed, within-budget plan still maps to exactly 1.
    budget_soft = (
        required_grounded_recall
        * budget_cost_completeness
        * budget_margin_score
    )
    budget_pass = float(
        cost_complete
        and budget > 0
        and estimated_cost <= budget
        and required_grounded_recall >= 1.0 - 1e-9
    )

    constraints = batch_item.get("local_constraint", {}) or {}
    hard_soft: Dict[str, float] = {"budget": float(budget_soft)}
    hard_pass: Dict[str, float] = {"budget": budget_pass}

    requested_cuisines = constraints.get("cuisine")
    if requested_cuisines:
        requested = (
            list(requested_cuisines)
            if isinstance(requested_cuisines, (list, tuple, set))
            else [requested_cuisines]
        )
        available = {
            canonical_value(cuisine)
            for entries in selected_entries.values()
            for entry in entries
            if entry.category == "restaurant"
            for cuisine in entry.cuisines
        }
        cuisine_score = _mean(
            [float(canonical_value(cuisine) in available) for cuisine in requested]
        )
        hard_soft["cuisine"] = cuisine_score
        hard_pass["cuisine"] = float(cuisine_score >= 1.0)

    house_rule = constraints.get("house rule")
    if house_rule:
        accommodations = [
            entry
            for entries in selected_entries.values()
            for entry in entries
            if entry.category == "accommodation"
        ]
        rule_bits = [
            float(accommodation_satisfies_house_rule(entry, house_rule))
            for entry in accommodations
        ]
        rule_score = _mean(rule_bits)
        hard_soft["room_rule"] = rule_score
        hard_pass["room_rule"] = float(bool(rule_bits) and rule_score >= 1.0)

    requested_room = constraints.get("room type")
    if requested_room:
        accommodations = [
            entry
            for entries in selected_entries.values()
            for entry in entries
            if entry.category == "accommodation"
        ]
        room_bits = [
            float(accommodation_satisfies_room_type(entry, requested_room))
            for entry in accommodations
        ]
        room_score = _mean(room_bits)
        hard_soft["room_type"] = room_score
        hard_pass["room_type"] = float(bool(room_bits) and room_score >= 1.0)

    transport_rule = constraints.get("transportation")
    if transport_rule:
        recognized_rule = canonical_value(transport_rule) in {
            "no flight",
            "no self-driving",
        }
        eligible_modes = [
            transportation_satisfies_rule(entry, transport_rule)
            for entries in selected_entries.values()
            for entry in entries
            if entry.category == "transportation"
        ]
        transport_score = float(recognized_rule and all(eligible_modes))
        hard_soft["transportation"] = transport_score
        hard_pass["transportation"] = transport_score

    commonsense_soft = _mean(list(commonsense_soft_values.values()))
    commonsense_pass_count = sum(commonsense_pass_values.values())
    commonsense_micro = commonsense_pass_count / len(commonsense_pass_values)
    commonsense_macro = float(commonsense_pass_count == len(commonsense_pass_values))
    hard_micro_soft = _mean(list(hard_soft.values()))
    hard_pass_count = sum(hard_pass.values())
    hard_micro = hard_pass_count / len(hard_pass)
    hard_macro = float(hard_pass_count == len(hard_pass))
    final_plan_success = float(commonsense_macro and hard_macro)
    plan_nonempty = float(
        len(plan) == days
        and any(
            canonical_value(row.get(field, "-")) != "-"
            for row in plan
            for field in PLAN_FIELDS
        )
    )
    roles_complete = bool(
        len(result.parsed) == 2
        and len(result.per_agent_assignments) == 2
        and all(
            parsed.parse_success
            and set(assignments) == owned_slots(agent_idx, days)
            and len(assignments) == parsed.raw_item_count
            for agent_idx, (parsed, assignments) in enumerate(
                zip(result.parsed, result.per_agent_assignments)
            )
        )
    )
    clean_merge = bool(
        result.overlap_count == 0
        and not result.conflict_slots
        and result.invalid_slot_count == 0
        and result.self_duplicate_count == 0
        and result.extra_assignment_count == 0
    )
    # Keep the paper-facing delivery metric separate from correctness: it asks
    # only whether the system handed back a non-empty plan.  The stricter local
    # completion diagnostic below additionally requires two complete role
    # actions and a value for every scaffold-required slot.
    required_plan_completion = float(
        len(plan) == days
        and roles_complete
        and clean_merge
        and assignment_coverage >= 1.0 - 1e-9
        and required_fill_rate >= 1.0 - 1e-9
    )

    details: Dict[str, Any] = {
        "assignment_coverage": float(assignment_coverage),
        "required_fill_rate": float(required_fill_rate),
        "required_grounded_recall": float(required_grounded_recall),
        "entity_grounding_precision": float(grounding_precision),
        "grounding_f1": float(grounding_f1),
        "commonsense_soft": float(commonsense_soft),
        "hard_constraint_soft": float(hard_micro_soft),
        "required_cost_completeness": float(required_cost_completeness),
        "budget_cost_completeness": float(budget_cost_completeness),
        "budget_margin_score": float(budget_margin_score),
        "budget_constraint_soft": float(budget_soft),
        "reference_parse_success": reference_parse_success,
        "estimated_cost": float(estimated_cost),
        "budget": float(budget),
        "budget_utilization": float(estimated_cost / budget) if budget > 0 else 0.0,
        "cost_complete": float(cost_complete),
        "reference_plan_nonempty": plan_nonempty,
        "ultimate/reference_plan_delivery": plan_nonempty,
        "ultimate/required_plan_completion": required_plan_completion,
        "ultimate/reference_grounding": float(grounding_precision >= 1.0 - 1e-9),
        "ultimate/reference_reasonable_route": float(route["pass"]),
        "ultimate/reference_route_scaffold_match": float(
            route["scaffold_match_rate"] >= 1.0 - 1e-9
        ),
        "ultimate/reference_complete_information": commonsense_pass_values[
            "complete_information"
        ],
        "ultimate/reference_within_current_city": commonsense_pass_values[
            "within_current_city"
        ],
        "ultimate/reference_within_reference": commonsense_pass_values[
            "within_reference"
        ],
        "ultimate/reference_transport_consistency": commonsense_pass_values[
            "transport_consistency"
        ],
        "ultimate/reference_restaurant_diversity": commonsense_pass_values[
            "restaurant_diversity"
        ],
        "ultimate/reference_attraction_diversity": commonsense_pass_values[
            "attraction_diversity"
        ],
        "ultimate/reference_minimum_nights": commonsense_pass_values["minimum_nights"],
        "ultimate/reference_commonsense_micro": float(commonsense_micro),
        "ultimate/reference_commonsense_macro": commonsense_macro,
        "ultimate/reference_budget_pass": budget_pass,
        "ultimate/reference_hard_micro": float(hard_micro),
        "ultimate/reference_hard_macro": hard_macro,
        "ultimate/reference_plan_success": final_plan_success,
        "route_parse_rate": float(route["parse_rate"]),
        "route_continuity": float(route["continuity_rate"]),
        "route_closed_loop": float(route["closed_loop"]),
        "route_visiting_city_count": float(route["visiting_city_count"]),
        "route_allowed_city_rate": float(route["allowed_city_rate"]),
        "route_scaffold_match_rate": float(route["scaffold_match_rate"]),
        "slot_validity": slot_validity,
        # Internal structured surface consumed by the decentralized evaluator.
        # It is removed before scalar logging, just like ``slot_validity``.
        "required_slot_validity": {
            slot: slot_validity.get(slot, 0.0) for slot in required
        },
        "_aggregate/commonsense_pass_count": float(commonsense_pass_count),
        "_aggregate/commonsense_applicable_count": float(len(commonsense_pass_values)),
        "_aggregate/hard_pass_count": float(hard_pass_count),
        "_aggregate/hard_applicable_count": float(len(hard_pass)),
        "_aggregate/required_valid_count": float(sum(required_valid_bits)),
        "_aggregate/required_slot_count": float(len(required_valid_bits)),
        "_aggregate/grounded_entity_count": float(sum(grounded_entity_bits)),
        "_aggregate/predicted_entity_count": float(len(grounded_entity_bits)),
        "_aggregate/required_cost_known_count": float(
            sum(required_cost_known_bits)
        ),
        "_aggregate/required_cost_slot_count": float(
            len(required_cost_known_bits)
        ),
        "reference_catalog_counts": {
            "restaurants": len(catalog.restaurants),
            "attractions": len(catalog.attractions),
            "accommodations": len(catalog.accommodations),
            "transportation": len(catalog.transportation),
        },
    }
    for key, value in commonsense_soft_values.items():
        details[f"constraint_soft/{key}"] = float(value)
    for key, value in commonsense_pass_values.items():
        details[f"constraint_pass/{key}"] = float(value)
    for key, value in hard_soft.items():
        details[f"hard_soft/{key}"] = float(value)
    for key, value in hard_pass.items():
        details[f"hard_pass/{key}"] = float(value)
    return details
