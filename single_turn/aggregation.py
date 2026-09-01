"""Deterministic aggregation for simultaneous TravelPlanner outputs."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from single_turn.parsing import Assignment, ParseResult, parse_assignments


PLAN_FIELDS: Tuple[str, ...] = (
    "current_city",
    "transportation",
    "breakfast",
    "attraction",
    "lunch",
    "dinner",
    "accommodation",
)

LOGISTICS_FIELDS = frozenset({"current_city", "transportation", "accommodation"})
EXPERIENCE_FIELDS = frozenset({"breakfast", "attraction", "lunch", "dinner"})
ROLE_FIELDS = (LOGISTICS_FIELDS, EXPERIENCE_FIELDS)
Slot = Tuple[int, str]


def canonical_value(value: object) -> str:
    text = "-" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text or "-"


def assignment_capacity(days: int, num_agents: int) -> int:
    return int(math.ceil((len(PLAN_FIELDS) * int(days)) / max(1, int(num_agents))))


def slot_owner(day: int, field: str, days: int) -> int:
    """Return the deterministic two-agent owner for one itinerary slot.

    Logistics always belongs to agent 0, daily experience belongs to agent 1,
    and even-day dinner slots move to agent 0 to balance odd-length trips.
    """

    day = int(day)
    days = int(days)
    if not 1 <= day <= days or field not in PLAN_FIELDS:
        raise ValueError(f"Invalid TravelPlanner slot: day={day}, field={field!r}")
    if field in LOGISTICS_FIELDS:
        return 0
    if field == "dinner" and day % 2 == 0:
        return 0
    return 1


def owned_slots(agent_idx: int, days: int) -> Set[Slot]:
    if agent_idx not in {0, 1}:
        raise ValueError("The initial TravelPlanner task has exactly two agents.")
    return {
        (day, field)
        for day in range(1, int(days) + 1)
        for field in PLAN_FIELDS
        if slot_owner(day, field, int(days)) == agent_idx
    }


@dataclass
class MergeResult:
    plan: List[Dict[str, object]]
    parsed: List[ParseResult]
    merged_assignments: Dict[Slot, str]
    per_agent_assignments: List[Dict[Slot, str]]
    conflict_slots: Set[Slot]
    overlap_count: int
    self_duplicate_count: int
    invalid_slot_count: int
    extra_assignment_count: int
    capacity: int

    @property
    def covered_slots(self) -> Set[Slot]:
        return set(self.merged_assignments)

    @property
    def agent_assignment_counts(self) -> List[int]:
        return [len(assignments) for assignments in self.per_agent_assignments]


def _validate_assignment(
    assignment: Assignment,
    *,
    days: int,
) -> Tuple[Slot, str] | None:
    try:
        day = int(assignment.day)
    except (TypeError, ValueError):
        return None
    field = str(assignment.field).strip().lower()
    if not 1 <= day <= days or field not in PLAN_FIELDS:
        return None
    value = "-" if assignment.value is None else str(assignment.value).strip() or "-"
    return (day, field), value


def merge_agent_assignments(
    agent_completions: Sequence[str],
    *,
    days: int,
    capacity: int | None = None,
) -> MergeResult:
    num_agents = len(agent_completions)
    cap = assignment_capacity(days, num_agents) if capacity is None else int(capacity)
    parsed = [
        parse_assignments(
            completion or "",
            expected_agent_id=agent_idx,
            capacity=cap,
            days=days,
            valid_fields=PLAN_FIELDS,
        )
        for agent_idx, completion in enumerate(agent_completions)
    ]

    invalid_slot_count = sum(result.invalid_item_count for result in parsed)
    extra_assignment_count = 0
    self_duplicate_count = 0
    per_agent: List[Dict[Slot, str]] = []

    for result in parsed:
        accepted: Dict[Slot, str] = {}
        # An over-capacity action is atomic and invalid.  Dropping only its tail
        # would still reward the "emit the whole plan and let the merger trim it"
        # policy observed in the first smoke run.
        allowed = result.assignments if result.capacity_valid else []
        extra_assignment_count += max(0, result.raw_item_count - cap)
        for assignment in allowed:
            validated = _validate_assignment(assignment, days=days)
            if validated is None:
                invalid_slot_count += 1
                continue
            slot, value = validated
            if slot in accepted:
                self_duplicate_count += 1
                continue
            accepted[slot] = value
        per_agent.append(accepted)

    by_slot: Dict[Slot, List[str]] = defaultdict(list)
    for assignments in per_agent:
        for slot, value in assignments.items():
            by_slot[slot].append(value)

    merged: Dict[Slot, str] = {}
    conflict_slots: Set[Slot] = set()
    overlap_count = 0
    for slot, values in by_slot.items():
        if len(values) == 1:
            merged[slot] = values[0]
            continue
        canonical = {canonical_value(value) for value in values}
        if len(canonical) == 1:
            merged[slot] = values[0]
            overlap_count += len(values) - 1
        else:
            conflict_slots.add(slot)

    plan: List[Dict[str, object]] = []
    for day in range(1, days + 1):
        row: Dict[str, object] = {"day": day}
        for field in PLAN_FIELDS:
            row[field] = merged.get((day, field), "-")
        plan.append(row)

    return MergeResult(
        plan=plan,
        parsed=parsed,
        merged_assignments=merged,
        per_agent_assignments=per_agent,
        conflict_slots=conflict_slots,
        overlap_count=overlap_count,
        self_duplicate_count=self_duplicate_count,
        invalid_slot_count=invalid_slot_count,
        extra_assignment_count=extra_assignment_count,
        capacity=cap,
    )
