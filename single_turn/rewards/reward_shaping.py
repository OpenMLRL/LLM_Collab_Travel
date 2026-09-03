"""Reward-only recovery and shaping for malformed Travel actions.

The benchmark evaluator deliberately remains strict.  This module provides a
separate, bounded surface for policy optimisation so that a completion which
has learned part of the JSON/action contract is distinguishable from an empty
or copied completion.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from single_turn.aggregation import (
    PLAN_FIELDS,
    MergeResult,
    canonical_value,
    merge_agent_assignments,
    owned_slots,
)
from single_turn.parsing import Assignment, parse_assignments
from single_turn.rewards.reference_evaluator import (
    ReferenceCatalog,
    match_entity,
    match_transport,
    parse_reference_catalog,
    split_attractions,
)


_QUOTED = r'''(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')'''
_KEY = r'''(?:"{name}"|'{name}'|{name})'''
_TRIPLE_RE = re.compile(
    _KEY.format(name="day")
    + r"\s*:\s*(?P<day>\d+)\s*,\s*"
    + _KEY.format(name="field")
    + rf"\s*:\s*(?P<field>{_QUOTED})\s*,\s*"
    + _KEY.format(name="value")
    + rf"\s*:\s*(?P<value>{_QUOTED})",
    flags=re.I | re.S,
)
_AGENT_ID_RE = re.compile(
    _KEY.format(name="agent_id") + r"\s*:\s*(?P<agent_id>\d+)",
    flags=re.I,
)
_ASSIGNMENTS_RE = re.compile(
    _KEY.format(name="assignments") + r"\s*:\s*(?P<container>.)?",
    flags=re.I | re.S,
)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _decode_quoted(value: str) -> str | None:
    for loader in (json.loads, ast.literal_eval):
        try:
            decoded = loader(value)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        return decoded if isinstance(decoded, str) else None
    return None


def recover_assignment_triples(text: str) -> List[Assignment]:
    """Recover only complete ``day, field, value`` triples with quoted values."""

    recovered: List[Assignment] = []
    for match in _TRIPLE_RE.finditer(text or ""):
        field = _decode_quoted(match.group("field"))
        value = _decode_quoted(match.group("value"))
        if field is None or value is None:
            continue
        recovered.append(Assignment(int(match.group("day")), field, value))
    return recovered


def _balanced_and_closed(text: str) -> bool:
    """Check JSON-style brackets while ignoring delimiters inside strings."""

    stripped = (text or "").strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return False
    stack: List[str] = []
    quote = ""
    escaped = False
    pairs = {"}": "{", "]": "["}
    for character in stripped:
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = ""
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            if not stack or stack.pop() != pairs[character]:
                return False
    return not quote and not stack


def _normalise_candidate(
    assignment: Assignment,
    *,
    agent_idx: int,
    days: int,
) -> Tuple[Tuple[int, str], str] | None:
    if isinstance(assignment.day, bool):
        return None
    try:
        day = int(assignment.day)
    except (TypeError, ValueError):
        return None
    field = str(assignment.field).strip().casefold()
    if not 1 <= day <= days or field not in PLAN_FIELDS:
        return None
    if (day, field) not in owned_slots(agent_idx, days):
        return None
    if not isinstance(assignment.value, str) or not assignment.value.strip():
        return None
    return (day, field), assignment.value.strip()


def _agent_recovery(
    text: str,
    *,
    agent_idx: int,
    days: int,
) -> Tuple[Dict[Tuple[int, str], str], Dict[str, float]]:
    expected = owned_slots(agent_idx, days)
    capacity = len(expected)
    parsed = parse_assignments(
        text or "",
        expected_agent_id=agent_idx,
        capacity=capacity,
        days=days,
        valid_fields=PLAN_FIELDS,
    )
    regex_assignments = recover_assignment_triples(text)
    # Parsed assignments and regex matches often describe the same objects.
    # Deduplicate before recovering the owned subset.
    candidates: List[Assignment] = []
    seen_raw = set()
    for candidate in [*parsed.assignments, *regex_assignments]:
        signature = (
            repr(candidate.day),
            repr(candidate.field),
            repr(candidate.value),
        )
        if signature not in seen_raw:
            seen_raw.add(signature)
            candidates.append(candidate)

    raw_candidate_count = max(parsed.raw_item_count, len(regex_assignments))
    over_capacity = raw_candidate_count > capacity
    accepted: Dict[Tuple[int, str], str] = {}
    valid_triple_count = 0
    for candidate in candidates:
        normalised = _normalise_candidate(
            candidate,
            agent_idx=agent_idx,
            days=days,
        )
        if normalised is None:
            continue
        slot, value = normalised
        valid_triple_count += 1
        # Reward recovery is deliberately non-atomic: one extra assignment
        # must not erase an otherwise useful agent contribution.  Strict
        # evaluation still rejects the whole over-capacity action.
        accepted.setdefault(slot, value)

    overflow_rate = max(0, raw_candidate_count - capacity) / max(
        1, raw_candidate_count
    )
    rejected_rate = max(0, raw_candidate_count - len(accepted)) / max(
        1, raw_candidate_count
    )

    stripped = (text or "").strip()
    id_match = _AGENT_ID_RE.search(stripped)
    assignments_match = _ASSIGNMENTS_RE.search(stripped)
    assignments_container = (
        assignments_match.group("container") if assignments_match else ""
    )
    signals = {
        "object_start": float(stripped.startswith("{")),
        "agent_id_key": float(id_match is not None),
        "agent_id_match": float(
            id_match is not None and int(id_match.group("agent_id")) == agent_idx
        ),
        "assignments_key": float(assignments_match is not None),
        "assignments_list": float(assignments_container == "["),
        "assignments_string": float(assignments_container in {'"', "'"}),
        "owned_triple_progress": len(accepted) / max(1, len(expected)),
        "closed_delimiters": float(_balanced_and_closed(stripped)),
        "regex_triple_count": float(len(regex_assignments)),
        "recovered_owned_count": float(len(accepted)),
        "recovery_over_capacity": float(over_capacity),
        "recovery_overflow_rate": float(overflow_rate),
        "recovery_rejected_rate": float(rejected_rate),
    }
    signals["format_progress"] = (
        0.10 * signals["object_start"]
        + 0.10 * signals["agent_id_key"]
        + 0.10 * signals["agent_id_match"]
        + 0.10 * signals["assignments_key"]
        + 0.20 * signals["assignments_list"]
        + 0.25 * signals["owned_triple_progress"]
        + 0.15 * signals["closed_delimiters"]
    )
    signals["valid_triple_count"] = float(valid_triple_count)
    return accepted, signals


def _synthetic_merge(
    assignments: Sequence[Mapping[Tuple[int, str], str]],
    *,
    days: int,
) -> MergeResult:
    completions = []
    for agent_idx, per_agent in enumerate(assignments):
        payload = {
            "agent_id": agent_idx,
            "assignments": [
                {"day": day, "field": field, "value": value}
                for (day, field), value in sorted(per_agent.items())
            ],
        }
        completions.append(json.dumps(payload, ensure_ascii=False))
    return merge_agent_assignments(completions, days=days)


def _slot_value_is_grounded(
    slot: Tuple[int, str],
    value: str,
    catalog: ReferenceCatalog,
    origin: str,
) -> bool:
    text = str(value or "").strip()
    if not text or canonical_value(text) == "-":
        return False
    field = slot[1]
    canonical = canonical_value(text)
    allowed_cities = set(catalog.allowed_cities)
    allowed_cities.add(canonical_value(origin))

    if field == "current_city":
        route = re.fullmatch(r"from\s+(.+?)\s+to\s+(.+?)", text, flags=re.I)
        if route:
            return all(
                canonical_value(city) in allowed_cities for city in route.groups()
            )
        return canonical in allowed_cities
    if field == "transportation":
        return match_transport(text, catalog) is not None
    if field in {"breakfast", "lunch", "dinner"}:
        return match_entity(text, catalog.restaurants) is not None
    if field == "accommodation":
        return match_entity(text, catalog.accommodations) is not None
    if field == "attraction":
        candidates = split_attractions(text)
        return bool(candidates) and all(
            match_entity(candidate, catalog.attractions) is not None
            for candidate in candidates
        )
    return False


def _recovered_grounding(
    recovered: Sequence[Mapping[Tuple[int, str], str]],
    *,
    batch_item: Mapping[str, Any],
) -> Dict[str, float]:
    catalog = parse_reference_catalog(str(batch_item.get("reference_information", "")))
    target = max(1, int(batch_item.get("days", 0) or 0))
    origin = str(batch_item.get("org", ""))
    per_agent_scores: List[float] = []
    useful_count = 0
    grounded_count = 0
    unique_grounded: set[str] = set()
    for assignments in recovered:
        useful = [
            (slot, value)
            for slot, value in assignments.items()
            if canonical_value(value) != "-"
        ]
        grounded = [
            (slot, value)
            for slot, value in useful
            if _slot_value_is_grounded(slot, value, catalog, origin)
        ]
        # Field-aware matching prevents category swaps, while value-only
        # uniqueness prevents repeating one restaurant across meal fields from
        # manufacturing extra progress.
        agent_unique = {canonical_value(value) for _slot, value in grounded}
        precision = len(grounded) / len(useful) if useful else 0.0
        progress = min(1.0, len(agent_unique) / target)
        per_agent_scores.append(precision * (progress**0.5))
        useful_count += len(useful)
        grounded_count += len(grounded)
        unique_grounded.update(agent_unique)

    mean_score = sum(per_agent_scores) / max(1, len(per_agent_scores))
    balanced_score = min(per_agent_scores, default=0.0)
    # Keep one agent's partial progress visible without allowing that agent to
    # fully substitute for its teammate.
    score = 0.5 * mean_score + 0.5 * balanced_score
    precision = grounded_count / useful_count if useful_count else 0.0
    progress = min(1.0, len(unique_grounded) / target)
    return {
        "quoted_value_count": float(
            sum(len(assignments) for assignments in recovered)
        ),
        "quoted_useful_value_count": float(useful_count),
        "quoted_grounded_count": float(grounded_count),
        "quoted_unique_grounded_count": float(len(unique_grounded)),
        "quoted_grounding_precision": float(precision),
        "quoted_grounding_progress": float(progress),
        "quoted_grounding_score": float(score),
    }


def _token_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:[.$:+/-][a-z0-9]+)*", (text or "").casefold())


@lru_cache(maxsize=512)
def _reference_ngrams(reference: str, ngram_size: int) -> frozenset[Tuple[str, ...]]:
    reference_tokens = _token_words(reference)
    return frozenset(
        tuple(reference_tokens[index : index + ngram_size])
        for index in range(len(reference_tokens) - ngram_size + 1)
    )


def _reference_copy_rate(text: str, reference: str, ngram_size: int = 8) -> float:
    output_tokens = _token_words(text)
    reference_ngrams = _reference_ngrams(reference, ngram_size)
    if len(output_tokens) < ngram_size or not reference_ngrams:
        return 0.0
    output_ngrams = [
        tuple(output_tokens[index : index + ngram_size])
        for index in range(len(output_tokens) - ngram_size + 1)
    ]
    raw_rate = sum(ngram in reference_ngrams for ngram in output_ngrams) / len(
        output_ngrams
    )
    # A legitimate long venue name can share an occasional eight-token span
    # with its catalog row.  Penalise sustained copying, not that necessary
    # grounding overlap.
    return _clamp((raw_rate - 0.20) / 0.60)


@dataclass(frozen=True)
class RewardShapingSurface:
    recovered_merge: MergeResult
    details: Dict[str, float]


def build_reward_shaping_surface(
    completions: Sequence[str],
    *,
    batch_item: Mapping[str, Any],
) -> RewardShapingSurface:
    """Build bounded dense signals without changing benchmark evaluation."""

    days = int(batch_item.get("days", 0) or 0)
    if days <= 0 or len(completions) != 2:
        raise ValueError("Reward shaping requires two agents and positive days.")
    recovered: List[Dict[Tuple[int, str], str]] = []
    agent_signals: List[Dict[str, float]] = []
    for agent_idx, text in enumerate(completions):
        assignments, signals = _agent_recovery(
            text,
            agent_idx=agent_idx,
            days=days,
        )
        recovered.append(assignments)
        agent_signals.append(signals)
    recovered_merge = _synthetic_merge(recovered, days=days)

    coverages = [
        len(assignments) / max(1, len(owned_slots(agent_idx, days)))
        for agent_idx, assignments in enumerate(recovered)
    ]
    mean_coverage = sum(coverages) / len(coverages)
    balanced_coverage = min(coverages)
    assignment_progress = 0.8 * mean_coverage + 0.2 * balanced_coverage

    format_progress = sum(
        signals["format_progress"] for signals in agent_signals
    ) / len(agent_signals)
    quoted = _recovered_grounding(recovered, batch_item=batch_item)

    reference = str(batch_item.get("reference_information", ""))
    copy_rates = [_reference_copy_rate(text, reference) for text in completions]
    overlong_rates = []
    for agent_idx, text in enumerate(completions):
        # A valid assignment normally needs far below 180 characters.  The
        # slack accommodates long venue names without tolerating raw tables.
        char_limit = 240 + 180 * len(owned_slots(agent_idx, days))
        overlong_rates.append(
            _clamp((len(text or "") - char_limit) / max(1, char_limit))
        )

    details: Dict[str, float] = {
        "format_progress": float(format_progress),
        "recovered_assignment_mean_coverage": float(mean_coverage),
        "recovered_assignment_balanced_coverage": float(balanced_coverage),
        "assignment_progress": float(assignment_progress),
        "reference_copy_rate": float(max(copy_rates, default=0.0)),
        "overlong_rate": float(max(overlong_rates, default=0.0)),
        "recovery_violation_rate": float(
            max(
                max(
                    signals["recovery_overflow_rate"],
                    signals["recovery_rejected_rate"],
                )
                for signals in agent_signals
            )
        ),
        **quoted,
    }
    for agent_idx, signals in enumerate(agent_signals):
        for key, value in signals.items():
            details[f"agent_{agent_idx}/{key}"] = float(value)
        details[f"agent_{agent_idx}/recovered_owned_coverage"] = float(
            coverages[agent_idx]
        )
        details[f"agent_{agent_idx}/reference_copy_rate"] = float(
            copy_rates[agent_idx]
        )
        details[f"agent_{agent_idx}/overlong_rate"] = float(
            overlong_rates[agent_idx]
        )
    return RewardShapingSurface(recovered_merge=recovered_merge, details=details)
