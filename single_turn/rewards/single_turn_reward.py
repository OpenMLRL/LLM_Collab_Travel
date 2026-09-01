"""Shared reward for one-shot, role-guided TravelPlanner collaboration.

The phase-one reward is deliberately self-contained: it uses the annotated plan
and sole-planning reference information shipped in the official train split.
It does not require the separate TravelPlanner database/evaluator checkout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from single_turn.aggregation import (
    EXPERIENCE_FIELDS,
    LOGISTICS_FIELDS,
    PLAN_FIELDS,
    MergeResult,
    canonical_value,
    merge_agent_assignments,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _tokens(value: Any) -> List[str]:
    text = canonical_value(value)
    text = re.sub(r"\bcost\s*:?\s*\$?\d+(?:\.\d+)?(?:\s*for\s*\d+\s*nights?)?", " ", text)
    return re.findall(r"[a-z0-9]+", text)


def _token_f1(predicted: Any, gold: Any) -> float:
    pred_tokens = _tokens(predicted)
    gold_tokens = _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts: Dict[str, int] = {}
    gold_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    overlap = sum(
        min(count, gold_counts.get(token, 0)) for token, count in pred_counts.items()
    )
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _split_entities(value: Any) -> List[str]:
    text = str(value or "").strip()
    return [part.strip() for part in text.split(";") if part.strip()]


def _attraction_similarity(predicted: Any, gold: Any) -> float:
    pred_entities = _split_entities(predicted)
    gold_entities = _split_entities(gold)
    if not pred_entities or not gold_entities:
        return _token_f1(predicted, gold)

    candidates = []
    for pred_idx, pred in enumerate(pred_entities):
        for gold_idx, target in enumerate(gold_entities):
            candidates.append((_token_f1(pred, target), pred_idx, gold_idx))
    candidates.sort(reverse=True)
    used_pred = set()
    used_gold = set()
    similarity_sum = 0.0
    for score, pred_idx, gold_idx in candidates:
        if pred_idx in used_pred or gold_idx in used_gold:
            continue
        used_pred.add(pred_idx)
        used_gold.add(gold_idx)
        similarity_sum += score
    precision = similarity_sum / max(1, len(pred_entities))
    recall = similarity_sum / max(1, len(gold_entities))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def slot_similarity(predicted: Any, gold: Any, field: str) -> float:
    pred_norm = canonical_value(predicted)
    gold_norm = canonical_value(gold)
    if pred_norm == gold_norm:
        return 1.0
    if pred_norm == "-" or gold_norm == "-":
        return 0.0
    if field == "attraction":
        return _attraction_similarity(predicted, gold)
    return _token_f1(predicted, gold)


def _gold_slots(batch_item: Dict[str, Any]) -> Dict[Tuple[int, str], str]:
    slots: Dict[Tuple[int, str], str] = {}
    for fallback_day, row in enumerate(batch_item.get("gold_plan", []) or [], start=1):
        day = int(row.get("day", row.get("days", fallback_day)))
        for field in PLAN_FIELDS:
            value = row.get(field, "-")
            slots[(day, field)] = "-" if value is None else str(value).strip() or "-"
    return slots


def _reference_text(batch_item: Dict[str, Any], field: str) -> str:
    records = batch_item.get("reference_records", []) or []
    if records:
        description_prefixes = {
            "breakfast": ("restaurants",),
            "lunch": ("restaurants",),
            "dinner": ("restaurants",),
            "attraction": ("attractions",),
            "accommodation": ("accommodations",),
            "transportation": ("flight", "self-driving", "taxi"),
        }
        allowed_prefixes = description_prefixes.get(field)
        chunks = []
        for record in records:
            if isinstance(record, dict):
                description = str(record.get("Description", ""))
                if allowed_prefixes and not description.casefold().startswith(
                    allowed_prefixes
                ):
                    continue
                content = str(record.get("Content", ""))
                if canonical_value(content).startswith("no valid information"):
                    continue
                chunks.extend([description, content])
        return canonical_value("\n".join(chunks))
    return canonical_value(batch_item.get("reference_information", ""))


def _contains_grounded_entity(reference: str, entity: str) -> bool:
    normalized = canonical_value(entity)
    compact = re.sub(r"[^a-z0-9]", "", normalized)
    if len(compact) < 3:
        return False
    if normalized in {"the", "and", "from", "to", "in", "at", "a", "an"}:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])"
    return re.search(pattern, reference) is not None


def _strip_city_suffix(entity: str, cities: Iterable[str]) -> str:
    result = entity.strip()
    result = re.sub(r";?\s*cost\s*:?\s*\$?\d+.*$", "", result, flags=re.I)
    for city in cities:
        if not city:
            continue
        result = re.sub(rf",\s*{re.escape(city)}\s*$", "", result, flags=re.I)
    return result.strip(" ,;")


def _is_grounded(
    value: str,
    *,
    field: str,
    batch_item: Dict[str, Any],
    gold_value: str,
) -> bool:
    if canonical_value(value) == "-":
        # An explicit empty value is grounded only when the annotated plan also
        # marks that slot empty. Otherwise a team could maximize coverage and
        # grounding by assigning "-" everywhere.
        return canonical_value(gold_value) == "-"
    if slot_similarity(value, gold_value, field) >= 0.98:
        return True

    reference = _reference_text(batch_item, field)
    if not reference:
        return False
    if field == "transportation":
        flight_numbers = re.findall(r"\bF\d+\b", value, flags=re.I)
        if flight_numbers:
            return all(number.casefold() in reference for number in flight_numbers)

    cities = [str(batch_item.get("org", "")), str(batch_item.get("dest", ""))]
    if field == "current_city":
        mentioned = [
            city
            for city in cities
            if city
            and re.search(
                rf"(?<![a-z0-9]){re.escape(city.casefold())}(?![a-z0-9])",
                value.casefold(),
            )
        ]
        return bool(mentioned)

    entities = _split_entities(value) or [value]
    normalized_entities = [
        canonical_value(_strip_city_suffix(entity, cities)) for entity in entities
    ]
    return bool(normalized_entities) and all(
        _contains_grounded_entity(reference, entity) for entity in normalized_entities
    )


def _effective_agent_counts(
    result: MergeResult,
    gold: Dict[Tuple[int, str], str],
) -> List[float]:
    counts: List[float] = []
    for assignments in result.per_agent_assignments:
        score = 0.0
        for slot, value in assignments.items():
            if slot not in result.merged_assignments:
                continue
            score += slot_similarity(value, gold[slot], slot[1])
        counts.append(score)
    return counts


def _balance_score(
    result: MergeResult,
    days: int,
    gold: Dict[Tuple[int, str], str],
) -> float:
    counts = _effective_agent_counts(result, gold)
    if not counts:
        return 0.0
    total_slots = len(PLAN_FIELDS) * days
    if len(counts) == 2:
        experience_target = min(len(EXPERIENCE_FIELDS) * days, result.capacity)
        targets = [total_slots - experience_target, experience_target]
    else:
        base, extra = divmod(total_slots, len(counts))
        targets = [base + int(index < extra) for index in range(len(counts))]
    difference = sum(abs(actual - target) for actual, target in zip(counts, targets))
    return max(0.0, 1.0 - difference / max(1, total_slots))


def _role_score(
    result: MergeResult,
    days: int,
    gold: Dict[Tuple[int, str], str],
) -> float:
    role_fields = [LOGISTICS_FIELDS, EXPERIENCE_FIELDS]
    scores = []
    for agent_idx, assignments in enumerate(result.per_agent_assignments):
        if agent_idx >= len(role_fields):
            continue
        primary = role_fields[agent_idx]
        achievable = min(len(primary) * days, result.capacity)
        primary_credit = sum(
            slot_similarity(value, gold[slot], slot[1])
            for slot, value in assignments.items()
            if slot in result.merged_assignments and slot[1] in primary
        )
        scores.append(primary_credit / max(1, achievable))
    return sum(scores) / len(scores) if scores else 0.0


@dataclass
class TravelRewardConfig:
    parse_weight: float = 0.05
    coverage_weight: float = 0.15
    slot_quality_weight: float = 0.35
    grounding_weight: float = 0.15
    balance_weight: float = 0.10
    role_weight: float = 0.10
    exact_bonus: float = 0.20
    overlap_penalty: float = 0.10
    conflict_penalty: float = 0.20
    lazy_agent_penalty: float = 0.15
    invalid_slot_penalty: float = 0.05
    extra_assignment_penalty: float = 0.05
    self_duplicate_penalty: float = 0.10
    empty_mismatch_penalty: float = 0.25
    min_reward: float = -0.4
    max_reward: float = 1.2

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TravelRewardConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in (raw or {}).items() if key in allowed})


@dataclass
class TravelJointReward:
    config: TravelRewardConfig = field(default_factory=TravelRewardConfig)
    last_details: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def reward_range(self) -> Tuple[float, float]:
        return self.config.min_reward, self.config.max_reward

    def __call__(self, *agent_completions, batch_items=None, prompts=None) -> List[float]:
        del prompts
        if batch_items is None:
            raise ValueError("TravelJointReward requires batch_items.")
        rewards: List[float] = []
        details: List[Dict[str, Any]] = []
        for sample_idx, batch_item in enumerate(batch_items):
            completions = []
            for agent_output in agent_completions:
                if isinstance(agent_output, str):
                    completions.append(agent_output if sample_idx == 0 else "")
                elif sample_idx < len(agent_output):
                    completions.append(agent_output[sample_idx])
                else:
                    completions.append("")
            reward, detail = score_single_turn_response(
                completions,
                batch_item=batch_item,
                config=self.config,
            )
            rewards.append(reward)
            details.append(detail)
        self.last_details = details
        return rewards


def score_single_turn_response(
    agent_completions: Sequence[str],
    *,
    batch_item: Dict[str, Any],
    config: TravelRewardConfig | None = None,
) -> Tuple[float, Dict[str, Any]]:
    cfg = config or TravelRewardConfig()
    days = int(batch_item.get("days", 0))
    total_slots = len(PLAN_FIELDS) * days
    if total_slots <= 0:
        raise ValueError("batch_item.days must be positive.")

    gold = _gold_slots(batch_item)
    if len(gold) != total_slots:
        raise ValueError(
            f"Expected {total_slots} gold slots, found {len(gold)}."
        )
    result = merge_agent_assignments(agent_completions, days=days)

    parse_score = sum(float(parsed.parse_success) for parsed in result.parsed) / max(
        1, len(result.parsed)
    )
    raw_coverage_score = len(result.covered_slots) / total_slots

    similarities = []
    grounding_values = []
    exact_slots = 0
    valid_covered_slots = 0
    for slot, gold_value in gold.items():
        predicted = result.merged_assignments.get(slot)
        if predicted is None:
            similarities.append(0.0)
            continue
        similarity = slot_similarity(predicted, gold_value, slot[1])
        similarities.append(similarity)
        exact_slots += int(similarity >= 1.0 - 1e-9)
        grounded = _is_grounded(
            predicted,
            field=slot[1],
            batch_item=batch_item,
            gold_value=gold_value,
        )
        grounding_values.append(float(grounded))
        valid_covered_slots += int(grounded)
    slot_quality = sum(similarities) / total_slots
    coverage_score = valid_covered_slots / total_slots
    grounding_score = (
        sum(grounding_values) / len(grounding_values) if grounding_values else 0.0
    )
    balance_score = _balance_score(result, days, gold)
    role_score = _role_score(result, days, gold)

    lazy_agents = sum(1 for count in result.agent_assignment_counts if count == 0)
    lazy_rate = lazy_agents / max(1, len(agent_completions))
    overlap_rate = min(1.0, result.overlap_count / total_slots)
    conflict_rate = min(1.0, len(result.conflict_slots) / total_slots)
    invalid_slot_rate = min(1.0, result.invalid_slot_count / total_slots)
    extra_assignment_rate = min(1.0, result.extra_assignment_count / total_slots)
    self_duplicate_rate = min(1.0, result.self_duplicate_count / total_slots)
    empty_mismatch_count = sum(
        1
        for slot, predicted in result.merged_assignments.items()
        if canonical_value(predicted) == "-" and canonical_value(gold[slot]) != "-"
    )
    empty_mismatch_rate = empty_mismatch_count / total_slots
    exact_match = exact_slots == total_slots and not result.conflict_slots

    reward = (
        cfg.parse_weight * parse_score
        + cfg.coverage_weight * coverage_score
        + cfg.slot_quality_weight * slot_quality
        + cfg.grounding_weight * grounding_score
        + cfg.balance_weight * balance_score
        + cfg.role_weight * role_score
        + (cfg.exact_bonus if exact_match else 0.0)
        - cfg.overlap_penalty * overlap_rate
        - cfg.conflict_penalty * conflict_rate
        - cfg.lazy_agent_penalty * lazy_rate
        - cfg.invalid_slot_penalty * invalid_slot_rate
        - cfg.extra_assignment_penalty * extra_assignment_rate
        - cfg.self_duplicate_penalty * self_duplicate_rate
        - cfg.empty_mismatch_penalty * empty_mismatch_rate
    )
    reward = _clamp(reward, cfg.min_reward, cfg.max_reward)

    detail: Dict[str, Any] = {
        "reward": float(reward),
        "exact_match": float(exact_match),
        "parse_success": float(parse_score),
        "coverage": float(coverage_score),
        "raw_coverage": float(raw_coverage_score),
        "slot_quality": float(slot_quality),
        "grounding": float(grounding_score),
        "balance_score": float(balance_score),
        "role_score": float(role_score),
        "exact_slots": float(exact_slots),
        "total_slots": float(total_slots),
        "covered_slots": float(valid_covered_slots),
        "raw_covered_slots": float(len(result.covered_slots)),
        "overlap_count": float(result.overlap_count),
        "overlap_rate": float(overlap_rate),
        "conflict_count": float(len(result.conflict_slots)),
        "conflict_rate": float(conflict_rate),
        "lazy_agents": float(lazy_agents),
        "lazy_rate": float(lazy_rate),
        "invalid_slot_count": float(result.invalid_slot_count),
        "invalid_slot_rate": float(invalid_slot_rate),
        "extra_assignment_count": float(result.extra_assignment_count),
        "extra_assignment_rate": float(extra_assignment_rate),
        "self_duplicate_count": float(result.self_duplicate_count),
        "self_duplicate_rate": float(self_duplicate_rate),
        "empty_mismatch_count": float(empty_mismatch_count),
        "empty_mismatch_rate": float(empty_mismatch_rate),
        "capacity_per_agent": float(result.capacity),
        "raw_assignment_count": float(
            sum(parsed.raw_item_count for parsed in result.parsed)
        ),
        "agent_assignment_counts": result.agent_assignment_counts,
        "merged_plan": result.plan,
        "reward_backend": "annotated_plan_reference",
    }
    return float(reward), detail


def make_reward(config: Dict[str, Any] | None = None) -> TravelJointReward:
    return TravelJointReward(TravelRewardConfig.from_dict(config or {}))
