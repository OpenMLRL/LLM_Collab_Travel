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
    PLAN_FIELDS,
    MergeResult,
    canonical_value,
    merge_agent_assignments,
    owned_slots,
)


# Keep this list stable so every evaluation step logs both positive and zero
# rates for each parser failure.  Logging only errors that happened in a batch
# would make W&B average each sparse series over its positive examples and show
# a misleading constant value of one.
PARSER_ERROR_CODES: Tuple[str, ...] = (
    "decode_failed",
    "not_strict_json",
    "top_level_schema",
    "top_level_type",
    "assignments_type",
    "invalid_agent_id",
    "agent_id_mismatch",
    "assignment_schema",
    "assignment_value",
    "self_duplicate",
    "capacity_exceeded",
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _harmonic_mean(values: Sequence[float]) -> float:
    """Return a balance-sensitive continuous team score in ``[0, 1]``."""

    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


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
    days: int,
    gold: Dict[Tuple[int, str], str],
    batch_item: Dict[str, Any],
) -> List[float]:
    counts: List[float] = []
    for agent_idx, assignments in enumerate(result.per_agent_assignments):
        primary = owned_slots(agent_idx, days) if agent_idx < 2 else set()
        score = 0.0
        for slot, value in assignments.items():
            if slot not in result.merged_assignments:
                continue
            if primary and slot not in primary:
                continue
            if canonical_value(gold[slot]) == "-":
                continue
            similarity = slot_similarity(value, gold[slot], slot[1])
            grounded = _is_grounded(
                value,
                field=slot[1],
                batch_item=batch_item,
                gold_value=gold[slot],
            )
            score += 0.5 * similarity + 0.5 * float(grounded)
        counts.append(score)
    return counts


def _semantic_role_targets(
    days: int,
    gold: Dict[Tuple[int, str], str],
) -> List[int]:
    return [
        sum(
            canonical_value(gold[slot]) != "-"
            for slot in owned_slots(agent_idx, days)
        )
        for agent_idx in range(2)
    ]


def _cooperative_contribution_score(
    result: MergeResult,
    days: int,
    gold: Dict[Tuple[int, str], str],
    batch_item: Dict[str, Any],
) -> Tuple[float, List[float]]:
    counts = _effective_agent_counts(result, days, gold, batch_item)
    targets = _semantic_role_targets(days, gold)
    ratios = [
        min(1.0, count / max(1, target))
        for count, target in zip(counts[:2], targets)
    ]
    if len(ratios) < 2 or not all(ratio > 0 for ratio in ratios):
        return 0.0, ratios
    return 2.0 * ratios[0] * ratios[1] / (ratios[0] + ratios[1]), ratios


def _balance_score(
    result: MergeResult,
    days: int,
    gold: Dict[Tuple[int, str], str],
    batch_item: Dict[str, Any],
) -> float:
    cooperative_score, _ = _cooperative_contribution_score(
        result, days, gold, batch_item
    )
    return cooperative_score


def _role_score(
    result: MergeResult,
    days: int,
    gold: Dict[Tuple[int, str], str],
) -> float:
    scores = []
    for agent_idx, assignments in enumerate(result.per_agent_assignments):
        if agent_idx >= 2:
            continue
        primary = owned_slots(agent_idx, days)
        achievable = sum(canonical_value(gold[slot]) != "-" for slot in primary)
        if achievable == 0:
            continue
        primary_credit = sum(
            slot_similarity(value, gold[slot], slot[1])
            for slot, value in assignments.items()
            if slot in result.merged_assignments
            and slot in primary
            and canonical_value(gold[slot]) != "-"
        )
        scores.append(primary_credit / max(1, achievable))
    if len(scores) < 2 or not all(score > 0 for score in scores):
        return 0.0
    return 2.0 * scores[0] * scores[1] / (scores[0] + scores[1])


@dataclass
class TravelRewardConfig:
    parse_weight: float = 0.05
    coverage_weight: float = 0.20
    slot_quality_weight: float = 0.45
    grounding_weight: float = 0.0
    empty_match_weight: float = 0.05
    balance_weight: float = 0.15
    role_weight: float = 0.10
    exact_bonus: float = 0.20
    cooperation_floor: float = 0.20
    validity_gate_weight: float = 0.50
    overlap_penalty: float = 0.15
    conflict_penalty: float = 0.30
    lazy_agent_penalty: float = 0.25
    invalid_slot_penalty: float = 0.15
    extra_assignment_penalty: float = 0.25
    self_duplicate_penalty: float = 0.15
    empty_mismatch_penalty: float = 0.25
    spurious_fill_penalty: float = 0.15
    min_reward: float = -0.5
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

    strict_format_score = sum(
        float(parsed.parse_success) for parsed in result.parsed
    ) / max(
        1, len(result.parsed)
    )
    decode_score = sum(float(parsed.decode_success) for parsed in result.parsed) / max(
        1, len(result.parsed)
    )
    strict_json_score = sum(float(parsed.strict_json) for parsed in result.parsed) / max(
        1, len(result.parsed)
    )
    schema_valid_score = sum(
        float(parsed.schema_valid) for parsed in result.parsed
    ) / max(1, len(result.parsed))
    agent_id_match_score = sum(
        float(parsed.agent_id_match) for parsed in result.parsed
    ) / max(1, len(result.parsed))
    capacity_valid_score = sum(
        float(parsed.capacity_valid) for parsed in result.parsed
    ) / max(1, len(result.parsed))
    agent_action_validity: List[float] = []
    agent_recoverable_action_validity: List[float] = []
    agent_soft_action_validity: List[float] = []
    ownership_validity: List[float] = []
    ownership_precision_scores: List[float] = []
    count_fidelity_scores: List[float] = []
    item_acceptance_scores: List[float] = []
    for agent_idx, (parsed, count, assignments) in enumerate(
        zip(
            result.parsed,
            result.agent_assignment_counts,
            result.per_agent_assignments,
        )
    ):
        owned = owned_slots(agent_idx, days) if agent_idx < 2 else set()
        ownership_valid = bool(owned) and set(assignments).issubset(owned)
        expected_count = len(owned)
        owned_assignment_count = sum(slot in owned for slot in assignments)
        ownership_precision = (
            owned_assignment_count / count if count > 0 else 0.0
        )
        count_fidelity = _clamp(
            1.0 - abs(count - expected_count) / max(1, expected_count),
            0.0,
            1.0,
        )
        item_acceptance = (
            count / parsed.raw_item_count if parsed.raw_item_count > 0 else 0.0
        )
        recoverable_action_valid = bool(
            parsed.decode_success
            and parsed.schema_valid
            and parsed.agent_id_match
            and parsed.capacity_valid
            and ownership_valid
            and count == expected_count
            and count == parsed.raw_item_count
        )

        # Strict JSON is deliberately excluded from this dense score.  A fully
        # recoverable action with a harmless prefix should retain its plan
        # learning signal, while ``parse_weight`` and the exact-match bonus
        # still make the strictly formatted action preferable.
        soft_action_validity = float(parsed.decode_success) * (
            float(parsed.schema_valid)
            + float(parsed.agent_id_match)
            + float(parsed.capacity_valid)
            + ownership_precision
            + count_fidelity
            + item_acceptance
        ) / 6.0

        ownership_validity.append(float(ownership_valid))
        ownership_precision_scores.append(float(ownership_precision))
        count_fidelity_scores.append(float(count_fidelity))
        item_acceptance_scores.append(float(item_acceptance))
        agent_recoverable_action_validity.append(
            float(recoverable_action_valid)
        )
        agent_soft_action_validity.append(float(soft_action_validity))
        agent_action_validity.append(
            float(
                parsed.parse_success
                and ownership_valid
                and count == expected_count
                and count == parsed.raw_item_count
            )
        )
    action_validity_score = sum(agent_action_validity) / max(
        1, len(agent_action_validity)
    )
    recoverable_action_validity_score = sum(
        agent_recoverable_action_validity
    ) / max(1, len(agent_recoverable_action_validity))
    soft_action_validity_score = sum(agent_soft_action_validity) / max(
        1, len(agent_soft_action_validity)
    )
    ownership_validity_score = sum(ownership_validity) / max(
        1, len(ownership_validity)
    )
    team_soft_action_validity = _harmonic_mean(agent_soft_action_validity)
    raw_coverage_score = len(result.covered_slots) / total_slots

    all_similarities = []
    semantic_similarities = []
    grounding_values = []
    exact_slots = 0
    valid_covered_slots = 0
    raw_semantic_covered_slots = 0
    empty_gold_slots = 0
    correct_empty_slots = 0
    for slot, gold_value in gold.items():
        predicted = result.merged_assignments.get(slot)
        if predicted is None:
            all_similarities.append(0.0)
            if canonical_value(gold_value) == "-":
                empty_gold_slots += 1
            else:
                semantic_similarities.append(0.0)
            continue
        similarity = slot_similarity(predicted, gold_value, slot[1])
        all_similarities.append(similarity)
        exact_slots += int(similarity >= 1.0 - 1e-9)
        if canonical_value(gold_value) == "-":
            empty_gold_slots += 1
            correct_empty_slots += int(canonical_value(predicted) == "-")
            continue
        semantic_similarities.append(similarity)
        raw_semantic_covered_slots += 1
        grounded = _is_grounded(
            predicted,
            field=slot[1],
            batch_item=batch_item,
            gold_value=gold_value,
        )
        grounding_values.append(float(grounded))
        valid_covered_slots += int(grounded)
    semantic_slot_count = len(semantic_similarities)
    slot_quality = sum(semantic_similarities) / max(1, semantic_slot_count)
    all_slot_quality = sum(all_similarities) / total_slots
    coverage_score = valid_covered_slots / max(1, semantic_slot_count)
    raw_semantic_coverage = raw_semantic_covered_slots / max(1, semantic_slot_count)
    grounding_precision = (
        sum(grounding_values) / len(grounding_values) if grounding_values else 0.0
    )
    empty_recall = (
        correct_empty_slots / empty_gold_slots if empty_gold_slots else 1.0
    )
    empty_match_score = empty_recall * coverage_score
    balance_score = _balance_score(result, days, gold, batch_item)
    role_score = _role_score(result, days, gold)
    cooperative_contribution, contribution_ratios = _cooperative_contribution_score(
        result, days, gold, batch_item
    )

    owned_nonconflict_counts = []
    owned_target_counts = []
    for agent_idx, assignments in enumerate(result.per_agent_assignments[:2]):
        owned = owned_slots(agent_idx, days)
        owned_target_counts.append(len(owned))
        owned_nonconflict_counts.append(
            sum(
                slot in owned and slot in result.merged_assignments
                for slot in assignments
            )
        )
    action_contribution_ratios = [
        min(1.0, count / max(1, target))
        for count, target in zip(owned_nonconflict_counts, owned_target_counts)
    ]
    minimum_action_contribution = min(action_contribution_ratios, default=0.0)
    lazy_deficit = 1.0 - minimum_action_contribution
    lazy_agents = sum(1 for ratio in action_contribution_ratios if ratio <= 0.0)
    lazy_rate = lazy_agents / max(1, len(agent_completions))
    proposed_slots = set().union(
        *(set(assignments) for assignments in result.per_agent_assignments)
    )
    proposed_slot_count = len(proposed_slots)
    raw_assignment_count = sum(parsed.raw_item_count for parsed in result.parsed)
    per_agent_distinct_counts = result.agent_assignment_counts
    collision_denominator = max(1, min(per_agent_distinct_counts, default=0))
    overlap_rate = min(1.0, result.overlap_count / collision_denominator)
    conflict_rate = min(
        1.0, len(result.conflict_slots) / collision_denominator
    )
    invalid_slot_rate = min(
        1.0, result.invalid_slot_count / max(1, raw_assignment_count)
    )
    extra_assignment_rate = min(
        1.0, result.extra_assignment_count / max(1, raw_assignment_count)
    )
    overcapacity_agent_rate = sum(
        float(not parsed.capacity_valid) for parsed in result.parsed
    ) / max(1, len(result.parsed))
    self_duplicate_rate = min(
        1.0, result.self_duplicate_count / max(1, raw_assignment_count)
    )
    empty_mismatch_count = sum(
        1
        for slot, predicted in result.merged_assignments.items()
        if canonical_value(predicted) == "-" and canonical_value(gold[slot]) != "-"
    )
    empty_mismatch_rate = empty_mismatch_count / max(1, semantic_slot_count)
    spurious_fill_count = sum(
        1
        for slot, predicted in result.merged_assignments.items()
        if canonical_value(predicted) != "-" and canonical_value(gold[slot]) == "-"
    )
    spurious_fill_rate = spurious_fill_count / max(1, empty_gold_slots)
    exact_match = exact_slots == total_slots and not result.conflict_slots
    team_action_valid = bool(agent_action_validity) and all(agent_action_validity)
    team_recoverable_action_valid = bool(agent_recoverable_action_validity) and all(
        agent_recoverable_action_validity
    )
    gate_weight = _clamp(cfg.validity_gate_weight, 0.0, 1.0)
    gate_floor = _clamp(cfg.cooperation_floor, 0.0, 1.0)
    validity_gate = gate_floor + (
        1.0 - gate_floor
    ) * team_soft_action_validity
    contribution_gate = gate_floor + (
        1.0 - gate_floor
    ) * cooperative_contribution
    if gate_weight <= 0.0:
        joint_gate_signal = cooperative_contribution
    elif gate_weight >= 1.0:
        joint_gate_signal = team_soft_action_validity
    elif team_soft_action_validity <= 0.0 or cooperative_contribution <= 0.0:
        joint_gate_signal = 0.0
    else:
        joint_gate_signal = (
            team_soft_action_validity ** gate_weight
            * cooperative_contribution ** (1.0 - gate_weight)
        )
    # Use one floor after combining the two continuous signals.  The previous
    # product of two separately floored gates could shrink the plan reward to
    # 0.04 solely because one strict-format bit was false.
    cooperation_gate = gate_floor + (
        1.0 - gate_floor
    ) * joint_gate_signal
    contribution_deficit = 1.0 - cooperative_contribution

    plan_score = (
        cfg.coverage_weight * coverage_score
        + cfg.slot_quality_weight * slot_quality
        # Grounded coverage, rather than conditional grounding precision, prevents
        # a team from receiving a large reward for one grounded slot.
        + cfg.grounding_weight * coverage_score
        + cfg.empty_match_weight * empty_match_score
        + cfg.balance_weight * balance_score
        + cfg.role_weight * role_score
    )

    reward = (
        cfg.parse_weight * strict_format_score
        + cooperation_gate * plan_score
        + (cfg.exact_bonus if exact_match and team_action_valid else 0.0)
        - cfg.overlap_penalty * overlap_rate
        - cfg.conflict_penalty * conflict_rate
        - cfg.lazy_agent_penalty * lazy_deficit
        - cfg.invalid_slot_penalty * invalid_slot_rate
        - cfg.extra_assignment_penalty * overcapacity_agent_rate
        - cfg.self_duplicate_penalty * self_duplicate_rate
        - cfg.empty_mismatch_penalty * empty_mismatch_rate
        - cfg.spurious_fill_penalty * spurious_fill_rate
    )
    reward = _clamp(reward, cfg.min_reward, cfg.max_reward)

    detail: Dict[str, Any] = {
        "reward": float(reward),
        "exact_match": float(exact_match),
        "rewarded_exact_match": float(exact_match and team_action_valid),
        "parse_success": float(strict_format_score),
        "decode_success": float(decode_score),
        "strict_json": float(strict_json_score),
        "schema_valid": float(schema_valid_score),
        "agent_id_match": float(agent_id_match_score),
        "capacity_valid": float(capacity_valid_score),
        "ownership_validity_mean": float(ownership_validity_score),
        "action_validity": float(action_validity_score),
        "recoverable_action_validity": float(recoverable_action_validity_score),
        "soft_action_validity": float(soft_action_validity_score),
        "agent_action_validity": agent_action_validity,
        "agent_recoverable_action_validity": agent_recoverable_action_validity,
        "agent_soft_action_validity": agent_soft_action_validity,
        "ownership_validity": ownership_validity,
        "ownership_precision": ownership_precision_scores,
        "count_fidelity": count_fidelity_scores,
        "item_acceptance": item_acceptance_scores,
        "team_action_valid": float(team_action_valid),
        "team_recoverable_action_valid": float(team_recoverable_action_valid),
        "team_soft_action_validity": float(team_soft_action_validity),
        "validity_gate": float(validity_gate),
        "contribution_gate": float(contribution_gate),
        "joint_gate_signal": float(joint_gate_signal),
        "cooperation_gate": float(cooperation_gate),
        "cooperative_contribution": float(cooperative_contribution),
        "contribution_ratios": contribution_ratios,
        "contribution_deficit": float(contribution_deficit),
        "action_contribution_ratios": action_contribution_ratios,
        "lazy_deficit": float(lazy_deficit),
        "plan_score": float(plan_score),
        "coverage": float(coverage_score),
        "raw_coverage": float(raw_coverage_score),
        "raw_semantic_coverage": float(raw_semantic_coverage),
        "slot_quality": float(slot_quality),
        "all_slot_quality": float(all_slot_quality),
        "grounding": float(grounding_precision),
        "empty_recall": float(empty_recall),
        "empty_match_score": float(empty_match_score),
        "semantic_slots": float(semantic_slot_count),
        "empty_gold_slots": float(empty_gold_slots),
        "correct_empty_slots": float(correct_empty_slots),
        "balance_score": float(balance_score),
        "role_score": float(role_score),
        "exact_slots": float(exact_slots),
        "total_slots": float(total_slots),
        "covered_slots": float(valid_covered_slots),
        "raw_semantic_covered_slots": float(raw_semantic_covered_slots),
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
        "overcapacity_agent_rate": float(overcapacity_agent_rate),
        "self_duplicate_count": float(result.self_duplicate_count),
        "self_duplicate_rate": float(self_duplicate_rate),
        "empty_mismatch_count": float(empty_mismatch_count),
        "empty_mismatch_rate": float(empty_mismatch_rate),
        "spurious_fill_count": float(spurious_fill_count),
        "spurious_fill_rate": float(spurious_fill_rate),
        "capacity_per_agent": float(result.capacity),
        "proposed_slot_count": float(proposed_slot_count),
        "raw_assignment_count": float(raw_assignment_count),
        "agent_assignment_counts": result.agent_assignment_counts,
        "parser_errors": [list(parsed.error_codes) for parsed in result.parsed],
        "merged_plan": result.plan,
        "reward_backend": "annotated_plan_reference",
    }

    per_agent_sequences = {
        "action_validity": agent_action_validity,
        "recoverable_action_validity": agent_recoverable_action_validity,
        "soft_action_validity": agent_soft_action_validity,
        "ownership_validity": ownership_validity,
        "ownership_precision": ownership_precision_scores,
        "count_fidelity": count_fidelity_scores,
        "item_acceptance": item_acceptance_scores,
        "contribution_ratio": contribution_ratios,
        "action_contribution_ratio": action_contribution_ratios,
        "assignment_count": result.agent_assignment_counts,
    }
    for metric_name, values in per_agent_sequences.items():
        for agent_idx, value in enumerate(values):
            detail[f"agent_{agent_idx}/{metric_name}"] = float(value)

    parser_error_sets = [set(parsed.error_codes) for parsed in result.parsed]
    detail["parser_error/any"] = sum(
        float(bool(error_codes)) for error_codes in parser_error_sets
    ) / max(1, len(parser_error_sets))
    for error_code in PARSER_ERROR_CODES:
        detail[f"parser_error/{error_code}"] = sum(
            float(error_code in error_codes) for error_codes in parser_error_sets
        ) / max(1, len(parser_error_sets))
    known_error_codes = set(PARSER_ERROR_CODES)
    for agent_idx, (parsed, error_codes) in enumerate(
        zip(result.parsed, parser_error_sets)
    ):
        agent_prefix = f"agent_{agent_idx}"
        detail[f"{agent_prefix}/decode_success"] = float(parsed.decode_success)
        detail[f"{agent_prefix}/strict_json"] = float(parsed.strict_json)
        detail[f"{agent_prefix}/schema_valid"] = float(parsed.schema_valid)
        detail[f"{agent_prefix}/agent_id_match"] = float(parsed.agent_id_match)
        detail[f"{agent_prefix}/capacity_valid"] = float(parsed.capacity_valid)
        detail[f"{agent_prefix}/raw_assignment_count"] = float(
            parsed.raw_item_count
        )
        detail[f"{agent_prefix}/parser_error/any"] = float(bool(error_codes))
        detail[f"{agent_prefix}/parser_error/other"] = float(
            bool(error_codes - known_error_codes)
        )
        for error_code in PARSER_ERROR_CODES:
            detail[f"{agent_prefix}/parser_error/{error_code}"] = float(
                error_code in error_codes
            )
    return float(reward), detail


def make_reward(config: Dict[str, Any] | None = None) -> TravelJointReward:
    return TravelJointReward(TravelRewardConfig.from_dict(config or {}))
