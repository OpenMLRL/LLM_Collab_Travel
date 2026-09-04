"""Reward-independent evaluation for simultaneous TravelPlanner actions.

This module owns the deterministic joint-action and final-itinerary metrics.
Both the training reward and the fixed evaluation logger consume this surface,
so changing reward weights cannot change the reported headline metrics.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Dict, List, Sequence, Tuple

from single_turn.aggregation import PLAN_FIELDS, merge_agent_assignments, owned_slots
from single_turn.rewards.reference_evaluator import evaluate_reference_plan


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
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def evaluate_single_turn_response(
    agent_completions: Sequence[str],
    *,
    batch_item: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate one aligned joint action without a target plan or reward config."""

    days = int(batch_item.get("days", 0))
    total_slots = len(PLAN_FIELDS) * days
    if total_slots <= 0:
        raise ValueError("batch_item.days must be positive.")
    if len(agent_completions) != 2:
        raise ValueError("The role-partitioned Travel task requires exactly 2 agents.")

    result = merge_agent_assignments(agent_completions, days=days)
    strict_format_score = sum(
        float(parsed.parse_success) for parsed in result.parsed
    ) / len(result.parsed)
    decode_score = sum(float(parsed.decode_success) for parsed in result.parsed) / len(
        result.parsed
    )
    strict_json_score = sum(
        float(parsed.strict_json) for parsed in result.parsed
    ) / len(result.parsed)
    schema_valid_score = sum(
        float(parsed.schema_valid) for parsed in result.parsed
    ) / len(result.parsed)
    agent_id_match_score = sum(
        float(parsed.agent_id_match) for parsed in result.parsed
    ) / len(result.parsed)
    capacity_valid_score = sum(
        float(parsed.capacity_valid) for parsed in result.parsed
    ) / len(result.parsed)

    agent_action_validity: List[float] = []
    agent_soft_action_validity: List[float] = []
    agent_owned_coverage: List[float] = []
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
        owned = owned_slots(agent_idx, days)
        expected_count = len(owned)
        owned_count = sum(slot in owned for slot in assignments)
        ownership_precision = owned_count / count if count else 0.0
        owned_coverage = owned_count / max(1, expected_count)
        count_fidelity = _clamp(
            1.0 - abs(count - expected_count) / max(1, expected_count),
            0.0,
            1.0,
        )
        item_acceptance = (
            count / parsed.raw_item_count if parsed.raw_item_count else 0.0
        )
        soft_validity = (
            float(parsed.decode_success)
            * (
                float(parsed.schema_valid)
                + float(parsed.agent_id_match)
                + float(parsed.capacity_valid)
                + ownership_precision
                + count_fidelity
                + item_acceptance
            )
            / 6.0
        )
        strict_action = bool(
            parsed.parse_success
            and set(assignments).issubset(owned)
            and count == expected_count
            and count == parsed.raw_item_count
        )
        agent_action_validity.append(float(strict_action))
        agent_soft_action_validity.append(float(soft_validity))
        agent_owned_coverage.append(float(owned_coverage))
        ownership_precision_scores.append(float(ownership_precision))
        count_fidelity_scores.append(float(count_fidelity))
        item_acceptance_scores.append(float(item_acceptance))

    team_action_valid = bool(
        all(agent_action_validity)
        and not result.conflict_slots
        and result.overlap_count == 0
        and result.invalid_slot_count == 0
        and result.self_duplicate_count == 0
        and result.extra_assignment_count == 0
    )
    team_soft_action_validity = _harmonic_mean(agent_soft_action_validity)

    plan_detail = evaluate_reference_plan(result, batch_item=batch_item)
    slot_validity = plan_detail.pop("slot_validity")
    required_slot_validity = plan_detail.pop("required_slot_validity")
    verified_contribution_ratios: List[float] = []
    required_grounded_contribution_ratios: List[float] = []
    for agent_idx, assignments in enumerate(result.per_agent_assignments):
        owned = owned_slots(agent_idx, days)
        # A conflicting assignment is deliberately removed by the merger. It
        # must not count as a valid contribution merely because the slot's
        # fallback "-" value happens to be acceptable to the plan checker.
        verified = sum(
            float(slot_validity.get(slot, 0.0))
            for slot in assignments
            if slot in owned and slot in result.merged_assignments
        )
        verified_contribution_ratios.append(verified / max(1, len(owned)))
        required_owned = owned & set(required_slot_validity)
        required_verified = sum(
            float(required_slot_validity.get(slot, 0.0))
            for slot in required_owned
            if slot in assignments and slot in result.merged_assignments
        )
        required_grounded_contribution_ratios.append(
            required_verified / len(required_owned) if required_owned else 0.0
        )
    cooperative_contribution = _harmonic_mean(verified_contribution_ratios)
    contribution_mean = sum(required_grounded_contribution_ratios) / len(
        required_grounded_contribution_ratios
    )
    contribution_min = min(required_grounded_contribution_ratios)
    # Keep some progress from the stronger role visible, but make the weaker
    # role dominate. A fully lazy teammate therefore caps this signal at 0.10.
    required_cooperative_contribution = (
        0.2 * contribution_mean + 0.8 * contribution_min
    )
    contribution_deficit = 1.0 - cooperative_contribution

    raw_assignment_count = sum(parsed.raw_item_count for parsed in result.parsed)
    overlap_rate = min(1.0, result.overlap_count / max(1, total_slots))
    conflict_rate = min(1.0, len(result.conflict_slots) / max(1, total_slots))
    invalid_action_count = (
        result.invalid_slot_count
        + result.self_duplicate_count
        + result.extra_assignment_count
    )
    invalid_action_rate = min(1.0, invalid_action_count / max(1, raw_assignment_count))

    both_agents_verified = float(
        bool(verified_contribution_ratios)
        and all(value >= 1.0 - 1e-9 for value in verified_contribution_ratios)
    )
    both_agents_required_grounded = float(
        bool(required_grounded_contribution_ratios)
        and all(
            value >= 1.0 - 1e-9
            for value in required_grounded_contribution_ratios
        )
    )
    conflict_free = float(result.overlap_count == 0 and not result.conflict_slots)
    collaboration_success = float(
        bool(plan_detail["ultimate/reference_plan_success"])
        and team_action_valid
        and bool(both_agents_verified)
    )

    detail: Dict[str, Any] = {
        "parse_success": float(strict_format_score),
        "decode_success": float(decode_score),
        "strict_json": float(strict_json_score),
        "schema_valid": float(schema_valid_score),
        "agent_id_match": float(agent_id_match_score),
        "capacity_valid": float(capacity_valid_score),
        "action_validity": sum(agent_action_validity) / len(agent_action_validity),
        "team_action_valid": float(team_action_valid),
        "team_soft_action_validity": float(team_soft_action_validity),
        "cooperative_contribution": float(cooperative_contribution),
        "required_cooperative_contribution": float(
            required_cooperative_contribution
        ),
        "contribution_deficit": float(contribution_deficit),
        "overlap_count": float(result.overlap_count),
        "overlap_rate": float(overlap_rate),
        "conflict_count": float(len(result.conflict_slots)),
        "conflict_rate": float(conflict_rate),
        "invalid_slot_count": float(result.invalid_slot_count),
        "self_duplicate_count": float(result.self_duplicate_count),
        "extra_assignment_count": float(result.extra_assignment_count),
        "invalid_action_rate": float(invalid_action_rate),
        "raw_assignment_count": float(raw_assignment_count),
        "raw_covered_slots": float(len(result.covered_slots)),
        "total_slots": float(total_slots),
        "agent_action_validity": agent_action_validity,
        "agent_soft_action_validity": agent_soft_action_validity,
        "agent_owned_coverage": agent_owned_coverage,
        "ownership_precision": ownership_precision_scores,
        "count_fidelity": count_fidelity_scores,
        "item_acceptance": item_acceptance_scores,
        "verified_contribution_ratios": verified_contribution_ratios,
        "required_grounded_contribution_ratios": (
            required_grounded_contribution_ratios
        ),
        "ultimate/team_action_success": float(team_action_valid),
        "ultimate/both_agent_verified_contribution": both_agents_verified,
        "ultimate/both_agent_required_grounded_contribution": (
            both_agents_required_grounded
        ),
        "ultimate/conflict_free": conflict_free,
        "ultimate/collaboration_success": collaboration_success,
        "merged_plan": result.plan,
        "parser_errors": [list(parsed.error_codes) for parsed in result.parsed],
        "evaluation_backend": "reference_constraint_scaffold_v2",
        **plan_detail,
    }

    per_agent_sequences = {
        "action_validity": agent_action_validity,
        "soft_action_validity": agent_soft_action_validity,
        "owned_slot_coverage": agent_owned_coverage,
        "ownership_precision": ownership_precision_scores,
        "count_fidelity": count_fidelity_scores,
        "item_acceptance": item_acceptance_scores,
        "verified_contribution": verified_contribution_ratios,
        "required_grounded_contribution": (
            required_grounded_contribution_ratios
        ),
        "assignment_count": result.agent_assignment_counts,
    }
    for metric_name, values in per_agent_sequences.items():
        for agent_idx, value in enumerate(values):
            detail[f"agent_{agent_idx}/{metric_name}"] = float(value)

    parser_error_sets = [set(parsed.error_codes) for parsed in result.parsed]
    detail["parser_error/any"] = sum(
        float(bool(error_codes)) for error_codes in parser_error_sets
    ) / len(parser_error_sets)
    for error_code in PARSER_ERROR_CODES:
        detail[f"parser_error/{error_code}"] = sum(
            float(error_code in error_codes) for error_codes in parser_error_sets
        ) / len(parser_error_sets)
    known_error_codes = set(PARSER_ERROR_CODES)
    for agent_idx, (parsed, error_codes) in enumerate(
        zip(result.parsed, parser_error_sets)
    ):
        prefix = f"agent_{agent_idx}"
        detail[f"{prefix}/decode_success"] = float(parsed.decode_success)
        detail[f"{prefix}/strict_json"] = float(parsed.strict_json)
        detail[f"{prefix}/schema_valid"] = float(parsed.schema_valid)
        detail[f"{prefix}/agent_id_match"] = float(parsed.agent_id_match)
        detail[f"{prefix}/capacity_valid"] = float(parsed.capacity_valid)
        detail[f"{prefix}/raw_assignment_count"] = float(parsed.raw_item_count)
        detail[f"{prefix}/parser_error/any"] = float(bool(error_codes))
        detail[f"{prefix}/parser_error/other"] = float(
            bool(error_codes - known_error_codes)
        )
        for error_code in PARSER_ERROR_CODES:
            detail[f"{prefix}/parser_error/{error_code}"] = float(
                error_code in error_codes
            )

    # Fail loudly if a future refactor reintroduces target-plan diagnostics or
    # emits a non-scalar headline metric.
    forbidden = {"gold_plan", "annotated_plan", "exact_match", "slot_quality"}
    if forbidden & set(detail):
        raise AssertionError("Reference-backed evaluation exposed a target plan.")
    if not all(
        isinstance(value, Real)
        for key, value in detail.items()
        if key.startswith("ultimate/")
    ):
        raise AssertionError("Ultimate metrics must be scalar values.")
    return detail
