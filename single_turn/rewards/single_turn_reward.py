"""Shared, reference-backed reward for one-shot TravelPlanner collaboration.

MAGRPO receives a scalar team reward, but never a target itinerary. Any joint
plan can score well when it follows the decentralized action contract, uses
facts from the supplied reference information, and satisfies the itinerary
constraints. Human ``annotated_plan`` / ``gold_plan`` fields are neither read
nor accepted by this scorer.

Headline metrics live in :mod:`single_turn.evaluation`; this module only maps
that reward-independent evaluation surface to a dense training scalar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from single_turn.evaluation import evaluate_single_turn_response
from single_turn.aggregation import owned_slots
from single_turn.rewards.reference_evaluator import evaluate_reference_plan
from single_turn.rewards.reward_shaping import build_reward_shaping_surface


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _bottleneck(values: List[float]) -> float:
    """Keep both roles visible while making the weaker role dominate."""

    if not values:
        return 0.0
    return 0.2 * (sum(values) / len(values)) + 0.8 * min(values)


def _geometric_quality(balance: float, quality: float) -> float:
    if balance <= 0.0 or quality <= 0.0:
        return 0.0
    return balance**0.65 * quality**0.35


@dataclass
class TravelRewardConfig:
    """Weights for the fixed, learnable strict-first team reward.

    Strict joint quality owns 91% of the positive range. Two small, bounded
    protocol/recovered-semantic channels keep malformed early rollouts
    rankable without allowing them to compete with valid collaboration.
    """

    protocol_progress_weight: float = 0.02
    recovered_semantic_weight: float = 0.03
    action_validity_weight: float = 0.04
    team_action_weight: float = 0.02
    strict_balance_weight: float = 0.14
    strict_quality_weight: float = 0.57
    strict_grounding_weight: float = 0.08
    final_success_bonus: float = 0.10

    assignment_coverage_weight: float = 0.20
    required_grounded_weight: float = 0.25
    grounding_f1_weight: float = 0.15
    commonsense_weight: float = 0.25
    hard_constraint_weight: float = 0.15
    grounding_support_floor: float = 0.10

    invalid_action_penalty: float = 0.10
    conflict_penalty: float = 0.05
    overlap_penalty: float = 0.05
    reference_copy_penalty: float = 0.10
    overlong_penalty: float = 0.05
    min_reward: float = -0.25
    max_reward: float = 1.00

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TravelRewardConfig":
        allowed = set(cls.__dataclass_fields__)
        values = {
            key: float(value) for key, value in (raw or {}).items() if key in allowed
        }
        return cls(**values)

    def validate(self) -> None:
        plan_weight = (
            self.assignment_coverage_weight
            + self.required_grounded_weight
            + self.grounding_f1_weight
            + self.commonsense_weight
            + self.hard_constraint_weight
        )
        if not math.isclose(plan_weight, 1.0, abs_tol=1e-9):
            raise ValueError(
                "Reference-backed plan weights must sum to 1.0; "
                f"found {plan_weight:.6f}."
            )
        positive_weights = (
            self.protocol_progress_weight,
            self.recovered_semantic_weight,
            self.action_validity_weight,
            self.team_action_weight,
            self.strict_balance_weight,
            self.strict_quality_weight,
            self.strict_grounding_weight,
            self.final_success_bonus,
        )
        if not math.isclose(sum(positive_weights), 1.0, abs_tol=1e-9):
            raise ValueError(
                "Strict positive reward weights must sum to 1.0; "
                f"found {sum(positive_weights):.6f}."
            )
        scalar_weights = (
            *positive_weights,
            self.overlap_penalty,
            self.conflict_penalty,
            self.invalid_action_penalty,
            self.reference_copy_penalty,
            self.overlong_penalty,
        )
        if any(weight < 0.0 for weight in scalar_weights):
            raise ValueError("Reward weights and penalties must be non-negative.")
        if not 0.0 <= self.grounding_support_floor <= 1.0:
            raise ValueError("grounding_support_floor must lie in [0, 1].")
        if self.min_reward >= self.max_reward:
            raise ValueError("min_reward must be smaller than max_reward.")


def _plan_quality(
    metrics: Dict[str, Any], config: TravelRewardConfig
) -> Tuple[float, float]:
    grounding = float(metrics["required_grounded_recall"])
    support = config.grounding_support_floor + (
        1.0 - config.grounding_support_floor
    ) * grounding
    quality = math.fsum(
        (
            support
            * config.assignment_coverage_weight
            * float(metrics["assignment_coverage"]),
            config.required_grounded_weight * grounding,
            config.grounding_f1_weight * float(metrics["grounding_f1"]),
            support
            * config.commonsense_weight
            * float(metrics["commonsense_soft"]),
            support
            * config.hard_constraint_weight
            * float(metrics["hard_constraint_soft"]),
        )
    )
    return float(quality), float(support)


@dataclass
class TravelJointReward:
    config: TravelRewardConfig = field(default_factory=TravelRewardConfig)
    last_details: List[Dict[str, Any]] = field(default_factory=list)
    _pending_details: List[Dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.config.validate()

    @property
    def reward_range(self) -> Tuple[float, float]:
        return self.config.min_reward, self.config.max_reward

    def __call__(
        self, *agent_completions, batch_items=None, prompts=None
    ) -> List[float]:
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
        self._pending_details.extend(details)
        return rewards

    def drain_details(self) -> List[Dict[str, Any]]:
        """Return reward diagnostics accumulated since the previous drain."""

        details = self._pending_details
        self._pending_details = []
        return details


def score_single_turn_response(
    agent_completions: Sequence[str],
    *,
    batch_item: Dict[str, Any],
    config: TravelRewardConfig | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """Map reward-independent joint evaluation to one dense team scalar."""

    cfg = config or TravelRewardConfig()
    cfg.validate()
    detail = evaluate_single_turn_response(
        agent_completions,
        batch_item=batch_item,
    )

    shaping = build_reward_shaping_surface(
        agent_completions,
        batch_item=batch_item,
    )
    recovered_plan = evaluate_reference_plan(
        shaping.recovered_merge,
        batch_item=batch_item,
    )
    recovered_plan.pop("slot_validity")
    recovered_required_slot_validity = recovered_plan.pop(
        "required_slot_validity"
    )
    recovered_required_ratios = []
    structural_progress_scores = []
    recovery_violation_scores = []
    for agent_idx, assignments in enumerate(
        shaping.recovered_merge.per_agent_assignments
    ):
        owned = owned_slots(agent_idx, int(batch_item["days"]))
        required_owned = owned & set(recovered_required_slot_validity)
        verified = sum(
            recovered_required_slot_validity.get(slot, 0.0)
            for slot in required_owned
            if slot in assignments
            and slot in shaping.recovered_merge.merged_assignments
        )
        recovered_required_ratios.append(
            verified / len(required_owned) if required_owned else 0.0
        )
        prefix = f"agent_{agent_idx}"
        structural_progress = sum(
            shaping.details[f"{prefix}/{key}"]
            for key in (
                "object_start",
                "agent_id_match",
                "assignments_list",
                "closed_delimiters",
            )
        ) / 4.0
        violation = max(
            shaping.details[f"{prefix}/recovery_overflow_rate"],
            shaping.details[f"{prefix}/recovery_rejected_rate"],
        )
        structural_progress_scores.append(structural_progress)
        recovery_violation_scores.append(violation)

    protocol_role_scores = [
        0.5 * shaping.details[f"agent_{agent_idx}/format_progress"]
        + 0.5 * detail[f"agent_{agent_idx}/soft_action_validity"]
        for agent_idx in range(len(agent_completions))
    ]
    protocol_progress = _bottleneck(protocol_role_scores)
    recovered_semantic_role_scores = [
        ratio * structural * (1.0 - violation) ** 2
        for ratio, structural, violation in zip(
            recovered_required_ratios,
            structural_progress_scores,
            recovery_violation_scores,
        )
    ]
    recovered_semantic_balance = _bottleneck(
        recovered_semantic_role_scores
    )

    plan_score, grounding_support = _plan_quality(detail, cfg)
    recovered_plan_score, recovered_grounding_support = _plan_quality(
        recovered_plan, cfg
    )
    recovered_composite_quality = _geometric_quality(
        recovered_semantic_balance, recovered_plan_score
    )

    strict_validity = float(detail["action_validity"])
    joint_validity = float(detail["team_action_valid"])
    semantic_contribution = float(detail["required_cooperative_contribution"])
    strict_grounding = float(detail["required_grounded_recall"])
    strict_composite_quality = _geometric_quality(
        semantic_contribution, plan_score
    )
    final_success = float(detail["ultimate/collaboration_success"])
    positive_components = {
        "protocol_progress": cfg.protocol_progress_weight * protocol_progress,
        "recovered_semantic": cfg.recovered_semantic_weight
        * recovered_composite_quality,
        "action_validity": cfg.action_validity_weight * strict_validity,
        "team_action": cfg.team_action_weight * joint_validity,
        "strict_balance": cfg.strict_balance_weight
        * joint_validity
        * semantic_contribution,
        "strict_quality": cfg.strict_quality_weight
        * joint_validity
        * strict_composite_quality,
        "strict_grounding": cfg.strict_grounding_weight
        * joint_validity
        * strict_grounding,
        "final_success": cfg.final_success_bonus * final_success,
    }
    penalty_components = {
        "invalid_action": cfg.invalid_action_penalty
        * max(
            detail["invalid_action_rate"],
            shaping.details["recovery_violation_rate"],
        ),
        "conflict": cfg.conflict_penalty * detail["conflict_rate"],
        "overlap": cfg.overlap_penalty * detail["overlap_rate"],
        "reference_copy": cfg.reference_copy_penalty
        * shaping.details["reference_copy_rate"],
        "overlong": cfg.overlong_penalty * shaping.details["overlong_rate"],
    }
    unclamped_reward = math.fsum(
        (
            *positive_components.values(),
            *(-value for value in penalty_components.values()),
        )
    )
    perfect_endpoint = (
        final_success == 1.0
        and strict_validity == 1.0
        and joint_validity == 1.0
        and semantic_contribution == 1.0
        and strict_grounding == 1.0
        and protocol_progress == 1.0
        and recovered_semantic_balance == 1.0
        and plan_score == 1.0
        and recovered_plan_score == 1.0
        and all(value == 0.0 for value in penalty_components.values())
    )
    if (
        perfect_endpoint
        and abs(unclamped_reward - 1.0) <= 8 * math.ulp(1.0)
    ):
        # Keep the semantic maximum stable across Python versions without
        # rounding genuinely near-perfect samples up to a perfect score.
        unclamped_reward = 1.0
    reward = _clamp(unclamped_reward, cfg.min_reward, cfg.max_reward)

    reward_detail = {
        **detail,
        "reward": float(reward),
        "unclamped_reward": float(unclamped_reward),
        "plan_score": float(plan_score),
        "strict_composite_quality": float(strict_composite_quality),
        "joint_validity_gate": float(joint_validity),
        "joint_quality_gate": float(
            joint_validity * strict_composite_quality
        ),
        "protocol_progress": float(protocol_progress),
        "recovered_semantic_balance": float(recovered_semantic_balance),
        "recovered_plan_score": float(recovered_plan_score),
        "recovered_composite_quality": float(recovered_composite_quality),
        "shaping/grounding_support": float(grounding_support),
        "shaping/recovered_grounding_support": float(
            recovered_grounding_support
        ),
        **{
            f"shaping/{key}": float(value)
            for key, value in shaping.details.items()
        },
        **{
            f"shaping/recovered_plan/{key}": float(recovered_plan[key])
            for key in (
                "assignment_coverage",
                "required_fill_rate",
                "required_grounded_recall",
                "entity_grounding_precision",
                "grounding_f1",
                "commonsense_soft",
                "hard_constraint_soft",
            )
        },
        **{
            f"shaping/agent_{agent_idx}/recovered_required_contribution": float(
                value
            )
            for agent_idx, value in enumerate(recovered_required_ratios)
        },
        **{
            f"shaping/agent_{agent_idx}/protocol_progress": float(value)
            for agent_idx, value in enumerate(protocol_role_scores)
        },
        **{
            f"shaping/agent_{agent_idx}/recovered_semantic_progress": float(
                value
            )
            for agent_idx, value in enumerate(recovered_semantic_role_scores)
        },
        **{
            f"reward_component/{key}": float(value)
            for key, value in positive_components.items()
        },
        **{
            f"reward_penalty/{key}": float(value)
            for key, value in penalty_components.items()
        },
        "reward_backend": "reference_constraint_learnable_budget_dense",
    }
    return float(reward), reward_detail


def make_reward(config: Dict[str, Any] | None = None) -> TravelJointReward:
    return TravelJointReward(TravelRewardConfig.from_dict(config or {}))
