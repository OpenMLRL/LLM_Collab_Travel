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


@dataclass
class TravelRewardConfig:
    # Additive positive components sum to 1.15 at exact validity.  Unlike the
    # former multiplicative cooperation gate, partial progress from either
    # agent survives in the team scalar.
    parse_weight: float = 0.05
    format_progress_weight: float = 0.10
    assignment_progress_weight: float = 0.10
    quoted_grounding_weight: float = 0.10
    plan_quality_weight: float = 0.70
    assignment_coverage_weight: float = 0.10
    required_grounded_weight: float = 0.15
    grounding_f1_weight: float = 0.15
    commonsense_weight: float = 0.35
    hard_constraint_weight: float = 0.25
    final_success_bonus: float = 0.10

    overlap_penalty: float = 0.10
    conflict_penalty: float = 0.20
    invalid_action_penalty: float = 0.10
    contribution_deficit_penalty: float = 0.15
    protocol_deficit_penalty: float = 0.15
    reference_copy_penalty: float = 0.25
    overlong_penalty: float = 0.15
    min_reward: float = -0.50
    max_reward: float = 1.15

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
        scalar_weights = (
            self.parse_weight,
            self.format_progress_weight,
            self.assignment_progress_weight,
            self.quoted_grounding_weight,
            self.plan_quality_weight,
            self.final_success_bonus,
            self.overlap_penalty,
            self.conflict_penalty,
            self.invalid_action_penalty,
            self.contribution_deficit_penalty,
            self.protocol_deficit_penalty,
            self.reference_copy_penalty,
            self.overlong_penalty,
        )
        if any(weight < 0.0 for weight in scalar_weights):
            raise ValueError("Reward weights and penalties must be non-negative.")
        if self.min_reward >= self.max_reward:
            raise ValueError("min_reward must be smaller than max_reward.")


@dataclass
class TravelJointReward:
    config: TravelRewardConfig = field(default_factory=TravelRewardConfig)
    last_details: List[Dict[str, Any]] = field(default_factory=list)

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
        return rewards


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
    shaping_plan = evaluate_reference_plan(
        shaping.recovered_merge,
        batch_item=batch_item,
    )
    shaping_slot_validity = shaping_plan.pop("slot_validity")
    verified_ratios = []
    for agent_idx, assignments in enumerate(
        shaping.recovered_merge.per_agent_assignments
    ):
        owned = owned_slots(agent_idx, int(batch_item["days"]))
        verified = sum(
            shaping_slot_validity.get(slot, 0.0)
            for slot in assignments
            if slot in owned
        )
        verified_ratios.append(verified / max(1, len(owned)))
    mean_verified = sum(verified_ratios) / len(verified_ratios)
    balanced_verified = min(verified_ratios)
    contribution_progress = 0.8 * mean_verified + 0.2 * balanced_verified
    contribution_deficit = 1.0 - contribution_progress

    # Commonsense/hard soft scores contain vacuously satisfied checks. Require
    # some reference-grounded support before those checks can dominate, while
    # retaining a small floor so early partial plans still have a gradient.
    grounding_support = 0.10 + 0.90 * shaping_plan["required_grounded_recall"]
    plan_score = (
        cfg.assignment_coverage_weight * shaping_plan["assignment_coverage"]
        + cfg.required_grounded_weight * shaping_plan["required_grounded_recall"]
        + cfg.grounding_f1_weight * shaping_plan["grounding_f1"]
        + grounding_support
        * (
            cfg.commonsense_weight * shaping_plan["commonsense_soft"]
            + cfg.hard_constraint_weight * shaping_plan["hard_constraint_soft"]
        )
    )
    assignment_progress = (
        0.25 * shaping.details["assignment_progress"]
        + 0.75 * shaping_plan["required_fill_rate"]
    )
    positive_components = {
        "strict_parse": cfg.parse_weight * detail["parse_success"],
        "format_progress": cfg.format_progress_weight
        * shaping.details["format_progress"],
        "assignment_progress": cfg.assignment_progress_weight
        * assignment_progress,
        "quoted_grounding": cfg.quoted_grounding_weight
        * shaping.details["quoted_grounding_score"],
        "plan_quality": cfg.plan_quality_weight * plan_score,
        "final_success": cfg.final_success_bonus
        * detail["ultimate/collaboration_success"],
    }
    penalty_components = {
        "overlap": cfg.overlap_penalty * detail["overlap_rate"],
        "conflict": cfg.conflict_penalty * detail["conflict_rate"],
        "invalid_action": cfg.invalid_action_penalty
        * max(
            detail["invalid_action_rate"],
            shaping.details["recovery_violation_rate"],
        ),
        "contribution_deficit": cfg.contribution_deficit_penalty
        * contribution_deficit,
        "protocol_deficit": cfg.protocol_deficit_penalty
        * (1.0 - detail["parse_success"]),
        "reference_copy": cfg.reference_copy_penalty
        * shaping.details["reference_copy_rate"],
        "overlong": cfg.overlong_penalty * shaping.details["overlong_rate"],
    }
    unclamped_reward = sum(positive_components.values()) - sum(
        penalty_components.values()
    )
    reward = unclamped_reward
    reward = _clamp(reward, cfg.min_reward, cfg.max_reward)

    reward_detail = {
        **detail,
        "reward": float(reward),
        "unclamped_reward": float(unclamped_reward),
        "plan_score": float(plan_score),
        "shaping/contribution_mean": float(mean_verified),
        "shaping/contribution_balanced": float(balanced_verified),
        "shaping/contribution_progress": float(contribution_progress),
        "shaping/contribution_deficit": float(contribution_deficit),
        "shaping/effective_assignment_progress": float(assignment_progress),
        "shaping/grounding_support": float(grounding_support),
        **{
            f"shaping/{key}": float(value)
            for key, value in shaping.details.items()
        },
        **{
            f"shaping/plan/{key}": float(shaping_plan[key])
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
            f"reward_component/{key}": float(value)
            for key, value in positive_components.items()
        },
        **{
            f"reward_penalty/{key}": float(value)
            for key, value in penalty_components.items()
        },
        "reward_backend": "reference_constraint_dense_v5",
    }
    return float(reward), reward_detail


def make_reward(config: Dict[str, Any] | None = None) -> TravelJointReward:
    return TravelJointReward(TravelRewardConfig.from_dict(config or {}))
