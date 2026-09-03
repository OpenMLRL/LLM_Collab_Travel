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


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class TravelRewardConfig:
    # Protocol signal. The plan terms below sum to one before gating.
    parse_weight: float = 0.05
    assignment_coverage_weight: float = 0.10
    required_grounded_weight: float = 0.15
    grounding_f1_weight: float = 0.15
    commonsense_weight: float = 0.35
    hard_constraint_weight: float = 0.25
    final_success_bonus: float = 0.10

    # A non-zero floor keeps early rollouts informative, while the joint gate
    # and deficit penalty still require useful contributions from both agents.
    cooperation_floor: float = 0.20
    overlap_penalty: float = 0.10
    conflict_penalty: float = 0.20
    invalid_action_penalty: float = 0.10
    contribution_deficit_penalty: float = 0.15
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
        if not 0.0 <= self.cooperation_floor <= 1.0:
            raise ValueError("cooperation_floor must lie in [0, 1].")
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

    gate_signal = math.sqrt(
        max(
            0.0,
            detail["team_soft_action_validity"] * detail["cooperative_contribution"],
        )
    )
    cooperation_gate = (
        cfg.cooperation_floor + (1.0 - cfg.cooperation_floor) * gate_signal
    )
    plan_score = (
        cfg.assignment_coverage_weight * detail["assignment_coverage"]
        + cfg.required_grounded_weight * detail["required_grounded_recall"]
        + cfg.grounding_f1_weight * detail["grounding_f1"]
        + cfg.commonsense_weight * detail["commonsense_soft"]
        + cfg.hard_constraint_weight * detail["hard_constraint_soft"]
    )
    reward = (
        cfg.parse_weight * detail["parse_success"]
        + cooperation_gate * plan_score
        + cfg.final_success_bonus * detail["ultimate/collaboration_success"]
        - cfg.overlap_penalty * detail["overlap_rate"]
        - cfg.conflict_penalty * detail["conflict_rate"]
        - cfg.invalid_action_penalty * detail["invalid_action_rate"]
        - cfg.contribution_deficit_penalty * detail["contribution_deficit"]
    )
    reward = _clamp(reward, cfg.min_reward, cfg.max_reward)

    reward_detail = {
        **detail,
        "reward": float(reward),
        "plan_score": float(plan_score),
        "cooperation_gate": float(cooperation_gate),
        "reward_backend": "reference_constraint_v1",
    }
    return float(reward), reward_detail


def make_reward(config: Dict[str, Any] | None = None) -> TravelJointReward:
    return TravelJointReward(TravelRewardConfig.from_dict(config or {}))
