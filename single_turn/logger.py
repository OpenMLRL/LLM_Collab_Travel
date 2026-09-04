"""Evaluation logging for single-turn TravelPlanner MAGRPO runs."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Mapping

import numpy as np

from single_turn.evaluation import evaluate_single_turn_response


def _prompt_key(prompt: str) -> str:
    return " ".join((prompt or "").split()).strip()


_EVAL_SCALAR_KEYS = (
    "action_validity",
    "ultimate/team_action_success",
    "required_cooperative_contribution",
    "required_grounded_recall",
    "entity_grounding_precision",
    "required_cost_completeness",
    "budget_constraint_soft",
    "route_scaffold_match_rate",
    "ultimate/reference_plan_delivery",
    "ultimate/required_plan_completion",
    "ultimate/reference_commonsense_micro",
    "ultimate/reference_hard_micro",
    "ultimate/reference_budget_pass",
    "ultimate/reference_plan_success",
    "ultimate/collaboration_success",
)

# These counts stay inside the evaluator/aggregator boundary.  They are needed
# to compute correct micro averages, but are deliberately never sent to W&B.
_AGGREGATE_SUPPORT_KEYS = (
    "_aggregate/commonsense_pass_count",
    "_aggregate/commonsense_applicable_count",
    "_aggregate/hard_pass_count",
    "_aggregate/hard_applicable_count",
    "_aggregate/required_valid_count",
    "_aggregate/required_slot_count",
    "_aggregate/grounded_entity_count",
    "_aggregate/predicted_entity_count",
    "_aggregate/required_cost_known_count",
    "_aggregate/required_cost_slot_count",
)


def _scalarize_reward_details(detail: Dict[str, Any]) -> Dict[str, float]:
    """Select only headline eval metrics plus hidden micro-average counts."""

    keys = (*_EVAL_SCALAR_KEYS, *_AGGREGATE_SUPPORT_KEYS)
    missing = [key for key in keys if key not in detail]
    if missing:
        raise KeyError(f"Evaluator omitted required metrics: {missing}")
    return {key: float(detail[key]) for key in keys}


def build_single_turn_eval_logger(
    eval_rows: Iterable[Dict[str, Any]],
    *,
    reward_config: Any = None,
    panel_size: int = 4,
) -> Callable[..., List[Dict[str, Any]]]:
    row_by_prompt: Dict[str, Dict[str, Any]] = {}
    panel_by_prompt: Dict[str, int] = {}
    if int(panel_size) < 1:
        raise ValueError("panel_size must be positive.")
    for row_idx, row in enumerate(eval_rows):
        key = _prompt_key(row.get("prompt") or row.get("query") or "")
        if key in row_by_prompt:
            raise ValueError(
                "The fixed eval split contains duplicate normalized prompts; "
                "metric rows would be ambiguous."
            )
        row_by_prompt[key] = row
        panel_by_prompt[key] = row_idx // int(panel_size) + 1

    def logger(
        agent_completions_turns: List[List[List[str]]],
        test_cases: List[str],
        entry_points: List[str],
        prompts: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        del test_cases, entry_points
        if not agent_completions_turns or prompts is None:
            return []
        metrics: List[Dict[str, Any]] = []
        for sample_idx, prompt in enumerate(prompts):
            row = row_by_prompt.get(_prompt_key(prompt))
            if row is None:
                raise KeyError(
                    "Evaluation prompt was not found in the fixed eval split; "
                    "refusing to shrink the metric denominator silently."
                )
            completions = []
            for agent_idx in range(len(agent_completions_turns)):
                per_sample = agent_completions_turns[agent_idx][sample_idx]
                completions.append(per_sample[0] if per_sample else "")
            reward_value = None
            reward_detail: Dict[str, Any] = {}
            if reward_config is not None:
                # Import lazily to keep the reward-independent evaluator free
                # from a reverse dependency on the training reward module.
                from single_turn.rewards.single_turn_reward import (
                    score_single_turn_response,
                )

                reward_value, reward_detail = score_single_turn_response(
                    completions,
                    batch_item=row,
                    config=reward_config,
                )
                # The reward scorer already returns the complete evaluator
                # detail.  Reuse it instead of evaluating the same plan twice.
                detail = reward_detail
            else:
                detail = evaluate_single_turn_response(
                    completions,
                    batch_item=row,
                )
            sample_metrics: Dict[str, Any] = {"sample_id": row.get("id", sample_idx)}
            for key, value in _scalarize_reward_details(detail).items():
                sample_metrics[f"turn_1/{key}"] = value
            sample_metrics["_eval_sample"] = {
                "panel_id": panel_by_prompt[_prompt_key(prompt)],
                "sample_id": str(row.get("id", sample_idx)),
                "days": int(row.get("days", 0)),
                "query": str(row.get("query") or row.get("prompt") or ""),
                "agent_0_output": completions[0],
                "agent_1_output": completions[1],
                "merged_plan": json.dumps(
                    detail["merged_plan"], ensure_ascii=False, sort_keys=True
                ),
                "reward": reward_value,
                "plan_score": reward_detail.get("plan_score"),
                "protocol_progress": reward_detail.get("protocol_progress"),
                "action_validity": detail["action_validity"],
                "team_action_success": detail["ultimate/team_action_success"],
                "required_cooperative_contribution": detail[
                    "required_cooperative_contribution"
                ],
                "required_grounded_recall": detail["required_grounded_recall"],
                "grounding_precision": detail["entity_grounding_precision"],
                "required_cost_completeness": detail[
                    "required_cost_completeness"
                ],
                "budget_constraint_soft": detail["budget_constraint_soft"],
                "budget_pass": detail["ultimate/reference_budget_pass"],
                "route_scaffold_match": detail["route_scaffold_match_rate"],
                "plan_delivery": detail[
                    "ultimate/reference_plan_delivery"
                ],
                "required_plan_completion": detail[
                    "ultimate/required_plan_completion"
                ],
                "commonsense_micro": detail[
                    "ultimate/reference_commonsense_micro"
                ],
                "hard_micro": detail["ultimate/reference_hard_micro"],
                "reference_plan_success": detail[
                    "ultimate/reference_plan_success"
                ],
                "collaboration_success": detail[
                    "ultimate/collaboration_success"
                ],
                "parser_errors": json.dumps(
                    detail["parser_errors"], ensure_ascii=False
                ),
            }
            metrics.append(sample_metrics)
        return metrics

    return logger


def aggregate_single_turn_metrics(
    metrics_list: List[Dict[str, Any]], num_turns: int = 1
) -> Dict[str, Any]:
    del num_turns
    if not metrics_list:
        return {}
    keys = sorted(
        {
            key
            for sample in metrics_list
            for key, value in sample.items()
            if key.startswith("turn_1/")
            and "/_aggregate/" not in key
            and isinstance(value, (int, float))
        }
    )
    aggregated = {
        key: float(
            np.mean([float(sample[key]) for sample in metrics_list if key in sample])
        )
        for key in keys
    }

    # Micro metrics must be ratios of global counts. Averaging per-example
    # ratios is wrong as soon as examples have different numbers of applicable
    # hard constraints or required slots.
    ratio_specs = {
        "turn_1/ultimate/reference_commonsense_micro": (
            "turn_1/_aggregate/commonsense_pass_count",
            "turn_1/_aggregate/commonsense_applicable_count",
        ),
        "turn_1/ultimate/reference_hard_micro": (
            "turn_1/_aggregate/hard_pass_count",
            "turn_1/_aggregate/hard_applicable_count",
        ),
        "turn_1/required_grounded_recall": (
            "turn_1/_aggregate/required_valid_count",
            "turn_1/_aggregate/required_slot_count",
        ),
        "turn_1/entity_grounding_precision": (
            "turn_1/_aggregate/grounded_entity_count",
            "turn_1/_aggregate/predicted_entity_count",
        ),
        "turn_1/required_cost_completeness": (
            "turn_1/_aggregate/required_cost_known_count",
            "turn_1/_aggregate/required_cost_slot_count",
        ),
    }
    for output_key, (numerator_key, denominator_key) in ratio_specs.items():
        numerator = sum(
            float(sample.get(numerator_key, 0.0)) for sample in metrics_list
        )
        denominator = sum(
            float(sample.get(denominator_key, 0.0)) for sample in metrics_list
        )
        aggregated[output_key] = numerator / denominator if denominator > 0 else 0.0

    precision = aggregated["turn_1/entity_grounding_precision"]
    recall = aggregated["turn_1/required_grounded_recall"]
    aggregated["turn_1/grounding_f1"] = (
        2.0 * precision * recall / (precision + recall)
        if precision > 0.0 and recall > 0.0
        else 0.0
    )

    sample_rows = [
        sample["_eval_sample"]
        for sample in metrics_list
        if isinstance(sample.get("_eval_sample"), Mapping)
    ]
    if sample_rows:
        try:
            import wandb
        except ImportError:
            pass
        else:
            columns = [
                "panel_id",
                "sample_id",
                "days",
                "query",
                "agent_0_output",
                "agent_1_output",
                "merged_plan",
                "reward",
                "plan_score",
                "protocol_progress",
                "action_validity",
                "team_action_success",
                "required_cooperative_contribution",
                "required_grounded_recall",
                "grounding_precision",
                "required_cost_completeness",
                "budget_constraint_soft",
                "budget_pass",
                "route_scaffold_match",
                "plan_delivery",
                "required_plan_completion",
                "commonsense_micro",
                "hard_micro",
                "reference_plan_success",
                "collaboration_success",
                "parser_errors",
            ]
            aggregated["turn_1/eval_samples"] = wandb.Table(
                columns=columns,
                # Scalar metrics use the complete eval subset.  Upload one
                # representative collaboration trace per checkpoint to avoid
                # repeatedly shipping four long prompts and generations.
                data=[
                    [sample_rows[0].get(column) for column in columns]
                ],
            )
    return aggregated
