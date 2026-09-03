"""Evaluation logging for single-turn TravelPlanner MAGRPO runs."""

from __future__ import annotations

from numbers import Real
from typing import Any, Callable, Dict, Iterable, List

import numpy as np

from single_turn.evaluation import evaluate_single_turn_response


def _prompt_key(prompt: str) -> str:
    return " ".join((prompt or "").split()).strip()


def _scalarize_reward_details(detail: Dict[str, Any]) -> Dict[str, float]:
    """Keep dense scalars; internal counts are consumed by the aggregator."""

    return {
        key: float(value) for key, value in detail.items() if isinstance(value, Real)
    }


def build_single_turn_eval_logger(
    eval_rows: Iterable[Dict[str, Any]],
) -> Callable[..., List[Dict[str, Any]]]:
    row_by_prompt: Dict[str, Dict[str, Any]] = {}
    for row in eval_rows:
        key = _prompt_key(row.get("prompt") or row.get("query") or "")
        if key in row_by_prompt:
            raise ValueError(
                "The fixed eval split contains duplicate normalized prompts; "
                "metric rows would be ambiguous."
            )
        row_by_prompt[key] = row

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
            detail = evaluate_single_turn_response(
                completions,
                batch_item=row,
            )
            sample_metrics: Dict[str, Any] = {"sample_id": row.get("id", sample_idx)}
            for key, value in _scalarize_reward_details(detail).items():
                sample_metrics[f"turn_1/{key}"] = value
            metrics.append(sample_metrics)
        return metrics

    return logger


def aggregate_single_turn_metrics(
    metrics_list: List[Dict[str, Any]], num_turns: int = 1
) -> Dict[str, float]:
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
    return aggregated
