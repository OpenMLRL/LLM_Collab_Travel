"""Evaluation logging for single-turn TravelPlanner MAGRPO runs."""

from __future__ import annotations

from numbers import Real
from typing import Any, Callable, Dict, Iterable, List

import numpy as np

from single_turn.rewards.single_turn_reward import (
    TravelRewardConfig,
    score_single_turn_response,
)


def _prompt_key(prompt: str) -> str:
    return " ".join((prompt or "").split()).strip()


def _scalarize_reward_details(detail: Dict[str, Any]) -> Dict[str, float]:
    """Keep the dense scalar diagnostics emitted by the reward scorer."""

    return {
        key: float(value)
        for key, value in detail.items()
        if isinstance(value, Real)
    }


def build_single_turn_eval_logger(
    eval_rows: Iterable[Dict[str, Any]],
    *,
    reward_config: Dict[str, Any] | None = None,
) -> Callable[..., List[Dict[str, Any]]]:
    row_by_prompt = {
        _prompt_key(row.get("prompt") or row.get("query") or ""): row
        for row in eval_rows
    }
    cfg = TravelRewardConfig.from_dict(reward_config or {})

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
                continue
            completions = []
            for agent_idx in range(len(agent_completions_turns)):
                per_sample = agent_completions_turns[agent_idx][sample_idx]
                completions.append(per_sample[0] if per_sample else "")
            _, detail = score_single_turn_response(
                completions,
                batch_item=row,
                config=cfg,
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
            if key.startswith("turn_1/") and isinstance(value, (int, float))
        }
    )
    return {
        key: float(np.mean([float(sample[key]) for sample in metrics_list if key in sample]))
        for key in keys
    }
