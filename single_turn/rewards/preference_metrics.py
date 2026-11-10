"""Preference metrics: task scores that can choose preference pairs instead of the reward.

A preference run names one with ``travel.preference_metric``, for example
``commonsense_micro``. The launcher passes the function to the CoMLRL trainer as
``preference_metric_func``; it takes the reward function's arguments and returns
one score per joint answer, higher is better. Without
``travel.preference_metric`` the reward chooses the pairs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def _reward_detail(key: str) -> Callable[[Optional[Dict[str, Any]]], Callable]:
    """A metric that reads one field of the reward scorer's per-answer detail."""

    def build(reward_config: Optional[Dict[str, Any]]) -> Callable:
        # Import lazily, as the reward factory does.
        from .single_turn_reward import make_reward

        # A scorer of its own, so the training reward's details stay intact.
        scorer = make_reward(reward_config)

        def metric(*agent_completions, batch_items=None, prompts=None):
            scorer(*agent_completions, batch_items=batch_items, prompts=prompts)
            scorer.drain_details()
            return [float(detail[key]) for detail in scorer.last_details]

        return metric

    return build


PREFERENCE_METRICS = {
    "commonsense_micro": _reward_detail("ultimate/reference_commonsense_micro"),
}


def get_preference_metric(
    name: Optional[str], reward_config: Optional[Dict[str, Any]] = None
) -> Optional[Callable]:
    """Return the named metric, or ``None`` (the reward chooses) for ``None``."""
    if name is None:
        return None
    if name not in PREFERENCE_METRICS:
        raise ValueError(
            f"Unknown travel.preference_metric {name!r}; available: "
            f"{', '.join(sorted(PREFERENCE_METRICS))}."
        )
    return PREFERENCE_METRICS[name](reward_config)
