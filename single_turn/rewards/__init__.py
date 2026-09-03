"""Reward factory for single-turn TravelPlanner collaboration."""

from typing import Any, Dict


def make_reward(config: Dict[str, Any] | None = None):
    """Import lazily so the reward-independent evaluator stays acyclic."""

    from .single_turn_reward import make_reward as _make_reward

    return _make_reward(config)


__all__ = ["make_reward"]
