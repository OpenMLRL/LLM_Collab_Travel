from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMLRL_ROOT = REPO_ROOT.parent / "CoMLRL"
for path in (COMLRL_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from comlrl.utils.reward_utils import call_reward_function
from single_turn.rewards import make_reward
from single_turn.rewards.preference_metrics import get_preference_metric
from single_turn.rewards.single_turn_reward import score_single_turn_response

from test_single_turn import all_dash_completions, fixture_item, valid_completions


def test_commonsense_micro_matches_the_reward_detail():
    reward = make_reward()
    metric = get_preference_metric("commonsense_micro", {})
    scores = []
    for outputs in (valid_completions(), all_dash_completions()):
        item = fixture_item()
        reward(*[[text] for text in outputs], batch_items=[item])
        pending = len(reward._pending_details)
        score = call_reward_function(
            metric,
            [item["prompt"]],
            [[text] for text in outputs],
            num_agents=2,
            batch_items=[item],
        )
        _, detail = score_single_turn_response(outputs, batch_item=item)
        assert score == [detail["ultimate/reference_commonsense_micro"]]
        # The metric scores with a scorer of its own.
        assert len(reward._pending_details) == pending
        scores.extend(score)
    assert scores[0] == 1.0 > scores[1]
