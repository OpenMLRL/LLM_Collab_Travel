"""Checkpoint resume keeps the Travel curriculum on its global epoch schedule."""

from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
COMLRL_ROOT = REPO_ROOT.parent / "CoMLRL"
for path in (COMLRL_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from comlrl.trainers.reinforce import MAGRPOConfig
from single_turn.train.structured_trainer import StructuredOutputMAGRPOTrainer


class _Interrupted(Exception):
    pass


class _TinyCurriculumTrainer(StructuredOutputMAGRPOTrainer):
    def __init__(self, stop_epoch=None):
        # Keep the real Travel wrapper/dataloader and the stock checkpoint loop.
        # Small linear updates replace only language-model generation and loss.
        self.args = MAGRPOConfig(
            num_agents=1,
            num_train_epochs=3,
            num_turns=1,
            agent_devices="cpu",
            eval_interval=0,
        )
        self.train_dataset = [{"target": 1.0}, {"target": 3.0}]
        self.curriculum_train_dataset = self.train_dataset[:1]
        self.curriculum_short_epochs = 2
        self.agents = [torch.nn.Linear(1, 1)]
        self.agent_devices = [torch.device("cpu")]
        self.optimizers = [torch.optim.AdamW(self.agents[0].parameters(), lr=0.03)]
        self.rollout_buffers = [[]]
        self.env_step = 0
        self._last_train_log_step = -1
        self.wandb_config = None
        self.wandb_initialized = False
        self.verbose = False
        self.num_agents = 1
        self._travel_training_active = False
        self._travel_train_epoch_cursor = 0
        self._travel_curriculum_stage = 1
        self._travel_train_detail_groups = []
        self.events = []
        self.stop_epoch = stop_epoch

    def _train_step_returns(self, batch_item, **kwargs):
        value = (
            batch_item["target"]
            + random.random()
            + float(np.random.random())
            + torch.rand(()).item()
        )
        self.events.append(
            (
                self._travel_train_epoch_cursor - 1,
                self._travel_curriculum_stage,
                batch_item["target"],
                value,
            )
        )
        self.rollout_buffers[0].append(value)
        self.env_step += 1

    def _drain_ready_buffers(self, ready_agents):
        assert ready_agents == [0]
        target = torch.tensor(self.rollout_buffers[0]).mean()
        optimizer = self.optimizers[0]
        optimizer.zero_grad()
        loss = (self.agents[0](torch.ones(1, 1)).squeeze() - target).square()
        loss.backward()
        optimizer.step()
        self.rollout_buffers[0].clear()

    def _checkpoint_commit(self, progress, force=False):
        super()._checkpoint_commit(progress, force=force)
        if progress["epoch"] == self.stop_epoch:
            raise _Interrupted()


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make(stop_epoch=None):
    _seed(37)
    trainer = _TinyCurriculumTrainer(stop_epoch)
    _seed(83)
    return trainer


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for value, target in zip(actual, expected):
            _assert_equal(value, target)
    else:
        assert actual == expected


@pytest.mark.parametrize("stop_epoch", [1, 2, 3])
def test_curriculum_resume_matches_uninterrupted_training(tmp_path, stop_epoch):
    baseline = _make()
    baseline.train()
    assert [event[:3] for event in baseline.events] == [
        (0, 0, 1.0),
        (1, 0, 1.0),
        (2, 1, 1.0),
        (2, 1, 3.0),
    ]
    expected_random = random.getstate()
    expected_numpy = np.random.get_state()
    expected_torch = torch.get_rng_state().clone()

    interrupted = _make(stop_epoch)
    with pytest.raises(_Interrupted):
        interrupted.train(checkpoint_dir=tmp_path)
    assert not interrupted._travel_training_active
    assert interrupted._travel_curriculum_stage == 1

    resumed = _make()
    _seed(991)
    resumed.train(resume_from_checkpoint=tmp_path / f"checkpoint-{stop_epoch}")
    assert interrupted.events + resumed.events == baseline.events
    assert resumed._travel_train_epoch_cursor == 3
    assert resumed.env_step == baseline.env_step == 4
    assert not resumed._travel_training_active
    assert resumed._travel_curriculum_stage == 1
    _assert_equal(resumed.agents[0].state_dict(), baseline.agents[0].state_dict())
    _assert_equal(
        resumed.optimizers[0].state_dict(), baseline.optimizers[0].state_dict()
    )
    assert random.getstate() == expected_random
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == expected_numpy[0]
    np.testing.assert_array_equal(actual_numpy[1], expected_numpy[1])
    assert actual_numpy[2:] == expected_numpy[2:]
    assert torch.equal(torch.get_rng_state(), expected_torch)
