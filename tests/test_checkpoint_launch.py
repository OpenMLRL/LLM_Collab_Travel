"""Launcher checkpoint options and W&B identity precede trainer construction."""

import copy
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT.parent / "CoMLRL", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import comlrl.utils
from single_turn.config import Config
from single_turn.train.checkpoint_launch import (
    parse_checkpoint_options,
    prepare_wandb_config,
)
from single_turn.train.train_magrpo import _wandb_config
from test_centralized_collaboration import TinyTokenizer
from test_single_turn import fixture_item


def config_with(data):
    config = Config.__new__(Config)
    config.data = copy.deepcopy(data)
    return config


@pytest.mark.parametrize("section", [None, {}, {"checkpoint_dir": None}])
def test_checkpoint_options_default_to_no_training_kwargs(section):
    assert parse_checkpoint_options(config_with({"checkpoint": section})) == {}


@pytest.mark.parametrize(
    "section",
    [
        False,
        [],
        {"unknown": 1},
        {"checkpoint_interval": True},
        {"checkpoint_interval": None},
        {"checkpoint_total_limit": 0},
        {"checkpoint_interval": 1.5},
        {"resume_from_checkpoint": " "},
        {"checkpoint_dir": 3},
    ],
)
def test_invalid_checkpoint_options_fail_before_io(section, monkeypatch):
    monkeypatch.setattr(
        comlrl.utils, "read_checkpoint_metadata", Mock(side_effect=AssertionError)
    )
    with pytest.raises(ValueError, match="checkpoint"):
        parse_checkpoint_options(config_with({"checkpoint": section}))


def test_options_forward_only_explicit_non_null_settings():
    values = {
        "resume_from_checkpoint": Path("saved/checkpoint-1"),
        "checkpoint_dir": "new/checkpoints",
        "checkpoint_interval": 2,
        "checkpoint_total_limit": None,
    }
    assert parse_checkpoint_options(config_with({"checkpoint": values})) == {
        "resume_from_checkpoint": "saved/checkpoint-1",
        "checkpoint_dir": "new/checkpoints",
        "checkpoint_interval": 2,
    }


@pytest.mark.parametrize("raw", [None, False, {"enabled": False}, {"mode": "disabled"}])
def test_disabled_logging_does_not_read_metadata(raw, monkeypatch):
    reader = Mock(side_effect=AssertionError("disabled W&B read metadata"))
    monkeypatch.setattr(comlrl.utils, "read_checkpoint_metadata", reader)
    config = config_with(
        {"wandb": raw, "checkpoint": {"resume_from_checkpoint": "saved"}}
    )
    assert _wandb_config(config, "output", {}) is None
    reader.assert_not_called()


@pytest.mark.parametrize(
    "explicit", [{"id": "chosen"}, {"resume": "never"}, {"id": None}]
)
def test_explicit_logging_identity_wins_without_metadata(explicit, monkeypatch):
    reader = Mock(side_effect=AssertionError("explicit W&B read metadata"))
    monkeypatch.setattr(comlrl.utils, "read_checkpoint_metadata", reader)
    config = config_with(
        {"wandb": explicit, "checkpoint": {"resume_from_checkpoint": "saved"}}
    )
    logging = _wandb_config(config, "output", {})
    for key, value in explicit.items():
        assert logging[key] == value
    reader.assert_not_called()


@pytest.mark.parametrize(
    "identity", [None, {"id": "saved", "project": "old-project", "entity": "old-team"}]
)
def test_resume_identity_preserves_other_logging_settings(identity, monkeypatch):
    reader = Mock(return_value={"wandb": identity})
    monkeypatch.setattr(comlrl.utils, "read_checkpoint_metadata", reader)
    config = config_with({"wandb": {"name": "name", "project": "new-project"}})
    original = {
        "name": "name",
        "project": "new-project",
        "dir": "new-output",
        "tags": ["travel"],
    }
    before = copy.deepcopy(original)
    result = prepare_wandb_config(config, original, {"resume_from_checkpoint": "saved"})
    assert original == before
    assert result == (
        {**original, **identity, "resume": "allow"} if identity else original
    )
    reader.assert_called_once_with("saved")


def test_plain_launch_does_not_read_metadata_or_change_logging(monkeypatch):
    reader = Mock(side_effect=AssertionError("plain launch read metadata"))
    monkeypatch.setattr(comlrl.utils, "read_checkpoint_metadata", reader)
    original = {"name": "name", "project": "project", "config_sections": {}}
    assert prepare_wandb_config(config_with({}), original, {}) is original


ENTRIES = (
    [
        ("train_magrpo", None, "structured_trainer", "StructuredOutputMAGRPOTrainer"),
        ("train_actor_critic", "iac", "actor_critic_trainer", "TravelIACTrainer"),
        ("train_actor_critic", "maac", "actor_critic_trainer", "TravelMAACTrainer"),
    ]
    + [
        ("train_mapl", algorithm, "preference_trainer", name)
        for algorithm, name in (
            ("madpo", "TravelMADPOTrainer"),
            ("madpo_iter", "TravelMADPOIterTrainer"),
            ("marlhf", "TravelMARLHFTrainer"),
            ("marlhf_iter", "TravelMARLHFIterTrainer"),
        )
    ]
    + [
        ("train_centralized", algorithm, "centralized_trainer", name)
        for algorithm, name in (
            ("magrpo", "TravelCentralizedMAGRPOTrainer"),
            ("madpo", "TravelCentralizedMADPOTrainer"),
            ("madpo_iter", "TravelCentralizedMADPOIterTrainer"),
            ("marlhf", "TravelCentralizedMARLHFTrainer"),
            ("marlhf_iter", "TravelCentralizedMARLHFIterTrainer"),
        )
    ]
)


@pytest.mark.parametrize("entry,algorithm,module,trainer_name", ENTRIES)
@pytest.mark.parametrize("resuming", [False, True])
def test_all_entrypoints_prepare_identity_before_construction_and_forward_train_options(
    tmp_path, monkeypatch, entry, algorithm, module, trainer_name, resuming
):
    entrypoint = importlib.import_module(f"single_turn.train.{entry}")
    trainer_module = importlib.import_module(f"single_turn.train.{module}")
    events, constructors = [], []
    identity = {"id": "saved-id", "project": "saved-project", "entity": "saved-team"}

    def metadata(path):
        events.append("metadata")
        assert path == "saved/checkpoint-1"
        return {"wandb": identity}

    def construct(**kwargs):
        events.append("construct")
        constructors.append(kwargs)
        config = kwargs["wandb_config"]
        if resuming:
            assert events == ["metadata", "construct"]
            assert all(config[key] == value for key, value in identity.items())
            assert config["resume"] == "allow"
        else:
            assert events == ["construct"]
            assert "id" not in config and "resume" not in config
        return SimpleNamespace(
            agents=[SimpleNamespace()],
            critics=[],
            tokenizers=[],
            env_step=0,
            preference_joint_candidates=0,
            preference_pairs_generated=0,
            wandb_initialized=False,
            evaluate=Mock(return_value={}),
            train=lambda **options: events.append(("train", options)),
        )

    monkeypatch.setattr(comlrl.utils, "read_checkpoint_metadata", metadata)
    monkeypatch.setattr(
        entrypoint, "_load_data", lambda config: ([fixture_item()], [fixture_item()])
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: TinyTokenizer()
    )
    monkeypatch.setattr(trainer_module, trainer_name, construct)
    section = (
        algorithm
        if algorithm in {"iac", "maac"}
        else ("magrpo" if algorithm in {None, "magrpo"} else "mapl")
    )
    overrides = [
        f"output.base_dir={tmp_path}",
        "output.save_final_model=false",
        "agent_model.gradient_checkpointing=false",
        "travel.curriculum_short_epochs=0",
        "wandb.enabled=true",
        f"{section}.eval_num_samples=1",
        f"{section}.num_train_epochs=1",
        "evaluation.final_num_samples=1",
    ]
    if section == "magrpo":
        overrides.append("magrpo.final_eval_num_samples=1")
    options = {}
    if resuming:
        options = {
            "resume_from_checkpoint": "saved/checkpoint-1",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "checkpoint_interval": 2,
            "checkpoint_total_limit": 3,
        }
        overrides.extend(f"checkpoint.{key}={value}" for key, value in options.items())
    argv = ["train"] + (["--algorithm", algorithm] if algorithm else [])
    monkeypatch.setattr(sys, "argv", argv + ["--override", *overrides])
    entrypoint.main()
    assert len(constructors) == 1
    assert events[-1] == ("train", options)
