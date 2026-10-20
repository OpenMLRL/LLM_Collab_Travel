"""One joint Travel actor for MAAC, and native joint actors for iterative MADPO/MARLHF."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import GPT2LMHeadModel

from comlrl.trainers.actor_critic import MAACConfig
from single_turn.aggregation import ordered_owned_slots
from single_turn.centralized_collaboration import TravelCentralizedCollaborationAdapter
from single_turn.centralized_native import (
    TravelNativeCentralizedAdapter,
    build_native_prefill,
)
from single_turn.rewards import make_reward
from single_turn.train.actor_critic_trainer import TravelMAACTrainer
from single_turn.train.centralized_trainer import (
    TravelNativeCentralizedMADPOIterTrainer,
    TravelNativeCentralizedMARLHFIterTrainer,
)
from test_actor_critic import ACTokenizer
from test_centralized_collaboration import TinyTokenizer
from test_preference_training import tiny_config
from test_single_turn import fixture_item

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "single_turn" / "configs"


class JointConcatAdapter(TravelCentralizedCollaborationAdapter):
    def build_prompt(self, batch_item, agent_prompts):
        return "joint context\n"


class JointNativeAdapter(TravelNativeCentralizedAdapter):
    def build_prompt(self, batch_item, agent_prompts):
        return "joint context\n"


def joint_maac(adapter):
    torch.manual_seed(42)
    trainer = TravelMAACTrainer(
        agents=[GPT2LMHeadModel(tiny_config())],
        critics=[GPT2LMHeadModel(tiny_config())],
        tokenizer=ACTokenizer(),
        args=MAACConfig(
            collaboration_mode="centralized",
            num_agents=2,
            num_turns=1,
            agent_devices="cpu",
            critic_devices="cpu",
            num_generations=2,
            num_train_epochs=1,
            rollout_buffer_size=2,
            train_batch_size=2,
            max_new_tokens=2048,
            eval_interval=0,
            eval_num_samples=1,
            advantage_normalization=False,
        ),
        reward_func=make_reward(),
        formatters=[lambda item: "role zero\n", lambda item: "role one\n"],
        train_dataset=[fixture_item()],
        eval_dataset=[fixture_item()],
        max_value_tokens=1,
        centralized_adapter=adapter,
    )
    trainer.verbose = False
    return trainer


class CentralizedTravelActorCriticTests(unittest.TestCase):
    def test_joint_maac_writes_the_adapter_layout_and_scores_both_roles(self):
        torch.set_num_threads(1)
        for adapter in (JointConcatAdapter(), JointNativeAdapter()):
            with self.subTest(adapter=type(adapter).__name__):
                trainer = joint_maac(adapter)
                self.assertEqual(trainer.num_agents, 1)
                item = fixture_item()
                samples = trainer._collect_rollouts(item)
                self.assertEqual([s.agent_idx for s in samples], [0, 0])
                for sample in samples:
                    self.assertTrue(
                        sample.completion.startswith(adapter.assistant_prefill(3))
                    )
                    roles = trainer.centralized_collaboration.split(
                        sample.completion, item
                    )
                    for role, text in enumerate(roles):
                        data = json.loads(text)
                        self.assertEqual(data["agent_id"], role)
                        self.assertEqual(
                            [(a["day"], a["field"]) for a in data["assignments"]],
                            ordered_owned_slots(role, 3),
                        )
                    mask = sample.metadata["completion_loss_mask"]
                    self.assertTrue(mask.any() and not mask.all())
                metrics = trainer.evaluate()
                self.assertIn("eval/reward", metrics)
                self.assertIn("eval/reference_plan_success", metrics)

    def test_entrypoint_builds_one_joint_maac_actor(self):
        from single_turn.train import train_actor_critic as entrypoint

        for config, adapter_cls in (
            (
                "single_turn_centralized_maac_config.yaml",
                TravelCentralizedCollaborationAdapter,
            ),
            (
                "single_turn_centralized_native_maac_config.yaml",
                TravelNativeCentralizedAdapter,
            ),
        ):
            with self.subTest(
                config=config
            ), tempfile.TemporaryDirectory() as directory:
                trainer = SimpleNamespace(
                    agents=[],
                    critics=[],
                    train=Mock(),
                    evaluate=Mock(return_value={"eval_full/reward": 0.5}),
                    wandb_initialized=False,
                )
                overrides = [
                    f"output.base_dir={directory}",
                    "output.save_final_model=false",
                    "wandb.enabled=false",
                    "agent_model.gradient_checkpointing=false",
                    "maac.eval_num_samples=1",
                    "maac.num_train_epochs=1",
                    "evaluation.final_num_samples=1",
                ]
                with patch.object(
                    entrypoint,
                    "_load_data",
                    return_value=([fixture_item()], [fixture_item()]),
                ), patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=TinyTokenizer(),
                ) as load_tokenizer, patch(
                    "single_turn.train.actor_critic_trainer.TravelMAACTrainer",
                    return_value=trainer,
                ) as construct, patch.object(
                    sys,
                    "argv",
                    [
                        "train",
                        "--algorithm",
                        "maac",
                        "--config",
                        str(CONFIGS / config),
                        "--override",
                        *overrides,
                    ],
                ), redirect_stdout(
                    io.StringIO()
                ):
                    entrypoint.main()
                kwargs = construct.call_args.kwargs
                self.assertEqual(kwargs["args"].collaboration_mode, "centralized")
                self.assertIsInstance(kwargs["centralized_adapter"], adapter_cls)
                self.assertNotIsInstance(kwargs["tokenizer"], list)
                self.assertEqual(load_tokenizer.call_count, 1)
                self.assertEqual(len(kwargs["formatters"]), 2)
                trainer.train.assert_called_once()


class NativeCentralizedPreferenceTests(unittest.TestCase):
    def test_native_iterative_trainers_use_the_native_layout(self):
        for cls in (
            TravelNativeCentralizedMADPOIterTrainer,
            TravelNativeCentralizedMARLHFIterTrainer,
        ):
            trainer = cls.__new__(cls)
            self.assertEqual(
                trainer._travel_assistant_prefill(0, {"days": 3}),
                build_native_prefill(3),
            )

    def test_reward_model_text_drops_the_native_prefill(self):
        trainer = TravelNativeCentralizedMARLHFIterTrainer.__new__(
            TravelNativeCentralizedMARLHFIterTrainer
        )
        text = trainer._format_joint_text(
            ["joint context\n" + build_native_prefill(3)], ["answer"]
        )
        self.assertEqual(json.loads(text)["role_contexts"], ["joint context\n"])

    def test_entrypoint_builds_native_iterative_joint_actors(self):
        from single_turn.train import train_centralized as entrypoint

        for method, name in (
            ("madpo_iter", "TravelNativeCentralizedMADPOIterTrainer"),
            ("marlhf_iter", "TravelNativeCentralizedMARLHFIterTrainer"),
        ):
            with self.subTest(
                method=method
            ), tempfile.TemporaryDirectory() as directory:
                trainer = SimpleNamespace(
                    agents=[SimpleNamespace()],
                    critics=[],
                    tokenizers=[],
                    env_step=0,
                    preference_joint_candidates=0,
                    preference_pairs_generated=0,
                    wandb_initialized=False,
                    evaluate=Mock(return_value={}),
                    train=Mock(),
                )
                overrides = [
                    f"output.base_dir={directory}",
                    "output.save_final_model=false",
                    "wandb.enabled=false",
                    "agent_model.gradient_checkpointing=false",
                    "travel.curriculum_short_epochs=0",
                    "mapl.eval_num_samples=1",
                    "mapl.num_train_epochs=1",
                    "evaluation.final_num_samples=1",
                ]
                config = (
                    CONFIGS / f"single_turn_centralized_native_{method}_config.yaml"
                )
                with patch.object(
                    entrypoint,
                    "_load_data",
                    return_value=([fixture_item()], [fixture_item()]),
                ), patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=TinyTokenizer(),
                ), patch(
                    f"single_turn.train.centralized_trainer.{name}",
                    return_value=trainer,
                ) as construct, patch.object(
                    sys,
                    "argv",
                    ["train", "--config", str(config), "--override", *overrides],
                ), redirect_stdout(
                    io.StringIO()
                ):
                    entrypoint.main()
                kwargs = construct.call_args.kwargs
                self.assertIsInstance(
                    kwargs["centralized_comparator_adapter"],
                    TravelNativeCentralizedAdapter,
                )
                trainer.train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
