from __future__ import annotations

import copy
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT.parent / "CoMLRL", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from transformers import GPT2LMHeadModel

from comlrl.trainers.preference import (
    MADPOConfig,
    MADPOIterConfig,
    MARLHFConfig,
    MARLHFIterConfig,
)
from comlrl.trainers.preference.collaboration import CentralizedCollaboration
from comlrl.trainers.reinforce import CentralizedMAGRPOConfig
from single_turn.centralized_collaboration import TravelCentralizedCollaborationAdapter
from single_turn.centralized_comparator import joint_slots
from single_turn.formatting import build_agent_json_prefill, get_single_turn_formatters
from single_turn.logger import (
    aggregate_single_turn_metrics,
    build_single_turn_eval_logger,
)
from single_turn.rewards import make_reward
from single_turn.structured_generation import FixedSlotJSONLogitsProcessor
from single_turn.train.centralized_trainer import (
    TravelCentralizedMAGRPOTrainer,
    TravelCentralizedMADPOTrainer,
    TravelCentralizedMADPOIterTrainer,
    TravelCentralizedMARLHFTrainer,
    TravelCentralizedMARLHFIterTrainer,
)
from single_turn.train.preference_trainer import TravelPreferenceMixin
from single_turn.train.structured_trainer import StructuredOutputMAGRPOTrainer
from single_turn.train.train_centralized import (
    ALGORITHMS,
    load_config,
    make_trainer_args,
)
from single_turn.train.train_mapl import (
    load_config as load_mapl,
    make_trainer_args as make_mapl_args,
)
import test_preference_training as preference_tests
from test_preference_training import tiny_config
from test_single_turn import fixture_item, valid_completions, all_dash_completions
from test_structured_generation import _TinyGenerationTokenizer, FakeTokenizer


def joint_text(roles):
    values = {
        (a["day"], a["field"]): a["value"]
        for role in roles
        for a in json.loads(role)["assignments"]
    }
    return json.dumps(
        {
            "agent_id": 0,
            "assignments": [
                {"day": d, "field": f, "value": values[d, f]} for d, f in joint_slots(3)
            ],
        }
    )


class TinyTokenizer(_TinyGenerationTokenizer):
    def encode(self, text, *, add_special_tokens):
        return super().encode(text, add_special_tokens=False)


class TinyAdapter(TravelCentralizedCollaborationAdapter):
    def build_prompt(self, batch_item, agent_prompts):
        return "joint context"


TRAINERS = (
    (TravelCentralizedMAGRPOTrainer, CentralizedMAGRPOConfig),
    (TravelCentralizedMADPOTrainer, MADPOConfig),
    (TravelCentralizedMADPOIterTrainer, MADPOIterConfig),
    (TravelCentralizedMARLHFTrainer, MARLHFConfig),
    (TravelCentralizedMARLHFIterTrainer, MARLHFIterConfig),
)


def make_tiny_trainer(cls, config_cls, directory):
    options = dict(
        collaboration_mode="centralized",
        num_agents=2,
        num_turns=1,
        agent_devices=["cpu"],
        parallel_training="none",
        num_train_epochs=1,
        eval_interval=0,
        eval_num_samples=1,
        max_new_tokens=2048,
        train_batch_size=1,
        rollout_buffer_size=1,
        advantage_normalization=False,
    )
    magrpo = cls is TravelCentralizedMAGRPOTrainer
    if not magrpo:
        options.update(preference_num_candidates=2, preference_pairs_per_sample=1)
    if cls in (TravelCentralizedMADPOIterTrainer, TravelCentralizedMARLHFIterTrainer):
        options.update(
            num_iterations=2,
            comparator_policy="current_copy",
            comparator_devices=["cpu"],
            preference_replay_dir=str(Path(directory) / "replay"),
            preference_replay_mode="lambda_decay",
            preference_replay_lambda=0.8,
            log_reward_distribution=True,
        )
    if magrpo or cls in (
        TravelCentralizedMARLHFTrainer,
        TravelCentralizedMARLHFIterTrainer,
    ):
        options.update(num_generations=2)
    if cls in (TravelCentralizedMARLHFTrainer, TravelCentralizedMARLHFIterTrainer):
        options.update(reward_model_device="cpu", reward_num_train_epochs=1)
    mode = {
        (
            "centralized_adapter" if magrpo else "centralized_comparator_adapter"
        ): TinyAdapter()
    }
    trainer = cls(
        agents=[GPT2LMHeadModel(tiny_config())],
        tokenizer=TinyTokenizer(),
        num_agents=2,
        args=config_cls(**options),
        reward_func=make_reward(),
        train_dataset=[fixture_item()],
        eval_dataset=[fixture_item()],
        formatters=get_single_turn_formatters(num_agents=2, use_chat_template=False),
        eval_logger=build_single_turn_eval_logger([fixture_item()]),
        eval_aggregator=aggregate_single_turn_metrics,
        chat_formatted_prompts=True,
        **mode,
    )
    trainer.verbose = False
    return trainer


def scripted_generation(
    self, agent, batch_items, *, agent_idx, num_return_sequences, **kwargs
):
    assert agent_idx == 0
    assert kwargs["slot_order_override"] == joint_slots(batch_items[0]["days"])
    idx = getattr(agent, "test_candidate_index", 0)
    n = num_return_sequences
    agent.test_candidate_index = idx + n
    fulls = [
        joint_text(
            valid_completions()
            if (idx + offset) % 4 in (0, 3)
            else all_dash_completions()
        )
        for offset in range(n)
    ]
    prefix = build_agent_json_prefill(0, 3)
    ids = [torch.tensor([ord(char) for char in full[len(prefix) :]]) for full in fulls]
    grammar = FixedSlotJSONLogitsProcessor(
        self.tokenizers[0], prompt_length=2, slots=joint_slots(3)
    )
    masks = [
        torch.tensor(mask, dtype=torch.bool)
        for mask in grammar.finalize_loss_masks(ids)
    ]
    prompt = kwargs["prompts_override"][0] + prefix
    return {
        "prompts": [prompt],
        "prompt_input_ids": torch.tensor([[2, 3]]),
        "prompt_attention_mask": torch.ones(1, 2),
        "batch_items": batch_items,
        "completions": [fulls],
        "completion_input_ids": [ids],
        "completion_attention_mask": [[torch.ones_like(tokens) for tokens in ids]],
        "completion_loss_mask": [masks],
        "response_lens": [len(tokens) for tokens in ids],
        "reference_kls": [0.0] * n,
    }


class CentralizedConfigTests(unittest.TestCase):
    def test_entrypoint_passes_one_model_and_the_correct_adapter_for_every_algorithm(
        self,
    ):
        from single_turn.train import train_centralized as entrypoint

        for algorithm, (cls, _) in zip(ALGORITHMS, TRAINERS):
            with self.subTest(
                algorithm=algorithm
            ), tempfile.TemporaryDirectory() as directory:
                actor = SimpleNamespace(gradient_checkpointing_enable=Mock())
                trainer = SimpleNamespace(
                    agents=[actor],
                    train=Mock(),
                    evaluate=Mock(return_value={"eval_full/reward": 0.5}),
                    env_step=0,
                    wandb_initialized=False,
                )
                section = "magrpo" if algorithm == "magrpo" else "mapl"
                overrides = [
                    f"output.base_dir={directory}",
                    "output.save_final_model=false",
                    "wandb.enabled=false",
                    f"{section}.eval_num_samples=1",
                    f"{section}.num_train_epochs=1",
                    "travel.curriculum_short_epochs=0",
                    (
                        "magrpo.final_eval_num_samples=1"
                        if algorithm == "magrpo"
                        else "evaluation.final_num_samples=1"
                    ),
                ]
                with patch.object(
                    entrypoint,
                    "_load_data",
                    return_value=([fixture_item()], [fixture_item()]),
                ), patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=TinyTokenizer(),
                ) as tokenizer, patch(
                    f"single_turn.train.centralized_trainer.{cls.__name__}",
                    return_value=trainer,
                ) as construct, patch.object(
                    sys,
                    "argv",
                    ["train", "--algorithm", algorithm, "--override", *overrides],
                ), redirect_stdout(
                    io.StringIO()
                ):
                    entrypoint.main()
                tokenizer.assert_called_once()
                kwargs = construct.call_args.kwargs
                self.assertEqual(kwargs["num_agents"], 2)
                self.assertIsNone(kwargs["agents"])
                self.assertNotIsInstance(kwargs["tokenizer"], (list, tuple))
                self.assertEqual(kwargs["args"].collaboration_mode, "centralized")
                self.assertIn(
                    (
                        "centralized_adapter"
                        if algorithm == "magrpo"
                        else "centralized_comparator_adapter"
                    ),
                    kwargs,
                )
                trainer.train.assert_called_once()
                trainer.evaluate.assert_called_once_with(num_eval_samples=1)
                self.assertEqual(len(list(Path(directory).glob("*/config.yaml"))), 1)
                (saved,) = Path(directory).glob("*/final_metrics.json")
                self.assertEqual(
                    json.loads(saved.read_text()), {"eval_full/reward": 0.5}
                )

    def test_new_configs_are_explicit_and_inherit_domain_budget(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                config = load_config(
                    ROOT
                    / f"single_turn/configs/single_turn_centralized_{algorithm}_config.yaml"
                )
                args = make_trainer_args(config)
                self.assertEqual(args.collaboration_mode, "centralized")
                self.assertEqual(args.num_agents, 2)
                self.assertEqual(args.agent_devices, ["cuda:0"])
                self.assertEqual(args.max_new_tokens, 2048)
                self.assertIn("centralized", config.get("output.base_dir"))
                self.assertIn("cenactor", config.get("wandb.name"))
                base = load_config(
                    ROOT / f"single_turn/configs/single_turn_{algorithm}_config.yaml"
                )
                for key in ("dataset", "travel_reward"):
                    self.assertEqual(config.get_section(key), base.get_section(key))
                if algorithm.endswith("_iter"):
                    self.assertEqual(args.num_iterations, 6)
                    self.assertEqual(args.comparator_generation_mode, "centralized")
                    self.assertEqual(args.preference_replay_lambda, 0.8)
                if algorithm != "magrpo":
                    dec = make_mapl_args(
                        load_mapl(
                            ROOT
                            / f"single_turn/configs/single_turn_{algorithm}_config.yaml"
                        )
                    )
                    self.assertEqual(dec.collaboration_mode, "decentralized")
                    self.assertEqual(dec.agent_devices, ["cuda:0", "cuda:0"])
                    self.assertEqual(dec.max_new_tokens, 1024)

    def test_rejects_wrong_entrypoint_mode_multiple_actors_or_devices(self):
        for override in (
            {"mapl": {"collaboration_mode": "decentralized"}},
            {"mapl": {"agent_devices": ["cuda:0", "cuda:0"]}},
            {"mapl": {"comparator_generation_mode": "decentralized"}},
            {"mapl": {"comparator_devices": ["cuda:0", "cuda:1"]}},
            {"agents": ["a", "b"]},
        ):
            config = load_config(
                ROOT
                / "single_turn/configs/single_turn_centralized_madpo_iter_config.yaml"
            )
            config.update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                make_trainer_args(config)

    def test_prompt_wraps_once_and_does_not_read_ground_truth(self):
        item = fixture_item()
        item.update(annotated_plan="SECRET_GOLD_1", gold_plan="SECRET_GOLD_2")
        tokenizer = FakeTokenizer()
        prompt = TravelCentralizedCollaborationAdapter(tokenizer).build_prompt(item, [])
        self.assertNotIn("SECRET_GOLD", prompt)
        self.assertEqual(prompt.count("<assistant>"), 1)
        self.assertIn("REFERENCE ROLE 0", prompt)
        self.assertIn("REFERENCE ROLE 1", prompt)
        self.assertIn("exactly 21 assignments", prompt)

    def test_reward_is_same_as_decentralized_and_bad_output_is_not_an_exception(self):
        reward = make_reward()
        roles = get_single_turn_formatters(num_agents=2, use_chat_template=False)
        runtime = CentralizedCollaboration(
            TravelCentralizedCollaborationAdapter(), roles, reward, 2
        )
        item = fixture_item()
        for outputs in (valid_completions(), all_dash_completions()):
            expected = reward([outputs[0]], [outputs[1]], batch_items=[item])
            self.assertEqual(
                runtime([joint_text(outputs)], batch_items=[item]), expected
            )
        self.assertEqual(runtime.reward_range, (-0.25, 1.0))
        self.assertEqual(
            runtime(["unfinished JSON"], batch_items=[item]),
            reward([""], [""], batch_items=[item]),
        )


class CentralizedLoopTests(unittest.TestCase):
    def test_joint_dpo_update_matches_masked_single_policy_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_tiny_trainer(
                TravelCentralizedMADPOTrainer, MADPOConfig, directory
            )
            with patch.object(
                StructuredOutputMAGRPOTrainer,
                "_generate_completions",
                scripted_generation,
            ):
                pairs = trainer._generate_preference_pairs_for_item(fixture_item())
            actor = trainer.agents[0]
            before = copy.deepcopy(actor.state_dict())
            delta = trainer._agent_logprob_delta(0, pairs[0])
            loss = -torch.nn.functional.logsigmoid(trainer.args.dpo_beta * delta)
            trainer.optimizers[0].zero_grad()
            loss.backward()
            # Keep a raw SGD step to compare gradients without optimizer state.
            expected = {
                key: param.detach().clone() - 0.01 * param.grad
                for key, param in actor.named_parameters()
            }
            actor.load_state_dict(before)
            trainer.optimizers[0] = torch.optim.SGD(actor.parameters(), lr=0.01)
            trainer._update_from_preference_batch(pairs)
            for key, param in actor.named_parameters():
                torch.testing.assert_close(param, expected[key], rtol=1e-5, atol=1e-7)

    def test_current_history_and_model_comparators_return_joint_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_tiny_trainer(
                TravelCentralizedMADPOIterTrainer, MADPOIterConfig, directory
            )
            for policy in ("current", "history", "model"):
                with self.subTest(policy=policy):
                    trainer.args.comparator_policy = policy
                    trainer._preference_trajectories = {}
                    with patch.object(
                        trainer,
                        "_get_centralized_comparator_agent",
                        return_value=trainer.agents[0],
                    ), patch.object(
                        trainer,
                        "_load_single_policy_checkpoint_agent",
                        return_value=trainer.agents[0],
                    ), patch.object(
                        trainer,
                        "_history_policy_checkpoint_path",
                        return_value="initial",
                    ), patch.object(
                        StructuredOutputMAGRPOTrainer,
                        "_generate_completions",
                        scripted_generation,
                    ):
                        result = trainer._generate_comparator_outputs_for_item(
                            fixture_item(), iteration_idx=0, num_candidates=2
                        )
                    self.assertEqual(len(result), 1)
                    self.assertEqual(len(result[0]["completions"][0]), 2)
                    self.assertTrue(
                        all(
                            len(json.loads(text)["assignments"]) == 21
                            for text in result[0]["completions"][0]
                        )
                    )

    def test_all_five_real_loops_update_one_actor_and_keep_original_eval(self):
        torch.set_num_threads(1)
        for cls, config_cls in TRAINERS:
            with self.subTest(
                algorithm=cls.__name__
            ), tempfile.TemporaryDirectory() as directory:
                trainer = make_tiny_trainer(cls, config_cls, directory)
                self.assertEqual(trainer.args.num_agents, 2)
                self.assertEqual(
                    (
                        trainer.num_agents,
                        len(trainer.agents),
                        len(trainer.optimizers),
                        len(trainer.tokenizers),
                    ),
                    (1, 1, 1, 1),
                )
                before = copy.deepcopy(trainer.agents[0].state_dict())
                with patch.object(
                    StructuredOutputMAGRPOTrainer,
                    "_generate_completions",
                    scripted_generation,
                ), patch(
                    "single_turn.train.preference_trainer.TravelRewardModelMixin._init_reward_model",
                    preference_tests.PreferenceLoopTests.tiny_reward_init,
                ):
                    trainer.train()
                    # The actual compact logger receives the original two roles.
                    metrics = trainer.evaluate(num_eval_samples=1)
                self.assertGreater(trainer.env_step, 0)
                self.assertTrue(
                    any(
                        not torch.equal(value, before[key])
                        for key, value in trainer.agents[0].state_dict().items()
                    )
                )
                self.assertIn("eval/reward", metrics)
                self.assertIn("eval/reference_plan_success", metrics)
                self.assertFalse(trainer._centralized_eval_items)
                if cls is TravelCentralizedMAGRPOTrainer:
                    self.assertEqual(
                        trainer.env_step, 2
                    )  # Joint responses, not role projections.
                    self.assertFalse(hasattr(trainer, "reward_model"))
                else:
                    self.assertGreater(trainer.preference_pairs_generated, 0)
                if cls in (
                    TravelCentralizedMADPOIterTrainer,
                    TravelCentralizedMARLHFIterTrainer,
                ):
                    self.assertEqual(
                        len(
                            list(
                                (Path(directory) / "reward_distributions").glob(
                                    "iteration_*.json"
                                )
                            )
                        ),
                        2,
                    )
                    for path in (Path(directory) / "reward_distributions").glob(
                        "iteration_*.json"
                    ):
                        distribution = json.loads(path.read_text())
                        self.assertEqual(
                            (distribution["reward_min"], distribution["reward_max"]),
                            (-0.25, 1.0),
                        )

    def test_real_generation_all_slots_and_original_joint_tokens_survive_replay(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_tiny_trainer(
                TravelCentralizedMADPOIterTrainer, MADPOIterConfig, directory
            )
            trainer.max_value_tokens = 2
            trainer._preference_trajectories = {}
            result = trainer._generate_completions(
                trainer.agents[0],
                [fixture_item()],
                num_return_sequences=1,
                max_new_tokens=2048,
            )
            text = result["completions"][0][0]
            assignments = json.loads(text)["assignments"]
            self.assertEqual(
                [(a["day"], a["field"]) for a in assignments], joint_slots(3)
            )
            tokens = result["completion_input_ids"][0][0]
            mask = result["completion_loss_mask"][0][0]
            self.assertTrue(mask.any())
            self.assertFalse(mask.all())
            self.assertEqual(
                build_agent_json_prefill(0, 3)
                + trainer.tokenizers[0].decode(tokens, skip_special_tokens=True),
                text,
            )
            cached = trainer._preference_tensors_from_text(
                0, result["prompts"][0], text, text
            )
            torch.testing.assert_close(cached.winner_completion_ids, tokens)
            torch.testing.assert_close(cached.winner_loss_mask, mask)

    def test_comparator_keeps_one_joint_output_and_cache_uses_identical_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_tiny_trainer(
                TravelCentralizedMADPOIterTrainer, MADPOIterConfig, directory
            )
            trainer.preference_generation_batch_size = 2
            trainer._preference_trajectories = {}
            prompt = trainer._build_centralized_comparator_prompt(fixture_item())
            self.assertEqual(prompt, trainer.formatters[0](fixture_item()))
            with patch.object(
                StructuredOutputMAGRPOTrainer,
                "_generate_completions",
                scripted_generation,
            ):
                outputs = trainer._generate_centralized_policy_output_for_agent(
                    trainer.agents[0], fixture_item(), prompt=prompt, num_candidates=3
                )
            self.assertEqual(len(outputs), 1)
            self.assertEqual(trainer.preference_joint_candidates, 3)
            self.assertEqual(len(outputs[0]["completions"][0]), 3)
            for text in outputs[0]["completions"][0]:
                self.assertEqual(len(json.loads(text)["assignments"]), 21)
                self.assertIn(
                    (0, prompt + build_agent_json_prefill(0, 3), text),
                    trainer._preference_trajectories,
                )

    def test_joint_replay_roundtrip_rejects_decentralized_records(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_tiny_trainer(
                TravelCentralizedMADPOIterTrainer, MADPOIterConfig, directory
            )
            with patch.object(
                StructuredOutputMAGRPOTrainer,
                "_generate_completions",
                scripted_generation,
            ):
                pairs = trainer._generate_preference_pairs_for_item(
                    fixture_item(), iteration_idx=0
                )
            self.assertTrue(pairs)
            record = json.loads(
                json.dumps(trainer._preference_pair_to_record(pairs[0]))
            )
            restored = trainer._preference_pair_from_record(record)
            self.assertEqual(len(restored.agent_tensors), 1)
            self.assertEqual(restored.winner_completions, pairs[0].winner_completions)
            for key in restored.agent_tensors[0].__dataclass_fields__:
                torch.testing.assert_close(
                    getattr(restored.agent_tensors[0], key),
                    getattr(pairs[0].agent_tensors[0], key),
                )
            record.pop("travel_collaboration_mode")
            with self.assertRaisesRegex(ValueError, "fresh joint-policy replay"):
                trainer._preference_pair_from_record(record)

    def test_context_overflow_is_caught_before_grammar_or_model_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_tiny_trainer(
                TravelCentralizedMADPOIterTrainer, MADPOIterConfig, directory
            )
            trainer.agents[0].config.max_position_embeddings = 100
            with patch.object(
                TravelPreferenceMixin, "_generate_completions"
            ) as generate:
                with self.assertRaisesRegex(ValueError, "refusing truncation"):
                    trainer._generate_completions(
                        trainer.agents[0], [fixture_item()], max_new_tokens=2048
                    )
                generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
