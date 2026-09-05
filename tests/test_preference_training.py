from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT.parent / "CoMLRL", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from comlrl.trainers.preference import MADPOIterConfig, MARLHFConfig, MARLHFIterConfig
from comlrl.trainers.preference.madpo import PreferencePair
from comlrl.trainers.reinforce import MAGRPOTrainer
from single_turn.formatting import build_agent_json_prefill
from single_turn.rewards import make_reward
from single_turn.train.preference_trainer import (
    TravelJointRewardModel, TravelMADPOIterTrainer, TravelMARLHFTrainer,
    TravelMARLHFIterTrainer, TravelPreferenceTensors,
)
from single_turn.train.structured_trainer import StructuredOutputMAGRPOTrainer
from single_turn.train.train_mapl import budget_report, load_config, make_trainer_args
from test_single_turn import fixture_item, valid_completions, all_dash_completions
from test_structured_generation import _PositionIndependentLM, _TinyGenerationTokenizer


def tiny_config():
    return GPT2Config(vocab_size=128, n_embd=8, n_layer=1, n_head=1, n_positions=4096,
                      resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, bos_token_id=1, eos_token_id=1)


def tensors():
    return TravelPreferenceTensors(
        torch.tensor([2, 3]), torch.tensor([4, 5, 6]), torch.tensor([4, 7, 8]),
        torch.tensor([False, True, True]), torch.tensor([False, True, True]),
    )


def pair():
    return PreferencePair(["p0", "p1"], ["win0", "win1"], ["lose0", "lose1"],
                          [tensors(), tensors()], .8, .2, .5)


def bare_trainer():
    trainer = TravelMADPOIterTrainer.__new__(TravelMADPOIterTrainer)
    trainer.agents = [_PositionIndependentLM(10), _PositionIndependentLM(10)]
    trainer.num_agents = 2
    trainer.args = SimpleNamespace(dpo_beta=.4)
    trainer.normalize_value_log_probs = True
    trainer.offload_inactive_actors = True
    trainer.optimizers = [torch.optim.SGD(agent.parameters(), lr=.2) for agent in trainer.agents]
    return trainer


class PreferenceAdapterTests(unittest.TestCase):
    def test_masked_logprob_includes_final_token(self):
        trainer = bare_trainer()
        data = tensors()
        actual = trainer._sequence_log_prob(0, data.prompt_input_ids, data.winner_completion_ids, data.winner_loss_mask)
        logits = trainer.agents[0].token_logits
        expected = torch.log_softmax(logits, -1)[torch.tensor([5, 6])].mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        torch.testing.assert_close(logits.grad, torch.autograd.grad(expected, logits)[0])
        with self.assertRaisesRegex(ValueError, "mask"):
            trainer._sequence_log_prob(0, data.prompt_input_ids, data.winner_completion_ids)

    def test_streaming_joint_dpo_matches_original_objective(self):
        actual, expected = bare_trainer(), bare_trainer()
        batch = [pair(), pair()]
        # Different advantages for different agents/pairs exercise joint coupling.
        batch[1].agent_tensors[1].winner_completion_ids = torch.tensor([4, 7, 9])
        deltas = expected._detached_agent_deltas(batch)
        for idx in range(2):
            expected.optimizers[idx].zero_grad()
            expected._agent_preference_loss(idx, batch, deltas).backward()
            expected.optimizers[idx].step()
        actual._update_from_preference_batch(batch)
        for left, right in zip(actual.agents, expected.agents):
            torch.testing.assert_close(left.token_logits, right.token_logits)

    def test_replay_preserves_exact_generated_tokens_and_masks(self):
        trainer = bare_trainer()
        original = pair()
        record = json.loads(json.dumps(trainer._preference_pair_to_record(original)))
        restored = trainer._preference_pair_from_record(record)
        for before, after in zip(original.agent_tensors, restored.agent_tensors):
            for key in TravelPreferenceTensors.__dataclass_fields__:
                torch.testing.assert_close(getattr(before, key), getattr(after, key))
        del record["travel_tensor_version"]
        with self.assertRaisesRegex(ValueError, "fresh"):
            trainer._preference_pair_from_record(record)

    def test_all_tied_preferences_fail_instead_of_silent_noop(self):
        trainer = bare_trainer()
        trainer.args.eval_interval = 0
        trainer.preference_pairs_generated = 0
        trainer._train_preference_algorithm = Mock()
        with self.assertRaisesRegex(RuntimeError, "No non-tied"):
            trainer.train()

    def test_eval_baseline_cached_until_actor_update(self):
        trainer = bare_trainer()
        trainer.args.eval_num_samples = 4
        trainer.env_step = 0
        with patch.object(StructuredOutputMAGRPOTrainer, "evaluate", return_value={"eval/reward": .4}) as evaluate:
            trainer.evaluate()
            trainer.evaluate()
            self.assertEqual(evaluate.call_count, 1)
            trainer._update_from_preference_batch([pair()])
            trainer.evaluate()
            self.assertEqual(evaluate.call_count, 2)

    def test_cpu_offload_restores_actors_even_if_phase_fails(self):
        trainer = bare_trainer()
        events = []
        actor = SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda:0"))]),
            to=Mock(side_effect=lambda device: events.append(str(device))),
        )
        trainer.agents = [actor]
        trainer.optimizers = [SimpleNamespace(state={}, zero_grad=Mock())]
        with patch("torch.cuda.empty_cache"), self.assertRaisesRegex(RuntimeError, "phase failed"):
            with trainer._offload_actors([0]):
                self.assertEqual(events, ["cpu"])
                raise RuntimeError("phase failed")
        self.assertEqual(events, ["cpu", "cuda:0"])

    def test_chunking_records_suffix_not_prefill_as_completion(self):
        trainer = bare_trainer()
        trainer.preference_generation_batch_size = 2
        trainer._preference_trajectories = {}
        trainer.preference_joint_candidates = 0
        calls = []

        def generate(self, agent, batch_items, *, agent_idx, num_return_sequences, **kwargs):
            calls.append(num_return_sequences)
            prefix = build_agent_json_prefill(agent_idx, 3)
            n = num_return_sequences
            return {"prompts": ["context" + prefix], "prompt_input_ids": [torch.tensor([2, 3])],
                    "completions": [[prefix + 'x"}]}' for _ in range(n)]],
                    "completion_input_ids": [[torch.tensor([5, 6]) for _ in range(n)]],
                    "completion_attention_mask": [[torch.ones(2) for _ in range(n)]],
                    "completion_loss_mask": [[torch.tensor([True, False]) for _ in range(n)]],
                    "response_lens": [2] * n, "reference_kls": [0.] * n}

        with patch.object(StructuredOutputMAGRPOTrainer, "_generate_completions", generate):
            result = trainer._generate_completions(trainer.agents[0], [{}], num_return_sequences=5)
        self.assertEqual(calls, [2, 2, 1])
        self.assertEqual(len(result["completions"][0]), 5)
        self.assertEqual(trainer.preference_joint_candidates, 5)
        text = result["completions"][0][0]
        data = trainer._preference_tensors_from_text(0, result["prompts"][0], text, text)
        self.assertEqual(data.winner_completion_ids.tolist(), [5, 6])
        self.assertEqual(data.winner_loss_mask.tolist(), [True, False])

    def test_reward_context_consistent_and_does_not_cut_embedded_json(self):
        trainer = TravelMARLHFTrainer.__new__(TravelMARLHFTrainer)
        prompts = ['role example {"agent_id": 0} then full reference', "other role"]
        actions = ["complete action 0", "complete action 1"]
        prefixed = [p + build_agent_json_prefill(i, 5) for i, p in enumerate(prompts)]
        self.assertEqual(trainer._format_joint_text(prompts, actions), trainer._format_joint_text(prefixed, actions))
        result = json.loads(trainer._format_joint_text(prefixed, actions))
        self.assertEqual(result["role_contexts"], prompts)
        self.assertEqual(result["joint_actions"], actions)

    def test_reward_model_scores_each_joint_action_without_truncation(self):
        trainer = TravelMARLHFTrainer.__new__(TravelMARLHFTrainer)
        trainer.reward_model = TravelJointRewardModel(GPT2Model(tiny_config()))
        trainer.reward_device = torch.device("cpu")
        trainer.args = SimpleNamespace(reward_max_length=6)
        calls = []

        def tokenizer(text, **kwargs):
            calls.append(kwargs)
            return {"input_ids": torch.full((1, len(text)), 3), "attention_mask": torch.ones((1, len(text)))}

        trainer.reward_tokenizer = tokenizer
        result = trainer._score_reward_texts(["abc", "defg"])
        self.assertEqual(tuple(result.shape), (2,))
        self.assertTrue(all(call["truncation"] is False for call in calls))
        result.sum().backward()
        self.assertIsNotNone(trainer.reward_model.reward_head.weight.grad)
        with self.assertRaisesRegex(ValueError, "instead of truncating"):
            trainer._score_reward_texts(["toolong"])

    def test_default_configs_keep_task_reward_and_budget_explicit(self):
        reference = load_config(ROOT / "single_turn/configs/single_turn_magrpo_config.yaml")
        for name in ("marlhf", "marlhf_iter", "madpo_iter"):
            config = load_config(ROOT / f"single_turn/configs/single_turn_{name}_config.yaml")
            args = make_trainer_args(config)
            self.assertEqual(config.get_section("travel_reward"), reference.get_section("travel_reward"))
            self.assertEqual(config.get_section("dataset"), reference.get_section("dataset"))
            self.assertEqual(args.agent_devices, ["cuda:0", "cuda:0"])
            if name.endswith("_iter"):
                self.assertTrue(args.log_reward_distribution)
            report = budget_report(config, args, 60)
            self.assertEqual(report.get("online_rl_joint_rollouts", report.get("pair_counted_env_steps_upper_bound")), 5040)
            config.update({"mapl": {"parallel_training": "mp"}})
            with self.assertRaisesRegex(ValueError, "parallel_training"):
                make_trainer_args(config)


class PreferenceLoggingTests(unittest.TestCase):
    def make_trainer(self, directory, *, enabled=True):
        trainer = bare_trainer()
        trainer.env_step = 0
        trainer.args.num_iterations = 2
        trainer.args.log_reward_distribution = enabled
        trainer.args.preference_replay_dir = str(Path(directory) / "preference_replay")
        trainer.wandb_config = None
        trainer.wandb_initialized = True
        trainer.verbose = False
        trainer.reward_func = make_reward()
        trainer._preference_replay_shards_state = [SimpleNamespace(num_pairs=2)]
        trainer._reset_iteration_reward_distribution()
        return trainer

    def test_iter_distributions_and_eval_can_log_at_same_env_step(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer._record_iteration_reward_distribution(
                target_rewards=[-.25, .5, 1.], comparator_rewards=[.08, .25, .75])
            selected = pair()
            selected.target_raw_reward = .5
            selected.comparator_raw_reward = .25
            fake = SimpleNamespace(run=object(), log=Mock(), define_metric=Mock())
            with patch("single_turn.train.preference_trainer.wandb", fake), \
                 patch.object(trainer, "_reward_distribution_line_image", return_value="line-plot") as line_plot, \
                 patch.object(trainer, "_reward_distribution_bar_image", return_value="bar-plot"):
                trainer._log_wandb_eval_metrics({"eval/reward": .4})
                trainer._log_iteration_replay(0, train_pairs=[selected], current_pair_count=1, train_pair_count=1)
                trainer._log_wandb_eval_metrics({
                    "eval/reward": .5, "train/loss": 7., "eval_full/reward": .7,
                    "eval/samples": "must not be uploaded", "turn_1/reward_mean": .5,
                })
            self.assertEqual(fake.log.call_count, 3)
            for call in fake.log.call_args_list:
                self.assertNotIn("step", call.kwargs)
                self.assertTrue(call.kwargs["commit"])
                self.assertEqual(call.args[0]["env_step"], 0)
            metrics = fake.log.call_args_list[1].args[0]
            self.assertEqual(metrics["iter/current_iteration"], 1)
            self.assertEqual(metrics["iter/total_preference_pairs"], 2)
            self.assertEqual(metrics["iter/reward_distribution/target_sample_count"], 3)
            self.assertEqual(metrics["iter/selected_reward_distribution/pair_count"], 1)
            self.assertIn("iter/reward_distribution/iteration_0001/line_plot", metrics)
            self.assertIn("iter/selected_reward_distribution/iteration_0001/bar_plot", metrics)
            self.assertEqual(fake.log.call_args_list[2].args[0], {"env_step": 0, "eval/reward": .5})
            fake.define_metric.assert_any_call("eval/*", step_metric="env_step")
            fake.define_metric.assert_any_call("iter/*", step_metric="iter/current_iteration")
            self.assertEqual(fake.define_metric.call_count, 4)
            for call in line_plot.call_args_list:
                edges = call.kwargs["edges"]
                self.assertEqual((edges[0], edges[-1], len(edges)), (-.25, 1., 17))
            record = json.loads((Path(directory) / "reward_distributions/iteration_0001.json").read_text())
            self.assertEqual((record["reward_min"], record["reward_max"], record["num_bins"]), (-.25, 1., 16))

    def test_distribution_disabled_keeps_only_iteration_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, enabled=False)
            fake = SimpleNamespace(run=object(), log=Mock(), define_metric=Mock())
            with patch("single_turn.train.preference_trainer.wandb", fake):
                trainer._log_iteration_replay(0, train_pairs=[], current_pair_count=0, train_pair_count=0)
            self.assertEqual(set(fake.log.call_args.args[0]), {
                "env_step", "iter/current_iteration", "iter/current_preference_pairs",
                "iter/total_preference_pairs", "iter/train_preference_pairs"})
            self.assertFalse((Path(directory) / "reward_distributions").exists())

    def test_distribution_uses_declared_raw_range_not_processed_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.reward_func = make_reward({"min_reward": -.5, "max_reward": 2.})
            trainer.reward_processor = lambda reward: reward * 10 + 7
            good, bad = valid_completions(), all_dash_completions()
            raw, processed = trainer._compute_raw_and_processed_rewards(
                ["prompt"], [[good[0], bad[0]], [good[1], bad[1]]], batch_items=[fixture_item()])
            self.assertEqual(processed, [value * 10 + 7 for value in raw])
            trainer._record_iteration_reward_distribution(target_rewards=raw, comparator_rewards=raw)
            self.assertEqual(trainer._iteration_reward_distribution["target"], raw)
            edges = trainer._reward_distribution_bin_edges()
            self.assertEqual((edges[0], edges[-1]), (-.5, 2.))


class PreferenceLoopTests(unittest.TestCase):
    def make_trainer(self, cls, config_cls, directory):
        options = dict(num_agents=2, agent_devices=["cpu", "cpu"], num_train_epochs=1,
                       eval_interval=0, eval_num_samples=1, preference_num_candidates=2,
                       preference_pairs_per_sample=1, max_new_tokens=1024,
                       train_batch_size=1, rollout_buffer_size=1, advantage_normalization=False)
        if cls != TravelMARLHFTrainer:
            options.update(num_iterations=2, comparator_policy="current_copy", comparator_devices=["cpu", "cpu"],
                           preference_replay_dir=str(Path(directory) / "preference_replay"),
                           log_reward_distribution=True,
                           preference_replay_mode="lambda_decay", preference_replay_lambda=.8)
        if cls != TravelMADPOIterTrainer:
            options.update(num_generations=2, reward_model_device="cpu", reward_num_train_epochs=1)
        trainer = cls(agents=[GPT2LMHeadModel(tiny_config()), GPT2LMHeadModel(tiny_config())],
                      num_agents=2, tokenizer=_TinyGenerationTokenizer(),
                      reward_func=make_reward(), train_dataset=[fixture_item()], eval_dataset=[fixture_item()],
                      formatters=[lambda item: "role0", lambda item: "role1"],
                      args=config_cls(**options), chat_formatted_prompts=True)
        trainer.verbose = False
        return trainer

    def test_real_structured_generation_produces_cacheable_masks(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(TravelMADPOIterTrainer, MADPOIterConfig, directory)
            trainer.max_value_tokens = 2
            trainer._preference_trajectories = {}
            for idx in range(2):
                generated = trainer._generate_completions(
                    trainer.agents[idx], [fixture_item()], agent_idx=idx,
                    num_return_sequences=1, max_new_tokens=2048,
                )
                full = generated["completions"][0][0]
                self.assertEqual(json.loads(full)["agent_id"], idx)
                data = trainer._preference_tensors_from_text(idx, generated["prompts"][0], full, full)
                self.assertEqual(len(data.winner_completion_ids), len(data.winner_loss_mask))
                self.assertGreater(data.winner_loss_mask.sum().item(), 0)
                self.assertLess(data.winner_loss_mask.sum().item(), len(data.winner_loss_mask))
                suffix = trainer.tokenizers[idx].decode(data.winner_completion_ids, skip_special_tokens=True)
                self.assertEqual(build_agent_json_prefill(idx, 3) + suffix, full)

    @staticmethod
    def scripted_generation(self, agent, batch_items, *, agent_idx, num_return_sequences, **kwargs):
        # Exercise actual CoMLRL candidate pairing/replay/training with the task
        # rewards and real tiny actor gradients, without a remote model download.
        idx = getattr(agent, "test_candidate_index", 0)
        agent.test_candidate_index = idx + 1
        # Vary comparator vs target and candidates; both roles see same cadence.
        good = (idx % 4) in (0, 3)
        full = (valid_completions() if good else all_dash_completions())[agent_idx]
        prefix = build_agent_json_prefill(agent_idx, 3)
        assert full.startswith(prefix)
        ids = torch.tensor([ord(char) for char in full[len(prefix):]])
        n = num_return_sequences
        return {"prompts": [f"role{agent_idx}" + prefix], "prompt_input_ids": torch.tensor([[2, 3]]),
                "prompt_attention_mask": torch.ones(1, 2), "batch_items": batch_items,
                "completions": [[full] * n], "completion_input_ids": [[ids.clone() for _ in range(n)]],
                "completion_attention_mask": [[torch.ones_like(ids) for _ in range(n)]],
                "completion_loss_mask": [[torch.ones_like(ids, dtype=torch.bool) for _ in range(n)]],
                "response_lens": [len(ids)] * n, "reference_kls": [0.] * n}

    @staticmethod
    def tiny_reward_init(self):
        self.reward_model = TravelJointRewardModel(GPT2Model(tiny_config()))
        self.reward_optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=.01)

        def tokenize(text, **kwargs):
            # Compact fake tokenizer for the RM plumbing smoke, not a truncation
            # strategy used in production. Both good/bad actions affect IDs.
            ids = torch.tensor([[2, 3, 4 if "Hotel" in text else 5]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        self.reward_tokenizer = tokenize

    def test_all_three_real_loops_with_tiny_actors(self):
        torch.set_num_threads(1)
        for cls, config_cls in ((TravelMADPOIterTrainer, MADPOIterConfig),
                                (TravelMARLHFTrainer, MARLHFConfig),
                                (TravelMARLHFIterTrainer, MARLHFIterConfig)):
            with self.subTest(algorithm=cls.__name__), tempfile.TemporaryDirectory() as directory:
                trainer = self.make_trainer(cls, config_cls, directory)
                before = [copy.deepcopy(actor.state_dict()) for actor in trainer.agents]
                with patch.object(StructuredOutputMAGRPOTrainer, "_generate_completions", self.scripted_generation), \
                     patch("single_turn.train.preference_trainer.TravelRewardModelMixin._init_reward_model", self.tiny_reward_init):
                    trainer.train()
                self.assertGreater(trainer.preference_pairs_generated, 0)
                self.assertGreater(trainer.env_step, 0)
                self.assertFalse(trainer._should_log_train(trainer.env_step))
                if cls != TravelMARLHFTrainer:
                    self.assertEqual(len(list((Path(directory) / "reward_distributions").glob("iteration_*.json"))), 2)
                for idx, actor in enumerate(trainer.agents):
                    self.assertTrue(any(not torch.equal(value, before[idx][key]) for key, value in actor.state_dict().items()))
                if cls != TravelMADPOIterTrainer:
                    # Even when the learned RM is active, benchmark eval must
                    # always route through the domain reward, then restore flags.
                    trainer._reward_model_active = True
                    flags = []
                    def inspect_eval(self, **kwargs):
                        flags.append(self._evaluating_with_task_reward)
                        return {}
                    with patch.object(MAGRPOTrainer, "evaluate", inspect_eval):
                        trainer.evaluate(num_eval_samples=1)
                    self.assertEqual(flags, [True])
                    self.assertFalse(trainer._evaluating_with_task_reward)


if __name__ == "__main__":
    unittest.main()
