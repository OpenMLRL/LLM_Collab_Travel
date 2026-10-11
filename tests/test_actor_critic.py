"""CPU integration checks for the two-role Travel actor-critic adapters."""

import copy
from contextlib import redirect_stdout
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

from comlrl.trainers.actor_critic import IACConfig, MAACConfig
from single_turn.config import Config
from single_turn.data import partition_rows
from single_turn.evaluation import evaluate_single_turn_response
from single_turn.formatting import build_agent_json_prefill
from single_turn.rewards import make_reward
from single_turn.rewards.single_turn_reward import score_single_turn_response
from single_turn.train.actor_critic_trainer import TravelIACTrainer, TravelMAACTrainer
from single_turn.train.train_actor_critic import make_trainer_args
from single_turn.train.train_centralized import load_config
from single_turn.train.train_magrpo import _load_data
from test_preference_training import tiny_config
from test_single_turn import fixture_item, valid_completions, all_dash_completions
from test_structured_generation import _TinyGenerationTokenizer


class ACTokenizer(_TinyGenerationTokenizer):
    def save_pretrained(self, directory):
        Path(directory, "tokenizer_config.json").write_text("{}")

    def __call__(self, prompt, **kwargs):
        assert kwargs.get("truncation") is False
        # Keep all prompt bytes so the forced assistant prefix is represented.
        ids = torch.tensor([self.encode(prompt, add_special_tokens=False)])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def make_trainer(algorithm, *, scripted=True, rows=None, generations=2):
    torch.manual_seed(42)
    cls, args_cls = (
        (TravelIACTrainer, IACConfig)
        if algorithm == "iac"
        else (TravelMAACTrainer, MAACConfig)
    )
    settings = dict(
        num_agents=2,
        num_turns=1,
        agent_devices="cpu",
        critic_devices="cpu",
        num_generations=generations,
        num_train_epochs=1,
        rollout_buffer_size=2,
        train_batch_size=2,
        max_new_tokens=1024,
        eval_interval=0,
        eval_num_samples=1,
        advantage_normalization=False,
        agent_learning_rate=1e-3,
        critic_learning_rate=1e-3,
    )
    if algorithm == "iac":
        settings["use_separate_critic"] = False
    models = [GPT2LMHeadModel(tiny_config()) for _ in range(2)]
    trainer = cls(
        agents=models,
        critics=[GPT2LMHeadModel(tiny_config())] if algorithm == "maac" else None,
        tokenizer=[ACTokenizer(), ACTokenizer()],
        args=args_cls(**settings),
        reward_func=make_reward(),
        formatters=[lambda item: "role zero\n", lambda item: "role one\n"],
        train_dataset=rows or [fixture_item()],
        eval_dataset=rows or [fixture_item()],
        max_value_tokens=64 if scripted else 1,
    )
    trainer.verbose = False
    if scripted:
        for role, actor in enumerate(trainer.agents):

            def generate(input_ids, num_return_sequences, _role=role, **kwargs):
                prefix = build_agent_json_prefill(_role, 3)
                choices = [valid_completions(), all_dash_completions()]
                tails = []
                for i in range(num_return_sequences):
                    text = choices[i % 2][_role]
                    assert text.startswith(prefix)
                    tails.append(torch.tensor([ord(c) for c in text[len(prefix) :]]))
                width = max(len(tail) for tail in tails)
                result = input_ids.expand(num_return_sequences, -1).clone()
                padded = torch.full((num_return_sequences, width), 1)
                for i, tail in enumerate(tails):
                    padded[i, : len(tail)] = tail
                return torch.cat([result, padded], dim=1)

            actor.model.generate = generate
    return trainer


class TravelActorCriticTests(unittest.TestCase):
    def test_reward_masks_logprob_and_real_parameter_updates(self):
        for algorithm in ("iac", "maac"):
            with self.subTest(algorithm=algorithm):
                trainer = make_trainer(algorithm)
                item = fixture_item()
                samples = trainer._collect_rollouts(item)
                self.assertEqual(len(samples), 4)
                for sample in samples:
                    expected_outputs = [valid_completions(), all_dash_completions()][
                        sample.metadata["generation_idx"]
                    ]
                    self.assertEqual(
                        sample.completion, expected_outputs[sample.agent_idx]
                    )
                    reward, _ = score_single_turn_response(
                        expected_outputs, batch_item=item
                    )
                    self.assertAlmostEqual(sample.reward.item(), reward, places=6)
                    mask = sample.metadata["completion_loss_mask"]
                    self.assertEqual(mask.shape, (sample.response_len,))
                    self.assertTrue(mask.any())
                    self.assertTrue((~mask).any())
                    actor = trainer.agents[sample.agent_idx]
                    ids = sample.full_input_ids[None]
                    logits = actor(
                        ids,
                        attention_mask=sample.attention_mask[None],
                        output_values=False,
                    ).logits
                    start = sample.prompt_len - 1
                    targets = ids[
                        :, sample.prompt_len : sample.prompt_len + sample.response_len
                    ]
                    logprobs = (
                        F.log_softmax(
                            logits[:, start : start + sample.response_len], dim=-1
                        )
                        .gather(-1, targets[..., None])
                        .squeeze(-1)
                    )
                    manual = logprobs[:, mask].mean(-1)
                    actual, _ = trainer._policy_eval(
                        actor,
                        ids,
                        sample.attention_mask[None],
                        sample.prompt_len,
                        sample.response_len,
                        output_values=False,
                        completion_loss_mask=mask,
                    )
                    torch.testing.assert_close(actual, manual)
                    torch.testing.assert_close(
                        sample.old_logprob, manual.detach().squeeze(0)
                    )
                    actor.zero_grad()
                    actual.sum().backward(retain_graph=True)
                    actual_grad = actor.model.transformer.wte.weight.grad.clone()
                    actor.zero_grad()
                    manual.sum().backward()
                    torch.testing.assert_close(
                        actor.model.transformer.wte.weight.grad, actual_grad
                    )
                self.assertFalse(trainer._travel_reward.drain_details())
                models = [*trainer.agents, *trainer.critics]
                before = [
                    {
                        name: parameter.detach().clone()
                        for name, parameter in model.named_parameters()
                    }
                    for model in models
                ]
                trainer.train()
                self.assertEqual(trainer.env_step, 2)
                for model, previous in zip(models, before):
                    self.assertTrue(
                        any(
                            not torch.equal(parameter, previous[name])
                            for name, parameter in model.named_parameters()
                        )
                    )
                    self.assertTrue(
                        all(
                            torch.isfinite(parameter).all()
                            for parameter in model.parameters()
                        )
                    )
                if algorithm == "iac":
                    self.assertEqual(len(trainer.critics), 0)
                    self.assertTrue(
                        all(actor.value_head is not None for actor in trainer.agents)
                    )
                else:
                    self.assertEqual(len(trainer.critics), 1)
                    self.assertTrue(
                        all(actor.value_head is None for actor in trainer.agents)
                    )
                self.assertTrue(all(not buffer for buffer in trainer.rollout_buffers))

    def test_eval_uses_joint_domain_metrics_without_training_or_logging_full_pool(self):
        for algorithm in ("iac", "maac"):
            rows = [fixture_item(), fixture_item()]
            rows[1]["budget"] = 1
            trainer = make_trainer(algorithm, rows=rows)
            before = [
                p.detach().clone() for m in trainer.agents for p in m.parameters()
            ]
            with patch.object(trainer, "_log_metrics") as logger:
                metrics = trainer.evaluate(num_eval_samples=2)
            expected = evaluate_single_turn_response(
                valid_completions(), batch_item=rows[0]
            )
            self.assertEqual(
                metrics["eval/reference_plan_success"],
                expected["ultimate/reference_plan_success"],
            )
            self.assertLess(
                metrics["eval_full/reference_plan_success"],
                metrics["eval/reference_plan_success"],
            )
            self.assertFalse(
                any(key.startswith("eval_full/") for key in logger.call_args.args[0])
            )
            self.assertEqual(trainer.env_step, 0)
            self.assertFalse(trainer._in_eval)
            for parameter, old in zip(
                (p for m in trainer.agents for p in m.parameters()), before
            ):
                self.assertTrue(torch.equal(parameter, old))

    def test_real_cpu_generation_closes_schema_and_retains_masks(self):
        item = fixture_item()
        item["days"] = 1
        trainer = make_trainer("iac", scripted=False, rows=[item], generations=1)
        trainer._in_eval = True
        samples = trainer._collect_rollouts(item)
        self.assertEqual(len(samples), 2)
        detail = evaluate_single_turn_response(
            [sample.completion for sample in samples], batch_item=item
        )
        self.assertEqual(detail["action_validity"], 1.0)
        for sample in samples:
            self.assertTrue(sample.completion.endswith("]}"))
            self.assertEqual(
                len(sample.metadata["completion_loss_mask"]), sample.response_len
            )
            self.assertFalse(sample.full_input_ids.is_inference())
            self.assertTrue(trainer.agents[sample.agent_idx].training)

    def test_configs_preserve_task_and_budget_and_reject_protocol_changes(self):
        root = Path(__file__).resolve().parents[1] / "single_turn/configs"
        for algorithm in ("iac", "maac"):
            config = load_config(root / f"single_turn_{algorithm}_config.yaml")
            args = make_trainer_args(config)
            self.assertEqual(
                args.num_train_epochs
                * args.num_generations
                * config.get("dataset.train_samples"),
                5040,
            )
            self.assertEqual(config.get("dataset.eval_samples"), 20)
            self.assertEqual(
                config.get("agent_model.name"), "Qwen/Qwen3-4B-Instruct-2507"
            )
            config.update({algorithm: {"temperature": 0.5}})
            self.assertEqual(make_trainer_args(config).temperature, 0.5)
            for change in (
                {"travel": {"normalize_value_log_probs": False}},
                {algorithm: {"num_agents": 1}},
                {algorithm: {"reference_kl_enabled": True}},
                {algorithm: {"critic_type": "q"}},
            ):
                broken = copy.deepcopy(config)
                broken.update(change)
                with self.assertRaises(ValueError):
                    make_trainer_args(broken)

    def test_final_checkpoint_retains_both_actors_and_value_heads(self):
        for algorithm in ("iac", "maac"):
            trainer = make_trainer(algorithm)
            with tempfile.TemporaryDirectory() as directory:
                trainer.save_model(directory)
                root = Path(directory)
                for role in range(2):
                    actor_path = root / f"agent_{role}"
                    saved = GPT2LMHeadModel.from_pretrained(
                        actor_path, local_files_only=True
                    )
                    for original, restored in zip(
                        trainer.agents[role].model.parameters(), saved.parameters()
                    ):
                        torch.testing.assert_close(original, restored)
                    self.assertTrue((actor_path / "tokenizer_config.json").exists())
                head_paths = (
                    [root / "agent_0/value_head.pt", root / "agent_1/value_head.pt"]
                    if algorithm == "iac"
                    else [root / "critic/value_head.pt"]
                )
                self.assertTrue(all(path.is_file() for path in head_paths))

    def test_cli_dry_run_never_loads_model_and_reports_fixed_split_seed(self):
        from single_turn.train.train_actor_critic import main

        for algorithm in ("iac", "maac"):
            output = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "train",
                    "--algorithm",
                    algorithm,
                    "--dry-run",
                    "--override",
                    "seed=123",
                    "dataset.seed=42",
                ],
            ), patch(
                "single_turn.train.train_actor_critic._load_data",
                return_value=([fixture_item()] * 60, [fixture_item()] * 20),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                side_effect=AssertionError("dry run loaded a tokenizer"),
            ), redirect_stdout(
                output
            ):
                main()
            report = json.loads(output.getvalue())
            self.assertEqual(report["dataset_seed"], 42)
            self.assertEqual(report["training_seed"], 123)
            self.assertEqual(report["online_joint_rollouts"], 5040)
            self.assertEqual(report["actor_models"], 2)

    def test_dataset_seed_is_independent_with_legacy_fallback(self):
        config = Config.__new__(Config)
        config.data = {
            "seed": 42,
            "dataset": {"name": "fixture", "train_samples": 4, "eval_samples": 4},
        }
        rows = [{"id": str(index), "days": 3, "level": "easy"} for index in range(8)]

        def load(_name, *, seed, train_samples, eval_samples, **kwargs):
            return partition_rows(
                rows, train_samples=train_samples, eval_samples=eval_samples, seed=seed
            )

        with patch(
            "single_turn.train.train_magrpo.load_single_turn_datasets", side_effect=load
        ):
            legacy = _load_data(config)
            config.update({"seed": 123, "dataset": {"seed": 42}})
            self.assertEqual(_load_data(config), legacy)
            del config.data["dataset"]["seed"]
            self.assertNotEqual(_load_data(config), legacy)


if __name__ == "__main__":
    unittest.main()
