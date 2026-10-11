from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from single_turn.aggregation import merge_agent_assignments, ordered_owned_slots
from single_turn.centralized_comparator import (
    TravelCentralizedComparatorAdapter,
    joint_slots,
    project_role_tokens,
)
from single_turn.formatting import build_agent_json_prefill
from single_turn.train.structured_trainer import StructuredOutputMAGRPOTrainer
from single_turn.train.train_mapl import load_config, make_trainer_args
from test_preference_training import ROOT, bare_trainer
from test_single_turn import fixture_item, valid_completions
from test_structured_generation import _TinyGenerationTokenizer
from test_structured_generation import FakeTokenizer
from comlrl.trainers.reinforce import MAGRPOTrainer


def joint_text(days=3, value="-"):
    return json.dumps(
        {
            "agent_id": 0,
            "assignments": [
                {"day": day, "field": field, "value": value}
                for day, field in joint_slots(days)
            ],
        }
    )


class CentralizedTravelTests(unittest.TestCase):
    def test_structured_joint_override_fills_all_roles_with_value_only_masks(self):
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer.chat_formatted_prompts = True
        trainer.force_json_prefix = trainer.constrain_json_skeleton = True
        trainer.max_value_tokens = 8
        trainer.stop_after_complete_json = True
        trainer._travel_eval_generation = False
        trainer.tokenizers = [FakeTokenizer()]
        trainer.formatters = [lambda item: "joint"]
        full = joint_text(days=1)
        prefix = build_agent_json_prefill(0, 1)
        suffix = full[len(prefix) :]
        tokens = torch.tensor([ord(c) for c in suffix])
        with patch.object(
            MAGRPOTrainer,
            "_generate_completions",
            return_value={
                "prompt_input_ids": torch.tensor([[2, 3]]),
                "completion_input_ids": [[tokens]],
                "completion_attention_mask": [[torch.ones(len(tokens))]],
                "completions": [[suffix]],
                "response_lens": [len(tokens)],
            },
        ) as generate:
            result = trainer._generate_completions(
                object(),
                [{"days": 1}],
                max_new_tokens=1024,
                slot_order_override=joint_slots(1),
            )
        self.assertEqual(result["completions"], [[full]])
        self.assertEqual(int(result["completion_loss_mask"][0][0].sum()), 14)
        self.assertEqual(
            generate.call_args.kwargs["logits_processor"][-1].slots,
            tuple(joint_slots(1)),
        )
        self.assertNotIn("slot_order_override", generate.call_args.kwargs)

    def test_split_preserves_every_value_and_existing_merge(self):
        item = fixture_item()
        roles = [json.loads(text) for text in valid_completions()]
        values = {
            (r["day"], r["field"]): r["value"]
            for role in roles
            for r in role["assignments"]
        }
        source = json.dumps(
            {
                "agent_id": 0,
                "assignments": [
                    {"day": d, "field": f, "value": values[d, f]}
                    for d, f in joint_slots(3)
                ],
            }
        )
        split = TravelCentralizedComparatorAdapter().parse_completion(source, item, 2)
        self.assertEqual([json.loads(s) for s in split], roles)
        self.assertEqual(sum(len(json.loads(s)["assignments"]) for s in split), 21)

    def test_joint_parser_rejects_missing_duplicate_wrong_role_and_nonstring(self):
        adapter = TravelCentralizedComparatorAdapter()
        for mutation in ("missing", "duplicate", "carrier", "value"):
            data = json.loads(joint_text())
            if mutation == "missing":
                data["assignments"].pop()
            if mutation == "duplicate":
                data["assignments"][1] = data["assignments"][0]
            if mutation == "carrier":
                data["agent_id"] = 1
            if mutation == "value":
                data["assignments"][0]["value"] = 10
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                adapter.parse_completion(json.dumps(data), {"days": 3}, 2)

    def test_prompt_excludes_annotations_and_exposes_both_roles(self):
        item = fixture_item()
        item["annotated_plan"] = "GOLD_SENTINEL_123"
        item["gold_plan"] = "GOLD_SENTINEL_456"
        prompt = TravelCentralizedComparatorAdapter().build_prompt(item, [])
        self.assertNotIn("GOLD_SENTINEL", prompt)
        self.assertIn("REFERENCE ROLE 0", prompt)
        self.assertIn("REFERENCE ROLE 1", prompt)
        self.assertIn("ONE centralized", prompt)
        self.assertIn("exactly 21 assignments", prompt)

    def test_projected_masks_cover_values_but_not_forced_schema_or_prefill(self):
        tokenizer = _TinyGenerationTokenizer()
        for value in ("Cafe, Beta", 'quoted "x"', "line\nbreak", "", "-", "旅途"):
            split = TravelCentralizedComparatorAdapter().parse_completion(
                joint_text(value=value), {"days": 3}, 2
            )
            for i, role in enumerate(split):
                text, tokens, mask = project_role_tokens(tokenizer, role, i, 3)
                self.assertEqual(json.loads(text), json.loads(role))
                suffix = tokenizer.decode(tokens, skip_special_tokens=True)
                self.assertEqual(build_agent_json_prefill(i, 3) + suffix, text)
                self.assertFalse(mask[-1])
                policy = tokenizer.decode(tokens[mask], skip_special_tokens=True)
                self.assertEqual(
                    policy,
                    json.dumps(value, ensure_ascii=False)[1:]
                    * len(ordered_owned_slots(i, 3)),
                )
                self.assertEqual(mask.numel(), tokens.numel())

    def test_central_config_supported_but_only_policy_index_zero(self):
        for algorithm in ("madpo_iter", "marlhf_iter"):
            config = load_config(
                ROOT / f"single_turn/configs/single_turn_{algorithm}_config.yaml"
            )
            config.update({"mapl": {"comparator_generation_mode": "centralized"}})
            self.assertEqual(
                make_trainer_args(config).comparator_generation_mode, "centralized"
            )
            config.update({"mapl": {"comparator_centralized_agent_index": 1}})
            with self.assertRaisesRegex(ValueError, "index 0"):
                make_trainer_args(config)

    def test_joint_generation_is_one_teacher_call_per_chunk_then_role_projection(self):
        trainer = bare_trainer()
        trainer.args.max_new_tokens = 1024
        trainer.args.comparator_centralized_agent_index = 0
        trainer.agents[0].config = SimpleNamespace(max_position_embeddings=32768)
        trainer.tokenizers = [_TinyGenerationTokenizer(), _TinyGenerationTokenizer()]
        trainer.formatters = [lambda item: "role0", lambda item: "role1"]
        trainer.centralized_comparator_adapter = TravelCentralizedComparatorAdapter()
        trainer.preference_generation_batch_size = 2
        trainer.preference_joint_candidates = 20
        trainer._preference_trajectories = {}
        calls = []

        def generate(self, agent, items, **kwargs):
            calls.append(kwargs)
            return {"completions": [[joint_text()] * kwargs["num_return_sequences"]]}

        with patch.object(
            trainer, "_offload_actors", return_value=nullcontext()
        ) as offload, patch.object(
            trainer, "_comparator_rng", return_value=nullcontext()
        ), patch.object(
            StructuredOutputMAGRPOTrainer, "_generate_completions", generate
        ):
            outputs = trainer._generate_centralized_policy_output_for_agent(
                trainer.agents[0], {"days": 3}, prompt="joint", num_candidates=3
            )
        offload.assert_called_once_with([1])
        self.assertEqual([r["num_return_sequences"] for r in calls], [2, 1])
        self.assertTrue(
            all(
                r["max_new_tokens"] == 2048
                and r["slot_order_override"] == joint_slots(3)
                for r in calls
            )
        )
        self.assertEqual(trainer.preference_joint_candidates, 23)
        self.assertEqual([len(r["completions"][0]) for r in outputs], [3, 3])
        for i in range(2):
            prompt = outputs[i]["prompts"][0]
            self.assertEqual(prompt, f"role{i}" + build_agent_json_prefill(i, 3))
            text = outputs[i]["completions"][0][0]
            tensors = trainer._preference_tensors_from_text(i, prompt, text, text)
            self.assertEqual(tensors.prompt_input_ids.tolist(), [2, 3])
            self.assertTrue(tensors.winner_loss_mask.any())

    def test_context_overflow_fails_before_generation(self):
        trainer = bare_trainer()
        trainer.args.max_new_tokens = 1024
        trainer.args.comparator_centralized_agent_index = 0
        trainer.tokenizers = [_TinyGenerationTokenizer()] * 2
        trainer.agents[0].config = SimpleNamespace(max_position_embeddings=100)
        with self.assertRaisesRegex(ValueError, "refusing truncation"):
            trainer._generate_centralized_policy_output_for_agent(
                trainer.agents[0], {"days": 3}, prompt="joint", num_candidates=20
            )

    def test_central_history_loaded_once_per_collection_and_released(self):
        trainer = bare_trainer()
        trainer.args.comparator_policy = "history"
        trainer._history_policy_checkpoint_path = Mock(return_value="initial")
        teacher = object()
        trainer._load_single_policy_checkpoint_agent = Mock(return_value=teacher)
        trainer._build_centralized_comparator_prompt = Mock(return_value="joint")
        trainer._generate_centralized_policy_output_for_agent = Mock(
            return_value="outputs"
        )
        trainer._clear_transient_comparator_agents = Mock()
        for item in ({"days": 3}, {"days": 5}):
            self.assertEqual(
                trainer._generate_centralized_comparator_outputs_for_item(
                    item, iteration_idx=0, num_candidates=20
                ),
                "outputs",
            )
        trainer._load_single_policy_checkpoint_agent.assert_called_once_with(
            "initial", 0
        )
        trainer._clear_travel_central_history()
        trainer._clear_transient_comparator_agents.assert_called_once_with([teacher])
        self.assertIsNone(trainer._travel_central_history_agent)


if __name__ == "__main__":
    unittest.main()
