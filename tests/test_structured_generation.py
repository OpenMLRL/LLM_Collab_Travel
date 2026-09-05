from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LogitsProcessor,
    LogitsProcessorList,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMLRL_ROOT = REPO_ROOT.parent / "CoMLRL"
for path in (COMLRL_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from comlrl.trainers.reinforce import MAGRPOTrainer
from single_turn.formatting import build_agent_json_prefill
from single_turn.aggregation import ordered_owned_slots
from single_turn.structured_generation import (
    CompleteJSONObjectCriteria,
    FixedSlotJSONLogitsProcessor,
    GreedyArgmaxLogitsProcessor,
    apply_chat_template,
    with_greedy_argmax_processor,
    wrap_formatter_with_chat_template,
)
from single_turn.train.structured_trainer import StructuredOutputMAGRPOTrainer
from single_turn.train.train_magrpo import _curriculum_plan
from single_turn.config import Config


class FakeTokenizer:
    chat_template = "fake-template"
    pad_token_id = 0
    eos_token_id = 0
    all_special_ids = [0]

    def __init__(self):
        self.last_messages = None
        self.last_add_generation_prompt = None
        self.last_tokenizer_kwargs = None

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        self.last_messages = messages
        self.last_add_generation_prompt = add_generation_prompt
        assert tokenize is False
        body = "\n".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        return body + "\n<assistant>"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return "".join(chr(int(token_id)) for token_id in token_ids)

    def __call__(self, prompts, **kwargs):
        self.last_tokenizer_kwargs = kwargs
        width = max(len(prompt) for prompt in prompts)
        return SimpleNamespace(
            input_ids=torch.zeros((len(prompts), width), dtype=torch.long)
        )


class _TinyBatchEncoding:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class _TinyGenerationTokenizer:
    pad_token_id = 1
    eos_token_id = 1
    pad_token = "<eos>"
    eos_token = "<eos>"
    all_special_ids = [0, 1]

    def __call__(self, prompts, **kwargs):
        del kwargs
        input_ids = torch.tensor([[2, 3] for _ in prompts], dtype=torch.long)
        return _TinyBatchEncoding(input_ids, torch.ones_like(input_ids))

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces=False,
    ):
        del clean_up_tokenization_spaces
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
        return "".join(
            chr(int(value))
            for value in values
            if not (skip_special_tokens and int(value) == self.eos_token_id)
        )


class _ForceDifferentJSONRows(LogitsProcessor):
    responses = (
        '-"}]}',
        'Beta"}]}',
        'from A to B"}]}',
        'quoted \\"x\\""}]}',
    )

    def __init__(self, prompt_length):
        self.prompt_length = int(prompt_length)

    def __call__(self, input_ids, scores):
        position = int(input_ids.shape[-1]) - self.prompt_length
        forced = torch.full_like(scores, -torch.inf)
        for row_idx, response in enumerate(self.responses):
            token_id = ord(response[position]) if position < len(response) else 1
            forced[row_idx, token_id] = 0.0
        return forced


class _PositionIndependentLM(torch.nn.Module):
    """Tiny differentiable LM for exact policy-loss assertions."""

    def __init__(self, vocab_size=8):
        super().__init__()
        self.token_logits = torch.nn.Parameter(torch.arange(vocab_size).float())

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        logits = self.token_logits.view(1, 1, -1).expand(
            input_ids.shape[0], input_ids.shape[1], -1
        )
        return SimpleNamespace(logits=logits)


class ChatTemplateTests(unittest.TestCase):
    def test_chat_template_adds_system_user_and_generation_prompt(self):
        tokenizer = FakeTokenizer()
        rendered = apply_chat_template(
            tokenizer,
            "raw travel instruction",
            system_prompt="json only",
        )
        self.assertEqual(
            tokenizer.last_messages,
            [
                {"role": "system", "content": "json only"},
                {"role": "user", "content": "raw travel instruction"},
            ],
        )
        self.assertTrue(tokenizer.last_add_generation_prompt)
        self.assertTrue(rendered.endswith("<assistant>"))

    def test_wrapped_formatter_preserves_external_prompt_call_contract(self):
        tokenizer = FakeTokenizer()

        def formatter(item, external_prompts=None):
            return f"{item['prompt']}::{external_prompts}"

        wrapped = wrap_formatter_with_chat_template(
            formatter,
            tokenizer,
            system_prompt=None,
        )
        rendered = wrapped({"prompt": "base"}, external_prompts="next")
        self.assertIn("<user>base::next", rendered)
        self.assertEqual(tokenizer.last_messages[0]["role"], "user")

    def test_missing_chat_template_fails_loudly(self):
        tokenizer = FakeTokenizer()
        tokenizer.chat_template = None
        with self.assertRaisesRegex(
            ValueError, "requires a tokenizer with a chat_template"
        ):
            apply_chat_template(tokenizer, "prompt")


class JSONGenerationConstraintTests(unittest.TestCase):
    def test_fixed_slot_processor_forces_schema_and_masks_it_from_loss(self):
        tokenizer = FakeTokenizer()
        slots = [(1, "current_city"), (1, "transportation")]
        processor = FixedSlotJSONLogitsProcessor(
            tokenizer,
            prompt_length=2,
            slots=slots,
            max_value_tokens=8,
            max_new_tokens=160,
        )
        generated = []

        def scores():
            input_ids = torch.tensor([[2, 3, *generated]], dtype=torch.long)
            return processor(input_ids, torch.zeros((1, 128)))

        self.assertTrue(torch.isfinite(scores()[0, ord("A")]))
        generated.append(ord("A"))
        self.assertTrue(torch.isfinite(scores()[0, ord('"')]))
        generated.append(ord('"'))

        # This is the exact failed-run boundary. Once the value quote closes,
        # ']' is impossible and the next skeleton token is forced to '}'.
        constrained = scores()
        self.assertEqual(torch.isfinite(constrained).sum().item(), 1)
        self.assertTrue(torch.isfinite(constrained[0, ord("}")]))
        self.assertFalse(torch.isfinite(constrained[0, ord("]")]))

        while True:
            constrained = scores()
            if processor._states[0].slot_index != 0:
                break
            generated.append(int(constrained[0].argmax().item()))
        generated.extend([ord("-"), ord('"')])
        while True:
            constrained = scores()
            if processor._states[0].mode == "complete":
                break
            generated.append(int(constrained[0].argmax().item()))

        generated_tensor = torch.tensor(generated, dtype=torch.long)
        mask = processor.finalize_loss_masks([generated_tensor])[0]
        response = build_agent_json_prefill(0, 1) + tokenizer.decode(
            generated_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        payload = json.loads(response)
        self.assertEqual(
            [(row["day"], row["field"]) for row in payload["assignments"]],
            slots,
        )
        self.assertEqual([row["value"] for row in payload["assignments"]], ["A", "-"])
        self.assertEqual(sum(mask), 4)
        self.assertEqual(len(mask), len(generated))

    def test_fixed_slot_processor_forces_a_close_at_value_token_cap(self):
        tokenizer = FakeTokenizer()
        processor = FixedSlotJSONLogitsProcessor(
            tokenizer,
            prompt_length=2,
            slots=[(1, "current_city")],
            max_value_tokens=1,
            max_new_tokens=16,
        )
        first = processor(
            torch.tensor([[2, 3]], dtype=torch.long), torch.zeros((1, 128))
        )
        self.assertTrue(torch.isfinite(first[0, ord("A")]))
        after_content = processor(
            torch.tensor([[2, 3, ord("A")]], dtype=torch.long),
            torch.zeros((1, 128)),
        )
        self.assertEqual(torch.isfinite(after_content).sum().item(), 1)
        self.assertTrue(torch.isfinite(after_content[0, ord('"')]))

    def test_value_choice_reserves_budget_for_a_worst_case_escape(self):
        tokenizer = FakeTokenizer()
        processor = FixedSlotJSONLogitsProcessor(
            tokenizer,
            prompt_length=2,
            slots=[(1, "current_city")],
            max_value_tokens=32,
            max_new_tokens=6,
        )
        generated = []
        while True:
            constrained = processor(
                torch.tensor([[2, 3, *generated]], dtype=torch.long),
                torch.zeros((1, 128)),
            )
            if processor._states[0].mode == "complete":
                break
            generated.append(int(constrained[0].argmax().item()))

        self.assertLessEqual(len(generated), 6)
        generated_tensor = torch.tensor(generated, dtype=torch.long)
        response = build_agent_json_prefill(0, 1) + tokenizer.decode(
            generated_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self.assertEqual(
            json.loads(response)["assignments"][0]["value"],
            "-",
        )

    def test_fixed_slot_processor_accepts_escaped_catalog_values(self):
        tokenizer = FakeTokenizer()
        for catalog_value in ('Yakima "Cruise-the-Ave"', "Line 1\nLine 2"):
            with self.subTest(catalog_value=catalog_value):
                processor = FixedSlotJSONLogitsProcessor(
                    tokenizer,
                    prompt_length=2,
                    slots=[(1, "current_city")],
                    max_value_tokens=64,
                    max_new_tokens=128,
                )
                generated = []
                # Remove only json.dumps' opening quote. Its escaped contents
                # and closing quote are exactly what the policy must emit.
                value_fragment = json.dumps(
                    catalog_value, ensure_ascii=False
                )[1:]
                for character in value_fragment:
                    constrained = processor(
                        torch.tensor([[2, 3, *generated]], dtype=torch.long),
                        torch.zeros((1, 128)),
                    )
                    self.assertTrue(
                        torch.isfinite(constrained[0, ord(character)]),
                        msg=f"grammar rejected {character!r} in {value_fragment!r}",
                    )
                    generated.append(ord(character))

                while True:
                    constrained = processor(
                        torch.tensor([[2, 3, *generated]], dtype=torch.long),
                        torch.zeros((1, 128)),
                    )
                    if processor._states[0].mode == "complete":
                        break
                    generated.append(int(constrained[0].argmax().item()))

                generated_tensor = torch.tensor(generated, dtype=torch.long)
                response = build_agent_json_prefill(0, 1) + tokenizer.decode(
                    generated_tensor,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                self.assertEqual(
                    json.loads(response)["assignments"][0]["value"],
                    catalog_value,
                )

    def test_forced_close_completes_a_partial_json_escape(self):
        tokenizer = FakeTokenizer()
        expected_values = {
            "\\": "\n",
            "\\u": "\u0000",
            "\\u1": "\u1000",
            "\\u12": "\u1200",
            "\\u123": "\u1230",
        }
        for partial_escape, expected_value in expected_values.items():
            with self.subTest(partial_escape=partial_escape):
                processor = FixedSlotJSONLogitsProcessor(
                    tokenizer,
                    prompt_length=2,
                    slots=[(1, "current_city")],
                    max_value_tokens=len(partial_escape),
                    max_new_tokens=32,
                )
                generated = [ord(character) for character in partial_escape]
                while True:
                    constrained = processor(
                        torch.tensor([[2, 3, *generated]], dtype=torch.long),
                        torch.zeros((1, 128)),
                    )
                    if processor._states[0].mode == "complete":
                        break
                    generated.append(int(constrained[0].argmax().item()))

                generated_tensor = torch.tensor(generated, dtype=torch.long)
                response = build_agent_json_prefill(0, 1) + tokenizer.decode(
                    generated_tensor,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                self.assertEqual(
                    json.loads(response)["assignments"][0]["value"],
                    expected_value,
                )

    def test_greedy_eval_processor_resolves_logit_ties_by_argmax(self):
        processor = GreedyArgmaxLogitsProcessor()
        scores = torch.tensor([[5.0, 5.0, 4.0], [1.0, 2.0, 2.0]])
        constrained = processor(torch.zeros((2, 1), dtype=torch.long), scores)
        self.assertEqual(
            torch.isfinite(constrained).nonzero().tolist(), [[0, 0], [1, 1]]
        )

    def test_greedy_eval_helper_preserves_existing_processors_without_mutation(self):
        existing = LogitsProcessorList([_ForceDifferentJSONRows(2)])
        combined = with_greedy_argmax_processor(existing)
        self.assertEqual(len(existing), 1)
        self.assertEqual(len(combined), 2)
        self.assertIs(combined[0], existing[0])
        self.assertIsInstance(combined[-1], GreedyArgmaxLogitsProcessor)

    def test_eval_flag_is_scoped_to_evaluation_and_restored(self):
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer._travel_eval_generation = False
        trainer.rotate_eval_subset = False
        trainer.eval_dataset = [{}]
        observed = []

        def fake_evaluate(base_trainer, num_eval_samples=None):
            observed.append(base_trainer._travel_eval_generation)
            return {"count": float(num_eval_samples)}

        with patch.object(MAGRPOTrainer, "evaluate", autospec=True) as evaluate:
            evaluate.side_effect = fake_evaluate
            result = trainer.evaluate(num_eval_samples=1)

        self.assertEqual(result, {"count": 1.0})
        self.assertEqual(observed, [True])
        self.assertFalse(trainer._travel_eval_generation)

    def test_eval_flag_is_restored_after_failure(self):
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer._travel_eval_generation = False
        trainer.rotate_eval_subset = False
        trainer.eval_dataset = [{}]
        with patch.object(
            MAGRPOTrainer, "evaluate", autospec=True, side_effect=RuntimeError("boom")
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                trainer.evaluate(num_eval_samples=1)
        self.assertFalse(trainer._travel_eval_generation)

    def test_travel_eval_logs_only_compact_metrics_in_one_wandb_step(self):
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer.eval_logger = lambda **_kwargs: [{"sample_id": "x"}]
        trainer.eval_aggregator = lambda _rows, num_turns: {
            "turn_1/eval_samples": "sample-table",
            "turn_1/action_validity": 0.5,
            "turn_1/ultimate/team_action_success": 0.5,
            "turn_1/required_cooperative_contribution": 0.5,
            "turn_1/required_grounded_recall": 0.5,
            "turn_1/entity_grounding_precision": 0.5,
            "turn_1/grounding_f1": 0.5,
            "turn_1/route_scaffold_match_rate": 0.5,
            "turn_1/ultimate/reference_plan_delivery": 0.5,
            "turn_1/ultimate/required_plan_completion": 0.5,
            "turn_1/ultimate/reference_commonsense_micro": 0.5,
            "turn_1/ultimate/reference_hard_micro": 0.5,
            "turn_1/ultimate/reference_plan_success": 0.5,
            "turn_1/ultimate/collaboration_success": 0.5,
        }
        trainer.args = SimpleNamespace(num_turns=1)
        trainer.wandb_initialized = True
        trainer.env_step = 240
        trainer.rotate_eval_subset = False
        fake_wandb = SimpleNamespace(
            run=object(),
            log=Mock(),
        )

        with patch(
            "single_turn.train.structured_trainer.wandb", fake_wandb
        ):
            metrics = trainer._log_eval_metrics(
                all_agent_completions_turns=[[["a0"]], [["a1"]]],
                all_test_cases=[""],
                all_entry_points=[""],
                all_prompts=["prompt"],
                extra_metrics={"eval/turn_1/reward_mean": 0.75},
            )

        self.assertEqual(metrics["eval/reward"], 0.75)
        self.assertEqual(metrics["eval/team_action_success"], 0.5)
        self.assertEqual(metrics["eval/reference_plan_delivery"], 0.5)
        self.assertEqual(metrics["eval/required_plan_completion"], 0.5)
        self.assertNotIn("eval/samples", metrics)
        self.assertEqual(
            set(metrics),
            {
                "eval/reward",
                "eval/action_validity",
                "eval/team_action_success",
                "eval/required_cooperative_contribution",
                "eval/required_grounded_recall",
                "eval/entity_grounding_precision",
                "eval/grounding_f1",
                "eval/route_scaffold_match",
                "eval/reference_plan_delivery",
                "eval/required_plan_completion",
                "eval/reference_commonsense_micro",
                "eval/reference_hard_micro",
                "eval/reference_plan_success",
                "eval/collaboration_success",
            },
        )
        fake_wandb.log.assert_called_once()
        self.assertEqual(fake_wandb.log.call_args.args[0], metrics)
        self.assertEqual(fake_wandb.log.call_args.kwargs["step"], 240)
        self.assertTrue(fake_wandb.log.call_args.kwargs["commit"])

    def test_final_eval_separates_full_pool_from_anchor_curve(self):
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer.eval_logger = lambda **_kwargs: [
            {"_eval_sample": {"reward": index / 10.0}} for index in range(5)
        ]
        trainer.eval_aggregator = lambda rows, num_turns: {
            "turn_1/eval_sample_count": float(len(rows)),
            "turn_1/eval_samples": f"table-{len(rows)}",
            "turn_1/ultimate/reference_plan_delivery": len(rows) / 10.0,
            "turn_1/ultimate/required_plan_completion": len(rows) / 10.0,
            "turn_1/ultimate/collaboration_success": len(rows) / 10.0,
        }
        trainer.args = SimpleNamespace(num_turns=1, eval_num_samples=2)
        trainer.wandb_initialized = True
        trainer.env_step = 5040
        fake_wandb = SimpleNamespace(run=object(), log=Mock())

        with patch(
            "single_turn.train.structured_trainer.wandb", fake_wandb
        ):
            metrics = trainer._log_eval_metrics(
                all_agent_completions_turns=[[["a0"]], [["a1"]]],
                all_test_cases=[""],
                all_entry_points=[""],
                all_prompts=["prompt"],
                extra_metrics={
                    "eval/turn_1/reward_mean": 0.8,
                    "eval/turn_1/expected_return": 0.8,
                },
            )

        self.assertNotIn("eval/samples", metrics)
        self.assertEqual(metrics["eval/reward"], 0.05)
        self.assertEqual(metrics["eval_full/reward"], 0.8)
        self.assertEqual(metrics["eval/reference_plan_delivery"], 0.2)
        self.assertEqual(metrics["eval_full/reference_plan_delivery"], 0.5)
        self.assertEqual(metrics["eval/required_plan_completion"], 0.2)
        self.assertEqual(metrics["eval_full/required_plan_completion"], 0.5)
        self.assertEqual(metrics["eval/collaboration_success"], 0.2)
        self.assertEqual(metrics["eval_full/collaboration_success"], 0.5)
        self.assertEqual(
            set(metrics),
            {
                "eval/reward",
                "eval/reference_plan_delivery",
                "eval/required_plan_completion",
                "eval/collaboration_success",
                "eval_full/reward",
                "eval_full/reference_plan_delivery",
                "eval_full/required_plan_completion",
                "eval_full/collaboration_success",
            },
        )
        fake_wandb.log.assert_called_once()
        self.assertEqual(
            fake_wandb.log.call_args.args[0],
            {
                "eval/reward": 0.05,
                "eval/reference_plan_delivery": 0.2,
                "eval/required_plan_completion": 0.2,
                "eval/collaboration_success": 0.2,
            },
        )
        self.assertTrue(fake_wandb.log.call_args.kwargs["commit"])

    def test_train_buffers_are_processed_without_wandb_uploads(self):
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer.wandb_initialized = True
        trainer.env_step = 40
        trainer.rollout_buffers = [["agent-0-buffer"], ["agent-1-buffer"]]
        trainer._parallel_agent_mode_enabled = Mock(return_value=False)

        def run_agent_tasks(process, *, agent_indices, parallel):
            return [process(agent_idx) for agent_idx in agent_indices]

        trainer._run_agent_tasks = Mock(side_effect=run_agent_tasks)
        trainer._process_buffer = Mock(side_effect=lambda agent_idx, buffer: {
            "log_entries": [{
                "agent_idx": agent_idx,
                "step": 40,
                "metrics": {"train/reward": 0.5, "turn_1/reward_mean": 0.5},
            }],
        })
        fake_wandb = SimpleNamespace(run=object(), log=Mock())
        with patch("comlrl.trainers.reinforce.magrpo.wandb", fake_wandb):
            trainer._drain_ready_buffers([0, 1])

        self.assertEqual(trainer._process_buffer.call_count, 2)
        trainer._process_buffer.assert_any_call(0, trainer.rollout_buffers[0])
        trainer._process_buffer.assert_any_call(1, trainer.rollout_buffers[1])
        fake_wandb.log.assert_not_called()

    def test_rotating_eval_cycles_through_held_out_pool(self):
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        original = [{"id": index} for index in range(5)]
        trainer.eval_dataset = original
        trainer.rotate_eval_subset = True
        trainer._eval_cursor = 0
        trainer.args = SimpleNamespace(eval_num_samples=2)
        snapshots = []

        def fake_evaluate(base_trainer, num_eval_samples=None):
            snapshots.append(
                [row["id"] for row in base_trainer.eval_dataset[:num_eval_samples]]
            )
            return {"count": float(num_eval_samples)}

        with patch.object(MAGRPOTrainer, "evaluate", autospec=True) as evaluate:
            evaluate.side_effect = fake_evaluate
            trainer.evaluate(num_eval_samples=2)
            trainer.evaluate(num_eval_samples=2)
            trainer.evaluate(num_eval_samples=2)
            trainer.evaluate(num_eval_samples=5)

        self.assertEqual(snapshots, [[0, 1], [2, 3], [4, 0], [1, 2, 3, 4, 0]])
        self.assertIs(trainer.eval_dataset, original)
        self.assertEqual(trainer._eval_cursor, 1)

    def test_rotating_eval_restores_dataset_after_failure(self):
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        original = [{"id": index} for index in range(5)]
        trainer.eval_dataset = original
        trainer.rotate_eval_subset = True
        trainer._eval_cursor = 2
        trainer.args = SimpleNamespace(eval_num_samples=2)

        with patch.object(
            MAGRPOTrainer, "evaluate", autospec=True, side_effect=RuntimeError("boom")
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                trainer.evaluate(num_eval_samples=2)
        self.assertIs(trainer.eval_dataset, original)
        self.assertEqual(trainer._eval_cursor, 2)

    def test_stop_criterion_ignores_braces_and_escapes_inside_strings(self):
        tokenizer = FakeTokenizer()
        criterion = CompleteJSONObjectCriteria(tokenizer, prompt_length=2)
        first = '{"value":"} and \\"quoted\\"","nested":{'
        first_ids = torch.tensor([[1, 2, *map(ord, first)]])
        self.assertFalse(criterion(first_ids, torch.empty(1)).item())

        complete = first + '"x":1}}'
        complete_ids = torch.tensor([[1, 2, *map(ord, complete)]])
        self.assertTrue(criterion(complete_ids, torch.empty(1)).item())

    def test_stop_criterion_tracks_return_sequences_independently(self):
        tokenizer = FakeTokenizer()
        criterion = CompleteJSONObjectCriteria(tokenizer, prompt_length=2)
        first_responses = ['{"a":1}', '{"b":  ']
        first_ids = torch.tensor(
            [[1, 2, *map(ord, response)] for response in first_responses]
        )
        self.assertEqual(
            criterion(first_ids, torch.empty(1)).tolist(),
            [True, False],
        )
        self.assertEqual(criterion.completed_response_lengths, (7, None))

        second_responses = [first_responses[0] + "  ", first_responses[1] + "2}"]
        second_ids = torch.tensor(
            [[1, 2, *map(ord, response)] for response in second_responses]
        )
        self.assertEqual(
            criterion(second_ids, torch.empty(1)).tolist(),
            [True, True],
        )
        self.assertEqual(criterion.completed_response_lengths, (7, 9))

    def test_stop_criterion_resumes_from_full_open_value_prefill(self):
        tokenizer = FakeTokenizer()
        prefix = build_agent_json_prefill(0, 3)
        criterion = CompleteJSONObjectCriteria(
            tokenizer,
            prompt_length=2,
            initial_text=prefix,
        )
        partial = "from Alpha to Beta"
        partial_ids = torch.tensor([[1, 2, *map(ord, partial)]])
        self.assertFalse(criterion(partial_ids, torch.empty(1)).item())

        suffix = partial + '"}]}'
        complete_ids = torch.tensor([[1, 2, *map(ord, suffix)]])
        self.assertTrue(criterion(complete_ids, torch.empty(1)).item())
        self.assertEqual(
            criterion.completed_response_lengths,
            (len(suffix),),
        )

    def test_stop_criterion_terminates_an_unclosed_assignments_array_as_invalid(self):
        tokenizer = FakeTokenizer()
        prefix = build_agent_json_prefill(0, 3)
        criterion = CompleteJSONObjectCriteria(
            tokenizer,
            prompt_length=2,
            initial_text=prefix,
        )
        # This is the dominant failed-run shape: the assignment object closes,
        # then a top-level brace is emitted before the assignments list's ']'.
        malformed = 'from Alpha to Beta"} }'
        malformed_ids = torch.tensor([[1, 2, *map(ord, malformed)]])
        self.assertTrue(criterion(malformed_ids, torch.empty(1)).item())
        self.assertEqual(criterion.completed_response_lengths, (len(malformed),))
        self.assertTrue(criterion._states[0].invalid)
        self.assertFalse(criterion._states[0].complete)

    def test_travel_trainer_injects_fresh_prefix_and_stop_constraints(self):
        tokenizer = FakeTokenizer()
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer.chat_formatted_prompts = True
        trainer.force_json_prefix = True
        trainer.stop_after_complete_json = True
        trainer._travel_eval_generation = True
        trainer.greedy_eval = True
        trainer.tokenizers = [tokenizer]
        trainer.formatters = [lambda item, external_prompts=None: item["prompt"]]
        prefix = build_agent_json_prefill(0, 3)
        suffix = 'from Alpha to Beta"}]}'

        with patch.object(
            MAGRPOTrainer,
            "_generate_completions",
            return_value={
                "ok": True,
                "prompt_input_ids": torch.tensor([[2, 3]]),
                "completion_input_ids": [
                    [torch.tensor([ord(char) for char in suffix])]
                ],
                "completion_attention_mask": [[torch.ones(len(suffix))]],
                "completions": [[suffix]],
                "response_lens": [len(suffix)],
            },
        ) as generate:
            result = trainer._generate_completions(
                object(),
                [{"prompt": "rendered chat prompt", "days": 3}],
                agent_idx=0,
                num_return_sequences=4,
            )

        self.assertTrue(result["ok"])
        call_kwargs = generate.call_args.kwargs
        self.assertIn("stopping_criteria", call_kwargs)
        self.assertIsInstance(
            call_kwargs["logits_processor"][-1], GreedyArgmaxLogitsProcessor
        )
        self.assertEqual(
            call_kwargs["prompts_override"], ["rendered chat prompt" + prefix]
        )
        self.assertNotIn("prompt_tokenizer_kwargs", call_kwargs)
        self.assertNotIn("response_length_resolver", call_kwargs)
        self.assertIsInstance(
            call_kwargs["stopping_criteria"][-1],
            CompleteJSONObjectCriteria,
        )
        self.assertEqual(
            call_kwargs["stopping_criteria"][-1].initial_text,
            prefix,
        )
        self.assertEqual(result["completions"], [[prefix + suffix]])
        self.assertEqual(result["response_lens"], [len(suffix)])
        self.assertNotIn("completion_loss_mask", result)

    def test_travel_trainer_injects_fixed_skeleton_and_value_loss_mask(self):
        tokenizer = FakeTokenizer()
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer.chat_formatted_prompts = True
        trainer.force_json_prefix = True
        trainer.constrain_json_skeleton = True
        trainer.max_value_tokens = 8
        trainer.normalize_value_log_probs = True
        trainer.stop_after_complete_json = True
        trainer._travel_eval_generation = True
        trainer.greedy_eval = True
        trainer.tokenizers = [tokenizer]
        trainer.formatters = [lambda item, external_prompts=None: item["prompt"]]

        slots = ordered_owned_slots(0, 1)
        payload = {
            "agent_id": 0,
            "assignments": [
                {"day": day, "field": field, "value": value}
                for (day, field), value in zip(slots, ("A", "-", "-"))
            ],
        }
        full_response = json.dumps(payload, ensure_ascii=False)
        prefix = build_agent_json_prefill(0, 1)
        self.assertTrue(full_response.startswith(prefix))
        generated_text = full_response[len(prefix) :]
        generated_tokens = torch.tensor(
            [ord(character) for character in generated_text], dtype=torch.long
        )

        with patch.object(
            MAGRPOTrainer,
            "_generate_completions",
            return_value={
                "prompt_input_ids": torch.tensor([[2, 3]]),
                "completion_input_ids": [[generated_tokens]],
                "completion_attention_mask": [[torch.ones(len(generated_tokens))]],
                "completions": [[generated_text]],
                "response_lens": [len(generated_tokens)],
            },
        ) as generate:
            result = trainer._generate_completions(
                object(),
                [{"prompt": "rendered chat prompt", "days": 1}],
                agent_idx=0,
                max_new_tokens=256,
            )

        processors = generate.call_args.kwargs["logits_processor"]
        self.assertIsInstance(processors[-2], FixedSlotJSONLogitsProcessor)
        self.assertIsInstance(processors[-1], GreedyArgmaxLogitsProcessor)
        self.assertEqual(result["completions"], [[full_response]])
        loss_mask = result["completion_loss_mask"][0][0]
        self.assertEqual(loss_mask.numel(), generated_tokens.numel())
        self.assertEqual(int(loss_mask.sum().item()), 6)
        self.assertLess(int(loss_mask.sum().item()), loss_mask.numel())

        packed = trainer._pack_completions_for_buffer(result)
        self.assertEqual(
            packed["completion_loss_mask"][0][0].tolist(), loss_mask.tolist()
        )

    def test_real_hf_generate_produces_the_complete_fixed_slot_schema(self):
        tokenizer = _TinyGenerationTokenizer()
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=128,
                n_positions=256,
                n_embd=16,
                n_layer=1,
                n_head=1,
                bos_token_id=0,
                eos_token_id=1,
                pad_token_id=1,
            )
        )
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer.chat_formatted_prompts = True
        trainer.force_json_prefix = True
        trainer.constrain_json_skeleton = True
        trainer.max_value_tokens = 1
        trainer.normalize_value_log_probs = True
        trainer.stop_after_complete_json = True
        trainer._travel_eval_generation = True
        trainer.greedy_eval = True
        trainer.tokenizer = tokenizer
        trainer.tokenizers = [tokenizer]
        trainer.formatters = [lambda item, external_prompts=None: "chat-rendered"]
        trainer.args = SimpleNamespace(
            temperature=0.7,
            top_p=0.8,
            top_k=None,
            reference_kl_enabled=False,
        )
        trainer.reference_models = []

        result = trainer._generate_completions(
            model,
            [{"prompt": "ignored", "days": 1}],
            agent_idx=0,
            num_return_sequences=2,
            max_new_tokens=192,
        )

        expected_slots = ordered_owned_slots(0, 1)
        self.assertEqual(len(result["completions"][0]), 2)
        for response in result["completions"][0]:
            payload = json.loads(response)
            self.assertEqual(payload["agent_id"], 0)
            self.assertEqual(
                [
                    (assignment["day"], assignment["field"])
                    for assignment in payload["assignments"]
                ],
                expected_slots,
            )
        for mask, tokens in zip(
            result["completion_loss_mask"][0],
            result["completion_input_ids"][0],
        ):
            self.assertEqual(mask.numel(), tokens.numel())
            self.assertEqual(int(mask.sum().item()), len(expected_slots))

    def test_travel_policy_loss_ignores_schema_tokens_and_normalizes_values(self):
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer.args = SimpleNamespace(
            advantage_normalization=False,
            reference_kl_enabled=False,
            reference_kl_coef=0.1,
        )
        trainer.advantage_mode = "mean"
        trainer.normalize_value_log_probs = True
        model = _PositionIndependentLM()
        completions_data = {
            "prompt_input_ids": torch.tensor([[0, 1]], dtype=torch.long),
            "completion_input_ids": [
                [
                    torch.tensor([2, 3, 4, 5], dtype=torch.long),
                    torch.tensor([2, 3], dtype=torch.long),
                ]
            ],
            "completion_loss_mask": [
                [
                    torch.tensor([1, 0, 0, 1], dtype=torch.bool),
                    torch.tensor([1, 1], dtype=torch.bool),
                ]
            ],
            "reference_kls": [],
        }
        loss = trainer._compute_loss_with_gradients(
            model, completions_data, returns=[1.0, -1.0]
        )
        log_probs = torch.log_softmax(model.token_logits, dim=-1)
        expected = (
            -((log_probs[2] + log_probs[5]) / 2.0)
            + ((log_probs[2] + log_probs[3]) / 2.0)
        ) / 2.0
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(model.token_logits.grad).all())

    def test_travel_policy_loss_rejects_a_misaligned_value_mask(self):
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer.args = SimpleNamespace(
            advantage_normalization=False,
            reference_kl_enabled=False,
            reference_kl_coef=0.1,
        )
        trainer.advantage_mode = "mean"
        trainer.normalize_value_log_probs = True
        completions_data = {
            "prompt_input_ids": torch.tensor([[0, 1]], dtype=torch.long),
            "completion_input_ids": [[torch.tensor([2, 3], dtype=torch.long)]],
            "completion_loss_mask": [[torch.tensor([1], dtype=torch.bool)]],
            "reference_kls": [],
        }
        with self.assertRaisesRegex(ValueError, "must have equal length"):
            trainer._compute_loss_with_gradients(
                _PositionIndependentLM(), completions_data, returns=[1.0]
            )

    def test_real_hf_generate_crops_each_stopped_sequence_before_padded_eos(self):
        tokenizer = _TinyGenerationTokenizer()
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=128,
                n_positions=64,
                n_embd=16,
                n_layer=1,
                n_head=1,
                bos_token_id=0,
                eos_token_id=1,
                pad_token_id=1,
            )
        )
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer.chat_formatted_prompts = True
        trainer.force_json_prefix = True
        trainer.stop_after_complete_json = True
        trainer.tokenizer = tokenizer
        trainer.tokenizers = [tokenizer]
        trainer.formatters = [lambda item, external_prompts=None: "chat-rendered"]
        trainer.args = SimpleNamespace(
            temperature=0.7,
            top_p=0.8,
            top_k=None,
            reference_kl_enabled=False,
        )
        trainer.reference_models = []
        prefix = build_agent_json_prefill(0, 3)

        with patch(
            "comlrl.trainers.reinforce.magrpo.apply_tokenizer_specials",
            lambda *_args, **_kwargs: None,
        ):
            result = trainer._generate_completions(
                model,
                [{"days": 3}],
                agent_idx=0,
                num_return_sequences=4,
                max_new_tokens=24,
                do_sample=True,
                logits_processor=LogitsProcessorList(
                    [_ForceDifferentJSONRows(prompt_length=2)]
                ),
            )

        expected = [prefix + response for response in _ForceDifferentJSONRows.responses]
        self.assertEqual(result["completions"], [expected])
        self.assertEqual(
            result["response_lens"],
            list(map(len, _ForceDifferentJSONRows.responses)),
        )
        self.assertEqual(
            result["completion_attention_mask"][0][0].tolist(),
            [1] * len(_ForceDifferentJSONRows.responses[0]),
        )
        self.assertNotIn("completion_loss_mask", result)

    def test_raw_prompt_mode_does_not_disable_tokenizer_special_tokens(self):
        tokenizer = FakeTokenizer()
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer.chat_formatted_prompts = False
        trainer.force_json_prefix = False
        trainer.stop_after_complete_json = False
        trainer.tokenizers = [tokenizer]
        trainer.formatters = [lambda item, external_prompts=None: item["prompt"]]
        base_result = {
            "prompt_input_ids": torch.tensor([[2, 3]]),
            "completion_input_ids": [[torch.tensor([ord("x")])]],
            "completion_attention_mask": [[torch.tensor([1, 1])]],
            "completions": [["x"]],
            "response_lens": [1],
        }

        with patch.object(
            MAGRPOTrainer,
            "_generate_completions",
            return_value=base_result,
        ) as generate:
            trainer._generate_completions(
                object(), [{"prompt": "raw prompt"}], agent_idx=0
            )

        self.assertEqual(generate.call_args.kwargs["prompts_override"], ["raw prompt"])
        self.assertNotIn("prompt_tokenizer_kwargs", generate.call_args.kwargs)
        self.assertNotIn("logits_processor", generate.call_args.kwargs)

    def test_greedy_eval_can_be_disabled(self):
        tokenizer = FakeTokenizer()
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer.chat_formatted_prompts = False
        trainer.force_json_prefix = False
        trainer.stop_after_complete_json = False
        trainer._travel_eval_generation = True
        trainer.greedy_eval = False
        trainer.tokenizers = [tokenizer]
        trainer.formatters = [lambda item, external_prompts=None: item["prompt"]]
        base_result = {
            "prompt_input_ids": torch.tensor([[2, 3]]),
            "completion_input_ids": [[torch.tensor([ord("x")])]],
            "completion_attention_mask": [[torch.tensor([1])]],
            "completions": [["x"]],
            "response_lens": [1],
        }
        with patch.object(
            MAGRPOTrainer, "_generate_completions", return_value=base_result
        ) as generate:
            trainer._generate_completions(
                object(), [{"prompt": "raw prompt"}], agent_idx=0
            )
        self.assertNotIn("logits_processor", generate.call_args.kwargs)


class TravelCurriculumTests(unittest.TestCase):
    def test_default_curriculum_preserves_5040_env_steps(self):
        config = Config(
            str(REPO_ROOT / "single_turn/configs/single_turn_magrpo_config.yaml")
        )
        rows = [
            {"source_index": index, "days": 3 if index < 30 else 5}
            for index in range(60)
        ]
        plan = _curriculum_plan(config, rows)
        self.assertEqual(len(plan["short_rows"]), 30)
        self.assertEqual(plan["short_epochs"], 8)
        self.assertEqual(plan["full_epochs"], 17)
        self.assertEqual(plan["expected_env_steps"], 5040)

    def test_curriculum_dataloader_switches_without_replacing_full_dataset(self):
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        full = [{"id": index} for index in range(4)]
        short = full[:2]
        trainer.train_dataset = full
        trainer.curriculum_train_dataset = short
        trainer.curriculum_short_epochs = 2
        trainer._travel_training_active = True
        trainer._travel_train_epoch_cursor = 0
        observed = []

        def fake_dataloader(base_trainer):
            observed.append(base_trainer.train_dataset)
            return base_trainer.train_dataset

        with patch.object(
            MAGRPOTrainer, "get_train_dataloader", autospec=True
        ) as get_dataloader:
            get_dataloader.side_effect = fake_dataloader
            self.assertIs(trainer.get_train_dataloader(), short)
            self.assertIs(trainer.get_train_dataloader(), short)
            self.assertIs(trainer.get_train_dataloader(), full)

        self.assertEqual(observed, [short, short, full])
        self.assertIs(trainer.train_dataset, full)
        self.assertEqual(trainer._travel_train_epoch_cursor, 3)

    def test_curriculum_train_resets_state_and_cleans_up_after_failure(self):
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer.args = SimpleNamespace(num_train_epochs=2)
        trainer._travel_training_active = False
        trainer._travel_train_epoch_cursor = 99
        trainer._travel_curriculum_stage = 1
        trainer._travel_train_detail_groups = []
        trainer.curriculum_short_epochs = 1

        def fake_train(base_trainer, **_kwargs):
            self.assertTrue(base_trainer._travel_training_active)
            base_trainer._travel_train_epoch_cursor = 1
            raise RuntimeError("boom")

        with patch.object(MAGRPOTrainer, "train", autospec=True) as train:
            train.side_effect = fake_train
            with self.assertRaisesRegex(RuntimeError, "boom"):
                trainer.train()
        self.assertFalse(trainer._travel_training_active)
        self.assertEqual(trainer._travel_curriculum_stage, 1)

    def test_train_reward_diagnostics_include_signal_and_end_metrics(self):
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer._travel_training_active = True
        trainer._travel_eval_generation = False
        trainer._travel_curriculum_stage = 0
        trainer._travel_train_detail_groups = []
        trainer.env_step = 8
        details = [
            {
                "action_validity": 1.0,
                "ultimate/team_action_success": 1.0,
                "required_cooperative_contribution": value,
                "required_grounded_recall": value,
                "grounding_f1": value,
                "plan_score": value,
                "strict_composite_quality": value,
                "protocol_progress": 1.0,
                "recovered_semantic_balance": value,
                "ultimate/required_plan_completion": value,
                "ultimate/collaboration_success": 0.0,
            }
            for value in (0.2, 0.4, 0.6, 0.8)
        ]
        trainer.reward_func = SimpleNamespace(drain_details=lambda: details)

        with patch.object(
            MAGRPOTrainer,
            "_compute_rewards",
            autospec=True,
            return_value=[0.1, 0.2, 0.3, 0.4],
        ):
            rewards = trainer._compute_rewards([], [[], []], batch_items=[{}])

        self.assertEqual(rewards, [0.1, 0.2, 0.3, 0.4])
        record = trainer._travel_train_detail_groups[0]
        self.assertEqual(record["_step"], 12.0)
        self.assertAlmostEqual(record["train/reward"], 0.25)
        self.assertGreater(record["train/reward_group_std"], 0.0)
        self.assertEqual(record["train/nonzero_advantage_rate"], 1.0)
        self.assertAlmostEqual(
            record["train/required_cooperative_contribution"], 0.5
        )
        self.assertAlmostEqual(record["train/required_plan_completion"], 0.5)
        self.assertEqual(record["train/curriculum_stage"], 0.0)

    def test_train_reward_diagnostics_are_attached_to_agent_zero_log(self):
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer._travel_train_detail_groups = [
            {"_step": 4.0, "train/reward": 0.2},
            {"_step": 8.0, "train/reward": 0.4},
        ]
        base_result = {
            "log_entries": [{"step": 8, "metrics": {"turn_1/reward_mean": 0.3}}]
        }
        with patch.object(
            MAGRPOTrainer, "_process_buffer", autospec=True, return_value=base_result
        ):
            result = trainer._process_buffer(0, [])
        self.assertAlmostEqual(
            result["log_entries"][0]["metrics"]["train/reward"], 0.3
        )
        self.assertEqual(trainer._travel_train_detail_groups, [])


if __name__ == "__main__":
    unittest.main()
