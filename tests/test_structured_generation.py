from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
from single_turn.structured_generation import (
    CompleteJSONObjectCriteria,
    apply_chat_template,
    wrap_formatter_with_chat_template,
)
from single_turn.train.structured_trainer import StructuredOutputMAGRPOTrainer


class FakeTokenizer:
    chat_template = "fake-template"
    pad_token_id = 0
    eos_token_id = 0

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

    def test_travel_trainer_injects_fresh_prefix_and_stop_constraints(self):
        tokenizer = FakeTokenizer()
        trainer = object.__new__(StructuredOutputMAGRPOTrainer)
        trainer.chat_formatted_prompts = True
        trainer.force_json_prefix = True
        trainer.stop_after_complete_json = True
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


if __name__ == "__main__":
    unittest.main()
