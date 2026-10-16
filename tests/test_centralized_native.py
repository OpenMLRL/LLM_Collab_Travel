from __future__ import annotations

import copy
import io
import json
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

from comlrl.trainers.preference.centralized import CentralizedComparatorParseError
from comlrl.trainers.preference.collaboration import CentralizedCollaboration
from comlrl.trainers.reinforce import CentralizedMAGRPOConfig
from single_turn.aggregation import ordered_owned_slots
from single_turn.centralized_collaboration import TravelCentralizedCollaborationAdapter
from single_turn.centralized_native import (
    NATIVE_SECTION_KEYS,
    SectionedFixedSlotJSONLogitsProcessor,
    TravelNativeCentralizedAdapter,
    build_native_prefill,
    native_slots,
)
from single_turn.formatting import get_single_turn_formatters
from single_turn.logger import (
    aggregate_single_turn_metrics,
    build_single_turn_eval_logger,
)
from single_turn.rewards import make_reward
from single_turn.train.centralized_trainer import TravelNativeCentralizedMAGRPOTrainer
from single_turn.train.structured_trainer import StructuredOutputMAGRPOTrainer
from single_turn.train.train_centralized import (
    centralized_prompt_mode,
    load_config,
    make_trainer_args,
)
from test_centralized_collaboration import TinyTokenizer
from test_preference_training import tiny_config
from test_single_turn import fixture_item, valid_completions, all_dash_completions
from test_structured_generation import FakeTokenizer

NATIVE_CONFIG = (
    ROOT / "single_turn/configs/single_turn_centralized_native_magrpo_config.yaml"
)


def native_text(roles):
    return json.dumps(
        {
            key: json.loads(role)["assignments"]
            for key, role in zip(NATIVE_SECTION_KEYS, roles)
        }
    )


class TinyNativeAdapter(TravelNativeCentralizedAdapter):
    def build_prompt(self, batch_item, agent_prompts):
        return "joint context"


def make_native_trainer(**overrides):
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
        num_generations=2,
        advantage_normalization=False,
    )
    options.update(overrides)
    trainer = TravelNativeCentralizedMAGRPOTrainer(
        agents=[GPT2LMHeadModel(tiny_config())],
        tokenizer=TinyTokenizer(),
        num_agents=2,
        args=CentralizedMAGRPOConfig(**options),
        reward_func=make_reward(),
        train_dataset=[fixture_item()],
        eval_dataset=[fixture_item()],
        formatters=get_single_turn_formatters(num_agents=2, use_chat_template=False),
        eval_logger=build_single_turn_eval_logger([fixture_item()]),
        eval_aggregator=aggregate_single_turn_metrics,
        chat_formatted_prompts=True,
        centralized_adapter=TinyNativeAdapter(),
    )
    trainer.verbose = False
    return trainer


def scripted_native_generation(
    self, agent, batch_items, *, agent_idx, num_return_sequences, **kwargs
):
    assert agent_idx == 0
    assert kwargs["slot_order_override"] == native_slots(batch_items[0]["days"])
    idx = getattr(agent, "test_candidate_index", 0)
    agent.test_candidate_index = idx + num_return_sequences
    fulls = [
        native_text(
            valid_completions()
            if (idx + offset) % 4 in (0, 3)
            else all_dash_completions()
        )
        for offset in range(num_return_sequences)
    ]
    prefix = build_native_prefill(3)
    ids = [torch.tensor([ord(char) for char in full[len(prefix) :]]) for full in fulls]
    grammar = SectionedFixedSlotJSONLogitsProcessor(
        self.tokenizers[0], prompt_length=2, slots=native_slots(3)
    )
    masks = [
        torch.tensor(mask, dtype=torch.bool)
        for mask in grammar.finalize_loss_masks(ids)
    ]
    return {
        "prompts": [kwargs["prompts_override"][0] + prefix],
        "prompt_input_ids": torch.tensor([[2, 3]]),
        "prompt_attention_mask": torch.ones(1, 2),
        "batch_items": batch_items,
        "completions": [fulls],
        "completion_input_ids": [ids],
        "completion_attention_mask": [[torch.ones_like(tokens) for tokens in ids]],
        "completion_loss_mask": [masks],
        "response_lens": [len(tokens) for tokens in ids],
        "reference_kls": [0.0] * num_return_sequences,
    }


class NativePromptTests(unittest.TestCase):
    def test_prompt_states_each_part_once_without_role_prompts_or_gold(self):
        item = fixture_item()
        item.update(annotated_plan="SECRET_GOLD_1", gold_plan="SECRET_GOLD_2")
        prompt = TravelNativeCentralizedAdapter(FakeTokenizer()).build_prompt(item, [])
        self.assertNotIn("SECRET_GOLD", prompt)
        self.assertEqual(prompt.count("<assistant>"), 1)
        for once in (
            "TRIP REQUEST:",
            "STRUCTURED CONSTRAINTS:",
            "SHARED REFERENCE-DERIVED ROUTE SCAFFOLD",
            "DETERMINISTIC TEAM BUDGET CONTRACT",
            "SUBTASK A CATALOG",
            "SUBTASK B CATALOG",
            "TRANSPORTATION OPTIONS",
            "RESTAURANT OPTIONS",
            "ASSISTANT PREFILL (already supplied",
        ):
            self.assertEqual(prompt.count(once), 1, once)
        for absent in ("decentralized", "cannot communicate", "REFERENCE ROLE"):
            self.assertNotIn(absent, prompt)
        self.assertIn(build_native_prefill(3), prompt)
        concatenated = TravelCentralizedCollaborationAdapter().build_prompt(item, [])
        self.assertLess(len(prompt), len(concatenated))

    def test_split_gives_the_unchanged_role_actions_and_reward(self):
        reward = make_reward()
        roles = get_single_turn_formatters(num_agents=2, use_chat_template=False)
        runtime = CentralizedCollaboration(
            TravelNativeCentralizedAdapter(), roles, reward, 2
        )
        item = fixture_item()
        for outputs in (valid_completions(), all_dash_completions()):
            split = TravelNativeCentralizedAdapter().parse_completion(
                native_text(outputs), item, 2
            )
            self.assertEqual(
                [json.loads(text) for text in split],
                [json.loads(text) for text in outputs],
            )
            expected = reward([outputs[0]], [outputs[1]], batch_items=[item])
            self.assertEqual(
                runtime([native_text(outputs)], batch_items=[item]), expected
            )
        self.assertEqual(
            runtime(["unfinished JSON"], batch_items=[item]),
            reward([""], [""], batch_items=[item]),
        )

    def test_split_rejects_swapped_answers_or_slot_order(self):
        adapter = TravelNativeCentralizedAdapter()
        data = json.loads(native_text(valid_completions()))
        swapped = json.dumps({key: data[key] for key in reversed(NATIVE_SECTION_KEYS)})
        reordered = copy.deepcopy(data)
        reordered[NATIVE_SECTION_KEYS[1]].reverse()
        for text in (swapped, json.dumps(reordered)):
            with self.subTest(text=text[:40]), self.assertRaises(
                CentralizedComparatorParseError
            ):
                adapter.parse_completion(text, fixture_item(), 2)


class NativeGrammarTests(unittest.TestCase):
    def test_blank_skeleton_is_two_answers_in_subtask_order(self):
        grammar = SectionedFixedSlotJSONLogitsProcessor(
            TinyTokenizer(), prompt_length=2, slots=native_slots(3)
        )
        blank = build_native_prefill(3) + "".join(
            '-"' + grammar._suffix_after(index) for index in range(len(grammar.slots))
        )
        data = json.loads(blank)
        self.assertEqual(list(data), list(NATIVE_SECTION_KEYS))
        for agent_idx, key in enumerate(NATIVE_SECTION_KEYS):
            self.assertEqual(
                [(a["day"], a["field"]) for a in data[key]],
                ordered_owned_slots(agent_idx, 3),
            )

    def test_only_value_tokens_enter_the_loss_mask(self):
        text = native_text(valid_completions())
        prefix = build_native_prefill(3)
        ids = torch.tensor([ord(char) for char in text[len(prefix) :]])
        grammar = SectionedFixedSlotJSONLogitsProcessor(
            TinyTokenizer(), prompt_length=2, slots=native_slots(3)
        )
        (mask,) = grammar.finalize_loss_masks([ids])
        values = [
            a["value"] for key in NATIVE_SECTION_KEYS for a in json.loads(text)[key]
        ]
        # Every value character plus its closing quote is a policy token.
        self.assertEqual(sum(mask), sum(len(json.dumps(v)) - 1 for v in values))

    def test_mixed_slot_order_is_rejected(self):
        with self.assertRaises(ValueError):
            SectionedFixedSlotJSONLogitsProcessor(
                TinyTokenizer(), prompt_length=2, slots=list(reversed(native_slots(3)))
            )


class NativeLoopTests(unittest.TestCase):
    def test_real_generation_writes_both_answers_with_value_masks(self):
        torch.set_num_threads(1)
        trainer = make_native_trainer()
        trainer.max_value_tokens = 2
        result = trainer._generate_completions(
            trainer.agents[0],
            [fixture_item()],
            num_return_sequences=1,
            max_new_tokens=2048,
        )
        text = result["completions"][0][0]
        data = json.loads(text)
        for agent_idx, key in enumerate(NATIVE_SECTION_KEYS):
            self.assertEqual(
                [(a["day"], a["field"]) for a in data[key]],
                ordered_owned_slots(agent_idx, 3),
            )
        tokens = result["completion_input_ids"][0][0]
        mask = result["completion_loss_mask"][0][0]
        self.assertTrue(mask.any())
        self.assertFalse(mask.all())
        self.assertEqual(
            build_native_prefill(3)
            + trainer.tokenizers[0].decode(tokens, skip_special_tokens=True),
            text,
        )

    def test_magrpo_updates_one_actor_and_evaluates_split_roles(self):
        torch.set_num_threads(1)
        trainer = make_native_trainer()
        self.assertEqual((trainer.num_agents, len(trainer.agents)), (1, 1))
        before = copy.deepcopy(trainer.agents[0].state_dict())
        with patch.object(
            StructuredOutputMAGRPOTrainer,
            "_generate_completions",
            scripted_native_generation,
        ):
            trainer.train()
            metrics = trainer.evaluate(num_eval_samples=1)
        self.assertEqual(trainer.env_step, 2)
        self.assertTrue(
            any(
                not torch.equal(value, before[key])
                for key, value in trainer.agents[0].state_dict().items()
            )
        )
        self.assertIn("eval/reward", metrics)
        self.assertIn("eval/reference_plan_success", metrics)


class NativeEntrypointTests(unittest.TestCase):
    def test_config_selects_native_mode_for_magrpo_only(self):
        config = load_config(NATIVE_CONFIG)
        args = make_trainer_args(config)
        self.assertEqual(centralized_prompt_mode(config), "native")
        self.assertEqual(args.max_new_tokens, 2048)
        self.assertEqual(args.agent_devices, ["cuda:0"])
        self.assertIn("native", config.get("wandb.name"))
        concatenated = load_config(
            ROOT / "single_turn/configs/single_turn_centralized_magrpo_config.yaml"
        )
        self.assertEqual(centralized_prompt_mode(concatenated), "concatenated")
        for key in ("dataset", "travel_reward", "magrpo"):
            self.assertEqual(config.get_section(key), concatenated.get_section(key))
        madpo = load_config(
            ROOT / "single_turn/configs/single_turn_centralized_madpo_config.yaml"
        )
        madpo.update({"travel": {"centralized_prompt": "native"}})
        with self.assertRaises(ValueError):
            centralized_prompt_mode(madpo)

    def test_entrypoint_builds_native_trainer_and_adapter(self):
        from single_turn.train import train_centralized as entrypoint

        with tempfile.TemporaryDirectory() as directory:
            actor = SimpleNamespace(gradient_checkpointing_enable=Mock())
            trainer = SimpleNamespace(
                agents=[actor],
                train=Mock(),
                evaluate=Mock(return_value={"eval_full/reward": 0.5}),
                env_step=0,
                wandb_initialized=False,
            )
            overrides = [
                f"output.base_dir={directory}",
                "output.save_final_model=false",
                "wandb.enabled=false",
                "magrpo.eval_num_samples=1",
                "magrpo.num_train_epochs=1",
                "magrpo.final_eval_num_samples=1",
                "travel.curriculum_short_epochs=0",
            ]
            with patch.object(
                entrypoint,
                "_load_data",
                return_value=([fixture_item()], [fixture_item()]),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=TinyTokenizer(),
            ), patch(
                "single_turn.train.centralized_trainer."
                "TravelNativeCentralizedMAGRPOTrainer",
                return_value=trainer,
            ) as construct, patch.object(
                sys,
                "argv",
                ["train", "--config", str(NATIVE_CONFIG), "--override", *overrides],
            ), redirect_stdout(
                io.StringIO()
            ):
                entrypoint.main()
            kwargs = construct.call_args.kwargs
            self.assertIsInstance(
                kwargs["centralized_adapter"], TravelNativeCentralizedAdapter
            )
            trainer.train.assert_called_once()
            (saved,) = Path(directory).glob("*/final_metrics.json")
            self.assertEqual(json.loads(saved.read_text()), {"eval_full/reward": 0.5})


if __name__ == "__main__":
    unittest.main()
