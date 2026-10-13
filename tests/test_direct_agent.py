"""CPU scientific-contract, exact-gradient and all-five-entrypath tests."""

import copy
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from test_single_turn import fixture_item, valid_plan_values
from single_turn.baseline.direct_agent.protocol import (
    ALGORITHMS,
    METRICS,
    Settings,
    eval_key,
    fingerprint,
    messages,
    score,
    slots,
    PREFILL,
    select_pairs,
    aggregate_metrics,
)
from single_turn.baseline.direct_agent.objectives import (
    ScalarModel,
    token_logps,
    dpo_loss,
    dpo_delta_gradient,
    normalized,
    grpo_loss,
)
from single_turn.baseline.direct_agent.run import Experiment, load_data
from single_turn.baseline.direct_agent.generation import generate, generate_prompt_batch
from single_turn.baseline.direct_agent.aggregate import aggregate


def row(index=0):
    result = fixture_item()
    result["id"] = str(index)
    return result


def answer(good=True):
    values = valid_plan_values()
    return json.dumps(
        {
            "assignments": [
                dict(day=d, field=f, value=values[d, f] if good else "-")
                for d, f in slots(3)
            ]
        }
    )


def tiny(vocab=32, context=128):
    torch.manual_seed(13)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=vocab,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=context,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return model


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return json.dumps(messages)

    def encode(self, text, **kwargs):
        return [1, 3, 4, 5]


class CharTokenizer(Tokenizer):
    all_special_ids = [0, 1, 2]

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(int(i)) for i in ids if int(i) not in self.all_special_ids)


class FusedTokenizer(CharTokenizer):
    def decode(self, ids, **kwargs):
        fused = {128: 'Beta"}', 129: 'Beta"},', 130: 'Beta"}]}'}
        return "".join(
            fused.get(int(i), chr(int(i)))
            for i in ids
            if int(i) not in self.all_special_ids
        )


def synthetic_generate(model, tokenizer, prompt, count, cfg, *, row):
    return [
        (
            torch.tensor([7 if i % 2 == 0 else 8, 9, 2]),
            answer(i % 2 == 0),
            torch.tensor([True, True, False]),
        )
        for i in range(count)
    ]


def synthetic_batch(model, tokenizer, prompts, cfg, *, rows):
    return [
        synthetic_generate(model, tokenizer, prompt, 1, cfg, row=r)[0]
        for prompt, r in zip(prompts, rows)
    ]


class DirectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_no_targets_or_coordinator(self):
        a, b = row(), row()
        b.update(annotated_plan="SECRET", gold_plan="LEAK")
        self.assertEqual(messages(a), messages(b))
        text = json.dumps(messages(a)).lower()
        for forbidden in ("secret", "leak", "coordinator", "agent_id", "role-specific"):
            self.assertNotIn(forbidden, text)

    def test_reward_bounds_and_direct_completion(self):
        good, blank = score(answer(), row()), score(answer(False), row())
        self.assertAlmostEqual(good["reward"], 1.0)
        self.assertEqual(good["final_pass"], 1.0)
        self.assertEqual(good["plan_completion"], 1.0)
        self.assertEqual(blank["plan_delivery"], 0.0)
        self.assertEqual(blank["final_pass"], 0.0)
        self.assertLess(blank["reward"], 0.1)
        for text in (
            "",
            "invalid",
            answer() + answer(),
            "```json" + answer() + "```",
            answer().replace('"day": 1', '"day": true', 1),
        ):
            self.assertEqual(score(text, row())["reward"], 0.0)
        changed = json.loads(answer())
        changed["assignments"][0] = changed["assignments"][1]
        self.assertEqual(score(json.dumps(changed), row())["parse_success"], 0.0)
        expensive = row()
        expensive["budget"] = 1
        self.assertEqual(score(answer(), expensive)["budget_pass"], 0.0)
        self.assertLess(score(answer(), expensive)["reward"], good["reward"])

    def test_micro_weighted_by_applicable_checks(self):
        a = score(answer(), row())
        b = score(answer(False), row())
        a["_hard_passed"], a["_hard_total"] = 1, 1
        b["_hard_passed"], b["_hard_total"] = 1, 3
        self.assertEqual(aggregate_metrics([a, b])[eval_key("hard_micro")], 0.5)

    def test_wandb_phase_axes(self):
        fake = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules, {"wandb": fake}
        ):
            cfg = Settings(output_dir=directory, wandb_enabled=True, max_new_tokens=8)
            Experiment(cfg, [row(0)], [row(1)], {}, Tokenizer(), tiny())
        fake.define_metric.assert_any_call(
            "eval/*", step_metric="progress/responses_seen"
        )
        fake.define_metric.assert_any_call(
            "preference/*", step_metric="progress/preference_responses"
        )
        fake.define_metric.assert_any_call(
            "optimization/reward_model_loss",
            step_metric="optimization/reward_model_pairs",
        )
        self.assertEqual(fake.init.call_args.kwargs["project"], "Travel")

    def test_budgets_and_ties(self):
        for algorithm in ("grpo", "iac", "rlhf"):
            self.assertEqual(
                Settings(algorithm=algorithm).budget()["online_responses"], 5040
            )
        self.assertEqual(
            Settings(algorithm="dpo").budget()["dpo_response_presentations_cap"], 5040
        )
        self.assertEqual(
            Settings(algorithm="dpo").budget()["preference_responses"], 4800
        )
        self.assertEqual(select_pairs([1, 1, 1], 16), [])
        self.assertEqual(select_pairs([0, 1, 0.5], 2), [(1, 0), (1, 2)])
        with patch(
            "single_turn.baseline.direct_agent.run.load_single_turn_datasets",
            return_value=([row()], [row()]),
        ):
            with self.assertRaises(ValueError):
                load_data(Settings(train_samples=1, eval_samples=1))

    def test_token_alignment_and_mask(self):
        model = tiny().eval()
        prompt = torch.tensor([1, 3, 4])
        for completion in (torch.tensor([7, 8, 2]), torch.tensor([2])):
            logits = model(
                torch.cat([prompt, completion[:-1]])[None], use_cache=False
            ).logits[0, prompt.numel() - 1 :]
            expected = -F.cross_entropy(logits, completion, reduction="none")
            torch.testing.assert_close(token_logps(model, prompt, completion), expected)
            mask = torch.arange(completion.numel()) % 2 == 0
            torch.testing.assert_close(
                token_logps(model, prompt, completion, mask), expected[mask]
            )
        with self.assertRaises(ValueError):
            token_logps(model, prompt, completion, torch.zeros_like(completion))

    def test_streamed_dpo_exact_gradient(self):
        model = tiny()
        full = copy.deepcopy(model)
        p, w, l, mask = (
            torch.tensor([1, 3]),
            torch.tensor([7, 2]),
            torch.tensor([8, 2]),
            torch.tensor([True, False]),
        )
        delta = (
            token_logps(full, p, w, mask).sum() - token_logps(full, p, l, mask).sum()
        )
        dpo_loss(delta, 0.3, 0.1).backward()
        with torch.no_grad():
            value = float(
                token_logps(model, p, w, mask).sum()
                - token_logps(model, p, l, mask).sum()
            )
        c = dpo_delta_gradient(value, 0.3, 0.1)
        (c * token_logps(model, p, w, mask).sum()).backward()
        (-c * token_logps(model, p, l, mask).sum()).backward()
        for a, b in zip(model.parameters(), full.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6)
        torch.testing.assert_close(normalized(torch.ones(4)), torch.zeros(4))
        self.assertAlmostEqual(
            float(dpo_loss(torch.tensor(0.3), 0.3, 0.1)), math.log(2), places=6
        )

    def test_streamed_reward_model_gradient(self):
        model = ScalarModel(tiny()).train()
        full = copy.deepcopy(model)
        w, l = torch.tensor([1, 7, 2]), torch.tensor([1, 8, 2])
        (-F.logsigmoid(full(w) - full(l))).backward()
        with torch.no_grad():
            c = -torch.sigmoid(-(model(w) - model(l)))
        (c * model(w)).backward()
        (-c * model(l)).backward()
        for a, b in zip(model.parameters(), full.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6)

    def test_real_grammar_batched_generation(self):
        cfg = Settings(max_new_tokens=500, max_value_tokens=2, generation_batch_size=2)
        out = generate(
            tiny(128, 1024),
            CharTokenizer(),
            torch.tensor([1, 3]),
            3,
            cfg,
            row={"days": 1},
        )
        self.assertEqual(len(out), 3)
        for tokens, text, mask in out:
            data = json.loads(text)
            self.assertEqual(
                [(a["day"], a["field"]) for a in data["assignments"]], slots(1)
            )
            self.assertEqual(len(tokens), len(mask))
            self.assertTrue(mask.any())
            self.assertFalse(mask.all())
            self.assertTrue(text.startswith(PREFILL))

    def test_fused_boundary_token_not_echoed_inside_value(self):
        cfg = Settings(
            max_new_tokens=500, max_value_tokens=2, generation_batch_size=2, top_k=1
        )
        model = tiny(131, 1024)
        model.lm_head.bias = torch.nn.Parameter(torch.zeros(131))
        with torch.no_grad():
            model.lm_head.bias[128:131] = torch.tensor([100.0, 101.0, 102.0])
        result = generate(
            model, FusedTokenizer(), torch.tensor([1, 3]), 2, cfg, row={"days": 1}
        )
        for tokens, text, mask in result:
            assignments = json.loads(text)["assignments"]
            self.assertEqual([item["value"] for item in assignments], ["Beta"] * 7)
            self.assertEqual(int(mask.sum()), 7)
            self.assertEqual(tokens[-1].item(), 130)

    def test_grammar_cache_does_not_change_team_decoder(self):
        from single_turn.structured_generation import FixedSlotJSONLogitsProcessor
        from single_turn.baseline.direct_agent.grammar import DirectJSONGrammar

        tok = FusedTokenizer()
        direct = DirectJSONGrammar(tok, prompt_length=1, slots=slots(1))
        team = FixedSlotJSONLogitsProcessor(tok, prompt_length=1, slots=slots(1))
        direct._prepare_vocab(131)
        team._prepare_vocab(131)
        self.assertTrue(direct._value_masks_cpu[(0, False)][129])
        self.assertFalse(team._value_masks_cpu[(0, False)][129])

    def test_mixed_prompt_lengths_batched_iac(self):
        cfg = Settings(max_new_tokens=1000, max_value_tokens=2, top_k=1)
        model = tiny(131, 2048)
        model.lm_head.bias = torch.nn.Parameter(torch.zeros(131))
        with torch.no_grad():
            model.lm_head.bias[128:131] = torch.tensor([100.0, 101.0, 102.0])
        prompts = [torch.tensor([1, 3]), torch.tensor([1, 4, 5, 6])]
        result = generate_prompt_batch(
            model, FusedTokenizer(), prompts, cfg, rows=[{"days": 1}, {"days": 2}]
        )
        for i, (tokens, text, mask) in enumerate(result):
            assignments = json.loads(text)["assignments"]
            self.assertEqual(
                [(a["day"], a["field"]) for a in assignments], slots(i + 1)
            )
            self.assertEqual(
                [a["value"] for a in assignments], ["Beta"] * (7 * (i + 1))
            )
            self.assertEqual(int(mask.sum()), 7 * (i + 1))

    def test_left_padding_preserves_greedy_answers(self):
        cfg = Settings(max_new_tokens=500, max_value_tokens=2, top_k=1)
        model, tok = tiny(128, 1024), CharTokenizer()
        prompts = [torch.tensor([1, 3]), torch.tensor([1, 4, 5, 6])]
        rows = [{"days": 1}, {"days": 1}]
        batch = generate_prompt_batch(model, tok, prompts, cfg, rows=rows)
        for i, (prompt, r) in enumerate(zip(prompts, rows)):
            single = generate(model, tok, prompt, 1, cfg, row=r)[0]
            torch.testing.assert_close(batch[i][0], single[0])
            self.assertEqual(batch[i][1], single[1])
            torch.testing.assert_close(batch[i][2], single[2])

    def test_all_five_entrypaths(self):
        for algorithm in ALGORITHMS:
            with self.subTest(
                algorithm=algorithm
            ), tempfile.TemporaryDirectory() as directory:
                cfg = Settings(
                    algorithm=algorithm,
                    train_samples=4,
                    eval_samples=2,
                    max_new_tokens=8,
                    periodic_eval_samples=1,
                    eval_every_responses=4,
                    online_epochs=1,
                    iac_epochs=1,
                    dpo_epochs=1,
                    grpo_generations=2,
                    preference_candidates=4,
                    preference_pairs=2,
                    prompt_batch_size=2,
                    preference_batch_size=2,
                    actor_device="cpu",
                    reward_device="cpu",
                    output_dir=directory,
                    wandb_enabled=False,
                    save_final_model=False,
                )
                train, ev = [row(i) for i in range(4)], [row(10), row(11)]
                manifest = {
                    "train_sha256": fingerprint(train),
                    "eval_sha256": fingerprint(ev),
                }
                model = tiny()
                before = model.model.embed_tokens.weight.detach().clone()
                ex = Experiment(cfg, train, ev, manifest, Tokenizer(), model)
                with patch(
                    "single_turn.baseline.direct_agent.run.generate", synthetic_generate
                ), patch(
                    "single_turn.baseline.direct_agent.run.generate_prompt_batch",
                    synthetic_batch,
                ), patch(
                    "single_turn.baseline.direct_agent.run.load_model",
                    side_effect=lambda *args: tiny(),
                ):
                    random.seed(100)
                    torch.manual_seed(100)
                    py, tr = random.getstate(), torch.get_rng_state().clone()
                    ex.evaluate("rng", 2)
                    self.assertEqual(py, random.getstate())
                    self.assertTrue(torch.equal(tr, torch.get_rng_state()))
                    result = ex.run()
                self.assertEqual(result["status"], "finished")
                self.assertEqual(
                    torch.equal(before, model.model.embed_tokens.weight),
                    algorithm == "raw",
                )
                if algorithm == "dpo":
                    self.assertEqual(result["dpo_presentations"], 16)
                if algorithm in ("dpo", "rlhf"):
                    self.assertEqual(result["preference_responses"], 16)

    def test_aggregation_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for alg in ALGORITHMS:
                for seed in (42, 43):
                    p = Path(directory) / f"{alg}-{seed}"
                    p.mkdir()
                    paths.append(p)
                    (p / "completed.json").write_text(
                        json.dumps(
                            {"status": "finished", "algorithm": alg, "seed": seed}
                        )
                    )
                    (p / "protocol.json").write_text(
                        json.dumps({"settings": {"eval_samples": 20}})
                    )
                    (p / "eval_final.json").write_text(
                        json.dumps(
                            {
                                "phase": "final",
                                "algorithm": alg,
                                "seed": seed,
                                "count": 20,
                                "protocol_sha256": "same",
                                "metrics": {eval_key(m): 0.5 for m in METRICS},
                            }
                        )
                    )
            self.assertEqual(
                aggregate(paths)["results"]["dpo"]["final_pass"]["mean"], 0.5
            )
            with self.assertRaises(ValueError):
                aggregate(paths[:-1])
            with self.assertRaises(ValueError):
                aggregate(paths + paths[:1])


if __name__ == "__main__":
    unittest.main()
