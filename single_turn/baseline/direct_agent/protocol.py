"""Shared prompts, task scores and experiment budgets for all five controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


ALGORITHMS = ("raw", "grpo", "iac", "rlhf", "dpo")
METRICS = (
    "reward",
    "parse_success",
    "plan_delivery",
    "plan_completion",
    "required_grounded_recall",
    "grounding_f1",
    "commonsense_micro",
    "commonsense_pass",
    "hard_micro",
    "hard_pass",
    "budget_pass",
    "final_pass",
)


def eval_key(metric: str) -> str:
    return f"eval/turn_1/{'reward_mean' if metric == 'reward' else metric}"


@dataclass
class Settings:
    algorithm: str = "grpo"
    seed: int = 42
    model: str = "Qwen/Qwen3-8B"
    model_revision: str | None = "b968826d9c46dd6066d109eabc6255188de91218"
    dataset: str = "osunlp/TravelPlanner"
    dataset_revision: str | None = "8736504ecfc31b7f8b7e40122873c337e83fff7c"
    data_seed: int = 42
    train_samples: int = 60
    eval_samples: int = 20
    max_new_tokens: int = 2048
    max_value_tokens: int = 32
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    generation_batch_size: int = 20
    prompt_batch_size: int = 4
    preference_candidates: int = 80
    preference_pairs: int = 16
    preference_batch_size: int = 4
    grpo_generations: int = 4
    online_epochs: int = 21
    iac_epochs: int = 84
    dpo_epochs: int = 3
    dpo_response_cap: int = 5040
    reward_epochs: int = 1
    actor_lr: float = 2e-5
    iac_lr: float = 5e-6
    reward_lr: float = 1e-5
    dpo_beta: float = 0.1
    ratio_clip: float = 0.2
    value_loss_coef: float = 0.6
    max_grad_norm: float = 1.0
    eval_every_responses: int = 480
    periodic_eval_samples: int = 4
    eval_seed: int = 1729
    actor_device: str = "cuda:0"
    reward_device: str = "cuda:1"
    output_dir: str = "output_travel_direct"
    wandb_enabled: bool = True
    wandb_entity: str = "OpenMLRL"
    wandb_project: str = "Travel"
    group: str = "travel-direct-qwen3-8b"
    save_final_model: bool = True

    def validate(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}")
        for key in (
            "train_samples",
            "eval_samples",
            "max_new_tokens",
            "generation_batch_size",
            "prompt_batch_size",
            "preference_candidates",
            "preference_pairs",
            "preference_batch_size",
            "online_epochs",
            "iac_epochs",
            "dpo_epochs",
            "reward_epochs",
            "eval_every_responses",
            "periodic_eval_samples",
            "max_value_tokens",
            "dpo_response_cap",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.preference_candidates < 2 or self.grpo_generations < 2:
            raise ValueError(
                "Preference and GRPO generation need at least two candidates"
            )
        if self.periodic_eval_samples > self.eval_samples:
            raise ValueError("periodic eval must fit the eval split")
        if not 0 < self.temperature or not 0 < self.top_p <= 1:
            raise ValueError("Sampling temperature and top_p must be valid")
        for key in (
            "actor_lr",
            "iac_lr",
            "reward_lr",
            "dpo_beta",
            "value_loss_coef",
            "max_grad_norm",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not 0 < self.ratio_clip < 1 or self.top_k < 1:
            raise ValueError("ratio_clip must be in (0,1) and top_k must be positive")

    def budget(self) -> dict[str, Any]:
        online = (
            self.train_samples
            * (
                self.iac_epochs
                if self.algorithm == "iac"
                else self.online_epochs * self.grpo_generations
            )
            if self.algorithm in ("grpo", "iac", "rlhf")
            else 0
        )
        preference = (
            self.train_samples * self.preference_candidates
            if self.algorithm in ("dpo", "rlhf")
            else 0
        )
        return {
            "online_responses": online,
            "preference_responses": preference,
            "preference_pair_cap": (
                self.train_samples * self.preference_pairs if preference else 0
            ),
            "dpo_response_presentations_cap": (
                min(
                    self.dpo_response_cap,
                    2 * self.train_samples * self.preference_pairs * self.dpo_epochs,
                )
                if self.algorithm == "dpo"
                else 0
            ),
            "not_equal_compute": "DPO response presentations reuse offline data; preference collection and eval cost extra.",
        }


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def select_pairs(rewards: list[float], limit: int) -> list[tuple[int, int]]:
    pairs = [
        (rewards[i] - rewards[j], i, j)
        for i in range(len(rewards))
        for j in range(len(rewards))
        if rewards[i] > rewards[j]
    ]
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [(i, j) for _, i, j in pairs[:limit]]


def slots(days: int):
    from single_turn.aggregation import PLAN_FIELDS

    return [(day, field) for day in range(1, days + 1) for field in PLAN_FIELDS]


PREFILL = '{"assignments": [{"day": 1, "field": "current_city", "value": "'


def messages(row: dict) -> list[dict[str, str]]:
    from single_turn.formatting import (
        _render_logistics_catalog,
        _render_experience_catalog,
        _route_scaffold,
    )
    from single_turn.rewards.reference_evaluator import (
        parse_reference_catalog,
        derive_route_scaffold,
    )

    catalog = parse_reference_catalog(str(row.get("reference_information", "")))
    if not catalog.parse_success:
        raise ValueError(f"Invalid reference catalog for {row.get('id')}")
    scaffold = derive_route_scaffold(row, catalog)
    catalogs = "\n\n".join(
        [
            _render_logistics_catalog(catalog, example=row, scaffold=scaffold),
            _render_experience_catalog(catalog, example=row, scaffold=scaffold),
        ]
    ).replace("team_cost", "party_cost")
    observation = {
        key: row.get(key)
        for key in (
            "query",
            "org",
            "dest",
            "days",
            "dates",
            "people_number",
            "budget",
            "local_constraint",
        )
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a travel planner. Complete the entire itinerary yourself in one JSON response. "
                "Choose all transportation, lodging, meals and attractions. Output no explanation or reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                "TRIP REQUEST\n"
                + json.dumps(observation, ensure_ascii=False)
                + "\n\nREFERENCE-DERIVED ROUTE\n"
                + "\n".join(_route_scaffold(row, catalog))
                + "\n\nAVAILABLE OPTIONS\n"
                + catalogs
                + "\n\nPLAN RULES\n"
                "Use the exact route current_city values above. On move days select a matching, dated "
                "eligible transport; use '-' for breakfast, attraction, lunch and dinner. On stay days "
                "transportation is '-', and all three meals plus exactly one attraction are required. "
                "Use a different restaurant for EVERY meal and a different attraction for each stay day. "
                "Lodging is required at each day's destination except the final day, where it is '-'. "
                "Choose eligible=yes options whose days include the selected day. Reuse lodging through "
                "a consecutive city stay to satisfy minimum nights. Respect every room type and house rule. "
                "Never mix self-driving with flight or taxi. Across meals cover every requested cuisine. "
                "Sum all party transport costs, all lodging nights and all meals; total must fit the budget. "
                "Catalog prices already include the party. Copy quoted option values exactly.\n\n"
                'OUTPUT CONTRACT\nReturn {"assignments": [{"day": integer, "field": string, '
                '"value": string}, ...]}. Include every slot once, in this order: '
                + json.dumps(slots(int(row["days"])))
                + ". The decoder supplies the JSON keys and slot order; you choose values only. "
                "The assistant prefix is already supplied; continue inside the first value string."
            ),
        },
    ]


def format_prompt(tokenizer, row: dict) -> str:
    return (
        tokenizer.apply_chat_template(
            messages(row),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + PREFILL
    )


def score(text: str, row: dict) -> dict[str, float]:
    """Score ONE plan directly: no two-role projection, merge or cooperation bonus."""
    from single_turn.aggregation import PLAN_FIELDS, MergeResult
    from single_turn.rewards.reference_evaluator import evaluate_reference_plan

    expected = slots(int(row["days"]))
    valid, values = True, {}
    try:
        data = json.loads(text)
        if not isinstance(data, dict) or set(data) != {"assignments"}:
            raise ValueError("Invalid envelope")
        assignments = data["assignments"]
        if not isinstance(assignments, list) or len(assignments) != len(expected):
            raise ValueError("Missing/extra assignments")
        for item, slot in zip(assignments, expected):
            if (
                not isinstance(item, dict)
                or set(item) != {"day", "field", "value"}
                or type(item["day"]) is not int
                or (item["day"], item["field"]) != slot
                or not isinstance(item["value"], str)
                or not item["value"].strip()
            ):
                raise ValueError("Invalid slot/value or duplicate")
            values[slot] = item["value"]
    except (TypeError, ValueError, KeyError):
        valid, values = False, {}
    plan = [
        dict(day=day, **{f: values.get((day, f), "-") for f in PLAN_FIELDS})
        for day in range(1, int(row["days"]) + 1)
    ]
    # This is only the evaluator's input container; there are no role actions.
    container = MergeResult(
        plan=plan,
        parsed=[],
        merged_assignments=values,
        per_agent_assignments=[],
        conflict_slots=set(),
        overlap_count=0,
        self_duplicate_count=0,
        invalid_slot_count=int(not valid),
        extra_assignment_count=0,
        capacity=len(expected),
    )
    d = evaluate_reference_plan(container, batch_item=row)
    recall = d["required_grounded_recall"]
    support = 0.10 + 0.90 * recall
    quality = (
        0.20 * support * d["assignment_coverage"]
        + 0.25 * recall
        + 0.15 * d["grounding_f1"]
        + support * (0.25 * d["commonsense_soft"] + 0.15 * d["hard_constraint_soft"])
    )
    result = {
        "reward": min(
            1.0, max(0.0, 0.90 * quality + 0.10 * d["ultimate/reference_plan_success"])
        ),
        "parse_success": 1.0,
        "plan_delivery": d["ultimate/reference_plan_delivery"],
        # The reference helper's completion diagnostic assumes two roles. Replace
        # only that diagnostic with the corresponding single-plan condition.
        "plan_completion": float(d["required_fill_rate"] >= 1.0 - 1e-9),
        "required_grounded_recall": recall,
        "grounding_f1": d["grounding_f1"],
        "commonsense_micro": d["ultimate/reference_commonsense_micro"],
        "commonsense_pass": d["ultimate/reference_commonsense_macro"],
        "hard_micro": d["ultimate/reference_hard_micro"],
        "hard_pass": d["ultimate/reference_hard_macro"],
        "budget_pass": d["ultimate/reference_budget_pass"],
        "final_pass": d["ultimate/reference_plan_success"],
    }
    result = {k: float(v) * float(valid) for k, v in result.items()}
    for kind in ("commonsense", "hard"):
        result[f"_{kind}_passed"] = d[f"_aggregate/{kind}_pass_count"] * float(valid)
        result[f"_{kind}_total"] = d[f"_aggregate/{kind}_applicable_count"]
    return result


def aggregate_metrics(records):
    if not records:
        raise ValueError("Cannot aggregate an empty evaluation")
    result = {eval_key(k): sum(r[k] for r in records) / len(records) for k in METRICS}
    for kind in ("commonsense", "hard"):
        result[eval_key(f"{kind}_micro")] = sum(
            r[f"_{kind}_passed"] for r in records
        ) / max(1.0, sum(r[f"_{kind}_total"] for r in records))
    return result


def protocol_metadata(settings: Settings) -> dict:
    return {
        "settings": asdict(settings),
        "protocol": "travel_direct_v1",
        "reward_formula": "0.90 * existing_plan_quality_Q + 0.10 * reference_final_pass; invalid JSON = 0",
        "reward_range": [0.0, 1.0],
        "num_actors": 1,
        "enable_thinking": False,
        "score_note": "Local reference-based Travel checks, not official benchmark results. No annotated plan or cooperation terms.",
        "eval_note": "Charts use fixed first 4 anchors; initial/final full 20 records and metrics are local artifacts + summary.",
        "loss_note": "Value tokens only; unwarped model log probabilities, as in the Travel team trainer. No KL.",
        "budget": settings.budget(),
    }
