"""Train MAGRPO on one-shot, role-guided TravelPlanner collaboration."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parent
DEFAULT_CONFIG = TASK_ROOT / "configs" / "single_turn_magrpo_config.yaml"
COMLRL_ROOT = REPO_ROOT.parent / "CoMLRL"
for path in (COMLRL_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from single_turn.aggregation import owned_slots
from single_turn.config import Config, add_config_args, parse_overrides
from single_turn.data import load_single_turn_datasets
from single_turn.formatting import get_single_turn_formatters
from single_turn.logger import (
    aggregate_single_turn_metrics,
    build_single_turn_eval_logger,
)
from single_turn.rewards import make_reward
from single_turn.rewards.reference_evaluator import parse_reference_catalog
from single_turn.rewards.single_turn_reward import score_single_turn_response
from single_turn.structured_generation import DEFAULT_SYSTEM_PROMPT


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _resolve_dataset_name(dataset_name: str) -> str:
    path = Path(dataset_name)
    if path.is_absolute() or path.exists():
        return str(path)
    repo_path = REPO_ROOT / dataset_name
    return str(repo_path) if repo_path.exists() else dataset_name


def _agent_names(config: Config) -> Optional[List[str]]:
    raw = config.get("agents")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or not all(isinstance(x, str) for x in raw):
        raise ValueError("agents must be a list of model names.")
    return [str(name) for name in raw]


def _load_data(config: Config):
    return load_single_turn_datasets(
        _resolve_dataset_name(str(config.get("dataset.name"))),
        config_name=str(config.get("dataset.config_name", "validation")),
        split=str(config.get("dataset.source_split", "validation")),
        train_samples=int(config.get("dataset.train_samples", 16)),
        eval_samples=int(config.get("dataset.eval_samples", 4)),
        seed=int(config.get("seed", 42)),
        days=config.get("dataset.filters.days"),
        levels=config.get("dataset.filters.levels"),
        visiting_city_numbers=config.get("dataset.filters.visiting_city_numbers"),
        max_reference_chars=config.get("dataset.filters.max_reference_chars"),
        select_shortest=config.get("dataset.filters.select_shortest"),
        stratify_by=config.get("dataset.partition.stratify_by"),
        interleave_eval=_bool(
            config.get("dataset.partition.interleave_eval"), default=False
        ),
        revision=config.get("dataset.revision"),
    )


def _build_reward_processor(config: Config):
    if not _bool(config.get("reward_processor.enabled", False)):
        return None
    from comlrl.utils.reward_processor import RewardProcessors

    processor = RewardProcessors.scale(
        factor=float(config.get("reward_processor.scale_factor", 1.0))
    )
    shift = config.get("reward_processor.shift")
    if shift is None:
        return processor
    shift_processor = RewardProcessors.shift(value=float(shift))
    return lambda value: shift_processor(processor(value))


def _wandb_config(config: Config, output_dir: str, trainer_cfg: Dict[str, Any]):
    section = config.get_section("wandb")
    if not _bool(section.get("enabled", True), default=True):
        return None
    return {
        "project": section.get("project", "travelplanner-collab"),
        "entity": section.get("entity", "OpenMLRL"),
        "name": section.get("name", "single_turn_magrpo"),
        "dir": section.get("dir", output_dir),
        "tags": section.get(
            "tags", ["magrpo", "travelplanner", "single-turn", "decentralized"]
        ),
        "config_sections": {
            "dataset": config.get_section("dataset"),
            "agent_model": config.get_section("agent_model"),
            "travel": config.get_section("travel"),
            "travel_reward": config.get_section("travel_reward"),
            "output": config.get_section("output"),
            "trainer": trainer_cfg,
        },
    }


def _contract_probe_completions(batch_item: Dict[str, Any]) -> List[str]:
    """Build strict all-dash actions to verify plumbing without a target plan."""

    days = int(batch_item["days"])
    return [
        json.dumps(
            {
                "agent_id": agent_idx,
                "assignments": [
                    {"day": day, "field": field, "value": "-"}
                    for day, field in sorted(owned_slots(agent_idx, days))
                ],
            },
            ensure_ascii=False,
        )
        for agent_idx in range(2)
    ]


def _curriculum_plan(
    config: Config, train_rows: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Resolve and validate the Travel-only short-trip curriculum."""

    magrpo = config.get_section("magrpo")
    epochs = int(magrpo.get("num_train_epochs", 1))
    short_epochs = int(config.get("travel.curriculum_short_epochs", 0))
    raw_days = config.get("travel.curriculum_days", [3])
    if not isinstance(raw_days, (list, tuple)) or not raw_days:
        raise ValueError("travel.curriculum_days must be a non-empty list.")
    curriculum_days = sorted({int(day) for day in raw_days})
    if short_epochs < 0:
        raise ValueError("travel.curriculum_short_epochs must be non-negative.")
    if short_epochs and short_epochs >= epochs:
        raise ValueError(
            "travel.curriculum_short_epochs must be smaller than "
            "magrpo.num_train_epochs."
        )

    short_rows = [
        row for row in train_rows if int(row.get("days", 0)) in curriculum_days
    ]
    if short_epochs and not short_rows:
        raise ValueError("The configured curriculum_days select no training rows.")
    full_ids = [str(row.get("source_index", row.get("id"))) for row in train_rows]
    if len(full_ids) != len(set(full_ids)):
        raise ValueError("The full Travel training split contains duplicate rows.")

    num_generations = int(magrpo.get("num_generations", 4))
    if num_generations < 2:
        raise ValueError("magrpo.num_generations must be at least 2 for MAGRPO.")
    full_epochs = epochs - short_epochs
    prompt_exposures = short_epochs * len(short_rows) + full_epochs * len(train_rows)
    buffer_size = int(magrpo.get("rollout_buffer_size", 1))
    if buffer_size < 1:
        raise ValueError("magrpo.rollout_buffer_size must be positive.")
    optimizer_updates = short_epochs * math.ceil(
        len(short_rows) / buffer_size
    ) + full_epochs * math.ceil(len(train_rows) / buffer_size)
    return {
        "short_rows": short_rows,
        "short_days": curriculum_days,
        "short_epochs": short_epochs,
        "full_epochs": full_epochs,
        "short_prompt_exposures": short_epochs * len(short_rows),
        "full_prompt_exposures": full_epochs * len(train_rows),
        "prompt_exposures": prompt_exposures,
        "expected_env_steps": prompt_exposures * num_generations,
        "optimizer_updates_per_agent": optimizer_updates,
    }


def _dry_run(
    config: Config,
    train_rows: Sequence[Dict[str, Any]],
    eval_rows,
    curriculum: Dict[str, Any],
) -> None:
    magrpo = config.get_section("magrpo")
    reward_cfg = config.get_section("travel_reward")
    sample = train_rows[0]
    completions = _contract_probe_completions(sample)
    reward, details = score_single_turn_response(
        completions,
        batch_item=sample,
        config=make_reward(reward_cfg).config,
    )
    num_generations = int(magrpo.get("num_generations", 4))
    epochs = int(magrpo.get("num_train_epochs", 1))
    expected_env_steps = int(curriculum["expected_env_steps"])
    formatters = get_single_turn_formatters(
        num_agents=int(magrpo.get("num_agents", 2)),
        role_mode=str(config.get("travel.role_mode", "partitioned_roles")),
        force_json_prefix=_bool(
            config.get("travel.force_json_prefix", True), default=True
        ),
    )
    all_rows = [*train_rows, *eval_rows]
    # Also validate that all held-out prompts can be indexed unambiguously by
    # the detailed evaluator before any model weights are loaded.
    build_single_turn_eval_logger(eval_rows)
    catalog_successes = [
        parse_reference_catalog(str(row.get("reference_information", ""))).parse_success
        for row in all_rows
    ]
    report = {
        "status": "ok",
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "epochs": epochs,
        "aligned_generations": num_generations,
        "expected_env_steps": expected_env_steps,
        "expected_agent_completions": expected_env_steps
        * int(magrpo.get("num_agents", 2)),
        "expected_optimizer_updates_per_agent": curriculum[
            "optimizer_updates_per_agent"
        ],
        "curriculum": {
            "short_days": curriculum["short_days"],
            "short_rows": len(curriculum["short_rows"]),
            "short_epochs": curriculum["short_epochs"],
            "short_env_steps": curriculum["short_prompt_exposures"]
            * num_generations,
            "full_rows": len(train_rows),
            "full_epochs": curriculum["full_epochs"],
            "full_env_steps": curriculum["full_prompt_exposures"]
            * num_generations,
        },
        "eval_at_end": _bool(magrpo.get("eval_at_end", True), default=True),
        "greedy_eval": _bool(config.get("travel.greedy_eval", True), default=True),
        "periodic_eval_samples": int(magrpo.get("eval_num_samples", 4)),
        "final_eval_samples": int(magrpo.get("final_eval_num_samples", len(eval_rows))),
        "rotate_eval_subset": _bool(
            magrpo.get("rotate_eval_subset", False), default=False
        ),
        "reward_backend": details["reward_backend"],
        "reward_range": list(make_reward(reward_cfg).reward_range),
        "all_dash_contract_probe_reward": reward,
        "all_dash_contract_probe_team_action_valid": details["team_action_valid"],
        "all_dash_contract_probe_required_grounded_recall": details[
            "required_grounded_recall"
        ],
        "all_dash_contract_probe_required_cooperative_contribution": details[
            "required_cooperative_contribution"
        ],
        "sample_reference_catalog_counts": details["reference_catalog_counts"],
        "reference_catalog_success_rows": sum(catalog_successes),
        "reference_catalog_total_rows": len(catalog_successes),
        "reference_char_range": [
            min(int(row["reference_chars"]) for row in all_rows),
            max(int(row["reference_chars"]) for row in all_rows),
        ],
        "source_split": sample["source_split"],
        "train_source_indices": [row["source_index"] for row in train_rows],
        "eval_source_indices": [row["source_index"] for row in eval_rows],
        "sample_days": sample["days"],
        "sample_prompt_chars": [len(formatter(sample)) for formatter in formatters],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MAGRPO on single-turn TravelPlanner collaboration."
    )
    add_config_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and verify prompts/reward without loading an LLM.",
    )
    args = parser.parse_args()
    config = Config(args.config or str(DEFAULT_CONFIG))
    if args.override:
        config.update(parse_overrides(args.override))

    train_rows, eval_rows = _load_data(config)
    curriculum = _curriculum_plan(config, train_rows)
    catalog_failures = [
        str(row.get("id", row.get("source_index", "unknown")))
        for row in [*train_rows, *eval_rows]
        if not parse_reference_catalog(
            str(row.get("reference_information", ""))
        ).parse_success
    ]
    if catalog_failures:
        preview = ", ".join(catalog_failures[:5])
        raise ValueError(
            "Reference catalog parsing failed before model loading for "
            f"{len(catalog_failures)} Travel rows: {preview}"
        )
    magrpo_section = config.get_section("magrpo")
    num_agents = int(magrpo_section.get("num_agents", 2))
    if num_agents != 2:
        raise ValueError(
            "The initial role-guided TravelPlanner task requires 2 agents."
        )
    if int(magrpo_section.get("num_turns", 1)) != 1:
        raise ValueError("This entrypoint intentionally supports exactly one turn.")
    if str(magrpo_section.get("joint_mode", "aligned")).lower() not in {
        "align",
        "aligned",
    }:
        raise ValueError("Use joint_mode=aligned to match the BFCL rollout budget.")
    periodic_eval_samples = int(magrpo_section.get("eval_num_samples", 4))
    final_eval_samples = int(
        magrpo_section.get("final_eval_num_samples", len(eval_rows))
    )
    if not 1 <= periodic_eval_samples <= len(eval_rows):
        raise ValueError("magrpo.eval_num_samples must lie within the eval pool.")
    if not 1 <= final_eval_samples <= len(eval_rows):
        raise ValueError("magrpo.final_eval_num_samples must lie within the eval pool.")

    if args.dry_run:
        _dry_run(config, train_rows, eval_rows, curriculum)
        return

    import torch
    from transformers import AutoTokenizer

    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_config = config.get_agent_model_config()
    agent_names = _agent_names(config)
    tokenizer_source = agent_names[0] if agent_names else model_config.name
    tokenizers = (
        [AutoTokenizer.from_pretrained(name) for name in agent_names]
        if agent_names
        else [AutoTokenizer.from_pretrained(tokenizer_source)]
    )
    for tokenizer in tokenizers:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        padding_side = config.get("tokenizer.padding_side")
        if padding_side:
            tokenizer.padding_side = str(padding_side)
        if model_config.special_tokens:
            tokenizer.add_special_tokens(model_config.special_tokens)

    output_base = str(config.get("output.base_dir", "output_single_turn_magrpo"))
    job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(output_base, f"job_{job_id}")
    os.makedirs(output_dir, exist_ok=True)
    config.save(os.path.join(output_dir, "config.yaml"))

    from comlrl.trainers.reinforce import MAGRPOConfig
    from single_turn.train.structured_trainer import StructuredOutputMAGRPOTrainer

    trainer_args = MAGRPOConfig(
        num_agents=num_agents,
        num_turns=1,
        parallel_training=str(magrpo_section.get("parallel_training", "mp")),
        agent_devices=magrpo_section.get("agent_devices", ["cuda:0", "cuda:1"]),
        num_train_epochs=int(magrpo_section.get("num_train_epochs", 32)),
        agent_learning_rate=float(magrpo_section.get("agent_learning_rate", 2e-5)),
        logging_steps=int(magrpo_section.get("logging_steps", 20)),
        num_generations=int(magrpo_section.get("num_generations", 4)),
        max_new_tokens=int(magrpo_section.get("max_new_tokens", 1024)),
        temperature=float(model_config.temperature),
        top_p=float(model_config.top_p),
        top_k=model_config.top_k,
        discount=float(magrpo_section.get("discount", 1.0)),
        joint_mode="aligned",
        early_termination_threshold=magrpo_section.get(
            "early_termination_threshold", None
        ),
        rollout_buffer_size=int(magrpo_section.get("rollout_buffer_size", 4)),
        train_batch_size=int(magrpo_section.get("train_batch_size", 4)),
        advantage_normalization=_bool(
            magrpo_section.get("advantage_normalization", True), default=True
        ),
        advantage_mode=str(magrpo_section.get("advantage_mode", "mean")),
        eval_interval=int(magrpo_section.get("eval_interval", 40)),
        eval_num_samples=int(magrpo_section.get("eval_num_samples", 4)),
        eval_batch_size=int(magrpo_section.get("eval_batch_size", 1)),
        reference_kl_enabled=_bool(
            magrpo_section.get("reference_kl_enabled", False), default=False
        ),
        reference_kl_coef=float(magrpo_section.get("reference_kl_coef", 0.1)),
        reference_devices=magrpo_section.get("reference_devices"),
    )

    reward_config = config.get_section("travel_reward")
    reward = make_reward(reward_config)
    formatter_tokenizers = tokenizers if agent_names else tokenizers * num_agents
    use_chat_template = _bool(
        config.get("travel.use_chat_template", True), default=True
    )
    force_json_prefix = _bool(
        config.get("travel.force_json_prefix", True), default=True
    )
    formatters = get_single_turn_formatters(
        num_agents=num_agents,
        role_mode=str(config.get("travel.role_mode", "partitioned_roles")),
        force_json_prefix=force_json_prefix,
        tokenizers=formatter_tokenizers,
        use_chat_template=use_chat_template,
        system_prompt=config.get("travel.system_prompt", DEFAULT_SYSTEM_PROMPT),
    )
    trainer = StructuredOutputMAGRPOTrainer(
        agent_model=model_config.name if not agent_names else None,
        agents=agent_names,
        num_agents=num_agents,
        tokenizer=tokenizers if agent_names else tokenizers[0],
        model_config={
            "torch_dtype": model_config.torch_dtype,
            "attn_implementation": model_config.attn_implementation,
            "special_tokens": model_config.special_tokens,
        },
        train_dataset=train_rows,
        eval_dataset=eval_rows,
        dataset_type="travelplanner",
        reward_func=reward,
        reward_processor=_build_reward_processor(config),
        formatters=formatters,
        external_transition=None,
        wandb_config=_wandb_config(config, output_dir, magrpo_section),
        eval_logger=build_single_turn_eval_logger(
            eval_rows,
            reward_config=reward.config,
            panel_size=int(magrpo_section.get("eval_num_samples", 4)),
        ),
        eval_aggregator=aggregate_single_turn_metrics,
        args=trainer_args,
        chat_formatted_prompts=use_chat_template,
        force_json_prefix=force_json_prefix,
        stop_after_complete_json=_bool(
            config.get("travel.stop_after_complete_json", True), default=True
        ),
        rotate_eval_subset=_bool(
            magrpo_section.get("rotate_eval_subset", False), default=False
        ),
        greedy_eval=_bool(config.get("travel.greedy_eval", True), default=True),
        curriculum_train_dataset=(
            curriculum["short_rows"] if curriculum["short_epochs"] else None
        ),
        curriculum_short_epochs=int(curriculum["short_epochs"]),
    )
    if _bool(config.get("agent_model.gradient_checkpointing", True), default=True):
        for agent in trainer.agents:
            if not hasattr(agent, "gradient_checkpointing_enable"):
                continue
            try:
                agent.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                agent.gradient_checkpointing_enable()
    trainer.train()

    # The periodic schedule evaluates at the start of each epoch. Log one
    # additional evaluation at the actual final env step so the W&B curve
    # includes the fully trained policy rather than ending one epoch early.
    if _bool(magrpo_section.get("eval_at_end", True), default=True):
        trainer.evaluate(num_eval_samples=final_eval_samples)

    if _bool(config.get("output.save_final_model", False)):
        for agent_idx, agent in enumerate(trainer.agents):
            save_dir = os.path.join(output_dir, f"agent_{agent_idx}")
            agent.save_pretrained(save_dir)
            trainer.tokenizers[agent_idx].save_pretrained(save_dir)


if __name__ == "__main__":
    main()
