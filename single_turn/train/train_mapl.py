"""Single-GPU Travel entrypoints for CoMLRL's MAPL preference algorithms."""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT.parent / "CoMLRL", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from single_turn.config import Config, add_config_args, parse_overrides
from single_turn.train.train_magrpo import (
    _agent_names, _bool, _build_reward_processor, _contract_probe_completions,
    _load_data, _wandb_config,
)

ALGORITHMS = ("marlhf", "marlhf_iter", "madpo_iter")


def load_config(path, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"Circular configuration inheritance: {path}")
    config = Config(str(path))
    parent = config.data.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent, (*seen, path))
        base.update(config.data)
        config.data = base.data
    # Shared domain settings are inherited, but MAGRPO's schedule is not.
    config.data.pop("magrpo", None)
    return config


def make_trainer_args(config):
    from comlrl.trainers.preference import MADPOIterConfig, MARLHFConfig, MARLHFIterConfig

    classes = {"marlhf": MARLHFConfig, "marlhf_iter": MARLHFIterConfig, "madpo_iter": MADPOIterConfig}
    if config.get_section("magrpo"):
        raise ValueError("Use mapl.* overrides for preference trainers, not magrpo.*.")
    algorithm = config.get("algorithm")
    if algorithm not in classes:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}.")
    section = config.get_section("mapl")
    unknown = set(section) - {field.name for field in fields(classes[algorithm])}
    if unknown:
        raise ValueError(f"Unknown mapl settings: {sorted(unknown)}")
    model = config.get_agent_model_config()
    args = classes[algorithm](**{
        "temperature": model.temperature, "top_p": model.top_p, "top_k": model.top_k,
        **section,
    })
    if args.num_agents != 2 or args.num_turns != 1 or args.joint_mode != "aligned":
        raise ValueError("Travel MAPL requires two agents, one turn, and aligned joint actions.")
    if args.parallel_training != "none" or args.reference_kl_enabled:
        raise ValueError("Travel MAPL requires parallel_training=none and reference_kl_enabled=false.")
    if isinstance(args.agent_devices, (list, tuple)) and len(set(args.agent_devices)) != 1:
        raise ValueError("Sequential Travel MAPL requires all actors on the same device.")
    if getattr(args, "rl_algorithm", "magrpo") != "magrpo":
        raise ValueError("This Travel MARLHF adapter currently supports the MAGRPO backend only.")
    if getattr(args, "comparator_policy", "current") not in {"current", "current_copy"}:
        raise ValueError("Travel MAPL currently supports current/current_copy comparators only.")
    if getattr(args, "comparator_generation_mode", "decentralized") != "decentralized":
        raise ValueError("Travel MAPL requires a decentralized comparator.")
    if getattr(args, "preference_scoring_reward", "task") != "task":
        raise ValueError("Travel preference labels must come from the task reward.")
    if getattr(args, "log_reward_distribution", False):
        raise ValueError("Travel MAPL keeps log_reward_distribution=false; only eval scalars are uploaded.")
    if args.eval_batch_size != 1 or getattr(args, "reward_train_batch_size", 1) != 1:
        raise ValueError("Use eval_batch_size=1 and reward_train_batch_size=1 for long Travel contexts.")
    if config.get("travel.curriculum_short_epochs", 0) != 0:
        raise ValueError("Travel MAPL uses all training rows from the first iteration; set curriculum_short_epochs=0.")
    for flag in ("use_chat_template", "force_json_prefix", "constrain_json_skeleton", "stop_after_complete_json"):
        if not _bool(config.get(f"travel.{flag}", True)):
            raise ValueError(f"Travel MAPL requires travel.{flag}=true.")
    if config.get("travel.role_mode") != "partitioned_roles":
        raise ValueError("Travel MAPL requires partitioned_roles.")
    return args


def budget_report(config, args, train_count):
    iterations = int(getattr(args, "num_iterations", 1))
    iterative = config.get("algorithm").endswith("_iter")
    comparator = (getattr(args, "comparator_num_candidates", None) or args.preference_num_candidates) if iterative else 0
    report = {
        "preference_joint_rollouts": iterations * train_count * (args.preference_num_candidates + comparator),
        "preference_refreshes": iterations,
        "eval_uses": "v9 task reward, not the learned reward model",
    }
    if config.get("algorithm") == "madpo_iter":
        pair_limit = args.preference_pairs_per_sample
        report["pair_counted_env_steps_upper_bound"] = (
            iterations * args.num_train_epochs * train_count * pair_limit * args.environment_steps_per_pair
            if pair_limit is not None and args.pair_selection != "all" and not args.preference_replay_sample_size
            else None
        )
        report["caveat"] = "Ties reduce pairs; replay reuses data. Pair-counted env steps are not rollout counts."
    else:
        report["online_rl_joint_rollouts"] = iterations * args.num_train_epochs * train_count * args.num_generations
        report["caveat"] = "Preference collection and evaluation cost are additional to online RL rollouts."
    return report


def main(default_algorithm=None):
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--algorithm", choices=ALGORITHMS, default=default_algorithm)
    parser.add_argument("--dry-run", action="store_true", help="Validate data/reward/config without loading model weights.")
    cli = parser.parse_args()
    algorithm = cli.algorithm or "madpo_iter"
    path = cli.config or REPO_ROOT / "single_turn" / "configs" / f"single_turn_{algorithm}_config.yaml"
    config = load_config(path)
    config.update(parse_overrides(cli.override))
    if cli.algorithm and config.get("algorithm") != cli.algorithm:
        raise ValueError("--algorithm and config.algorithm disagree; select the matching configuration.")
    args = make_trainer_args(config)
    train_rows, eval_rows = _load_data(config)
    final_count = int(config.get("evaluation.final_num_samples", len(eval_rows)))
    if not train_rows or not 1 <= args.eval_num_samples <= len(eval_rows) or not 1 <= final_count <= len(eval_rows):
        raise ValueError("Training must be nonempty; periodic/final eval counts must fit the held-out pool.")
    from single_turn.rewards import make_reward
    reward = make_reward(config.get_section("travel_reward"))
    report = {
        "algorithm": config.get("algorithm"), "seed": config.get("seed"),
        "train_samples": len(train_rows), "eval_samples": len(eval_rows),
        "periodic_eval_samples": args.eval_num_samples, "agent_devices": args.agent_devices,
        "reward_range": [reward.config.min_reward, reward.config.max_reward],
        **budget_report(config, args, len(train_rows)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if cli.dry_run:
        # Verify the actual joint-action reward interface, without a gold plan.
        actions = _contract_probe_completions(train_rows[0])
        print("Contract probe:", reward([actions[0]], [actions[1]], batch_items=[train_rows[0]]))
        return

    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from single_turn.formatting import get_single_turn_formatters
    from single_turn.logger import aggregate_single_turn_metrics, build_single_turn_eval_logger
    from single_turn.train.preference_trainer import TravelMADPOIterTrainer, TravelMARLHFTrainer, TravelMARLHFIterTrainer

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = config.get_agent_model_config()
    names = _agent_names(config)
    if names is not None and len(names) != 2:
        raise ValueError("agents must list exactly two model sources.")
    tokenizers = [AutoTokenizer.from_pretrained(name) for name in (names or [model.name, model.name])]
    for tokenizer in tokenizers:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.special_tokens:
            tokenizer.add_special_tokens(model.special_tokens)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job = os.environ.get("SLURM_JOB_ID", "local")
    output = Path(config.get("output.base_dir")) / f"job_{job}_seed{seed}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    if hasattr(args, "preference_replay_dir") and args.preference_replay_dir is None:
        args.preference_replay_dir = str(output / "preference_replay")
    config.update({"mapl": vars(args), "output": {"run_dir": str(output)}})
    config.save(str(output / "config.yaml"))
    wandb_config = _wandb_config(config, str(output), vars(args))
    if wandb_config is not None:
        wandb_config["dir"] = config.get("wandb.dir") or str(output)
        wandb_config["output_dir"] = str(output)
        wandb_config["name"] = f"{wandb_config['name']}-seed{seed}-{job}"
    trainers = {"madpo_iter": TravelMADPOIterTrainer, "marlhf": TravelMARLHFTrainer, "marlhf_iter": TravelMARLHFIterTrainer}
    trainer = trainers[config.get("algorithm")](
        agent_model=model.name if names is None else None, agents=names, num_agents=2,
        tokenizer=tokenizers, model_config={"torch_dtype": model.torch_dtype,
            "attn_implementation": model.attn_implementation, "special_tokens": model.special_tokens},
        train_dataset=train_rows, eval_dataset=eval_rows, dataset_type="travelplanner",
        reward_func=reward, reward_processor=_build_reward_processor(config),
        formatters=get_single_turn_formatters(num_agents=2, role_mode="partitioned_roles",
            force_json_prefix=True, tokenizers=tokenizers, use_chat_template=True,
            system_prompt=config.get("travel.system_prompt")),
        wandb_config=wandb_config, args=args,
        eval_logger=build_single_turn_eval_logger(eval_rows, reward_config=reward.config, panel_size=args.eval_num_samples),
        eval_aggregator=aggregate_single_turn_metrics,
        chat_formatted_prompts=True, force_json_prefix=True, constrain_json_skeleton=True,
        max_value_tokens=int(config.get("travel.max_value_tokens", 32)),
        normalize_value_log_probs=_bool(config.get("travel.normalize_value_log_probs", True)),
        role_mode="partitioned_roles", stop_after_complete_json=True,
        rotate_eval_subset=False, greedy_eval=_bool(config.get("travel.greedy_eval", True)),
        curriculum_short_epochs=0,
        preference_generation_batch_size=int(config.get("travel.preference_generation_batch_size", 1)),
        offload_inactive_actors=_bool(config.get("travel.offload_inactive_actors", True)),
    )
    trainer.verbose = _bool(config.get("output.verbose", True))
    if _bool(config.get("agent_model.gradient_checkpointing", True)):
        for agent in trainer.agents:
            agent.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer.train()
    if _bool(config.get("evaluation.at_end", True)):
        trainer.evaluate(num_eval_samples=final_count)
    print(f"Finished: env_steps={trainer.env_step}, preference_joint_candidates={trainer.preference_joint_candidates}, preference_pairs={trainer.preference_pairs_generated}")
    if _bool(config.get("output.save_final_model", True)):
        for idx, agent in enumerate(trainer.agents):
            target = output / f"agent_{idx}"
            agent.save_pretrained(target)
            trainer.tokenizers[idx].save_pretrained(target)
    if trainer.wandb_initialized:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
