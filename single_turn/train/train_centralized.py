"""Train one joint Travel actor (and a joint comparator for iterative MAPL).

This is an independent opt-in entrypoint. The decentralized entrypoints and
their defaults do not import this module or the centralized trainers.
"""

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
    _agent_names, _bool, _build_reward_processor, _curriculum_plan, _load_data,
    _wandb_config,
)
from single_turn.train.train_mapl import budget_report, make_trainer_args as mapl_args

# The shared helpers above locate the conventional sibling CoMLRL checkout.
# An explicit isolated core checkout takes precedence for this NEW entrypoint
# only; existing entrypoints keep their original dependency resolution.
if os.environ.get("TRAVEL_COMLRL_ROOT"):
    core_root = Path(os.environ["TRAVEL_COMLRL_ROOT"]).expanduser().resolve()
    if not (core_root / "comlrl").is_dir():
        raise ValueError("TRAVEL_COMLRL_ROOT must contain the comlrl package.")
    sys.path.insert(0, str(core_root))

ALGORITHMS = ("magrpo", "madpo", "madpo_iter", "marlhf", "marlhf_iter")
MAGRPO_EVAL_SETTINGS = {"eval_at_end", "final_eval_num_samples", "rotate_eval_subset"}


def load_config(path, seen=()):
    """Keep MAGRPO's section until the selected algorithm is known."""
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"Circular configuration inheritance: {path}")
    config = Config(str(path))
    parent = config.data.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent, (*seen, path))
        base.update(config.data)
        config.data = base.data
    return config


def make_trainer_args(config):
    try:
        from comlrl.trainers.reinforce import CentralizedMAGRPOConfig
        from comlrl.trainers.preference import MADPOConfig
        if "collaboration_mode" not in {field.name for field in fields(MADPOConfig)}:
            raise ImportError("CoMLRL is missing centralized preference actors")
    except ImportError as exc:
        raise RuntimeError(
            "The centralized entrypoint requires CoMLRL's centralized actor APIs "
            "(cen commit 998ea13 or a revision containing it). Use a separate "
            "CoMLRL checkout/environment; do not switch a running job's source."
        ) from exc
    algorithm = config.get("algorithm")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}.")
    section_name = "magrpo" if algorithm == "magrpo" else "mapl"
    section = config.get_section(section_name)
    if section.get("collaboration_mode") != "centralized":
        raise ValueError("Use a centralized config with collaboration_mode=centralized.")
    names = _agent_names(config)
    if names is not None and len(names) != 1:
        raise ValueError("Centralized Travel accepts exactly one actor model source.")
    devices = section.get("agent_devices")
    if not (isinstance(devices, str) or isinstance(devices, (list, tuple)) and len(devices) == 1):
        raise ValueError("Centralized Travel requires exactly one explicit actor device.")
    if algorithm.endswith("_iter"):
        if section.get("comparator_generation_mode") != "centralized":
            raise ValueError("Centralized collaboration requires a centralized comparator.")
        devices = section.get("comparator_devices")
        if devices is not None and not (isinstance(devices, str) or isinstance(devices, (list, tuple)) and len(devices) == 1):
            raise ValueError("Centralized Travel requires at most one comparator device.")
    if algorithm != "magrpo":
        # Drop only the inherited MAGRPO schedule, leaving domain/data/reward intact.
        config.data.pop("magrpo", None)
        return mapl_args(config)

    if config.get_section("mapl"):
        raise ValueError("Use magrpo.* settings for centralized MAGRPO, not mapl.*.")
    field_names = {field.name for field in fields(CentralizedMAGRPOConfig)}
    unknown = set(section) - field_names - MAGRPO_EVAL_SETTINGS
    if unknown:
        raise ValueError(f"Unknown magrpo settings: {sorted(unknown)}")
    model = config.get_agent_model_config()
    args = CentralizedMAGRPOConfig(**{
        "temperature": model.temperature, "top_p": model.top_p, "top_k": model.top_k,
        **{key: value for key, value in section.items() if key in field_names},
    })
    if args.num_agents != 2 or args.num_turns != 1 or args.joint_mode != "aligned":
        raise ValueError("Centralized Travel requires two logical roles, one turn, aligned actions.")
    if args.parallel_training != "none" or args.reference_kl_enabled or args.eval_batch_size != 1:
        raise ValueError("Use parallel_training=none, reference_kl_enabled=false, eval_batch_size=1.")
    for flag in ("use_chat_template", "force_json_prefix", "constrain_json_skeleton", "stop_after_complete_json"):
        if not _bool(config.get(f"travel.{flag}", True)):
            raise ValueError(f"Centralized Travel requires travel.{flag}=true.")
    if config.get("travel.role_mode") != "partitioned_roles":
        raise ValueError("Keep partitioned_roles for the unchanged reward/eval boundary.")
    return args


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--algorithm", choices=ALGORITHMS)
    parser.add_argument("--dry-run", action="store_true", help="Check data, joint contract and reward without loading weights.")
    cli = parser.parse_args()
    algorithm = cli.algorithm or "marlhf_iter"
    config = load_config(cli.config or REPO_ROOT / "single_turn/configs" / f"single_turn_centralized_{algorithm}_config.yaml")
    overrides = parse_overrides(cli.override)
    config.update(overrides)
    if cli.algorithm and config.get("algorithm") != cli.algorithm:
        raise ValueError("--algorithm and config.algorithm disagree.")
    wrong_section = "mapl" if config.get("algorithm") == "magrpo" else "magrpo"
    if wrong_section in overrides:
        raise ValueError(f"Do not pass {wrong_section}.* overrides for {config.get('algorithm')}.")
    args = make_trainer_args(config)
    algorithm = config.get("algorithm")
    magrpo = algorithm == "magrpo"
    section_name = "magrpo" if magrpo else "mapl"
    train_rows, eval_rows = _load_data(config)
    final_count = int(config.get(
        "magrpo.final_eval_num_samples" if magrpo else "evaluation.final_num_samples", len(eval_rows)
    ))
    if not train_rows or not 1 <= args.eval_num_samples <= len(eval_rows) or not 1 <= final_count <= len(eval_rows):
        raise ValueError("Training must be nonempty; periodic/final eval counts must fit the held-out pool.")

    from single_turn.centralized_collaboration import TravelCentralizedCollaborationAdapter
    from single_turn.centralized_comparator import joint_slots
    from single_turn.formatting import get_single_turn_formatters
    from single_turn.rewards.reference_evaluator import parse_reference_catalog
    from single_turn.rewards import make_reward

    for row in [*train_rows, *eval_rows]:
        if not parse_reference_catalog(str(row.get("reference_information", ""))).parse_success:
            raise ValueError(f"Reference catalog parsing failed: {row.get('id')}")
    curriculum = _curriculum_plan(config, train_rows) if magrpo else None
    reward = make_reward(config.get_section("travel_reward"))
    report = {
        "algorithm": algorithm, "collaboration_mode": "centralized",
        "num_actor_models": 1, "num_roles": 2, "agent_devices": args.agent_devices,
        "comparator_generation_mode": getattr(args, "comparator_generation_mode", None),
        "train_samples": len(train_rows), "eval_samples": len(eval_rows),
        "periodic_eval_samples": args.eval_num_samples, "final_eval_samples": final_count,
        "reward_range": list(reward.reward_range), "joint_max_new_tokens": args.max_new_tokens,
        **({"expected_env_steps": curriculum["expected_env_steps"]} if magrpo else budget_report(config, args, len(train_rows))),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if cli.dry_run:
        adapter = TravelCentralizedCollaborationAdapter()
        for row in [*train_rows, *eval_rows]:
            adapter.build_prompt(row, [])
            joint = json.dumps({"agent_id": 0, "assignments": [
                {"day": day, "field": field, "value": "-"} for day, field in joint_slots(row["days"])
            ]})
            roles = adapter.parse_completion(joint, row, 2)
            reward([roles[0]], [roles[1]], batch_items=[row])
            reward.drain_details()
        print(f"Joint contract/reward probes passed for {len(train_rows) + len(eval_rows)} rows (all-dash actions are not successful plans).")
        return

    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from single_turn.logger import aggregate_single_turn_metrics, build_single_turn_eval_logger
    from single_turn.train.centralized_trainer import (
        TravelCentralizedMAGRPOTrainer, TravelCentralizedMADPOTrainer,
        TravelCentralizedMADPOIterTrainer, TravelCentralizedMARLHFTrainer,
        TravelCentralizedMARLHFIterTrainer,
    )

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = config.get_agent_model_config()
    names = _agent_names(config)
    tokenizer = AutoTokenizer.from_pretrained(names[0] if names else model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.special_tokens:
        tokenizer.add_special_tokens(model.special_tokens)
    if config.get("tokenizer.padding_side"):
        tokenizer.padding_side = str(config.get("tokenizer.padding_side"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(config.get("output.base_dir")) / f"seed{seed}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    if hasattr(args, "preference_replay_dir") and args.preference_replay_dir is None:
        args.preference_replay_dir = str(output / "preference_replay")
    config.update({section_name: vars(args), "output": {"run_dir": str(output)}})
    config.save(str(output / "config.yaml"))
    wandb_config = _wandb_config(config, str(output), vars(args))
    if wandb_config is not None:
        wandb_config["dir"] = config.get("wandb.dir") or str(output)
        wandb_config["output_dir"] = str(output)
        wandb_config["name"] = f"{wandb_config['name']}-seed{seed}"
    adapter = TravelCentralizedCollaborationAdapter(tokenizer)
    trainers = dict(zip(ALGORITHMS, (
        TravelCentralizedMAGRPOTrainer, TravelCentralizedMADPOTrainer,
        TravelCentralizedMADPOIterTrainer, TravelCentralizedMARLHFTrainer,
        TravelCentralizedMARLHFIterTrainer,
    )))
    mode_kwargs = {"centralized_adapter" if magrpo else "centralized_comparator_adapter": adapter}
    if magrpo:
        mode_kwargs.update(
            curriculum_short_epochs=curriculum["short_epochs"],
            curriculum_train_dataset=curriculum["short_rows"] if curriculum["short_epochs"] else None,
        )
    else:
        mode_kwargs.update(
            curriculum_short_epochs=0,
            preference_generation_batch_size=int(config.get("travel.preference_generation_batch_size", 20)),
            offload_inactive_actors=_bool(config.get("travel.offload_inactive_actors", True)),
        )
    trainer = trainers[algorithm](
        agent_model=model.name if names is None else None, agents=names, num_agents=2,
        tokenizer=tokenizer, model_config={"torch_dtype": model.torch_dtype,
            "attn_implementation": model.attn_implementation, "special_tokens": model.special_tokens},
        train_dataset=train_rows, eval_dataset=eval_rows, dataset_type="travelplanner",
        reward_func=reward, reward_processor=_build_reward_processor(config),
        formatters=get_single_turn_formatters(num_agents=2, use_chat_template=False),
        args=args, wandb_config=wandb_config,
        eval_logger=build_single_turn_eval_logger(eval_rows, reward_config=reward.config, panel_size=args.eval_num_samples),
        eval_aggregator=aggregate_single_turn_metrics,
        chat_formatted_prompts=True, force_json_prefix=True, constrain_json_skeleton=True,
        max_value_tokens=int(config.get("travel.max_value_tokens", 32)),
        normalize_value_log_probs=_bool(config.get("travel.normalize_value_log_probs", True)),
        role_mode="partitioned_roles", stop_after_complete_json=True,
        rotate_eval_subset=False, greedy_eval=_bool(config.get("travel.greedy_eval", True)),
        **mode_kwargs,
    )
    trainer.verbose = _bool(config.get("output.verbose", True))
    if _bool(config.get("agent_model.gradient_checkpointing", True)):
        trainer.agents[0].gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer.train()
    if _bool(config.get("magrpo.eval_at_end" if magrpo else "evaluation.at_end", True)):
        trainer.evaluate(num_eval_samples=final_count)
    print(f"Finished centralized {algorithm}: env_steps={trainer.env_step}, actor_models={len(trainer.agents)}")
    if _bool(config.get("output.save_final_model", True)):
        # One actor checkpoint, never agent_1. Iterative checkpoint/replay layout
        # is owned by this run's unique directory.
        trainer.agents[0].save_pretrained(output / "agent_0")
        tokenizer.save_pretrained(output / "agent_0")
    if trainer.wandb_initialized:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
