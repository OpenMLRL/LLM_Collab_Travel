"""Train two Travel roles with independent values or a shared centralized critic."""

import argparse
from dataclasses import fields
import json
from pathlib import Path
import random
import sys

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parent
for path in (REPO_ROOT.parent / "CoMLRL", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from single_turn.config import add_config_args, parse_overrides, run_id
from single_turn.train.checkpoint_launch import parse_checkpoint_options
from single_turn.train.train_centralized import load_config
from single_turn.train.train_magrpo import (
    _agent_names,
    _bool,
    _build_reward_processor,
    _load_data,
    _wandb_config,
)


def make_trainer_args(config):
    from comlrl.trainers.actor_critic import IACConfig, MAACConfig

    algorithm = config.get("algorithm")
    if algorithm not in {"iac", "maac"}:
        raise ValueError("algorithm must be iac or maac.")
    section = config.get_section(algorithm)
    cls = IACConfig if algorithm == "iac" else MAACConfig
    unknown = set(section) - {field.name for field in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {algorithm} settings: {sorted(unknown)}")
    model = config.get_agent_model_config()
    args = cls(
        **{
            "temperature": model.temperature,
            "top_p": model.top_p,
            "top_k": model.top_k,
            **section,
        }
    )
    if args.num_agents != 2 or args.num_turns != 1 or args.critic_type != "v":
        raise ValueError(
            "Travel actor-critic requires two roles, one turn and a V critic."
        )
    if args.reference_kl_enabled or args.eval_batch_size != 1:
        raise ValueError("Use reference_kl_enabled=false and eval_batch_size=1.")
    if config.get("travel.curriculum_short_epochs", 0):
        raise ValueError(
            "Travel actor-critic uses the full training split from epoch one."
        )
    for flag in (
        "use_chat_template",
        "force_json_prefix",
        "constrain_json_skeleton",
        "stop_after_complete_json",
        "normalize_value_log_probs",
    ):
        if not _bool(config.get(f"travel.{flag}", True)):
            raise ValueError(f"Travel actor-critic requires travel.{flag}=true.")
    if config.get("travel.role_mode") != "partitioned_roles":
        raise ValueError("Travel actor-critic requires partitioned_roles.")
    return args


def main(default_algorithm=None):
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--algorithm", choices=("iac", "maac"), default=default_algorithm
    )
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()
    algorithm = cli.algorithm or "iac"
    config = load_config(
        cli.config or TASK_ROOT / "configs" / f"single_turn_{algorithm}_config.yaml"
    )
    config.data.pop("magrpo", None)
    updates = parse_overrides(cli.override)
    if set(updates) & {"magrpo", "mapl"}:
        raise ValueError("Use iac.* or maac.* overrides with this entrypoint.")
    config.update(updates)
    checkpoint_options = parse_checkpoint_options(config)
    if config.get("algorithm") != algorithm:
        raise ValueError("--algorithm and config.algorithm disagree.")
    args = make_trainer_args(config)
    train_rows, eval_rows = _load_data(config)
    final_count = int(config.get("evaluation.final_num_samples", len(eval_rows)))
    if (
        not train_rows
        or not 1 <= args.eval_num_samples <= len(eval_rows)
        or not 1 <= final_count <= len(eval_rows)
    ):
        raise ValueError(
            "Train data must be nonempty and eval counts must fit the held-out pool."
        )
    report = dict(
        algorithm=algorithm,
        actor_models=2,
        num_roles=2,
        critic="independent" if algorithm == "iac" else "centralized",
        train_samples=len(train_rows),
        eval_samples=len(eval_rows),
        dataset_seed=config.get("dataset.seed", config.get("seed", 42)),
        training_seed=config.get("seed", 42),
        online_joint_rollouts=len(train_rows)
        * args.num_train_epochs
        * args.num_generations,
        agent_devices=args.agent_devices,
        critic_devices=args.critic_devices,
        value_only_policy_loss=True,
    )
    if cli.dry_run:
        print(json.dumps(report, indent=2))
        return

    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from single_turn.formatting import get_single_turn_formatters
    from single_turn.rewards import make_reward
    from single_turn.structured_generation import DEFAULT_SYSTEM_PROMPT
    from single_turn.train.actor_critic_trainer import (
        TravelIACTrainer,
        TravelMAACTrainer,
    )

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = config.get_agent_model_config()
    names = _agent_names(config)
    if names is not None and len(names) != 2:
        raise ValueError(
            "Travel actor-critic requires exactly two actor model sources."
        )
    tokenizers = [
        AutoTokenizer.from_pretrained(name) for name in (names or [model.name] * 2)
    ]
    for tokenizer in tokenizers:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = config.get("tokenizer.padding_side", "left")
        if model.special_tokens:
            tokenizer.add_special_tokens(model.special_tokens)
    output = Path(config.get("output.base_dir")) / run_id()
    output.mkdir(parents=True, exist_ok=False)
    config.update({algorithm: vars(args), "output": {"run_dir": str(output)}})
    config.save(str(output / "config.yaml"))
    wandb_config = _wandb_config(config, str(output), vars(args))
    if wandb_config is not None:
        wandb_config.update(
            dir=str(output),
            output_dir=str(output),
            name=f"{wandb_config['name']}-seed{seed}",
        )
        wandb_config["config_sections"]["seed"] = seed
    critic = config.get_critic_model_config(required=False) or model
    model_kwargs = {
        "torch_dtype": model.torch_dtype,
        "attn_implementation": model.attn_implementation,
    }
    critic_kwargs = {
        "torch_dtype": critic.torch_dtype,
        "attn_implementation": critic.attn_implementation,
    }
    cls = TravelIACTrainer if algorithm == "iac" else TravelMAACTrainer
    trainer = cls(
        agent_model=model.name if names is None else None,
        agents=names,
        critic_model=(
            critic.name if algorithm == "maac" or args.use_separate_critic else None
        ),
        tokenizer=tokenizers,
        args=args,
        model_config={
            "model_kwargs": model_kwargs,
            "critic_model_kwargs": critic_kwargs,
        },
        reward_func=make_reward(config.get_section("travel_reward")),
        reward_processor=_build_reward_processor(config),
        formatters=get_single_turn_formatters(
            num_agents=2,
            tokenizers=tokenizers,
            use_chat_template=True,
            system_prompt=config.get("travel.system_prompt", DEFAULT_SYSTEM_PROMPT),
        ),
        train_dataset=train_rows,
        eval_dataset=eval_rows,
        wandb_config=wandb_config,
        max_value_tokens=int(config.get("travel.max_value_tokens", 32)),
        greedy_eval=_bool(config.get("travel.greedy_eval", True)),
    )
    trainer.verbose = _bool(config.get("output.verbose", True))
    if _bool(config.get("agent_model.gradient_checkpointing", True)):
        for wrapped in [*trainer.agents, *trainer.critics]:
            if wrapped is not None:
                wrapped.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                wrapped.model.config.use_cache = False
    trainer.train(**checkpoint_options)
    metrics = trainer.evaluate(num_eval_samples=final_count)
    (output / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if _bool(config.get("output.save_final_model", False)):
        trainer.save_model(str(output / "final"))
    if trainer.wandb_initialized:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
