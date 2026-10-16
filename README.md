# LLM Collaboration – TravelPlanner

A one-turn decentralized [TravelPlanner](https://huggingface.co/datasets/osunlp/TravelPlanner) task for [CoMLRL](https://github.com/OpenMLRL/CoMLRL). Two agents receive the same travel request and plan different parts of the itinerary at the same time: agent 0 handles logistics (cities, transportation, and accommodation), and agent 1 handles daily experiences (meals and attractions). A deterministic merger combines their answers into one itinerary, which receives a shared reward. Human reference plans are never used, so any grounded itinerary that satisfies the constraints can succeed.

## Installation

CoMLRL can be installed from PyPI or conda-forge (`conda install -c conda-forge comlrl`). Users should also install a PyTorch build compatible with their device.

```bash
pip install comlrl
git clone https://github.com/OpenMLRL/LLM_Collab_Travel.git
cd LLM_Collab_Travel
pip install -r requirements.txt
```

A `CoMLRL/` checkout next to this repository takes precedence over the installed package. The centralized and actor-critic entrypoints need a recent CoMLRL, which such a checkout provides.

## Quick Start

A dry run checks the data split, prompts, reward range, and step budget without loading a model:

```bash
python single_turn/train/train_magrpo.py --dry-run
```

The following command trains the two agents with MAGRPO on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u single_turn/train/train_magrpo.py
```

On one GPU, both agents share the device and are updated in turn:

```bash
CUDA_VISIBLE_DEVICES=0 python -u single_turn/train/train_magrpo.py \
  --override magrpo.parallel_training=none 'magrpo.agent_devices=["cuda:0"]'
```

For long runs, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `TOKENIZERS_PARALLELISM=false` are recommended. Runs log to the W&B project `Travel`; `--override wandb.enabled=false` disables W&B.

## Training Scripts

All entrypoints are in `single_turn/train/` and accept `--config`, `--override`, and `--dry-run`.

| Script | Method |
| --- | --- |
| `train_magrpo.py` | MAGRPO, starting with a short-trip curriculum |
| `train_madpo.py`, `train_madpo_iter.py` | MADPO and its iterative variant |
| `train_marlhf.py`, `train_marlhf_iter.py` | MARLHF and its iterative variant |
| `train_iac.py`, `train_maac.py` | IAC and MAAC |
| `train_centralized.py --algorithm <name>` | One model for both roles, with `magrpo`, `madpo`, `madpo_iter`, `marlhf`, or `marlhf_iter` |

For centralized MAGRPO, `single_turn_centralized_native_magrpo_config.yaml` sets `travel.centralized_prompt: native`: one planner prompt that states the request once and asks for one answer per subtask, instead of the concatenated role prompts ([details](docs/task_details.md#native-centralized-prompt-magrpo)).

The preference methods default to one GPU, and one job per GPU is recommended. Their iterative variants compare against a `current_copy` comparator, the current actor with an independent random stream; `history` and `model` comparators are also supported. The defaults use Qwen3-4B-Instruct-2507 agents.

## Task

- **Data:** a custom 60/20 split of the TravelPlanner validation set (3- and 5-day trips, easy and medium levels, seed 42). It is not the official validation benchmark. Periodic evaluation uses a fixed four-example anchor, and the final evaluation uses all 20 held-out examples.
- **Actions:** each agent fills every slot it owns as one JSON object, choosing exact values from a compact catalog for its role. The JSON structure is generated automatically, so the policy chooses only the values.
- **Reward:** a dense shared reward in [−0.25, 1.0] that combines action validity, grounding in the reference information, commonsense and hard constraints, the budget, and both agents' contributions. A strict, fully grounded, constraint-valid plan with contributions from both agents receives 1.0.
- **Metrics:** reward-independent `eval/*` metrics, including `reference_plan_delivery`, `required_plan_completion`, `reference_commonsense_micro`, `reference_hard_micro`, `reference_plan_success`, `team_action_success`, and `collaboration_success`.

[Task details](docs/task_details.md) gives the full specification: the curriculum, action contract, reward formula, metric definitions, and the settings of each training method.

## Options

**Checkpoints.** A top-level `checkpoint` section is passed to CoMLRL's `trainer.train()`; without it, no training checkpoints are saved.

```yaml
checkpoint:
  checkpoint_dir: ./checkpoints/travel
  checkpoint_interval: 1
  checkpoint_total_limit: 2
  # resume_from_checkpoint: ./checkpoints/travel/checkpoint-1
```

A resumed run continues logging to the saved W&B run unless `wandb.id` or `wandb.resume` is set.

**Tests.** The tests use tiny CPU models and need no model download:

```bash
python -m unittest discover -s tests -v
```
