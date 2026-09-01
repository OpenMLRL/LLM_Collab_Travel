# LLM Collaboration — TravelPlanner

This repository implements the first, intentionally narrow TravelPlanner
collaboration experiment: two role-partitioned LLM agents produce partial itinerary
slot assignments **simultaneously**, a deterministic merger forms one complete
plan, and both agents receive the same plan-level reward.

The initial algorithm is MAGRPO through the sibling `CoMLRL` repository. There
is one turn, no agent-to-agent transcript, and no LLM aggregator.

## Task

Both agents receive the same official TravelPlanner train example and the same
sole-planning reference information.

- Agent 0 owns `current_city`, `transportation`, and `accommodation` on every
  day, plus dinner on even-numbered days.
- Agent 1 owns `breakfast`, `attraction`, and `lunch` on every day, plus dinner
  on odd-numbered days.
- Each agent must emit exactly one assignment for every owned slot, including
  an explicit `"-"` when the correct value is empty.
- The partition is exhaustive and disjoint. For a `D`-day trip, Agent 0 owns
  `3D + floor(D/2)` slots and Agent 1 owns `3D + ceil(D/2)` slots; neither may
  exceed `ceil(7D/2)` assignments.
- The final itinerary contains seven fields per day:
  `current_city`, `transportation`, `breakfast`, `attraction`, `lunch`,
  `dinner`, and `accommodation`.

The default model uses PyTorch SDPA plus gradient checkpointing. Do not switch
back to BFCL's `eager` attention setting: TravelPlanner reference prompts are
much longer, and eager attention's quadratic memory footprint can OOM even on
high-memory GPUs.

An action must be exactly one JSON object, with no Markdown fence, prose,
prefix/suffix, or second object:

```json
{
  "agent_id": 0,
  "assignments": [
    {
      "day": 1,
      "field": "transportation",
      "value": "Flight Number: F3573659, from St. Petersburg to Rockford"
    }
  ]
}
```

The example illustrates the schema only; a real response contains every slot
owned by that agent. The top-level object must have exactly `agent_id` and
`assignments`, `agent_id` must match the prompted agent, and every assignment
must have exactly `day` (integer), `field` (valid field), and `value` (string).
The parser can recover assignments from malformed early-training outputs to
keep content feedback dense, but only the strict one-object form receives
action-validity credit or can earn the exact-plan bonus.

The merger is deliberately non-intelligent:

- one proposal for a slot: accept it;
- the same proposal from multiple agents: accept once and count overlap;
- different proposals for one slot: mark a conflict and leave the slot empty;
- no proposal: leave the slot empty;
- explicit `"-"`: count the slot as intentionally filled.

## Dense joint reward v2

The official train configuration has 45 human-annotated plans. This first
implementation uses a deterministic 40/5 train/eval partition and scores the
merged plan against the annotated plan plus the row's sole-planning reference
information. It therefore runs without downloading the separate TravelPlanner
database.

Every agent receives the same scalar. Let `A` be mean strict-format validity,
`T` indicate that both actions are strict, complete, and ownership-compliant,
and `C` be the harmonic mean of the two agents' useful semantic contribution
ratios. The cooperative gate and plan score are:

```text
G = (0.20 + 0.80 T) * (0.20 + 0.80 C)

P = 0.30 grounded nonempty coverage
  + 0.35 annotated-plan nonempty slot quality
  + 0.05 gated explicit-empty recall
  + 0.15 role-target contribution balance
  + 0.10 owned-slot quality

R = 0.05 A + G * P
  + 0.20 strict exact-plan bonus
  - 0.15 cross-agent overlap rate
  - 0.30 cross-agent conflict rate
  - 0.25 owned-action coverage deficit of the less active agent
  - 0.15 invalid-slot rate
  - 0.25 over-capacity-agent rate
  - 0.15 within-agent duplicate rate
  - 0.25 explicit-empty mismatch rate
  - 0.10 nonempty fill on an annotated-empty slot
```

The result is clamped to `[-0.5, 1.2]`. A strict, ownership-compliant,
conflict-free exact plan scores `1.2`. Both agents always receive this same
joint value; there is no separately optimized individual reward.

Annotated `"-"` slots are excluded from the denominators for coverage, slot
quality, balance, role quality, and cooperative contribution. Empty-slot credit
is instead multiplied by nonempty coverage, so predicting `"-"` everywhere
cannot exploit the many empty fields in TravelPlanner. Requiring two-sided
contribution and using the cooperative gate similarly prevents one agent from
constructing the whole plan while the other emits a token assignment.

This is a dense phase-one proxy, not yet the full official constraint reward:
valid alternative itineraries can score below the single annotated target.
The next evaluator backend should call TravelPlanner's per-sample commonsense
and hard-constraint evaluators while preserving the same reward interface.

## BFCL-matched rollout budget

The current BFCL native-parallel MAGRPO configuration executes:

```text
160 prompts x 8 epochs x 4 aligned generations = 5120 logged env steps
```

TravelPlanner has only 45 annotated train rows, so this repository uses a
disjoint 40/5 split and:

```text
40 train prompts x 32 epochs x 4 aligned generations = 5120 logged env steps
```

This also preserves 320 optimizer-buffer updates per agent with buffer size 4.
The BFCL README's older “approximately 2560” statement no longer matches its
current eight-epoch config; this project follows the effective config.

The first 12-hour cluster diagnostic intentionally overrides the default to 16
epochs (`40 x 16 x 4 = 2560` env steps) so runtime, VRAM, reward components,
and sampled team outputs can be inspected before committing to the full
BFCL-matched run. That diagnostic override does not change the repository's
32-epoch default.

## Run

The expected directory layout is:

```text
GitHub/
  CoMLRL/
  LLM_Collab_Travel/
```

Install dependencies, then verify data, prompts, aggregation, reward, and step
count without loading a model:

```bash
cd /Users/ninoliu/Documents/GitHub/LLM_Collab_Travel
pip install -r requirements.txt
python single_turn/train/train_magrpo.py --dry-run
```

Run MAGRPO on two GPUs:

```bash
python single_turn/train/train_magrpo.py
```

Override config values in the same style as the BFCL repository:

```bash
python single_turn/train/train_magrpo.py \
  --override agent_model.name=Qwen/Qwen2.5-7B-Instruct \
             magrpo.agent_devices='["cuda:0", "cuda:1"]'
```

The default configuration is
`single_turn/configs/single_turn_magrpo_config.yaml`.

W&B logging uses the `Travel` project. Give diagnostic and full runs distinct
names so their rollout budgets remain visible in the dashboard:

```bash
python single_turn/train/train_magrpo.py \
  --override wandb.project=Travel wandb.name=Travel-local-smoke
```

## Tests

The unit tests are pure Python and do not load models:

```bash
python -m unittest discover -s tests -v
```

They cover parser behavior, conflict/overlap aggregation, joint-reward ordering,
official annotated-plan normalization, deterministic data partitioning, and
role-specific prompts.

## Development workflow

Repository edits are intentionally left uncommitted for review in GitHub
Desktop. Commit, push, and remote-cluster synchronization happen only when the
reviewer explicitly requests them; temporary Slurm launch scripts and profiling
artifacts stay outside this repository.
