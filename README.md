# LLM Collaboration — TravelPlanner

This repository implements the first, intentionally narrow TravelPlanner
collaboration experiment: two role-guided LLM agents produce partial itinerary
slot assignments **simultaneously**, a deterministic merger forms one complete
plan, and both agents receive the same plan-level reward.

The initial algorithm is MAGRPO through the sibling `CoMLRL` repository. There
is one turn, no agent-to-agent transcript, and no LLM aggregator.

## Task

Both agents receive the same official TravelPlanner train example and the same
sole-planning reference information.

- Agent 0 receives a soft `LOGISTICS AND FEASIBILITY` reminder.
- Agent 1 receives a soft `DAILY EXPERIENCE` reminder.
- The action space remains the same for both agents; either may fill any field.
- Each agent may emit at most `ceil(7 * days / 2)` assignments.
- The final itinerary contains seven fields per day:
  `current_city`, `transportation`, `breakfast`, `attraction`, `lunch`,
  `dinner`, and `accommodation`.

The default model uses PyTorch SDPA plus gradient checkpointing. Do not switch
back to BFCL's `eager` attention setting: TravelPlanner reference prompts are
much longer, and eager attention's quadratic memory footprint can OOM even on
high-memory GPUs.

An action is JSON:

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

The merger is deliberately non-intelligent:

- one proposal for a slot: accept it;
- the same proposal from multiple agents: accept once and count overlap;
- different proposals for one slot: mark a conflict and leave the slot empty;
- no proposal: leave the slot empty;
- explicit `"-"`: count the slot as intentionally filled.

## Phase-one shared reward

The official train configuration has 45 human-annotated plans. This first
implementation uses a deterministic 40/5 train/eval partition and scores the
merged plan against the annotated plan plus the row's sole-planning reference
information. It therefore runs without downloading the separate TravelPlanner
database.

Every agent receives the same scalar:

```text
R = 0.05 parse
  + 0.15 grounded/correct slot coverage
  + 0.35 annotated-plan slot quality
  + 0.15 reference grounding
  + 0.10 role-aware contribution balance
  + 0.10 primary-role coverage
  + 0.20 exact-plan bonus
  - 0.10 overlap rate
  - 0.20 conflict rate
  - 0.15 lazy-agent rate
  - 0.05 invalid-slot rate
  - 0.05 over-capacity rate
  - 0.10 within-agent duplicate rate
  - 0.25 explicit-empty mismatch rate
```

The result is clamped to `[-0.4, 1.2]`. A conflict-free complementary exact
plan scores `1.1` before the configured training-time scale factor. There is no
individual role reward.

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

## Tests

The unit tests are pure Python and do not load models:

```bash
python -m unittest discover -s tests -v
```

They cover parser behavior, conflict/overlap aggregation, joint-reward ordering,
official annotated-plan normalization, deterministic data partitioning, and
role-specific prompts.
