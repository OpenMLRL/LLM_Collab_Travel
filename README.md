# LLM Collaboration — TravelPlanner

This repository implements a one-turn decentralized TravelPlanner task. Two
LLM agents see the same request and reference information, act simultaneously,
and receive one shared MAGRPO reward after a deterministic merger builds the
team itinerary.

The task is environment-based rather than imitation-based. Human target plans
are removed during data normalization and are never used by the prompt, reward,
or evaluation logger. Any itinerary can succeed if it is grounded and satisfies
the explicit constraints.

## Curriculum data

The default run uses the 180-row TravelPlanner `validation` configuration as a
source for a custom research split. In that source, trip length and number of
visiting cities are coupled:

| Days | Visiting cities | Easy | Medium | Hard |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 20 | 20 | 20 |
| 5 | 2 | 20 | 20 | 20 |
| 7 | 3 | 20 | 20 | 20 |

Consequently, `days=3` and `visiting_city_number=2` has no examples. The
default scalable curriculum keeps the four easiest useful cells and excludes
all hard, seven-day, and three-city queries:

```text
source split: validation
dataset revision: 8736504ecfc31b7f8b7e40122873c337e83fff7c
days: [3, 5]
level: [easy, medium]
visiting_city_number: [1, 2]
candidate rows: 80
train rows: 60
held-out eval rows: 20
seed: 42
```

The 80 reference contexts contain 11,673–35,838 characters, with a median of
20,395.5 versus 25,868 for the full validation source. The split is stratified
by `(days, level)`: every 20-row cell contributes 15 train rows and 5 eval
rows. The eval rows are interleaved as five balanced four-example panels:

```text
P1: [19, 74, 29, 90]
P2: [5, 68, 20, 80]
P3: [14, 63, 39, 94]
P4: [4, 71, 22, 88]
P5: [9, 70, 33, 92]
```

Every panel contains one `3/easy`, `3/medium`, `5/easy`, and `5/medium`
example. Periodic evaluation rotates through the panels, so each call uses
only four rows while every five calls cover the full held-out pool. The final
evaluation always uses all 20 rows.

This is deliberately a custom training split over validation-source queries.
It must not be reported as the official TravelPlanner validation benchmark.
The 20 held-out examples are the fixed internal evaluation pool for this
curriculum experiment.

The rollout budget remains aligned with the BFCL experiment:

```text
60 train prompts × 21 epochs × 4 aligned generations = 5040 joint env steps
```

This is within 1.6% of the current BFCL 5120-step budget.

One aligned generation is one simultaneous two-agent joint action; the two
agents do not add another factor of two to `env_step`.

## Decentralized action contract

Agent 0 owns logistics and feasibility:

- `current_city`, `transportation`, and `accommodation` for every day;
- `dinner` on even-numbered days.

Agent 1 owns daily experience:

- `breakfast`, `attraction`, and `lunch` for every day;
- `dinner` on odd-numbered days.

The partition is exhaustive and disjoint. Each agent emits exactly one JSON
assignment for every owned slot, including explicit `"-"` values where the
itinerary convention permits an empty value. A deterministic merger combines
the assignments without an LLM aggregator.

The plan conventions are stated in both prompts:

- a travel day uses `current_city="from A to B"` and requires matching
  transportation;
- a stay day uses one city, has no transportation, and requires three meals
  plus an attraction;
- accommodation is required except on the final return day;
- the route starts and ends at the origin and visits the requested number of
  cities;
- selected entities must come from the supplied reference information.

## Reference/constraint reward v4

Each row's reference information is parsed once into catalogs for restaurants,
attractions, accommodations, flights, taxis, and self-driving routes. The
catalog retains entity cities and the metadata needed for cost, room, cuisine,
minimum-night, and transportation checks.

For each merged plan the scorer calculates:

- strict and soft action validity for both agents;
- owned-slot coverage and verified contribution by each agent;
- reference grounding precision and required grounded recall;
- route continuity, closed-loop travel, city count, and city consistency;
- required information, restaurant/attraction diversity, transportation
  consistency, and minimum-night compliance;
- estimated total cost and all applicable user constraints.

Let `P` be the harmonic mean of both agents' soft action validity and `C` the
harmonic mean of their verified contribution ratios. The joint gate is:

```text
G = 0.20 + 0.80 × sqrt(P × C)
```

The dense plan score is:

```text
Q = 0.10 × assignment coverage
  + 0.15 × required grounded recall
  + 0.15 × grounding F1
  + 0.35 × commonsense soft score
  + 0.25 × applicable hard-constraint soft score
```

The shared MAGRPO reward is:

```text
R = 0.05 × mean strict parse success
  + G × Q
  + 0.10 × collaboration success
  - 0.10 × overlap rate
  - 0.20 × conflict rate
  - 0.10 × invalid-action rate
  - 0.15 × (1 - C)
```

`R` is clamped to `[-0.50, 1.15]`. A strict, fully grounded, constraint-valid
plan with verified contributions from both agents receives `1.15`. The
terminal bonus is small; dense component scores still distinguish imperfect
rollouts early in training. Filling every slot with `"-"` cannot exploit the
budget or grounding terms because required grounded recall becomes zero.

## End metrics

The training reward is a learning signal, not the headline result. A
reward-independent joint evaluator computes the fixed-evaluation metrics
directly from the two actions and reference catalog; changing reward weights
does not change these values. In W&B, the names below appear under
`eval/turn_1/ultimate/`:

| Metric | Definition | Better |
| --- | --- | :---: |
| `team_action_success` | Both agents produce strict, correctly owned, complete, conflict-free actions | ↑ |
| `both_agent_verified_contribution` | Every owned slot from both agents is independently valid | ↑ |
| `conflict_free` | No duplicate or conflicting cross-agent slots | ↑ |
| `reference_grounding` | Every emitted non-empty entity is found in the appropriate reference catalog | ↑ |
| `reference_reasonable_route` | Route parses, is continuous, visits the requested city count, and returns home | ↑ |
| `reference_complete_information` | Every team slot is explicitly assigned and every structurally required slot is non-empty | ↑ |
| `reference_within_current_city` | Every selected entity or transportation option matches that day's city or route | ↑ |
| `reference_transport_consistency` | Movement days have valid dated transport; stay days do not | ↑ |
| `reference_restaurant_diversity` | No restaurant is reused | ↑ |
| `reference_attraction_diversity` | No attraction is reused | ↑ |
| `reference_minimum_nights` | Consecutive accommodation choices satisfy listed minimum stays | ↑ |
| `reference_commonsense_micro` | Passed local commonsense checks divided by all local commonsense checks | ↑ |
| `reference_commonsense_macro` | All local commonsense checks pass for the example | ↑ |
| `reference_budget_pass` | A complete grounded itinerary has estimated cost within budget | ↑ |
| `reference_hard_micro` | Passed applicable local hard checks divided by applicable local hard checks | ↑ |
| `reference_hard_macro` | Every applicable local hard check passes | ↑ |
| `reference_plan_success` | Both reference commonsense macro and reference hard macro pass | ↑ |
| `collaboration_success` | Reference plan success plus team action success and full two-agent contribution | ↑ |

The main table should report at least:

```text
reward
reference_commonsense_micro / reference_commonsense_macro
reference_hard_micro / reference_hard_macro
reference_plan_success
team_action_success
both_agent_verified_contribution
collaboration_success
required_grounded_recall
entity_grounding_precision
```

For paper-style reporting, use `initial / final / delta` columns. The micro and
dense metrics show incremental learning even when the all-or-nothing final
success rate remains zero early in training.

Easy examples have no cuisine, room-type, house-rule, or transportation
restriction; medium examples add one such constraint. Hard examples, which
contain three active local constraints, remain outside this curriculum.

These are self-contained reference-backed checks inspired by TravelPlanner's
public evaluator. They should not be labeled as official benchmark metrics
until results are separately run through the official database evaluator.

## Run

Expected directory layout:

```text
GitHub/
  CoMLRL/
  LLM_Collab_Travel/
```

After activating the environment, verify the actual split, prompts, reference
catalog, reward range, and 5040-step budget without loading a model:

```bash
cd /path/to/GitHub/LLM_Collab_Travel
python single_turn/train/train_magrpo.py --dry-run
```

Run MAGRPO on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
python -u single_turn/train/train_magrpo.py
```

For one B200, keep both actors on the single logical device and train them
sequentially:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
python -u single_turn/train/train_magrpo.py \
  --override magrpo.parallel_training=none \
             'magrpo.agent_devices=["cuda:0"]'
```

W&B defaults to project `Travel` and run name
`Travel-magrpo-qwen3-4b-reference-constraints-v4-60train`.

## Tests

The unit tests do not load an LLM:

```bash
python -m unittest discover -s tests -v
```

They verify that different valid itineraries can both receive maximum reward,
target-plan fields cannot change the score, ungrounded and all-empty plans are
penalized, route and collaboration failures change the end metrics, weighted
micro aggregation is correct, and the structured-generation path still crops
each completed JSON object safely.

## Development workflow

Repository edits are intentionally left uncommitted for review in GitHub
Desktop. Commit, push, and remote-cluster synchronization happen only when
explicitly requested. Temporary Slurm scripts and profiling artifacts stay
outside this repository.
