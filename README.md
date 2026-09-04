# LLM Collaboration — TravelPlanner

This repository implements a one-turn decentralized TravelPlanner task. Two
LLM agents see the same request, constraints, and reference-derived route
scaffold, act simultaneously, and receive one shared MAGRPO reward after a
deterministic merger builds the team itinerary. Each agent receives a compact
catalog tailored to its assigned work instead of the original raw tables.

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
example. Periodic evaluation always uses P1 as a four-row anchor, making every
point on the learning curve directly comparable. The final evaluation uses all
20 rows. The remaining panel grouping is retained for optional diagnostics.

This is deliberately a custom training split over validation-source queries.
It must not be reported as the official TravelPlanner validation benchmark.
The 20 held-out examples are the fixed internal evaluation pool for this
curriculum experiment.

The rollout budget remains aligned with the BFCL experiment. Training first
uses all 30 three-day rows for eight epochs, then the complete 60-row split for
17 epochs:

```text
8 × 30 × 4 + 17 × 60 × 4 = 5040 joint env steps
```

This is within 1.6% of the current BFCL 5120-step budget.
Both phases contain a whole number of ten-prompt rollout buffers, so this still
produces exactly 126 optimizer updates per agent. The curriculum changes only
the order and frequency of training rows; the held-out anchor never selects a
training stage or checkpoint.

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

Both prompts contain the same movement/stay scaffold, derived only from dated
route descriptions in the reference slice. Agent 0 receives exact
transportation values, accommodation metadata, and restaurant options needed
for its even-day dinner duty. Agent 1 receives exact restaurant and attraction
values. Addresses, coordinates, phone numbers, URLs, and unrelated table
columns are omitted. The generation adapter also prefills the fixed JSON
header through the opening quote of the first owned value, so the sampled
policy starts by choosing task content instead of relearning the outer schema.

The plan conventions are stated in both prompts:

- a travel day uses `current_city="from A to B"` and requires matching
  transportation;
- a stay day uses one city, has no transportation, and requires three meals
  plus an attraction;
- accommodation is required except on the final return day;
- the route starts and ends at the origin and visits the requested number of
  cities;
- selected entities must come from the supplied reference information.

## Learnable strict-first reference/constraint reward v7

Each row's reference information is parsed once into catalogs for restaurants,
attractions, accommodations, flights, taxis, and self-driving routes. The
catalog retains entity cities and the metadata needed for cost, room, cuisine,
minimum-night, and transportation checks.

For each strict merged plan the scorer calculates:

- strict action validity for the reward-independent end metrics;
- exact per-agent action validity and verified two-agent contribution;
- reference grounding precision and required grounded recall;
- route continuity, closed-loop travel, city count, and city consistency;
- required information, restaurant/attraction diversity, transportation
  consistency, and minimum-night compliance;
- estimated total cost and all applicable user constraints.

To prevent vacuously satisfied checks from rewarding an all-dash itinerary,
let `S = 0.10 + 0.90 × required grounded recall`. The unit strict plan-quality
score is:

```text
Q = S × 0.20 × assignment coverage
  + 0.25 × required grounded recall
  + 0.15 × grounding F1
  + S × (0.25 × commonsense soft score
       + 0.15 × applicable hard-constraint soft score)
```

Let `V` be the mean exact action validity of the two agents and `J` strict joint
action validity. For each role, measure the fraction of its structurally
required slots that are grounded and valid. The required mask and city/route
validity semantics come from the same dated-reference scaffold shown in both
prompts, never from either agent's output. Let
`B = 0.20 × mean + 0.80 × min` across the two roles. This bottleneck
prevents a syntactically valid all-dash agent from free-riding on its teammate
while keeping partial progress dense.
An empty required set counts as zero contribution, and `from X to X` is invalid;
the logistics role therefore cannot erase the experience role's work surface.
The strict joint-quality term is the geometric composite
`C = B^0.65 × Q^0.35`, or zero if either input is zero. Compared with the old
`B × Q` product, this preserves the same ordering while creating more separation
where early policies have a weak role.

Two bounded learning channels operate below the strict gate. `P` is the same
0.20-mean/0.80-min bottleneck over each role's format progress and soft action
validity. `E` applies the same geometric composite to safely recovered required
assignments and their recovered reference quality. Recovery uses the fixed
reference-derived required-slot mask and rejects wrong-role, overflow, and
route-changing assignments. Let `G` be strict required grounded recall and `U`
strict final collaboration success. The shared MAGRPO reward is:

```text
R = 0.02 × P
  + 0.03 × E
  + 0.04 × V
  + 0.02 × J
  + 0.14 × J × B
  + 0.57 × J × C
  + 0.08 × J × G
  + 0.10 × U
  - 0.10 × invalid/rejected-action rate
  - 0.05 × conflict rate
  - 0.05 × overlap rate
  - 0.10 × sustained reference-copy rate
  - 0.05 × overlength rate
```

`R` is clamped to `[-0.25, 1.00]`. A strict, fully grounded,
constraint-valid plan with verified contributions from both agents receives
`1.00`.

Reward-only recovery extracts complete `(day, field, value)` triples solely to
rank malformed actions. The two early learning channels total `0.05` and never
feed strict `Q`, `B`, `G`, or any ultimate metric. Two invalid agents therefore
receive at most `0.05` positive reward; with exactly one invalid agent the cap is
`0.07` before penalties. A strict all-dash joint action receives exactly `0.08`,
far below a grounded collaboration. Copying a raw reference table is explicitly
penalized.

The JSON stopping state tracks both object braces and array brackets. A
completion that emits the final `}` before closing `assignments` with `]` is no
longer mistaken for a complete object and cropped early.

## Training stability

The default keeps `advantage_mode=mean` but disables per-prompt unit-variance
normalization. Earlier runs either amplified near-zero numerical noise or
optimized recovery while strict collaboration fell to zero. V7 keeps 91% of
all positive reward behind strict joint quality/final success, while its 5%
bounded pre-validity surface makes distinct early outputs rankable.

The rollout and train buffers are 10 prompts. This changes neither model memory
residency nor the 5040-env-step budget, and averages a broader set of prompts per
update. Learning rate, number of generations, and maximum response length are
unchanged.

Periodic evaluation uses greedy argmax generation on the same P1 examples;
training generation remains stochastic. This removes sampling noise from the
within-run anchor comparison. The short curriculum evaluates every 120 env
steps and the full phase every 240 env steps. Because the training distribution
changes at step 960, the fixed held-out eval curve—not a smoothed cross-phase
training average—is the primary convergence curve.

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
| `both_agent_required_grounded_contribution` | Both roles ground every structurally required slot they own | ↑ |
| `conflict_free` | No duplicate or conflicting cross-agent slots | ↑ |
| `reference_grounding` | Every emitted non-empty entity is found in the appropriate reference catalog | ↑ |
| `reference_reasonable_route` | Route parses, is continuous, visits the requested city count, and returns home | ↑ |
| `reference_route_scaffold_match` | Every generated move/stay leg matches the shared reference-derived scaffold | ↑ |
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
required_cooperative_contribution
required_grounded_recall
entity_grounding_precision
```

For paper-style reporting, use `initial / final / delta` columns. The micro and
dense metrics show incremental learning even when the all-or-nothing final
success rate remains zero early in training.

Every periodic evaluation also logs `eval/turn_1/eval_samples`, a W&B table
containing the query, both raw agent actions, merged plan, reward, parser
errors, and headline end metrics for the four anchor examples. Compact aliases
such as `eval/reward`, `eval/team_action_success`, and
`eval/collaboration_success` are logged alongside the full metric tree so
existing W&B panels do not depend on internal turn prefixes. At the terminal
20-example evaluation, full-pool values use `eval_full/*`; the first four
generated rows are also aggregated under `eval/*`, so the last anchor point has
the same denominator as the rest of that curve.

Training logs expose `train/reward` plus the reward-independent
`train/team_action_success`, `train/required_cooperative_contribution`,
`train/required_grounded_recall`, `train/grounding_f1`,
`train/reference_commonsense_soft`, `train/reference_hard_soft`, and
`train/collaboration_success`. `train/reward_group_std`,
`train/nonzero_advantage_rate`, and `train/meaningful_advantage_rate` show
whether the four aligned generations still provide an actual MAGRPO ranking
signal. The corresponding `eval/*` metrics on the fixed anchor are the ones to
use for initial/final/delta claims.
`eval/headline_summary` is a W&B table with `initial / current / delta` for the
fixed anchor, and the same changes are available as `eval/delta/*` scalars.

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
`Travel-magrpo-qwen3-4b-learnable-v7-60train`.

## Tests

The unit tests do not load an LLM:

```bash
python -m unittest discover -s tests -v
```

They verify that different valid itineraries can both receive maximum reward,
target-plan fields cannot change the score, ungrounded and all-dash plans are
penalized, malformed one-agent shortcuts cannot receive a high score,
reward-only repair cannot weaken ultimate metrics, W&B eval tables contain the
actual two-agent actions, compact role catalogs expose every owned choice,
unavailable flights still preserve the shared route scaffold, and the
structured-generation path does not accept an unclosed assignments array.

## Development workflow

Repository edits are intentionally left uncommitted for review in GitHub
Desktop. Commit, push, and remote-cluster synchronization happen only when
explicitly requested. Temporary Slurm scripts and profiling artifacts stay
outside this repository.
