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

- `current_city`, `transportation`, and `accommodation` for every day.

Agent 1 owns daily experience:

- `breakfast`, `attraction`, `lunch`, and `dinner` for every day.

The partition is exhaustive and disjoint. Each agent emits exactly one JSON
assignment for every owned slot, including explicit `"-"` values where the
itinerary convention permits an empty value. A deterministic merger combines
the assignments without an LLM aggregator.

Both prompts contain the same movement/stay scaffold, derived only from dated
route descriptions in the reference slice. Agent 0 receives exact
transportation values and accommodation metadata. Agent 1 receives exact
restaurant and attraction values. Addresses, coordinates, phone numbers, URLs,
and unrelated table columns are omitted. The generation adapter prefills the
first assignment and then forces the complete role-owned JSON skeleton. The
policy chooses only value text and when to close each value; deterministic
schema tokens are excluded from its loss, and value log probabilities are
length-normalized. This prevents one bracket mistake—or the length of a
multi-day JSON response—from dominating a Travel policy update.
The fixed skeleton is intentionally available only for `partitioned_roles`;
reference KL is disabled because schema tokens are excluded from this
domain-specific policy loss.

Each prompt also receives a target-free role budget contract. The contract
computes the cheapest constraint-compatible logistics and experience floors
from the supplied catalog, then splits the remaining slack between the two
roles. The two caps sum exactly to the user budget, so simultaneous agents can
coordinate joint cost without seeing one another's action. Catalog entries show
full-party costs and hard-constraint eligibility, and a per-role day checklist
marks every owned slot as required, exact, or intentionally empty.

The plan conventions are stated in both prompts:

- a travel day uses `current_city="from A to B"` and requires matching
  transportation;
- a stay day uses one city, has no transportation, and requires three meals
  plus an attraction;
- accommodation is required except on the final return day;
- the route starts and ends at the origin and visits the requested number of
  cities;
- selected entities must come from the supplied reference information.

## Reference/constraint reward

The task uses the fixed dense reward formula below together with hard-aware
budget prompts and duplicate-catalog handling. The reward backend is named
`reference_constraint_learnable_budget_dense`.

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

The terminal budget pass remains strict: every required slot must be grounded,
every emitted entity must be costable, and the complete plan must be within
budget. Its training-only soft surface is dense:

```text
budget soft = required grounded recall
            × emitted cost completeness
            × over-budget margin
```

Emitted cost completeness covers every required transportation,
accommodation, and meal slot plus any optional cost-bearing slot the policy
chooses to fill. The separately logged required-cost completeness metric keeps
a stable required-slot denominator. The margin is `1` within budget and falls
linearly to `0` as known cost moves from one to two times the budget. This
preserves the final metric while avoiding a zero reward cliff when one entity
is still malformed.

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
optimized recovery while strict collaboration fell to zero. The fixed reward
preserves a strict success endpoint while its bounded 5% pre-validity surface
makes distinct early outputs rankable.

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
reward-independent joint evaluator computes diagnostics directly from the two
actions and reference catalog; changing reward weights does not change them.
Detailed constraint and parser diagnostics remain available internally, while
W&B publishes only this compact surface:

| Metric | Definition | Better |
| --- | --- | :---: |
| `eval/reward` | Shared MAGRPO reward | ↑ |
| `eval/action_validity` | Mean strict validity of the two agent actions | ↑ |
| `eval/team_action_success` | Both actions are strict, correctly owned, complete, and conflict-free | ↑ |
| `eval/required_cooperative_contribution` | Weaker role's grounded coverage of its required work | ↑ |
| `eval/required_grounded_recall` | Grounded required slots divided by all required slots | ↑ |
| `eval/entity_grounding_precision` | Supported emitted entities divided by emitted entities | ↑ |
| `eval/grounding_f1` | Harmonic mean of grounding precision and required recall | ↑ |
| `eval/required_cost_completeness` | Required priced slots with a known catalog cost | ↑ |
| `eval/reference_budget_soft` | Dense grounding × cost-completeness × budget-margin progress | ↑ |
| `eval/reference_budget_pass` | Strict complete, grounded plan within the total budget | ↑ |
| `eval/route_scaffold_match` | Generated move/stay legs matching the reference-derived scaffold | ↑ |
| `eval/reference_plan_delivery` | A non-empty plan is delivered (paper-style delivery, independent of correctness) | ↑ |
| `eval/required_plan_completion` | Both strict role actions fill every scaffold-required slot | ↑ |
| `eval/reference_commonsense_micro` | Passed commonsense checks divided by applicable checks | ↑ |
| `eval/reference_hard_micro` | Passed hard constraints divided by applicable hard constraints | ↑ |
| `eval/reference_plan_success` | All commonsense and hard checks pass | ↑ |
| `eval/collaboration_success` | Plan success plus valid team action and full two-role contribution | ↑ |

The main table should report at least:

```text
reward
reference_plan_delivery
required_plan_completion
reference_commonsense_micro
reference_hard_micro
reference_plan_success
team_action_success
collaboration_success
required_cooperative_contribution
required_grounded_recall
entity_grounding_precision
required_cost_completeness
reference_budget_soft
reference_budget_pass
```

For paper-style reporting, use `initial / final / delta` columns. The micro and
dense metrics show incremental learning even when the all-or-nothing final
success rate remains zero early in training.

MAGRPO uploads only the fixed-anchor `eval/*` scalar curves above. Training
metrics (`train/*` and stock `turn_1/*` aliases), full-pool `eval_full/*`
metrics, and the `eval/samples` table are not uploaded. No W&B sample table is
constructed.

The terminal evaluation still evaluates all 20 held-out examples and returns
the full-pool results to the caller. Only its first four examples contribute
to the final W&B `eval/*` point, so it has the same denominator as earlier
points. Per-sample evaluation details and training diagnostics remain internal.
Use the fixed-anchor curves for initial/final/delta comparisons.

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
`Travel-magrpo-qwen3-4b-hard-budget-60train`.

## MAPL preference algorithms (one GPU)

Travel now provides three adapters over the existing CoMLRL implementations:

| Entry point | Training procedure | Default online/update budget |
| --- | --- | --- |
| `single_turn/train/train_marlhf.py` | Rank sampled joint actions with the task reward, fit a joint reward model once, then MAGRPO | 21 epochs × 60 prompts × 4 generations = 5040 online joint rollouts |
| `single_turn/train/train_marlhf_iter.py` | Refresh comparisons and refit the reward model before each online stage | 7 iterations × 3 epochs × 60 × 4 = 5040 online joint rollouts |
| `single_turn/train/train_madpo_iter.py` | Iteratively compare joint actions and apply joint DPO, with λ=0.8 replay | At most 7 iterations × 60 prompts × 6 pairs × 2 = 5040 pair-counted env steps |

All three reuse the existing role prompts, strict JSON generation, deterministic
merge, fixed task reward, 60/20 split, and four-example fixed eval anchor. Unlike
MAGRPO's short-trip warm-up, these configurations use all 60 training rows from
the outset. No target/annotated plan is used to generate preference labels.
The iterative comparator is `current_copy`: the same pre-update actor weights
with an independent random-number stream, not a separately trained critic. On
the same device CoMLRL reuses those weights instead of cloning two more actors.

MARLHF's learned training reward is **unbounded**; its scale is not the task
reward scale. Evaluation always bypasses the learned model and uses the fixed
task reward in **[-0.25, 1.0]** and the usual success metrics, making eval curves
comparable across algorithms. All MAPL trainers upload the same `eval/*` scalars.
MADPO iterative and MARLHF iterative additionally upload `iter/*`: iteration
number, new/total/training preference-pair counts, candidate reward distributions
(target vs comparator), and selected/replayed preference reward distributions.
The default `mapl.log_reward_distribution=true` enables the distributions;
setting it to false keeps the small iteration counters without distribution
plots. Non-iterative MARLHF has no iterative refresh loop or `iter/*` panel.
Training, full-pool eval, and sample-table uploads remain disabled.

Distribution plots use 16 fixed bins over the current task reward's declared
`reward_range`, currently **[-0.25, 1.0]**: x = raw task reward, y = sample count.
They are not distributions of MARLHF's learned model scores, advantages, or DPO
losses. The default reward processor is the identity (scale 1, no shift), so raw
and processed task rewards coincide. A future processor scale/shift does not
change these raw-reward plots; changing the task reward's declared min/max does.
The underlying distribution JSON is saved beside preference replay in each
iterative run's `reward_distributions` directory, including when W&B is disabled.

MAPL eval curves use the explicit `env_step` x-axis; iteration scalar curves use
`iter/current_iteration`. W&B's internal history step advances independently,
so an iteration refresh and an eval at the same env step cannot drop each
other's logs. The reward x-axis inside a distribution plot is separate from
either training-progress axis.

Preference collection is additional compute: default MARLHF collects 1200
joint candidates once, while each iterative configuration collects 16,800
across seven refreshes (20 target + 20 comparator candidates per prompt).
Ties produce no preference, so actual pair counts and completed update steps
can be lower. MADPO's pair-counted steps include replay and are not fresh
rollouts. These settings approximate the old plotted step budget, **not equal
total compute**. The dry-run and final console report distinguish these counts.

The defaults put both Qwen3-4B-Instruct-2507 actors on `cuda:0`, use SDPA and
gradient checkpointing, and generate candidates one at a time. During actor
updates, the inactive actor and its optimizer states move to CPU. During reward
model fitting both actors move to CPU; the reward decoder has a scalar head
without LM vocabulary logits. After fitting, the reward optimizer is released
and the frozen reward model stays available for online scoring. Allow ample host
RAM for offloading (requesting 128 GB is a reasonable starting point).
This is designed for one B200 but has **not yet been profiled on a B200**;
host transfer time and long-context activations still matter.

Travel-specific preference records store the exact generated continuation IDs
and value-token masks, including in replay. The assistant prefill is not
duplicated in DPO targets, forced schema tokens carry no policy loss, and the
final selected token is included. Value-token averaging follows Travel's
existing policy setting. The joint DPO gradient is evaluated with one live
winner/loser graph at a time; this requires zero dropout, as in the default
Qwen3 actors. CoMLRL and the existing MAGRPO entrypoint are not modified.

Configuration files support relative `extends`; common `mapl.*` overrides apply
to the selected algorithm. Validate without loading any model weights:

```bash
python single_turn/train/train_marlhf.py --dry-run
python single_turn/train/train_marlhf_iter.py --dry-run
python single_turn/train/train_madpo_iter.py --dry-run
```

Run **one job per GPU**, for example:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
python -u single_turn/train/train_madpo_iter.py --override seed=42
```

Replace the entrypoint with either MARLHF entrypoint for that algorithm. For a
minimal iterative smoke test, reduce collection as well as online updates:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
python -u single_turn/train/train_madpo_iter.py --override \
  dataset.train_samples=1 dataset.eval_samples=1 \
  mapl.num_iterations=1 mapl.num_train_epochs=1 \
  mapl.preference_num_candidates=2 mapl.preference_pairs_per_sample=1 \
  mapl.rollout_buffer_size=1 mapl.train_batch_size=1 \
  mapl.eval_num_samples=1 mapl.eval_interval=0 \
  evaluation.final_num_samples=1 wandb.enabled=false output.save_final_model=false
```

For non-iterative MARLHF omit `mapl.num_iterations`; MARLHF also supports
`mapl.num_generations=2` for smoke runs. A tied smoke sample can produce zero
updates; an all-tied run exits with an explicit error rather than reporting a
successful training run. Check the final `preference_pairs` and `env_steps`.
Each run writes its resolved
configuration and final actors under a unique job/seed/time output directory;
iterative preference replay stays in that run directory.

## Tests

The tests require no pretrained model download; preference-loop tests use tiny
randomly initialized CPU models and scripted candidates:

```bash
python -m unittest discover -s tests -v
```

They verify that different valid itineraries can both receive maximum reward,
target-plan fields cannot change the score, ungrounded and all-dash plans are
penalized, malformed one-agent shortcuts cannot receive a high score,
reward-only repair cannot weaken ultimate metrics, W&B receives only fixed-anchor
eval scalars, compact role catalogs expose every owned choice,
unavailable flights still preserve the shared route scaffold, and the
structured-generation path does not accept an unclosed assignments array.
Preference tests also verify masked/final-token likelihoods, exact replay token
round trips, streamed joint-DPO gradient equivalence, complete reward-model
inputs, and all three training loops, including iterative refresh.

## Development workflow

Repository edits are intentionally left uncommitted for review in GitHub
Desktop. Commit, push, and remote-cluster synchronization happen only when
explicitly requested. Temporary Slurm scripts and profiling artifacts stay
outside this repository.
