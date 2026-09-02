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
keep content feedback dense. Strict validity, recoverable validity, and soft
validity are deliberately separate signals; only the strict one-object form
receives strict action-validity credit or can earn the exact-plan bonus.

### Structured generation

The default generation path enforces this contract instead of relying on the
reward to teach it from zero:

1. Each role prompt is rendered with the tokenizer's native chat template as a
   system message, user message, and assistant-generation prompt.
2. The opening `{` is supplied as assistant prefill, then restored in the
   reward-facing response. It is context rather than a sampled policy token.
3. A lexical JSON stopping criterion tracks strings and escape characters and
   stops each sampled sequence independently after its first complete
   top-level object.

These controls apply to both training and evaluation. They prevent the failure
mode where a raw causal prompt continues the final instruction bullet before
the JSON, and prevent a correct first object from being followed by a second
object or repeated prose. The relevant defaults are
`travel.use_chat_template=true`, `travel.force_json_prefix=true`, and
`travel.stop_after_complete_json=true`. Chat-template mode fails loudly when a
tokenizer has no template; disable it only when intentionally using a base
model that expects raw text.

The merger is deliberately non-intelligent:

- one proposal for a slot: accept it;
- the same proposal from multiple agents: accept once and count overlap;
- different proposals for one slot: mark a conflict and leave the slot empty;
- no proposal: leave the slot empty;
- explicit `"-"`: count the slot as intentionally filled.

## Dense joint reward v3

The official train configuration has 45 human-annotated plans. This first
implementation uses a deterministic 40/5 train/eval partition and scores the
merged plan against the annotated plan plus the row's sole-planning reference
information. It therefore runs without downloading the separate TravelPlanner
database.

Every agent receives the same scalar. The validity terms are:

- `A`: mean strict parse success. A strict action is exactly one JSON object
  with the required schema, matching agent ID, legal values, no duplicate, and
  no capacity overflow.
- `Q_i`: recoverable action validity for agent `i`. The whole completion need
  not itself be strict JSON; the recovered payload must still satisfy the
  schema, ID, capacity, ownership, exact owned-slot count, and full
  item-acceptance checks.
- `V_i`: soft action validity for agent `i`. With `D_i` for decode success, its
  six equally weighted components are schema validity, agent-ID match,
  capacity validity, ownership precision, assignment-count fidelity, and
  accepted-item ratio:

```text
V_i = D_i * (schema_i + id_i + capacity_i + ownership_precision_i
             + count_fidelity_i + item_acceptance_i) / 6
```

Let `V = H(V_0, V_1)` be the harmonic mean of the two soft validity scores and
`C` be the harmonic mean of their useful semantic contribution ratios. The
default equal-weight geometric blend and the single cooperative gate are:

```text
J = sqrt(V * C)                    # 0 when either signal is 0
G = 0.20 + 0.80 J

P = 0.20 grounded nonempty coverage
  + 0.45 annotated-plan nonempty slot quality
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
  - 0.15 nonempty fill on an annotated-empty slot
```

Here the `0.20` exact bonus is awarded only when the merged plan is exact and
both actions pass strict action validity. The result is clamped to
`[-0.5, 1.2]`. A strict, ownership-compliant, conflict-free exact plan scores
`1.2`. If the same exact payloads are recoverable but one agent adds surrounding
text, the plan gate remains `1.0`, but `A` falls to `0.5`, its weighted
contribution becomes `0.025`, and the exact bonus is withheld, yielding
`0.975`. Thus strict output remains clearly preferable without erasing the
content-learning signal. Both agents always receive this same joint value;
there is no separately optimized individual reward.

`travel_reward.validity_gate_weight` controls the geometric blend's validity
exponent and defaults to `0.50`; the contribution exponent is its complement.
For diagnostics, `validity_gate = 0.20 + 0.80 V` and
`contribution_gate = 0.20 + 0.80 C` are logged separately, but they are not
multiplied together. This replaces v2's brittle strict-team AND gate, whose two
separately floored factors could reduce the plan multiplier to `0.04`.

Annotated `"-"` slots are excluded from the denominators for coverage, slot
quality, balance, role quality, and cooperative contribution. Empty-slot credit
is instead multiplied by nonempty coverage, so predicting `"-"` everywhere
cannot exploit the many empty fields in TravelPlanner. Requiring two-sided
contribution and using the cooperative gate similarly prevents one agent from
constructing the whole plan while the other emits a token assignment.

Increasing slot-quality weight from `0.35` to `0.45`, reducing coverage weight
from `0.30` to `0.20`, and increasing the spurious-fill penalty from `0.10` to
`0.15` makes “fill every slot with any grounded candidate” less attractive.
This is still a dense phase-one proxy, not yet the full official constraint
reward: valid alternative itineraries can score below the single annotated target.
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

One aligned generation is one simultaneous joint action, so the two agents are
not an additional factor of two in `env_step`. A run ending at 2560 steps used
a 16-epoch override; it was not an early stop. The repository default is the
full 32-epoch, 5120-step budget shown above.

## MAGRPO stability and evaluation

Travel keeps the stock CoMLRL MAGRPO optimization semantics. Structured-output
handling is domain-local: completed JSON responses are physically cropped before
they enter the rollout buffer, so padded generation tails cannot become policy
targets even when using an unmodified CoMLRL checkout.

Periodic eval runs at each epoch boundary, and `eval_at_end=true` adds an
evaluation at the actual final step (`5120`) so the curve no longer ends one
epoch before the fully trained policy.

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

Evaluation logging includes team-level reward components and stable per-agent
diagnostics under `turn_1/agent_0/...` and `turn_1/agent_1/...`, including:

- `decode_success`, `strict_json`, `schema_valid`, `agent_id_match`, and
  `capacity_valid`;
- `ownership_validity`, `ownership_precision`, `count_fidelity`,
  `item_acceptance`, strict `action_validity`, `recoverable_action_validity`,
  and `soft_action_validity`;
- semantic and raw action contribution ratios, assignment counts, and raw
  assignment counts;
- `parser_error/any` and a binary series for every parser error such as
  `not_strict_json`, `decode_failed`, `agent_id_mismatch`, or
  `capacity_exceeded`.

Every known parser-error series logs zeros as well as ones, avoiding misleading
sparse W&B curves. Team summaries include `team_action_valid`,
`team_recoverable_action_valid`, `team_soft_action_validity`,
`joint_gate_signal`, `validity_gate`, `contribution_gate`, and
`cooperation_gate`.

Training logs additionally expose `loss`, `grad_norm`, `group_reward_std`,
`group_return_std`, `advantage_raw_std`, and response-length mean/max for each
agent, together with prompt-length mean/max to catch context truncation.
Unprefixed versions are cross-agent means. In particular,
`group_reward_std` makes it visible when all four MAGRPO generations receive
the same reward and the relative-advantage signal has disappeared.

## Tests

The unit tests are pure Python and do not load models:

```bash
python -m unittest discover -s tests -v
```

They cover parser behavior, conflict/overlap aggregation, strict versus
recoverable reward behavior, dense per-agent logging, reward ordering for
grounded-but-wrong plans, official annotated-plan normalization, deterministic
data partitioning, role-specific prompts, chat-template rendering, forced JSON
prefixes, and per-sequence complete-object stopping.

## Development workflow

Repository edits are intentionally left uncommitted for review in GitHub
Desktop. Commit, push, and remote-cluster synchronization happen only when the
reviewer explicitly requests them; temporary Slurm launch scripts and profiling
artifacts stay outside this repository.
