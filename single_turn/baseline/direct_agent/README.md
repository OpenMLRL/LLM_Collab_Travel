# Direct single-agent Travel controls

Independent raw / GRPO / IAC / RLHF / DPO baselines, analogous to the BFCL
direct-answer controls. One released **Qwen/Qwen3-8B** checkpoint (not Base),
non-thinking, produces the complete itinerary. No coordinator, role tags,
second policy, role projection or agent merger. Existing Travel/CoMLRL trainers
and configs are unchanged.

## Fixed domain protocol

- Pinned TravelPlanner validation partition: days 3/5, easy/medium, visiting
  city counts 1/2. Same existing stratified 60 train / 20 eval split, data seed
  42 independently of optimization seed. No annotated/gold plan is retained.
- Full compact catalogs and reference-derived route; no per-role budget caps.
  The planner sees party-wide costs and the overall budget. These are the same
  catalog eligibility helpers used by the team domain, not a hidden target plan.
- Output is `{"assignments": [{"day": 1, "field": "current_city", "value": ...}, ...]}`,
  every field/day in canonical order. A direct-only extension of the existing
  Travel fixed-slot grammar accepts fused Qwen boundary tokens (e.g. a value
  followed by `"},` in one token), and forces only the remaining schema suffix.
  Masking these tokens caused Qwen3-8B to substitute apostrophe delimiters inside
  otherwise valid entity strings. The existing team decoder is not modified.
  Only selected value tokens (including voluntary closing/boundary tokens)
  enter actor/reference likelihoods. Standalone forced schema/prefill/EOS are excluded.
  All methods, including raw, use this same action space and decoding.
- Full-plan budget 2048 tokens, each value up to 32 tokens; temperature .7,
  top-p .8, top-k 20; generation microbatch 20. No silent prompt truncation.

## Reward and metrics

Reward is **[0,1]**, `0.90 Q + 0.10 reference_final_pass`, strict invalid output
gets 0. `Q` is the existing Travel dense plan-quality component:

```
g = required grounded recall; s = .10 + .90 g
Q = .20 s assignment_coverage + .25 g + .15 grounding_F1
    + s (.25 commonsense_soft + .15 hard_constraint_soft)
```

This deliberately removes role balance, joint-action validity, overlap and
cooperation terms. It is **not scalar-equivalent** to the two-agent reward.
Compare final pass, commonsense/hard pass and grounding across architectures,
not their differently composed rewards. These are local reference-based checks,
not an official TravelPlanner leaderboard evaluation. The domain-specific
delivery/completion distinction is retained without its two-role assumption.

`eval/turn_1/*` logs reward, valid JSON, delivery, required completion, grounded
recall/F1, commonsense micro/all-pass, hard micro/all-pass, budget and final pass.
Micro metrics divide total passed checks by total applicable checks. All plotted
evaluations, including initial/final, use the same first 4 stratified anchors.
Initial/final **full 20** evaluations are saved locally and in run summary for
aggregation, not in `eval_full` panels. No train panel or sample tables.
`turn_1`, `optimization`, and `preference` retain a small set of useful metrics.
Hidden x-axis field: `progress/responses_seen`. Eval RNG is isolated/restored,
with the same seed-dependent sampling policy across methods/checkpoints.
Preference curves instead use `progress/preference_responses`, and RM fitting
uses its pair counter, so these offline phases do not collapse onto actor step0.

## Methods and budgets

Full BF16 fine-tuning, SDPA, non-reentrant checkpointing; no LoRA/quantization.
Algorithm definitions follow BFCL direct controls:

| Method | Training |
| --- | --- |
| raw | No updates; evaluate base released checkpoint |
| GRPO | Group-normalized task reward, clipped token-mean objective, one update per rollout, no reference KL |
| IAC | One rollout/question, shared-backbone V(prompt), normalized R−V, sequence-sum policy loss plus .6 value MSE |
| RLHF | Offline task-oracle preference pairs → independent scalar reward model → GRPO actor against frozen learned reward |
| DPO | Fixed initial reference, beta .1, exact cached reference differences; streamed winner/loser backward |

RLHF/DPO are **non-iter**. RLHF preferences are synthetic task-score labels,
not human preferences or annotated plans. Learned reward and true task reward
are logged separately. Reward-model pairwise gradients are also streamed
exactly to avoid retaining two long-response graphs.

GRPO/RLHF: 60 × 21 epochs × 4 = **5040 online responses**.
IAC: 60 × 84 epochs = **5040 online responses**.
Its four independent questions per optimizer batch are decoded concurrently,
with left padding and separate per-question grammars. Each question still has
exactly one rollout; batching changes the RNG stream, not the IAC objective or
update grouping. This avoids serial full-itinerary decoding underutilizing a GPU.
Offline RLHF/DPO: 60 × 80 = **4800 candidates**, up to **16 strict pairs/question**;
largest reward-gap pairs, ties excluded. DPO up to 3 passes, capped at **5040
response presentations**; actual count may be smaller when too few strict
pairs exist. These are reuses, not newly generated environment steps. This
matches the approximate Travel online budget, not equal total compute:
offline collection, reward fitting and evaluation are additional costs.

## Run and aggregate

From repository root, with installed torch/transformers/datasets/wandb:

```bash
python -m single_turn.baseline.direct_agent.run \
  --config single_turn/baseline/direct_agent/qwen3_8b.json \
  --algorithm grpo --seed 42 --output-dir output_direct/grpo-seed42
```

Choose `raw`, `grpo`, `iac`, `rlhf`, or `dpo`. `--dry-run` validates the split
without loading weights. CPU tests: `python -m unittest discover -s tests -p
test_direct_agent.py -v`. Outputs must be new directories. Snapshots can be
deployed without committing them; cluster submission scripts remain outside
the repository.

For the paired seeds42/43, pass all ten completed output directories to
`python -m single_turn.baseline.direct_agent.aggregate ... --output means.json`.
It refuses missing, failed, mixed-protocol, partial-eval or duplicate-seed runs,
then reports each metric's mean, sample SD and paired difference from raw.
Two seeds provide preliminary evidence, not a reliable confidence interval.
