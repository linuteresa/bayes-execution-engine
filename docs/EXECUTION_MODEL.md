# Execution model

Implementation: `core/graph.py`, `nodes/`, `core/runtime.py` and `core/numbers.py`.

## Planning and dependency context

Planner → Executor → Replanner shares a typed `PlanExecuteState`. The planner must return
a bounded nonempty string-step list. The replanner must return an explicit `plan` or
`response` action with the corresponding required fields. Malformed/extra/wrong-type
fields trigger bounded format retries and then `invalid_model_output`; malformed output
cannot masquerade as a successful measured answer.

An executor step receives the goal, the step and bounded prior results. This supports
dependent work such as deriving a value and then explaining it. The prompt explicitly
states there is no database, file or live-system access and prohibits invented records.
The model still provides knowledge/reasoning only; prompt instructions do not prove truth.

## Evidence version 2

The default executor samples the model `EXECUTOR_SAMPLES` times at
`EXECUTOR_TEMPERATURE`. It compares each pair with word-overlap cosine, or embedding cosine
when `EMBEDDINGS_BASE_URL` is set. Numeric-claim multisets and associated unit signatures
are checked before agreement/medoid selection. Different claims receive zero pairwise
agreement. Parsing preserves signs, decimals, grouped values, scientific notation,
fractions and percentage equivalences. Short valid answers such as `Paris` or `42` are
answerable; empty text and explicit refusal markers are degraded.

| Ordinal signal | Measurement | Degradation |
|---|---|---|
| `DataQuality` | Mean guarded pairwise similarity | Lower similarity is worse |
| `TaskStatus` | Fraction of samples that are answerable | More refusals/empty answers is worse |
| `ToolReliability` | Fraction of distinct grouped answers | More dispersion is worse |

The three ordinals run from 0 to 4. Similarity thresholds for grouping are 0.8 for word
overlap and 0.9 for embeddings. The most similar sample, the medoid, becomes the answer;
continuous signals are retained for evaluation/scaling. A one-sample run has no observed
self-consistency comparison, even though its conventional agreement value is 1.

Numeric guards reduce obvious false agreement but are not semantic verification. Different
entities, negations, relations and numerical explanations can still evade them; extra
reasoning numbers can also make equivalent answers look different. `ToolReliability` is
currently an answer-dispersion signal, not a measurement of actual external tool uptime.

## Confidence and bounded control

Each evidence triple is resolved against the Dirichlet posterior. The step score is
`P(CERTAIN) + P(HIGH)` and is paired with `good_credible_interval`. The MAP outcome's
probability and interval are separate fields. Every step score is visible to the replanner.

A score below 0.5 queues one forced `Re-check:` step unless every sample refused. A re-check
is not recursively re-checked. Completed/duplicate plan steps are filtered. The run ends
when the replanner emits a valid response, no further useful steps remain, the step budget
is reached, or a deadline/error stops execution. When a budget is exhausted, the response
explicitly reports the stop and available results.

The returned `confidence_scope` is `weakest_step`: the smallest latest-attempt step score,
with a re-check replacing its preceding attempt. This is a heuristic summary. A synthesized
final answer can introduce errors that this score does not measure. The single-step
calibration harness does not validate it as final-answer probability.

Per-request inference timeouts, maximum generated tokens, bounded format attempts,
`MAX_STEPS` and an absolute job deadline limit work. Guards run before and after nodes;
lease loss prevents stale checkpoint-reference publication and final completion. Recovery
can repeat model calls. See [system design](SYSTEM_DESIGN.md) for attempt isolation.

## Feedback and alternative paths

The API commits one final-answer verdict and its per-step count updates transactionally
in the configured job store. Workers share those counts. This is coarse credit assignment:
every recorded step receives the same outcome label. Standalone engine `.npy` state is
separate and is not automatically imported into service feedback.

`demo.py --sim` and tests use a deterministic scripted sampler through the same evidence
computation. The normal CLI/API returns an explicit error if the live model is unavailable.
`EXECUTOR_EVIDENCE=keywords` deliberately opts into `core/signals.py`, a legacy single-shot
heuristic with no self-consistency sampling. It is not the default path or a validation of
the sampling-based confidence claim.

For the available historical results, versioned collection and known limits, see
[calibration methodology](CALIBRATION.md). The committed real-model rows predate the
current numeric/short-answer/context and grading corrections.
