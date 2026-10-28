# Calibration methodology and scope

Implementation: `eval/`; entry point: `python -m eval.run_eval --help`.

## What is measured

Each question is treated as **one execution step**. The harness samples the executor,
records its evidence, resolves the same Bayesian step score, selects a consensus answer
and grades it against gold. This isolates a single-step confidence signal. It does not
measure planner quality, dependent multi-step reasoning, replanner synthesis or the final
weakest-step score as a probability of correctness.

The committed real-model rows predate evidence version 2 and grading version 2. Their
reports have been regenerated from stored scores and labels with corrected tie handling;
no model or embedding requests were made and raw historical rows are unchanged. They do
not establish calibration of the current executor. Missing acquisition details remain
unknown rather than being inferred from the current environment.

## Collect, resume, re-score and analyse

```bash
pip install -r requirements.txt -r requirements-eval.txt
python -m eval.run_eval --model llama --dataset hf:triviaqa+hf:gsm8k \
  --limit 250 --samples 4 --keep-samples --out eval/results/new-run
# Repeat the identical acquisition command with --resume after an interruption.
python -m eval.run_eval --rows eval/results/embeddings/rows.jsonl --out /tmp/bayes-analysis
```

Collection writes and fsyncs each completed JSONL observation before moving on. A manifest
records completion, expected items and a checksum of its durable prefix. Resume validates
experiment/dataset configuration, saved question identities and that checksum; a torn
last append can be removed, while malformed complete rows are errors. Existing output is
not overwritten without explicit `--overwrite`.

Every new row embeds versioned acquisition metadata: the dataset digest and selection,
sample count, temperature/seed scope, similarity backend/configuration, code fingerprint,
dependency versions, grading version, model request label, and source experiment when
replaying. The model request label does not independently identify server-loaded weights;
record their revision/hash separately. A CLI seed controls dataset order and the simulator,
not the random seed of a live inference server.

`--rows PATH` preserves stored evidence, confidence and correctness labels and recomputes
metrics only. `--model replay:PATH` requires stored samples and re-runs the current
consensus/evidence/grading logic. Replaying samples under evidence v2 is a new scoring
experiment, not a correction that silently changes a historical record. Changing the
similarity backend can also change which medoid answer is graded. Both replay paths
preserve simulated/unknown provenance conservatively.

Analysis records its own code/dependencies/configuration and source-row checksum separately
from acquisition. Reproduction instructions for historical reports use `--rows`, not a
made-up bundled dataset or the current model server configuration.

## Labels and baselines

Graders prefer the final explicitly labelled/boxed answer over alternatives mentioned in
reasoning. Alias matching respects token boundaries and rejects explicit negation or
ambiguity; numeric grading handles signs, decimals, grouped numbers, scientific notation
and fractions. These are deterministic heuristics, not semantic judges. They can reject
valid paraphrases and still miss subtle incorrect answers. Audit a representative sample
of labels before making stronger claims. Historical labels have not been retroactively
changed to the revised rules.

Baselines are raw pairwise self-consistency and `exp(mean logprob)` when every sampled
response has a usable token-logprob value. The latter transforms the mean of per-response
mean token logprobs; it is a geometric likelihood score, not a calibrated probability that
the answer is correct. Missing baseline observations remove that comparison rather than
silently comparing a different subset of questions.

## Metrics

| Metric | Interpretation |
|---|---|
| ECE/reliability bins | Compare mean confidence and empirical accuracy in fixed-width bins; bin count matters |
| Brier | Mean squared error of the score against correctness; proper probability score |
| AUROC | Ranking of right above wrong answers; ties receive half credit; undefined with one label |
| Risk-coverage/AURC | Expected retained accuracy/risk as the answered share changes |

Within equal-score groups, selective prediction uses the expected accuracy of uniform
selection. Row permutations cannot manufacture abstention gains. Intermediate points
require random selection within the boundary tie; a confidence threshold alone includes
or excludes the whole group. AURC is the mean risk over observation-level coverages.

Bootstrap intervals resample questions; paired comparisons resample the same question
indices for both signals. A confidence interval spanning zero does not establish an
improvement. Repeated questions are kept together during the train/test split, and item
IDs must be unique to avoid replay collisions. Bootstrap intervals currently treat rows
as independent; workloads with repeated questions need an appropriate clustered bootstrap
before those intervals are interpreted as population uncertainty.

## Learning and dimensionality reduction

Rows split into question-disjoint train/test groups, approximately stratified by label.
The train half supplies `observe()` counts: correct → `CERTAIN`, wrong → `AMBIGUOUS`.
Prior and posterior are scored on the same held-out half. Learning curves use that fixed
test split. The PCA grid is fitted on training signals only, then evaluated on those same
held-out questions. Coverage comparisons use training coverage consistently for each grid.

Quantile binning balances marginal axes, not every Cartesian cell. Sparse contexts retain
their prior. Posterior credible intervals describe uncertainty under the model assumptions;
they are not frequentist accuracy guarantees or proof of calibration.

## Historical results after tie correction

| Stored run | Items | ECE | Brier | AUROC |
|---|---:|---:|---:|---:|
| Embedding agreement | 500 | 0.144 | 0.246 | 0.654 |
| Word overlap | 500 | 0.145 | 0.259 | 0.601 |
| First run, older evidence/prompt/model build | 500 | 0.163 | 0.239 | 0.629 |

For historical embedding scores, TriviaQA accuracy is 0.432 → **0.611** at 50% coverage;
GSM8K is 0.392 → **0.444**. Word-overlap equivalents are 0.428 → **0.511** and
0.392 → **0.431**. The first run gives 0.424 → **0.604** and 0.220 → **0.221**.
These values replace input-order-dependent tie results. AUROC/ECE/Brier are unchanged
because historical scores and labels are unchanged.

The embedding GSM8K AUROC interval spans chance. The first-run held-out learning ECE change
is −0.055 [−0.107, 0.012], so the observed point improvement is not established statistically.
Differences between first and later acquisition runs are not controlled model comparisons:
model builds, server and prompts changed. The paired word/embedding sample comparison is
historical; it does not isolate or evaluate the current numeric-claim guard.

Reports: [embeddings](../eval/results/embeddings/REPORT.md),
[word overlap](../eval/results/word-overlap/REPORT.md),
[first run](../eval/results/REPORT.md). The
[simulated example](../eval/results/example-simulated/REPORT.md) was refreshed with current
code and includes a versioned manifest, but the sampler is told the gold answer. It proves
pipeline execution only and is banner-stamped as not a calibration result.

## Next evidence needed

Run fresh live-model collections with recorded weights/server configuration, audit grading
labels, evaluate multiple sampling runs, and add independent correctness signals. Separately
run a held-out end-to-end evaluation of final responses and stopping/rechecking decisions.
Self-consistency can be maximally high for a consistently wrong method, so reliable
arithmetic or factual verification requires evidence beyond repetition.
