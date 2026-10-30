# Calibration report — historical single-step scores

**Scope: single-step answer confidence.** Each question is evaluated as one execution step. These metrics do not validate confidence in a multi-step final response, the replanner, or the agent's end-to-end stopping policy.

**Analysis only:** this report recomputes metrics from saved answers and labels. Stored confidence, evidence, samples and correctness labels are preserved; no model or embedding requests were made. Risk-coverage now uses expected uniform selection within equal-confidence groups.

**Historical acquisition:** these rows predate versioned experiment manifests and the current evidence/grading implementation. Missing sampling and agreement settings remain unknown. Results characterize the saved scores and labels; they do not establish calibration of the current executor. The learning study refits the current engine on the historical evidence.

## Provenance

| Field | Value |
|---|---|
| Generated (UTC) | unknown (legacy rows) |
| Answerer | `local-openai-compatible` |
| Model | `qwen2.5-3b-instruct-q4_k_m` |
| Simulated | no |
| Dataset spec | `hf:gsm8k+hf:triviaqa` |
| Items | 500 |
| Samples per item | 4 |
| Agreement measure | `unknown (legacy rows)` |
| Provenance schema | legacy / incomplete |
| Experiment ID | `unknown` |
| Provenance status | legacy_partial |

Composition: arithmetic × 250, mixed × 250.

## Fix 1 — is the confidence calibrated?

ECE and Brier are scale-sensitive, so a raw score never meant to be a probability can discriminate well and still calibrate badly. AUROC and AURC are rank-based and compare the engine and the baselines on equal terms — read those for *does the number know anything*, and ECE for *does the number mean what it says*. Brackets are 95% bootstrap CIs over items.

| Confidence signal | ECE ↓ | Brier ↓ | AUROC ↑ | AURC ↓ | Acc @100% | Acc @80% |
|---|---|---|---|---|---|---|
| Bayesian engine (125-cell Dirichlet) | 0.145 | 0.259 [0.245, 0.273] | 0.601 [0.553, 0.651] | 0.519 | 0.410 | 0.443 |
| Baseline: self-consistency agreement | 0.355 | 0.373 [0.343, 0.404] | 0.603 [0.552, 0.653] | 0.525 | 0.410 | 0.448 |
| Baseline: mean token logprob | 0.426 | 0.406 [0.374, 0.437] | 0.665 [0.615, 0.713] | 0.476 | 0.410 | 0.455 |

### Verdict

- Confidence separates correct from incorrect answers at AUROC 0.601.
- ECE is 0.145 over 10 bins; the engine is overconfident on average (mean confidence 0.550 vs accuracy 0.410).
- Abstaining on the least-confident 20% lifts accuracy on the remainder from 0.410 to 0.443.
- The engine is **not** measurably better than *Baseline: self-consistency agreement* on AUROC (-0.002 [-0.031, 0.028]; CI straddles 0). On this dataset the 125-cell layer is not adding discrimination over the raw signal.
- The engine is **worse** than *Baseline: mean token logprob* on AUROC by -0.063 [-0.101, -0.026] (CI excludes 0). The extra machinery is costing discrimination here.

### Reliability bins (engine)

| Confidence bin | Items | Mean confidence | Empirical accuracy | Gap |
|---|---|---|---|---|
| 0.1–0.2 | 6 | 0.178 | 0.000 | +0.178 |
| 0.2–0.3 | 30 | 0.245 | 0.167 | +0.078 |
| 0.3–0.4 | 41 | 0.338 | 0.366 | -0.028 |
| 0.4–0.5 | 157 | 0.458 | 0.389 | +0.069 |
| 0.5–0.6 | 23 | 0.573 | 0.391 | +0.182 |
| 0.6–0.7 | 115 | 0.659 | 0.400 | +0.259 |
| 0.7–0.8 | 128 | 0.718 | 0.539 | +0.179 |

### Per-dataset breakdown

One averaged number hides the most useful thing in this run: self-consistency has a known failure mode — a model that is *consistently* wrong looks confident — and arithmetic is where that bites, because a wrong method reproduces the same wrong answer every sample.

| Dataset | Items | Accuracy | ECE ↓ | AUROC ↑ | AUROC (agreement) | Acc @100% | Acc @50% |
|---|---|---|---|---|---|---|---|
| gsm8k | 250 | 0.392 | 0.154 | 0.549 [0.476, 0.621] | 0.531 | 0.392 | 0.431 |
| triviaqa | 250 | 0.428 | 0.146 | 0.647 [0.578, 0.711] | 0.670 | 0.428 | 0.511 |

## Fix 2 — closing the learning loop

`engine.observe()` was called on 250 training rows (250 observations); both engines are then scored on the same 250 **held-out** rows.

| Engine | ECE ↓ | Brier ↓ | AUROC ↑ | Acc @100% |
|---|---|---|---|---|
| Prior only (as shipped) | 0.141 | 0.254 [0.234, 0.273] | 0.624 [0.555, 0.691] | 0.412 |
| Posterior (observed) | 0.042 | 0.235 [0.222, 0.247] | 0.618 [0.554, 0.686] | 0.412 |

ECE change from observing (posterior − prior, negative is better): **-0.100 [-0.127, -0.005]**.

### Credible-interval width vs effective sample size

| ESS bucket | Mean ESS | Mean 95% CI width | Items |
|---|---|---|---|
| 13.0–18.0 | 14.8 | 0.440 | 35 |
| 18.0–33.0 | 20.0 | 0.400 | 45 |
| 33.0–39.0 | 33.4 | 0.328 | 54 |
| 39.0–77.0 | 61.4 | 0.254 | 116 |

### Learning curve (held-out)

| Train observations | ECE ↓ | AUROC ↑ | Mean CI width | Mean ESS |
|---|---|---|---|---|
| 0 | 0.141 | 0.624 | 0.484 | 13.0 |
| 42 | 0.106 | 0.612 | 0.431 | 18.2 |
| 83 | 0.099 | 0.609 | 0.394 | 22.8 |
| 125 | 0.050 | 0.607 | 0.372 | 27.5 |
| 167 | 0.045 | 0.594 | 0.353 | 32.7 |
| 208 | 0.054 | 0.597 | 0.334 | 37.2 |
| 250 | 0.042 | 0.618 | 0.322 | 41.4 |

## Context coverage — is a 125-cell table the right size?

Across all 500 evaluated items the engine visited **40 of 125** contexts (32.0%); 10 contexts saw 10+ observations, and the busiest held 128.

Cells never visited stay at their prior no matter how long the system runs, so low coverage is the empirical argument for reducing dimensionality rather than growing the table — which is what `scaling/latent_bayes.py` does.

### Same rows through the latent (PCA) grid

The executor's 6 continuous signals would give 15,625 naive contexts at 5 bins each. Projected to 125 quantile-binned latent cells, **54** were visited (explained variance 0.859).

| Model | ECE ↓ | Brier ↓ | AUROC ↑ | Contexts visited |
|---|---|---|---|---|
| Hand-designed 125-cell grid | 0.042 | 0.235 [0.222, 0.247] | 0.618 [0.554, 0.686] | 33/125 (train) |
| Latent PCA grid | 0.030 | 0.238 [0.221, 0.255] | 0.591 [0.517, 0.661] | 54/125 (train) |

## Figures

**Reliability diagram**

![Reliability diagram](reliability.png)

**Risk-coverage curve**

![Risk-coverage curve](risk_coverage.png)

**Learning loop**

![Learning loop](learning.png)

## How to reproduce

```bash
python -m eval.run_eval --rows eval/results/word-overlap/rows.jsonl --bins 10 --bootstrap 2000 --seed 0 --abstain 0.2 --test-fraction 0.5
```

## Limitations

- The acquisition agreement configuration is unknown for these legacy rows.
- Grading is normalized alias containment / numeric match: a correct answer phrased unusually can be scored wrong, which shows up as apparent overconfidence.
- Self-consistency measures agreement, not truth. A model that is confidently and consistently wrong scores high here by construction.
- Intermediate risk-coverage points assume uniform random selection within a boundary tie group. A confidence threshold alone cannot separate equal scores.

