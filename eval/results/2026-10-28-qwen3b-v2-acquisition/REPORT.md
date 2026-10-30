# Calibration report

**Scope: single-step answer confidence.** Each question is evaluated as one execution step. These metrics do not validate confidence in a multi-step final response, the replanner, or the agent's end-to-end stopping policy.

**Analysis only:** this report recomputes metrics from saved answers and labels. Stored confidence, evidence, samples and correctness labels are preserved; no model or embedding requests were made. Risk-coverage now uses expected uniform selection within equal-confidence groups.

## Provenance

| Field | Value |
|---|---|
| Generated (UTC) | 2026-10-29T02:20:22Z |
| Answerer | `local-openai-compatible` |
| Model | `qwen2.5-3b-instruct-q4_k_m` |
| Weights revision | `ollama-manifest:357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` |
| Weights SHA-256 | `5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6` |
| Inference server | `llama.cpp-0.7.0-build11669-620d2225c` |
| Simulated | no |
| Dataset spec | `jsonl:eval/data/calibration-500.jsonl` |
| Items | 500 |
| Samples per item | 4 |
| Agreement measure | `word-overlap` |
| Provenance schema | 2 |
| Experiment ID | `95deeaee1bed4f825cfdf18764a1cc9879b7bea1b4fd882635fd51faaf4289a5` |
| Provenance status | versioned |

Composition: arithmetic × 250, mixed × 250.

## Fix 1 — is the confidence calibrated?

ECE and Brier are scale-sensitive, so a raw score never meant to be a probability can discriminate well and still calibrate badly. AUROC and AURC are rank-based and compare the engine and the baselines on equal terms — read those for *does the number know anything*, and ECE for *does the number mean what it says*. Brackets are 95% bootstrap CIs over items.

| Confidence signal | ECE ↓ | Brier ↓ | AUROC ↑ | AURC ↓ | Acc @100% | Acc @80% |
|---|---|---|---|---|---|---|
| Bayesian engine (125-cell Dirichlet) | 0.165 | 0.264 [0.245, 0.282] | 0.570 [0.520, 0.622] | 0.462 | 0.462 | 0.470 |
| Baseline: self-consistency agreement | 0.402 | 0.403 [0.367, 0.436] | 0.496 [0.443, 0.548] | 0.481 | 0.462 | 0.438 |
| Baseline: mean token logprob | 0.371 | 0.353 [0.325, 0.382] | 0.770 [0.728, 0.809] | 0.334 | 0.462 | 0.545 |

### Verdict

- Confidence separates correct from incorrect answers at AUROC 0.570.
- ECE is 0.165 over 10 bins; the engine is underconfident on average (mean confidence 0.425 vs accuracy 0.462).
- Abstaining on the least-confident 20% lifts accuracy on the remainder from 0.462 to 0.470.
- The engine beats *Baseline: self-consistency agreement* on AUROC by 0.074 [0.054, 0.096] (paired bootstrap CI excludes 0).
- The engine is **worse** than *Baseline: mean token logprob* on AUROC by -0.200 [-0.248, -0.149] (CI excludes 0). The extra machinery is costing discrimination here.

### Reliability bins (engine)

| Confidence bin | Items | Mean confidence | Empirical accuracy | Gap |
|---|---|---|---|---|
| 0.1–0.2 | 20 | 0.188 | 0.100 | +0.088 |
| 0.2–0.3 | 173 | 0.231 | 0.451 | -0.220 |
| 0.3–0.4 | 85 | 0.338 | 0.482 | -0.145 |
| 0.4–0.5 | 46 | 0.468 | 0.261 | +0.207 |
| 0.5–0.6 | 36 | 0.573 | 0.500 | +0.073 |
| 0.6–0.7 | 38 | 0.653 | 0.316 | +0.337 |
| 0.7–0.8 | 102 | 0.718 | 0.667 | +0.051 |

### Per-dataset breakdown

Datasets can have different score and accuracy distributions, so the pooled ranking can differ substantially from ranking within each dataset. Agreement measures consistency rather than correctness; differing intermediate numeric claims can also lower agreement even when final answers match.

| Dataset | Items | Accuracy | ECE ↓ | AUROC ↑ | AUROC (agreement) | Acc @100% | Acc @50% |
|---|---|---|---|---|---|---|---|
| gsm8k | 250 | 0.556 | 0.265 | 0.573 [0.509, 0.635] | 0.506 | 0.556 | 0.592 |
| triviaqa | 250 | 0.368 | 0.182 | 0.799 [0.747, 0.849] | 0.788 | 0.368 | 0.585 |

## Fix 2 — closing the learning loop

`engine.observe()` was called on 250 training rows (250 observations); both engines are then scored on the same 250 **held-out** rows.

| Engine | ECE ↓ | Brier ↓ | AUROC ↑ | Acc @100% |
|---|---|---|---|---|
| Prior only (as shipped) | 0.141 | 0.253 [0.229, 0.279] | 0.611 [0.542, 0.677] | 0.460 |
| Posterior (observed) | 0.060 | 0.227 [0.213, 0.242] | 0.672 [0.603, 0.732] | 0.460 |

ECE change from observing (posterior − prior, negative is better): **-0.081 [-0.160, -0.005]**.

### Credible-interval width vs effective sample size

| ESS bucket | Mean ESS | Mean 95% CI width | Items |
|---|---|---|---|
| 13.0–17.0 | 14.5 | 0.430 | 33 |
| 17.0–48.0 | 21.4 | 0.383 | 58 |
| 48.0–61.0 | 48.0 | 0.278 | 38 |
| 61.0–81.0 | 72.1 | 0.225 | 121 |

### Learning curve (held-out)

| Train observations | ECE ↓ | AUROC ↑ | Mean CI width | Mean ESS |
|---|---|---|---|---|
| 0 | 0.141 | 0.611 | 0.460 | 13.0 |
| 42 | 0.094 | 0.642 | 0.421 | 18.1 |
| 83 | 0.106 | 0.608 | 0.380 | 24.4 |
| 125 | 0.080 | 0.629 | 0.352 | 29.8 |
| 167 | 0.059 | 0.645 | 0.330 | 35.7 |
| 208 | 0.055 | 0.674 | 0.312 | 41.4 |
| 250 | 0.060 | 0.672 | 0.297 | 49.1 |

## Context coverage — is a 125-cell table the right size?

Across all 500 evaluated items the engine visited **45 of 125** contexts (36.0%); 8 contexts saw 10+ observations, and the busiest held 135.

Cells never visited stay at their prior no matter how long the system runs, so low coverage is the empirical argument for reducing dimensionality rather than growing the table — which is what `scaling/latent_bayes.py` does.

### Same rows through the latent (PCA) grid

The executor's 6 continuous signals would give 15,625 naive contexts at 5 bins each. Projected to 125 quantile-binned latent cells, **51** were visited (explained variance 0.914).

| Model | ECE ↓ | Brier ↓ | AUROC ↑ | Contexts visited |
|---|---|---|---|---|
| Hand-designed 125-cell grid | 0.060 | 0.227 [0.213, 0.242] | 0.672 [0.603, 0.732] | 36/125 (train) |
| Latent PCA grid | 0.098 | 0.240 [0.221, 0.260] | 0.621 [0.551, 0.689] | 51/125 (train) |

## Figures

**Reliability diagram**

![Reliability diagram](reliability.png)

**Risk-coverage curve**

![Risk-coverage curve](risk_coverage.png)

**Learning loop**

![Learning loop](learning.png)

## How to reproduce

```bash
python -m eval.run_eval --rows eval/results/2026-10-28-qwen3b-v2-acquisition/rows.jsonl --bins 10 --bootstrap 2000 --seed 0 --abstain 0.2 --test-fraction 0.5
```

## Limitations

- Agreement used word-overlap cosine; paraphrases can read as disagreement and contradictions with similar wording can look consistent.
- Grading is normalized alias containment / numeric match: a correct answer phrased unusually can be scored wrong, which shows up as apparent overconfidence.
- Self-consistency measures agreement, not truth. A model that is confidently and consistently wrong scores high here by construction.
- Intermediate risk-coverage points assume uniform random selection within a boundary tie group. A confidence threshold alone cannot separate equal scores.
