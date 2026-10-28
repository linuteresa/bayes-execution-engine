# Calibration report

## Provenance

| Field | Value |
|---|---|
| Generated (UTC) | 2026-10-28T17:45:07Z |
| Answerer | `replay of local-openai-compatible` |
| Model | `qwen2.5-3b-instruct-q4_k_m` |
| Simulated | no |
| Dataset spec | `bundled` |
| Items | 500 |
| Samples per item | 4 |
| Agreement measure | `embeddings:all-minilm-l6-v2` |

Composition: arithmetic × 250, mixed × 250.

## Fix 1 — is the confidence calibrated?

ECE and Brier are scale-sensitive, so a raw score never meant to be a probability can discriminate well and still calibrate badly. AUROC and AURC are rank-based and compare the engine and the baselines on equal terms — read those for *does the number know anything*, and ECE for *does the number mean what it says*. Brackets are 95% bootstrap CIs over items.

| Confidence signal | ECE ↓ | Brier ↓ | AUROC ↑ | AURC ↓ | Acc @100% | Acc @80% |
|---|---|---|---|---|---|---|
| Bayesian engine (125-cell Dirichlet) | 0.144 | 0.246 [0.233, 0.258] | 0.654 [0.605, 0.701] | 0.469 | 0.412 | 0.470 |
| Baseline: self-consistency agreement | 0.434 | 0.424 [0.391, 0.457] | 0.617 [0.565, 0.666] | 0.521 | 0.412 | 0.448 |
| Baseline: mean token logprob | 0.424 | 0.406 [0.374, 0.436] | 0.654 [0.604, 0.703] | 0.479 | 0.412 | 0.458 |

### Verdict

- Confidence separates correct from incorrect answers at AUROC 0.654.
- ECE is 0.144 over 10 bins; the engine is overconfident on average (mean confidence 0.556 vs accuracy 0.412).
- Abstaining on the least-confident 20% lifts accuracy on the remainder from 0.412 to 0.470.
- The engine beats *Baseline: self-consistency agreement* on AUROC by 0.037 [0.004, 0.072] (paired bootstrap CI excludes 0).
- The engine is **not** measurably better than *Baseline: mean token logprob* on AUROC (-0.000 [-0.034, 0.035]; CI straddles 0). On this dataset the 125-cell layer is not adding discrimination over the raw signal.

### Reliability bins (engine)

| Confidence bin | Items | Mean confidence | Empirical accuracy | Gap |
|---|---|---|---|---|
| 0.1–0.2 | 2 | 0.192 | 0.000 | +0.192 |
| 0.2–0.3 | 23 | 0.262 | 0.043 | +0.218 |
| 0.3–0.4 | 29 | 0.338 | 0.241 | +0.096 |
| 0.4–0.5 | 188 | 0.462 | 0.324 | +0.138 |
| 0.5–0.6 | 40 | 0.573 | 0.550 | +0.023 |
| 0.6–0.7 | 99 | 0.671 | 0.465 | +0.207 |
| 0.7–0.8 | 119 | 0.718 | 0.580 | +0.138 |

### Per-dataset breakdown

One averaged number hides the most useful thing in this run: self-consistency has a known failure mode — a model that is *consistently* wrong looks confident — and arithmetic is where that bites, because a wrong method reproduces the same wrong answer every sample.

| Dataset | Items | Accuracy | ECE ↓ | AUROC ↑ | AUROC (agreement) | Acc @100% | Acc @50% |
|---|---|---|---|---|---|---|---|
| gsm8k | 250 | 0.392 | 0.193 | 0.554 [0.480, 0.623] | 0.521 | 0.392 | 0.448 |
| triviaqa | 250 | 0.432 | 0.108 | 0.743 [0.680, 0.797] | 0.695 | 0.432 | 0.600 |

## Fix 2 — closing the learning loop

`engine.observe()` was called on 250 training rows (250 observations); both engines are then scored on the same 250 **held-out** rows.

| Engine | ECE ↓ | Brier ↓ | AUROC ↑ | Acc @100% |
|---|---|---|---|---|
| Prior only (as shipped) | 0.138 | 0.234 [0.218, 0.253] | 0.703 [0.637, 0.762] | 0.412 |
| Posterior (observed) | 0.085 | 0.217 [0.205, 0.230] | 0.679 [0.616, 0.741] | 0.412 |

ECE change from observing (posterior − prior, negative is better): **-0.054 [-0.117, 0.006]**.

### Credible-interval width vs effective sample size

| ESS bucket | Mean ESS | Mean 95% CI width | Items |
|---|---|---|---|
| 13.0–22.0 | 15.7 | 0.439 | 50 |
| 22.0–30.0 | 25.7 | 0.365 | 38 |
| 30.0–41.0 | 36.3 | 0.313 | 41 |
| 41.0–80.0 | 58.8 | 0.256 | 121 |

### Learning curve (held-out)

| Train observations | ECE ↓ | AUROC ↑ | Mean CI width | Mean ESS |
|---|---|---|---|---|
| 0 | 0.138 | 0.703 | 0.487 | 13.0 |
| 42 | 0.089 | 0.719 | 0.438 | 17.7 |
| 83 | 0.110 | 0.718 | 0.400 | 22.4 |
| 125 | 0.079 | 0.725 | 0.370 | 27.1 |
| 167 | 0.056 | 0.718 | 0.347 | 32.0 |
| 208 | 0.084 | 0.673 | 0.331 | 37.2 |
| 250 | 0.085 | 0.679 | 0.318 | 41.5 |

## Context coverage — is a 125-cell table the right size?

Across all 500 evaluated items the engine visited **39 of 125** contexts (31.2%); 10 contexts saw 10+ observations, and the busiest held 119.

Cells never visited stay at their prior no matter how long the system runs, so low coverage is the empirical argument for reducing dimensionality rather than growing the table — which is what `scaling/latent_bayes.py` does.

### Same rows through the latent (PCA) grid

The executor's 6 continuous signals would give 15,625 naive contexts at 5 bins each. Projected to 125 quantile-binned latent cells, **50** were visited (explained variance 0.847).

| Model | ECE ↓ | Brier ↓ | AUROC ↑ | Contexts visited |
|---|---|---|---|---|
| Hand-designed 125-cell grid | 0.085 | 0.217 [0.205, 0.230] | 0.679 [0.616, 0.741] | 39/125 |
| Latent PCA grid | 0.064 | 0.235 [0.218, 0.252] | 0.609 [0.540, 0.682] | 50/125 |

## Figures

**Reliability diagram**

![Reliability diagram](reliability.png)

**Risk-coverage curve**

![Risk-coverage curve](risk_coverage.png)

**Learning loop**

![Learning loop](learning.png)

## How to reproduce

```bash
python -m eval.run_eval --model llama --dataset bundled --samples 4
```

## Limitations

- Agreement is bag-of-words cosine, so paraphrases read as disagreement and the confidence signal is pessimistic on verbose answers.
- Grading is normalized alias containment / numeric match: a correct answer phrased unusually can be scored wrong, which shows up as apparent overconfidence.
- Self-consistency measures agreement, not truth. A model that is confidently and consistently wrong scores high here by construction.
- Risk-coverage ties are broken by row order, so the curve is slightly pessimistic when many items share a confidence — which happens whenever the evidence is a coarse 5-level ordinal.

