# Calibration report

## Provenance

| Field | Value |
|---|---|
| Generated (UTC) | 2026-09-16T00:57:21Z |
| Answerer | `local-openai-compatible` |
| Model | `qwen2.5:3b-instruct` |
| Simulated | no |
| Dataset spec | `hf:triviaqa+hf:gsm8k` |
| Items | 500 |
| Samples per item | 4 |

Composition: arithmetic × 250, mixed × 250.

## Fix 1 — is the confidence calibrated?

ECE and Brier are scale-sensitive, so a raw score never meant to be a probability can discriminate well and still calibrate badly. AUROC and AURC are rank-based and compare the engine and the baselines on equal terms — read those for *does the number know anything*, and ECE for *does the number mean what it says*. Brackets are 95% bootstrap CIs over items.

| Confidence signal | ECE ↓ | Brier ↓ | AUROC ↑ | AURC ↓ | Acc @100% | Acc @80% |
|---|---|---|---|---|---|---|
| Bayesian engine (125-cell Dirichlet) | 0.166 | 0.233 [0.219, 0.246] | 0.660 [0.610, 0.711] | 0.559 | 0.318 | 0.372 |
| Baseline: self-consistency agreement | 0.399 | 0.377 [0.346, 0.407] | 0.684 [0.636, 0.733] | 0.563 | 0.318 | 0.375 |
| Baseline: mean token logprob | 0.458 | 0.401 [0.372, 0.429] | 0.733 [0.683, 0.779] | 0.514 | 0.318 | 0.370 |

### Verdict

- Confidence separates correct from incorrect answers at AUROC 0.660.
- ECE is 0.166 over 10 bins; the engine is overconfident on average (mean confidence 0.484 vs accuracy 0.318).
- Abstaining on the least-confident 20% lifts accuracy on the remainder from 0.318 to 0.372.
- The engine is **not** measurably better than *Baseline: self-consistency agreement* on AUROC (-0.024 [-0.054, 0.004]; CI straddles 0). On this dataset the 125-cell layer is not adding discrimination over the raw signal.
- The engine is **worse** than *Baseline: mean token logprob* on AUROC by -0.072 [-0.107, -0.037] (CI excludes 0). The extra machinery is costing discrimination here.

### Reliability bins (engine)

| Confidence bin | Items | Mean confidence | Empirical accuracy | Gap |
|---|---|---|---|---|
| 0.1–0.2 | 21 | 0.180 | 0.048 | +0.132 |
| 0.2–0.3 | 62 | 0.255 | 0.145 | +0.109 |
| 0.3–0.4 | 59 | 0.338 | 0.237 | +0.100 |
| 0.4–0.5 | 173 | 0.455 | 0.324 | +0.132 |
| 0.5–0.6 | 33 | 0.573 | 0.364 | +0.209 |
| 0.6–0.7 | 88 | 0.671 | 0.341 | +0.330 |
| 0.7–0.8 | 64 | 0.718 | 0.578 | +0.140 |

### Per-dataset breakdown

One averaged number hides the most useful thing in this run: self-consistency has a known failure mode — a model that is *consistently* wrong looks confident — and arithmetic is where that bites, because a wrong method reproduces the same wrong answer every sample.

| Dataset | Items | Accuracy | ECE ↓ | AUROC ↑ | AUROC (agreement) | Acc @100% | Acc @50% |
|---|---|---|---|---|---|---|---|
| gsm8k | 250 | 0.220 | 0.270 | 0.568 [0.482, 0.647] | 0.606 | 0.220 | 0.236 |
| triviaqa | 250 | 0.416 | 0.074 | 0.741 [0.679, 0.804] | 0.741 | 0.416 | 0.566 |

## Fix 2 — closing the learning loop

`engine.observe()` was called on 250 training rows (250 observations); both engines are then scored on the same 250 **held-out** rows.

| Engine | ECE ↓ | Brier ↓ | AUROC ↑ | Acc @100% |
|---|---|---|---|---|
| Prior only (as shipped) | 0.176 | 0.224 [0.206, 0.243] | 0.700 [0.636, 0.764] | 0.316 |
| Posterior (observed) | 0.095 | 0.205 [0.187, 0.223] | 0.665 [0.590, 0.733] | 0.316 |

ECE change from observing (posterior − prior, negative is better): **-0.081 [-0.100, -0.043]**.

### Credible-interval width vs effective sample size

| ESS bucket | Mean ESS | Mean 95% CI width | Items |
|---|---|---|---|
| 13.0–18.0 | 15.5 | 0.434 | 48 |
| 18.0–24.0 | 20.4 | 0.395 | 45 |
| 24.0–33.0 | 25.3 | 0.372 | 47 |
| 33.0–47.0 | 38.5 | 0.308 | 52 |
| 47.0–55.0 | 50.9 | 0.266 | 58 |

### Learning curve (held-out)

| Train observations | ECE ↓ | AUROC ↑ | Mean CI width | Mean ESS |
|---|---|---|---|---|
| 0 | 0.176 | 0.700 | 0.485 | 13.0 |
| 42 | 0.145 | 0.695 | 0.436 | 15.6 |
| 83 | 0.112 | 0.693 | 0.410 | 19.3 |
| 125 | 0.100 | 0.700 | 0.392 | 22.2 |
| 167 | 0.109 | 0.671 | 0.374 | 25.5 |
| 208 | 0.129 | 0.670 | 0.362 | 28.1 |
| 250 | 0.095 | 0.665 | 0.350 | 31.2 |

## Context coverage — is a 125-cell table the right size?

Across all 500 evaluated items the engine visited **39 of 125** contexts (31.2%); 12 contexts saw 10+ observations, and the busiest held 70.

Cells never visited stay at their prior no matter how long the system runs, so low coverage is the empirical argument for reducing dimensionality rather than growing the table — which is what `scaling/latent_bayes.py` does.

### Same rows through the latent (PCA) grid

The executor's 6 continuous signals would give 15,625 naive contexts at 5 bins each. Projected to 125 quantile-binned latent cells, **66** were visited (explained variance 0.867).

| Model | ECE ↓ | Brier ↓ | AUROC ↑ | Contexts visited |
|---|---|---|---|---|
| Hand-designed 125-cell grid | 0.095 | 0.205 [0.187, 0.223] | 0.665 [0.590, 0.733] | 39/125 |
| Latent PCA grid | 0.067 | 0.208 [0.189, 0.227] | 0.649 [0.577, 0.720] | 66/125 |

## Figures

**Reliability diagram**

![Reliability diagram](reliability.png)

**Risk-coverage curve**

![Risk-coverage curve](risk_coverage.png)

**Learning loop**

![Learning loop](learning.png)

## How to reproduce

```bash
python -m eval.run_eval --model llama --dataset hf:triviaqa+hf:gsm8k --samples 4
```

## Limitations

- Agreement is bag-of-words cosine, so paraphrases read as disagreement and the confidence signal is pessimistic on verbose answers.
- Grading is normalized alias containment / numeric match: a correct answer phrased unusually can be scored wrong, which shows up as apparent overconfidence.
- Self-consistency measures agreement, not truth. A model that is confidently and consistently wrong scores high here by construction.
- Risk-coverage ties are broken by row order, so the curve is slightly pessimistic when many items share a confidence — which happens whenever the evidence is a coarse 5-level ordinal.

