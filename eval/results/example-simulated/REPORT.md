# Calibration report — SIMULATED (not a result)

> **⚠️ SIMULATED RUN — NOT A CALIBRATION RESULT.**
> These numbers come from `eval.answerers.SimulatedAnswerer`, a deterministic
> stand-in that is *told the gold answer* and fabricates samples from it. The run
> exists to prove the harness computes its metrics correctly end to end with no
> model available. It says nothing about any real model's calibration. For a
> reportable number, start `llama-server` and re-run with `--model llama`.


## Provenance

| Field | Value |
|---|---|
| Generated (UTC) | 2026-09-16T00:08:54Z |
| Answerer | `simulated` |
| Model | `simulated-answerer` |
| Simulated | **yes — see banner** |
| Dataset spec | `bundled` |
| Items | 80 |
| Samples per item | 4 |

Composition: arithmetic × 20, easy × 20, hard × 20, medium × 20.

## Fix 1 — is the confidence calibrated?

ECE and Brier are scale-sensitive, so a raw score never meant to be a probability can discriminate well and still calibrate badly. AUROC and AURC are rank-based and compare the engine and the baselines on equal terms — read those for *does the number know anything*, and ECE for *does the number mean what it says*. Brackets are 95% bootstrap CIs over items.

| Confidence signal | ECE ↓ | Brier ↓ | AUROC ↑ | AURC ↓ | Acc @100% | Acc @80% |
|---|---|---|---|---|---|---|
| Bayesian engine (125-cell Dirichlet) | 0.448 | 0.363 [0.329, 0.396] | 0.607 [0.455, 0.761] | 0.297 | 0.738 | 0.812 |
| Baseline: self-consistency agreement | 0.294 | 0.198 [0.170, 0.231] | 0.876 [0.747, 0.975] | 0.097 | 0.738 | 0.875 |
| Baseline: mean token logprob | 0.202 | 0.195 [0.166, 0.225] | 0.789 [0.652, 0.908] | 0.122 | 0.738 | 0.828 |

### Verdict

- Confidence separates correct from incorrect answers at AUROC 0.607.
- ECE is 0.448 over 10 bins; the engine is underconfident on average (mean confidence 0.332 vs accuracy 0.738).
- Abstaining on the least-confident 20% lifts accuracy on the remainder from 0.738 to 0.812.
- The engine is **worse** than *Baseline: self-consistency agreement* on AUROC by -0.269 [-0.380, -0.156] (CI excludes 0). The extra machinery is costing discrimination here.
- The engine is **worse** than *Baseline: mean token logprob* on AUROC by -0.182 [-0.345, -0.005] (CI excludes 0). The extra machinery is costing discrimination here.

### Reliability bins (engine)

| Confidence bin | Items | Mean confidence | Empirical accuracy | Gap |
|---|---|---|---|---|
| 0.2–0.3 | 16 | 0.260 | 0.438 | -0.177 |
| 0.3–0.4 | 60 | 0.338 | 0.850 | -0.512 |
| 0.4–0.5 | 1 | 0.415 | 1.000 | -0.585 |
| 0.5–0.6 | 3 | 0.573 | 0.000 | +0.573 |

## Fix 2 — closing the learning loop

`engine.observe()` was called on 40 training rows (40 observations); both engines are then scored on the same 40 **held-out** rows.

| Engine | ECE ↓ | Brier ↓ | AUROC ↑ | Acc @100% |
|---|---|---|---|---|
| Prior only (as shipped) | 0.423 | 0.353 [0.301, 0.398] | 0.613 [0.429, 0.805] | 0.725 |
| Posterior (observed) | 0.236 | 0.094 [0.064, 0.131] | 0.983 [0.940, 1.000] | 0.725 |

ECE change from observing (posterior − prior, negative is better): **-0.187 [-0.300, -0.044]**.

### Credible-interval width vs effective sample size

| ESS bucket | Mean ESS | Mean 95% CI width | Items |
|---|---|---|---|
| 13.0–16.0 | 14.0 | 0.453 | 5 |
| 16.0–39.0 | 33.0 | 0.314 | 35 |

### Learning curve (held-out)

| Train observations | ECE ↓ | AUROC ↑ | Mean CI width | Mean ESS |
|---|---|---|---|---|
| 0 | 0.423 | 0.613 | 0.479 | 13.0 |
| 7 | 0.379 | 0.848 | 0.438 | 16.4 |
| 13 | 0.379 | 0.983 | 0.419 | 19.1 |
| 20 | 0.235 | 0.926 | 0.387 | 23.0 |
| 27 | 0.304 | 0.983 | 0.367 | 25.3 |
| 33 | 0.287 | 0.983 | 0.356 | 26.9 |
| 40 | 0.236 | 0.983 | 0.332 | 30.6 |

## Context coverage — is a 125-cell table the right size?

Across all 80 evaluated items the engine visited **6 of 125** contexts (4.8%); 2 contexts saw 10+ observations, and the busiest held 51.

Cells never visited stay at their prior no matter how long the system runs, so low coverage is the empirical argument for reducing dimensionality rather than growing the table — which is what `scaling/latent_bayes.py` does.

### Same rows through the latent (PCA) grid

The executor's 6 continuous signals would give 15,625 naive contexts at 5 bins each. Projected to 125 quantile-binned latent cells, **10** were visited (explained variance 0.987).

| Model | ECE ↓ | Brier ↓ | AUROC ↑ | Contexts visited |
|---|---|---|---|---|
| Hand-designed 125-cell grid | 0.236 | 0.094 [0.064, 0.131] | 0.983 [0.940, 1.000] | 6/125 |
| Latent PCA grid | 0.174 | 0.076 [0.051, 0.105] | 0.991 [0.969, 1.000] | 10/125 |

## Figures

**Reliability diagram**

![Reliability diagram](reliability.png)

**Risk-coverage curve**

![Risk-coverage curve](risk_coverage.png)

**Learning loop**

![Learning loop](learning.png)

## How to reproduce

```bash
python -m eval.run_eval --model sim --dataset bundled --samples 4
```

## Limitations

- Agreement is bag-of-words cosine, so paraphrases read as disagreement and the confidence signal is pessimistic on verbose answers.
- Grading is normalized alias containment / numeric match: a correct answer phrased unusually can be scored wrong, which shows up as apparent overconfidence.
- Self-consistency measures agreement, not truth. A model that is confidently and consistently wrong scores high here by construction.
- Risk-coverage ties are broken by row order, so the curve is slightly pessimistic when many items share a confidence — which happens whenever the evidence is a coarse 5-level ordinal.

