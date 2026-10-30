# Calibration report — SIMULATED (not a result)

> **⚠️ SIMULATED RUN — NOT A CALIBRATION RESULT.**
> These numbers come from `eval.answerers.SimulatedAnswerer`, a deterministic
> stand-in that is *told the gold answer* and fabricates samples from it. The run
> exists to prove the harness computes its metrics correctly end to end with no
> model available. It says nothing about any real model's calibration. For a
> reportable number, start `llama-server` and re-run with `--model llama`.


**Scope: single-step answer confidence.** Each question is evaluated as one execution step. These metrics do not validate confidence in a multi-step final response, the replanner, or the agent's end-to-end stopping policy.

**Analysis only:** this report recomputes metrics from saved answers and labels. Stored confidence, evidence, samples and correctness labels are preserved; no model or embedding requests were made. Risk-coverage now uses expected uniform selection within equal-confidence groups.

## Provenance

| Field | Value |
|---|---|
| Generated (UTC) | 2026-10-28T18:51:42Z |
| Answerer | `simulated` |
| Model | `simulated-answerer` |
| Simulated | **yes — see banner** |
| Dataset spec | `bundled` |
| Items | 80 |
| Samples per item | 4 |
| Agreement measure | `word-overlap` |
| Provenance schema | 2 |
| Experiment ID | `7d78fdf423e105546305cdfd2769ae58eb141838d71438c8e390e3609911b3f3` |
| Provenance status | versioned |

Composition: arithmetic × 20, easy × 20, hard × 20, medium × 20.

## Fix 1 — is the confidence calibrated?

ECE and Brier are scale-sensitive, so a raw score never meant to be a probability can discriminate well and still calibrate badly. AUROC and AURC are rank-based and compare the engine and the baselines on equal terms — read those for *does the number know anything*, and ECE for *does the number mean what it says*. Brackets are 95% bootstrap CIs over items.

| Confidence signal | ECE ↓ | Brier ↓ | AUROC ↑ | AURC ↓ | Acc @100% | Acc @80% |
|---|---|---|---|---|---|---|
| Bayesian engine (125-cell Dirichlet) | 0.440 | 0.364 [0.330, 0.399] | 0.600 [0.448, 0.756] | 0.285 | 0.738 | 0.808 |
| Baseline: self-consistency agreement | 0.314 | 0.201 [0.166, 0.242] | 0.877 [0.750, 0.975] | 0.097 | 0.738 | 0.859 |
| Baseline: mean token logprob | 0.202 | 0.195 [0.166, 0.225] | 0.789 [0.652, 0.908] | 0.122 | 0.738 | 0.828 |

### Verdict

- Confidence separates correct from incorrect answers at AUROC 0.600.
- ECE is 0.440 over 10 bins; the engine is underconfident on average (mean confidence 0.327 vs accuracy 0.738).
- Abstaining on the least-confident 20% lifts accuracy on the remainder from 0.738 to 0.808.
- The engine is **worse** than *Baseline: self-consistency agreement* on AUROC by -0.276 [-0.393, -0.162] (CI excludes 0). The extra machinery is costing discrimination here.
- The engine is **worse** than *Baseline: mean token logprob* on AUROC by -0.189 [-0.352, -0.015] (CI excludes 0). The extra machinery is costing discrimination here.

### Reliability bins (engine)

| Confidence bin | Items | Mean confidence | Empirical accuracy | Gap |
|---|---|---|---|---|
| 0.2–0.3 | 17 | 0.244 | 0.471 | -0.227 |
| 0.3–0.4 | 59 | 0.338 | 0.847 | -0.510 |
| 0.4–0.5 | 2 | 0.456 | 0.500 | -0.044 |
| 0.5–0.6 | 2 | 0.573 | 0.000 | +0.573 |

## Fix 2 — closing the learning loop

`engine.observe()` was called on 40 training rows (40 observations); both engines are then scored on the same 40 **held-out** rows.

| Engine | ECE ↓ | Brier ↓ | AUROC ↑ | Acc @100% |
|---|---|---|---|---|
| Prior only (as shipped) | 0.398 | 0.353 [0.300, 0.400] | 0.607 [0.427, 0.799] | 0.725 |
| Posterior (observed) | 0.207 | 0.099 [0.068, 0.140] | 0.958 [0.887, 1.000] | 0.725 |

ECE change from observing (posterior − prior, negative is better): **-0.191 [-0.308, -0.049]**.

### Credible-interval width vs effective sample size

| ESS bucket | Mean ESS | Mean 95% CI width | Items |
|---|---|---|---|
| 13.0–16.8 | 15.2 | 0.439 | 8 |
| 16.8–38.0 | 33.7 | 0.292 | 32 |

### Learning curve (held-out)

| Train observations | ECE ↓ | AUROC ↑ | Mean CI width | Mean ESS |
|---|---|---|---|---|
| 0 | 0.398 | 0.607 | 0.478 | 13.0 |
| 7 | 0.382 | 0.920 | 0.446 | 16.4 |
| 13 | 0.312 | 0.958 | 0.412 | 19.1 |
| 20 | 0.261 | 0.958 | 0.372 | 22.9 |
| 27 | 0.242 | 0.958 | 0.351 | 25.2 |
| 33 | 0.234 | 0.958 | 0.346 | 26.2 |
| 40 | 0.207 | 0.958 | 0.321 | 30.0 |

## Context coverage — is a 125-cell table the right size?

Across all 80 evaluated items the engine visited **7 of 125** contexts (5.6%); 2 contexts saw 10+ observations, and the busiest held 50.

Cells never visited stay at their prior no matter how long the system runs, so low coverage is the empirical argument for reducing dimensionality rather than growing the table — which is what `scaling/latent_bayes.py` does.

### Same rows through the latent (PCA) grid

The executor's 6 continuous signals would give 15,625 naive contexts at 5 bins each. Projected to 125 quantile-binned latent cells, **11** were visited (explained variance 0.988).

| Model | ECE ↓ | Brier ↓ | AUROC ↑ | Contexts visited |
|---|---|---|---|---|
| Hand-designed 125-cell grid | 0.207 | 0.099 [0.068, 0.140] | 0.958 [0.887, 1.000] | 5/125 (train) |
| Latent PCA grid | 0.194 | 0.079 [0.054, 0.109] | 0.991 [0.962, 1.000] | 11/125 (train) |

## Figures

**Reliability diagram**

![Reliability diagram](reliability.png)

**Risk-coverage curve**

![Risk-coverage curve](risk_coverage.png)

**Learning loop**

![Learning loop](learning.png)

## How to reproduce

```bash
python -m eval.run_eval --rows eval/results/example-simulated/rows.jsonl --bins 10 --bootstrap 2000 --seed 0 --abstain 0.2 --test-fraction 0.5
```

## Limitations

- Agreement used word-overlap cosine; paraphrases can read as disagreement and contradictions with similar wording can look consistent.
- Grading is normalized alias containment / numeric match: a correct answer phrased unusually can be scored wrong, which shows up as apparent overconfidence.
- Self-consistency measures agreement, not truth. A model that is confidently and consistently wrong scores high here by construction.
- Intermediate risk-coverage points assume uniform random selection within a boundary tie group. A confidence threshold alone cannot separate equal scores.

