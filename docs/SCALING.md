# Scaling the Conflict Resolver to 10,000+ Environmental States

The core engine reasons over a hand-designed 125-context grid. That is interpretable
but does not scale with the *input* dimensionality. This document describes the
strategy and the working proof-of-concept in `scaling/latent_bayes.py`.

## 1. Why the naive approach fails

Suppose a production agent exposes `D` noisy telemetry channels (retriever scores,
HTTP statuses, schema-validation results, latencies, circuit-breaker states, …). Naive
discretisation into `b` bins each gives `b^D` contexts:

```
D = 8, b = 5   ->   5^8 = 390,625 contexts
```

To learn a Dirichlet per context you need observations *per context*. With realistic
data volumes almost every context is visited zero or one time, so every posterior stays
pinned to its prior. The model "runs" but learns nothing. This is the curse of
dimensionality applied to CPTs.

## 2. Strategy: reduce, then reason

Real agent signals are **highly correlated** — a failing tool degrades latency,
validation, *and* retriever scores simultaneously. So the intrinsic dimensionality is
far lower than `D`. We exploit that:

```
raw signals (R^D)
   │  StandardScaler            (decorrelate scale)
   ▼
   │  PCA  ->  k components      (linear manifold; UMAP/autoencoder are drop-in)
   ▼
   │  quantile binning per axis  (k axes × b bins  ->  b^k DENSE contexts)
   ▼
conjugate Dirichlet–Multinomial update   (same math as the core engine)
```

With `k = 3, b = 5` the latent grid is the same 125 contexts the core engine already
uses — but now fed by an arbitrarily high-dimensional input. The mapping
`b^D (huge) -> b^k (small)` is the whole idea.

## 3. Proof-of-concept results

`GeneralDirichletGrid` generalises the core engine to arbitrary `(n_factors, n_bins,
n_outcomes)`; `LatentBayesPipeline` chains scaler → PCA → quantile bins → grid. Running
`python scaling/latent_bayes.py` on synthetic 8-D signals driven by 3 latent factors:

| Metric | Value |
|---|---|
| Raw dimensionality `D` | 8 |
| Naive state space `5⁸` | 390,625 |
| Latent state space `5³` | 125 |
| **Compression ratio** | **3,125×** |
| Variance retained (3 PCA components) | 0.89 |
| **Latent Bayesian test accuracy** | **0.64** |
| Majority-class baseline | 0.22 |

The classifier built on the 125-state latent grid beats the majority baseline by ~3×,
confirming the reduced representation retains the decision-relevant signal. Tests in
`tests/test_scaling.py` assert the compression ratio, the variance floor, reproducibility,
and that accuracy clears the baseline by a margin.

## 4. Design notes & honest limitations

- **PCA is a placeholder for "any encoder".** The pipeline depends only on a
  `fit_transform`/`transform` interface, so UMAP, a VAE, or a learned encoder slot in
  unchanged. PCA is the right *first* choice: cheap, deterministic, and interpretable
  (explained-variance ratio tells you how much you lost).
- **Quantile binning** gives roughly uniform occupancy per latent bin, which maximises
  per-context sample size — the opposite failure mode to naive equal-width bins.
- **Linearity limit.** PCA captures linear structure; if the latent manifold is
  strongly nonlinear, explained variance drops and you should switch encoders. The
  pipeline surfaces this via `explained_variance()` so the limitation is observable,
  not hidden.
- **Bins vs. outcomes.** Latent grids need not be `5³`; `GeneralDirichletGrid` supports
  any shape. Keep `b^k` in the low hundreds to preserve density.

## 5. When to use which

| Situation | Approach |
|---|---|
| ≤ 3 interpretable, independent signals | Core 125-state engine directly. |
| Many correlated noisy channels | `LatentBayesPipeline` (PCA → 125-state grid). |
| Strongly nonlinear signal manifold | Same pipeline, swap PCA for UMAP/autoencoder. |
