# Scaling to 10,000+ States

Implementation: `scaling/latent_bayes.py`.

## Why the naive approach fails

`D` noisy signals discretised into `b` bins = `b^D` contexts (`5⁸ = 390,625`). With realistic
data almost every context is visited 0–1 times, so every posterior stays pinned to its prior —
the model runs but learns nothing (curse of dimensionality on the CPT).

## Strategy: reduce, then reason

Agent signals are highly correlated, so intrinsic dimensionality is low. Exploit it:

1. **Standardise** the raw signal vector.
2. **PCA** → a few latent components (a stand-in for UMAP/autoencoder — same interface).
3. **Quantile-bin** each latent axis → a small, *dense* grid (e.g. `5³ = 125`).
4. **Conjugate Dirichlet update** in latent space — same engine as the core.

The mapping `b^D (huge) → b^k (small)` is the whole idea.

## Proof-of-concept (`python scaling/latent_bayes.py`)

8-D synthetic signals driven by 3 latent factors:

| Naive states | Latent states | Compression | Variance kept | Latent accuracy | Baseline |
|---|---|---|---|---|---|
| 390,625 (`5⁸`) | 125 (`5³`) | 3,125× | 0.89 | 0.64 | 0.22 |

The 125-state latent classifier beats the majority baseline ~3×, confirming the reduced
representation keeps the decision-relevant signal. Asserted in `tests/test_scaling.py`.

## Notes & limits

- PCA is the cheap, interpretable first choice; `explained_variance()` surfaces what's lost.
  Nonlinear manifold ⇒ swap in UMAP/autoencoder (interface unchanged).
- Quantile bins keep occupancy uniform, maximising per-cell sample size.
- Keep `b^k` in the low hundreds to preserve density.
