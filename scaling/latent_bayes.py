"""
Scaling the Bayesian conflict resolver from 125 to 10,000+ environmental states.

The core engine reasons over a hand-designed 3-signal / 5-state grid (125 contexts).
That is interpretable but does not scale: if a production agent exposes, say, 8 noisy
telemetry signals discretized into 5 bins each, the naive context space explodes to
``5**8 = 390,625`` states. Estimating a Dirichlet per state is then hopeless -- almost
every state is visited zero or one time, so the posterior never moves off its prior.

Strategy: *reduce, then reason*.
--------------------------------
1. Standardize the raw, high-dimensional, correlated signal vector.
2. Project it onto a low-dimensional latent manifold with PCA (a stand-in for any
   manifold-learning method -- UMAP / autoencoder / random projection all slot in
   behind the same interface). Real agent signals are highly correlated, so a handful
   of components captures most of the variance.
3. Discretize each latent axis into quantile bins, yielding a small, *dense* context
   grid (e.g. 3 components x 5 bins = the same 125 states the core engine already uses).
4. Run the conjugate Dirichlet-Multinomial update in that latent grid.

This keeps the statistical strength of conjugacy (each latent state now receives many
observations) while letting the *input* dimensionality grow arbitrarily. The mapping
``raw_states (10^4+)  ->  latent_states (10^2)`` is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Optional, Tuple

import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass
class GeneralDirichletGrid:
    """Conjugate Dirichlet-Multinomial model over an arbitrary discrete context grid.

    Generalizes the core engine: ``n_factors`` discrete factors, ``n_bins`` states
    each, predicting one of ``n_outcomes`` outcomes. Storage is dense over
    ``n_bins ** n_factors`` contexts -- which is exactly why we keep that number small
    by reducing dimensionality first.
    """

    n_factors: int
    n_bins: int = 5
    n_outcomes: int = 5
    prior: float = 1.0
    alpha: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.n_factors = _positive_integer(self.n_factors, "n_factors")
        self.n_bins = _positive_integer(self.n_bins, "n_bins")
        self.n_outcomes = _positive_integer(self.n_outcomes, "n_outcomes")
        if (isinstance(self.prior, (bool, np.bool_)) or not isinstance(self.prior, Real)
                or not np.isfinite(self.prior) or self.prior <= 0):
            raise ValueError("prior must be finite and positive")
        if not np.isfinite(self.prior * self.n_outcomes):
            raise ValueError("prior overflows the concentration sum")
        self.n_contexts = self.n_bins ** self.n_factors
        self.alpha = np.full((self.n_contexts, self.n_outcomes), self.prior, dtype=float)

    def _ctx(self, bins: np.ndarray) -> int:
        bins = np.asarray(bins)
        if bins.shape != (self.n_factors,) or bins.dtype.kind not in "iu":
            raise ValueError(f"bins must contain exactly {self.n_factors} integer indices")
        if ((bins < 0) | (bins >= self.n_bins)).any():
            raise ValueError(f"bin indices must be in 0..{self.n_bins - 1}")
        idx = 0
        for b in bins:
            idx = idx * self.n_bins + int(b)
        return idx

    def observe(self, bins: np.ndarray, outcome: int) -> None:
        if (isinstance(outcome, (bool, np.bool_)) or not isinstance(outcome, Integral)
                or not 0 <= outcome < self.n_outcomes):
            raise ValueError(f"outcome must be an integer in 0..{self.n_outcomes - 1}")
        self.alpha[self._ctx(bins), int(outcome)] += 1.0

    def predictive(self, bins: np.ndarray) -> np.ndarray:
        a = self.alpha[self._ctx(bins)]
        return a / a.sum()

    def predict(self, bins: np.ndarray) -> int:
        return int(np.argmax(self.predictive(bins)))


@dataclass
class LatentBayesPipeline:
    """High-dim signals -> PCA latent grid -> conjugate Bayesian outcome model."""

    n_components: int = 3
    n_bins: int = 5
    n_outcomes: int = 5
    random_state: int = 0

    def __post_init__(self) -> None:
        if not _HAVE_SKLEARN:  # pragma: no cover
            raise ImportError("scikit-learn is required for the scaling pipeline")
        self.n_components = _positive_integer(self.n_components, "n_components")
        self.n_bins = _positive_integer(self.n_bins, "n_bins")
        self.n_outcomes = _positive_integer(self.n_outcomes, "n_outcomes")
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.grid = GeneralDirichletGrid(self.n_components, self.n_bins, self.n_outcomes)
        self._edges: Optional[list] = None

    # ---------------------------------------------------------------- fitting
    def fit_projection(self, X: np.ndarray) -> "LatentBayesPipeline":
        """Learn the projection and reset counts whose bin meanings have changed.

        A failed fit leaves the previous fitted model intact.
        """
        X = np.asarray(X)
        if X.ndim != 2 or X.dtype.kind not in "iuf" or not np.isfinite(X).all():
            raise ValueError("signals must be a finite real-valued matrix")
        scaler = StandardScaler()
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        Z = pca.fit_transform(scaler.fit_transform(X))
        qs = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        edges = [np.percentile(Z[:, i], qs) for i in range(self.n_components)]
        self.scaler, self.pca, self._edges = scaler, pca, edges
        self.grid = GeneralDirichletGrid(self.n_components, self.n_bins, self.n_outcomes)
        return self

    def to_bins(self, X: np.ndarray) -> np.ndarray:
        """Project raw signals to discrete latent bin indices, shape (n, n_components)."""
        if self._edges is None:
            raise ValueError("fit_projection must be called before projecting signals")
        Z = self.pca.transform(self.scaler.transform(np.atleast_2d(X)))
        bins = np.empty_like(Z, dtype=int)
        for i in range(self.n_components):
            bins[:, i] = np.clip(np.digitize(Z[:, i], self._edges[i]), 0, self.n_bins - 1)
        return bins

    def update(self, X: np.ndarray, outcomes: np.ndarray) -> "LatentBayesPipeline":
        """Conjugate update from observed (signal, outcome) pairs."""
        bins = self.to_bins(X)
        outcomes = np.asarray(outcomes)
        if outcomes.shape != (len(bins),) or outcomes.dtype.kind not in "iu":
            raise ValueError("outcomes must contain one integer label per signal row")
        if ((outcomes < 0) | (outcomes >= self.n_outcomes)).any():
            raise ValueError(f"outcomes must be in 0..{self.n_outcomes - 1}")
        for b, y in zip(bins, outcomes, strict=True):
            self.grid.observe(b, int(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.grid.predict(b) for b in self.to_bins(X)])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Posterior-predictive distribution per row, shape ``(n, n_outcomes)``.

        The calibration eval needs a *probability*, not an argmax label: it scores
        ``P(good outcome)`` from the latent grid against the same held-out items the
        125-cell engine is scored on (``eval/learning.py``). That comparison is what
        turns this module from a standalone PoC into a measured alternative.
        """
        return np.array([self.grid.predictive(b) for b in self.to_bins(X)])

    def coverage(self) -> int:
        """Latent contexts with at least one observation (vs ``grid.n_contexts``)."""
        return int((self.grid.alpha > self.grid.prior).any(axis=1).sum())

    # -------------------------------------------------------------- reporting
    def explained_variance(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())

    def state_reduction(self, raw_dim: int) -> Tuple[int, int]:
        """Return (naive_raw_states, latent_states) for a given raw dimensionality."""
        return self.n_bins ** raw_dim, self.grid.n_contexts


# --------------------------------------------------------------------------
# Synthetic data + runnable demo.
# --------------------------------------------------------------------------
def make_synthetic_agent_signals(
    n_samples: int = 4000, raw_dim: int = 8, latent_dim: int = 3, noise: float = 0.6, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate correlated high-dim signals driven by a few latent degradation factors.

    The label is a discretized function of the latent factors, so a good pipeline
    should recover it after projecting back down. Mirrors reality: many telemetry
    channels, few underlying causes.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_samples, latent_dim))
    mixing = rng.normal(size=(latent_dim, raw_dim))
    X = latent @ mixing + noise * rng.normal(size=(n_samples, raw_dim))
    severity = latent.sum(axis=1)
    edges = np.percentile(severity, [20, 40, 60, 80])
    y = np.digitize(severity, edges)  # 5 ordinal outcome classes
    return X, y


def run_demo() -> dict:
    X, y = make_synthetic_agent_signals()
    n = len(X)
    split = int(0.7 * n)
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

    pipe = LatentBayesPipeline(n_components=3, n_bins=5).fit_projection(Xtr)
    pipe.update(Xtr, ytr)
    preds = pipe.predict(Xte)

    acc = float((preds == yte).mean())
    raw_states, latent_states = pipe.state_reduction(raw_dim=X.shape[1])
    report = {
        "raw_dim": X.shape[1],
        "naive_raw_states": raw_states,
        "latent_states": latent_states,
        "compression_ratio": raw_states / latent_states,
        "explained_variance": round(pipe.explained_variance(), 3),
        "test_accuracy": round(acc, 3),
        "majority_baseline": round(float(np.bincount(yte).max() / len(yte)), 3),
    }
    return report


if __name__ == "__main__":
    import json

    print(json.dumps(run_demo(), indent=2))
