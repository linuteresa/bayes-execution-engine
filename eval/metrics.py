"""Calibration and selective-prediction metrics.

Everything here operates on two aligned arrays:

``scores``   -- a per-item confidence in ``[0, 1]`` (engine confidence, or a baseline).
``correct``  -- a per-item binary outcome, 1 = the answer matched gold.

Two families of metric, and the distinction matters when reading the report:

* **Calibration** (ECE, Brier) asks "when the model says 0.7, is it right 70% of the
  time?". It is scale-sensitive: a raw score that was never meant to be a probability
  (self-consistency agreement, exp(mean logprob)) can discriminate perfectly and still
  post a terrible ECE. Those baselines are reported, but compared on discrimination.
* **Discrimination** (AUROC, risk-coverage/AURC) asks "does a higher score mean more
  likely correct?". It is rank-based, so it compares raw scores and calibrated
  probabilities on equal terms. This is the fair engine-vs-baseline comparison.

No dependency beyond numpy, so the metrics are unit-testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


def _as_arrays(scores: Sequence[float], correct: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(correct, dtype=float)
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and correct {y.shape} must align")
    if s.size == 0:
        raise ValueError("empty evaluation set")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores contain NaN/inf")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("correct must be binary 0/1")
    return s, y


# --------------------------------------------------------------------- calibration
@dataclass
class ReliabilityBin:
    """One bucket of a reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    empirical_accuracy: float

    @property
    def gap(self) -> float:
        """Signed calibration gap; positive = overconfident."""
        return self.mean_confidence - self.empirical_accuracy


def reliability_bins(
    scores: Sequence[float], correct: Sequence[int], n_bins: int = 10
) -> List[ReliabilityBin]:
    """Bucket items into equal-width confidence bins and measure empirical accuracy.

    This is the raw material of the reliability diagram: a perfectly calibrated model
    puts every bin on the diagonal (``mean_confidence == empirical_accuracy``).
    Empty bins are dropped rather than reported as 0 accuracy.
    """
    s, y = _as_arrays(scores, correct)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=False puts x in bin i when edges[i-1] <= x < edges[i];
    # clip so that exactly-1.0 scores land in the last bin rather than overflowing.
    idx = np.clip(np.digitize(s, edges[1:-1], right=False), 0, n_bins - 1)

    out: List[ReliabilityBin] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(
            ReliabilityBin(
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                count=n,
                mean_confidence=float(s[mask].mean()),
                empirical_accuracy=float(y[mask].mean()),
            )
        )
    return out


def expected_calibration_error(
    scores: Sequence[float], correct: Sequence[int], n_bins: int = 10
) -> float:
    """ECE: count-weighted mean absolute gap between confidence and accuracy.

    ``ECE = sum_b (n_b / N) * |acc_b - conf_b|``. 0 is perfect; 0.5 is the worst a
    binary predictor can do. Sensitive to ``n_bins`` -- report the bin count alongside.
    """
    s, y = _as_arrays(scores, correct)
    total = 0.0
    for b in reliability_bins(s, y, n_bins):
        total += (b.count / s.size) * abs(b.gap)
    return float(total)


def maximum_calibration_error(
    scores: Sequence[float], correct: Sequence[int], n_bins: int = 10
) -> float:
    """MCE: the worst single-bin calibration gap (ignores how few items are in it)."""
    bins = reliability_bins(scores, correct, n_bins)
    return float(max((abs(b.gap) for b in bins), default=0.0))


def brier_score(scores: Sequence[float], correct: Sequence[int]) -> float:
    """Mean squared error of the confidence against the 0/1 outcome. Lower is better.

    A proper scoring rule: it rewards calibration *and* discrimination jointly, which
    is why it is reported next to ECE rather than instead of it.
    """
    s, y = _as_arrays(scores, correct)
    return float(np.mean((s - y) ** 2))


# ------------------------------------------------------------------ discrimination
def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks of ``x`` (1-based), ties receiving their average rank -- scipy's rankdata."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auroc(scores: Sequence[float], correct: Sequence[int]) -> float:
    """AUROC of the score as a detector of correctness (Mann-Whitney U, ties handled).

    0.5 = the score carries no information about whether the answer is right; 1.0 = it
    ranks every correct answer above every incorrect one. Undefined (returns NaN) when
    every item shares the same label -- a degenerate eval set with no wrong answers
    cannot test a confidence signal at all, which is why the dataset needs a spread of
    difficulty.
    """
    s, y = _as_arrays(scores, correct)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass
class RiskCoveragePoint:
    coverage: float
    accuracy: float

    @property
    def risk(self) -> float:
        return 1.0 - self.accuracy


@dataclass
class RiskCoverageCurve:
    points: List[RiskCoveragePoint] = field(default_factory=list)
    aurc: float = 0.0
    full_accuracy: float = 0.0

    def accuracy_at_coverage(self, coverage: float) -> float:
        """Accuracy retained when answering only the most-confident ``coverage`` share."""
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage must be in (0, 1]")
        best = min(self.points, key=lambda p: abs(p.coverage - coverage))
        return best.accuracy


def risk_coverage(scores: Sequence[float], correct: Sequence[int]) -> RiskCoverageCurve:
    """Selective prediction: accuracy as a function of how much you choose to answer.

    Items are ranked by confidence, then we sweep the answering threshold from "only the
    single most confident item" to "answer everything". This is the business-legible
    view: *abstain on the least-confident X%, keep accuracy Y on the rest*.

    Ties are broken by the input order via a stable sort, so the curve is deterministic
    for a given row ordering but slightly pessimistic when many scores are identical.
    """
    s, y = _as_arrays(scores, correct)
    order = np.argsort(-s, kind="mergesort")  # stable: deterministic under ties
    y_sorted = y[order]
    cum_acc = np.cumsum(y_sorted) / np.arange(1, y.size + 1)
    points = [
        RiskCoveragePoint(coverage=float(k + 1) / y.size, accuracy=float(cum_acc[k]))
        for k in range(y.size)
    ]
    return RiskCoverageCurve(
        points=points,
        aurc=float(np.mean(1.0 - cum_acc)),
        full_accuracy=float(y.mean()),
    )


# ----------------------------------------------------------------------- bootstrap
@dataclass
class Interval:
    point: float
    low: float
    high: float

    def as_dict(self) -> dict:
        return {"point": self.point, "ci_low": self.low, "ci_high": self.high}


def _bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_ci(
    statistic: Callable[[np.ndarray, np.ndarray], float],
    scores: Sequence[float],
    correct: Sequence[int],
    *,
    n_boot: int = 2000,
    mass: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI for any metric, resampling *items* with replacement.

    Resampling items (not predictions) is the right unit here: the uncertainty we care
    about is "would this number hold on another draw of questions?".
    """
    s, y = _as_arrays(scores, correct)
    point = float(statistic(s, y))
    draws = []
    for idx in _bootstrap_indices(s.size, n_boot, seed):
        try:
            draws.append(float(statistic(s[idx], y[idx])))
        except (ValueError, ZeroDivisionError):
            continue  # degenerate resample (e.g. all-correct) -- skip, don't crash
    arr = np.asarray([d for d in draws if np.isfinite(d)], dtype=float)
    if arr.size == 0:
        return Interval(point, float("nan"), float("nan"))
    tail = (1.0 - mass) / 2.0
    return Interval(point, float(np.quantile(arr, tail)), float(np.quantile(arr, 1.0 - tail)))


def bootstrap_diff(
    statistic: Callable[[np.ndarray, np.ndarray], float],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    correct: Sequence[int],
    *,
    n_boot: int = 2000,
    mass: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Paired bootstrap CI for ``statistic(a) - statistic(b)`` on the same items.

    This is the test that decides whether the 125-cell Dirichlet layer earns its
    complexity over a baseline: pairing on items removes question difficulty as a
    source of variance, so a CI that excludes 0 is real separation and one that
    straddles 0 says the extra machinery is not measurably helping.
    """
    a, y = _as_arrays(scores_a, correct)
    b, _ = _as_arrays(scores_b, correct)
    point = float(statistic(a, y) - statistic(b, y))
    draws = []
    for idx in _bootstrap_indices(a.size, n_boot, seed):
        try:
            draws.append(float(statistic(a[idx], y[idx]) - statistic(b[idx], y[idx])))
        except (ValueError, ZeroDivisionError):
            continue
    arr = np.asarray([d for d in draws if np.isfinite(d)], dtype=float)
    if arr.size == 0:
        return Interval(point, float("nan"), float("nan"))
    tail = (1.0 - mass) / 2.0
    return Interval(point, float(np.quantile(arr, tail)), float(np.quantile(arr, 1.0 - tail)))


# ------------------------------------------------------------------------ rollup
def evaluate_scores(
    scores: Sequence[float],
    correct: Sequence[int],
    *,
    n_bins: int = 10,
    n_boot: int = 2000,
    seed: int = 0,
    abstain_fraction: float = 0.2,
) -> Dict[str, object]:
    """Full metric rollup for one confidence signal."""
    s, y = _as_arrays(scores, correct)
    curve = risk_coverage(s, y)
    coverage = max(1.0 - abstain_fraction, 1.0 / s.size)
    return {
        "n": int(s.size),
        "accuracy": float(y.mean()),
        "mean_confidence": float(s.mean()),
        "ece": expected_calibration_error(s, y, n_bins),
        "mce": maximum_calibration_error(s, y, n_bins),
        "ece_bins": n_bins,
        "brier": bootstrap_ci(
            lambda a, b: brier_score(a, b), s, y, n_boot=n_boot, seed=seed
        ).as_dict(),
        "auroc": bootstrap_ci(
            lambda a, b: auroc(a, b), s, y, n_boot=n_boot, seed=seed
        ).as_dict(),
        "aurc": curve.aurc,
        "accuracy_full_coverage": curve.full_accuracy,
        "abstain_fraction": abstain_fraction,
        "accuracy_at_coverage": curve.accuracy_at_coverage(coverage),
        "reliability": [
            {
                "lower": b.lower,
                "upper": b.upper,
                "count": b.count,
                "mean_confidence": b.mean_confidence,
                "empirical_accuracy": b.empirical_accuracy,
                "gap": b.gap,
            }
            for b in reliability_bins(s, y, n_bins)
        ],
    }


__all__ = [
    "ReliabilityBin",
    "RiskCoverageCurve",
    "RiskCoveragePoint",
    "Interval",
    "reliability_bins",
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "auroc",
    "risk_coverage",
    "bootstrap_ci",
    "bootstrap_diff",
    "evaluate_scores",
]
