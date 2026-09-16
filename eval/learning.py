"""Close the loop: actually call ``engine.observe`` and measure whether it helps.

Before this module the engine shipped with ``observe`` implemented, documented, unit
tested -- and never called outside tests. The posterior was always the prior, so
``resolve_conflict`` was a deterministic closed-form function of three ordinals: a
defensible interpretable scoring function, but not a *learned* one, and the conjugacy
argument was unexercised.

What happens here, all on rows already collected by :mod:`eval.harness` (no extra model
calls -- the evidence triples are replayable):

1. Split the rows into train/test.
2. Feed the train half back through ``observe`` with
   :func:`eval.harness.outcome_label` as the outcome.
3. Score prior-only and posterior confidences on the **held-out** half, so the
   comparison is honest rather than the posterior grading its own training data.
4. Report credible-interval width against ESS, which is the visible form of the
   "conjugacy gives free uncertainty that shrinks with data" claim.
5. Report context coverage -- how many of the 125 cells were ever visited. A few
   hundred items cannot populate 125 contexts, and saying so is the empirical case for
   the dimensionality reduction in ``scaling/latent_bayes.py``, which is fitted on the
   same rows here for a like-for-like comparison.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from bayesian_engine.bayes_engine import (
    N_CONTEXTS,
    DirichletBayesianEngine,
    context_index,
    good_probability,
)
from eval.harness import SIGNAL_KEYS, EvalRow, outcome_label

# Outcome states counted as "good" by the reported confidence, P(CERTAIN) + P(HIGH).
_GOOD_STATES = (0, 1)


@dataclass
class Split:
    train: List[EvalRow]
    test: List[EvalRow]


def split_rows(
    rows: Sequence[EvalRow], *, test_fraction: float = 0.5, seed: int = 0
) -> Split:
    """Shuffle and split, stratified by correctness.

    Stratifying keeps the train and test halves at comparable accuracy; an unstratified
    split on a few hundred items can easily hand the test half most of the wrong
    answers and make the posterior look worse than it is.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    rng = random.Random(seed)
    train: List[EvalRow] = []
    test: List[EvalRow] = []
    for label in (0, 1):
        bucket = [r for r in rows if r.correct == label]
        rng.shuffle(bucket)
        cut = int(round(len(bucket) * (1.0 - test_fraction)))
        train.extend(bucket[:cut])
        test.extend(bucket[cut:])
    rng.shuffle(train)
    rng.shuffle(test)
    if not train or not test:
        raise ValueError("split produced an empty half; need more rows")
    return Split(train=train, test=test)


def fit_engine(
    train: Sequence[EvalRow], *, prior_strength: float = 8.0, prior_spread: float = 1.0
) -> DirichletBayesianEngine:
    """Conjugate-update a fresh engine from labelled rows. This is the learning step."""
    engine = DirichletBayesianEngine(
        prior_strength=prior_strength, prior_spread=prior_spread
    )
    for row in train:
        engine.observe(
            row.task_status,
            row.data_quality,
            row.tool_reliability,
            outcome_label(row.correct),
        )
    return engine


def score_rows(rows: Sequence[EvalRow], engine: DirichletBayesianEngine) -> Dict[str, list]:
    """Replay rows through an engine; returns confidences, CI widths, ESS, labels."""
    confidences, widths, ess, correct = [], [], [], []
    for row in rows:
        summary = engine.resolve(
            {
                "TaskStatus": row.task_status,
                "DataQuality": row.data_quality,
                "ToolReliability": row.tool_reliability,
            }
        ).as_dict()
        low, high = summary["credible_interval"]
        confidences.append(good_probability(summary))
        widths.append(float(high) - float(low))
        ess.append(float(summary["effective_sample_size"]))
        correct.append(int(row.correct))
    return {
        "confidence": confidences,
        "ci_width": widths,
        "ess": ess,
        "correct": correct,
    }


# ------------------------------------------------------------------- coverage
def context_coverage(rows: Sequence[EvalRow]) -> Dict[str, object]:
    """How much of the 125-cell table a run of this size actually touches.

    The honest framing for the README: if N items only ever land in a handful of
    contexts, then most cells stay at their prior no matter how much data arrives, and
    the table's granularity is doing less work than its size suggests.
    """
    counts = np.zeros(N_CONTEXTS, dtype=int)
    for row in rows:
        counts[context_index(row.task_status, row.data_quality, row.tool_reliability)] += 1
    visited = int((counts > 0).sum())
    occupied = counts[counts > 0]
    top = sorted(
        ((int(i), int(c)) for i, c in enumerate(counts) if c > 0),
        key=lambda kv: -kv[1],
    )[:10]
    return {
        "n_contexts": N_CONTEXTS,
        "contexts_visited": visited,
        "contexts_visited_fraction": visited / N_CONTEXTS,
        "n_observations": int(counts.sum()),
        "max_count_in_one_context": int(occupied.max()) if occupied.size else 0,
        "median_count_in_visited": float(np.median(occupied)) if occupied.size else 0.0,
        "contexts_with_at_least_10": int((counts >= 10).sum()),
        "top_contexts": [{"context_index": i, "count": c} for i, c in top],
    }


def ci_width_vs_ess(scored: Dict[str, list], n_buckets: int = 5) -> List[Dict[str, float]]:
    """Bucket held-out rows by ESS and report mean credible-interval width.

    The expected shape is monotone decreasing: more observations in a context means a
    tighter interval. That is the measurable form of "alpha_0 IS the effective sample
    size", as opposed to an architectural assertion about it.
    """
    ess = np.asarray(scored["ess"], dtype=float)
    widths = np.asarray(scored["ci_width"], dtype=float)
    if ess.size == 0:
        return []
    edges = np.unique(np.quantile(ess, np.linspace(0, 1, n_buckets + 1)))
    if edges.size < 2:
        return [
            {
                "ess_low": float(ess.min()),
                "ess_high": float(ess.max()),
                "mean_ess": float(ess.mean()),
                "mean_ci_width": float(widths.mean()),
                "count": int(ess.size),
            }
        ]
    out = []
    for b in range(edges.size - 1):
        lo, hi = edges[b], edges[b + 1]
        mask = (ess >= lo) & (ess <= hi if b == edges.size - 2 else ess < hi)
        if not mask.any():
            continue
        out.append(
            {
                "ess_low": float(lo),
                "ess_high": float(hi),
                "mean_ess": float(ess[mask].mean()),
                "mean_ci_width": float(widths[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return out


# --------------------------------------------------------------- latent variant
def _signal_matrix(rows: Sequence[EvalRow]) -> np.ndarray:
    return np.asarray(
        [[float(r.signals.get(k, 0.0)) for k in SIGNAL_KEYS] for r in rows], dtype=float
    )


def latent_confidences(
    train: Sequence[EvalRow],
    test: Sequence[EvalRow],
    *,
    n_components: int = 3,
    n_bins: int = 5,
) -> Optional[Dict[str, object]]:
    """Fit the PCA latent grid on the same rows and score the same held-out items.

    Instead of hand-binning three signals into 125 hand-chosen cells, this standardizes
    the six *continuous* measurements the executor already produces, projects them onto
    a few latent axes and quantile-bins those into a dense grid. Quantile bins are the
    point: they are populated by construction, so coverage does not collapse the way it
    does on the hand-designed table.

    Returns ``None`` when scikit-learn is missing, so the eval still runs without it.
    """
    try:
        from scaling.latent_bayes import LatentBayesPipeline
    except ImportError:  # pragma: no cover - optional dependency
        return None

    X_train, X_test = _signal_matrix(train), _signal_matrix(test)
    n_components = max(1, min(n_components, X_train.shape[1]))
    # PCA needs at least as many samples as components, and quantile edges need spread.
    if X_train.shape[0] <= n_components or np.allclose(X_train.std(axis=0), 0):
        return None

    pipe = LatentBayesPipeline(n_components=n_components, n_bins=n_bins)
    pipe.fit_projection(X_train)
    pipe.update(X_train, np.array([outcome_label(r.correct) for r in train]))

    probs = pipe.predict_proba(X_test)
    confidence = probs[:, list(_GOOD_STATES)].sum(axis=1)
    return {
        "confidence": [float(c) for c in confidence],
        "correct": [int(r.correct) for r in test],
        "latent_contexts": int(pipe.grid.n_contexts),
        "latent_contexts_visited": pipe.coverage(),
        "explained_variance": float(pipe.explained_variance()),
        "n_raw_signals": int(X_train.shape[1]),
        "naive_raw_contexts": int(n_bins ** X_train.shape[1]),
    }


def learning_curve(
    rows: Sequence[EvalRow],
    *,
    seed: int = 0,
    steps: int = 6,
    test_fraction: float = 0.5,
) -> List[Dict[str, float]]:
    """ECE/accuracy on a fixed test half as the train half is fed in incrementally.

    Answers "does more observed data make the confidence better calibrated, and how
    fast?" rather than only "is the posterior better than the prior at the end?".
    """
    from eval.metrics import auroc, expected_calibration_error

    split = split_rows(rows, test_fraction=test_fraction, seed=seed)
    n_train = len(split.train)
    sizes = sorted({int(round(n_train * f / steps)) for f in range(steps + 1)})
    curve = []
    for size in sizes:
        engine = fit_engine(split.train[:size])
        scored = score_rows(split.test, engine)
        curve.append(
            {
                "n_train": size,
                "ece": expected_calibration_error(scored["confidence"], scored["correct"]),
                "auroc": auroc(scored["confidence"], scored["correct"]),
                "mean_ci_width": float(np.mean(scored["ci_width"])),
                "mean_ess": float(np.mean(scored["ess"])),
            }
        )
    return curve


def run_learning_study(
    rows: Sequence[EvalRow],
    *,
    seed: int = 0,
    test_fraction: float = 0.5,
    n_bins: int = 10,
    n_boot: int = 2000,
) -> Dict[str, object]:
    """Fix 2 end to end: prior-only vs posterior vs latent posterior on held-out rows."""
    from eval.metrics import bootstrap_diff, evaluate_scores, expected_calibration_error

    split = split_rows(rows, test_fraction=test_fraction, seed=seed)
    prior_engine = DirichletBayesianEngine()
    posterior_engine = fit_engine(split.train)

    prior_scored = score_rows(split.test, prior_engine)
    post_scored = score_rows(split.test, posterior_engine)
    correct = prior_scored["correct"]

    result: Dict[str, object] = {
        "seed": seed,
        "n_train": len(split.train),
        "n_test": len(split.test),
        "prior_only": evaluate_scores(
            prior_scored["confidence"], correct, n_bins=n_bins, n_boot=n_boot, seed=seed
        ),
        "posterior": evaluate_scores(
            post_scored["confidence"], correct, n_bins=n_bins, n_boot=n_boot, seed=seed
        ),
        "ece_delta_posterior_minus_prior": bootstrap_diff(
            lambda s, y: expected_calibration_error(s, y, n_bins),
            post_scored["confidence"],
            prior_scored["confidence"],
            correct,
            n_boot=n_boot,
            seed=seed,
        ).as_dict(),
        "ci_width_vs_ess": {
            "prior_only": ci_width_vs_ess(prior_scored),
            "posterior": ci_width_vs_ess(post_scored),
        },
        "coverage_train": context_coverage(split.train),
        "coverage_all": context_coverage(rows),
        "learning_curve": learning_curve(
            rows, seed=seed, test_fraction=test_fraction
        ),
        "engine_total_observations": posterior_engine.total_observations(),
        "engine_contexts_visited": posterior_engine.coverage(),
    }

    latent = latent_confidences(split.train, split.test)
    if latent is not None:
        result["latent"] = {
            **{k: v for k, v in latent.items() if k not in ("confidence", "correct")},
            "metrics": evaluate_scores(
                latent["confidence"], latent["correct"], n_bins=n_bins, n_boot=n_boot, seed=seed
            ),
        }
    return result


__all__ = [
    "Split",
    "split_rows",
    "fit_engine",
    "score_rows",
    "context_coverage",
    "ci_width_vs_ess",
    "latent_confidences",
    "learning_curve",
    "run_learning_study",
]
