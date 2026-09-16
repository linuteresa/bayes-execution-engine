"""Figures for the calibration report.

matplotlib is an eval-only dependency (``requirements-eval.txt``), so every function
here degrades to a no-op returning ``None`` when it is missing. The numbers in the
report never depend on the figures rendering -- a run without matplotlib produces the
same tables, just without the PNGs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

_ENGINE_COLOR = "#1f4e79"
_BASELINE_COLORS = ("#c2571a", "#4a7c59", "#7a4a7c")


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: CI and servers have no display
        import matplotlib.pyplot as plt

        return plt
    except Exception:  # pragma: no cover - optional dependency
        return None


def reliability_diagram(
    series: Dict[str, dict], path: Path | str, *, title: str = "Reliability"
) -> Optional[Path]:
    """Plot empirical accuracy against predicted confidence, one line per signal.

    ``series`` maps a label to a metric rollup from ``eval.metrics.evaluate_scores``.
    The diagonal is perfect calibration; a line below it is overconfident.
    """
    plt = _pyplot()
    if plt is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(6.2, 6.6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1, label="perfect calibration")

    colors = [_ENGINE_COLOR, *_BASELINE_COLORS]
    for (label, rollup), color in zip(series.items(), colors, strict=False):
        bins = rollup.get("reliability") or []
        if not bins:
            continue
        xs = [b["mean_confidence"] for b in bins]
        ys = [b["empirical_accuracy"] for b in bins]
        sizes = [max(18.0, 4.0 * b["count"]) for b in bins]
        ax.plot(xs, ys, "-o", color=color, markersize=4, linewidth=1.5, label=label)
        ax.scatter(xs, ys, s=sizes, color=color, alpha=0.18)
        ax_hist.plot(xs, [b["count"] for b in bins], "-o", color=color, markersize=3, linewidth=1)

    ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    ax_hist.set_xlabel("predicted confidence")
    ax_hist.set_ylabel("items per bin")
    ax_hist.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def risk_coverage_plot(
    curves: Dict[str, Sequence], path: Path | str, *, title: str = "Risk-coverage"
) -> Optional[Path]:
    """Accuracy retained as a function of how many items you choose to answer.

    ``curves`` maps a label to a ``RiskCoverageCurve``. Reading right to left is the
    business sentence: give up the least-confident tail, keep this much accuracy.
    """
    plt = _pyplot()
    if plt is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    colors = [_ENGINE_COLOR, *_BASELINE_COLORS]
    for (label, curve), color in zip(curves.items(), colors, strict=False):
        xs = [p.coverage for p in curve.points]
        ys = [p.accuracy for p in curve.points]
        ax.plot(xs, ys, color=color, linewidth=1.6, label=f"{label} (AURC {curve.aurc:.3f})")
        ax.axhline(curve.full_accuracy, color="#999999", linestyle="--", linewidth=1)

    ax.set_xlabel("coverage (fraction of items answered, most confident first)")
    ax.set_ylabel("accuracy on answered items")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def learning_panels(study: dict, path: Path | str) -> Optional[Path]:
    """Two panels for the loop-closing result.

    Left: held-out ECE as train observations accumulate, prior-only as a flat reference
    (it cannot move -- that is the point). Right: mean credible-interval width against
    ESS, the visible form of intervals shrinking with data.
    """
    plt = _pyplot()
    if plt is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    curve: List[dict] = list(study.get("learning_curve") or [])
    buckets: List[dict] = list(
        (study.get("ci_width_vs_ess") or {}).get("posterior") or []
    )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    if curve:
        ax_l.plot(
            [c["n_train"] for c in curve],
            [c["ece"] for c in curve],
            "-o",
            color=_ENGINE_COLOR,
            markersize=4,
            label="posterior (observed)",
        )
        prior_ece = (study.get("prior_only") or {}).get("ece")
        if prior_ece is not None:
            ax_l.axhline(
                prior_ece, color=_BASELINE_COLORS[0], linestyle="--", label="prior only"
            )
    ax_l.set_xlabel("training observations fed to engine.observe()")
    ax_l.set_ylabel("ECE on held-out items")
    ax_l.set_title("Calibration vs observed data")
    ax_l.legend(fontsize=8)
    ax_l.grid(alpha=0.25)

    if buckets:
        ax_r.plot(
            [b["mean_ess"] for b in buckets],
            [b["mean_ci_width"] for b in buckets],
            "-o",
            color=_ENGINE_COLOR,
            markersize=4,
        )
    ax_r.set_xlabel("effective sample size (alpha_0)")
    ax_r.set_ylabel("mean 95% credible-interval width")
    ax_r.set_title("Uncertainty shrinks with data")
    ax_r.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


__all__ = ["reliability_diagram", "risk_coverage_plot", "learning_panels"]
