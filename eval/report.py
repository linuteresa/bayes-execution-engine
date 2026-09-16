"""Render eval results as a self-describing markdown report.

Two rules this module enforces so a stale or synthetic artefact can never be mistaken
for a calibration result:

* every report carries a provenance block (model, dataset, item count, timestamp);
* a run whose answerer was simulated is stamped with a warning banner at the top and
  the word SIMULATED in its title, because those numbers describe the harness, not a
  model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

UNKNOWN_PROVENANCE_BANNER = (
    "> **⚠️ PROVENANCE UNKNOWN — THIS MAY NOT BE A REAL CALIBRATION RESULT.**\n"
    "> These rows carry no record of which answerer produced them, so this report\n"
    "> cannot certify that a real model was involved rather than the simulated\n"
    "> stand-in in `eval.answerers`. Treat the numbers below as unverified until the\n"
    "> run is reproduced with `--model llama`.\n"
)

SIMULATED_BANNER = (
    "> **⚠️ SIMULATED RUN — NOT A CALIBRATION RESULT.**\n"
    "> These numbers come from `eval.answerers.SimulatedAnswerer`, a deterministic\n"
    "> stand-in that is *told the gold answer* and fabricates samples from it. The run\n"
    "> exists to prove the harness computes its metrics correctly end to end with no\n"
    "> model available. It says nothing about any real model's calibration. For a\n"
    "> reportable number, start `llama-server` and re-run with `--model llama`.\n"
)


def _simulated_cell(flag: Optional[bool]) -> str:
    if flag is True:
        return "**yes — see banner**"
    if flag is False:
        return "no"
    return "**unknown — see banner**"


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        if value != value:  # NaN
            return "n/a"
    except TypeError:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_interval(interval: Optional[dict], digits: int = 3) -> str:
    """``0.812 [0.744, 0.873]`` -- point estimate with its bootstrap CI."""
    if not interval:
        return "n/a"
    point = _fmt(interval.get("point"), digits)
    low = _fmt(interval.get("ci_low"), digits)
    high = _fmt(interval.get("ci_high"), digits)
    if low == "n/a" or high == "n/a":
        return point
    return f"{point} [{low}, {high}]"


def _excludes_zero(interval: Optional[dict]) -> Optional[bool]:
    if not interval:
        return None
    low, high = interval.get("ci_low"), interval.get("ci_high")
    if low is None or high is None or low != low or high != high:
        return None
    return bool(low > 0 or high < 0)


SIGNAL_LABELS = {
    "engine": "Bayesian engine (125-cell Dirichlet)",
    "self_consistency": "Baseline: self-consistency agreement",
    "mean_token_logprob": "Baseline: mean token logprob",
}

#: Same signals, short enough for a figure legend.
SHORT_LABELS = {
    "engine": "engine (Dirichlet)",
    "self_consistency": "self-consistency",
    "mean_token_logprob": "mean token logprob",
}


def _headline_table(signals: Dict[str, dict], abstain: float) -> List[str]:
    kept = int(round((1.0 - abstain) * 100))
    lines = [
        f"| Confidence signal | ECE ↓ | Brier ↓ | AUROC ↑ | AURC ↓ | Acc @100% | Acc @{kept}% |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, rollup in signals.items():
        if not rollup:
            continue
        lines.append(
            "| {label} | {ece} | {brier} | {auroc} | {aurc} | {acc_full} | {acc_cov} |".format(
                label=SIGNAL_LABELS.get(key, key),
                ece=_fmt(rollup.get("ece")),
                brier=_fmt_interval(rollup.get("brier")),
                auroc=_fmt_interval(rollup.get("auroc")),
                aurc=_fmt(rollup.get("aurc")),
                acc_full=_fmt(rollup.get("accuracy_full_coverage")),
                acc_cov=_fmt(rollup.get("accuracy_at_coverage")),
            )
        )
    return lines


def _reliability_table(rollup: dict) -> List[str]:
    lines = [
        "| Confidence bin | Items | Mean confidence | Empirical accuracy | Gap |",
        "|---|---|---|---|---|",
    ]
    for b in rollup.get("reliability", []):
        lines.append(
            f"| {b['lower']:.1f}–{b['upper']:.1f} | {b['count']} | "
            f"{b['mean_confidence']:.3f} | {b['empirical_accuracy']:.3f} | {b['gap']:+.3f} |"
        )
    return lines


def _verdict(results: dict) -> List[str]:
    """State plainly what the run does and does not establish."""
    engine = (results.get("signals") or {}).get("engine") or {}
    comparisons = results.get("comparisons") or {}
    lines: List[str] = []

    auroc = (engine.get("auroc") or {}).get("point")
    if auroc is not None and auroc == auroc:
        if auroc < 0.55:
            lines.append(
                f"- Confidence barely separates correct from incorrect answers "
                f"(AUROC {auroc:.3f}, where 0.5 is chance). On this run the signal is "
                f"close to uninformative."
            )
        else:
            lines.append(
                f"- Confidence separates correct from incorrect answers at AUROC "
                f"{auroc:.3f}."
            )

    ece = engine.get("ece")
    if ece is not None:
        direction = "over" if _mean_gap(engine) > 0 else "under"
        lines.append(
            f"- ECE is {ece:.3f} over {engine.get('ece_bins', 10)} bins; the engine is "
            f"{direction}confident on average "
            f"(mean confidence {_fmt(engine.get('mean_confidence'))} vs accuracy "
            f"{_fmt(engine.get('accuracy'))})."
        )

    acc_full = engine.get("accuracy_full_coverage")
    acc_cov = engine.get("accuracy_at_coverage")
    abstain = engine.get("abstain_fraction")
    if None not in (acc_full, acc_cov, abstain):
        lines.append(
            f"- Abstaining on the least-confident {abstain:.0%} lifts accuracy on the "
            f"remainder from {acc_full:.3f} to {acc_cov:.3f}."
        )

    for name, comp in comparisons.items():
        delta = comp.get("auroc_delta")
        excludes = _excludes_zero(delta)
        if excludes is None:
            continue
        baseline = SIGNAL_LABELS.get(name.replace("engine_vs_", ""), name)
        if excludes and (delta.get("point") or 0) > 0:
            lines.append(
                f"- The engine beats *{baseline}* on AUROC by "
                f"{_fmt_interval(delta)} (paired bootstrap CI excludes 0)."
            )
        elif excludes:
            lines.append(
                f"- The engine is **worse** than *{baseline}* on AUROC by "
                f"{_fmt_interval(delta)} (CI excludes 0). The extra machinery is "
                f"costing discrimination here."
            )
        else:
            lines.append(
                f"- The engine is **not** measurably better than *{baseline}* on AUROC "
                f"({_fmt_interval(delta)}; CI straddles 0). On this dataset the "
                f"125-cell layer is not adding discrimination over the raw signal."
            )
    return lines


def _accuracy_at(rollup: dict, coverage: float) -> Optional[float]:
    """Accuracy at a coverage level, read from the row-level curve computed upstream.

    ``evaluate_scores`` stores decile points of the real risk-coverage curve, so this is
    a lookup rather than a reconstruction. It used to re-derive the value from the
    reliability bins by taking a fractional slice of the bin straddling the cutoff,
    which assumes uniform correctness within that bin and was measurably wrong -- it
    understated TriviaQA's accuracy at 50% coverage as 0.566 against a true 0.600.
    """
    points = rollup.get("coverage_points") or []
    if not points:
        return None
    nearest = min(points, key=lambda p: abs(p["coverage"] - coverage))
    # Exact-decile lookup only. Snapping a nearby request onto a decile would return a
    # real number under a label that misstates which coverage it belongs to, which is a
    # subtler version of the bug this function was rewritten to fix.
    if abs(nearest["coverage"] - coverage) > 1e-6:
        return None
    return nearest["accuracy"]


def _mean_gap(rollup: dict) -> float:
    conf = rollup.get("mean_confidence")
    acc = rollup.get("accuracy")
    if conf is None or acc is None:
        return 0.0
    return float(conf) - float(acc)


def _learning_section(study: dict) -> List[str]:
    lines = [
        "## Fix 2 — closing the learning loop",
        "",
        f"`engine.observe()` was called on {study.get('n_train', 0)} training rows "
        f"({_fmt(study.get('engine_total_observations'), 0)} observations); both engines "
        f"are then scored on the same {study.get('n_test', 0)} **held-out** rows.",
        "",
        "| Engine | ECE ↓ | Brier ↓ | AUROC ↑ | Acc @100% |",
        "|---|---|---|---|---|",
    ]
    for key, label in (("prior_only", "Prior only (as shipped)"), ("posterior", "Posterior (observed)")):
        rollup = study.get(key) or {}
        lines.append(
            f"| {label} | {_fmt(rollup.get('ece'))} | {_fmt_interval(rollup.get('brier'))} | "
            f"{_fmt_interval(rollup.get('auroc'))} | {_fmt(rollup.get('accuracy_full_coverage'))} |"
        )

    delta = study.get("ece_delta_posterior_minus_prior")
    lines += [
        "",
        f"ECE change from observing (posterior − prior, negative is better): "
        f"**{_fmt_interval(delta)}**.",
        "",
    ]

    buckets = (study.get("ci_width_vs_ess") or {}).get("posterior") or []
    if buckets:
        lines += [
            "### Credible-interval width vs effective sample size",
            "",
            "| ESS bucket | Mean ESS | Mean 95% CI width | Items |",
            "|---|---|---|---|",
        ]
        for b in buckets:
            lines.append(
                f"| {b['ess_low']:.1f}–{b['ess_high']:.1f} | {b['mean_ess']:.1f} | "
                f"{b['mean_ci_width']:.3f} | {b['count']} |"
            )
        lines.append("")

    curve = study.get("learning_curve") or []
    if curve:
        lines += [
            "### Learning curve (held-out)",
            "",
            "| Train observations | ECE ↓ | AUROC ↑ | Mean CI width | Mean ESS |",
            "|---|---|---|---|---|",
        ]
        for c in curve:
            lines.append(
                f"| {c['n_train']} | {_fmt(c['ece'])} | {_fmt(c['auroc'])} | "
                f"{_fmt(c['mean_ci_width'])} | {_fmt(c['mean_ess'], 1)} |"
            )
        lines.append("")
    return lines


def _coverage_section(study: dict) -> List[str]:
    cov = study.get("coverage_all") or {}
    visited = cov.get("contexts_visited", 0)
    total = cov.get("n_contexts", 125)
    lines = [
        "## Context coverage — is a 125-cell table the right size?",
        "",
        f"Across all {cov.get('n_observations', 0)} evaluated items the engine visited "
        f"**{visited} of {total}** contexts "
        f"({cov.get('contexts_visited_fraction', 0.0):.1%}); "
        f"{cov.get('contexts_with_at_least_10', 0)} contexts saw 10+ observations, and the "
        f"busiest held {cov.get('max_count_in_one_context', 0)}.",
        "",
        "Cells never visited stay at their prior no matter how long the system runs, so "
        "low coverage is the empirical argument for reducing dimensionality rather than "
        "growing the table — which is what `scaling/latent_bayes.py` does.",
        "",
    ]

    latent = study.get("latent")
    if latent:
        metrics = latent.get("metrics") or {}
        lines += [
            "### Same rows through the latent (PCA) grid",
            "",
            f"The executor's {latent.get('n_raw_signals')} continuous signals would give "
            f"{latent.get('naive_raw_contexts'):,} naive contexts at 5 bins each. Projected "
            f"to {latent.get('latent_contexts')} quantile-binned latent cells, "
            f"**{latent.get('latent_contexts_visited')}** were visited "
            f"(explained variance {_fmt(latent.get('explained_variance'))}).",
            "",
            "| Model | ECE ↓ | Brier ↓ | AUROC ↑ | Contexts visited |",
            "|---|---|---|---|---|",
            f"| Hand-designed 125-cell grid | {_fmt((study.get('posterior') or {}).get('ece'))} | "
            f"{_fmt_interval((study.get('posterior') or {}).get('brier'))} | "
            f"{_fmt_interval((study.get('posterior') or {}).get('auroc'))} | "
            f"{visited}/{total} |",
            f"| Latent PCA grid | {_fmt(metrics.get('ece'))} | "
            f"{_fmt_interval(metrics.get('brier'))} | {_fmt_interval(metrics.get('auroc'))} | "
            f"{latent.get('latent_contexts_visited')}/{latent.get('latent_contexts')} |",
            "",
        ]
    return lines


def render_report(results: dict) -> str:
    """Build the full markdown report from a results dict."""
    meta = results.get("metadata") or {}
    # Tri-state on purpose: True / False / None(unknown). `bool(None)` would quietly
    # promote "we don't know" into "it's real", which is the one mistake this banner
    # exists to prevent.
    simulated_flag = meta.get("is_simulated")
    simulated = simulated_flag is True
    unknown_provenance = simulated_flag is None
    dataset = results.get("dataset") or {}
    signals = results.get("signals") or {}
    engine = signals.get("engine") or {}

    if simulated:
        title = "Calibration report — SIMULATED (not a result)"
    elif unknown_provenance:
        title = "Calibration report — UNVERIFIED PROVENANCE"
    else:
        title = "Calibration report"
    lines: List[str] = [f"# {title}", ""]
    if simulated:
        lines += [SIMULATED_BANNER, ""]
    elif unknown_provenance:
        lines += [UNKNOWN_PROVENANCE_BANNER, ""]

    lines += [
        "## Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Generated (UTC) | {meta.get('timestamp_utc', 'unknown')} |",
        f"| Answerer | `{meta.get('answerer', 'unknown')}` |",
        f"| Model | `{meta.get('model', 'unknown')}` |",
        f"| Simulated | {_simulated_cell(simulated_flag)} |",
        f"| Dataset spec | `{meta.get('dataset_spec', 'unknown')}` |",
        f"| Items | {meta.get('n_items', dataset.get('n_items', 0))} |",
        f"| Samples per item | {meta.get('n_samples_per_item', 0)} |",
        "",
        "Composition: "
        + ", ".join(f"{k} × {v}" for k, v in (dataset.get("by_difficulty") or {}).items())
        + ".",
        "",
        "## Fix 1 — is the confidence calibrated?",
        "",
        "ECE and Brier are scale-sensitive, so a raw score never meant to be a probability "
        "can discriminate well and still calibrate badly. AUROC and AURC are rank-based and "
        "compare the engine and the baselines on equal terms — read those for "
        "*does the number know anything*, and ECE for *does the number mean what it says*. "
        "Brackets are 95% bootstrap CIs over items.",
        "",
    ]
    lines += _headline_table(signals, float(engine.get("abstain_fraction", 0.2)))
    lines += ["", "### Verdict", ""]
    lines += _verdict(results) or ["- Not enough data to draw a conclusion."]
    lines += ["", "### Reliability bins (engine)", ""]
    lines += _reliability_table(engine)
    lines += [""]

    by_source = results.get("by_source") or {}
    if by_source:
        lines += [
            "### Per-dataset breakdown",
            "",
            "One averaged number hides the most useful thing in this run: self-consistency "
            "has a known failure mode — a model that is *consistently* wrong looks confident "
            "— and arithmetic is where that bites, because a wrong method reproduces the same "
            "wrong answer every sample.",
            "",
            "| Dataset | Items | Accuracy | ECE ↓ | AUROC ↑ | AUROC (agreement) | Acc @100% | Acc @50% |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for source, rollup in by_source.items():
            half = _fmt(_accuracy_at(rollup, 0.5))
            lines.append(
                f"| {source} | {rollup.get('n')} | {_fmt(rollup.get('accuracy'))} | "
                f"{_fmt(rollup.get('ece'))} | {_fmt_interval(rollup.get('auroc'))} | "
                f"{_fmt(rollup.get('auroc_self_consistency'))} | "
                f"{_fmt(rollup.get('accuracy_full_coverage'))} | {half} |"
            )
        lines += [""]

    study = results.get("learning")
    if study:
        lines += _learning_section(study)
        lines += _coverage_section(study)

    figures = results.get("figures") or {}
    if figures:
        lines += ["## Figures", ""]
        for label, rel in figures.items():
            lines += [f"**{label}**", "", f"![{label}]({rel})", ""]

    lines += [
        "## How to reproduce",
        "",
        "```bash",
        f"python -m eval.run_eval --model {'sim' if simulated else 'llama'} "
        f"--dataset {meta.get('dataset_spec', 'bundled')} "
        f"--samples {meta.get('n_samples_per_item', 4)}",
        "```",
        "",
        "## Limitations",
        "",
        "- Agreement is bag-of-words cosine, so paraphrases read as disagreement and the "
        "confidence signal is pessimistic on verbose answers.",
        "- Grading is normalized alias containment / numeric match: a correct answer phrased "
        "unusually can be scored wrong, which shows up as apparent overconfidence.",
        "- Self-consistency measures agreement, not truth. A model that is confidently and "
        "consistently wrong scores high here by construction.",
        "- Risk-coverage ties are broken by row order, so the curve is slightly pessimistic "
        "when many items share a confidence — which happens whenever the evidence is a "
        "coarse 5-level ordinal.",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_report(results: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(results), encoding="utf-8")
    return path


__all__ = [
    "render_report",
    "write_report",
    "SIMULATED_BANNER",
    "UNKNOWN_PROVENANCE_BANNER",
    "SIGNAL_LABELS",
    "SHORT_LABELS",
]
