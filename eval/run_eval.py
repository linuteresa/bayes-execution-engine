"""CLI: run the calibration eval end to end.

    python -m eval.run_eval --model llama --dataset hf:triviaqa+hf:gsm8k --limit 200
    python -m eval.run_eval --model sim --dataset bundled        # offline smoke test
    python -m eval.run_eval --rows eval/results/rows.jsonl       # re-analyse, no model

Stages: collect rows (the only stage that touches the model) -> score the engine and the
baselines -> run the learning study -> render figures and a markdown report. Because the
rows file holds the evidence triples and the continuous signals, every stage after the
first is a pure function of it, and `--rows` replays an old run without re-sampling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from eval import figures as figures_mod
from eval.answerers import build_answerer, is_simulated
from eval.datasets import describe, load_items
from eval.harness import EvalRow, read_rows, run_harness, run_metadata, write_rows
from eval.learning import run_learning_study
from eval.metrics import auroc, bootstrap_diff, brier_score, evaluate_scores, risk_coverage
from eval.report import SHORT_LABELS, write_report

DEFAULT_OUT = Path("eval/results")


def _logprob_confidence(rows: Sequence[EvalRow]) -> Optional[List[float]]:
    """Turn mean token logprob into a [0, 1] score: ``exp(mean logprob)``.

    That is the model's average per-token probability -- the standard cheap confidence
    baseline. It is available only if the server returned logprobs for every item;
    a partial signal would make the comparison unpaired, so it is all or nothing.
    """
    values = [r.mean_logprob for r in rows]
    if any(v is None for v in values):
        return None
    import math

    return [math.exp(min(0.0, float(v))) for v in values]


def collect_rows(args: argparse.Namespace) -> tuple[List[EvalRow], dict]:
    items = load_items(args.dataset, limit=args.limit, seed=args.seed)
    answerer = build_answerer(args.model, temperature=args.temperature, seed=args.seed)

    if is_simulated(answerer):
        print(
            "NOTE: --model sim fabricates answers from the gold labels. The report will "
            "be stamped SIMULATED and is not a calibration result.",
            file=sys.stderr,
        )

    def progress(done: int, total: int, row: EvalRow) -> None:
        if args.quiet:
            return
        mark = "ok " if row.correct else "MISS"
        print(
            f"[{done:>4}/{total}] {mark} conf={row.confidence:.3f} "
            f"agree={row.agreement:.2f} {row.item_id}",
            file=sys.stderr,
        )

    rows = run_harness(
        items,
        answerer,
        n_samples=args.samples,
        keep_samples=args.keep_samples,
        progress=progress,
    )
    meta = run_metadata(
        answerer, items, n_samples=args.samples, dataset_spec=args.dataset
    )
    meta["dataset"] = describe(items)
    return rows, meta


def analyse(rows: Sequence[EvalRow], meta: dict, args: argparse.Namespace) -> dict:
    correct = [r.correct for r in rows]
    n_classes = len(set(correct))
    if n_classes < 2:
        print(
            "WARNING: every item has the same correctness label. AUROC and the "
            "risk-coverage curve are undefined -- the question set needs a wider "
            "difficulty spread to test a confidence signal at all.",
            file=sys.stderr,
        )

    engine_conf = [r.confidence for r in rows]
    agreement = [r.agreement for r in rows]
    logprob_conf = _logprob_confidence(rows)

    def rollup(scores):
        return evaluate_scores(
            scores,
            correct,
            n_bins=args.bins,
            n_boot=args.bootstrap,
            seed=args.seed,
            abstain_fraction=args.abstain,
        )

    signals: Dict[str, Optional[dict]] = {
        "engine": rollup(engine_conf),
        "self_consistency": rollup(agreement),
        "mean_token_logprob": rollup(logprob_conf) if logprob_conf else None,
    }
    if logprob_conf is None:
        print(
            "NOTE: the server returned no token logprobs, so that baseline is omitted. "
            "Run llama-server with logprobs enabled to include it.",
            file=sys.stderr,
        )

    comparisons: Dict[str, dict] = {}
    for name, baseline in (
        ("self_consistency", agreement),
        ("mean_token_logprob", logprob_conf),
    ):
        if baseline is None or n_classes < 2:
            continue
        comparisons[f"engine_vs_{name}"] = {
            "auroc_delta": bootstrap_diff(
                lambda s, y: auroc(s, y),
                engine_conf,
                baseline,
                correct,
                n_boot=args.bootstrap,
                seed=args.seed,
            ).as_dict(),
            "brier_delta": bootstrap_diff(
                lambda s, y: brier_score(s, y),
                engine_conf,
                baseline,
                correct,
                n_boot=args.bootstrap,
                seed=args.seed,
            ).as_dict(),
        }

    results: dict = {
        "metadata": meta,
        "dataset": meta.get("dataset", {}),
        "signals": {k: v for k, v in signals.items() if v},
        "comparisons": comparisons,
        "by_source": _per_source(rows, args),
    }

    if not args.no_learning:
        try:
            results["learning"] = run_learning_study(
                rows,
                seed=args.seed,
                test_fraction=args.test_fraction,
                n_bins=args.bins,
                n_boot=args.bootstrap,
            )
        except ValueError as exc:
            print(f"NOTE: learning study skipped ({exc}).", file=sys.stderr)

    # Figures are optional; the tables above never depend on them rendering.
    out = Path(args.out)
    written: Dict[str, str] = {}
    series = {SHORT_LABELS.get(k, k): v for k, v in results["signals"].items()}
    rel = figures_mod.reliability_diagram(
        series, out / "reliability.png", title="Reliability — engine vs baselines"
    )
    if rel:
        written["Reliability diagram"] = rel.name
    curves = {SHORT_LABELS["engine"]: risk_coverage(engine_conf, correct)}
    if n_classes >= 2:
        curves[SHORT_LABELS["self_consistency"]] = risk_coverage(agreement, correct)
    rc = figures_mod.risk_coverage_plot(curves, out / "risk_coverage.png")
    if rc:
        written["Risk-coverage curve"] = rc.name
    if results.get("learning"):
        lp = figures_mod.learning_panels(results["learning"], out / "learning.png")
        if lp:
            written["Learning loop"] = lp.name
    if written:
        results["figures"] = written
    return results


def _per_source(rows: Sequence[EvalRow], args: argparse.Namespace) -> Dict[str, dict]:
    """Score each dataset separately.

    One averaged number can hide the most useful thing in the run. A confidence signal
    built on self-consistency has a known failure mode -- a model that is *consistently*
    wrong looks confident -- and arithmetic is exactly where that bites, because a wrong
    method produces the same wrong answer every sample. Averaging trivia and maths into
    a single AUROC would report that as mediocre discrimination everywhere, rather than
    good discrimination in one regime and near-chance in the other.
    """
    groups: Dict[str, List[EvalRow]] = {}
    for row in rows:
        groups.setdefault(row.source, []).append(row)
    if len(groups) < 2:
        return {}

    out: Dict[str, dict] = {}
    for source, group in sorted(groups.items()):
        labels = [r.correct for r in group]
        # A subset with one label makes AUROC undefined; report accuracy and move on.
        if len(group) < 20 or len(set(labels)) < 2:
            continue
        out[source] = evaluate_scores(
            [r.confidence for r in group],
            labels,
            n_bins=args.bins,
            n_boot=args.bootstrap,
            seed=args.seed,
            abstain_fraction=args.abstain,
        )
        out[source]["auroc_self_consistency"] = auroc([r.agreement for r in group], labels)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval.run_eval",
        description="Measure whether the engine's confidence is calibrated.",
    )
    p.add_argument(
        "--model",
        default="sim",
        help="'llama' for the local llama.cpp server (the reportable path), "
        "'sim' for the offline stand-in (default: sim)",
    )
    p.add_argument(
        "--dataset",
        default="bundled",
        help="bundled | jsonl:<path> | hf:triviaqa | hf:gsm8k | hf:simpleqa; "
        "combine with '+' (default: bundled)",
    )
    p.add_argument("--limit", type=int, default=None, help="max items per dataset component")
    p.add_argument("--samples", type=int, default=4, help="self-consistency samples per item")
    p.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=10, help="reliability/ECE bin count")
    p.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples")
    p.add_argument(
        "--abstain",
        type=float,
        default=0.2,
        help="fraction of least-confident items to abstain on in the headline number",
    )
    p.add_argument("--test-fraction", type=float, default=0.5, help="held-out share for Fix 2")
    p.add_argument("--no-learning", action="store_true", help="skip the learning study")
    p.add_argument("--keep-samples", action="store_true", help="store raw samples in rows.jsonl")
    p.add_argument("--rows", default=None, help="re-analyse an existing rows.jsonl (no model)")
    p.add_argument("--out", default=str(DEFAULT_OUT), help=f"output dir (default: {DEFAULT_OUT})")
    p.add_argument("--quiet", action="store_true", help="suppress per-item progress")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.rows:
        rows = read_rows(args.rows)
        meta_path = Path(args.rows).with_name("metrics.json")
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8")).get("metadata", {})
        meta.setdefault("dataset_spec", args.dataset)
        meta.setdefault("n_items", len(rows))
        meta.setdefault("n_samples_per_item", rows[0].n_samples if rows else args.samples)
        meta["dataset"] = meta.get("dataset") or {"n_items": len(rows)}
    else:
        rows, meta = collect_rows(args)
        write_rows(rows, out / "rows.jsonl")

    results = analyse(rows, meta, args)
    (out / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_path = write_report(results, out / "REPORT.md")

    engine = results["signals"]["engine"]
    print(
        f"\nn={engine['n']}  accuracy={engine['accuracy']:.3f}  "
        f"ECE={engine['ece']:.3f}  AUROC={engine['auroc']['point']:.3f}  "
        f"acc@{1 - args.abstain:.0%}coverage={engine['accuracy_at_coverage']:.3f}"
    )
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
