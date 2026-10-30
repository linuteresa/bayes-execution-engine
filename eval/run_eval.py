"""CLI: run the calibration eval end to end.

    python -m eval.run_eval --model llama --dataset hf:triviaqa+hf:gsm8k --limit 200
    python -m eval.run_eval --model sim --dataset bundled        # offline smoke test
    python -m eval.run_eval --rows eval/results/rows.jsonl       # re-analyse, no model
    EMBEDDINGS_BASE_URL=http://localhost:8081/v1 \
      python -m eval.run_eval --model replay:eval/results/word-overlap/rows.jsonl  # re-score samples

Stages: collect rows (the only stage that touches the model) -> score the engine and the
baselines -> run the learning study -> render figures and a markdown report. Because the
rows file holds the evidence triples and the continuous signals, every stage after the
first is a pure function of it, and `--rows` replays an old run without re-sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from eval import figures as figures_mod
from eval.answerers import build_answerer, is_simulated
from eval.datasets import describe, load_items
from eval.harness import (
    EvalRow,
    RowWriter,
    metadata_from_rows,
    read_rows,
    run_harness,
    run_metadata,
    validate_resume_rows,
)
from eval.learning import run_learning_study
from eval.metrics import auroc, bootstrap_diff, brier_score, evaluate_scores, risk_coverage
from eval.provenance import (
    SCHEMA_VERSION,
    code_digest,
    dependency_versions,
    experiment_fingerprint,
    file_digest,
    json_compatible,
    utc_now,
    write_json_atomic,
)
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
    answerer = build_answerer(args.model, temperature=args.temperature, seed=args.seed)
    # A replay brings its own items: the exact questions whose samples it stored.
    items = getattr(answerer, "items", None) or load_items(
        args.dataset, limit=args.limit, seed=args.seed
    )

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

    meta = run_metadata(
        answerer, items, n_samples=args.samples, dataset_spec=args.dataset,
        temperature=args.temperature, seed=args.seed, limit=args.limit,
        keep_samples=args.keep_samples,
    )
    meta["dataset"] = describe(items)
    out = Path(args.out)
    rows_path, manifest_path = out / "rows.jsonl", out / "manifest.json"
    initial_rows: List[EvalRow] = []
    if args.resume:
        if not manifest_path.exists():
            raise ValueError("--resume requires a versioned manifest.json in --out")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous = manifest.get("experiment", {})
        if previous.get("experiment_id") != experiment_fingerprint(previous):
            raise ValueError("saved experiment metadata fingerprint is invalid")
        if manifest.get("schema_version") != SCHEMA_VERSION or previous.get("experiment_id") != meta["experiment_id"]:
            raise ValueError("resume configuration differs from the saved experiment")
        meta = previous
        initial_rows = read_rows(rows_path, repair_truncated_tail=True) if rows_path.exists() else []
        validate_resume_rows(initial_rows, items, meta)
        saved = manifest.get("collection", {})
        saved_count = saved.get("completed_items", 0)
        if not isinstance(saved_count, int) or not 0 <= saved_count <= len(initial_rows):
            raise ValueError("saved manifest row count is inconsistent with rows.jsonl")
        if saved.get("rows_sha256"):
            prefix = b"".join(rows_path.read_bytes().splitlines(keepends=True)[:saved_count])
            if hashlib.sha256(prefix).hexdigest() != saved["rows_sha256"]:
                raise ValueError("saved rows checksum differs from the experiment manifest")
    elif not args.overwrite and (rows_path.exists() or manifest_path.exists()):
        raise ValueError("output already contains a run; use --resume, a new --out, or --overwrite")

    completed = len(initial_rows)
    active_writer = None

    def save_manifest(status: str) -> None:
        write_json_atomic({
            "schema_version": SCHEMA_VERSION,
            "experiment": meta,
            "collection": {
                "status": status, "completed_items": completed, "expected_items": len(items),
                "updated_at_utc": utc_now(),
                "rows_sha256": active_writer.sha256 if active_writer else (
                    file_digest(rows_path) if rows_path.exists() else hashlib.sha256(b"").hexdigest()
                ),
            },
        }, manifest_path)

    with RowWriter(rows_path, resume=args.resume, overwrite=args.overwrite) as writer:
        active_writer = writer
        save_manifest("collecting")
        def save_row(row: EvalRow) -> None:
            nonlocal completed
            writer.append(row)
            completed += 1
            save_manifest("collecting")

        rows = run_harness(
            items, answerer, n_samples=args.samples, keep_samples=args.keep_samples,
            progress=progress, on_row=save_row, experiment=meta, initial_rows=initial_rows,
        )
    save_manifest("complete")
    return rows, meta


def analyse(rows: Sequence[EvalRow], meta: dict, args: argparse.Namespace) -> dict:
    correct = [r.correct for r in rows]
    n_classes = len(set(correct))
    if n_classes < 2:
        print(
            "WARNING: every item has the same correctness label. AUROC is "
            "undefined and risk-coverage is constant -- the question set needs a wider "
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
        "analysis": {
            "schema_version": SCHEMA_VERSION, "timestamp_utc": utc_now(),
            "code_sha256": code_digest(analysis=True),
            "dependencies": dependency_versions(),
            "seed": args.seed, "bootstrap": args.bootstrap, "bins": args.bins,
            "abstain_fraction": args.abstain, "test_fraction": args.test_fraction,
            "learning": not args.no_learning,
            "risk_coverage_ties": "expected_uniform_selection_within_equal_scores",
            "confidence_scope": "single_step",
            "stored_confidence_evidence_and_labels_preserved": bool(args.rows),
            "source_rows": str(args.rows) if args.rows else str(Path(args.out) / "rows.jsonl"),
            "source_rows_sha256": file_digest(args.rows or Path(args.out) / "rows.jsonl"),
        },
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


def _count(rows: Sequence[EvalRow], attr: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        key = str(getattr(row, attr))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


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
        "'sim' for the offline stand-in, 'replay:<rows.jsonl>' to re-score a run's "
        "stored samples under the current agreement measure (default: sim)",
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
    p.add_argument("--resume", action="store_true", help="resume --out after validating dataset and experiment configuration")
    p.add_argument("--overwrite", action="store_true", help="replace an existing collected run in --out")
    p.add_argument("--out", default=str(DEFAULT_OUT), help=f"output dir (default: {DEFAULT_OUT})")
    p.add_argument("--quiet", action="store_true", help="suppress per-item progress")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples < 1 or args.bins < 1 or args.bootstrap < 1:
        raise ValueError("--samples, --bins and --bootstrap must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if not 0 <= args.abstain < 1 or not 0 < args.test_fraction < 1:
        raise ValueError("--abstain must be in [0, 1), --test-fraction in (0, 1)")
    if args.resume and (args.rows or args.overwrite):
        raise ValueError("--resume cannot be combined with --rows or --overwrite")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.rows:
        rows = read_rows(args.rows)
        if not rows:
            print(f"no rows in {args.rows}", file=sys.stderr)
            return 1
        # Provenance comes from the rows, not from whatever metrics.json happens to sit
        # next to them: a rows file can be moved or shared alone, and stale adjacent
        # metadata could label a different run entirely.
        meta = metadata_from_rows(rows)
        if meta["is_simulated"] is None:
            print(
                "WARNING: these rows carry no provenance (written before it was "
                "recorded, or mixed from several runs). The report cannot certify "
                "whether they came from a real model, and will say so.",
                file=sys.stderr,
            )
        meta["dataset"] = {
            "n_items": len(rows),
            "by_difficulty": _count(rows, "difficulty"),
            "by_source": _count(rows, "source"),
        }
        meta["analysis_only"] = True
        meta["source_rows"] = str(args.rows)
        meta["source_rows_sha256"] = file_digest(args.rows)
        expected = rows[0].experiment.get("n_items")
        meta["collection_status"] = "partial" if expected and len(rows) < expected else "complete_or_legacy_unknown"
    else:
        rows, meta = collect_rows(args)

    results = analyse(rows, meta, args)
    write_json_atomic(json_compatible(results), out / "metrics.json")
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
