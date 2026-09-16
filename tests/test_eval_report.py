"""Tests for report rendering and the CLI, including the anti-fabrication guards."""

import json

from eval.report import render_report
from eval.run_eval import main

_BASE = {
    "metadata": {
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "answerer": "llama.cpp",
        "model": "Qwen2.5-7B",
        "is_simulated": False,
        "dataset_spec": "bundled",
        "n_items": 80,
        "n_samples_per_item": 4,
    },
    "dataset": {"n_items": 80, "by_difficulty": {"easy": 40, "hard": 40}},
    "signals": {
        "engine": {
            "n": 80,
            "accuracy": 0.6,
            "mean_confidence": 0.7,
            "ece": 0.12,
            "mce": 0.3,
            "ece_bins": 10,
            "brier": {"point": 0.2, "ci_low": 0.17, "ci_high": 0.24},
            "auroc": {"point": 0.78, "ci_low": 0.68, "ci_high": 0.87},
            "aurc": 0.2,
            "accuracy_full_coverage": 0.6,
            "abstain_fraction": 0.2,
            "accuracy_at_coverage": 0.8,
            "reliability": [
                {
                    "lower": 0.6,
                    "upper": 0.7,
                    "count": 80,
                    "mean_confidence": 0.7,
                    "empirical_accuracy": 0.6,
                    "gap": 0.1,
                }
            ],
        }
    },
    "comparisons": {},
}


def test_real_run_has_no_simulated_banner():
    report = render_report(_BASE)
    assert "SIMULATED" not in report
    assert "Qwen2.5-7B" in report
    assert "AUROC" in report


def test_simulated_run_is_banner_stamped():
    """A synthetic run must be impossible to mistake for a calibration result."""
    results = json.loads(json.dumps(_BASE))
    results["metadata"]["is_simulated"] = True
    report = render_report(results)
    assert report.startswith("# Calibration report — SIMULATED (not a result)")
    assert "NOT A CALIBRATION RESULT" in report
    assert "--model llama" in report


def test_verdict_reports_the_abstention_headline():
    report = render_report(_BASE)
    assert "Abstaining on the least-confident 20%" in report
    assert "0.600 to 0.800" in report


def test_verdict_calls_out_a_losing_engine():
    """An unflattering result must be stated, not omitted."""
    results = json.loads(json.dumps(_BASE))
    results["comparisons"] = {
        "engine_vs_self_consistency": {
            "auroc_delta": {"point": -0.2, "ci_low": -0.31, "ci_high": -0.09}
        }
    }
    report = render_report(results)
    assert "worse" in report


def test_verdict_calls_out_an_inconclusive_comparison():
    results = json.loads(json.dumps(_BASE))
    results["comparisons"] = {
        "engine_vs_self_consistency": {
            "auroc_delta": {"point": 0.02, "ci_low": -0.08, "ci_high": 0.13}
        }
    }
    report = render_report(results)
    assert "not** measurably better" in report


def test_cli_runs_end_to_end_offline(tmp_path):
    """The whole pipeline, no model, no network."""
    code = main(
        [
            "--model", "sim",
            "--dataset", "bundled",
            "--limit", "40",
            "--samples", "3",
            "--bootstrap", "100",
            "--out", str(tmp_path),
            "--quiet",
        ]
    )
    assert code == 0

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["metadata"]["is_simulated"] is True
    assert metrics["signals"]["engine"]["n"] == 40
    assert "learning" in metrics

    report = (tmp_path / "REPORT.md").read_text()
    assert "SIMULATED" in report
    assert (tmp_path / "rows.jsonl").exists()


def test_cli_can_reanalyse_rows_without_a_model(tmp_path):
    """Rows are the unit of work: analysis must never need to re-sample."""
    main(["--model", "sim", "--dataset", "bundled", "--limit", "40", "--samples", "3",
          "--bootstrap", "100", "--out", str(tmp_path), "--quiet"])
    first = json.loads((tmp_path / "metrics.json").read_text())["signals"]["engine"]["ece"]

    out2 = tmp_path / "replay"
    code = main(["--rows", str(tmp_path / "rows.jsonl"), "--bootstrap", "100",
                 "--out", str(out2), "--quiet"])
    assert code == 0
    second = json.loads((out2 / "metrics.json").read_text())["signals"]["engine"]["ece"]
    assert first == second
