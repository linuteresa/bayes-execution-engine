"""Regression tests for the PR #9 review findings.

Each test names the defect it pins so a future refactor that reintroduces it fails
here rather than silently producing a plausible-looking calibration number.
"""

import json

import numpy as np
import pytest

from bayesian_engine.bayes_engine import (
    GOOD_STATES,
    DirichletBayesianEngine,
    good_probability,
)
from eval.answerers import SimulatedAnswerer
from eval.datasets import load_bundled
from eval.harness import metadata_from_rows, read_rows, run_harness, write_rows
from eval.report import render_report


# ------------------------------------------------- credible interval mismatch
def test_good_interval_bounds_the_reported_confidence():
    """The stored interval must describe P(CERTAIN)+P(HIGH), not the MAP state.

    Previously the eval paired `good_probability(summary)` with the MAP state's Beta
    interval -- two different quantities. For degraded evidence the interval did not
    even contain the confidence it travelled with.
    """
    engine = DirichletBayesianEngine()
    for task in range(5):
        summary = engine.resolve(
            {"TaskStatus": task, "DataQuality": task, "ToolReliability": task}
        ).as_dict()
        conf = good_probability(summary)
        low, high = summary["good_credible_interval"]
        assert low <= conf <= high, f"interval {low, high} excludes confidence {conf}"
        assert 0.0 <= low <= high <= 1.0


def test_good_interval_differs_from_the_map_interval():
    """They are genuinely different quantities -- this is not a cosmetic rename."""
    summary = DirichletBayesianEngine().resolve(
        {"TaskStatus": 4, "DataQuality": 4, "ToolReliability": 4}
    ).as_dict()
    assert summary["good_credible_interval"] != summary["credible_interval"]
    assert not (
        summary["credible_interval"][0]
        <= good_probability(summary)
        <= summary["credible_interval"][1]
    )


def test_set_interval_reduces_to_the_single_state_case():
    """Dirichlet aggregation: a one-element set must equal the old per-state interval."""
    engine = DirichletBayesianEngine()
    evidence = {"TaskStatus": 2, "DataQuality": 1, "ToolReliability": 3}
    assert engine._beta_credible_interval(evidence, 0) == pytest.approx(
        engine._beta_credible_interval(evidence, [0])
    )


def test_good_interval_tightens_as_evidence_accumulates():
    engine = DirichletBayesianEngine()
    ctx = {"TaskStatus": 1, "DataQuality": 1, "ToolReliability": 1}
    before = engine.resolve(ctx).as_dict()["good_credible_interval"]
    for _ in range(300):
        engine.observe(1, 1, 1, GOOD_STATES[0])
    after = engine.resolve(ctx).as_dict()["good_credible_interval"]
    assert (after[1] - after[0]) < (before[1] - before[0])


# --------------------------------------------------------- provenance in rows
def test_rows_carry_provenance(tmp_path):
    """rows.jsonl must be self-describing: it is the documented unit of exchange."""
    rows = run_harness(load_bundled()[:5], SimulatedAnswerer(seed=0), n_samples=3)
    assert all(r.is_simulated is True for r in rows)
    assert all(r.answerer == "simulated" for r in rows)

    path = write_rows(rows, tmp_path / "rows.jsonl")
    payload = json.loads(path.read_text().splitlines()[0])
    assert payload["is_simulated"] is True   # survives the file, not just the object


def test_replay_of_orphaned_rows_cannot_claim_to_be_real(tmp_path):
    """A rows file moved away from its metrics.json must not render as a real result."""
    rows = run_harness(load_bundled()[:5], SimulatedAnswerer(seed=0), n_samples=3)
    path = write_rows(rows, tmp_path / "rows.jsonl")   # no metrics.json beside it
    meta = metadata_from_rows(read_rows(path), dataset_spec="bundled")
    assert meta["is_simulated"] is True
    assert "SIMULATED" in render_report({"metadata": meta, "signals": {"engine": {}}})


def test_rows_without_provenance_are_unknown_not_real(tmp_path):
    """Legacy rows must degrade to 'unknown', never default to 'real'."""
    rows = run_harness(load_bundled()[:5], SimulatedAnswerer(seed=0), n_samples=3)
    path = write_rows(rows, tmp_path / "rows.jsonl")
    stripped = []
    for line in path.read_text().splitlines():
        obj = json.loads(line)
        obj.pop("is_simulated"), obj.pop("answerer"), obj.pop("model")
        stripped.append(json.dumps(obj))
    path.write_text("\n".join(stripped) + "\n")

    meta = metadata_from_rows(read_rows(path), dataset_spec="bundled")
    assert meta["is_simulated"] is None          # unknown, not False
    report = render_report({"metadata": meta, "signals": {"engine": {}}})
    assert "UNVERIFIED PROVENANCE" in report
    assert "PROVENANCE UNKNOWN" in report


def test_mixed_provenance_is_also_unknown():
    rows = run_harness(load_bundled()[:4], SimulatedAnswerer(seed=0), n_samples=3)
    rows[0].is_simulated = False                 # rows spliced from two runs
    assert metadata_from_rows(rows, dataset_spec="bundled")["is_simulated"] is None


# ------------------------------------------------- logprob baseline integrity
def test_partial_logprobs_withdraw_the_baseline(monkeypatch):
    """A mean over some samples isn't comparable with one over all N."""
    from eval import answerers

    class _Patchy:
        """Returns logprobs on the first sample only."""

        def __init__(self, logprobs):
            self.calls = 0

        def invoke(self, _prompt):
            self.calls += 1
            meta = (
                {"logprobs": {"content": [{"token": "a", "logprob": -0.5}]}}
                if self.calls == 1
                else {}
            )
            return type("M", (), {"content": "x", "response_metadata": meta})()

    monkeypatch.setitem(
        __import__("sys").modules,
        "core.llm",
        type("M", (), {"build_llm": staticmethod(lambda **kw: _Patchy(kw.get("logprobs")))}),
    )
    a = answerers.LlamaAnswerer()
    a.sampler_for(load_bundled()[0], n_samples=4)
    for _ in range(4):
        a.invoke("p")
    assert a.mean_logprob() is None      # 1 of 4 is not a usable item-level value


def test_complete_logprobs_are_reported(monkeypatch):
    from eval import answerers

    class _Full:
        def __init__(self, logprobs):
            pass

        def invoke(self, _prompt):
            return type(
                "M",
                (),
                {
                    "content": "x",
                    "response_metadata": {
                        "logprobs": {"content": [{"token": "a", "logprob": -1.0}]}
                    },
                },
            )()

    monkeypatch.setitem(
        __import__("sys").modules,
        "core.llm",
        type("M", (), {"build_llm": staticmethod(lambda **kw: _Full(kw.get("logprobs")))}),
    )
    a = answerers.LlamaAnswerer()
    a.sampler_for(load_bundled()[0], n_samples=3)
    for _ in range(3):
        a.invoke("p")
    assert a.mean_logprob() == pytest.approx(-1.0)


# --------------------------------------------------------------- gold answers
def test_every_loaded_item_has_a_gold_answer():
    """An item with no gold is a guaranteed false negative that poisons the labels."""
    for item in load_bundled():
        assert item.answers, f"{item.id} has no gold answer"
        assert all(str(a).strip() for a in item.answers)


def test_triviaqa_items_without_usable_aliases_are_dropped(monkeypatch):
    from eval import datasets as ds

    fake = [
        {"question_id": "keep", "question": "q1",
         "answer": {"value": "Paris", "aliases": ["Paris"]}},
        {"question_id": "drop", "question": "q2",
         "answer": {"value": "x" * 90, "aliases": ["y" * 80]}},   # all aliases too long
    ]
    monkeypatch.setattr(ds, "_load_first_available", lambda *a, **k: fake)
    items = ds._load_triviaqa(None)
    assert [i.id for i in items] == ["keep"]


# ------------------------------------------------- optional-dependency degrade
def test_learning_study_survives_missing_sklearn(monkeypatch):
    """scaling.latent_bayes imports fine without sklearn and only raises on construction."""
    import scaling.latent_bayes as lb
    from eval.learning import latent_confidences, run_learning_study, split_rows

    rows = run_harness(load_bundled()[:40], SimulatedAnswerer(seed=0), n_samples=3)

    def _boom(*_a, **_k):
        raise ImportError("scikit-learn is required for the scaling pipeline")

    monkeypatch.setattr(lb, "LatentBayesPipeline", _boom)
    split = split_rows(rows, seed=0)
    assert latent_confidences(split.train, split.test) is None   # degrades, not dies

    study = run_learning_study(rows, seed=0, n_boot=50)
    assert "latent" not in study
    assert study["posterior"]["ece"] >= 0.0                       # the rest still runs


# ------------------------------------------------------- coverage from rows
def test_float_noise_is_clamped_but_real_violations_raise():
    """1.0000000000000002 must not kill a run; 1.4 must still be rejected."""
    from eval.metrics import brier_score, expected_calibration_error

    # An ulp above 1.0 -- what cosine similarity of identical vectors actually returns.
    assert brier_score([1.0 + 2.2e-16, 0.0], [1, 0]) == pytest.approx(0.0)
    assert expected_calibration_error([1.0 + 2.2e-16], [1]) == pytest.approx(0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([1.4, 0.2], [1, 0])


def test_cosine_never_leaves_the_unit_interval():
    """Fix at source: identical bags used to return just over 1.0."""
    from collections import Counter

    from nodes.llm_executor import _cosine

    for text in ("the capital of france is paris", "a", "x y z x y z"):
        bag = Counter(text.split())
        assert 0.0 <= _cosine(bag, bag) <= 1.0
        assert _cosine(bag, bag) == pytest.approx(1.0)
    assert _cosine(Counter(), Counter("ab")) == 0.0


def test_report_coverage_matches_the_true_curve():
    """Acc @50% must come from row-level scores, not a fractional reliability bin."""
    from eval.metrics import evaluate_scores, risk_coverage
    from eval.report import _accuracy_at

    rng = np.random.default_rng(3)
    scores = rng.random(300)
    correct = (rng.random(300) < scores).astype(int)
    rollup = evaluate_scores(scores, correct, n_boot=50)
    assert _accuracy_at(rollup, 0.5) == pytest.approx(
        risk_coverage(scores, correct).accuracy_at_coverage(0.5)
    )


def test_report_refuses_to_interpolate_a_coverage_it_lacks():
    """Snapping 0.37 onto the 0.4 decile would mislabel which coverage the number is."""
    from eval.metrics import evaluate_scores
    from eval.report import _accuracy_at

    rollup = evaluate_scores([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], n_boot=20)
    assert _accuracy_at(rollup, 0.37) is None
    assert _accuracy_at(rollup, 0.5) is not None   # a decile we do hold


def test_conflict_telemetry_pairs_each_confidence_with_its_own_interval():
    """Dashboards must not band the gating confidence with the MAP state's interval.

    The logger binds its handler to sys.stdout at creation and sets propagate=False, so
    capsys cannot see it; attach a handler to the logger itself instead.
    """
    import logging

    from core.telemetry import get_logger, log_conflict_resolution

    captured = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(getattr(record, "extra_fields", {}))

    logger = get_logger()
    handler = _Capture()
    logger.addHandler(handler)
    try:
        summary = DirichletBayesianEngine().resolve(
            {"TaskStatus": 4, "DataQuality": 4, "ToolReliability": 4}
        ).as_dict()
        log_conflict_resolution(task="t", evidence={"TaskStatus": 4}, summary=summary)
    finally:
        logger.removeHandler(handler)

    assert captured, "no telemetry event was emitted"
    payload = captured[-1]
    low, high = payload["credible_interval"]
    assert low <= payload["confidence"] <= high, "confidence outside its own interval"
    # Both quantities are reported, and they are genuinely different.
    assert payload["map_credible_interval"] != payload["credible_interval"]
    assert payload["map_confidence"] != payload["confidence"]
