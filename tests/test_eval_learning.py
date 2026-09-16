"""Tests for the learning loop -- the part that finally calls engine.observe()."""

import pytest

from bayesian_engine.bayes_engine import DirichletBayesianEngine, context_index
from eval.answerers import SimulatedAnswerer
from eval.datasets import load_bundled
from eval.harness import run_harness
from eval.learning import (
    ci_width_vs_ess,
    context_coverage,
    fit_engine,
    latent_confidences,
    learning_curve,
    run_learning_study,
    score_rows,
    split_rows,
)


@pytest.fixture(scope="module")
def rows():
    return run_harness(load_bundled(), SimulatedAnswerer(seed=0), n_samples=4)


def test_split_is_stratified_and_disjoint(rows):
    split = split_rows(rows, test_fraction=0.5, seed=0)
    train_ids = {r.item_id for r in split.train}
    test_ids = {r.item_id for r in split.test}
    assert not (train_ids & test_ids)
    assert len(train_ids) + len(test_ids) == len(rows)
    train_acc = sum(r.correct for r in split.train) / len(split.train)
    test_acc = sum(r.correct for r in split.test) / len(split.test)
    assert abs(train_acc - test_acc) < 0.12  # stratification keeps the halves comparable


def test_fit_engine_actually_observes(rows):
    """The regression this whole exercise exists to prevent: observe() never called."""
    split = split_rows(rows, seed=0)
    engine = fit_engine(split.train)
    assert engine.total_observations() == pytest.approx(len(split.train))
    assert engine.coverage() >= 1
    # A prior-only engine has no observations at all -- the shipped state.
    assert DirichletBayesianEngine().total_observations() == pytest.approx(0.0)


def test_observing_moves_the_posterior_off_the_prior(rows):
    split = split_rows(rows, seed=0)
    prior = score_rows(split.test, DirichletBayesianEngine())
    posterior = score_rows(split.test, fit_engine(split.train))
    assert prior["confidence"] != posterior["confidence"]
    # Every observation adds one pseudo-count, so ESS can only grow.
    assert sum(posterior["ess"]) > sum(prior["ess"])


def test_credible_intervals_shrink_as_evidence_accumulates():
    """'alpha_0 IS the effective sample size' -- as a measurement, not an assertion."""
    engine = DirichletBayesianEngine()
    ctx = {"TaskStatus": 1, "DataQuality": 1, "ToolReliability": 1}
    before = engine.resolve(ctx)
    for _ in range(200):
        engine.observe(1, 1, 1, 0)
    after = engine.resolve(ctx)
    width_before = before.credible_high - before.credible_low
    width_after = after.credible_high - after.credible_low
    assert width_after < width_before
    assert after.effective_sample_size > before.effective_sample_size


def test_ci_width_vs_ess_is_monotone_decreasing(rows):
    split = split_rows(rows, seed=0)
    buckets = ci_width_vs_ess(score_rows(split.test, fit_engine(split.train)))
    assert buckets
    widths = [b["mean_ci_width"] for b in buckets]
    assert widths == sorted(widths, reverse=True)


def test_context_coverage_counts_visited_cells(rows):
    cov = context_coverage(rows)
    assert cov["n_contexts"] == 125
    assert 1 <= cov["contexts_visited"] <= 125
    assert cov["n_observations"] == len(rows)
    # The honest finding: a few hundred items cannot populate 125 cells.
    assert cov["contexts_visited"] < 125


def test_context_coverage_matches_explicit_indices():
    class _Row:
        def __init__(self, t, d, r):
            self.task_status, self.data_quality, self.tool_reliability = t, d, r

    cov = context_coverage([_Row(0, 0, 0), _Row(0, 0, 0), _Row(4, 4, 4)])
    assert cov["contexts_visited"] == 2
    assert cov["max_count_in_one_context"] == 2
    assert {c["context_index"] for c in cov["top_contexts"]} == {
        context_index(0, 0, 0),
        context_index(4, 4, 4),
    }


def test_learning_curve_starts_at_the_prior(rows):
    curve = learning_curve(rows, seed=0)
    assert curve[0]["n_train"] == 0
    assert curve[-1]["n_train"] > 0
    # More data can only tighten intervals, whatever happens to ECE.
    assert curve[-1]["mean_ci_width"] < curve[0]["mean_ci_width"]
    assert curve[-1]["mean_ess"] > curve[0]["mean_ess"]


def test_latent_pipeline_scores_the_same_held_out_items(rows):
    pytest.importorskip("sklearn")
    split = split_rows(rows, seed=0)
    latent = latent_confidences(split.train, split.test)
    assert latent is not None
    assert len(latent["confidence"]) == len(split.test)
    assert all(0.0 <= c <= 1.0 for c in latent["confidence"])
    # Quantile bins are populated by construction, unlike the hand-designed grid.
    assert latent["latent_contexts_visited"] >= 1
    assert latent["naive_raw_contexts"] > latent["latent_contexts"]


def test_learning_study_reports_both_engines_on_held_out_data(rows):
    study = run_learning_study(rows, seed=0, n_boot=200)
    assert study["n_train"] + study["n_test"] == len(rows)
    for key in ("prior_only", "posterior"):
        assert 0.0 <= study[key]["ece"] <= 1.0
        assert study[key]["n"] == study["n_test"]
    assert set(study["ece_delta_posterior_minus_prior"]) == {"point", "ci_low", "ci_high"}
    assert study["coverage_all"]["n_contexts"] == 125


def test_split_rejects_impossible_fractions(rows):
    with pytest.raises(ValueError):
        split_rows(rows, test_fraction=0.0)
    with pytest.raises(ValueError):
        split_rows(rows[:1], test_fraction=0.5)
