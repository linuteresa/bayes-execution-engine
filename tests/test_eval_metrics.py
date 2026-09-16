"""Unit tests for the calibration metrics.

These are the numbers the README's headline claim rests on, so they are tested against
hand-computable cases rather than against themselves.
"""

import numpy as np
import pytest

from eval.metrics import (
    auroc,
    bootstrap_ci,
    bootstrap_diff,
    brier_score,
    evaluate_scores,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
    risk_coverage,
)


def test_perfect_calibration_has_zero_ece():
    """10 items at confidence 0.7 of which exactly 7 are right => ECE 0."""
    scores = [0.7] * 10
    correct = [1] * 7 + [0] * 3
    assert expected_calibration_error(scores, correct, n_bins=10) == pytest.approx(0.0)


def test_total_overconfidence_has_maximal_ece():
    scores = [1.0] * 8
    correct = [0] * 8
    assert expected_calibration_error(scores, correct, n_bins=10) == pytest.approx(1.0)


def test_ece_is_count_weighted():
    """A big well-calibrated bin must dominate a tiny badly-calibrated one."""
    scores = [0.5] * 98 + [1.0, 1.0]
    correct = [1] * 49 + [0] * 49 + [0, 0]
    # 98 items contribute 0; 2 items contribute |1.0 - 0| * 2/100 = 0.02
    assert expected_calibration_error(scores, correct, n_bins=10) == pytest.approx(0.02)


def test_mce_ignores_bin_size():
    scores = [0.5] * 98 + [1.0, 1.0]
    correct = [1] * 49 + [0] * 49 + [0, 0]
    assert maximum_calibration_error(scores, correct, n_bins=10) == pytest.approx(1.0)


def test_reliability_bins_partition_the_items():
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    correct = (rng.random(200) < scores).astype(int)
    bins = reliability_bins(scores, correct, n_bins=10)
    assert sum(b.count for b in bins) == 200
    assert all(b.lower <= b.mean_confidence <= b.upper for b in bins)


def test_reliability_bin_includes_exactly_one():
    """A score of exactly 1.0 belongs in the top bin, not off the end."""
    bins = reliability_bins([1.0, 1.0], [1, 0], n_bins=10)
    assert len(bins) == 1
    assert bins[0].upper == pytest.approx(1.0)
    assert bins[0].count == 2


def test_brier_score_known_value():
    # (0.8-1)^2 + (0.3-0)^2 = 0.04 + 0.09, mean = 0.065
    assert brier_score([0.8, 0.3], [1, 0]) == pytest.approx(0.065)


def test_auroc_perfect_and_chance():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(0.0)


def test_auroc_all_ties_is_one_half():
    """A constant score carries no information -- exactly chance, not 1.0."""
    assert auroc([0.5] * 6, [1, 0, 1, 0, 1, 0]) == pytest.approx(0.5)


def test_auroc_undefined_without_both_labels():
    assert np.isnan(auroc([0.9, 0.1], [1, 1]))


def test_risk_coverage_improves_for_a_good_ranker():
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    correct = [1, 1, 1, 0, 0]
    curve = risk_coverage(scores, correct)
    assert curve.full_accuracy == pytest.approx(0.6)
    assert curve.accuracy_at_coverage(0.6) == pytest.approx(1.0)
    assert curve.aurc < 0.4  # much better than answering everything


def test_risk_coverage_flat_for_an_uninformative_ranker():
    correct = [1, 0, 1, 0]
    curve = risk_coverage([0.5] * 4, correct)
    assert curve.accuracy_at_coverage(1.0) == pytest.approx(0.5)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    scores = rng.random(300)
    correct = (rng.random(300) < scores).astype(int)
    interval = bootstrap_ci(lambda s, y: brier_score(s, y), scores, correct, n_boot=400, seed=3)
    assert interval.low <= interval.point <= interval.high
    assert interval.high - interval.low < 0.2


def test_bootstrap_diff_detects_a_real_gap():
    """A perfect ranker vs a constant one: the AUROC difference must exclude 0."""
    rng = np.random.default_rng(2)
    correct = rng.integers(0, 2, size=200)
    good = correct + rng.normal(0, 0.05, size=200)
    flat = np.full(200, 0.5)
    delta = bootstrap_diff(lambda s, y: auroc(s, y), good, flat, correct, n_boot=400, seed=5)
    assert delta.point > 0.4
    assert delta.low > 0


def test_bootstrap_diff_straddles_zero_for_equivalent_signals():
    rng = np.random.default_rng(4)
    correct = rng.integers(0, 2, size=200)
    a = rng.random(200)
    b = rng.random(200)
    delta = bootstrap_diff(lambda s, y: auroc(s, y), a, b, correct, n_boot=400, seed=6)
    assert delta.low < 0 < delta.high


def test_evaluate_scores_rollup_shape():
    rng = np.random.default_rng(7)
    scores = rng.random(150)
    correct = (rng.random(150) < scores).astype(int)
    out = evaluate_scores(scores, correct, n_boot=200, seed=0, abstain_fraction=0.2)
    assert out["n"] == 150
    assert 0.0 <= out["ece"] <= 1.0
    assert set(out["brier"]) == {"point", "ci_low", "ci_high"}
    assert out["abstain_fraction"] == pytest.approx(0.2)
    assert sum(b["count"] for b in out["reliability"]) == 150


def test_metrics_reject_malformed_input():
    with pytest.raises(ValueError):
        brier_score([0.5, 0.5], [1])
    with pytest.raises(ValueError):
        brier_score([], [])
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [2])
