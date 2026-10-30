"""Tests for the PCA latent-space scaling pipeline."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from scaling.latent_bayes import (  # noqa: E402
    GeneralDirichletGrid,
    LatentBayesPipeline,
    make_synthetic_agent_signals,
    run_demo,
)


def test_general_grid_context_count():
    grid = GeneralDirichletGrid(n_factors=3, n_bins=5)
    assert grid.n_contexts == 125
    assert grid.alpha.shape == (125, 5)


def test_general_grid_conjugate_update():
    grid = GeneralDirichletGrid(n_factors=2, n_bins=3, n_outcomes=4)
    bins = np.array([1, 2])
    for _ in range(10):
        grid.observe(bins, outcome=3)
    assert grid.predict(bins) == 3


def test_pipeline_reduces_state_space():
    X, y = make_synthetic_agent_signals(n_samples=500, raw_dim=8)
    pipe = LatentBayesPipeline(n_components=3, n_bins=5).fit_projection(X)
    naive, latent = pipe.state_reduction(raw_dim=8)
    assert naive == 5 ** 8
    assert latent == 125
    assert naive / latent > 1000


def test_pipeline_beats_majority_baseline():
    report = run_demo()
    assert report["test_accuracy"] > report["majority_baseline"] + 0.2
    assert report["explained_variance"] > 0.7


def test_pipeline_is_reproducible():
    a = run_demo()
    b = run_demo()
    assert a == b


def test_bins_within_range():
    X, _ = make_synthetic_agent_signals(n_samples=300, raw_dim=6)
    pipe = LatentBayesPipeline(n_components=3, n_bins=5).fit_projection(X)
    bins = pipe.to_bins(X)
    assert bins.min() >= 0 and bins.max() <= 4
    assert bins.shape == (len(X), 3)


@pytest.mark.parametrize("bins", [[-1, 0], [0, 3], [0], [0, 1, 2], [0.5, 1.5], [True, False]])
def test_grid_rejects_invalid_bins_without_updating_another_context(bins):
    grid = GeneralDirichletGrid(n_factors=2, n_bins=3)
    before = grid.alpha.copy()
    with pytest.raises(ValueError):
        grid.observe(bins, 0)
    np.testing.assert_array_equal(grid.alpha, before)


@pytest.mark.parametrize("outcome", [-1, 5, 0.5, True])
def test_grid_rejects_invalid_outcome(outcome):
    grid = GeneralDirichletGrid(n_factors=2)
    before = grid.alpha.copy()
    with pytest.raises(ValueError):
        grid.observe([0, 0], outcome)
    np.testing.assert_array_equal(grid.alpha, before)


@pytest.mark.parametrize("kwargs", [
    {"n_factors": 0}, {"n_factors": 1.5}, {"n_factors": 2, "n_bins": 0},
    {"n_factors": 2, "n_outcomes": -1}, {"n_factors": 2, "prior": 0},
    {"n_factors": 2, "prior": float("nan")},
])
def test_grid_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        GeneralDirichletGrid(**kwargs)


@pytest.mark.parametrize("labels", [[0], [0, 1, 2], [0, -1], [0, 5], [0.0, 0.5]])
def test_pipeline_update_validates_all_labels_before_mutating(labels):
    X = np.array([[0.0], [1.0]])
    pipe = LatentBayesPipeline(n_components=1).fit_projection(X)
    before = pipe.grid.alpha.copy()
    with pytest.raises(ValueError):
        pipe.update(X, np.array(labels))
    np.testing.assert_array_equal(pipe.grid.alpha, before)


def test_projection_refit_discards_counts_attached_to_old_bins():
    X = np.arange(20.0).reshape(-1, 1)
    pipe = LatentBayesPipeline(n_components=1, n_bins=2).fit_projection(X)
    pipe.update(X, np.array([0] * 10 + [1] * 10))
    assert pipe.coverage() == 2
    pipe.fit_projection(X + 1000)
    assert pipe.coverage() == 0
    np.testing.assert_allclose(pipe.predict_proba([[1019]]), [[0.2] * 5])


def test_failed_projection_refit_preserves_existing_model():
    X = np.arange(20.0).reshape(-1, 1)
    pipe = LatentBayesPipeline(n_components=1).fit_projection(X)
    pipe.update(X, np.zeros(len(X), dtype=int))
    before = pipe.predict_proba(X)
    with pytest.raises(ValueError):
        pipe.fit_projection(np.array([[float("nan")]]))
    np.testing.assert_array_equal(pipe.predict_proba(X), before)


def test_projection_requires_a_fitted_model():
    with pytest.raises(ValueError, match="fit_projection"):
        LatentBayesPipeline().to_bins(np.zeros((2, 3)))
