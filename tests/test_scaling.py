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
