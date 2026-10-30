"""Unit tests for the conjugate Dirichlet-Multinomial Bayesian engine."""

import numpy as np
import pytest

from bayesian_engine.bayes_engine import (
    N_CONTEXTS,
    N_STATES,
    STATE_NAMES,
    DirichletBayesianEngine,
    context_index,
    context_triple,
    resolve_conflict,
)


def test_context_index_roundtrip():
    seen = set()
    for t in range(N_STATES):
        for d in range(N_STATES):
            for r in range(N_STATES):
                idx = context_index(t, d, r)
                seen.add(idx)
                assert context_triple(idx) == (t, d, r)
    assert len(seen) == N_CONTEXTS == 125


def test_context_index_validates_range():
    with pytest.raises(ValueError):
        context_index(5, 0, 0)


def test_prior_is_proper_and_normalised():
    eng = DirichletBayesianEngine()
    cpt = eng.cpt()
    assert cpt.shape == (N_STATES, N_CONTEXTS)
    # Every context column of the posterior-predictive CPT sums to 1.
    assert np.allclose(cpt.sum(axis=0), 1.0)
    # Proper prior: strictly positive concentration everywhere.
    assert (eng.alpha > 0).all()


def test_prior_is_deterministic():
    a = DirichletBayesianEngine().alpha
    b = DirichletBayesianEngine().alpha
    assert np.array_equal(a, b)  # no randomness, unlike the old implementation


def test_prior_monotonicity():
    """Best signals -> CERTAIN; worst signals -> AMBIGUOUS."""
    eng = DirichletBayesianEngine()
    good = eng.resolve({"TaskStatus": 0, "DataQuality": 0, "ToolReliability": 0})
    bad = eng.resolve({"TaskStatus": 4, "DataQuality": 4, "ToolReliability": 4})
    assert good.state == "CERTAIN"
    assert bad.state == "AMBIGUOUS"


def test_expected_outcome_increases_with_degradation():
    """The MAP outcome index should be monotonic in aggregate degradation."""
    eng = DirichletBayesianEngine()
    last = -1
    for level in range(N_STATES):
        idx = eng.resolve(
            {"TaskStatus": level, "DataQuality": level, "ToolReliability": level}
        ).state_index
        assert idx >= last
        last = idx


def test_conjugate_update_shifts_posterior():
    """Observing CERTAIN outcomes pulls the predictive toward CERTAIN."""
    eng = DirichletBayesianEngine()
    ctx = {"TaskStatus": 2, "DataQuality": 2, "ToolReliability": 2}
    before = eng.resolve(ctx).distribution["CERTAIN"]
    for _ in range(50):
        eng.observe(2, 2, 2, 0)
    after = eng.resolve(ctx).distribution["CERTAIN"]
    assert after > before + 0.3


def test_effective_sample_size_grows_with_data():
    eng = DirichletBayesianEngine()
    ctx = {"TaskStatus": 1, "DataQuality": 1, "ToolReliability": 1}
    ess0 = eng.resolve(ctx).effective_sample_size
    eng.fit([(1, 1, 1, 0)] * 20)
    ess1 = eng.resolve(ctx).effective_sample_size
    assert ess1 == pytest.approx(ess0 + 20, abs=1e-6)


def test_reset_restores_prior():
    eng = DirichletBayesianEngine()
    snapshot = eng.alpha.copy()
    eng.fit([(0, 0, 0, 4)] * 10)
    eng.reset()
    assert np.array_equal(eng.alpha, snapshot)


def test_partial_evidence_marginalises():
    """Querying with a missing parent still yields a valid distribution."""
    eng = DirichletBayesianEngine()
    summary = eng.resolve({"DataQuality": 4})  # TaskStatus, ToolReliability unknown
    assert abs(sum(summary.distribution.values()) - 1.0) < 1e-9
    assert summary.state in STATE_NAMES.values()


def test_credible_interval_brackets_confidence():
    eng = DirichletBayesianEngine()
    s = eng.resolve({"TaskStatus": 0, "DataQuality": 0, "ToolReliability": 0})
    assert 0.0 <= s.credible_low <= s.confidence <= s.credible_high <= 1.0


def test_resolve_conflict_backwards_compatible_keys():
    out = resolve_conflict({"TaskStatus": 2, "DataQuality": 1, "ToolReliability": 3})
    for key in ("state", "state_index", "confidence", "distribution"):
        assert key in out
    assert 0.0 <= out["confidence"] <= 1.0
    assert abs(sum(out["distribution"].values()) - 1.0) < 1e-9


def test_invalid_outcome_rejected():
    eng = DirichletBayesianEngine()
    with pytest.raises(ValueError):
        eng.observe(0, 0, 0, 5)


def test_pgmpy_network_optional():
    """If pgmpy is installed, the equivalent network must validate."""
    pytest.importorskip("pgmpy")
    eng = DirichletBayesianEngine()
    model = eng.build_network()
    assert model.check_model()
    cpd = model.get_cpds("Outcome")
    assert cpd.values.size == N_STATES * N_CONTEXTS  # 625


@pytest.mark.parametrize("value", [-1, 5, 0.5, 1.0, True, float("nan"), "1"])
def test_context_indices_reject_noninteger_and_out_of_range_values(value):
    with pytest.raises(ValueError):
        context_index(value, 0, 0)


@pytest.mark.parametrize("value", [-1, 125, 0.5, True])
def test_inverse_context_index_validates_input(value):
    with pytest.raises(ValueError):
        context_triple(value)


@pytest.mark.parametrize("kwargs", [
    {"prior_strength": -1}, {"prior_strength": float("inf")},
    {"prior_spread": 0}, {"prior_spread": -1}, {"prior_spread": float("nan")},
])
def test_invalid_prior_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DirichletBayesianEngine(**kwargs)


def test_tiny_positive_spread_still_produces_a_valid_prior():
    engine = DirichletBayesianEngine(prior_spread=1e-300)
    assert np.isfinite(engine.alpha).all()
    assert np.allclose(engine.cpt().sum(axis=0), 1)


@pytest.mark.parametrize("count", [-1, 0.5, True, float("nan"), float("inf"), 10 ** 1000])
def test_invalid_counts_leave_posterior_unchanged(count):
    engine = DirichletBayesianEngine()
    before = engine.alpha.copy()
    with pytest.raises(ValueError):
        engine.observe(0, 0, 0, 0, count=count)
    np.testing.assert_array_equal(engine.alpha, before)


def test_invalid_batch_fit_is_atomic():
    engine = DirichletBayesianEngine()
    before = engine.alpha.copy()
    with pytest.raises(ValueError):
        engine.fit([(0, 0, 0, 0), (0, 0, 0, -1)])
    np.testing.assert_array_equal(engine.alpha, before)


@pytest.mark.parametrize("bad_value", [-1, float("nan"), float("inf"), "invalid"])
def test_invalid_saved_counts_do_not_replace_existing_posterior(tmp_path, bad_value):
    engine = DirichletBayesianEngine()
    engine.observe(0, 0, 0, 0)
    before = engine.alpha.copy()
    counts = np.zeros((N_CONTEXTS, N_STATES), dtype=object if isinstance(bad_value, str) else float)
    counts[0, 0] = bad_value
    path = tmp_path / "invalid.npy"
    np.save(path, counts)
    with pytest.raises(ValueError):
        engine.load(path)
    np.testing.assert_array_equal(engine.alpha, before)


def test_partial_interval_follows_uniform_average_with_unequal_context_counts():
    engine = DirichletBayesianEngine()
    engine.observe(0, 0, 0, 0, count=10000)
    evidence = {"TaskStatus": 0, "DataQuality": 0}
    summary = engine.resolve(evidence)
    assert summary.good_confidence == pytest.approx(0.6780106)
    assert summary.good_credible_low < summary.good_confidence < summary.good_credible_high
    # Independent high-resolution posterior simulation gave [0.575, 0.777].
    assert summary.good_credible_low == pytest.approx(0.575, abs=0.015)
    assert summary.good_credible_high == pytest.approx(0.777, abs=0.015)
    assert summary.as_dict() == engine.resolve(evidence).as_dict()


def test_full_evidence_interval_remains_exact_beta_quantiles():
    from scipy.stats import beta

    engine = DirichletBayesianEngine()
    engine.observe(1, 2, 3, 0, count=20)
    a = engine.alpha[context_index(1, 2, 3)]
    expected = beta.ppf([0.025, 0.975], a[:2].sum(), a[2:].sum())
    summary = engine.resolve({"TaskStatus": 1, "DataQuality": 2, "ToolReliability": 3})
    assert [summary.good_credible_low, summary.good_credible_high] == pytest.approx(expected)


def test_unknown_evidence_is_rejected_instead_of_silently_marginalized():
    with pytest.raises(ValueError, match="unknown evidence"):
        DirichletBayesianEngine().resolve({"DataQualty": 4})


def test_concurrent_initialization_returns_one_shared_engine(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import bayesian_engine.bayes_engine as be

    monkeypatch.setattr(be, "_DEFAULT_ENGINE", None)
    monkeypatch.setattr(be, "_COUNT_SOURCE", None)
    monkeypatch.delenv("ENGINE_STATE_PATH", raising=False)
    original = be.DirichletBayesianEngine
    created = []
    initialization_started = threading.Event()
    release_initialization = threading.Event()

    def slow_constructor():
        instance = original()
        created.append(instance)
        initialization_started.set()
        assert release_initialization.wait(5)
        return instance

    monkeypatch.setattr(be, "DirichletBayesianEngine", slow_constructor)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(be.get_default_engine)
        assert initialization_started.wait(5)
        second = pool.submit(be.get_default_engine)
        release_initialization.set()
        assert first.result(timeout=5) is second.result(timeout=5)
    assert len(created) == 1


def test_shared_count_source_refreshes_without_accumulating_snapshots(monkeypatch):
    import bayesian_engine.bayes_engine as be

    monkeypatch.setattr(be, "_DEFAULT_ENGINE", None)
    monkeypatch.setattr(be, "_COUNT_SOURCE", None)
    counts = np.zeros((N_CONTEXTS, N_STATES))
    be.set_count_source(lambda: counts)
    engine = be.get_default_engine()
    assert engine.total_observations() == 0
    counts[0, 0] = 2
    assert be.get_default_engine() is engine
    assert engine.total_observations() == 2
    assert be.get_default_engine().total_observations() == 2
    before = engine.alpha.copy()
    counts[1, 0] = -1
    with pytest.raises(ValueError):
        be.get_default_engine()
    np.testing.assert_array_equal(engine.alpha, before)


def test_concurrent_snapshot_saves_use_separate_temporary_files(monkeypatch, tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import bayesian_engine.bayes_engine as be

    engines = [DirichletBayesianEngine(), DirichletBayesianEngine()]
    engines[0].observe(0, 0, 0, 0)
    engines[1].observe(1, 1, 1, 0)
    barrier = threading.Barrier(2)
    original_replace = be.os.replace
    temp_paths = []

    def synchronized_replace(source, target):
        temp_paths.append(source)
        barrier.wait(timeout=5)
        original_replace(source, target)

    monkeypatch.setattr(be.os, "replace", synchronized_replace)
    path = tmp_path / "counts.npy"
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda engine: engine.save(path), engines))
    assert len(set(temp_paths)) == 2
    restored = DirichletBayesianEngine()
    restored.load(path)
    assert restored.total_observations() == pytest.approx(1)
    assert list(tmp_path.iterdir()) == [path]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
