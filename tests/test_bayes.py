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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
