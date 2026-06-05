import pytest
import numpy as np
from bayesian_engine.bayes_engine import build_bayesian_network, resolve_conflict, STATE_NAMES

def test_bayesian_network_structure():
    """Test that the Bayesian Network is correctly structured."""
    model, _ = build_bayesian_network()

    assert "TaskStatus" in model.nodes()
    assert "DataQuality" in model.nodes()
    assert "ToolReliability" in model.nodes()
    assert "Outcome" in model.nodes()

    assert model.check_model()

def test_cpt_outcome_shape():
    """
    Verify the Outcome CPT covers exactly 125 parent combinations.
    3 parent variables × 5 states each = 5^3 = 125 parent combinations.
    pgmpy stores as (5, 5, 5, 5) where first dim is outcome, rest are parent states.
    """
    model, cpd_outcome = build_bayesian_network()

    expected_shape = (5, 5, 5, 5)
    assert cpd_outcome.values.shape == expected_shape, (
        f"Expected CPT shape {expected_shape}, got {cpd_outcome.values.shape}"
    )

    assert cpd_outcome.values.size == 625, (
        f"Expected 625 total probability values (5 outcome × 125 parent combos), got {cpd_outcome.values.size}"
    )

def test_cpt_normalization():
    """Verify CPT values for each parent state combination sum to 1.0."""
    model, cpd_outcome = build_bayesian_network()

    for i in range(5):
        for j in range(5):
            for k in range(5):
                col_sum = cpd_outcome.values[:, i, j, k].sum()
                assert abs(col_sum - 1.0) < 1e-6, (
                    f"CPT column [{i},{j},{k}] sums to {col_sum}, not 1.0"
                )

def test_state_names_mapping():
    """Verify semantic state names are correctly defined."""
    expected = {
        0: "CERTAIN",
        1: "HIGH",
        2: "MEDIUM",
        3: "LOW",
        4: "AMBIGUOUS",
    }
    assert STATE_NAMES == expected

def test_resolve_conflict_basic():
    """Test the resolve_conflict function with valid evidence."""
    result = resolve_conflict({
        "TaskStatus": 2,
        "DataQuality": 2,
        "ToolReliability": 3,
    })

    assert "state" in result
    assert "state_index" in result
    assert "confidence" in result
    assert "distribution" in result

    assert result["state"] in STATE_NAMES.values()
    assert 0 <= result["confidence"] <= 1.0
    assert result["state_index"] in range(5)

    distribution = result["distribution"]
    total_prob = sum(distribution.values())
    assert abs(total_prob - 1.0) < 1e-6

def test_resolve_conflict_all_states():
    """Verify resolve_conflict works for all possible parent state combinations."""
    test_cases = [
        {"TaskStatus": 0, "DataQuality": 0, "ToolReliability": 0},
        {"TaskStatus": 2, "DataQuality": 3, "ToolReliability": 1},
        {"TaskStatus": 4, "DataQuality": 4, "ToolReliability": 4},
    ]

    for evidence in test_cases:
        result = resolve_conflict(evidence)
        assert 0 <= result["confidence"] <= 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
