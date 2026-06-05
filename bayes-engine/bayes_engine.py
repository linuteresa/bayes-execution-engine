import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

STATE_NAMES = {0: "CERTAIN", 1: "HIGH", 2: "MEDIUM", 3: "LOW", 4: "AMBIGUOUS"}
STATE_MAP = {v: k for k, v in STATE_NAMES.items()}

def build_bayesian_network():
    """
    Build a Bayesian Network with 3 parent variables and 1 outcome.
    Parents: TaskStatus, DataQuality, ToolReliability (5 states each)
    Outcome: Confidence level (5 semantic states)
    Total: 5 * 5 * 5 = 125 parent combinations
    """
    model = DiscreteBayesianNetwork([
        ('TaskStatus', 'Outcome'),
        ('DataQuality', 'Outcome'),
        ('ToolReliability', 'Outcome'),
    ])

    cpd_task = TabularCPD(
        variable='TaskStatus',
        variable_card=5,
        values=[[0.2], [0.2], [0.2], [0.2], [0.2]]
    )

    cpd_data = TabularCPD(
        variable='DataQuality',
        variable_card=5,
        values=[[0.2], [0.2], [0.2], [0.2], [0.2]]
    )

    cpd_tool = TabularCPD(
        variable='ToolReliability',
        variable_card=5,
        values=[[0.2], [0.2], [0.2], [0.2], [0.2]]
    )

    values_outcome = np.random.dirichlet(np.ones(5), size=125)
    cpd_outcome = TabularCPD(
        variable='Outcome',
        variable_card=5,
        values=values_outcome.T,
        evidence=['TaskStatus', 'DataQuality', 'ToolReliability'],
        evidence_card=[5, 5, 5]
    )

    model.add_cpds(cpd_task, cpd_data, cpd_tool, cpd_outcome)
    assert model.check_model()

    return model, cpd_outcome

def resolve_conflict(evidence: dict):
    """
    Run Variable Elimination inference to resolve ambiguous/conflicting data.
    evidence: dict mapping variable names to state indices (0-based)
    Returns: dict with state name, confidence, and full distribution
    """
    model, _ = build_bayesian_network()
    infer = VariableElimination(model)

    result = infer.query(variables=['Outcome'], evidence=evidence)
    probabilities = result.values.flatten()

    max_idx = np.argmax(probabilities)
    max_prob = float(probabilities[max_idx])
    state_name = STATE_NAMES[max_idx]

    return {
        "state": state_name,
        "state_index": int(max_idx),
        "confidence": max_prob,
        "distribution": {STATE_NAMES[i]: float(p) for i, p in enumerate(probabilities)}
    }
