"""Unit tests for self-consistency execution and the LLM executor path."""

from nodes.executor import executor_node
from nodes.llm_executor import execute_step_with_llm


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeModel:
    """Returns scripted outputs, cycling through them on each invoke()."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def invoke(self, _prompt):
        out = self.outputs[self.calls % len(self.outputs)]
        self.calls += 1
        return FakeMessage(out)


def test_consistent_samples_are_high_quality():
    model = FakeModel(["The capital of France is Paris."])
    r = execute_step_with_llm(model, "name the capital of France", n_samples=4)
    assert r.consistency > 0.95
    assert r.evidence == {"TaskStatus": 0, "DataQuality": 0, "ToolReliability": 0}
    assert "Paris" in r.answer


def test_divergent_samples_signal_conflict():
    model = FakeModel([
        "The capital is Paris in northern France.",
        "Quantum chromodynamics describes the strong nuclear force.",
        "Bananas are an excellent source of dietary potassium.",
        "The 1929 stock market crash began on Black Tuesday.",
    ])
    r = execute_step_with_llm(model, "ambiguous step", n_samples=4)
    assert r.consistency < 0.3
    assert r.evidence["DataQuality"] >= 3      # low agreement
    assert r.evidence["ToolReliability"] >= 3  # all answers distinct


def test_refusals_lower_task_status():
    model = FakeModel(["I don't know.", "I do not know.", "I cannot answer that.", "Unable to."])
    r = execute_step_with_llm(model, "unanswerable step", n_samples=4)
    assert r.evidence["TaskStatus"] >= 3       # nothing answerable


def test_medoid_picks_the_consensus_answer():
    model = FakeModel([
        "Paris is the capital of France.",
        "The capital of France is Paris.",
        "Paris, the capital city of France.",
        "Mitochondria are the powerhouse of the cell.",  # outlier
    ])
    r = execute_step_with_llm(model, "capital of France", n_samples=4)
    assert "Paris" in r.answer                 # never the outlier


def test_single_sample_is_trivially_consistent():
    model = FakeModel(["A concise factual answer."])
    r = execute_step_with_llm(model, "step", n_samples=1)
    assert r.consistency == 1.0


def _state(plan, question="test goal"):
    return {"input": question, "plan": plan, "past_steps": [], "response": "", "confidence_score": 1.0}


def test_executor_llm_path_returns_real_answer():
    model = FakeModel(["The answer is forty-two."])
    config = {"configurable": {"executor_model": model}}
    result = executor_node(_state(["compute the answer"]), config)
    assert result["past_steps"][0][1] == "The answer is forty-two."
    assert 0.0 < result["confidence_score"] <= 1.0
    assert result["plan"] == []


def test_executor_llm_path_low_confidence_on_disagreement():
    consistent = FakeModel(["Paris is the capital of France."])
    divergent = FakeModel([
        "Paris sits on the Seine in France.",
        "Photosynthesis converts sunlight into chemical energy.",
        "The Treaty of Westphalia was signed in 1648.",
        "Octopuses have three hearts and blue blood.",
    ])
    c_conf = executor_node(_state(["x"]), {"configurable": {"executor_model": consistent}})["confidence_score"]
    d_conf = executor_node(_state(["x"]), {"configurable": {"executor_model": divergent}})["confidence_score"]
    assert d_conf < c_conf  # disagreement must reduce confidence


def test_executor_falls_back_to_mock_without_model():
    result = executor_node(_state(["query the database"]), config=None)
    assert "query_result" in result["past_steps"][0][1]
    assert 0.0 < result["confidence_score"] < 1.0
