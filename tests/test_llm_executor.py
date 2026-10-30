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


def test_paraphrases_count_as_one_answer():
    """Reworded copies of one answer are not 'distinct answers'; exact-text dedup said they were."""
    model = FakeModel([
        "The system has five active users today.",
        "Today the system has five active users.",
        "The system has five active users today!",
        "Five active users are on the system today.",
    ])
    r = execute_step_with_llm(model, "count active users", n_samples=4)
    assert r.evidence["ToolReliability"] == 0


def test_no_access_answer_counts_as_unanswered():
    """An honest 'I don't have that data' must lower confidence, not read as a real answer."""
    model = FakeModel(["I don't have access to the user database for this system."])
    r = execute_step_with_llm(model, "list active users", n_samples=4)
    assert r.evidence["TaskStatus"] == 4


def test_step_prompt_forbids_inventing_data():
    seen = []

    class CapturingModel(FakeModel):
        def invoke(self, prompt):
            seen.append(prompt)
            return super().invoke(prompt)

    execute_step_with_llm(CapturingModel(["ok done"]), "list active users", n_samples=1)
    assert "never invent" in seen[0].lower()


def test_answers_with_different_numbers_are_different_answers():
    """Word overlap alone calls '3 users' and '12 users' the same answer; the numbers decide."""
    model = FakeModel([
        "There are 3 active users today.",
        "There are 12 active users today.",
        "There are 5 active users today.",
        "There are 7 active users today.",
    ])
    r = execute_step_with_llm(model, "count active users", n_samples=4)
    assert r.evidence["ToolReliability"] == 4


class _FakeEmbeddings:
    """Maps each text to a fixed vector, so a test can say what 'same meaning' is."""

    def __init__(self, vectors):
        self.vectors = vectors

    def embed_documents(self, texts):
        return [self.vectors[t] for t in texts]


def test_embeddings_see_agreement_that_word_overlap_misses(monkeypatch):
    """'Yes.' and 'Correct, that is right.' share no words but mean the same thing."""
    same = [1.0, 0.0]
    fake = _FakeEmbeddings({"Yes.": same, "Correct, that is right.": same})
    monkeypatch.setattr("nodes.llm_executor.build_embeddings", lambda **_kwargs: fake)
    model = FakeModel(["Yes.", "Correct, that is right."])

    r = execute_step_with_llm(model, "is it right?", n_samples=4)

    assert r.consistency > 0.99
    assert r.evidence["DataQuality"] == 0
    assert r.evidence["ToolReliability"] == 0


def test_embeddings_still_split_answers_with_different_numbers(monkeypatch):
    """Embeddings blur numbers too (0.84 measured live), so the number check stays."""
    v = [1.0, 0.0]
    fake = _FakeEmbeddings({"There are 3 users.": v, "There are 12 users.": v})
    monkeypatch.setattr("nodes.llm_executor.build_embeddings", lambda **_kwargs: fake)
    model = FakeModel(["There are 3 users.", "There are 12 users."])

    r = execute_step_with_llm(model, "count users", n_samples=2)

    assert r.evidence["ToolReliability"] == 4
