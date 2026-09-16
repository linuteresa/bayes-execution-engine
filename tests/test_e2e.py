"""Tests for the executor DAG node and graph routing logic."""

import pytest

from nodes.executor import executor_node, simple_executor
from nodes.replanner import _filter_repeated_steps, replanner_node


def _state(plan):
    return {"input": "test", "plan": plan, "past_steps": [], "response": "", "confidence_score": 1.0}


def test_executor_clean_step_is_high_confidence():
    """A stable mock tool agrees with itself, so the posterior lands on a good outcome.

    Not 1.0: confidence is now P(CERTAIN)+P(HIGH) from the posterior on *measured*
    agreement, so even a perfectly consistent step keeps the prior's residual doubt.
    """
    result = executor_node(_state(["summarize data"]))
    assert result["confidence_score"] > 0.6
    assert len(result["plan"]) == 0
    assert len(result["past_steps"]) == 1


def test_stable_tool_beats_flaky_tool_on_confidence():
    """The whole point of one evidence path: disagreement, not keywords, lowers confidence."""
    stable = executor_node(_state(["summarize data"]))["confidence_score"]
    flaky = executor_node(_state(["query conflicting data sources"]))["confidence_score"]
    assert flaky < stable


def test_executor_evidence_defaults_to_self_consistency(monkeypatch):
    """The keyword extractor must not be on the default path."""
    monkeypatch.delenv("EXECUTOR_EVIDENCE", raising=False)
    import core.signals

    def _boom(*_args, **_kwargs):
        raise AssertionError("keyword extractor must not run by default")

    monkeypatch.setattr(core.signals, "extract_evidence", _boom)
    executor_node(_state(["query conflicting data sources"]))


def test_keyword_evidence_is_opt_in(monkeypatch):
    """...but stays reachable for no-sampling deployments when asked for explicitly."""
    monkeypatch.setenv("EXECUTOR_EVIDENCE", "keywords")
    result = executor_node(_state(["query conflicting data sources"]))
    assert 0.0 < result["confidence_score"] < 1.0


def test_executor_conflict_triggers_bayes():
    """An ambiguous observation must run the Bayesian update and lower confidence."""
    result = executor_node(_state(["query conflicting data sources"]))
    assert 0.0 < result["confidence_score"] < 1.0


def test_executor_conflict_confidence_is_deterministic():
    """The new engine is deterministic -- same conflict, same confidence."""
    a = executor_node(_state(["query conflicting data sources"]))["confidence_score"]
    b = executor_node(_state(["query conflicting data sources"]))["confidence_score"]
    assert a == b


def test_executor_empty_plan():
    result = executor_node(_state([]))
    assert result == {"response": "No tasks in plan"}


def test_executor_consumes_plan_sequentially():
    state = _state(["step 1", "step 2", "step 3"])
    r1 = executor_node(state)
    assert len(r1["plan"]) == 2
    state2 = {**state, "plan": r1["plan"], "past_steps": r1["past_steps"]}
    r2 = executor_node(state2)
    assert len(r2["plan"]) == 1


def test_simple_executor_routing():
    assert "search_result" in simple_executor("search the index")
    assert "query_result" in simple_executor("query the db")
    assert "validate_result" in simple_executor("validate inputs")
    assert simple_executor("do thing").startswith("executed")


def test_mock_tool_sampler_is_deterministic_but_divergent():
    """A flaky mock tool must actually return different readings across samples."""
    from nodes.executor import _MockToolSampler

    a = [_MockToolSampler("query the db").invoke("") for _ in range(1)]
    sampler = _MockToolSampler("query the db")
    readings = {sampler.invoke("") for _ in range(4)}
    assert len(readings) > 1                      # genuine disagreement to measure
    assert a[0] == _MockToolSampler("query the db").invoke("")  # reproducible

    stable = _MockToolSampler("summarize data")
    assert len({stable.invoke("") for _ in range(4)}) == 1


def test_replanner_filters_completed_and_duplicate_steps():
    completed = {"step 1", "step 2"}
    steps = [" Step 1 ", "step 3", "step 3", "step 2", "step 4"]

    assert _filter_repeated_steps(steps, completed) == ["step 3", "step 4"]


def test_replanner_falls_back_to_remaining_plan_when_model_repeats_completed(monkeypatch):
    class FakeReplanner:
        def invoke(self, _payload):
            return '{"action": "plan", "steps": ["step 1"]}'

    monkeypatch.setattr("nodes.replanner.create_replanner", lambda _model: FakeReplanner())
    state = {
        "input": "test",
        "plan": ["step 2"],
        "past_steps": [("step 1", "executed: step 1")],
        "response": "",
        "confidence_score": 1.0,
    }

    assert replanner_node(state, {"configurable": {"model": object()}}) == {"plan": ["step 2"]}


def test_graph_routing():
    """should_continue routes correctly. Requires langgraph (skip otherwise)."""
    pytest.importorskip("langgraph")
    from core.graph import should_continue

    assert should_continue({"response": "done", "plan": ["x"]}) == "END"
    assert should_continue({"response": "", "plan": ["x"]}) == "executor"
    assert should_continue({"response": "", "plan": []}) == "END"


def test_run_execution_engine_returns_friendly_error_when_llm_is_down(monkeypatch):
    from core.graph import run_execution_engine

    def _raise_connection_error(*args, **kwargs):
        raise RuntimeError("Connection error")

    monkeypatch.setattr("core.graph.build_llm", _raise_connection_error)

    result = run_execution_engine("what is life?")

    assert result["steps_executed"] == 0
    assert result["confidence_score"] == 0.0
    assert "Local LLM is unavailable" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
