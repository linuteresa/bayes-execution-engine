"""Tests for the executor DAG node and graph routing logic."""

import pytest

from nodes.executor import executor_node, simple_executor
from nodes.replanner import _filter_repeated_steps, replanner_node


def _state(plan):
    return {"input": "test", "plan": plan, "past_steps": [], "response": "", "confidence_score": 1.0}


def test_executor_clean_step_keeps_confidence():
    result = executor_node(_state(["summarize data"]))
    assert result["confidence_score"] == 1.0
    assert len(result["plan"]) == 0
    assert len(result["past_steps"]) == 1


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
