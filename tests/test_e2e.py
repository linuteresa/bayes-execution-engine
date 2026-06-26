"""Tests for the executor DAG node and graph routing logic."""

import pytest

from nodes.executor import executor_node, simple_executor


def _state(plan):
    return {"input": "test", "plan": plan, "past_steps": [], "response": "", "confidence_score": 1.0}


def test_executor_clean_step_keeps_confidence():
    result = executor_node(_state(["search for data"]))
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


def test_graph_routing():
    """should_continue routes correctly. Requires langgraph (skip otherwise)."""
    pytest.importorskip("langgraph")
    from core.graph import should_continue

    assert should_continue({"response": "done", "plan": ["x"]}) == "END"
    assert should_continue({"response": "", "plan": ["x"]}) == "executor"
    assert should_continue({"response": "", "plan": []}) == "END"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
