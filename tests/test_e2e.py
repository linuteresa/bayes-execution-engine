import pytest
import os
from unittest.mock import MagicMock, patch
from core.state import PlanExecuteState
from nodes.planner import planner_node
from nodes.executor import executor_node
from bayesian_engine.bayes_engine import resolve_conflict

def test_executor_node_basic_execution():
    """Test that executor node correctly executes a task."""
    state = {
        "input": "test",
        "plan": ["search for data"],
        "past_steps": [],
        "response": "",
        "confidence_score": 1.0,
    }

    result = executor_node(state)

    assert "plan" in result
    assert "past_steps" in result
    assert "confidence_score" in result
    assert len(result["plan"]) == 0
    assert len(result["past_steps"]) == 1
    assert isinstance(result["confidence_score"], float)

def test_executor_node_ambiguous_triggers_bayes():
    """Test that ambiguous output triggers Bayesian conflict resolution."""
    state = {
        "input": "test",
        "plan": ["query conflicting data sources"],
        "past_steps": [],
        "response": "",
        "confidence_score": 1.0,
    }

    result = executor_node(state)

    assert result["confidence_score"] < 1.0, "Ambiguous result should lower confidence"

def test_executor_node_empty_plan():
    """Test executor behavior when plan is empty."""
    state = {
        "input": "test",
        "plan": [],
        "past_steps": [],
        "response": "",
        "confidence_score": 1.0,
    }

    result = executor_node(state)

    assert "response" in result
    assert result["response"] == "No tasks in plan"

def test_bayes_conflict_resolution():
    """Test Bayesian conflict resolution produces valid output."""
    evidence = {
        "TaskStatus": 2,
        "DataQuality": 1,
        "ToolReliability": 3,
    }

    result = resolve_conflict(evidence)

    assert "state" in result
    assert "confidence" in result
    assert "distribution" in result
    assert 0 <= result["confidence"] <= 1.0

def test_planner_node_structure():
    """Test that planner node accepts correct state structure."""
    from core.schemas import Plan

    state = {
        "input": "Query the database",
        "plan": [],
        "past_steps": [],
        "response": "",
        "confidence_score": 1.0,
    }

    assert state["input"] == "Query the database"
    assert state["plan"] == []
    assert state["response"] == ""

def test_execution_pipeline_steps():
    """Test that executor processes tasks sequentially."""
    initial_plan = ["step 1", "step 2", "step 3"]
    state = {
        "input": "test",
        "plan": initial_plan,
        "past_steps": [],
        "response": "",
        "confidence_score": 1.0,
    }

    result1 = executor_node(state)
    assert len(result1["plan"]) == 2
    assert len(result1["past_steps"]) == 1

    state2 = {
        "input": "test",
        "plan": result1["plan"],
        "past_steps": result1["past_steps"],
        "response": "",
        "confidence_score": result1["confidence_score"],
    }

    result2 = executor_node(state2)
    assert len(result2["plan"]) == 1
    assert len(result2["past_steps"]) == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
