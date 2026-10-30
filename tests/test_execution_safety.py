"""Adversarial graph, dependency, deadline and evidence regressions from review."""

import json
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver

from core.graph import checkpoint_thread_id, run_execution_engine
from core.llm import build_llm
from core.numbers import numeric_claims, numeric_values
from core.runtime import ModelOutputError
from nodes.executor import executor_node
from nodes.llm_executor import dependency_context, execute_step_with_llm
from nodes.replanner import replanner_node


@pytest.fixture(autouse=True)
def isolated_engine(monkeypatch):
    import bayesian_engine.bayes_engine as be
    monkeypatch.setattr(be, "_COUNT_SOURCE", None)
    monkeypatch.setattr(be, "_DEFAULT_ENGINE", None)
    monkeypatch.delenv("ENGINE_STATE_PATH", raising=False)
    monkeypatch.delenv("EMBEDDINGS_BASE_URL", raising=False)


def install_model(monkeypatch, fn):
    monkeypatch.setattr("core.graph.build_llm", lambda **kw: RunnableLambda(
        lambda prompt: AIMessage(content=fn(prompt if isinstance(prompt, str) else prompt.to_string()))
    ))


@pytest.mark.parametrize("output", ["not json", '{"steps":"ABC"}', '{"steps":null}',
                                    '{"steps":[]}', '{"steps":[42]}', '{"steps":[" "]}'])
def test_invalid_planner_cannot_execute_or_claim_confidence(monkeypatch, output):
    calls = []
    install_model(monkeypatch, lambda prompt: calls.append(prompt) or output)
    result = run_execution_engine("an objective")
    assert result["error_code"] == "invalid_model_output"
    assert result["steps_executed"] == 0
    assert result["confidence_score"] == 0
    assert len(calls) == 2  # repair is bounded; executor was never reached


def test_planner_can_repair_invalid_output_once(monkeypatch):
    planning = iter(['{"steps":"bad"}', '{"steps":["derive value"]}'])
    def respond(prompt):
        if "orchestrator architecture planner" in prompt:
            return next(planning)
        if "task replanner" in prompt:
            return '{"action":"response","response":"37"}'
        return "37"
    install_model(monkeypatch, respond)
    result = run_execution_engine("derive a value")
    assert result["steps_executed"] == 1
    assert "error" not in result


@pytest.mark.parametrize("output", ['{"action":"unknown","steps":[]}',
                                    '{"action":"plan","steps":null}',
                                    '{"action":"response","response":42}',
                                    '{"action":"response","response":" "}'])
def test_replanner_rejects_wrong_decision_types(monkeypatch, output):
    monkeypatch.setattr("nodes.replanner.create_replanner", lambda _: Mock(invoke=lambda _: output))
    with pytest.raises(ModelOutputError):
        replanner_node({"input":"x", "plan":[], "past_steps":[("step", "answer")],
                        "step_confidences":[.8]}, {"configurable":{}})


def test_second_step_consumes_result_it_cannot_derive_from_the_goal(monkeypatch):
    seen = []
    def respond(prompt):
        if "orchestrator architecture planner" in prompt:
            return '{"steps":["derive value","double prior result"]}'
        if "task replanner" in prompt:
            return '{"action":"plan","steps":[]}'
        seen.append(prompt)
        if "Step to execute: derive value" in prompt:
            return "Intermediate result: 37"
        return "Doubled result: 74" if '"result": "Intermediate result: 37"' in prompt else "No prior result"
    install_model(monkeypatch, respond)
    result = run_execution_engine("Perform the two dependent operations")
    assert result["past_steps"][1][1] == "Doubled result: 74"
    assert all('"step_id":' in p for p in seen[4:])


def test_recheck_receives_prior_attempt_and_reason():
    seen = []
    model = Mock(invoke=lambda p: seen.append(p) or "Verified 37")
    executor_node({"input":"verify", "plan":["Re-check: derive value"],
                   "past_steps":[("derive value", "Uncertain value: 37")],
                   "step_confidences":[.3]}, {"configurable":{"executor_model":model}})
    assert "Uncertain value: 37" in seen[0]
    assert '"recheck_reason"' in seen[0]


def test_dependency_context_is_bounded_and_retains_recent_result(monkeypatch):
    monkeypatch.setenv("EXECUTOR_CONTEXT_CHARS", "800")
    context = dependency_context([("older", "a" * 20000), ("newest", "LATEST_VALUE")], [.8,.7], "use it")
    assert "LATEST_VALUE" in context
    assert len(context) <= 800
    assert json.loads(context)["results"][-1]["task"] == "newest"


def stable_model(prompt):
    if "orchestrator architecture planner" in prompt:
        return '{"steps":["derive value"]}'
    if "task replanner" in prompt:
        return '{"action":"response","response":"done"}'
    return "A stable answer"


def test_same_session_new_questions_get_isolated_runs(monkeypatch):
    install_model(monkeypatch, stable_model)
    saver = MemorySaver()
    first = run_execution_engine("first", checkpointer=saver, thread_id="session")
    second = run_execution_engine("second", checkpointer=saver, thread_id="session")
    assert first["steps_executed"] == second["steps_executed"] == 1
    assert first["run_id"] != second["run_id"]


def test_completed_run_resume_does_not_reexecute_and_duplicate_start_is_rejected(monkeypatch):
    calls = []
    install_model(monkeypatch, lambda p: calls.append(p) or stable_model(p))
    saver = MemorySaver()
    first = run_execution_engine("goal", checkpointer=saver, thread_id="session", run_id="run")
    count = len(calls)
    resumed = run_execution_engine("goal", checkpointer=saver, thread_id="session", run_id="run", resume=True)
    assert resumed == first
    assert len(calls) == count
    duplicate = run_execution_engine("goal", checkpointer=saver, thread_id="session", run_id="run")
    assert duplicate["error_code"] == "invalid_request"


def test_resume_after_replanner_crash_keeps_completed_executor_result(monkeypatch):
    calls = []
    crashed = False
    def respond(prompt):
        nonlocal crashed
        calls.append(prompt)
        if "task replanner" in prompt and not crashed:
            crashed = True
            raise RuntimeError("worker terminated")
        return stable_model(prompt)
    install_model(monkeypatch, respond)
    saver = MemorySaver()
    with pytest.raises(RuntimeError, match="worker terminated"):
        run_execution_engine("goal", checkpointer=saver, thread_id="s", run_id="r")
    result = run_execution_engine("goal", checkpointer=saver, thread_id="s", run_id="r", resume=True)
    assert result["steps_executed"] == 1
    assert sum("Step to execute:" in p for p in calls) == 4


def test_resume_without_checkpoint_restarts_fresh(monkeypatch):
    install_model(monkeypatch, stable_model)
    assert run_execution_engine("goal", checkpointer=MemorySaver(), run_id="r", resume=True)["steps_executed"] == 1


def test_cancelled_inference_cannot_commit_its_node_output(monkeypatch):
    cancelled = threading.Event()
    def respond(prompt):
        if "Step to execute:" in prompt:
            cancelled.set()
        return stable_model(prompt)
    install_model(monkeypatch, respond)
    saver = MemorySaver()
    result = run_execution_engine("goal", checkpointer=saver, thread_id="s", run_id="r", cancel_event=cancelled)
    assert result["error_code"] == "execution_cancelled"
    state = saver.get_tuple({"configurable":{"thread_id":checkpoint_thread_id("s", "r")}})
    assert state.checkpoint["channel_values"].get("past_steps", []) == []


def test_expired_run_does_not_start_a_model(monkeypatch):
    build = Mock()
    monkeypatch.setattr("core.graph.build_llm", build)
    result = run_execution_engine("goal", deadline=time.time() - 1)
    assert result["error_code"] == "execution_timeout"
    build.assert_not_called()


@pytest.mark.parametrize("answer", ["Paris.", "42", "Yes.", "東京", "sí"])
def test_short_and_unicode_answers_are_answerable(answer):
    result = execute_step_with_llm(Mock(invoke=lambda _:answer), "question", n_samples=4)
    assert result.evidence["TaskStatus"] == 0


@pytest.mark.parametrize("a,b", [("42", "-42"), ("1.5", "5.1"), ("10 USD", "10 EUR")])
def test_numeric_contradictions_reduce_agreement_and_remain_distinct(a,b):
    answers = iter([f"The result is {a}.", f"The result is {b}."] * 2)
    result = execute_step_with_llm(Mock(invoke=lambda _:next(answers)), "question", n_samples=4)
    assert result.consistency < .5
    assert result.evidence["ToolReliability"] > 0


@pytest.mark.parametrize("a,b", [("42", "42.0"), ("1e3", "1000"), ("3/4", "0.75"),
                                ("12.5%", "0.125"), ("100 cm", "1 m")])
def test_equivalent_numeric_literals_compare_exactly(a,b):
    assert numeric_claims(a) == numeric_claims(b)


def test_numeric_multiplicity_is_retained():
    assert numeric_values("3 and 3") != numeric_values("3")


def test_llm_client_has_explicit_request_generation_and_retry_budgets():
    model = build_llm()
    assert model.root_client.timeout == 60
    assert model.max_tokens == 2048
    assert model.max_retries == 0


def test_nonfinite_embeddings_are_rejected(monkeypatch):
    monkeypatch.setattr("nodes.llm_executor.build_embeddings", lambda **_kwargs: Mock(
        embed_documents=lambda _: [[float("nan"), 0]] * 2))
    with pytest.raises(ValueError, match="invalid vectors"):
        execute_step_with_llm(Mock(invoke=lambda _:"answer"), "question", n_samples=2)


@pytest.mark.parametrize("n", [0, -1, True, 65, 1.5])
def test_invalid_sample_budget_is_rejected(n):
    with pytest.raises(ValueError, match="n_samples"):
        execute_step_with_llm(Mock(), "question", n_samples=n)


def test_step_confidence_aggregate_rejects_unaligned_or_invalid_inputs():
    from core.graph import overall_confidence
    with pytest.raises(ValueError):
        overall_confidence([("step", "answer")], [])
    with pytest.raises(ValueError):
        overall_confidence([("step", "answer")], [np.nan])
