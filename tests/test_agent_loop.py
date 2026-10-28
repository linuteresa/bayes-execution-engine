"""Replanner guardrails, run-level confidence, and the feedback loop that teaches the engine."""

import numpy as np
import pytest

from nodes.replanner import _filter_repeated_steps, replanner_node


class _ModelMustNotRun:
    def invoke(self, _payload):
        raise AssertionError("the replanner model should not have been called")


def _state(past_steps, confidences, plan=()):
    return {
        "input": "test",
        "plan": list(plan),
        "past_steps": list(past_steps),
        "step_confidences": list(confidences),
        "response": "",
        "confidence_score": confidences[-1] if confidences else 1.0,
    }


def _no_model(monkeypatch):
    monkeypatch.setattr("nodes.replanner.create_replanner", lambda _model: _ModelMustNotRun())


# --------------------------------------------------------------- loop guards
def test_step_with_previous_result_tacked_on_is_a_repeat():
    """The live model 'reworded' steps by appending their old output; that is still the same step."""
    completed = {"compile the list"}
    steps = ["Compile the list: | User ID | Tasks |", "compile the list for group b"]
    assert _filter_repeated_steps(steps, completed) == ["compile the list for group b"]


def test_replanner_stops_at_the_step_cap(monkeypatch):
    _no_model(monkeypatch)
    monkeypatch.setenv("MAX_STEPS", "2")
    state = _state([("a", "result a"), ("b", "result b")], [0.9, 0.9], plan=["c"])

    out = replanner_node(state, {"configurable": {"model": object()}})

    assert "plan" not in out
    assert "result a" in out["response"] and "result b" in out["response"]


# ------------------------------------------------------------- forced retry
def test_low_confidence_step_is_rechecked_once(monkeypatch):
    _no_model(monkeypatch)
    state = _state([("a", "x"), ("count users", "maybe 5")], [0.9, 0.3], plan=["next"])

    out = replanner_node(state, {"configurable": {"model": object()}})

    assert out == {"plan": ["Re-check: count users", "next"]}


def test_a_recheck_is_not_rechecked_again(monkeypatch):
    calls = []

    class FakeReplanner:
        def invoke(self, payload):
            calls.append(payload)
            return '{"action": "response", "response": "done"}'

    monkeypatch.setattr("nodes.replanner.create_replanner", lambda _model: FakeReplanner())
    state = _state([("count users", "maybe 5"), ("Re-check: count users", "maybe 7")], [0.3, 0.3])

    assert replanner_node(state, {"configurable": {"model": object()}}) == {"response": "done"}
    assert len(calls) == 1


# ------------------------------------------------------ run-level confidence
def test_overall_confidence_is_the_weakest_step():
    from core.graph import overall_confidence

    assert overall_confidence([("a", ""), ("b", "")], [0.7, 0.3]) == 0.3


def test_overall_confidence_uses_the_recheck_not_the_original():
    from core.graph import overall_confidence

    steps = [("a", ""), ("Re-check: a", ""), ("b", "")]
    assert overall_confidence(steps, [0.3, 0.6, 0.5]) == 0.5


# ------------------------------------------------------------ feedback loop
def test_executor_records_each_steps_evidence():
    from nodes.executor import executor_node

    out = executor_node(_state([], [], plan=["query conflicting data sources"]))
    assert len(out["step_evidence"]) == 1
    assert set(out["step_evidence"][0]) == {"TaskStatus", "DataQuality", "ToolReliability"}


@pytest.fixture
def fresh_engine(monkeypatch, tmp_path):
    """Isolate the shared engine (and its saved state) from other tests."""
    import bayesian_engine.bayes_engine as be

    monkeypatch.setenv("ENGINE_STATE_PATH", str(tmp_path / "engine.npy"))
    monkeypatch.setattr(be, "_DEFAULT_ENGINE", None)
    return be


def _done_job(job_id, evidence):
    from persistence.job_store import Job
    from service import api

    api._store.create(Job(id=job_id, status="done", result={"step_evidence": evidence}))
    return api


def test_wrong_answer_feedback_lowers_confidence_in_that_context(fresh_engine):
    from service.api import FeedbackRequest

    ev = {"TaskStatus": 0, "DataQuality": 0, "ToolReliability": 0}
    before = fresh_engine.good_probability(fresh_engine.resolve_conflict(ev))
    api = _done_job("fb-wrong", [ev, ev])

    api.submit_feedback("fb-wrong", FeedbackRequest(correct=False))

    after = fresh_engine.good_probability(fresh_engine.resolve_conflict(ev))
    assert after < before
    assert fresh_engine.get_default_engine().total_observations() == pytest.approx(2)


def test_feedback_survives_a_restart(fresh_engine, monkeypatch):
    from service.api import FeedbackRequest

    ev = {"TaskStatus": 1, "DataQuality": 1, "ToolReliability": 1}
    api = _done_job("fb-persist", [ev])
    api.submit_feedback("fb-persist", FeedbackRequest(correct=True))
    learned = fresh_engine.get_default_engine().alpha.copy()

    monkeypatch.setattr(fresh_engine, "_DEFAULT_ENGINE", None)  # simulate a new process

    np.testing.assert_allclose(fresh_engine.get_default_engine().alpha, learned)


def test_feedback_is_accepted_once_per_job(fresh_engine):
    from fastapi import HTTPException

    from service.api import FeedbackRequest

    api = _done_job("fb-twice", [{"TaskStatus": 0, "DataQuality": 0, "ToolReliability": 0}])
    api.submit_feedback("fb-twice", FeedbackRequest(correct=True))
    with pytest.raises(HTTPException) as exc:
        api.submit_feedback("fb-twice", FeedbackRequest(correct=True))
    assert exc.value.status_code == 409


def test_feedback_needs_a_finished_job(fresh_engine):
    from fastapi import HTTPException

    from persistence.job_store import Job
    from service import api
    from service.api import FeedbackRequest

    api._store.create(Job(id="fb-running", status="running"))
    with pytest.raises(HTTPException) as exc:
        api.submit_feedback("fb-running", FeedbackRequest(correct=True))
    assert exc.value.status_code == 409
    with pytest.raises(HTTPException) as exc:
        api.submit_feedback("no-such-job", FeedbackRequest(correct=True))
    assert exc.value.status_code == 404


# ---------------------------------------------------- refusals and fallback
def test_a_step_every_sample_refused_is_not_rechecked(monkeypatch):
    """Asking again cannot produce data the model does not have; skip the wasted retry."""
    calls = []

    class FakeReplanner:
        def invoke(self, payload):
            calls.append(payload)
            return '{"action": "response", "response": "no data available"}'

    monkeypatch.setattr("nodes.replanner.create_replanner", lambda _model: FakeReplanner())
    state = _state([("list users", "I do not have access")], [0.3])
    state["step_evidence"] = [{"TaskStatus": 4, "DataQuality": 0, "ToolReliability": 0}]

    assert replanner_node(state, {"configurable": {"model": object()}}) == {
        "response": "no data available"
    }


def test_fallback_answer_has_no_repeats_and_prefers_the_recheck():
    from nodes.replanner import _fallback_summary

    steps = [
        ("count users", "maybe 5"),
        ("Re-check: count users", "There are 5 users."),
        ("list tasks", "No access."),
        ("list owners", "No access."),
    ]
    assert _fallback_summary(steps) == "There are 5 users. No access."
