"""Attempt isolation prevents late checkpoint writes from changing recovery state."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from core.runtime import ExecutionCancelled
from persistence.fenced_checkpointer import FencedCheckpointer, checkpoint_reference
from persistence.job_store import InMemoryJobStore, Job


class _State(TypedDict):
    value: str
    complete: bool


def _graph(saver, measure=None, *, interrupt_after=None):
    builder = StateGraph(_State)
    builder.add_node("measure", measure or (lambda _: {"value": "committed"}))
    builder.add_node("finish", lambda _: {"complete": True})
    builder.add_edge(START, "measure")
    builder.add_edge("measure", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=saver, interrupt_after=interrupt_after)


def _config(thread):
    return {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}


def test_recovery_clones_exact_full_checkpoint_and_ignores_pending_and_newer_writes():
    delegate = MemorySaver()
    references = []
    first = FencedCheckpointer(delegate, lambda ref: references.append(ref) or True)
    _graph(first, interrupt_after=["measure"]).invoke(
        {"value": "", "complete": False}, _config("old-attempt"),
    )
    committed = deepcopy(references[-1])
    source = delegate.get_tuple(committed)
    delegate.put_writes(committed, [("value", "uncommitted")], "pending-task")
    # A late old worker can write a newer physical checkpoint. It must not alter
    # which exact checkpoint the job store authorized for recovery.
    stale = deepcopy(source.checkpoint)
    stale["id"] = "zzzz-newer-unpublished"
    stale["channel_values"]["value"] = "stale"
    version = delegate.get_next_version(stale["channel_versions"]["value"], None)
    stale["channel_versions"]["value"] = version
    delegate.put(committed, stale, source.metadata, {"value": version})
    assert delegate.get_tuple(_config("old-attempt")).checkpoint["id"] == stale["id"]

    recovered_refs = []
    second = FencedCheckpointer(delegate, lambda ref: recovered_refs.append(ref) or True)
    restored = second.restore_from(committed, _config("new-attempt"))
    restored_tuple = delegate.get_tuple(restored)
    assert restored_tuple.checkpoint["channel_values"]["value"] == "committed"
    assert not restored_tuple.pending_writes
    assert recovered_refs[-1]["configurable"]["thread_id"] == "new-attempt"
    result = _graph(second).invoke(None, restored)
    assert result == {"value": "committed", "complete": True}


def test_late_worker_cannot_replace_published_recovery_reference():
    store = InMemoryJobStore()
    delegate = MemorySaver()
    store.create(Job(id="job"))
    first_job = store.claim_next("old-owner", 30)
    paused, release = threading.Event(), threading.Event()

    def stale_measure(_state):
        paused.set()
        assert release.wait(5)
        return {"value": "stale"}

    first = FencedCheckpointer(
        delegate, lambda ref: store.record_checkpoint("job", "old-owner", ref),
    )
    with ThreadPoolExecutor(1) as pool:
        running = pool.submit(
            _graph(first, stale_measure).invoke, {"value": "", "complete": False},
            _config(first_job.attempt_id),
        )
        assert paused.wait(5)
        # Simulate loss of the process between node computation and saver.put.
        store.update("job", lease_expires=0)
        recovered_job = store.claim_next("new-owner", 30)
        second = FencedCheckpointer(
            delegate, lambda ref: store.record_checkpoint("job", "new-owner", ref),
        )
        restored = second.restore_from(recovered_job.checkpoint_ref, _config(recovered_job.attempt_id))
        result = _graph(second, lambda _: {"value": "fresh"}).invoke(None, restored)
        assert result == {"value": "fresh", "complete": True}
        authorized = deepcopy(store.get("job").checkpoint_ref)
        release.set()
        with pytest.raises(ExecutionCancelled):
            running.result(timeout=5)
    assert store.get("job").checkpoint_ref == authorized
    assert delegate.get_tuple(authorized).checkpoint["channel_values"]["value"] == "fresh"
    # Recovery reads the authorized full checkpoint, not the old thread's latest.
    third = FencedCheckpointer(delegate, lambda _: True)
    final_ref = third.restore_from(authorized, _config("third-attempt"))
    assert _graph(third).invoke(None, final_ref)["value"] == "fresh"
    store.close()


def test_missing_exact_recovery_checkpoint_fails_instead_of_using_latest():
    delegate = MemorySaver()
    first = FencedCheckpointer(delegate, lambda _: True)
    _graph(first).invoke({"value": "", "complete": False}, _config("old"))
    missing = {"configurable": {"thread_id": "old", "checkpoint_ns": "", "checkpoint_id": "missing"}}
    with pytest.raises(ValueError, match="unavailable"):
        FencedCheckpointer(delegate, lambda _: True).restore_from(missing, _config("new"))


def test_published_checkpoint_id_cannot_be_reused_for_different_state():
    delegate = MemorySaver()
    references = []
    saver = FencedCheckpointer(delegate, lambda ref: references.append(ref) or True)
    _graph(saver).invoke({"value": "", "complete": False}, _config("attempt"))
    saved = delegate.get_tuple(references[-1])
    changed = deepcopy(saved.checkpoint)
    changed["channel_values"]["value"] = "different"
    with pytest.raises(ValueError, match="different state"):
        saver.put(saved.config, changed, saved.metadata, changed["channel_versions"])
    assert delegate.get_tuple(references[-1]).checkpoint["channel_values"]["value"] == "committed"


def test_reference_contains_only_exact_saver_address():
    reference = checkpoint_reference({"configurable": {
        "thread_id": "attempt", "checkpoint_id": "id", "model": object(), "api_key": "secret",
    }})
    assert reference == {"configurable": {
        "thread_id": "attempt", "checkpoint_ns": "", "checkpoint_id": "id",
    }}
    with pytest.raises(ValueError):
        checkpoint_reference(_config("attempt"))


def test_recovery_requires_a_different_physical_attempt_thread():
    reference = {"configurable": {"thread_id": "same", "checkpoint_id": "id"}}
    with pytest.raises(ValueError, match="new attempt"):
        FencedCheckpointer(MemorySaver(), lambda _: True).restore_from(reference, _config("same"))


@pytest.mark.parametrize("elapsed_budget", [False, True])
def test_execution_engine_restores_completed_authorized_attempt_without_rebuilding_models(monkeypatch, elapsed_budget):
    import core.graph as graph

    builds = []
    planning = []
    monkeypatch.setattr(graph, "build_llm", lambda **_: builds.append(True) or object())

    def planner(_state, _config):
        planning.append(True)
        return {"plan": ["measure"]}

    monkeypatch.setattr(graph, "planner_node", planner)
    monkeypatch.setattr(graph, "executor_node", lambda _state, _config: {
        "plan": [], "past_steps": [("measure", "measured")], "step_confidences": [0.8],
        "step_evidence": [{"TaskStatus": 0, "DataQuality": 0, "ToolReliability": 0}],
    })
    monkeypatch.setattr(graph, "replanner_node", lambda _state, _config: {"response": "done"})
    store = InMemoryJobStore()
    saver = MemorySaver()
    store.create(Job(id="engine", prompt="goal", thread_id="session", run_id="run"))
    first = store.claim_next("first", 30)
    result = graph.run_execution_engine(
        "goal", checkpointer=saver, thread_id="session", run_id="run", attempt_id=first.attempt_id,
        checkpoint_callback=lambda ref: store.record_checkpoint("engine", "first", ref),
    )
    assert result["response"] == "done"
    published = store.get("engine").checkpoint_ref
    assert published is not None
    store.update("engine", lease_expires=0,
                 deadline_at=time.time() - 1 if elapsed_budget else time.time() + 30)
    second = store.claim_next("second", 30)
    resumed = graph.run_execution_engine(
        "goal", checkpointer=saver, thread_id="session", run_id="run", resume=True,
        attempt_id=second.attempt_id, checkpoint_ref=second.checkpoint_ref,
        deadline=second.deadline_at,
        checkpoint_callback=lambda ref: store.record_checkpoint("engine", "second", ref),
    )
    assert resumed == result
    assert len(builds) == 2 and len(planning) == 1
    assert store.get("engine").checkpoint_ref != published
    assert graph.checkpoint_thread_id("session", "run") != graph.checkpoint_thread_id(
        "session", "run", first.attempt_id,
    )
    store.close()


def test_incomplete_authorized_recovery_still_requires_time_before_building_models(monkeypatch):
    import core.graph as graph

    builds = []
    monkeypatch.setattr(graph, "build_llm", lambda **_: builds.append(True) or object())
    monkeypatch.setattr(graph, "planner_node", lambda _state, _config: {"plan": ["measure"]})

    def interrupted(_state, _config):
        raise RuntimeError("worker crashed before its answer")

    monkeypatch.setattr(graph, "executor_node", interrupted)
    store = InMemoryJobStore()
    saver = MemorySaver()
    store.create(Job(id="incomplete", prompt="goal"))
    first = store.claim_next("first", 30)
    with pytest.raises(RuntimeError, match="worker crashed"):
        graph.run_execution_engine(
            "goal", checkpointer=saver, run_id="run", attempt_id=first.attempt_id,
            checkpoint_callback=lambda ref: store.record_checkpoint("incomplete", "first", ref),
        )
    store.update("incomplete", lease_expires=0, deadline_at=time.time() - 1)
    second = store.claim_next("second", 30)
    result = graph.run_execution_engine(
        "goal", checkpointer=saver, run_id="run", resume=True, attempt_id=second.attempt_id,
        checkpoint_ref=second.checkpoint_ref, deadline=second.deadline_at,
        checkpoint_callback=lambda ref: store.record_checkpoint("incomplete", "second", ref),
    )
    assert result["error_code"] == "execution_timeout"
    assert len(builds) == 2
    store.close()


def test_execution_engine_does_not_restart_when_an_exact_persistent_reference_is_missing():
    from core.graph import run_execution_engine

    result = run_execution_engine(
        "goal", checkpointer=MemorySaver(), run_id="run", resume=True, attempt_id="new",
        checkpoint_ref={"configurable": {"thread_id": "old", "checkpoint_id": "missing"}},
        checkpoint_callback=lambda _: True,
    )
    assert result["error_code"] == "invalid_request"
    assert "unavailable" in result["error"]
