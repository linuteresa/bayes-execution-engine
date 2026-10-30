"""Opt-in real backend tests. Set TEST_REDIS_URL / TEST_POSTGRES_DSN to run."""
from __future__ import annotations

import os
import uuid
from typing import TypedDict

import pytest

from persistence.checkpointer import build_checkpointer
from persistence.job_store import FeedbackConflict, Job, RedisJobStore


def test_live_redis_store_shared_feedback_and_dispatch(monkeypatch):
    url = os.getenv("TEST_REDIS_URL")
    if not url:
        pytest.skip("set TEST_REDIS_URL for real Redis integration")
    pytest.importorskip("redis")
    monkeypatch.setenv("REDIS_JOB_PREFIX", "test-bayes-" + uuid.uuid4().hex)
    first, second = RedisJobStore(url), RedisJobStore(url)
    try:
        first.create(Job(id="one", prompt="question", thread_id="session", run_id="run"))
        job = second.claim_next("worker", 30)
        assert job.run_id == "run" and job.thread_id == "session"
        assert first.claim_next("other", 30) is None
        assert second.finish("one", "worker", status="done", result={"step_evidence": [
            {"TaskStatus": 0, "DataQuality": 0, "ToolReliability": 0}
        ]})
        assert first.record_feedback("one", True) == 1
        with pytest.raises(FeedbackConflict):
            second.record_feedback("one", False)
        assert second.observation_counts()[0, 0] == 1
    finally:
        first._r.delete(first._jobs, first._counts, first._feedback, first._pending, first._terminal)
        first.close()
        second.close()


@pytest.mark.parametrize("backend,env_name,module_name", [
    ("redis", "TEST_REDIS_URL", "langgraph.checkpoint.redis"),
    ("postgres", "TEST_POSTGRES_DSN", "langgraph.checkpoint.postgres"),
])
def test_live_persistent_saver_roundtrip(monkeypatch, backend, env_name, module_name):
    url = os.getenv(env_name)
    if not url:
        pytest.skip(f"set {env_name} for real {backend} integration")
    pytest.importorskip(module_name)
    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        value: int

    workflow = StateGraph(State)
    workflow.add_node("step", lambda state: {"value": state["value"] + 1})
    workflow.add_edge(START, "step")
    workflow.add_edge("step", END)
    monkeypatch.setenv("CHECKPOINTER", backend)
    monkeypatch.setenv("REDIS_URL" if backend == "redis" else "POSTGRES_DSN", url)
    thread_id = "test-bayes-" + uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    with build_checkpointer() as saver:
        try:
            graph = workflow.compile(checkpointer=saver)
            assert graph.invoke({"value": 1}, config=config)["value"] == 2
            assert saver.get_tuple(config).checkpoint["channel_values"]["value"] == 2
        finally:
            saver.delete_thread(thread_id)
