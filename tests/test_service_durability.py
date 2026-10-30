"""Failure, concurrency, recovery, and HTTP regressions for the execution service."""
from __future__ import annotations

import contextlib
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from persistence.checkpointer import build_checkpointer
from persistence.job_store import (
    FeedbackConflict,
    InMemoryJobStore,
    Job,
    QueueFull,
    SQLiteJobStore,
)

EV = {"TaskStatus": 1, "DataQuality": 2, "ToolReliability": 3}


def _wait_for(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("worker did not reach expected state")


def test_feedback_commits_once_across_connections_and_survives_restart(tmp_path):
    path = str(tmp_path / "shared.sqlite3")
    stores = [SQLiteJobStore(path), SQLiteJobStore(path)]
    stores[0].create(Job(id="one", status="done", result={"step_evidence": [EV, EV]}))
    barrier = threading.Barrier(2)

    def feedback(store):
        barrier.wait()
        try:
            return store.record_feedback("one", True)
        except FeedbackConflict:
            return "conflict"

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(feedback, stores))
    assert sorted(results, key=str) == [2, "conflict"]
    for store in stores:
        store.close()
    reopened = SQLiteJobStore(path)
    assert reopened.get("one").feedback is True
    assert reopened.observation_counts()[38, 0] == 2
    assert reopened.observation_counts().sum() == 2
    reopened.close()


def test_feedback_failure_rolls_back_verdict_event_and_all_counts(tmp_path, monkeypatch):
    store = SQLiteJobStore(str(tmp_path / "atomic.sqlite3"))
    store.create(Job(id="rollback", status="done", result={"step_evidence": [EV]}))
    original = store._write

    def broken_write(*args):
        raise OSError("injected failure after counts were written")

    monkeypatch.setattr(store, "_write", broken_write)
    with pytest.raises(OSError):
        store.record_feedback("rollback", False)
    assert store.get("rollback").feedback is None
    assert store.observation_counts().sum() == 0
    monkeypatch.setattr(store, "_write", original)
    assert store.record_feedback("rollback", False) == 1
    assert store.observation_counts()[38, 4] == 1
    store.close()


def test_concurrent_admission_does_not_exceed_capacity(tmp_path):
    path = str(tmp_path / "capacity.sqlite3")
    stores = [SQLiteJobStore(path), SQLiteJobStore(path)]
    barrier = threading.Barrier(2)

    def submit(pair):
        index, store = pair
        barrier.wait()
        try:
            store.create(Job(id=str(index)), max_pending=1)
            return True
        except QueueFull:
            return False

    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(submit, enumerate(stores))) == [False, True]
    for store in stores:
        store.close()


def test_expired_lease_recovered_once_and_old_owner_cannot_publish(tmp_path, monkeypatch):
    import persistence.job_store as js
    now = [1000.0]
    monkeypatch.setattr(js.time, "time", lambda: now[0])
    path = str(tmp_path / "lease.sqlite3")
    first, second = SQLiteJobStore(path), SQLiteJobStore(path)
    first.create(Job(id="recover", prompt="question", thread_id="session", run_id="run"))
    claimed = first.claim_next("old-worker", 30)
    assert claimed.attempts == 1
    assert second.claim_next("new-worker", 30) is None
    now[0] += 31
    recovered = second.claim_next("new-worker", 30)
    assert (recovered.prompt, recovered.thread_id, recovered.run_id, recovered.attempts) == (
        "question", "session", "run", 2,
    )
    assert not first.finish("recover", "old-worker", status="done", result={"response": "stale"})
    assert not first.renew("recover", "old-worker", 30)
    assert second.finish("recover", "new-worker", status="done", result={"response": "fresh"})
    assert first.get("recover").result["response"] == "fresh"
    first.close()
    second.close()


def test_retention_removes_results_but_keeps_learned_counts():
    store = InMemoryJobStore()
    store.create(Job(id="old", status="done", result={"step_evidence": [EV]}))
    store.record_feedback("old", True)
    store.update("old", updated_at=1)  # update sets its own time; set time through SQL below
    store._db.execute("UPDATE jobs SET updated_at=1 WHERE id='old'")
    store.create(Job(id="pending"))
    assert store.prune(60) == 1
    assert store.get("old") is None
    assert store.get("pending") is not None
    assert store.observation_counts().sum() == 1
    store.close()


@pytest.mark.parametrize("backend,module_name,class_name", [
    ("postgres", "langgraph.checkpoint.postgres", "PostgresSaver"),
    ("redis", "langgraph.checkpoint.redis", "RedisSaver"),
])
def test_persistent_saver_context_is_entered_setup_and_closed(monkeypatch, backend, module_name, class_name):
    events = []
    saver = types.SimpleNamespace(setup=lambda: events.append("setup"))

    class Factory:
        @classmethod
        @contextlib.contextmanager
        def from_conn_string(cls, _url):
            events.append("open")
            try:
                yield saver
            finally:
                events.append("close")

    monkeypatch.setenv("CHECKPOINTER", backend)
    monkeypatch.setitem(sys.modules, module_name, types.SimpleNamespace(**{class_name: Factory}))
    with build_checkpointer() as actual:
        assert actual is saver
        assert events == ["open", "setup"]
    assert events == ["open", "setup", "close"]


def test_requested_missing_saver_is_a_startup_error(monkeypatch):
    monkeypatch.setenv("CHECKPOINTER", "postgres")
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", None)
    with pytest.raises(ImportError), build_checkpointer():
        pass


def test_dispatcher_recovers_expired_job_with_persisted_identity(tmp_path, monkeypatch):
    from service import api
    store = SQLiteJobStore(str(tmp_path / "recover.sqlite3"))
    store.create(Job(id="restart", prompt="recover me", thread_id="session", run_id="unique-run",
                     deadline_at=time.time() + 30))
    store.claim_next("dead-process", 30)
    store.update("restart", lease_expires=0)
    calls = []

    def fake_run(question, **kwargs):
        calls.append((question, kwargs))
        return {"response": "recovered", "confidence_score": 0.7}

    monkeypatch.setattr(api, "run_execution_engine", fake_run)
    dispatcher = api.Dispatcher(store, None, workers=1, poll_seconds=0.01)
    dispatcher.start()
    try:
        _wait_for(lambda: store.get("restart").status == "done")
    finally:
        dispatcher.stop()
    question, config = calls[0]
    assert question == "recover me"
    assert config["run_id"] == "unique-run" and config["thread_id"] == "session"
    assert config["resume"] is True
    store.close()


def test_heartbeat_keeps_active_job_from_being_reclaimed(tmp_path, monkeypatch):
    from service import api
    path = str(tmp_path / "heartbeat.sqlite3")
    store, observer = SQLiteJobStore(path), SQLiteJobStore(path)
    store.create(Job(id="slow"))
    started, release = threading.Event(), threading.Event()

    def fake_run(*args, **kwargs):
        started.set()
        release.wait(3)
        return {"response": "ok"}

    monkeypatch.setattr(api, "run_execution_engine", fake_run)
    dispatcher = api.Dispatcher(store, None, workers=1, lease_seconds=0.09, poll_seconds=0.01)
    dispatcher.start()
    try:
        assert started.wait(1)
        time.sleep(0.2)
        assert observer.claim_next("other", 30) is None
    finally:
        release.set()
        dispatcher.stop()
        observer.close()
        store.close()


def test_http_validates_prompt_reports_capacity_and_processes_jobs(tmp_path, monkeypatch):
    from service import api
    monkeypatch.setenv("JOB_STORE", "sqlite")
    monkeypatch.setenv("JOB_DB_PATH", str(tmp_path / "http.sqlite3"))
    monkeypatch.setenv("CHECKPOINTER", "none")
    monkeypatch.setenv("JOB_WORKERS", "1")
    monkeypatch.setenv("JOB_MAX_PENDING", "1")
    started, release = threading.Event(), threading.Event()

    def fake_run(*args, **kwargs):
        started.set()
        release.wait(3)
        return {"response": "ok", "step_evidence": [EV]}

    monkeypatch.setattr(api, "run_execution_engine", fake_run)
    with TestClient(api.app) as client:
        assert client.post("/jobs", json={"question": "   "}).status_code == 422
        assert client.post("/jobs", json={"question": "x" * 16001}).status_code == 422
        accepted = client.post("/jobs", json={"question": "first", "thread_id": "session"})
        assert accepted.status_code == 202
        job_id = accepted.json()["job_id"]
        assert started.wait(1)
        assert client.post("/jobs", json={"question": "second"}).status_code == 429
        release.set()
        _wait_for(lambda: client.get(f"/jobs/{job_id}").json()["status"] == "done")
        assert client.post(f"/jobs/{job_id}/feedback", json={"correct": True}).status_code == 200
        assert client.post(f"/jobs/{job_id}/feedback", json={"correct": False}).status_code == 409
        assert np.sum(api._store.observation_counts()) == 1


@pytest.mark.parametrize("body", [None, ["bad"], {"question": 7}, {"question": "   "}, {"question": "x" * 16001}])
def test_flask_rejects_invalid_requests(body):
    from app import app
    with app.test_client() as client:
        assert client.post("/ask", json=body).status_code == 400


def test_flask_model_unavailability_is_an_http_error(monkeypatch):
    import app as ui
    monkeypatch.setattr(ui, "run_execution_engine", lambda _: {"error": "Local LLM unavailable"})
    with ui.app.test_client() as client:
        response = client.post("/ask", json={"question": "hello"})
        assert response.status_code == 503
        assert response.json["error"] == "Local LLM unavailable"


def test_ui_renders_model_markup_as_text_and_handles_error_results():
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the browser handler without external dependencies")
    script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
    constructor() {
        this.value = 'question'; this.style = {}; this.children = [];
        this.classes = new Set();
        this.classList = {add: c => this.classes.add(c), remove: c => this.classes.delete(c)};
    }
    addEventListener(_type, callback) { this.listener = callback; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    set innerHTML(_value) { throw new Error('Untrusted content passed to an HTML sink'); }
}
const elements = {};
const document = {
    getElementById: id => elements[id] ??= new Element(),
    createElement: () => new Element()
};
const payload = '<img src=x onerror="window.compromised=true">';
let data = {response:'answer', steps_executed:1, confidence_score:.5, past_steps:[[payload,payload]]};
const context = vm.createContext({document, fetch:async()=>({ok:true,json:async()=>data})});
const code = fs.readFileSync(process.argv[2], 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
vm.runInContext(code, context);
(async () => {
    await elements.queryForm.listener({preventDefault(){}});
    const row = elements.stepsList.children[0];
    assert.equal(row.children[1].textContent, payload);
    assert.equal(row.children[3].textContent, 'Result: '+payload);
    assert(elements.result.classes.has('show'));
    data = {error:'Model unavailable'};
    await elements.queryForm.listener({preventDefault(){}});
    assert.equal(elements.error.textContent, 'Error: Model unavailable');
    assert(elements.error.classes.has('show'));
    assert(!elements.result.classes.has('show'));
})().catch(error => { console.error(error); process.exitCode=1; });
'''
    subprocess.run(
        [node, "-", str(Path(__file__).resolve().parents[1] / "templates" / "index.html")],
        input=script, text=True, check=True, capture_output=True, timeout=10,
    )


def test_checkpoint_retention_cleans_saver_before_deleting_job():
    from core.graph import checkpoint_thread_id
    from service.api import Dispatcher

    store = InMemoryJobStore()
    job = Job(id="expire", thread_id="session", run_id="run", status="done")
    store.create(job)
    store._db.execute("UPDATE jobs SET updated_at=1 WHERE id='expire'")
    deleted = []
    saver = types.SimpleNamespace(delete_thread=lambda key: deleted.append(key))
    dispatcher = Dispatcher(store, saver, retention_seconds=60)
    dispatcher.prune()
    assert deleted == [checkpoint_thread_id("session", "run")]
    assert store.get("expire") is None
    store.close()


def test_lost_lease_cancels_runtime_and_cannot_publish_stale_result(monkeypatch):
    from service import api

    store = InMemoryJobStore()
    store.create(Job(id="lost"))
    job = store.claim_next("owner", 1)
    monkeypatch.setattr(store, "renew", lambda *args: False)
    seen = []

    def fake_run(*args, **kwargs):
        assert kwargs["cancel_event"].wait(1)
        seen.append("cancelled")
        return {"response": "stale"}

    monkeypatch.setattr(api, "run_execution_engine", fake_run)
    dispatcher = api.Dispatcher(store, None, workers=1, lease_seconds=0.03)
    dispatcher._run_job(job, "owner")
    assert seen == ["cancelled"]
    assert store.get("lost").status == "running"
    assert store.get("lost").result is None
    store.close()


def test_api_key_protects_jobs_and_feedback_but_not_health(tmp_path, monkeypatch):
    from service import api

    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("JOB_STORE", "sqlite")
    monkeypatch.setenv("JOB_DB_PATH", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("CHECKPOINTER", "none")
    monkeypatch.setenv("JOB_WORKERS", "1")
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/jobs", json={"question": "hello"}).status_code == 401
        assert client.get("/jobs/unknown").status_code == 401
        assert client.post("/jobs/unknown/feedback", json={"correct": True}).status_code == 401
        response = client.get("/jobs/unknown", headers={"Authorization": "Bearer test-secret"})
        assert response.status_code == 404


def test_expired_worker_cannot_renew_or_finish_even_before_takeover(monkeypatch):
    import persistence.job_store as js

    now = [1000.0]
    monkeypatch.setattr(js.time, "time", lambda: now[0])
    store = InMemoryJobStore()
    store.create(Job(id="expired"))
    store.claim_next("old", 10)
    now[0] = 1011.0
    assert not store.renew("expired", "old", 10)
    assert not store.finish("expired", "old", status="done")
    assert store.get("expired").lease_expires == 1010.0
    store.close()


def test_checkpoint_reference_only_advances_under_live_lease(tmp_path, monkeypatch):
    import persistence.job_store as js

    now = [1000.0]
    monkeypatch.setattr(js.time, "time", lambda: now[0])
    store = SQLiteJobStore(str(tmp_path / "checkpoint-fence.sqlite3"))
    store.create(Job(id="fenced"))
    first = store.claim_next("old", 10)
    first_ref = {"configurable": {"thread_id": "old-attempt", "checkpoint_ns": "", "checkpoint_id": "1"}}
    assert store.record_checkpoint("fenced", "old", first_ref)
    now[0] += 11
    stale_ref = {"configurable": {"thread_id": "old-attempt", "checkpoint_ns": "", "checkpoint_id": "2"}}
    assert not store.record_checkpoint("fenced", "old", stale_ref)
    second = store.claim_next("new", 10)
    assert second.checkpoint_ref == first_ref
    assert second.attempt_id != first.attempt_id
    assert second.attempt_ids == [first.attempt_id, second.attempt_id]
    assert not store.record_checkpoint("fenced", "old", stale_ref)
    new_ref = {"configurable": {"thread_id": "new-attempt", "checkpoint_ns": "", "checkpoint_id": "3"}}
    assert store.record_checkpoint("fenced", "new", new_ref)
    assert store.get("fenced").checkpoint_ref == new_ref
    store.close()


def test_dispatcher_uses_a_fresh_ownership_token_for_every_claim(monkeypatch):
    from service import api

    store = InMemoryJobStore()
    store.create(Job(id="first"))
    store.create(Job(id="second"))
    tokens = []
    original_claim = store.claim_next

    def claim(owner, seconds):
        tokens.append(owner)
        return original_claim(owner, seconds)

    monkeypatch.setattr(store, "claim_next", claim)
    monkeypatch.setattr(api, "run_execution_engine", lambda *args, **kwargs: {"response": "ok"})
    dispatcher = api.Dispatcher(store, None, workers=1, poll_seconds=0.01)
    dispatcher.start()
    try:
        _wait_for(lambda: store.get("second").status == "done")
    finally:
        dispatcher.stop()
    assert len(tokens) >= 2
    assert len(tokens) == len(set(tokens))
    store.close()


def test_dispatcher_real_graph_publishes_and_recovers_completed_checkpoint(tmp_path, monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver

    import core.graph as graph
    import nodes.planner as planner
    import nodes.replanner as replanner
    from nodes.executor import _MockToolSampler
    from service import api

    planner_calls = []

    class Planner:
        def invoke(self, _payload):
            planner_calls.append(True)
            return '{"steps": ["summarize"]}'

    monkeypatch.setattr(graph, "build_llm", lambda **kwargs: _MockToolSampler("summarize"))
    monkeypatch.setattr(planner, "create_planner", lambda model: Planner())
    monkeypatch.setattr(replanner, "create_replanner", lambda model: types.SimpleNamespace(
        invoke=lambda _payload: '{"action":"response", "response":"completed"}',
    ))
    monkeypatch.setenv("EXECUTOR_SAMPLES", "2")
    store = SQLiteJobStore(str(tmp_path / "graph-service.sqlite3"))
    store.create(Job(id="graph", prompt="explain", deadline_at=time.time() + 30))
    terminal_write_attempted = threading.Event()
    original_finish = store.finish

    def lose_terminal_write(*args, **kwargs):
        terminal_write_attempted.set()
        return False

    monkeypatch.setattr(store, "finish", lose_terminal_write)
    saver = MemorySaver()
    dispatcher = api.Dispatcher(store, saver, workers=1, poll_seconds=0.01)
    dispatcher.start()
    try:
        assert terminal_write_attempted.wait(3)
    finally:
        dispatcher.stop()
    first = store.get("graph")
    assert first.status == "running", first.error
    assert first.checkpoint_ref is not None
    assert len(planner_calls) == 1
    # Simulate a crash after the final checkpoint but before the terminal job write.
    monkeypatch.setattr(store, "finish", original_finish)
    store.update("graph", status="running", result=None, lease_owner="dead", lease_expires=0)
    recovered = api.Dispatcher(store, saver, workers=1, poll_seconds=0.01)
    recovered.start()
    try:
        _wait_for(lambda: store.get("graph").status in ("done", "error"))
    finally:
        recovered.stop()
    job = store.get("graph")
    assert job.status == "done", job.error
    assert job.result["response"] == "completed"
    assert job.attempts == 2 and len(planner_calls) == 1
    assert job.checkpoint_ref != first.checkpoint_ref
    assert list(saver.list(None)) == []  # Terminal results do not retain memory snapshots.
    store.close()


def test_memory_checkpoint_loss_retries_from_prompt_after_restart(tmp_path, monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver

    from service import api

    store = SQLiteJobStore(str(tmp_path / "memory-restart.sqlite3"))
    store.create(Job(id="memory-restart", prompt="original prompt"))
    store.claim_next("dead-process", 10)
    store.update("memory-restart", lease_expires=0, checkpoint_ref={"configurable": {
        "thread_id": "lost-process-memory", "checkpoint_ns": "", "checkpoint_id": "missing",
    }})
    seen = []

    def fake_run(question, **kwargs):
        seen.append((question, kwargs["checkpoint_ref"], kwargs["resume"]))
        return {"response": "fresh retry"}

    monkeypatch.setattr(api, "run_execution_engine", fake_run)
    dispatcher = api.Dispatcher(store, MemorySaver(), workers=1, poll_seconds=0.01)
    dispatcher.start()
    try:
        _wait_for(lambda: store.get("memory-restart").status == "done")
    finally:
        dispatcher.stop()
    assert seen == [("original prompt", None, True)]
    store.close()
