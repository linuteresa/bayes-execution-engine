"""Opt-in process crashes against real service backends and a scripted HTTP model.

These tests validate persistence and the real OpenAI client, not model quality.
Enable TEST_SERVICE_RECOVERY=1 and configure TEST_REDIS_URL / TEST_POSTGRES_DSN.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from core.graph import checkpoint_thread_id
from persistence.checkpointer import build_checkpointer
from persistence.job_store import RedisJobStore, SQLiteJobStore


def _wait(predicate, *, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    pytest.fail("timed out waiting for live recovery state")


@contextmanager
def _scripted_model():
    counts = Counter()
    lock = threading.Lock()
    blocked, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = "\n".join(message["content"] for message in request["messages"])
            if "task replanner" in prompt:
                kind, answer = "replanner", json.dumps({"action": "response", "response": "Paris"})
            elif "orchestrator architecture planner" in prompt:
                kind, answer = "planner", json.dumps({"steps": ["Name the capital of France"]})
            else:
                assert "executing ONE step" in prompt
                kind, answer = "executor", "Paris"
            with lock:
                counts[kind] += 1
                pause = kind == "replanner" and counts[kind] == 1
            if pause:
                blocked.set()
                release.wait(60)
            body = json.dumps({
                "id": "scripted", "object": "chat.completion", "created": 0,
                "model": "scripted-recovery-only",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The first API process is deliberately killed mid-request.

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", counts, blocked, release
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def _api(env, log_path):
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "service.api:app", "--host", "127.0.0.1",
             "--port", str(port), "--log-level", "warning"],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2,
                          headers={"Authorization": "Bearer recovery-test"}, trust_env=False) as client:
            try:
                def ready():
                    if process.poll() is not None:
                        pytest.fail(f"API exited at startup:\n{log_path.read_text()}")
                    try:
                        return client.get("/health").status_code == 200
                    except httpx.TransportError:
                        return False

                _wait(ready)
                yield process, client
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


@pytest.mark.parametrize("backend", ["memory", "redis", "postgres"])
def test_live_api_recovers_after_sigkill_without_repeating_committed_steps(backend, tmp_path, monkeypatch):
    if os.getenv("TEST_SERVICE_RECOVERY") != "1":
        pytest.skip("set TEST_SERVICE_RECOVERY=1 for subprocess/HTTP crash recovery")
    for name in ({"redis": ["TEST_REDIS_URL"], "postgres": ["TEST_POSTGRES_DSN"]}.get(backend, [])):
        if not os.getenv(name):
            pytest.skip(f"set {name} for {backend} crash recovery")
    if backend == "redis":
        pytest.importorskip("langgraph.checkpoint.redis")
    if backend == "postgres":
        pytest.importorskip("langgraph.checkpoint.postgres")

    prefix = "recovery-test-" + uuid.uuid4().hex
    settings = {
        "JOB_STORE": "redis" if backend == "redis" else "sqlite",
        "JOB_DB_PATH": str(tmp_path / "jobs.sqlite3"), "REDIS_JOB_PREFIX": prefix,
        "REDIS_URL": os.getenv("TEST_REDIS_URL", "redis://127.0.0.1:16379/0"),
        "POSTGRES_DSN": os.getenv("TEST_POSTGRES_DSN", ""), "CHECKPOINTER": backend,
        "API_KEY": "recovery-test", "JOB_WORKERS": "1", "JOB_LEASE_SECONDS": "2",
        "JOB_TIMEOUT_SECONDS": "120", "LLM_TIMEOUT_SECONDS": "60",
        "LLAMA_MODEL": "scripted-recovery-only", "LLAMA_CPP_API_KEY": "not-needed",
        "EXECUTOR_SAMPLES": "4", "EXECUTOR_EVIDENCE": "self-consistency",
        "LANGGRAPH_STRICT_MSGPACK": "true",
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    for name in ("OTEL_EXPORTER_OTLP_ENDPOINT", "EMBEDDINGS_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    store = RedisJobStore(settings["REDIS_URL"]) if backend == "redis" else SQLiteJobStore(settings["JOB_DB_PATH"])
    job_id = None
    try:
        with _scripted_model() as (url, counts, blocked, release):
            env = {**os.environ, "LLAMA_CPP_BASE_URL": url}
            with _api(env, tmp_path / "first.log") as (first, client):
                accepted = client.post("/jobs", json={"question": "What is the capital of France?"})
                assert accepted.status_code == 202, accepted.text
                job_id = accepted.json()["job_id"]
                assert blocked.wait(30), (tmp_path / "first.log").read_text()
                def published():
                    job = store.get(job_id)
                    return job if job.checkpoint_ref else None

                before = _wait(published)
                if backend != "memory":
                    with build_checkpointer() as saver:
                        def completed_step():
                            job = store.get(job_id)
                            saved = saver.get_tuple(job.checkpoint_ref)
                            steps = saved.checkpoint["channel_values"].get("past_steps", [])
                            return job if [tuple(step) for step in steps] == [
                                ("Name the capital of France", "Paris")
                            ] else None

                        before = _wait(completed_step)
                first.kill()
                first.wait(timeout=5)
            release.set()
            # The second process must wait out the real lease; no direct store edits.
            with _api(env, tmp_path / "second.log") as (_, client):
                def complete():
                    response = client.get(f"/jobs/{job_id}")
                    assert response.status_code == 200
                    job = response.json()
                    assert job["status"] != "error", (job, (tmp_path / "second.log").read_text())
                    return job if job["status"] == "done" else None

                result = _wait(complete)
                assert result["result"]["response"] == "Paris"
                assert result["attempts"] == 2
                assert result["result"]["steps_executed"] == 1
                assert counts["planner"] == (2 if backend == "memory" else 1)
                assert counts["executor"] == (8 if backend == "memory" else 4)
                assert result["checkpoint_ref"] != before.checkpoint_ref
                # Two independent live API processes race to apply the same verdict.
                with _api(env, tmp_path / "third.log") as (_, other):
                    with ThreadPoolExecutor(2) as pool:
                        responses = list(pool.map(
                            lambda c: c.post(f"/jobs/{job_id}/feedback", json={"correct": True}),
                            [client, other],
                        ))
                    assert sorted(r.status_code for r in responses) == [200, 409]
                    assert other.get(f"/jobs/{job_id}").json()["feedback"] is True
                assert store.observation_counts().sum() == 1
    finally:
        if job_id and backend != "memory":
            job = store.get(job_id)
            with build_checkpointer() as saver:
                for attempt in job.attempt_ids:
                    saver.delete_thread(checkpoint_thread_id(job.thread_id, job.run_id, attempt))
        if backend == "redis":
            store._r.delete(store._jobs, store._counts, store._feedback, store._pending, store._terminal)
        store.close()
