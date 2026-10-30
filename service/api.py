"""Submit/poll API with durable dispatch, leased recovery, and atomic feedback."""
from __future__ import annotations

import asyncio
import hmac
import math
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field, field_validator

from bayesian_engine.bayes_engine import set_count_source
from core.graph import checkpoint_thread_id, run_execution_engine
from core.runtime import MAX_PROMPT_CHARS_CEILING, positive_float, positive_int
from core.telemetry import log_event
from persistence.checkpointer import build_checkpointer
from persistence.job_store import (
    FeedbackConflict,
    InMemoryJobStore,
    Job,
    JobStore,
    QueueFull,
    build_job_store,
)

load_dotenv()
# Importing routes never opens a file or starts workers; production state is owned
# by lifespan. This demo store also allows isolated direct-function unit tests.
_store: JobStore = InMemoryJobStore()
_dispatcher = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1,
                          max_length=positive_int("MAX_PROMPT_CHARS", 16000, maximum=MAX_PROMPT_CHARS_CEILING))
    thread_id: str | None = Field(None, min_length=1, max_length=128)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be blank")
        return value


class JobAccepted(BaseModel):
    job_id: str
    status: str


class FeedbackRequest(BaseModel):
    correct: bool = Field(..., strict=True, description="Was the job's final answer right?")


class Dispatcher:
    """Workers claim persisted jobs directly; there is no volatile pending queue."""

    def __init__(self, store: JobStore, checkpointer, *, workers: int = 8,
                 lease_seconds: float = 30, retention_seconds: float = 86400,
                 poll_seconds: float = 0.25) -> None:
        durations = (lease_seconds, retention_seconds, poll_seconds)
        if workers < 1 or any(not math.isfinite(value) or value <= 0 for value in durations):
            raise ValueError("worker, lease, retention and polling limits must be positive")
        self.store, self.checkpointer = store, checkpointer
        self.workers, self.lease_seconds = workers, lease_seconds
        self.retention_seconds, self.poll_seconds = retention_seconds, poll_seconds
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for _ in range(self.workers):
            thread = threading.Thread(target=self._worker, daemon=True)
            thread.start()
            self._threads.append(thread)

    def wake(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        # Keep the saver and store alive until active bounded model requests finish.
        for thread in self._threads:
            thread.join()

    def prune(self) -> None:
        expired = self.store.expired_jobs(self.retention_seconds)
        for job in expired:
            self._delete_checkpoints(job)
        self.store.prune(self.retention_seconds, [job.id for job in expired])

    def _delete_checkpoints(self, job: Job) -> None:
        if self.checkpointer is not None:
            keys = {checkpoint_thread_id(job.thread_id, job.run_id)}
            keys.update(checkpoint_thread_id(job.thread_id, job.run_id, attempt_id)
                        for attempt_id in job.attempt_ids)
            for key in keys:
                self.checkpointer.delete_thread(key)

    def _worker(self) -> None:
        next_cleanup = 0.0
        while not self._stop.is_set():
            try:
                if time.monotonic() >= next_cleanup:
                    self.prune()
                    next_cleanup = time.monotonic() + 60
                # A new token per claim prevents an old callback becoming valid
                # again when this same worker later reclaims its expired job.
                owner = uuid.uuid4().hex
                job = self.store.claim_next(owner, self.lease_seconds)
                if job is not None:
                    self._run_job(job, owner)
                    continue
            except Exception as exc:  # Store outages retry; unclaimed jobs stay durable.
                log_event("worker.error", error=str(exc))
            self._wake.wait(self.poll_seconds)
            self._wake.clear()

    def _run_job(self, job: Job, owner: str) -> None:
        finished = threading.Event()
        lost_lease = threading.Event()

        def heartbeat() -> None:
            while not finished.wait(max(0.01, self.lease_seconds / 3)):
                try:
                    if not self.store.renew(job.id, owner, self.lease_seconds):
                        lost_lease.set()
                        return
                except Exception as exc:
                    # Ownership is uncertain after a store outage. Never publish a
                    # result under that uncertainty; the lease allows recovery.
                    lost_lease.set()
                    log_event("worker.heartbeat_error", job_id=job.id, error=str(exc))
                    return

        renewer = threading.Thread(target=heartbeat, daemon=True)
        renewer.start()
        try:
            reference = job.checkpoint_ref
            if self.checkpointer is None:
                reference = None
            elif isinstance(self.checkpointer, MemorySaver) and reference is not None:
                # Memory checkpoints disappear on process restart. Retry the pure
                # model workflow from its prompt; durable backends fail on missing
                # published references instead of silently discarding progress.
                if self.checkpointer.get_tuple(reference) is None:
                    reference = None
            result = run_execution_engine(
                job.prompt, checkpointer=self.checkpointer, thread_id=job.thread_id,
                run_id=job.run_id, resume=job.attempts > 1, deadline=job.deadline_at,
                cancel_event=lost_lease,
                attempt_id=job.attempt_id, checkpoint_ref=reference,
                checkpoint_callback=lambda ref: self.store.record_checkpoint(job.id, owner, ref),
            )
            changes = {"status": "error" if result.get("error") else "done", "result": result,
                       "error": result.get("error")}
        except Exception as exc:
            changes = {"status": "error", "error": str(exc)}
        finally:
            finished.set()
            renewer.join()
        if lost_lease.is_set():
            log_event("job.lease_lost", job_id=job.id)
            return
        if self.store.finish(job.id, owner, **changes):
            log_event(f"job.{changes['status']}", job_id=job.id)
            if isinstance(self.checkpointer, MemorySaver):
                # The durable terminal result contains all data needed by polling
                # and feedback. Completed in-memory snapshots need no retention.
                self._delete_checkpoints(job)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _dispatcher
    from core.telemetry import configure_telemetry

    previous_store = _store
    provider = configure_telemetry()
    store = None
    try:
        positive_float("JOB_TIMEOUT_SECONDS", 300)
        positive_int("JOB_MAX_PENDING", 128)
        store = build_job_store()
        with build_checkpointer() as saver:
            _store = store
            set_count_source(store.observation_counts)
            dispatcher = Dispatcher(
                store, saver, workers=positive_int("JOB_WORKERS", 8, maximum=64),
                lease_seconds=positive_float("JOB_LEASE_SECONDS", 30),
                retention_seconds=positive_float("JOB_RETENTION_SECONDS", 86400),
            )
            _dispatcher = dispatcher
            dispatcher.start()
            try:
                yield
            finally:
                await asyncio.to_thread(dispatcher.stop)
                _dispatcher = None
    finally:
        _dispatcher = None
        set_count_source(None)
        if store is not None:
            store.close()
        _store = previous_store
        if provider is not None:
            provider.shutdown()


app = FastAPI(title="Bayes Execution Engine", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def require_api_key(request, call_next):
    api_key = os.getenv("API_KEY")
    if api_key and request.url.path != "/health":
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied.encode(), f"Bearer {api_key}".encode()):
            return JSONResponse(status_code=401, content={"detail": "invalid API key"},
                                headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/jobs", response_model=JobAccepted, status_code=202)
def submit_job(req: AskRequest) -> JobAccepted:
    job_id = uuid.uuid4().hex
    timeout = positive_float("JOB_TIMEOUT_SECONDS", 300)
    capacity = positive_int("JOB_MAX_PENDING", 128)
    job = Job(id=job_id, prompt=req.question, thread_id=req.thread_id or job_id,
              run_id=job_id, deadline_at=time.time() + timeout)
    try:
        _store.create(job, max_pending=capacity)
    except QueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "5"}) from exc
    if _dispatcher is not None:
        _dispatcher.wake()
    log_event("job.submitted", job_id=job_id, thread_id=job.thread_id)
    return JobAccepted(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = _store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()


@app.post("/jobs/{job_id}/feedback")
def submit_feedback(job_id: str, req: FeedbackRequest) -> dict:
    """Commit a verdict and all count increments together in the shared store."""
    try:
        observations = _store.record_feedback(job_id, req.correct)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    except FeedbackConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    log_event("job.feedback", job_id=job_id, correct=req.correct, observations=observations)
    return {"job_id": job_id, "observations": observations}
