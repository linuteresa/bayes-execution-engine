"""
Asynchronous FastAPI service for the Bayes Execution Engine.

Why async + a job queue
-----------------------
A Plan-and-Execute run makes several LLM calls and can take tens of seconds. Doing
that inside a synchronous request invites gateway/load-balancer timeouts and ties up
a worker per in-flight prompt. Instead, we use the classic *submit / poll* pattern:

    POST /jobs        -> 202 Accepted, returns {job_id}        (returns instantly)
    GET  /jobs/{id}   -> job status, and the result once done

The heavy work runs off the request thread. Here it runs as a FastAPI ``BackgroundTask``
backed by a thread pool, which is enough to prove the decoupling. In production the
background task is replaced by a real broker (Kafka / RabbitMQ + Celery) consuming
from a queue -- the API contract above does not change. See docs/SYSTEM_DESIGN.md.

State (jobs + LangGraph checkpoints) lives in a pluggable store so the service is
stateless and horizontally scalable.

Run with:  uvicorn service.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bayesian_engine.bayes_engine import get_default_engine, outcome_label
from core.graph import run_execution_engine
from core.telemetry import log_event
from persistence.checkpointer import build_checkpointer
from persistence.job_store import Job, build_job_store

app = FastAPI(title="Bayes Execution Engine", version="1.0.0")

_store = build_job_store()
_pool = ThreadPoolExecutor(max_workers=8)
_feedback_lock = threading.Lock()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User prompt to orchestrate")
    thread_id: str | None = Field(None, description="Session id for state isolation/resume")


class JobAccepted(BaseModel):
    job_id: str
    status: str


class FeedbackRequest(BaseModel):
    correct: bool = Field(..., description="Was the job's final answer right?")


def _run_job(job_id: str, question: str, thread_id: str) -> None:
    _store.update(job_id, status="running")
    try:
        checkpointer = build_checkpointer()
        result = run_execution_engine(question, checkpointer=checkpointer, thread_id=thread_id)
        if result.get("error"):
            _store.update(job_id, status="error", error=result["error"], result=result)
            log_event("job.error", job_id=job_id, error=result["error"])
        else:
            _store.update(job_id, status="done", result=result)
            log_event("job.done", job_id=job_id, confidence=result.get("confidence_score"))
    except Exception as exc:  # noqa: BLE001
        _store.update(job_id, status="error", error=str(exc))
        log_event("job.error", job_id=job_id, error=str(exc))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/jobs", response_model=JobAccepted, status_code=202)
def submit_job(req: AskRequest) -> JobAccepted:
    job_id = uuid.uuid4().hex
    thread_id = req.thread_id or job_id
    _store.create(Job(id=job_id, prompt=req.question))
    _pool.submit(_run_job, job_id, req.question, thread_id)
    log_event("job.submitted", job_id=job_id, thread_id=thread_id)
    return JobAccepted(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = _store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()


@app.post("/jobs/{job_id}/feedback")
def submit_feedback(job_id: str, req: FeedbackRequest) -> dict:
    """Teach the engine from a verdict on a finished job.

    The verdict is per run, so every step's evidence gets the same outcome -- coarse
    credit assignment, but the same conjugate update the calibration eval uses.
    """
    with _feedback_lock:  # check-then-write, so two verdicts for one job can't both land
        job = _store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"job is {job.status}, not done")
        if job.feedback is not None:
            raise HTTPException(status_code=409, detail="feedback already recorded")

        engine = get_default_engine()
        evidence = (job.result or {}).get("step_evidence", [])
        for ev in evidence:
            engine.observe(
                ev["TaskStatus"], ev["DataQuality"], ev["ToolReliability"],
                outcome_label(req.correct),
            )
        path = os.getenv("ENGINE_STATE_PATH")
        if path:
            engine.save(path)
        _store.update(job_id, feedback=req.correct)

    log_event("job.feedback", job_id=job_id, correct=req.correct, observations=len(evidence))
    return {"job_id": job_id, "observations": len(evidence)}
