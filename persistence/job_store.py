"""
Durable job store for the async execution service.

Long-running LLM orchestration must not block an HTTP request. The API enqueues a
job, returns a job id immediately, and the engine writes the result back here. A
client polls until the job is ``done``.

Two interchangeable backends:
  * ``InMemoryJobStore``  -- zero dependencies, fine for a single process / demos.
  * ``RedisJobStore``     -- survives restarts and is shared across workers, so the
    service scales horizontally behind a load balancer.

``build_job_store()`` picks one from the ``JOB_STORE`` / ``REDIS_URL`` env vars.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued -> running -> done | error
    prompt: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JobStore(Protocol):
    def create(self, job: Job) -> None: ...
    def get(self, job_id: str) -> Optional[Job]: ...
    def update(self, job_id: str, **changes: Any) -> None: ...


class InMemoryJobStore:
    """Thread-safe in-process store. State is lost on restart."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in changes.items():
                setattr(job, k, v)
            job.updated_at = time.time()


class RedisJobStore:
    """Redis-backed store. Survives restarts; shared across workers/replicas."""

    def __init__(self, url: Optional[str] = None, ttl_seconds: int = 86400) -> None:
        import redis  # imported lazily so redis stays optional

        self._r = redis.Redis.from_url(url or os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        self._ttl = ttl_seconds

    def _key(self, job_id: str) -> str:
        return f"job:{job_id}"

    def create(self, job: Job) -> None:
        self._r.set(self._key(job.id), json.dumps(job.to_dict()), ex=self._ttl)

    def get(self, job_id: str) -> Optional[Job]:
        raw = self._r.get(self._key(job_id))
        return Job(**json.loads(raw)) if raw else None

    def update(self, job_id: str, **changes: Any) -> None:
        job = self.get(job_id)
        if job is None:
            return
        for k, v in changes.items():
            setattr(job, k, v)
        job.updated_at = time.time()
        self._r.set(self._key(job_id), json.dumps(job.to_dict()), ex=self._ttl)


def build_job_store() -> JobStore:
    backend = os.getenv("JOB_STORE", "memory").lower()
    if backend == "redis":
        return RedisJobStore()
    return InMemoryJobStore()
