"""Durable dispatch, worker leases, and exactly-once feedback accounting.

SQLite is the default for a single host. Redis supports multiple hosts. Memory is
an explicit demo backend. A feedback event and its counts commit together; workers
read those shared counts instead of overwriting independent engine snapshots.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class QueueFull(Exception):
    """The configured number of active jobs has been reached."""


class FeedbackConflict(Exception):
    """Feedback requires a completed job without an existing verdict."""


@dataclass
class Job:
    id: str
    status: str = "queued"
    prompt: str = ""
    thread_id: str = ""
    run_id: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    feedback: bool | None = None
    deadline_at: float | None = None
    lease_owner: str | None = None
    lease_expires: float = 0.0
    attempts: int = 0
    attempt_id: str = ""
    attempt_ids: list[str] = field(default_factory=list)
    checkpoint_ref: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.thread_id = self.thread_id or self.id
        self.run_id = self.run_id or self.id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _feedback_contexts(job: Job) -> list[int]:
    contexts = []
    for ev in (job.result or {}).get("step_evidence", []):
        values = [ev[name] for name in ("TaskStatus", "DataQuality", "ToolReliability")]
        if any(type(v) is not int or not 0 <= v < 5 for v in values):
            raise ValueError("invalid feedback evidence")
        contexts.append(values[0] * 25 + values[1] * 5 + values[2])
    return contexts


class JobStore(Protocol):
    def create(self, job: Job, max_pending: int | None = None) -> None: ...
    def get(self, job_id: str) -> Job | None: ...
    def update(self, job_id: str, **changes: Any) -> None: ...
    def claim_next(self, owner: str, lease_seconds: float) -> Job | None: ...
    def renew(self, job_id: str, owner: str, lease_seconds: float) -> bool: ...
    def finish(self, job_id: str, owner: str, **changes: Any) -> bool: ...
    def record_checkpoint(self, job_id: str, owner: str, checkpoint_ref: dict) -> bool: ...
    def record_feedback(self, job_id: str, correct: bool) -> int: ...
    def observation_counts(self) -> np.ndarray: ...
    def expired_jobs(self, retention_seconds: float) -> list[Job]: ...
    def prune(self, retention_seconds: float, job_ids: list[str] | None = None) -> int: ...
    def close(self) -> None: ...


class SQLiteJobStore:
    """Transactions serialize admission, dispatch, and feedback across processes.

    Use a local filesystem, not NFS. Each API process owns one connection; SQLite's
    WAL and BEGIN IMMEDIATE provide coordination between those connections.
    """

    def __init__(self, path: str = "./data/jobs.sqlite3") -> None:
        if path != ":memory:":
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(Path(path).expanduser()) if path != ":memory:" else path,
                                   timeout=30, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT NOT NULL,
                lease_owner TEXT, lease_expires REAL NOT NULL,
                created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS jobs_dispatch ON jobs(status, lease_expires, created_at);
            CREATE TABLE IF NOT EXISTS feedback (
                job_id TEXT PRIMARY KEY, correct INTEGER NOT NULL, evidence TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS counts (
                context INTEGER NOT NULL, outcome INTEGER NOT NULL, n INTEGER NOT NULL,
                PRIMARY KEY(context, outcome)
            );
        """)

    @contextmanager
    def _transaction(self):
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    @staticmethod
    def _read(db, job_id: str) -> Job | None:
        row = db.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
        return Job(**json.loads(row[0])) if row else None

    @staticmethod
    def _write(db, job: Job) -> None:
        db.execute("""INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)
                      ON CONFLICT(id) DO UPDATE SET status=excluded.status,
                      payload=excluded.payload, lease_owner=excluded.lease_owner,
                      lease_expires=excluded.lease_expires, updated_at=excluded.updated_at""",
                   (job.id, job.status, json.dumps(job.to_dict()), job.lease_owner,
                    job.lease_expires, job.created_at, job.updated_at))

    def create(self, job: Job, max_pending: int | None = None) -> None:
        with self._transaction() as db:
            if self._read(db, job.id) is not None:
                raise ValueError("job already exists")
            if max_pending is not None:
                pending = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
                if pending >= max_pending:
                    raise QueueFull("job queue is full")
            self._write(db, job)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._read(self._db, job_id)

    def update(self, job_id: str, **changes: Any) -> None:
        with self._transaction() as db:
            job = self._read(db, job_id)
            if job is not None:
                for key, value in changes.items():
                    setattr(job, key, value)
                job.updated_at = time.time()
                self._write(db, job)

    def claim_next(self, owner: str, lease_seconds: float) -> Job | None:
        with self._transaction() as db:
            now = time.time()
            row = db.execute("""SELECT payload FROM jobs WHERE status='queued'
                                OR (status='running' AND lease_expires<=?)
                                ORDER BY created_at LIMIT 1""", (now,)).fetchone()
            if row is None:
                return None
            job = Job(**json.loads(row[0]))
            job.status, job.lease_owner = "running", owner
            job.lease_expires = now + lease_seconds
            job.attempts += 1
            job.attempt_id = uuid.uuid4().hex
            job.attempt_ids.append(job.attempt_id)
            job.updated_at = now
            self._write(db, job)
            return job

    def renew(self, job_id: str, owner: str, lease_seconds: float) -> bool:
        return self._owned_update(job_id, owner, lease_seconds=lease_seconds)

    def finish(self, job_id: str, owner: str, **changes: Any) -> bool:
        return self._owned_update(job_id, owner, lease_owner=None, lease_expires=0.0, **changes)

    def record_checkpoint(self, job_id: str, owner: str, checkpoint_ref: dict) -> bool:
        return self._owned_update(job_id, owner, checkpoint_ref=checkpoint_ref)

    def _owned_update(self, job_id: str, owner: str, *, lease_seconds: float | None = None,
                      **changes: Any) -> bool:
        with self._transaction() as db:
            job = self._read(db, job_id)
            now = time.time()
            if (job is None or job.status != "running" or job.lease_owner != owner
                    or job.lease_expires <= now):
                return False
            if lease_seconds is not None:
                changes["lease_expires"] = now + lease_seconds
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = time.time()
            self._write(db, job)
            return True

    def record_feedback(self, job_id: str, correct: bool) -> int:
        with self._transaction() as db:
            job = self._read(db, job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status != "done" or job.feedback is not None:
                raise FeedbackConflict("feedback requires a completed job without a verdict")
            contexts = _feedback_contexts(job)
            db.execute("INSERT INTO feedback VALUES (?, ?, ?, ?)",
                       (job_id, int(correct), json.dumps(contexts), time.time()))
            for context in contexts:
                db.execute("""INSERT INTO counts VALUES (?, ?, 1)
                              ON CONFLICT(context,outcome) DO UPDATE SET n=n+1""",
                           (context, 0 if correct else 4))
            job.feedback, job.updated_at = correct, time.time()
            self._write(db, job)
            return len(contexts)

    def observation_counts(self) -> np.ndarray:
        with self._lock:
            rows = self._db.execute("SELECT context,outcome,n FROM counts").fetchall()
        result = np.zeros((125, 5), dtype=float)
        for context, outcome, n in rows:
            result[context, outcome] = n
        return result

    def prune(self, retention_seconds: float, job_ids: list[str] | None = None) -> int:
        with self._transaction() as db:
            cutoff = time.time() - retention_seconds
            query = "SELECT id FROM jobs WHERE status IN ('done','error') AND updated_at<?"
            if job_ids is None:
                ids = [row[0] for row in db.execute(query, (cutoff,))]
            else:
                ids = []
                for start in range(0, len(job_ids), 500):
                    batch = job_ids[start:start + 500]
                    placeholders = ','.join('?' for _ in batch)
                    ids.extend(row[0] for row in db.execute(
                        query + f" AND id IN ({placeholders})", (cutoff, *batch),
                    ))
            db.executemany("DELETE FROM feedback WHERE job_id=?", [(job_id,) for job_id in ids])
            db.executemany("DELETE FROM jobs WHERE id=?", [(job_id,) for job_id in ids])
            return len(ids)

    def expired_jobs(self, retention_seconds: float) -> list[Job]:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM jobs WHERE status IN ('done','error') AND updated_at<?",
                (time.time() - retention_seconds,),
            ).fetchall()
        return [Job(**json.loads(row[0])) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()


class InMemoryJobStore(SQLiteJobStore):
    """Explicit demo backend; provides the same transaction semantics in one process."""

    def __init__(self) -> None:
        super().__init__(":memory:")


class RedisJobStore:
    """Shared store using WATCH transactions; requires redis-py and a Redis server.

    All mutations watch the jobs hash, making feedback and lease transitions atomic
    across replicas. This favors correctness at this service's bounded queue size.
    """

    def __init__(self, url: str | None = None, ttl_seconds: int = 86400) -> None:
        import redis
        self._r = redis.Redis.from_url(
            url or os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            socket_connect_timeout=5, socket_timeout=5,
        )
        self._r.ping()
        self._prefix = os.getenv("REDIS_JOB_PREFIX", "bayes")
        self._jobs = f"{self._prefix}:jobs"
        self._counts = f"{self._prefix}:counts"
        self._feedback = f"{self._prefix}:feedback"
        self._pending = f"{self._prefix}:pending"
        self._terminal = f"{self._prefix}:terminal"
        self._ttl = ttl_seconds

    def _mutate(self, operation):
        import redis
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with self._r.pipeline() as pipe:
                try:
                    pipe.watch(self._jobs)
                    writes, deletes, event, result = operation(pipe)
                    pipe.multi()
                    for job in writes:
                        pipe.hset(self._jobs, job.id, json.dumps(job.to_dict()))
                        if job.status in ('queued', 'running'):
                            eligible = job.created_at if job.status == 'queued' else job.lease_expires
                            pipe.zadd(self._pending, {job.id: eligible})
                            pipe.zrem(self._terminal, job.id)
                        else:
                            pipe.zrem(self._pending, job.id)
                            pipe.zadd(self._terminal, {job.id: job.updated_at})
                    if deletes:
                        pipe.hdel(self._jobs, *deletes)
                        pipe.hdel(self._feedback, *deletes)
                        pipe.zrem(self._pending, *deletes)
                        pipe.zrem(self._terminal, *deletes)
                    if event is not None:
                        job_id, correct, contexts = event
                        pipe.hset(self._feedback, job_id, json.dumps([correct, contexts]))
                        for context in contexts:
                            pipe.hincrby(self._counts, f"{context}:{0 if correct else 4}", 1)
                    pipe.execute()
                    return result
                except redis.WatchError:
                    continue
        raise TimeoutError("Redis job transaction contention exceeded 5 seconds")

    def create(self, job: Job, max_pending: int | None = None) -> None:
        def op(pipe):
            if pipe.hexists(self._jobs, job.id):
                raise ValueError("job already exists")
            if max_pending is not None and pipe.zcard(self._pending) >= max_pending:
                raise QueueFull("job queue is full")
            return [job], [], None, None
        self._mutate(op)

    def get(self, job_id: str) -> Job | None:
        return self._read(self._r, job_id)

    def _read(self, reader, job_id: str) -> Job | None:
        raw = reader.hget(self._jobs, job_id)
        return Job(**json.loads(raw)) if raw else None

    def update(self, job_id: str, **changes: Any) -> None:
        def op(pipe):
            job = self._read(pipe, job_id)
            if job is None:
                return [], [], None, None
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = time.time()
            return [job], [], None, None
        self._mutate(op)

    def claim_next(self, owner: str, lease_seconds: float) -> Job | None:
        def op(pipe):
            now = time.time()
            ids = pipe.zrangebyscore(self._pending, '-inf', now, start=0, num=1)
            if not ids:
                return [], [], None, None
            job = self._read(pipe, ids[0])
            job.status, job.lease_owner = 'running', owner
            job.lease_expires, job.updated_at = now + lease_seconds, now
            job.attempts += 1
            job.attempt_id = uuid.uuid4().hex
            job.attempt_ids.append(job.attempt_id)
            return [job], [], None, job
        return self._mutate(op)

    def _owned_update(self, job_id: str, owner: str, *, lease_seconds: float | None = None,
                      **changes: Any) -> bool:
        def op(pipe):
            job = self._read(pipe, job_id)
            now = time.time()
            if (job is None or job.status != 'running' or job.lease_owner != owner
                    or job.lease_expires <= now):
                return [], [], None, False
            if lease_seconds is not None:
                changes["lease_expires"] = now + lease_seconds
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = time.time()
            return [job], [], None, True
        return self._mutate(op)

    def renew(self, job_id: str, owner: str, lease_seconds: float) -> bool:
        return self._owned_update(job_id, owner, lease_seconds=lease_seconds)

    def finish(self, job_id: str, owner: str, **changes: Any) -> bool:
        return self._owned_update(job_id, owner, lease_owner=None, lease_expires=0.0, **changes)

    def record_checkpoint(self, job_id: str, owner: str, checkpoint_ref: dict) -> bool:
        return self._owned_update(job_id, owner, checkpoint_ref=checkpoint_ref)

    def record_feedback(self, job_id: str, correct: bool) -> int:
        def op(pipe):
            job = self._read(pipe, job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status != 'done' or job.feedback is not None or pipe.hexists(self._feedback, job_id):
                raise FeedbackConflict('feedback requires a completed job without a verdict')
            contexts = _feedback_contexts(job)
            job.feedback, job.updated_at = correct, time.time()
            return [job], [], (job_id, correct, contexts), len(contexts)
        return self._mutate(op)

    def observation_counts(self) -> np.ndarray:
        result = np.zeros((125, 5), dtype=float)
        for key, value in self._r.hgetall(self._counts).items():
            context, outcome = (int(part) for part in key.decode().split(':'))
            result[context, outcome] = int(value)
        return result

    def prune(self, retention_seconds: float, job_ids: list[str] | None = None) -> int:
        def op(pipe):
            ids = pipe.zrangebyscore(self._terminal, '-inf', time.time() - retention_seconds)
            if job_ids is not None:
                candidates = {job_id.encode() for job_id in job_ids}
                ids = [job_id for job_id in ids if job_id in candidates]
            return [], ids, None, len(ids)
        return self._mutate(op)

    def expired_jobs(self, retention_seconds: float) -> list[Job]:
        cutoff = time.time() - retention_seconds
        ids = self._r.zrangebyscore(self._terminal, '-inf', cutoff)
        if not ids:
            return []
        jobs = [Job(**json.loads(raw)) for raw in self._r.hmget(self._jobs, ids) if raw]
        return [job for job in jobs if job.status in ('done', 'error') and job.updated_at < cutoff]

    def close(self) -> None:
        self._r.close()


def build_job_store() -> JobStore:
    backend = os.getenv("JOB_STORE", "sqlite").lower()
    if backend == "sqlite":
        return SQLiteJobStore(os.getenv("JOB_DB_PATH", "./data/jobs.sqlite3"))
    if backend == "redis":
        return RedisJobStore()
    if backend == "memory":
        return InMemoryJobStore()
    raise ValueError(f"unknown JOB_STORE backend: {backend}")
