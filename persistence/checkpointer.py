"""
LangGraph checkpointer factory for fault-tolerant, concurrent execution.

LangGraph keeps the ``PlanExecuteState`` in memory by default, so a crash mid-DAG
loses all progress and two concurrent runs can only be isolated by separate
processes. A *checkpointer* persists state after every node transition, keyed by a
``thread_id``. That buys us three things:

  * Fault tolerance -- a restarted worker resumes a run from its last checkpoint.
  * Concurrency / multi-tenancy -- each session is a distinct ``thread_id``, so many
    users' runs share one process with zero state bleed.
  * Horizontal scaling -- with a shared Postgres/Redis saver, any stateless worker
    behind the load balancer can pick up any thread.

``build_checkpointer()`` selects a backend from ``CHECKPOINTER`` / connection-string
env vars and degrades gracefully to an in-memory saver if the backend libs are absent.
"""

from __future__ import annotations

import os
from typing import Any, Optional


def build_checkpointer() -> Optional[Any]:
    """Return a LangGraph checkpointer based on the ``CHECKPOINTER`` env var.

    Values: ``memory`` (default), ``redis``, ``postgres``, ``none``.
    """
    backend = os.getenv("CHECKPOINTER", "memory").lower()

    if backend == "none":
        return None

    if backend == "redis":
        try:
            from langgraph.checkpoint.redis import RedisSaver

            return RedisSaver.from_conn_string(
                os.getenv("REDIS_URL", "redis://localhost:6379/0")
            )
        except Exception:
            pass

    if backend == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            return PostgresSaver.from_conn_string(
                os.getenv("POSTGRES_DSN", "postgresql://localhost:5432/bayes")
            )
        except Exception:
            pass

    try:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    except Exception:
        return None
