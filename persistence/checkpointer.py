"""Lifespan-managed LangGraph savers. Requested backends fail explicitly."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def build_checkpointer() -> Iterator[Any]:
    """Enter and initialize a saver; keep its connection open for the whole run.

    Persistent saver factories themselves return context managers. Missing packages,
    setup failures and invalid backend names are startup errors, never memory fallback.
    """
    backend = os.getenv("CHECKPOINTER", "memory").lower()
    if backend == "none":
        yield None
    elif backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        yield MemorySaver()
    elif backend == "redis":
        from langgraph.checkpoint.redis import RedisSaver
        with RedisSaver.from_conn_string(os.getenv("REDIS_URL", "redis://localhost:6379/0")) as saver:
            saver.setup()
            yield saver
    elif backend == "postgres":
        from langgraph.checkpoint.postgres import PostgresSaver
        with PostgresSaver.from_conn_string(os.getenv("POSTGRES_DSN", "postgresql://localhost:5432/bayes")) as saver:
            saver.setup()
            yield saver
    else:
        raise ValueError(f"unknown CHECKPOINTER backend: {backend}")
