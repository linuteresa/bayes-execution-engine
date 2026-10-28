"""Validated execution limits shared by graph nodes and model clients."""

from __future__ import annotations

import math
import os
import time


class ModelOutputError(ValueError):
    """The model did not produce a valid orchestration decision."""


class ExecutionCancelled(TimeoutError):
    """The worker no longer owns this execution."""


def positive_int(name: str, default: int, *, maximum: int = 100000) -> int:
    value = int(os.getenv(name, str(default)))
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def remaining_seconds(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    if not math.isfinite(deadline):
        raise ValueError("deadline must be finite")
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Execution deadline exceeded")
    return remaining


def check_deadline(config: dict | None) -> float | None:
    settings = (config or {}).get("configurable", {})
    event = settings.get("cancel_event")
    if event is not None and event.is_set():
        raise ExecutionCancelled("Execution cancelled after worker lease loss")
    return remaining_seconds(settings.get("deadline"))


def bounded_model(model, deadline: float | None):
    """Apply a per-call deadline to the real chat client; retain fixture interfaces."""
    remaining = remaining_seconds(deadline)
    if remaining is not None:
        from langchain_openai import ChatOpenAI

        if isinstance(model, ChatOpenAI):
            return model.bind(timeout=min(remaining, positive_float("LLM_TIMEOUT_SECONDS", 60)))
    return model
