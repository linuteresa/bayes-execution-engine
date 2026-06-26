"""
Structured JSON logging and lightweight tracing for the execution engine.

Every node emits machine-parseable events instead of free-text prints, so that a
log pipeline (Datadog / Loki / CloudWatch) or an OpenTelemetry collector can build
dashboards and alerts on, e.g., the rate of low-confidence conflict resolutions or
the distribution of the 125-state evidence matrix.

We deliberately keep zero hard dependencies: if ``OTEL_EXPORTER_OTLP_ENDPOINT`` is
set and the OpenTelemetry SDK is installed, spans are exported; otherwise tracing is
a no-op and only structured logs are written.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

_LOGGER_NAME = "bayes_engine"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)  # type: ignore[attr-defined]
        return json.dumps(payload, default=str)


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    return logger


def log_event(event: str, **fields: Any) -> None:
    """Emit a structured JSON log line."""
    get_logger().info(event, extra={"extra_fields": fields})


def log_conflict_resolution(
    *,
    task: str,
    evidence: Dict[str, int],
    summary: Dict[str, Any],
    trace_id: Optional[str] = None,
) -> None:
    """Specialised event for Bayesian conflict resolution.

    Emits the confidence score, the resolved state, the credible interval, and the
    full evidence matrix coordinates so alerts can fire on degraded data quality.
    """
    log_event(
        "bayes.conflict_resolved",
        trace_id=trace_id,
        task=task,
        evidence=evidence,
        resolved_state=summary.get("state"),
        confidence=summary.get("confidence"),
        credible_interval=summary.get("credible_interval"),
        effective_sample_size=summary.get("effective_sample_size"),
        distribution=summary.get("distribution"),
    )


@contextmanager
def span(name: str, trace_id: Optional[str] = None, **attributes: Any) -> Iterator[str]:
    """Trace a unit of work.

    Uses OpenTelemetry if available + configured, otherwise emits start/end JSON logs
    with a duration. Yields the trace id so callers can correlate child events.
    """
    tid = trace_id or uuid.uuid4().hex
    start = time.perf_counter()

    otel_span_cm = None
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        try:  # pragma: no cover - exercised only when OTEL is installed/configured
            from opentelemetry import trace as _otel_trace

            tracer = _otel_trace.get_tracer(_LOGGER_NAME)
            otel_span_cm = tracer.start_as_current_span(name)
            otel_span_cm.__enter__()
        except Exception:
            otel_span_cm = None

    log_event(f"{name}.start", trace_id=tid, **attributes)
    try:
        yield tid
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        log_event(f"{name}.end", trace_id=tid, duration_ms=round(duration_ms, 2))
        if otel_span_cm is not None:  # pragma: no cover
            otel_span_cm.__exit__(None, None, None)
