"""
Structured JSON logging and lightweight tracing for the execution engine.

Every node emits machine-parseable events instead of free-text prints, so that a
log pipeline (Datadog / Loki / CloudWatch) or an OpenTelemetry collector can build
dashboards and alerts on, e.g., the rate of low-confidence conflict resolutions or
the distribution of the 125-state evidence matrix.

I deliberately keep zero hard dependencies: if ``OTEL_EXPORTER_OTLP_ENDPOINT`` is
set and the OpenTelemetry SDK is installed, spans are exported; otherwise tracing is
a no-op and only structured logs are written.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

_LOGGER_NAME = "bayes_engine"
_TRACE_ID: ContextVar[str | None] = ContextVar("engine_trace_id", default=None)
_PROVIDER = None
_PROVIDER_LOCK = threading.Lock()


def configure_telemetry():
    """Configure OTLP/gRPC export once when explicitly requested by environment.

    Without an endpoint, structured JSON logging remains available with no SDK.
    Explicit tracing configuration fails visibly when its dependencies are missing.
    """
    global _PROVIDER
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None
    with _PROVIDER_LOCK:
        if _PROVIDER is None:
            try:
                from opentelemetry import trace
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
            except ImportError as exc:
                raise RuntimeError("OTLP tracing requires requirements-telemetry.txt") from exc
            provider = TracerProvider(resource=Resource.create({
                "service.name": os.getenv("OTEL_SERVICE_NAME", "bayes-execution-engine")
            }))
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            trace.set_tracer_provider(provider)
            _PROVIDER = provider
        return _PROVIDER


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
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
    if not fields.get("trace_id") and _TRACE_ID.get():
        fields["trace_id"] = _TRACE_ID.get()
    get_logger().info(event, extra={"extra_fields": fields})


def log_conflict_resolution(
    *,
    task: str,
    evidence: Dict[str, int],
    summary: Dict[str, Any],
    trace_id: Optional[str] = None,
) -> None:
    """Specialised event for Bayesian conflict resolution.

    Emits both confidence scalars with the interval that belongs to each, plus the full
    evidence coordinates so alerts can fire on degraded data quality.

    The two pairs are *not* interchangeable. ``map_confidence`` is ``P(MAP state)`` and
    is bounded by ``map_credible_interval``; ``confidence`` is ``P(CERTAIN)+P(HIGH)``,
    the number the executor gates on, and is bounded by ``credible_interval``. Emitting
    the MAP interval alongside the gating confidence -- as this used to -- attaches an
    uncertainty band to a quantity it does not describe, and dashboards built on it
    would alert on the wrong thing.
    """
    log_event(
        "bayes.conflict_resolved",
        trace_id=trace_id,
        task=task,
        evidence=evidence,
        resolved_state=summary.get("state"),
        confidence=summary.get("good_confidence", summary.get("confidence")),
        credible_interval=summary.get(
            "good_credible_interval", summary.get("credible_interval")
        ),
        map_confidence=summary.get("confidence"),
        map_credible_interval=summary.get("credible_interval"),
        effective_sample_size=summary.get("effective_sample_size"),
        distribution=summary.get("distribution"),
    )


@contextmanager
def span(name: str, trace_id: Optional[str] = None, **attributes: Any) -> Iterator[str]:
    """Trace a unit of work.

    Uses OpenTelemetry if available + configured, otherwise emits start/end JSON logs
    with a duration. Yields the trace id so callers can correlate child events.
    """
    provider = configure_telemetry()
    tid = trace_id or uuid.uuid4().hex
    token = _TRACE_ID.set(tid)
    start = time.perf_counter()

    otel_span_cm = None
    if provider is not None:
        from opentelemetry import trace as _otel_trace

        tracer = _otel_trace.get_tracer(_LOGGER_NAME)
        otel_span_cm = tracer.start_as_current_span(name)
        otel_span = otel_span_cm.__enter__()
        for key, value in attributes.items():
            if value is not None:
                otel_span.set_attribute(key, value)

    log_event(f"{name}.start", trace_id=tid, **attributes)
    try:
        yield tid
    except BaseException as exc:
        if otel_span_cm is not None:
            from opentelemetry.trace import Status, StatusCode
            otel_span.record_exception(exc)
            otel_span.set_status(Status(StatusCode.ERROR, str(exc)))
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        log_event(f"{name}.end", trace_id=tid, duration_ms=round(duration_ms, 2))
        if otel_span_cm is not None:  # pragma: no cover
            otel_span_cm.__exit__(None, None, None)
        _TRACE_ID.reset(token)
