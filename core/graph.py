"""Shared LangGraph wiring for the Plan-and-Execute engine.

Both the CLI (``main.py``) and the web/API layers import from here so the graph is
defined exactly once.
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from typing import Any, Callable, Dict, Literal, Optional

from langgraph.graph import END, START, StateGraph

from core.llm import build_llm
from core.runtime import (
    MAX_PROMPT_CHARS_CEILING,
    ExecutionCancelled,
    ModelOutputError,
    check_deadline,
    positive_float,
    positive_int,
)
from core.state import PlanExecuteState
from core.telemetry import span
from nodes.executor import executor_node
from nodes.planner import planner_node
from nodes.replanner import RECHECK_PREFIX, replanner_node
from persistence.fenced_checkpointer import FencedCheckpointer

_LLM_UNAVAILABLE_PHRASES = (
    "connection error",
    "connection refused",
    "actively refused",
    "failed to connect",
    "api connection error",
)


def should_continue(state: PlanExecuteState) -> Literal["executor", "END"]:
    """Route to executor while plan steps remain, otherwise finish."""
    if state.get("response"):
        return "END"
    if state.get("plan"):
        return "executor"
    return "END"


def overall_confidence(past_steps, step_confidences) -> float:
    """Run-level confidence: the weakest step, counting only each step's latest attempt.

    A re-check replaces the attempt it re-checks, so a step that was redone and came
    back confident no longer drags the run down.
    """
    latest: Dict[str, float] = {}
    for (task, _), conf in zip(past_steps, step_confidences, strict=True):
        if not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError("step confidence must be finite and in [0, 1]")
        latest[task.removeprefix(RECHECK_PREFIX)] = conf
    return min(latest.values(), default=0.0)


def checkpoint_thread_id(session_id: str | None, run_id: str,
                         attempt_id: str | None = None) -> str:
    """Isolate runs within a session, and worker attempts within a leased run.

    The two-argument form retains the address used by direct, non-worker callers.
    """
    key = f"{session_id or ''}\0{run_id}"
    if attempt_id is not None:
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("attempt_id must be a nonempty string")
        key += f"\0attempt:{attempt_id}"
    return hashlib.sha256(key.encode()).hexdigest()


def _guarded(node):
    def run_node(state, config):
        check_deadline(config)
        result = node(state, config)
        # Stop cancelled work at node boundaries. Leased workers additionally use
        # attempt-isolated saver writes and an ownership-fenced recovery pointer.
        check_deadline(config)
        return result
    return run_node


def build_graph(checkpointer: Optional[Any] = None):
    """Compile the Plan-and-Execute StateGraph.

    Pass a LangGraph ``checkpointer`` (e.g. a Redis/Postgres saver) to persist state
    across process restarts and enable concurrent, isolated sessions by thread id.
    """
    workflow = StateGraph(PlanExecuteState)
    workflow.add_node("planner", _guarded(planner_node))
    workflow.add_node("executor", _guarded(executor_node))
    workflow.add_node("replanner", _guarded(replanner_node))

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "replanner")
    workflow.add_conditional_edges(
        "replanner", should_continue, {"executor": "executor", "END": END}
    )
    return workflow.compile(checkpointer=checkpointer)


def run_execution_engine(
    user_input: str,
    *,
    checkpointer: Optional[Any] = None,
    thread_id: Optional[str] = None,
    run_id: Optional[str] = None,
    resume: bool = False,
    deadline: float | None = None,
    cancel_event: Any = None,
    attempt_id: str | None = None,
    checkpoint_ref: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], bool] | None = None,
) -> Dict[str, Any]:
    """Start an isolated run, or resume the same explicit run after interruption.

    A new question always gets a new run ID even when its session ID is reused.
    Resuming without a checkpoint (e.g. a crash before the first write) starts fresh.
    Leased workers supply a unique attempt_id and an ownership-checking callback;
    only the exact checkpoint_ref that callback published is copied on recovery.
    Completed snapshots can be recovered after the inference deadline; new model
    work always requires time remaining and cancellation checks still apply.
    The confidence is a weakest-step score, not a calibrated final-answer probability.
    """
    try:
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must be a nonempty string")
        if len(user_input) > positive_int("MAX_PROMPT_CHARS", 16000, maximum=MAX_PROMPT_CHARS_CEILING):
            raise ValueError("user_input exceeds MAX_PROMPT_CHARS")
        if resume and not run_id:
            raise ValueError("resume requires an explicit run_id")
        if checkpoint_ref is not None and not resume:
            raise ValueError("checkpoint_ref requires resume=True")
        if checkpoint_callback is not None and not attempt_id:
            raise ValueError("checkpoint_callback requires an explicit attempt_id")
        # Worker mode is all-or-nothing. A saver plus an attempt_id without a callback
        # used to run unfenced AND never publish a reference, so the job's
        # checkpoint_ref stayed None and the next attempt silently restarted from
        # scratch -- both halves of durable recovery failing with no error at all.
        if attempt_id is not None and checkpointer is not None and checkpoint_callback is None:
            raise ValueError("attempt_id with a saver requires an ownership-checking callback")
        if checkpoint_ref is not None and (checkpointer is None or checkpoint_callback is None):
            raise ValueError("checkpoint_ref requires a saver and ownership-checking callback")
        run_id = run_id or uuid.uuid4().hex
        deadline = deadline if deadline is not None else time.time() + positive_float("JOB_TIMEOUT_SECONDS", 300)

        initial_state = {
            "input": user_input,
            "plan": [],
            "past_steps": [],
            "step_confidences": [],
            "step_evidence": [],
            "response": "",
            "confidence_score": 0.0,
        }
        config: Dict[str, Any] = {
            "configurable": {"deadline": deadline, "cancel_event": cancel_event,
                             "thread_id": checkpoint_thread_id(thread_id, run_id, attempt_id),
                             "checkpoint_ns": ""},
            "recursion_limit": 2 * positive_int("MAX_STEPS", 10, maximum=100) + 4,
        }
        if not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        # Recovery can reuse a completed answer after an outage exceeded the
        # inference budget. Ownership/cancellation still govern publishing it.
        cancellation_config = {"configurable": {"cancel_event": cancel_event}}
        check_deadline(cancellation_config)
        if not resume:
            check_deadline(config)
        if checkpointer is not None and checkpoint_callback is not None:
            checkpointer = FencedCheckpointer(checkpointer, checkpoint_callback)
            if checkpoint_ref is not None:
                restored = checkpointer.restore_from(checkpoint_ref, config)
                config["configurable"].update(restored["configurable"])
        app_graph = build_graph(checkpointer=checkpointer)
        checkpoint = app_graph.get_state(config) if checkpointer is not None else None
        has_checkpoint = checkpoint is not None and bool(checkpoint.values)
        if has_checkpoint and not resume:
            raise ValueError("run_id already exists; use resume=True or a new run_id")
        if has_checkpoint and checkpoint.values.get("input") != user_input:
            raise ValueError("resume input does not match the saved run")
        with span("engine.run", thread_id=thread_id, run_id=run_id):
            if resume and has_checkpoint and not checkpoint.next:
                check_deadline(cancellation_config)
                final_state = checkpoint.values
            else:
                remaining = check_deadline(config)
                config["configurable"].update({
                    "model": build_llm(temperature=0.0, timeout=remaining),
                    "executor_model": build_llm(
                        temperature=positive_float("EXECUTOR_TEMPERATURE", 0.7),
                        timeout=remaining,
                    ),
                })
                final_state = app_graph.invoke(None if resume and has_checkpoint else initial_state,
                                              config=config)

        past_steps = [tuple(step) for step in final_state.get("past_steps", [])]
        step_confidences = final_state.get("step_confidences", [])
        if not past_steps or not final_state.get("response"):
            raise ModelOutputError("Execution produced no measured answer")
        return {
            "response": final_state.get("response", "No response"),
            "steps_executed": len(past_steps),
            "confidence_score": overall_confidence(past_steps, step_confidences),
            "confidence_scope": "weakest_step",
            "run_id": run_id,
            "session_id": thread_id,
            "past_steps": past_steps,
            "step_confidences": step_confidences,
            "step_evidence": final_state.get("step_evidence", []),
        }
    except Exception as exc:  # noqa: BLE001
        if _is_llm_unavailable_error(exc) or isinstance(exc, (ModelOutputError, TimeoutError, ValueError)):
            code = "invalid_request"
            error = str(exc)
            if _is_llm_unavailable_error(exc):
                code = "llm_unavailable"
                error = ("Local LLM is unavailable. Start llama-server at the configured "
                         "LLAMA_CPP_BASE_URL and try again.")
            elif isinstance(exc, ExecutionCancelled):
                code = "execution_cancelled"
            elif isinstance(exc, TimeoutError):
                code = "execution_timeout"
            elif isinstance(exc, ModelOutputError):
                code = "invalid_model_output"
            return {
                "response": "",
                "steps_executed": 0,
                "confidence_score": 0.0,
                "past_steps": [],
                "step_confidences": [], "step_evidence": [],
                "error": error,
                "error_code": code,
                "run_id": run_id,
            }
        raise


def _is_llm_unavailable_error(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    return any(phrase in text for phrase in _LLM_UNAVAILABLE_PHRASES)
