"""Executor node: runs one DAG step and resolves conflicts via real Bayesian inference."""

from __future__ import annotations

from bayesian_engine.bayes_engine import resolve_conflict
from core.signals import extract_evidence, is_conflict
from core.state import PlanExecuteState
from core.telemetry import log_conflict_resolution, log_event


def simple_executor(task: str) -> str:
    """Mock execution backend for demonstration.

    In production this is where a tool/MCP call happens. The string it returns is
    treated as an *observation* that gets mapped into ordinal evidence -- it is no
    longer used as a magic keyword switch for the Bayesian update.
    """
    t = task.lower()
    if "search" in t:
        return "search_result: Found 5 relevant documents"
    if "query" in t:
        return "query_result: Retrieved 3 records with conflicting timestamps"
    if "validate" in t:
        return "validate_result: Data validation uncertain - 2 sources disagree"
    return f"executed: {task}"


def executor_node(state: PlanExecuteState, config=None) -> dict:
    """Execute the first task in the plan DAG.

    Steps:
      1. Pop the first task and execute it.
      2. Map the *real* result into an ordinal evidence vector (no hardcoding).
      3. If the observation is conflicting/ambiguous, run a Bayesian update and use
         the posterior-predictive confidence as the step's confidence score.
      4. Append (task, result) to past_steps and shrink the plan.
    """
    if not state.get("plan"):
        return {"response": "No tasks in plan"}

    current_task = state["plan"][0]
    result = simple_executor(current_task)

    confidence = 1.0
    if is_conflict(result):
        evidence = extract_evidence(current_task, result).as_evidence()
        summary = resolve_conflict(evidence)
        confidence = summary["confidence"]
        log_conflict_resolution(task=current_task, evidence=evidence, summary=summary)
    else:
        log_event("executor.step", task=current_task, result=result, confidence=confidence)

    return {
        "plan": state["plan"][1:],
        "past_steps": [(current_task, result)],
        "confidence_score": confidence,
    }
