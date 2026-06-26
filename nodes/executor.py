"""Executor node: runs one DAG step and resolves conflicts via real Bayesian inference.

Two execution modes:

* **LLM mode** (when an ``executor_model`` is supplied) -- the step is run by the model
  via self-consistency sampling (``nodes.llm_executor``). The model's *measured*
  agreement becomes real Bayesian evidence, so the conflict resolver fires on genuine
  model uncertainty and the consensus answer flows to the replanner.
* **Mock mode** (no model) -- a deterministic keyword stub simulating enterprise tools
  that sometimes return conflicting data. Keeps the engine runnable and unit-testable
  without a live LLM.
"""

from __future__ import annotations

from bayesian_engine.bayes_engine import resolve_conflict
from core.signals import extract_evidence, is_conflict
from core.state import PlanExecuteState
from core.telemetry import log_conflict_resolution, log_event
from nodes.llm_executor import execute_step_with_llm

_CONFLICT_STATE_INDEX = 2


def simple_executor(task: str) -> str:
    """Deterministic mock backend used when no LLM is available (tests / offline).

    Simulates tools that sometimes return clean results and sometimes conflicting or
    uncertain data. The returned string is treated as an *observation*, not a keyword
    switch.
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
    """Execute the first task in the plan DAG and resolve any conflict probabilistically."""
    if not state.get("plan"):
        return {"response": "No tasks in plan"}

    current_task = state["plan"][0]
    configurable = (config or {}).get("configurable", {}) if config else {}
    sampler = configurable.get("executor_model")

    if sampler is not None:
        result_text, confidence = _execute_with_llm(sampler, current_task, state.get("input", ""))
    else:
        result_text, confidence = _execute_mock(current_task)

    return {
        "plan": state["plan"][1:],
        "past_steps": [(current_task, result_text)],
        "confidence_score": confidence,
    }


def _good_probability(summary: dict) -> float:
    """Posterior probability the step's outcome is high quality: P(CERTAIN) + P(HIGH).

    A principled scalar derived from the full Bayesian posterior over outcome quality --
    high when all three signals are good, low when the model disagrees with itself.
    """
    dist = summary["distribution"]
    return float(dist.get("CERTAIN", 0.0) + dist.get("HIGH", 0.0))


def _execute_with_llm(sampler, task: str, goal: str) -> tuple[str, float]:
    exec_result = execute_step_with_llm(sampler, task, goal)
    evidence = exec_result.evidence
    summary = resolve_conflict(evidence)
    confidence = _good_probability(summary)
    answer = exec_result.answer or f"(no result produced for: {task})"

    if summary["state_index"] >= _CONFLICT_STATE_INDEX:

        log_conflict_resolution(
            task=task,
            evidence=evidence,
            summary={**summary, "consistency": round(exec_result.consistency, 3)},
        )
    else:
        log_event(
            "executor.step",
            task=task,
            confidence=confidence,
            consistency=round(exec_result.consistency, 3),
        )
    return answer, confidence


def _execute_mock(task: str) -> tuple[str, float]:
    result = simple_executor(task)
    confidence = 1.0
    if is_conflict(result):
        evidence = extract_evidence(task, result).as_evidence()
        summary = resolve_conflict(evidence)
        confidence = summary["confidence"]
        log_conflict_resolution(task=task, evidence=evidence, summary=summary)
    else:
        log_event("executor.step", task=task, result=result, confidence=confidence)
    return result, confidence
