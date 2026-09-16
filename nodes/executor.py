"""Executor node: runs one DAG step and resolves conflicts via real Bayesian inference.

**One evidence path.** Whether the step is executed by a live model or by the offline
mock tool, the three ordinal signals fed to the Bayesian engine come from the same
place: ``nodes.llm_executor`` samples the backend N times and *measures* how much the
samples agree. Conflicts therefore fire on observed disagreement, which is what the
README claims, rather than on substring matches for words like "conflict" or "failed".

That was not always true. The offline path used to derive evidence by keyword-matching
the result text (``core.signals``), so the repo shipped two contradictory notions of
what evidence *is*, and the one the README described was only half of it. The keyword
extractor still exists for deployments with no sampling budget -- a single-shot tool
that returns text and nothing else -- but it is now opt-in via
``EXECUTOR_EVIDENCE=keywords`` and is never the default. See ``core/signals.py``.

Backends:

* **LLM mode** (an ``executor_model`` in the config) -- the real model, sampled at
  temperature > 0.
* **Mock mode** (no model) -- ``_MockToolSampler``, a deterministic fixture that
  simulates enterprise tools which sometimes return conflicting data. It is a *fake
  tool*, not a fake evidence path: its samples are measured exactly like a model's, so
  the offline suite exercises the real machinery.
"""

from __future__ import annotations

import os

from bayesian_engine.bayes_engine import good_probability, resolve_conflict
from core.state import PlanExecuteState
from core.telemetry import log_conflict_resolution, log_event
from nodes.llm_executor import execute_step_with_llm

_CONFLICT_STATE_INDEX = 2

# A flaky-tool fixture: for each trigger, the alternative readings the "tool" can
# return. A single entry means the tool is stable and every sample agrees; several
# mean the sources genuinely disagree, which the sampler surfaces as low agreement.
_MOCK_RESPONSES: dict[str, tuple[str, ...]] = {
    "search": ("search_result: Found 5 relevant documents",),
    "query": (
        "query_result: Retrieved 3 records, latest timestamp 2024-03-11",
        "query_result: Retrieved 5 records, latest timestamp 2023-11-02",
        "query_result: Retrieved 3 records, no timestamp column present",
        "query_result: Upstream replica returned 12 rows for the same key",
    ),
    "validate": (
        "validate_result: Schema check passed on 2 of 3 sources",
        "validate_result: Source B reports a different total than source A",
        "validate_result: Validation inconclusive, 2 sources disagree",
        "validate_result: All three sources agree on 41 records",
    ),
}


def _mock_responses_for(task: str) -> tuple[str, ...]:
    t = task.lower()
    for trigger, responses in _MOCK_RESPONSES.items():
        if trigger in t:
            return responses
    return (f"executed: {task}",)


class _MockToolSampler:
    """Deterministic stand-in backend used when no LLM is configured.

    Cycles through the scripted readings for a task, so repeated sampling of a flaky
    tool yields genuinely divergent observations and repeated sampling of a stable one
    yields identical ones. Deterministic, so the offline test suite stays reproducible.
    """

    def __init__(self, task: str):
        self._responses = _mock_responses_for(task)
        self._calls = 0

    def invoke(self, _prompt: str) -> str:
        out = self._responses[self._calls % len(self._responses)]
        self._calls += 1
        return out


def simple_executor(task: str) -> str:
    """The mock tool's single-shot reading -- its first scripted response."""
    return _mock_responses_for(task)[0]


def executor_node(state: PlanExecuteState, config=None) -> dict:
    """Execute the first task in the plan DAG and resolve any conflict probabilistically."""
    if not state.get("plan"):
        return {"response": "No tasks in plan"}

    current_task = state["plan"][0]
    configurable = (config or {}).get("configurable", {}) if config else {}
    sampler = configurable.get("executor_model")
    mode = "llm"
    if sampler is None:
        sampler, mode = _MockToolSampler(current_task), "mock"

    result_text, confidence = _execute_step(sampler, current_task, state.get("input", ""), mode)

    return {
        "plan": state["plan"][1:],
        "past_steps": [(current_task, result_text)],
        "confidence_score": confidence,
    }


def _execute_step(sampler, task: str, goal: str, mode: str) -> tuple[str, float]:
    exec_result = execute_step_with_llm(sampler, task, goal)
    evidence = (
        _keyword_evidence(task, exec_result.answer)
        if _evidence_mode() == "keywords"
        else exec_result.evidence
    )
    summary = resolve_conflict(evidence)
    confidence = good_probability(summary)
    answer = exec_result.answer or f"(no result produced for: {task})"

    if summary["state_index"] >= _CONFLICT_STATE_INDEX:
        log_conflict_resolution(
            task=task,
            evidence=evidence,
            summary={
                **summary,
                "consistency": round(exec_result.consistency, 3),
                "backend": mode,
                "evidence_mode": _evidence_mode(),
            },
        )
    else:
        log_event(
            "executor.step",
            task=task,
            confidence=confidence,
            consistency=round(exec_result.consistency, 3),
            backend=mode,
            evidence_mode=_evidence_mode(),
        )
    return answer, confidence


def _evidence_mode() -> str:
    """``self-consistency`` (default) or the opt-in legacy ``keywords`` extractor."""
    return os.getenv("EXECUTOR_EVIDENCE", "self-consistency").strip().lower()


def _keyword_evidence(task: str, result: str) -> dict:
    """Legacy no-sampling fallback. Imported lazily so the default path never needs it."""
    from core.signals import extract_evidence

    return extract_evidence(task, result).as_evidence()


__all__ = ["executor_node", "simple_executor"]
