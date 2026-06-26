"""
Map raw tool output into ordinal Bayesian evidence.

The Bayesian engine reasons over three ordinal signals (TaskStatus, DataQuality,
ToolReliability). Those signals have to come from *somewhere* real. In this demo the
"somewhere" is the text returned by a tool call; in a production system it would be
structured tool metadata (HTTP status, row counts, schema-validation results,
retriever scores, circuit-breaker state, ...).

This module isolates that mapping so the engine never hardcodes evidence. Each
extractor returns an index in 0..4 where 0 = CERTAIN (best) and 4 = AMBIGUOUS (worst),
matching ``bayesian_engine.STATE_NAMES``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

# Keyword -> degradation weight. Higher weight pushes a signal toward AMBIGUOUS.
_CONFLICT_TERMS = {
    "conflict": 4,
    "disagree": 4,
    "contradict": 4,
    "uncertain": 3,
    "ambiguous": 3,
    "missing": 3,
    "stale": 2,
    "partial": 2,
    "timeout": 3,
    "retry": 2,
    "error": 3,
    "failed": 4,
}
_POSITIVE_TERMS = {
    "found": 0,
    "retrieved": 1,
    "validated": 0,
    "success": 0,
    "confirmed": 0,
    "executed": 1,
}


def _clip(v: int) -> int:
    return max(0, min(4, v))


@dataclass
class EvidenceVector:
    task_status: int
    data_quality: int
    tool_reliability: int

    def as_evidence(self) -> Dict[str, int]:
        return {
            "TaskStatus": self.task_status,
            "DataQuality": self.data_quality,
            "ToolReliability": self.tool_reliability,
        }


def extract_evidence(task: str, result: str) -> EvidenceVector:
    """Derive an ordinal evidence vector from the executed task and its result.

    Heuristic but deterministic and fully transparent:

    * DataQuality   -- driven by conflict/uncertainty language in the result.
    * TaskStatus    -- driven by whether the result reads like a clean completion.
    * ToolReliability -- driven by error/retry/timeout signals plus a small penalty
      for low result specificity (no numbers => vaguer output).
    """
    text = f"{result}".lower()

    conflict_score = max(
        (w for term, w in _CONFLICT_TERMS.items() if term in text), default=0
    )
    positive_score = min(
        (w for term, w in _POSITIVE_TERMS.items() if term in text), default=4
    )


    data_quality = _clip(conflict_score)


    task_status = _clip(max(positive_score, conflict_score - 1))


    reliability_terms = ("error", "failed", "timeout", "retry")
    reliability_hit = max(
        (_CONFLICT_TERMS[t] for t in reliability_terms if t in text), default=0
    )
    specificity_bonus = 0 if re.search(r"\d", text) else 1
    tool_reliability = _clip(reliability_hit + specificity_bonus)

    return EvidenceVector(task_status, data_quality, tool_reliability)


def is_conflict(result: str) -> bool:
    """Cheap gate: does this result warrant a Bayesian update at all?"""
    text = result.lower()
    return any(
        term in text
        for term in ("conflict", "uncertain", "disagree", "ambiguous", "missing", "error", "failed", "timeout")
    )
