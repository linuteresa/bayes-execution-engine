"""Run the real execution path over gold-answer questions and log one row per item.

This is the data-collection half of the eval. It deliberately calls the same
``execute_step_with_llm`` -> ``resolve_conflict`` -> ``good_probability`` chain the
executor node uses, so the confidences being scored are the confidences the agent
actually reports. Nothing is re-derived for the benefit of the eval.

Each :class:`EvalRow` keeps everything needed to replay the analysis offline: the
ordinal evidence triple, the continuous signals behind it, the engine's confidence and
credible interval, the ESS, the medoid answer, the baselines, and correctness against
gold. Sampling is the expensive part (N model calls per item), so rows are written to
JSONL and every later stage -- metrics, the learning loop, figures -- reads that file
instead of touching the model again.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from bayesian_engine.bayes_engine import (
    STATE_MAP,
    DirichletBayesianEngine,
    good_probability,
)
from eval.answerers import is_simulated
from eval.datasets import EvalItem
from eval.grading import grade
from nodes.llm_executor import execute_step_with_llm

SIGNAL_KEYS = (
    "agreement_mean",
    "agreement_min",
    "answerability",
    "distinct_ratio",
    "length_mean",
    "length_cv",
)


@dataclass
class EvalRow:
    """One evaluated question: what we measured, what we predicted, what was true."""

    item_id: str
    question: str
    difficulty: str
    source: str
    grader: str

    # --- ordinal evidence fed to the Bayesian engine (the 125-cell context) ---
    task_status: int
    data_quality: int
    tool_reliability: int

    # --- engine output ---
    confidence: float
    credible_low: float
    credible_high: float
    effective_sample_size: float
    map_state: str

    # --- baselines the engine has to beat ---
    agreement: float
    mean_logprob: Optional[float]

    # --- answer + label ---
    answer: str
    gold: List[str]
    correct: int

    n_samples: int
    signals: Dict[str, float] = field(default_factory=dict)
    samples: List[str] = field(default_factory=list)

    @property
    def context(self) -> tuple[int, int, int]:
        return (self.task_status, self.data_quality, self.tool_reliability)

    def as_dict(self) -> dict:
        return asdict(self)


def _prompt_for(item: EvalItem) -> str:
    """Turn a QA item into a single plan step.

    The harness evaluates one step, not a whole plan: multi-step planning would mix
    planner quality into a number that is supposed to be about the confidence signal.
    """
    return item.question


def run_harness(
    items: Sequence[EvalItem],
    answerer,
    *,
    n_samples: int = 4,
    engine: Optional[DirichletBayesianEngine] = None,
    keep_samples: bool = False,
    progress: Optional[Callable[[int, int, "EvalRow"], None]] = None,
) -> List[EvalRow]:
    """Evaluate every item and return the logged rows.

    ``engine`` defaults to a fresh prior-only :class:`DirichletBayesianEngine` -- the
    state the shipped system is actually in, which is what Fix 1 is measuring. The
    learning loop later refits an engine from these same rows.
    """
    eng = engine if engine is not None else DirichletBayesianEngine()
    rows: List[EvalRow] = []

    for i, item in enumerate(items):
        sampler = answerer.sampler_for(item, n_samples=n_samples)
        result = execute_step_with_llm(
            sampler, _prompt_for(item), item.question, n_samples=n_samples
        )
        summary = eng.resolve(result.evidence).as_dict()
        correct = grade(result.answer, item.answers, item.grader)

        row = EvalRow(
            item_id=item.id,
            question=item.question,
            difficulty=item.difficulty,
            source=item.source,
            grader=item.grader,
            task_status=int(result.evidence["TaskStatus"]),
            data_quality=int(result.evidence["DataQuality"]),
            tool_reliability=int(result.evidence["ToolReliability"]),
            confidence=good_probability(summary),
            credible_low=float(summary["credible_interval"][0]),
            credible_high=float(summary["credible_interval"][1]),
            effective_sample_size=float(summary["effective_sample_size"]),
            map_state=str(summary["state"]),
            agreement=float(result.consistency),
            mean_logprob=_safe_logprob(sampler),
            answer=result.answer,
            gold=list(item.answers),
            correct=int(bool(correct)),
            n_samples=n_samples,
            signals={k: float(result.signals.get(k, 0.0)) for k in SIGNAL_KEYS},
            samples=list(result.samples) if keep_samples else [],
        )
        rows.append(row)
        if progress is not None:
            progress(i + 1, len(items), row)

    return rows


def _safe_logprob(sampler) -> Optional[float]:
    getter = getattr(sampler, "mean_logprob", None)
    if getter is None:
        return None
    try:
        value = getter()
    except Exception:  # noqa: BLE001 - a missing baseline must not fail the run
        return None
    return None if value is None else float(value)


# ------------------------------------------------------------------ persistence
def write_rows(rows: Sequence[EvalRow], path: Path | str) -> Path:
    """Write rows as JSONL so every downstream stage is a pure function of this file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.as_dict(), ensure_ascii=False) + "\n")
    return path


def read_rows(path: Path | str) -> List[EvalRow]:
    """Load rows written by :func:`write_rows`."""
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(EvalRow(**json.loads(line)))
    return rows


def run_metadata(answerer, items, *, n_samples: int, dataset_spec: str) -> dict:
    """Provenance stamped onto every artefact.

    ``is_simulated`` is the important field: it is what stops a pipeline smoke-test run
    from being read later as a calibration result for a real model.
    """
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "answerer": getattr(answerer, "name", type(answerer).__name__),
        "model": getattr(answerer, "model_name", "unknown"),
        "is_simulated": is_simulated(answerer),
        "dataset_spec": dataset_spec,
        "n_items": len(items),
        "n_samples_per_item": n_samples,
    }


def outcome_label(correct: int) -> int:
    """Map a binary correctness label onto the engine's 5-state ordinal outcome axis.

    A gold-answer eval yields one bit per item, but ``observe`` expects an outcome in
    ``0..4``. We collapse onto the endpoints: correct -> ``CERTAIN``, wrong ->
    ``AMBIGUOUS``. Since the reported confidence is ``P(CERTAIN) + P(HIGH)`` and HIGH
    then accrues no counts, the posterior confidence in a well-observed context
    converges to that context's empirical accuracy -- which is exactly the quantity a
    calibrated confidence is supposed to equal. Intermediate outcome states stay
    reachable from the prior, so low-count contexts are still smoothly regularized.
    """
    return STATE_MAP["CERTAIN"] if correct else STATE_MAP["AMBIGUOUS"]


__all__ = [
    "EvalRow",
    "run_harness",
    "write_rows",
    "read_rows",
    "run_metadata",
    "outcome_label",
    "SIGNAL_KEYS",
]
