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

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from bayesian_engine.bayes_engine import (
    DirichletBayesianEngine,
    good_probability,
    outcome_label,  # lives in the engine so the service can use it; re-exported here
)
from eval.datasets import EvalItem, validate_item_ids
from eval.grading import grade
from eval.provenance import (
    SCHEMA_VERSION,
    digest,
    experiment_fingerprint,
    inferred_dataset_spec,
    make_experiment,
    utc_now,
)
from nodes.llm_executor import execute_step_with_llm, similarity_backend

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
    #: 95% credible interval for ``confidence`` (the summed good-state probability).
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

    # --- provenance, stamped on every row ---
    # Deliberately duplicated per row rather than kept only in a sibling metrics.json.
    # rows.jsonl is the documented unit of exchange, and a file that can be moved or
    # shared on its own must carry the one fact that stops a synthetic run being read as
    # a real calibration result. `None` means unknown, which is NOT the same as False.
    is_simulated: Optional[bool] = None
    answerer: str = ""
    model: str = ""
    experiment: Dict[str, object] = field(default_factory=dict)

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
    on_row: Optional[Callable[["EvalRow"], None]] = None,
    experiment: Optional[dict] = None,
    initial_rows: Sequence[EvalRow] = (),
) -> List[EvalRow]:
    """Evaluate every item and return the logged rows.

    ``engine`` defaults to a fresh prior-only :class:`DirichletBayesianEngine` -- the
    state the shipped system is actually in, which is what Fix 1 is measuring. The
    learning loop later refits an engine from these same rows.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    validate_item_ids(items)
    eng = engine if engine is not None else DirichletBayesianEngine()
    if initial_rows and experiment is None:
        raise ValueError("resuming requires explicit experiment provenance")
    experiment = experiment or run_metadata(
        answerer, items, n_samples=n_samples, keep_samples=keep_samples
    )
    if engine is not None:
        experiment = deepcopy(experiment)
        experiment["execution"]["engine"] = "provided_engine"
        experiment["execution"]["state_sha256"] = digest({
            "prior_strength": engine.prior_strength, "prior_spread": engine.prior_spread,
            "counts": engine.observation_counts().tolist(),
        })
        experiment["experiment_id"] = experiment_fingerprint(experiment)
    if engine is not None and initial_rows:
        raise ValueError("resuming a custom learned engine is unsupported")
    rows: List[EvalRow] = list(initial_rows)
    validate_resume_rows(rows, items, experiment)
    completed = {row.item_id for row in rows}

    for item in items:
        if item.id in completed:
            continue
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
            # The interval for P(CERTAIN)+P(HIGH), i.e. for `confidence` itself -- not
            # summary["credible_interval"], which bounds the MAP state's probability and
            # need not even contain this number.
            credible_low=float(summary["good_credible_interval"][0]),
            credible_high=float(summary["good_credible_interval"][1]),
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
            is_simulated=experiment.get("is_simulated"),
            answerer=str(getattr(answerer, "name", type(answerer).__name__)),
            model=str(getattr(answerer, "model_name", "unknown")),
            experiment=deepcopy(experiment),
        )
        # A writer callback is deliberately separate from progress: --quiet must not
        # suppress durable writes, and a failed write must not count an item as saved.
        if on_row is not None:
            on_row(row)
        rows.append(row)
        if progress is not None:
            progress(len(rows), len(items), row)

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
    validate_row_ids(rows)
    with RowWriter(path, overwrite=True) as writer:
        for row in rows:
            writer.append(row)
    return path


def read_rows(path: Path | str, *, repair_truncated_tail: bool = False) -> List[EvalRow]:
    """Load rows written by :func:`write_rows`."""
    path = Path(path)
    rows = []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    offset = 0
    for index, line in enumerate(lines):
        if line.strip():
            try:
                rows.append(EvalRow(**json.loads(line)))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not (repair_truncated_tail and index == len(lines) - 1
                        and not line.endswith(b"\n")):
                    raise
                # A crash may tear only the final append. Never hide malformed complete
                # lines or alter a valid preceding row while recovering it.
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                    stream.flush()
                    os.fsync(stream.fileno())
                break
        offset += len(line)
    validate_row_ids(rows)
    return rows


def validate_row_ids(rows: Sequence[EvalRow]) -> None:
    seen = set()
    for row in rows:
        if not row.item_id or row.item_id in seen:
            raise ValueError(f"duplicate or empty item id {row.item_id!r} in rows")
        seen.add(row.item_id)


def validate_resume_rows(rows: Sequence[EvalRow], items: Sequence[EvalItem], experiment: dict) -> None:
    validate_row_ids(rows)
    expected = {item.id: item for item in items}
    for row in rows:
        item = expected.get(row.item_id)
        if item is None or (
            row.question, row.gold, row.grader, row.difficulty, row.source, row.n_samples
        ) != (
            item.question, item.answers, item.grader, item.difficulty, item.source,
            experiment["n_samples_per_item"],
        ):
            raise ValueError(f"saved item {row.item_id!r} differs from the requested dataset")
        if not row.experiment or row.experiment.get("experiment_id") != experiment.get("experiment_id"):
            raise ValueError(f"saved item {row.item_id!r} has incompatible experiment provenance")


class RowWriter:
    """Append one complete JSONL observation and fsync it before returning."""

    def __init__(self, path: Path | str, *, resume: bool = False, overwrite: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a" if resume else "w" if overwrite else "x", encoding="utf-8")
        self._digest = hashlib.sha256(self.path.read_bytes())
        if resume and self.path.stat().st_size:
            # A valid final line without a newline must still be separated from the next
            # append. Invalid truncated tails are repaired before opening this writer.
            with self.path.open("rb") as saved:
                saved.seek(-1, os.SEEK_END)
                if saved.read(1) != b"\n":
                    self.stream.write("\n")
                    self._digest.update(b"\n")

    def append(self, row: EvalRow) -> None:
        line = json.dumps(row.as_dict(), ensure_ascii=False, allow_nan=False) + "\n"
        self.stream.write(line)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self._digest.update(line.encode("utf-8"))

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.stream.close()


def metadata_from_rows(rows: Sequence[EvalRow], *, dataset_spec: Optional[str] = None) -> dict:
    """Rebuild provenance from the rows themselves, for replay without a model.

    Returns ``is_simulated=None`` when the rows predate per-row provenance or disagree
    among themselves. Unknown provenance must stay unknown: silently defaulting it to
    "real" is exactly how a synthetic run gets published as a calibration result.
    """
    flags = {r.is_simulated for r in rows}
    answerers = {r.answerer for r in rows if r.answerer}
    models = {r.model for r in rows if r.model}
    experiments = {json.dumps(r.experiment, sort_keys=True) for r in rows}
    versioned = len(experiments) == 1 and bool(rows) and rows[0].experiment.get("schema_version") == SCHEMA_VERSION
    meta = deepcopy(rows[0].experiment) if versioned else {}
    # Caller defaults cannot relabel a stored experiment. Infer known dataset families
    # for old rows and explicitly retain the fact that their full acquisition is unknown.
    return {
        **meta,
        "timestamp_utc": meta.get("timestamp_utc", "unknown (legacy rows)"),
        "analysed_at_utc": utc_now(),
        "answerer": answerers.pop() if len(answerers) == 1 else "unknown",
        "model": models.pop() if len(models) == 1 else "unknown",
        "is_simulated": flags.pop() if len(flags) == 1 else None,
        "dataset_spec": meta.get("dataset_spec", inferred_dataset_spec(rows)),
        "provenance_status": meta.get("provenance_status", "legacy_partial"),
        "confidence_scope": "single_step",
        "similarity": meta.get("similarity", "unknown (legacy rows)"),
        "n_items": len(rows),
        "n_samples_per_item": rows[0].n_samples if rows else 0,
        "replayed_from_rows": True,
        "historical_scores": not versioned,
    }


def run_metadata(
    answerer, items, *, n_samples: int, dataset_spec: Optional[str] = None,
    temperature: Optional[float] = None, seed: Optional[int] = None,
    limit: Optional[int] = None, keep_samples: bool = False,
) -> dict:
    """Provenance stamped onto every artefact.

    ``is_simulated`` is the important field: it is what stops a pipeline smoke-test run
    from being read later as a calibration result for a real model.
    """
    return make_experiment(
        answerer, items, n_samples=n_samples, dataset_spec=dataset_spec,
        temperature=temperature, seed=seed, limit=limit, keep_samples=keep_samples,
        similarity=similarity_backend(),
    )


__all__ = [
    "EvalRow",
    "run_harness",
    "write_rows",
    "read_rows",
    "run_metadata",
    "metadata_from_rows",
    "outcome_label",
    "SIGNAL_KEYS",
]
