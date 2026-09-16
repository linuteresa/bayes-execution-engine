"""Question sets with verifiable gold answers.

A calibration eval is only meaningful on items the system sometimes gets **wrong**. On
an all-easy set every confidence is high and every answer correct, AUROC is undefined
and the reliability diagram has one point. So every loader here aims for a difficulty
spread, and :func:`describe` reports that spread so a degenerate run is visible in the
report rather than silently flattering.

Specs accepted by :func:`load_items`:

===========================  ==============================================================
``bundled``                  80 hand-checked items shipped in ``eval/data`` (easy -> hard
                             trivia + 20 arithmetic word problems). Runs offline; big
                             enough to smoke-test, small enough to commit.
``jsonl:<path>``             Your own file, one JSON object per line.
``hf:triviaqa``              TriviaQA ``rc.nocontext`` validation split.
``hf:gsm8k``                 GSM8K ``main`` test split (grader: numeric).
``hf:simpleqa``              SimpleQA test split (community mirror).
===========================  ==============================================================

The ``hf:`` loaders need ``pip install datasets`` and one network download; the headline
300-500 item run in the README uses them. ``bundled`` needs neither.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DATA_DIR = Path(__file__).parent / "data"
BUNDLED_PATH = DATA_DIR / "dev_questions.jsonl"


@dataclass
class EvalItem:
    """One gradeable question."""

    id: str
    question: str
    answers: List[str]
    grader: str = "contains"
    difficulty: str = "unknown"
    source: str = "custom"
    metadata: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _item_from_obj(obj: dict, index: int, source: str) -> EvalItem:
    question = obj.get("question") or obj.get("problem") or obj.get("prompt")
    if not question:
        raise ValueError(f"item {index} in {source} has no 'question' field")
    answers = obj.get("answers")
    if answers is None:
        single = obj.get("answer")
        answers = [single] if single is not None else []
    if isinstance(answers, str):
        answers = [answers]
    answers = [str(a) for a in answers if str(a).strip()]
    if not answers:
        raise ValueError(f"item {index} in {source} has no gold answer")
    return EvalItem(
        id=str(obj.get("id") or f"{source}-{index:04d}"),
        question=str(question),
        answers=answers,
        grader=str(obj.get("grader") or "contains"),
        difficulty=str(obj.get("difficulty") or "unknown"),
        source=str(obj.get("source") or source),
        metadata=dict(obj.get("metadata") or {}),
    )


def load_jsonl(path: Path | str) -> List[EvalItem]:
    """Load items from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"question file not found: {path}")
    items: List[EvalItem] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            items.append(_item_from_obj(json.loads(line), i, path.stem))
    if not items:
        raise ValueError(f"no items in {path}")
    return items


def load_bundled() -> List[EvalItem]:
    """The committed offline dev set."""
    return load_jsonl(BUNDLED_PATH)


# ------------------------------------------------------------------ HuggingFace
def _require_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "hf: datasets need `pip install datasets` (see requirements-eval.txt)"
        ) from exc
    return datasets


def _load_first_available(candidates, *, split: str):
    """Load the first repo id that resolves, reporting every failure if none do.

    Dataset ids get renamed and namespaced over time, and a bare alias that works on one
    `datasets` version raises on another. Falling through a short candidate list keeps
    the eval runnable across versions instead of failing on a stale identifier.
    """
    datasets_mod = _require_datasets()
    errors = []
    for repo_id, config in candidates:
        try:
            if config is None:
                return datasets_mod.load_dataset(repo_id, split=split)
            return datasets_mod.load_dataset(repo_id, config, split=split)
        except Exception as exc:  # noqa: BLE001 - try the next id, report all at the end
            errors.append(f"  {repo_id}: {exc.__class__.__name__}: {exc}")
    raise RuntimeError(
        "could not load any candidate dataset id:\n" + "\n".join(errors)
    )


def _load_triviaqa(limit: Optional[int]) -> List[EvalItem]:
    ds = _load_first_available(
        [("mandarjoshi/trivia_qa", "rc.nocontext"), ("trivia_qa", "rc.nocontext")],
        split="validation",
    )
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    items = []
    for i, row in enumerate(ds):
        answer = row.get("answer") or {}
        aliases = list(answer.get("normalized_aliases") or answer.get("aliases") or [])
        value = answer.get("value")
        if value:
            aliases.insert(0, value)
        if not aliases:
            continue
        # Long aliases are mostly noise and inflate false positives under containment
        # grading; keep the short, answer-shaped ones.
        answers = [a for a in dict.fromkeys(aliases) if len(str(a)) <= 60][:20]
        if not answers:
            # Every alias was filtered out. Keeping the item would give it no gold
            # answer at all, so the grader would mark every prediction wrong and the
            # item would enter the eval as a guaranteed false negative -- contaminating
            # the very labels the calibration numbers are computed against.
            continue
        items.append(
            EvalItem(
                id=str(row.get("question_id") or f"triviaqa-{i:04d}"),
                question=str(row["question"]),
                answers=answers,
                grader="contains",
                difficulty="mixed",
                source="triviaqa",
            )
        )
    return items


def _load_gsm8k(limit: Optional[int]) -> List[EvalItem]:
    # `datasets` >= 5 requires namespaced repo ids, so the bare "gsm8k" alias that older
    # docs and tutorials use now fails outright. Try the canonical id first and keep the
    # legacy ones as fallbacks for older `datasets` installs.
    ds = _load_first_available(
        [("openai/gsm8k", "main"), ("gsm8k", "main")], split="test"
    )
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    items = []
    for i, row in enumerate(ds):
        # GSM8K solutions end with "#### <final answer>".
        final = str(row["answer"]).split("####")[-1].strip()
        if not final:
            continue
        items.append(
            EvalItem(
                id=f"gsm8k-{i:04d}",
                question=str(row["question"]),
                answers=[final],
                grader="numeric",
                difficulty="arithmetic",
                source="gsm8k",
            )
        )
    return items


def _load_simpleqa(limit: Optional[int]) -> List[EvalItem]:
    ds = _load_first_available([("basicv8vc/SimpleQA", None)], split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [
        EvalItem(
            id=f"simpleqa-{i:04d}",
            question=str(row["problem"]),
            answers=[str(row["answer"])],
            grader="contains",
            difficulty="hard",
            source="simpleqa",
        )
        for i, row in enumerate(ds)
    ]


_HF_LOADERS = {
    "triviaqa": _load_triviaqa,
    "gsm8k": _load_gsm8k,
    "simpleqa": _load_simpleqa,
}


# ----------------------------------------------------------------------- public
def load_items(
    spec: str = "bundled",
    *,
    limit: Optional[int] = None,
    seed: int = 0,
    shuffle: bool = True,
) -> List[EvalItem]:
    """Resolve a dataset spec into items.

    Multiple sets can be mixed with ``+`` (``hf:triviaqa+hf:gsm8k``), which is how you
    build the difficulty spread the metrics need. ``limit`` applies per component, so
    ``--limit 250`` over two sets yields ~500 items.
    """
    parts = [p.strip() for p in str(spec).split("+") if p.strip()]
    if not parts:
        raise ValueError("empty dataset spec")

    items: List[EvalItem] = []
    for part in parts:
        if part == "bundled":
            items.extend(load_bundled())
        elif part.startswith("jsonl:"):
            items.extend(load_jsonl(part[len("jsonl:") :]))
        elif part.startswith("hf:"):
            name = part[len("hf:") :]
            if name not in _HF_LOADERS:
                raise ValueError(f"unknown hf dataset {name!r}; try {sorted(_HF_LOADERS)}")
            # Over-fetch then subsample so the slice isn't just the head of the file.
            items.extend(_HF_LOADERS[name](limit * 4 if limit else None))
        else:
            raise ValueError(
                f"unknown dataset spec {part!r}; use bundled | jsonl:<path> | hf:<name>"
            )

    if shuffle:
        random.Random(seed).shuffle(items)
    if limit is not None and len(parts) == 1:
        items = items[:limit]
    elif limit is not None:
        # Keep `limit` per component so a mixed spec stays balanced.
        per_source: Dict[str, int] = {}
        kept = []
        for it in items:
            n = per_source.get(it.source, 0)
            if n < limit:
                per_source[it.source] = n + 1
                kept.append(it)
        items = kept
    if not items:
        raise ValueError(f"dataset spec {spec!r} produced no items")
    return items


def describe(items: Sequence[EvalItem]) -> Dict[str, object]:
    """Summarise composition so the report can show what was actually evaluated."""
    by_difficulty: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    for it in items:
        by_difficulty[it.difficulty] = by_difficulty.get(it.difficulty, 0) + 1
        by_source[it.source] = by_source.get(it.source, 0) + 1
    return {
        "n_items": len(items),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "by_source": dict(sorted(by_source.items())),
    }


__all__ = ["EvalItem", "load_items", "load_jsonl", "load_bundled", "describe", "BUNDLED_PATH"]
