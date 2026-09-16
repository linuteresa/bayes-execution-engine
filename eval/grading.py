"""Grade a free-form model answer against gold answers.

The engine returns prose ("The capital of France is Paris."), while gold answers are
short strings ("Paris"). Two graders cover the question types the harness uses:

* ``contains`` -- normalized alias containment, the standard generative-QA criterion
  for short-answer sets like TriviaQA/SimpleQA.
* ``numeric``  -- pull the final number out of the answer and compare with tolerance,
  the standard criterion for GSM8K-style arithmetic.

Grading is deliberately strict-but-dumb: no model-based judging. A grader that is
itself an LLM would make the calibration numbers depend on a second uncalibrated
system, which is exactly the circularity this eval exists to avoid.
"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Iterable, Optional, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = {ord(c): " " for c in string.punctuation}
# Matches 1234, 1,234.5, -12, .5 -- trailing '.' is left to normalization.
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")


def normalize_answer(text: str) -> str:
    """Lowercase, strip accents/punctuation/articles, collapse whitespace.

    The standard SQuAD/TriviaQA normalization, so scores here are comparable with
    published numbers on the same sets.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def exact_match(prediction: str, golds: Sequence[str]) -> bool:
    """True when the whole normalized prediction equals a normalized gold alias."""
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(g) for g in golds if str(g).strip())


def contains_match(prediction: str, golds: Sequence[str]) -> bool:
    """True when a normalized gold alias appears as a whole-token span in the answer.

    Token-boundary matching, not raw substring: raw ``in`` would score "Austria" as a
    hit for gold "Austr" and, worse, mark a gold of "no" correct inside "nothing".
    """
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return False
    for gold in golds:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        span = len(gold_tokens)
        for start in range(len(pred_tokens) - span + 1):
            if pred_tokens[start : start + span] == gold_tokens:
                return True
    return False


def extract_numbers(text: str) -> list[float]:
    """All numbers in ``text``, comma separators removed, in order of appearance."""
    out: list[float] = []
    for raw in _NUMBER.findall(str(text or "")):
        cleaned = raw.replace(",", "").rstrip(".")
        if cleaned in ("", "-", "."):
            continue
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def numeric_match(prediction: str, golds: Sequence[str], *, rel_tol: float = 1e-6) -> bool:
    """Compare the prediction's *final* number against the gold number.

    Final rather than first: chain-of-thought style answers restate intermediate
    quantities before the result, so the last number is the conventional choice.
    """
    pred_numbers = extract_numbers(prediction)
    if not pred_numbers:
        return False
    pred = pred_numbers[-1]
    for gold in golds:
        gold_numbers = extract_numbers(gold)
        if not gold_numbers:
            continue
        target = gold_numbers[-1]
        tolerance = max(rel_tol, abs(target) * rel_tol)
        if abs(pred - target) <= tolerance:
            return True
    return False


GRADERS = {
    "contains": contains_match,
    "exact": exact_match,
    "numeric": numeric_match,
}


def grade(prediction: str, golds: Iterable[str], grader: Optional[str] = None) -> bool:
    """Grade ``prediction`` with the named grader (default ``contains``)."""
    name = grader or "contains"
    if name not in GRADERS:
        raise ValueError(f"unknown grader {name!r}; choose from {sorted(GRADERS)}")
    return bool(GRADERS[name](prediction, list(golds)))


__all__ = [
    "normalize_answer",
    "exact_match",
    "contains_match",
    "numeric_match",
    "extract_numbers",
    "grade",
    "GRADERS",
]
