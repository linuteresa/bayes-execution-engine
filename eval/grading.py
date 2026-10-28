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

import math
import re
import string
import unicodedata
from typing import Iterable, Optional, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = {ord(c): " " for c in string.punctuation}
# Matches 1234, 1,234.5, -12, .5 -- trailing '.' is left to normalization.
_DECIMAL = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUMBER = re.compile(rf"({_DECIMAL})(?:\s*/\s*({_DECIMAL}))?")
_FINAL = re.compile(
    r"(?im)\b(?:final\s+answer|the\s+answer|answer)\s*(?:is\b|:|=)\s*"
)
_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_LATEX_FRACTION = re.compile(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_UNCERTAIN = re.compile(r"\b(?:or|either|maybe|perhaps|not sure|don't know|do not know)\b", re.I)
_NEGATED = re.compile(
    r"\b(?:not|isn't|isn’t|wasn't|wasn’t|wrong|incorrect|rather than|instead of)\b", re.I
)
GRADING_VERSION = 2


def final_answer_span(prediction: str) -> str:
    """Prefer the final explicitly labelled answer over numbers/names in reasoning.

    This is a deterministic safeguard, not semantic judging. Free-form containment
    remains an approximate grader; structured final answers are the preferred input.
    """
    text = str(prediction or "").strip()
    boxed = list(_BOXED.finditer(text))
    labelled = list(_FINAL.finditer(text))
    spans = [(m.start(), m.group(1)) for m in boxed]
    spans.extend((m.start(), text[m.end():].split("\n", 1)[0]) for m in labelled)
    if spans:
        return max(spans, key=lambda entry: entry[0])[1].strip()
    return text


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
    candidate = final_answer_span(prediction)
    if _UNCERTAIN.search(candidate):
        return False
    # Reject explicitly negated mentions, rather than rewarding an answer merely for
    # naming an alternative it dismisses. The word "no" itself can be a valid gold.
    clauses = [c for c in re.split(r"[;.!?\n]", candidate) if not _NEGATED.search(c)]
    pred_tokens = normalize_answer(" ; ".join(clauses)).split()
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
    text = str(text or "").replace("−", "-")
    text = _LATEX_FRACTION.sub(r"\1/\2", text)
    for numerator, denominator in _NUMBER.findall(text):
        try:
            value = float(numerator.replace(",", "").rstrip("."))
            if denominator:
                value /= float(denominator.replace(",", "").rstrip("."))
            if math.isfinite(value):
                out.append(value)
        except (ValueError, ZeroDivisionError):
            continue
    return out


def _last_number(text: str) -> Optional[float]:
    text = _LATEX_FRACTION.sub(r"\1/\2", str(text or "").replace("−", "-"))
    tokens = list(_NUMBER.finditer(text))
    if not tokens:
        return None
    # Parse only the final complete numeric token. An invalid final fraction/overflow
    # must not cause an earlier intermediate result to be graded as the answer.
    values = extract_numbers(tokens[-1].group(0))
    return values[-1] if values else None


def numeric_match(prediction: str, golds: Sequence[str], *, rel_tol: float = 1e-6) -> bool:
    """Compare the prediction's *final* number against the gold number.

    Final rather than first: chain-of-thought style answers restate intermediate
    quantities before the result, so the last number is the conventional choice.
    """
    candidate = final_answer_span(prediction)
    if _UNCERTAIN.search(candidate) or _NEGATED.search(candidate):
        return False
    pred = _last_number(candidate)
    if pred is None:
        return False
    for gold in golds:
        target = _last_number(gold)
        if target is None:
            continue
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
    "GRADING_VERSION",
    "final_answer_span",
]
