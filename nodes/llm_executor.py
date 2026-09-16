"""
Self-consistency execution: turn a single plan step into a real answer *and* a
genuine, measured uncertainty signal for the Bayesian conflict resolver.

The problem with a naive LLM executor
-------------------------------------
If you just ask the model to execute a step once, you get an answer but no honest
notion of *how reliable it is* -- and the Bayesian conflict resolver has nothing real
to act on. So this module executes each step by **sampling the model several times**
(temperature > 0) and measuring how much the samples agree. Disagreement among a
model's own samples is a well-established proxy for uncertainty/hallucination
(self-consistency, Wang et al. 2022).

From the samples we derive the three ordinal evidence signals the engine expects --
from *real measurements*, not keywords:

* ``DataQuality``    -- semantic agreement across samples (high agreement = high quality).
* ``TaskStatus``     -- answerability: fraction of samples that aren't refusals/empties.
* ``ToolReliability``-- answer dispersion: how many distinct answers were produced.

The medoid (the sample most similar to all the others) is returned as the consensus
answer. Agreement is measured with dependency-free bag-of-words cosine similarity, so
nothing beyond the standard library + numpy is required; swap in embeddings for a
semantic upgrade without changing the interface.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9]+")
_REFUSAL_MARKERS = (
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "cannot answer",
    "can't answer",
    "unable to",
    "no information",
    "as an ai",
    "i cannot",
)


@dataclass
class ExecutionResult:
    answer: str
    evidence: dict[str, int]
    consistency: float
    samples: list[str] = field(default_factory=list)
    #: The continuous measurements the ordinal ``evidence`` is discretized from.
    #: The agent only needs the five-level bins, but the calibration eval logs the
    #: raw values (they are what the latent/PCA scaling path in ``scaling/`` reduces),
    #: and discarding them would make those numbers unreconstructable after the fact.
    signals: dict[str, float] = field(default_factory=dict)


def _content(resp) -> str:
    text = getattr(resp, "content", None)
    return (text if text is not None else str(resp)).strip()


def _sample_once(model, step: str, goal: str) -> str:
    prompt = (
        "You are executing ONE step of a larger plan.\n"
        f"Overall goal: {goal}\n"
        f"Step to execute: {step}\n"
        "Return only the concise, factual result of this step."
    )
    return _content(model.invoke(prompt))


def _vec(s: str) -> Counter:
    return Counter(_WORD.findall(s.lower()))


def _cosine(a: Counter, b: Counter) -> float:
    """Cosine similarity of two bag-of-words counts, in [0, 1].

    Counts are non-negative, so the true value cannot leave [0, 1] -- but the division
    can land a hair outside it (1.0000000000000002 for identical vectors). Downstream
    that value is treated as a probability, so clamp here rather than let float noise
    escape into the calibration metrics.
    """
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if not na or not nb:
        return 0.0
    return max(0.0, min(1.0, num / (na * nb)))


def _is_degenerate(s: str) -> bool:
    text = s.strip().lower()
    if len(_WORD.findall(text)) < 2:
        return True
    return any(marker in text for marker in _REFUSAL_MARKERS)


def _clip4(x: float) -> int:
    return max(0, min(4, int(round(x))))


def execute_step_with_llm(
    model,
    step: str,
    goal: str = "",
    *,
    n_samples: int | None = None,
) -> ExecutionResult:
    """Execute one step via self-consistency sampling and return answer + evidence."""
    n = n_samples if n_samples is not None else int(os.getenv("EXECUTOR_SAMPLES", "4"))
    n = max(1, n)

    samples = [_sample_once(model, step, goal) for _ in range(n)]
    vecs = [_vec(s) for s in samples]

    # Pairwise semantic agreement.
    sims: list[float] = []
    if n > 1:
        sims = [
            _cosine(vecs[i], vecs[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        consistency = sum(sims) / len(sims) if sims else 1.0
    else:
        consistency = 1.0


    best_idx, best_score = 0, -1.0
    for i in range(n):
        score = sum(_cosine(vecs[i], vecs[j]) for j in range(n) if j != i)
        if score > best_score:
            best_idx, best_score = i, score
    answer = samples[best_idx] if samples else ""


    answerability = sum(1 for s in samples if not _is_degenerate(s)) / n
    normalised = {re.sub(r"\s+", " ", s.strip().lower()) for s in samples}
    distinct_ratio = (len(normalised) - 1) / (n - 1) if n > 1 else 0.0

    evidence = {
        "TaskStatus": _clip4((1.0 - answerability) * 4),
        "DataQuality": _clip4((1.0 - consistency) * 4),
        "ToolReliability": _clip4(distinct_ratio * 4),
    }

    lengths = [len(_WORD.findall(s.lower())) for s in samples]
    mean_len = sum(lengths) / n
    var_len = sum((v - mean_len) ** 2 for v in lengths) / n
    signals = {
        "agreement_mean": float(consistency),
        "agreement_min": float(min(sims)) if n > 1 and sims else 1.0,
        "answerability": float(answerability),
        "distinct_ratio": float(distinct_ratio),
        "length_mean": float(mean_len),
        # Coefficient of variation: scale-free spread in answer length. Verbose,
        # waffling samples vary in length; confident ones converge on one form.
        "length_cv": float(math.sqrt(var_len) / mean_len) if mean_len > 0 else 0.0,
    }

    return ExecutionResult(
        answer=answer,
        evidence=evidence,
        consistency=consistency,
        samples=samples,
        signals=signals,
    )
