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
answer. Agreement is bag-of-words cosine by default (no extra service needed), or
embedding cosine when ``EMBEDDINGS_BASE_URL`` points at an embedding server -- see
``_similarity_matrix``. Both feed the same evidence computation.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from core.llm import build_embeddings

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
    "don't have access",
    "do not have access",
)
#: Two samples at least this similar are the same answer, reworded. Word overlap and
#: embeddings need different cut-offs: on MiniLM, true rewordings scored 0.95-0.97 but a
#: contradiction ("Sydney" vs "Canberra") scored 0.88, so 0.8 would merge them.
_SAME_ANSWER = 0.8
_SAME_ANSWER_EMBEDDINGS = 0.9


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
        "Return only the concise, factual result of this step.\n"
        "You have no access to any database, file or live system. If the step needs data "
        "you do not have, say so plainly. Never invent names, IDs, numbers or records."
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


def similarity_backend() -> str:
    """Which agreement measure is active, for provenance: runs on different ones differ."""
    if os.getenv("EMBEDDINGS_BASE_URL"):
        return f"embeddings:{os.getenv('EMBEDDINGS_MODEL', 'local-embeddings')}"
    return "word-overlap"


def _similarity_matrix(samples: list[str], vecs: list[Counter]) -> tuple[list[list[float]], float]:
    """Pairwise similarity in [0, 1], plus the "same answer" cut-off for that measure.

    Word overlap by default. With ``EMBEDDINGS_BASE_URL`` set, cosine of embeddings, so
    "Yes." and "Correct, that's right." agree although they share no words.
    """
    n = len(samples)
    embedder = build_embeddings()
    if embedder is None:
        return [[_cosine(vecs[i], vecs[j]) for j in range(n)] for i in range(n)], _SAME_ANSWER

    u = np.array(embedder.embed_documents([s or " " for s in samples]), dtype=float)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    sim = np.clip(u @ u.T, 0.0, 1.0)  # unrelated text can dip below 0
    nonempty = np.array([bool(s) for s in samples])
    sim *= np.outer(nonempty, nonempty)  # an empty sample agrees with nothing, as with words
    return sim.tolist(), _SAME_ANSWER_EMBEDDINGS


def _is_degenerate(s: str) -> bool:
    text = s.strip().lower()
    if len(_WORD.findall(text)) < 2:
        return True
    return any(marker in text for marker in _REFUSAL_MARKERS)


def _count_distinct(vecs: list[Counter], sim: list[list[float]], same_answer: float) -> int:
    """Number of different answers, grouping rewordings of the same one.

    Counting exact strings made every long sample "distinct" at temperature > 0, so
    ToolReliability saturated at its worst value and carried no information.

    Word overlap can't see that "3 users" and "12 users" disagree, so two samples are
    only the same answer if they also state the same numbers.
    """
    reps: list[int] = []
    for i, v in enumerate(vecs):
        if not any(sim[i][r] >= same_answer and _numbers(v) == _numbers(vecs[r]) for r in reps):
            reps.append(i)
    return len(reps)


def _numbers(v: Counter) -> set[str]:
    return {t for t in v if t.isdigit()}


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
    sim, same_answer = _similarity_matrix(samples, vecs)

    # Pairwise agreement.
    sims: list[float] = []
    if n > 1:
        sims = [sim[i][j] for i in range(n) for j in range(i + 1, n)]
        consistency = sum(sims) / len(sims) if sims else 1.0
    else:
        consistency = 1.0


    best_idx, best_score = 0, -1.0
    for i in range(n):
        score = sum(sim[i][j] for j in range(n) if j != i)
        if score > best_score:
            best_idx, best_score = i, score
    answer = samples[best_idx] if samples else ""


    answerability = sum(1 for s in samples if not _is_degenerate(s)) / n
    distinct = _count_distinct(vecs, sim, same_answer)
    distinct_ratio = (distinct - 1) / (n - 1) if n > 1 else 0.0

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
