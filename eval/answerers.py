"""Sample sources for the eval harness.

The harness drives the *real* execution path (``nodes.llm_executor.execute_step_with_llm``),
which needs an object exposing ``.invoke(prompt)``. Two are provided:

``LlamaAnswerer``
    The production path: the local llama.cpp server via ``core.llm.build_llm`` at the
    executor's sampling temperature. It also captures per-response mean token logprob,
    which is one of the two baselines the engine has to beat.

``SimulatedAnswerer``
    A deterministic stand-in that fabricates samples from the item's own gold answer.
    It exists so the harness, metrics and figures are testable in CI with no model, and
    so a contributor can run the whole pipeline end to end offline.

    Numbers produced with ``SimulatedAnswerer`` are **not** evidence about any model.
    The simulator is told the right answer, so its confidence/correctness relationship
    is one this module wrote by hand. Every artefact it produces is stamped
    ``"is_simulated": true`` and the report refuses to present it as a calibration
    result. Only ``LlamaAnswerer`` runs count.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional

from eval.datasets import EvalItem


class _Msg:
    """Minimal message object matching what ``llm_executor`` reads."""

    def __init__(self, content: str):
        self.content = content


def _mean_logprob(response) -> Optional[float]:
    """Mean token logprob from an OpenAI-compatible response, if the server sent one.

    llama-server returns logprobs only when asked and only on some builds, so this is
    best-effort: ``None`` means the baseline is unavailable, not that it is zero.
    """
    meta = getattr(response, "response_metadata", None) or {}
    payload = meta.get("logprobs") or {}
    tokens = payload.get("content") or []
    values = [
        t["logprob"]
        for t in tokens
        if isinstance(t, dict) and isinstance(t.get("logprob"), (int, float))
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


# --------------------------------------------------------------------------- live
@dataclass
class LlamaAnswerer:
    """Samples from the local llama.cpp server -- the real, reportable path."""

    temperature: float = 0.7
    request_logprobs: bool = True
    name: str = "llama.cpp"
    is_simulated: bool = False

    def __post_init__(self) -> None:
        from core.llm import build_llm

        self._model = build_llm(
            temperature=self.temperature, logprobs=self.request_logprobs
        )
        self._logprobs: List[float] = []
        self.model_name = os.getenv("LLAMA_MODEL", "local-gguf")

    def sampler_for(self, item: EvalItem, n_samples: int = 4) -> "LlamaAnswerer":
        self._logprobs = []
        return self

    def invoke(self, prompt: str):
        response = self._model.invoke(prompt)
        lp = _mean_logprob(response)
        if lp is not None:
            self._logprobs.append(lp)
        return response

    def mean_logprob(self) -> Optional[float]:
        if not self._logprobs:
            return None
        return float(sum(self._logprobs) / len(self._logprobs))


# ---------------------------------------------------------------------- simulated
_WRONG_POOL = (
    "The records are inconclusive on this point.",
    "Approximately four hundred and twelve.",
    "It was first described in the eighteenth century.",
    "Somewhere in the southern hemisphere, most likely.",
    "The Treaty of Utrecht settled the matter in 1713.",
    "Roughly seventeen per cent, by most estimates.",
    "That honour belongs to the city of Valparaiso.",
    "The value is close to nine thousand and eighty.",
)

_PARAPHRASES = (
    "The answer is {a}.",
    "It is {a}.",
    "Based on the step, the result is {a}.",
    "{a} is the correct result here.",
)


@dataclass
class SimulatedAnswerer:
    """Deterministic fake model for offline pipeline tests. NOT a calibration result.

    Behaviour is a seeded function of the item id, so runs are reproducible. Items are
    sorted into three bands by a hash and their declared difficulty:

    * *knows*        -- every sample paraphrases the gold answer (high agreement, correct)
    * *partly knows* -- some samples right, some wrong (middling agreement)
    * *guesses*      -- every sample is a different wrong string (low agreement, wrong)

    That is a hand-written confidence/correctness relationship. It proves the plumbing
    computes ECE/AUROC/risk-coverage correctly; it proves nothing about a real model.
    """

    seed: int = 0
    name: str = "simulated"
    is_simulated: bool = True
    model_name: str = "simulated-answerer"

    def __post_init__(self) -> None:
        self._queue: List[str] = []
        self._logprob: Optional[float] = None

    # Difficulty -> probability the simulator "knows" the answer.
    _SKILL = {
        "easy": 0.92,
        "medium": 0.72,
        "arithmetic": 0.55,
        "mixed": 0.50,
        "hard": 0.28,
        "unknown": 0.50,
    }

    def _uniform(self, item: EvalItem, salt: str) -> float:
        digest = hashlib.sha256(f"{self.seed}:{item.id}:{salt}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    def sampler_for(self, item: EvalItem, n_samples: int = 4) -> "SimulatedAnswerer":
        gold = item.answers[0]
        skill = self._SKILL.get(item.difficulty, 0.5)
        roll = self._uniform(item, "band")

        samples: List[str] = []
        if roll < skill:                       # knows it: consistent + correct
            for k in range(n_samples):
                samples.append(_PARAPHRASES[k % len(_PARAPHRASES)].format(a=gold))
        elif roll < skill + 0.18:              # partly: mixed evidence
            for k in range(n_samples):
                if self._uniform(item, f"mix{k}") < 0.5:
                    samples.append(_PARAPHRASES[k % len(_PARAPHRASES)].format(a=gold))
                else:
                    pool_i = int(self._uniform(item, f"w{k}") * len(_WRONG_POOL))
                    samples.append(_WRONG_POOL[pool_i % len(_WRONG_POOL)])
        else:                                  # guesses: scattered + wrong
            for k in range(n_samples):
                pool_i = int(self._uniform(item, f"g{k}") * len(_WRONG_POOL))
                samples.append(_WRONG_POOL[(pool_i + k) % len(_WRONG_POOL)])

        self._queue = samples
        # A plausible stand-in logprob: more confident when the samples agree.
        self._logprob = -0.25 - 1.6 * self._uniform(item, "lp") * (0.3 if roll < skill else 1.0)
        return self

    def invoke(self, prompt: str) -> _Msg:
        if not self._queue:
            return _Msg("No further information is available.")
        return _Msg(self._queue.pop(0))

    def mean_logprob(self) -> Optional[float]:
        return self._logprob


def build_answerer(kind: str, *, temperature: float = 0.7, seed: int = 0):
    """Factory: ``llama`` for the real path, ``sim`` for the offline stand-in."""
    if kind in ("llama", "llama.cpp", "live", "real"):
        return LlamaAnswerer(temperature=temperature)
    if kind in ("sim", "simulated", "mock"):
        return SimulatedAnswerer(seed=seed)
    raise ValueError(f"unknown answerer {kind!r}; use 'llama' or 'sim'")


def is_simulated(answerer) -> bool:
    return bool(getattr(answerer, "is_simulated", False))


__all__ = [
    "LlamaAnswerer",
    "SimulatedAnswerer",
    "build_answerer",
    "is_simulated",
]
