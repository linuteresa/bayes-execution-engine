"""
Side-by-side confidence demo.

Shows the whole point of the engine in one screen: a question the model answers
*consistently* gets HIGH confidence, while one where the model's own samples *disagree*
gets LOW confidence and a flagged conflict -- all from real self-consistency
measurement, not keywords.

Usage:
    python demo.py            # uses your local llama.cpp model if reachable,
                              # otherwise falls back to a deterministic simulation
    python demo.py --sim      # force the simulation (no model needed)

With a live model, start llama-server first (see QUICKSTART.md).
"""

from __future__ import annotations

import random
import sys

from bayesian_engine.bayes_engine import resolve_conflict
from nodes.llm_executor import execute_step_with_llm

CONFIDENT_PROMPT = "What is the capital of France?"
AMBIGUOUS_PROMPT = "What will the exact price of Bitcoin be next Tuesday?"


def good_probability(summary: dict) -> float:
    """Confidence = posterior probability the answer is high quality: P(CERTAIN)+P(HIGH)."""
    dist = summary["distribution"]
    return dist["CERTAIN"] + dist["HIGH"]


# --------------------------------------------------------------------------- live
def build_live_model():
    """Return a real llama.cpp-backed sampler, or None if it can't be built."""
    try:
        from core.llm import build_llm

        return build_llm(temperature=0.7)
    except Exception:
        return None


# ---------------------------------------------------------------------- simulation
class _Msg:
    def __init__(self, content: str):
        self.content = content


class SimModel:
    """Deterministic-ish stand-in: consistent on the known prompt, scattered on the
    speculative one. Lets the demo run with no model server."""

    _SCATTER = [
        "Around 50,000 US dollars.",
        "Probably close to 80k.",
        "It is impossible to predict precisely.",
        "Maybe somewhere near 42,000.",
        "Could be 100k or higher, hard to say.",
    ]

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def invoke(self, prompt: str) -> _Msg:
        if "capital of france" in prompt.lower():
            return _Msg("The capital of France is Paris.")
        return _Msg(self._rng.choice(self._SCATTER))


# --------------------------------------------------------------------------- report
def evaluate(model, prompt: str) -> dict:
    result = execute_step_with_llm(model, prompt, goal=prompt, n_samples=5)
    summary = resolve_conflict(result.evidence)
    return {
        "answer": result.answer,
        "consistency": result.consistency,
        "evidence": result.evidence,
        "state": summary["state"],
        "confidence": good_probability(summary),
        "conflict": summary["state_index"] >= 2,
    }


def print_row(label: str, prompt: str, r: dict) -> None:
    print(f"\n  {label}: {prompt}")
    print(f"    self-consistency : {r['consistency']:.2f}")
    print(f"    evidence         : {r['evidence']}")
    print(f"    bayes outcome    : {r['state']}  (conflict={r['conflict']})")
    print(f"    CONFIDENCE       : {r['confidence']:.2f}")
    print(f"    consensus answer : {r['answer'][:80]}")


def main() -> None:
    force_sim = "--sim" in sys.argv
    model = None if force_sim else build_live_model()
    mode = "LIVE (local llama.cpp model)"

    if model is not None:
        try:
            execute_step_with_llm(model, "ping", goal="ping", n_samples=1)
        except Exception as exc:
            print(f"[demo] Live model unreachable ({exc}); using simulation.\n")
            model = None

    if model is None:
        model = SimModel()
        mode = "SIMULATION (no model server needed)"

    print("=" * 70)
    print(f"CONFIDENCE DEMO  --  mode: {mode}")
    print("=" * 70)

    confident = evaluate(model, CONFIDENT_PROMPT)
    ambiguous = evaluate(model, AMBIGUOUS_PROMPT)
    print_row("CONFIDENT", CONFIDENT_PROMPT, confident)
    print_row("AMBIGUOUS", AMBIGUOUS_PROMPT, ambiguous)

    print("\n" + "-" * 70)
    delta = confident["confidence"] - ambiguous["confidence"]
    print(
        f"Confidence gap: {confident['confidence']:.2f} (confident) "
        f"vs {ambiguous['confidence']:.2f} (ambiguous)  ->  Δ {delta:+.2f}"
    )
    print("The engine assigns higher confidence where the model agrees with itself.")
    print("-" * 70)


if __name__ == "__main__":
    main()
