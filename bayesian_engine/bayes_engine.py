"""
Dirichlet–Multinomial conjugate Bayesian engine for agentic conflict resolution.

Why this exists
---------------
When the Plan-and-Execute DAG hits conflicting / uncertain tool output, I do NOT
ask the LLM to "guess again". We run a real Bayesian update over a small, fully
interpretable model and pick the action with the highest *posterior-predictive*
confidence, together with a calibrated uncertainty estimate.

The probabilistic model
-----------------------
We model the conditional distribution of a discrete ``Outcome`` given three
observed signals (the "context"):

    TaskStatus, DataQuality, ToolReliability   -- 5 ordinal states each
    Outcome                                    -- 5 ordinal states

For every one of the ``5 * 5 * 5 = 125`` parent contexts, the conditional
``P(Outcome | context)`` is a **Categorical** distribution. Its conjugate prior is
the **Dirichlet** distribution. This is the entire reason Dirichlet is the right
choice (and not, say, a Gaussian or a hand-tuned softmax):

  * Conjugacy: Dirichlet prior + Multinomial likelihood  ->  Dirichlet posterior,
    in closed form. Updating is just ``alpha_posterior = alpha_prior + counts``.
    No gradient steps, no sampling, fully reproducible, O(1) online updates.
  * The Dirichlet lives on the probability simplex, so every posterior-predictive
    column is automatically a valid distribution that sums to 1 -- by construction,
    not by a normalizing hack.
  * The concentration sum ``alpha_0 = sum(alpha)`` IS the model's effective sample
    size, which gives us free, principled uncertainty quantification (credible
    intervals shrink as we observe more data).

This replaces the previous implementation, which generated a *random* CPT on every
call (``np.random.dirichlet(...)`` with no prior and no data) and therefore returned
essentially random answers. Here the CPT is a deterministic function of an
informative prior plus whatever evidence has been observed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

STATE_NAMES: Dict[int, str] = {0: "CERTAIN", 1: "HIGH", 2: "MEDIUM", 3: "LOW", 4: "AMBIGUOUS"}
STATE_MAP: Dict[str, int] = {v: k for k, v in STATE_NAMES.items()}

PARENTS: Tuple[str, str, str] = ("TaskStatus", "DataQuality", "ToolReliability")
N_STATES = 5
N_CONTEXTS = N_STATES ** len(PARENTS)  # 125


def context_index(task: int, data: int, tool: int) -> int:
    """Map a (TaskStatus, DataQuality, ToolReliability) triple to a flat 0..124 index."""
    for name, v in zip(PARENTS, (task, data, tool), strict=False):
        if not 0 <= v < N_STATES:
            raise ValueError(f"{name} state {v} out of range 0..{N_STATES - 1}")
    return task * (N_STATES ** 2) + data * N_STATES + tool


def context_triple(idx: int) -> Tuple[int, int, int]:
    """Inverse of :func:`context_index`."""
    task, rem = divmod(idx, N_STATES ** 2)
    data, tool = divmod(rem, N_STATES)
    return task, data, tool


@dataclass
class PosteriorSummary:
    """Result of a conflict-resolution query."""

    state: str
    state_index: int
    confidence: float
    distribution: Dict[str, float]
    credible_low: float
    credible_high: float
    effective_sample_size: float

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "state_index": self.state_index,
            "confidence": self.confidence,
            "distribution": self.distribution,
            "credible_interval": [self.credible_low, self.credible_high],
            "effective_sample_size": self.effective_sample_size,
        }


@dataclass
class DirichletBayesianEngine:
    """A conjugate Dirichlet–Multinomial Bayesian network over 125 contexts.

    Parameters
    ----------
    prior_strength:
        How strongly the informative prior pulls each context's outcome toward its
        ordinal "expected" state. ``prior_strength=0`` recovers a flat (uninformative)
        Jeffreys-style prior. Larger values encode stronger domain belief.
    prior_spread:
        Std-dev (in outcome-index units) of the Gaussian bump used to build the
        informative prior. Controls how peaked each context's prior is.
    """

    prior_strength: float = 8.0
    prior_spread: float = 1.0
    # alpha has shape (125, 5): a Dirichlet concentration vector per context.
    alpha: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.alpha = self._build_informative_prior()
        self._prior = self.alpha.copy()  # keep a copy so we can reset / inspect.

    # ------------------------------------------------------------------ priors
    def _build_informative_prior(self) -> np.ndarray:
        """Construct a monotonic, ordinal-aware Dirichlet prior for all 125 contexts.

        Intuition: state index 0 = CERTAIN (best), 4 = AMBIGUOUS (worst). The more
        degraded the three input signals are, the more prior mass the Outcome should
        place on degraded states. We encode this as a Gaussian bump centred at the
        normalized aggregate degradation of the three parents.
        """
        alpha = np.empty((N_CONTEXTS, N_STATES), dtype=float)
        outcomes = np.arange(N_STATES)
        max_degradation = (N_STATES - 1) * len(PARENTS)  # 12
        for idx in range(N_CONTEXTS):
            task, data, tool = context_triple(idx)
            degradation = (task + data + tool) / max_degradation  # 0..1
            center = degradation * (N_STATES - 1)                 # 0..4
            bump = np.exp(-0.5 * ((outcomes - center) / self.prior_spread) ** 2)
            bump /= bump.sum()
            # base 1.0 => a proper prior with at least uniform support everywhere.
            alpha[idx] = 1.0 + self.prior_strength * bump
        return alpha

    # ------------------------------------------------------------- conjugacy
    def observe(self, task: int, data: int, tool: int, outcome: int, count: int = 1) -> None:
        """Online conjugate update: ``alpha_posterior = alpha_prior + counts``."""
        if not 0 <= outcome < N_STATES:
            raise ValueError(f"outcome {outcome} out of range")
        self.alpha[context_index(task, data, tool), outcome] += count

    def fit(self, samples: Iterable[Tuple[int, int, int, int]]) -> "DirichletBayesianEngine":
        """Batch conjugate update from ``(task, data, tool, outcome)`` observations."""
        for task, data, tool, outcome in samples:
            self.observe(task, data, tool, outcome)
        return self

    def reset(self) -> None:
        """Discard observed data, returning to the pure prior."""
        self.alpha = self._prior.copy()

    # --------------------------------------------------------- predictive CPT
    def cpt(self) -> np.ndarray:
        """Posterior-predictive CPT, shape ``(5_outcome, 125_context)``.

        Column = E[theta | data] = alpha / alpha_0. Each column sums to 1 by
        construction (a property of the Dirichlet mean, not a renormalization).
        """
        return (self.alpha / self.alpha.sum(axis=1, keepdims=True)).T

    def _predictive_for_context(self, idx: int) -> np.ndarray:
        a = self.alpha[idx]
        return a / a.sum()

    def _marginal_predictive(self, evidence: Dict[str, int]) -> Tuple[np.ndarray, float]:
        """Posterior-predictive over Outcome given *full or partial* evidence.

        Missing parents are marginalized out under a uniform parent prior. This is
        what makes the network structure (rather than a flat lookup table) earn its
        keep: we can still answer when a signal is unobserved.
        """
        fixed = {name: evidence[name] for name in PARENTS if name in evidence}
        for name, v in fixed.items():
            if not 0 <= v < N_STATES:
                raise ValueError(f"{name} evidence {v} out of range")

        probs = np.zeros(N_STATES)
        ess_acc = 0.0
        n_contexts = 0
        ranges = [
            [fixed[name]] if name in fixed else range(N_STATES) for name in PARENTS
        ]
        for task in ranges[0]:
            for data in ranges[1]:
                for tool in ranges[2]:
                    idx = context_index(task, data, tool)
                    probs += self._predictive_for_context(idx)
                    ess_acc += self.alpha[idx].sum()
                    n_contexts += 1
        probs /= n_contexts
        return probs, ess_acc / n_contexts

    # --------------------------------------------------------------- queries
    def resolve(self, evidence: Dict[str, int]) -> PosteriorSummary:
        """Resolve a conflict: return the MAP outcome plus calibrated uncertainty."""
        probs, ess = self._marginal_predictive(evidence)
        map_idx = int(np.argmax(probs))
        map_p = float(probs[map_idx])

        # 95% credible interval for the MAP probability under the Dirichlet posterior
        # (Beta marginal: theta_k ~ Beta(alpha_k, alpha_0 - alpha_k)).
        low, high = self._beta_credible_interval(evidence, map_idx)

        return PosteriorSummary(
            state=STATE_NAMES[map_idx],
            state_index=map_idx,
            confidence=map_p,
            distribution={STATE_NAMES[i]: float(p) for i, p in enumerate(probs)},
            credible_low=low,
            credible_high=high,
            effective_sample_size=float(ess),
        )

    def _beta_credible_interval(
        self, evidence: Dict[str, int], outcome_idx: int, mass: float = 0.95
    ) -> Tuple[float, float]:

        ranges = [
            [evidence[name]] if name in evidence else range(N_STATES) for name in PARENTS
        ]
        a_k, a_rest, n = 0.0, 0.0, 0
        for task in ranges[0]:
            for data in ranges[1]:
                for tool in ranges[2]:
                    a = self.alpha[context_index(task, data, tool)]
                    a_k += a[outcome_idx]
                    a_rest += a.sum() - a[outcome_idx]
                    n += 1
        a_k /= n
        a_rest /= n
        try:
            from scipy.stats import beta  # optional dependency
            tail = (1.0 - mass) / 2.0
            return float(beta.ppf(tail, a_k, a_rest)), float(beta.ppf(1 - tail, a_k, a_rest))
        except Exception:

            a0 = a_k + a_rest
            mean = a_k / a0
            var = a_k * a_rest / (a0 ** 2 * (a0 + 1))
            sd = float(np.sqrt(var))
            return max(0.0, mean - 1.96 * sd), min(1.0, mean + 1.96 * sd)

    # ----------------------------------------------------- optional pgmpy view
    def build_network(self):
        """Return an equivalent ``pgmpy`` DiscreteBayesianNetwork.

        Useful for visualization and for exact inference with *arbitrary* query/
        evidence patterns. Imported lazily so pgmpy stays an optional dependency.
        """
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.models import DiscreteBayesianNetwork

        model = DiscreteBayesianNetwork([(p, "Outcome") for p in PARENTS])
        uniform = [[1.0 / N_STATES]] * N_STATES
        cpds = [TabularCPD(p, N_STATES, uniform) for p in PARENTS]
        cpds.append(
            TabularCPD(
                "Outcome",
                N_STATES,
                self.cpt(),
                evidence=list(PARENTS),
                evidence_card=[N_STATES] * len(PARENTS),
            )
        )
        model.add_cpds(*cpds)
        assert model.check_model()
        return model


# --------------------------------------------------------------------------
# Module-level convenience API (deterministic, backwards compatible).
# --------------------------------------------------------------------------
_DEFAULT_ENGINE: Optional[DirichletBayesianEngine] = None


def get_default_engine() -> DirichletBayesianEngine:
    """Lazily build a shared engine with the informative prior (no randomness)."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = DirichletBayesianEngine()
    return _DEFAULT_ENGINE


def build_bayesian_network():
    """Backwards-compatible helper: returns ``(pgmpy_model, outcome_cpd)``."""
    engine = get_default_engine()
    model = engine.build_network()
    outcome_cpd = model.get_cpds("Outcome")
    return model, outcome_cpd


def resolve_conflict(evidence: Dict[str, int]) -> dict:
    """Resolve conflicting/ambiguous evidence into a confident outcome.

    ``evidence`` maps any subset of {TaskStatus, DataQuality, ToolReliability} to a
    state index 0..4. Returns a dict with the MAP state, confidence, full
    distribution, a 95% credible interval, and the effective sample size.
    """
    return get_default_engine().resolve(evidence).as_dict()


__all__ = [
    "DirichletBayesianEngine",
    "PosteriorSummary",
    "resolve_conflict",
    "build_bayesian_network",
    "get_default_engine",
    "context_index",
    "context_triple",
    "STATE_NAMES",
    "STATE_MAP",
    "PARENTS",
    "N_STATES",
    "N_CONTEXTS",
]
