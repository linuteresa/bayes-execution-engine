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

import os
import tempfile
import threading
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np

STATE_NAMES: Dict[int, str] = {0: "CERTAIN", 1: "HIGH", 2: "MEDIUM", 3: "LOW", 4: "AMBIGUOUS"}
STATE_MAP: Dict[str, int] = {v: k for k, v in STATE_NAMES.items()}

#: Outcome states counted as "good" by the reported confidence, P(CERTAIN) + P(HIGH).
GOOD_STATES: Tuple[int, int] = (0, 1)

PARENTS: Tuple[str, str, str] = ("TaskStatus", "DataQuality", "ToolReliability")
N_STATES = 5
N_CONTEXTS = N_STATES ** len(PARENTS)  # 125


def _index(value: int, size: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in 0..{size - 1}")
    if not 0 <= value < size:
        raise ValueError(f"{name} {value} out of range 0..{size - 1}")
    return int(value)


def _finite_parameter(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def context_index(task: int, data: int, tool: int) -> int:
    """Map a (TaskStatus, DataQuality, ToolReliability) triple to a flat 0..124 index."""
    task, data, tool = (
        _index(v, N_STATES, name) for name, v in zip(PARENTS, (task, data, tool), strict=True)
    )
    return task * (N_STATES ** 2) + data * N_STATES + tool


def context_triple(idx: int) -> Tuple[int, int, int]:
    """Inverse of :func:`context_index`."""
    task, rem = divmod(_index(idx, N_CONTEXTS, "context index"), N_STATES ** 2)
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
    #: ``P(CERTAIN) + P(HIGH)`` -- the scalar the executor and the eval actually report.
    good_confidence: float = 0.0
    #: Credible interval for ``good_confidence``. Distinct from ``credible_low/high``,
    #: which bound the MAP state's probability: those are two different quantities and
    #: pairing a confidence with the wrong one makes the uncertainty meaningless.
    good_credible_low: float = 0.0
    good_credible_high: float = 0.0

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "state_index": self.state_index,
            "confidence": self.confidence,
            "distribution": self.distribution,
            "credible_interval": [self.credible_low, self.credible_high],
            "effective_sample_size": self.effective_sample_size,
            "good_confidence": self.good_confidence,
            "good_credible_interval": [self.good_credible_low, self.good_credible_high],
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
    _lock: object = field(default_factory=threading.RLock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.prior_strength = _finite_parameter(self.prior_strength, "prior_strength")
        self.prior_spread = _finite_parameter(self.prior_spread, "prior_spread", positive=True)
        self.alpha = self._build_informative_prior()
        with np.errstate(over="ignore"):
            valid = np.isfinite(self.alpha).all() and np.isfinite(self.alpha.sum(axis=1)).all()
        if not valid:
            raise ValueError("prior parameters overflow the concentration sum")
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
            # Subtract the nearest squared distance first: very small valid spreads
            # must still produce a normalized bump rather than underflowing to zero.
            distances = (outcomes - center) ** 2
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                log_bump = -0.5 * (distances - distances.min()) / self.prior_spread / self.prior_spread
            log_bump[distances == distances.min()] = 0.0
            bump = np.exp(log_bump)
            bump /= bump.sum()
            # base 1.0 => a proper prior with at least uniform support everywhere.
            alpha[idx] = 1.0 + self.prior_strength * bump
        return alpha

    # ------------------------------------------------------------- conjugacy
    def observe(self, task: int, data: int, tool: int, outcome: int, count: int = 1) -> None:
        """Online conjugate update: ``alpha_posterior = alpha_prior + counts``."""
        idx = context_index(task, data, tool)
        outcome = _index(outcome, N_STATES, "outcome")
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 0:
            raise ValueError("count must be a nonnegative integer")
        try:
            count = float(count)
        except OverflowError as exc:
            raise ValueError("count would overflow the posterior") from exc
        with self._lock:
            with np.errstate(over="ignore"):
                updated = self.alpha[idx, outcome] + count
                valid = np.isfinite(updated) and np.isfinite(self.alpha[idx].sum() + count)
            if not valid:
                raise ValueError("count would overflow the posterior")
            self.alpha[idx, outcome] = updated

    def fit(self, samples: Iterable[Tuple[int, int, int, int]]) -> "DirichletBayesianEngine":
        """Batch conjugate update from ``(task, data, tool, outcome)`` observations."""
        increments = np.zeros_like(self.alpha)
        for task, data, tool, outcome in samples:
            increments[context_index(task, data, tool), _index(outcome, N_STATES, "outcome")] += 1
        with self._lock:
            self.replace_counts(self.observation_counts() + increments)
        return self

    def reset(self) -> None:
        """Discard observed data, returning to the pure prior."""
        with self._lock:
            self.alpha = self._prior.copy()

    # ----------------------------------------------------------- introspection
    def observation_counts(self) -> np.ndarray:
        """Observed counts only, shape ``(125, 5)``: ``alpha_posterior - alpha_prior``.

        Conjugacy makes this exact rather than an estimate -- the prior contributes a
        fixed pseudo-count, so subtracting it recovers the real data. Useful for
        answering the question that decides whether a 125-cell table is the right
        size: how many contexts has this deployment *ever* visited?
        """
        with self._lock:
            return self.alpha - self._prior

    def coverage(self) -> int:
        """How many of the 125 contexts have at least one real observation."""
        return int((self.observation_counts().sum(axis=1) > 0).sum())

    def total_observations(self) -> float:
        """Total real observations across all contexts."""
        return float(self.observation_counts().sum())

    # ------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        """Atomically write a snapshot using a unique temporary file.

        Snapshots do not merge independent writers. Shared deployments must record
        observations transactionally and register their aggregate via set_count_source.
        """
        counts = self.observation_counts()
        path = os.fspath(path)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=os.path.dirname(os.path.abspath(path)),
                prefix=f".{os.path.basename(path)}.", suffix=".tmp", delete=False,
            ) as fh:
                tmp = fh.name
                np.save(fh, counts, allow_pickle=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if tmp is not None and os.path.exists(tmp):
                os.unlink(tmp)

    def load(self, path: str) -> None:
        """Restore counts written by :meth:`save` on top of this engine's prior."""
        self.replace_counts(np.load(path, allow_pickle=False))

    def replace_counts(self, counts: np.ndarray) -> None:
        """Validate and install an authoritative observation snapshot atomically."""
        counts = np.asarray(counts)
        if counts.shape != self._prior.shape:
            raise ValueError(f"saved counts have shape {counts.shape}, expected {self._prior.shape}")
        if counts.dtype.kind not in "iuf" or not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("counts must be finite nonnegative real numbers")
        with np.errstate(over="ignore"):
            posterior = self._prior + counts
            valid = np.isfinite(posterior).all() and np.isfinite(posterior.sum(axis=1)).all()
        if not valid:
            raise ValueError("counts overflow the posterior")
        with self._lock:
            self.alpha = posterior

    # --------------------------------------------------------- predictive CPT
    def cpt(self) -> np.ndarray:
        """Posterior-predictive CPT, shape ``(5_outcome, 125_context)``.

        Column = E[theta | data] = alpha / alpha_0. Each column sums to 1 by
        construction (a property of the Dirichlet mean, not a renormalization).
        """
        with self._lock:
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
        alphas = self._matching_alphas(evidence)
        totals = alphas.sum(axis=1)
        return (alphas / totals[:, None]).mean(axis=0), float(totals.mean())

    def _matching_alphas(self, evidence: Dict[str, int]) -> np.ndarray:
        unknown = set(evidence) - set(PARENTS)
        if unknown:
            raise ValueError(f"unknown evidence names: {sorted(unknown)}")
        fixed = {name: _index(value, N_STATES, name) for name, value in evidence.items()}
        ranges = [
            [fixed[name]] if name in fixed else range(N_STATES) for name in PARENTS
        ]
        indices = [context_index(task, data, tool)
                   for task in ranges[0] for data in ranges[1] for tool in ranges[2]]
        with self._lock:
            return self.alpha[indices].copy()

    # --------------------------------------------------------------- queries
    def resolve(self, evidence: Dict[str, int]) -> PosteriorSummary:
        """Resolve against one consistent posterior snapshot.

        Full-evidence intervals are exact Beta marginals; partial-evidence intervals
        approximate the posterior of the uniform average with reproducibly seeded draws.
        """
        with self._lock:
            return self._resolve(evidence)

    def _resolve(self, evidence: Dict[str, int]) -> PosteriorSummary:
        probs, ess = self._marginal_predictive(evidence)
        map_idx = int(np.argmax(probs))
        map_p = float(probs[map_idx])

        # 95% credible interval for the MAP probability under the Dirichlet posterior
        # (Beta marginal: theta_k ~ Beta(alpha_k, alpha_0 - alpha_k)).
        low, high = self._beta_credible_interval(evidence, map_idx)
        # The reported confidence sums two states, so it needs its own bound.
        good_low, good_high = self._beta_credible_interval(evidence, GOOD_STATES)

        return PosteriorSummary(
            state=STATE_NAMES[map_idx],
            state_index=map_idx,
            confidence=map_p,
            distribution={STATE_NAMES[i]: float(p) for i, p in enumerate(probs)},
            credible_low=low,
            credible_high=high,
            effective_sample_size=float(ess),
            good_confidence=float(probs[list(GOOD_STATES)].sum()),
            good_credible_low=good_low,
            good_credible_high=good_high,
        )

    def _beta_credible_interval(
        self,
        evidence: Dict[str, int],
        outcomes: "int | Iterable[int]",
        mass: float = 0.95,
    ) -> Tuple[float, float]:
        """Credible interval for the total probability of one or more outcome states.

        For a fully observed context the Dirichlet's aggregation property applies: if
        ``theta ~ Dirichlet(alpha)`` then the sum of any subset ``S`` of its components
        is marginally ``Beta(sum_{k in S} alpha_k, alpha_0 - sum_{k in S} alpha_k)``.
        For partial evidence, independently draw the Beta marginal in every matching
        context and average with the same uniform weights as the predictive mean.
        The quantiles use 16,384 draws and seed 0 for deterministic approximation.
        Averaging concentration vectors would describe a different distribution and
        incorrectly weight well-observed contexts more heavily.

        Passing a set matters because the number this system *reports* is
        ``P(CERTAIN) + P(HIGH)`` -- a sum of two states. Bounding a single MAP state
        instead would attach an interval to a different quantity than the confidence it
        travels with.
        """
        raw_idxs = [outcomes] if isinstance(outcomes, Integral) else list(outcomes)
        idxs = sorted({_index(i, N_STATES, "outcome") for i in raw_idxs})
        mass = _finite_parameter(mass, "mass", positive=True)
        if mass >= 1:
            raise ValueError("mass must be less than 1")
        alphas = self._matching_alphas(evidence)
        if not idxs:
            return 0.0, 0.0
        if len(idxs) == N_STATES:
            return 1.0, 1.0
        rest = [i for i in range(N_STATES) if i not in idxs]
        a_k, a_rest = alphas[:, idxs].sum(axis=1), alphas[:, rest].sum(axis=1)
        tail = (1.0 - mass) / 2.0
        if len(alphas) > 1:
            rng = np.random.default_rng(0)
            draws = np.zeros(16384)
            for selected, other in zip(a_k, a_rest, strict=True):
                draws += rng.beta(selected, other, size=len(draws))
            draws /= len(alphas)
            low, high = np.quantile(draws, [tail, 1 - tail])
            return float(low), float(high)
        a_k, a_rest = float(a_k[0]), float(a_rest[0])
        try:
            from scipy.stats import beta  # optional dependency
            return float(beta.ppf(tail, a_k, a_rest)), float(beta.ppf(1 - tail, a_k, a_rest))
        except ImportError:
            # The minimal installation uses the same reproducible approximation.
            draws = np.random.default_rng(0).beta(a_k, a_rest, size=16384)
            low, high = np.quantile(draws, [tail, 1 - tail])
            return float(low), float(high)

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
_DEFAULT_ENGINE_LOCK = threading.RLock()
_COUNT_SOURCE: Optional[Callable[[], np.ndarray]] = None


def set_count_source(source: Optional[Callable[[], np.ndarray]]) -> None:
    """Register an authoritative transactional aggregate, or disable refreshing.

    The source returns finite nonnegative counts with shape ``(125, 5)``. It takes
    precedence over ENGINE_STATE_PATH. Record updates in that store rather than
    mutating the returned engine: each getter refresh replaces its snapshot.
    """
    if source is not None and not callable(source):
        raise ValueError("count source must be callable or None")
    global _COUNT_SOURCE
    with _DEFAULT_ENGINE_LOCK:
        _COUNT_SOURCE = source


def get_default_engine() -> DirichletBayesianEngine:
    """Lazily build a shared engine with the informative prior (no randomness).

    If ``ENGINE_STATE_PATH`` points at counts saved from feedback, they are loaded on
    top of the prior, so what the engine learned survives a restart.
    """
    global _DEFAULT_ENGINE
    with _DEFAULT_ENGINE_LOCK:
        if _DEFAULT_ENGINE is None:
            engine = DirichletBayesianEngine()
            path = os.getenv("ENGINE_STATE_PATH")
            if _COUNT_SOURCE is None and path and os.path.exists(path):
                engine.load(path)
            _DEFAULT_ENGINE = engine
        if _COUNT_SOURCE is not None:
            _DEFAULT_ENGINE.replace_counts(_COUNT_SOURCE())
        return _DEFAULT_ENGINE


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
    if not isinstance(correct, (Integral, bool, np.bool_)) or correct not in (0, 1):
        raise ValueError("correct must be a binary label")
    return STATE_MAP["CERTAIN"] if correct else STATE_MAP["AMBIGUOUS"]


def build_bayesian_network():
    """Backwards-compatible helper: returns ``(pgmpy_model, outcome_cpd)``."""
    engine = get_default_engine()
    model = engine.build_network()
    outcome_cpd = model.get_cpds("Outcome")
    return model, outcome_cpd


def good_probability(summary: dict) -> float:
    """Collapse a posterior summary to one scalar: ``P(CERTAIN) + P(HIGH)``.

    This is *the* confidence number the rest of the system reports -- the executor
    gates on it, the demo prints it, and the calibration eval scores it. It lives here
    so those three agree by construction rather than by three copies of one expression.

    It is a posterior-predictive probability of a good outcome, so it is on the right
    scale to be checked against empirical accuracy (see ``eval/metrics.py``).

    The matching uncertainty is ``summary["good_credible_interval"]``. Do **not** pair
    this number with ``summary["credible_interval"]``: that bounds the MAP state's
    probability, which is a different quantity and need not even contain this one.
    """
    dist = summary["distribution"]
    return float(dist.get("CERTAIN", 0.0) + dist.get("HIGH", 0.0))


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
    "good_probability",
    "GOOD_STATES",
    "build_bayesian_network",
    "get_default_engine",
    "set_count_source",
    "outcome_label",
    "context_index",
    "context_triple",
    "STATE_NAMES",
    "STATE_MAP",
    "PARENTS",
    "N_STATES",
    "N_CONTEXTS",
]
