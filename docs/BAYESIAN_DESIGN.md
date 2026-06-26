# Bayesian Design: Why Dirichlet, and Why This Granularity

This document defends the probabilistic modelling choices in `bayesian_engine/bayes_engine.py`
from first principles. The goal is to show the choices are *derived*, not decorative.

## 1. The estimation problem

When a DAG step returns conflicting or uncertain output, we want to know the
distribution over the *true outcome quality* given what we observed. We summarise an
observation as three ordinal signals:

```
TaskStatus, DataQuality, ToolReliability  ∈ {CERTAIN, HIGH, MEDIUM, LOW, AMBIGUOUS}
Outcome                                   ∈ {CERTAIN, HIGH, MEDIUM, LOW, AMBIGUOUS}
```

For a fixed context `c = (TaskStatus, DataQuality, ToolReliability)`, the outcome is a
draw from a **Categorical** distribution with unknown parameter vector
`θ_c = (θ_c1, …, θ_c5)`, `Σ θ_ck = 1`. There are `5³ = 125` such vectors to estimate —
one per context. This is exactly a conditional probability table (CPT).

## 2. Why a Dirichlet prior (and not anything else)

We need a prior over `θ_c`, a point on the 4-simplex. The candidates:

| Option | Problem |
|---|---|
| Point estimate / softmax of hand-tuned logits | No uncertainty; can't update with data; overconfident. |
| Gaussian over logits (logistic-normal) | No closed-form posterior; needs MCMC/VI; not conjugate. |
| **Dirichlet** | **Conjugate to the Categorical/Multinomial. Closed-form posterior. Lives on the simplex.** |

The Categorical/Multinomial likelihood has the Dirichlet as its **conjugate prior**.
That is the decisive property. With prior `θ_c ~ Dir(α_c)` and observed outcome counts
`n_c = (n_c1, …, n_c5)`:

```
posterior:            θ_c | data  ~  Dir(α_c + n_c)
posterior predictive: P(Outcome = k | c, data) = (α_ck + n_ck) / Σ_j (α_cj + n_cj)
```

Consequences we actually exploit in code:

- **O(1) online updates.** `engine.observe(...)` is one array increment
  (`α_posterior = α_prior + counts`). No retraining, no sampling. Fully reproducible.
- **Valid distributions for free.** The posterior-predictive mean is a normalised
  vector by construction — every CPT column sums to 1 because it is a Dirichlet mean,
  not because we divided by a total. (`test_prior_is_proper_and_normalised`.)
- **Uncertainty for free.** The marginal of each component is
  `θ_ck ~ Beta(α_ck, α₀ − α_ck)` with `α₀ = Σ α_c`. We report a 95% credible interval
  from that Beta, and `α₀` *is* the effective sample size — so confidence intervals
  shrink as evidence accumulates (`test_effective_sample_size_grows_with_data`).

> The v0 code wrote `np.random.dirichlet(np.ones(5), size=125)` on every call. That
> *samples* a fresh random CPT from a flat Dirichlet and ignores all data — it uses the
> Dirichlet as a random-number generator, not as a prior. The whole point of conjugacy
> (updating with evidence) was absent, which is why results were effectively random.

## 3. The informative prior

A flat prior `α = 1` (Bayes–Laplace) is a valid default but throws away domain
knowledge: we *know* that good signals should predict good outcomes. We encode an
**ordinal, monotonic** prior. With state index `0 = CERTAIN` (best) … `4 = AMBIGUOUS`
(worst), define the aggregate degradation of a context:

```
d(c) = (TaskStatus + DataQuality + ToolReliability) / 12   ∈ [0, 1]
```

and place a Gaussian bump over outcomes centred at `4 · d(c)`:

```
α_ck = 1 + s · N(k ; μ = 4·d(c), σ = spread),   normalised bump
```

- `s = prior_strength` controls how strongly the prior commits (set `s = 0` to recover
  the flat prior).
- The base `+1` keeps the prior **proper** with support on every outcome, so no
  posterior probability is ever exactly zero.

This yields the monotonicity the tests assert: the MAP outcome index is non-decreasing
in aggregate degradation (`test_expected_outcome_increases_with_degradation`), the
all-good context resolves to `CERTAIN`, the all-bad context to `AMBIGUOUS`
(`test_prior_monotonicity`).

## 4. Why this *granularity* (the honest version of "why 125")

125 is `5³`, i.e. **3 signals × 5 ordinal levels**. It is a granularity decision, not
an optimum. The trade-off:

- **Resolution.** Five levels is the standard psychometric granularity (Likert);
  three is too coarse to distinguish "uncertain" (one weak source) from "contradictory"
  (two sources disagree).
- **Statistical density.** Each context must be visited often enough that its posterior
  moves off the prior. With `C` contexts and `N` observations, expected counts per cell
  scale as `N / C`. Pushing `C` up (more signals or more levels) makes cells sparse and
  the model reverts to its prior everywhere — the curse of dimensionality.
- **Interpretability.** 125 cells can be inspected, audited, and reasoned about by a
  human reviewing a conflict decision. A 400k-cell table cannot.

So the defensible claim is: *given three reasonably independent quality signals and a
5-level ordinal scale, 125 contexts is the granularity that balances resolution against
density and interpretability.* If you need richer inputs, you reduce dimensionality
first (see [SCALING.md](SCALING.md)) rather than inflating this table.

## 5. Why keep the Bayesian network structure at all?

Because conditioning on **partial** evidence is then well-defined. If only
`DataQuality` is observed, the engine marginalises the unobserved parents under their
prior — `P(Outcome | DataQuality) = Σ P(Outcome | c) P(other parents)` — which a flat
125-row lookup table cannot do. `build_network()` exposes an equivalent `pgmpy`
`DiscreteBayesianNetwork` for exact inference over arbitrary query/evidence patterns and
for visualisation. (`test_partial_evidence_marginalises`, `test_pgmpy_network_optional`.)

## 6. What an interviewer can verify in 5 minutes

- `α_posterior = α_prior + counts` — `observe()` / `fit()`.
- Posterior-predictive = `α / α₀` — `cpt()`, columns provably sum to 1.
- Credible interval from the Beta marginal — `_beta_credible_interval()`.
- Determinism — two engines built fresh have identical `α` (`test_prior_is_deterministic`).
- Monotonic, ordinal prior — `_build_informative_prior()`.
