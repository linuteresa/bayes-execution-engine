# Bayesian Design: Why Dirichlet, Why This Granularity

Implementation: `bayesian_engine/bayes_engine.py`.

## The problem

For a fixed context `c = (TaskStatus, DataQuality, ToolReliability)` (each ordinal, 5 levels),
the outcome quality is a draw from a **Categorical** with unknown `θ_c`. There are `5³ = 125`
such vectors to estimate,  a conditional probability table.

## Why Dirichlet

The Categorical/Multinomial's **conjugate prior is the Dirichlet** — the decisive property:

- **Conjugacy:** `θ_c ~ Dir(α_c)` + counts `n_c` ⇒ posterior `Dir(α_c + n_c)`, closed form.
  Updating = one array add (`observe`), O(1), reproducible, no sampling.
- **Simplex support:** the posterior-predictive mean `α/α₀` is a valid distribution by
  construction - every CPT column sums to 1 with no renormalization.
- **Free uncertainty:** each component is `Beta(α_k, α₀−α_k)`, giving a 95% credible interval;
  `α₀ = Σα` is the effective sample size, so intervals shrink as data arrives.

> The v0 code called `np.random.dirichlet(...)` per request — sampling a *random* CPT and
> ignoring all data. It used the Dirichlet as an RNG, not a prior. Conjugacy (the point) was
> absent, so results were random.

## The informative prior

A flat `α = 1` is valid but throws away domain knowledge. We encode an **ordinal, monotonic**
prior: with `0 = CERTAIN` (best) … `4 = AMBIGUOUS` (worst), place a Gaussian bump over outcomes
centred at the aggregate degradation `d(c) = (Task+Data+Tool)/12`, scaled by `prior_strength`,
plus a `+1` base to keep it proper. Result: the MAP outcome rises monotonically with
degradation (all-good ⇒ CERTAIN, all-bad ⇒ AMBIGUOUS). Set `prior_strength=0` for the flat prior.

## Why 125 

`125 = 5³` is a **granularity choice**, not an optimum:

- Too coarse ⇒ can't separate "uncertain" (one weak source) from "contradictory" (sources disagree).
- Too fine ⇒ cells get `~N/C` observations; raise `C` and the posterior stays at its prior.
- 125 cells stay human-auditable; 400k cells don't.

Five levels = standard Likert granularity; three signals keep cells dense. Richer input ⇒
reduce dimensions first ([SCALING.md](SCALING.md)), don't inflate the table.

## Why keep a Bayesian network

Conditioning on **partial** evidence is then well-defined: an unobserved parent is marginalized
under its prior. A flat lookup table can't do that. `build_network()` exposes an equivalent
`pgmpy` model for exact inference and visualization.
