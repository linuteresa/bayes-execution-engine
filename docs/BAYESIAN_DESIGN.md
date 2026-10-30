# Bayesian design

Implementation: `bayesian_engine/bayes_engine.py`.

## Full context

A context is `(TaskStatus, DataQuality, ToolReliability)`, three ordinals in `0..4`.
For each of the 125 contexts, outcome quality is Categorical with a Dirichlet prior.
Counts update concentrations in closed form: `alpha_posterior = alpha_prior + counts`.
The posterior predictive distribution is `alpha / sum(alpha)`.

An informative prior places a Gaussian-shaped bump over ordered outcomes according to
aggregate degradation, plus one unit of concentration per outcome. `prior_strength=0`
selects a uniform prior. Parameters, evidence names/indices and counts are validated;
indices must be integers, counts nonnegative integers, and posterior concentrations finite.
Queries and updates use consistent locked snapshots.

For one context, the probability of an outcome or a subset of outcomes is Beta-distributed
under Dirichlet aggregation. With SciPy installed, its credible interval uses exact Beta
quantiles. Without SciPy, reproducibly seeded draws approximate those quantiles.
`good_confidence = P(CERTAIN) + P(HIGH)` therefore has its own `good_credible_interval`.
The MAP state's probability/interval describes a different quantity.

The concentration sum is the effective sample size of that context. Adding data increases
concentration but can change both the mean and variance; credible-interval width is not
unconditionally monotone for every possible sequence of surprising outcomes.

## Partial evidence

Unknown evidence names and invalid levels are rejected. Missing signals are marginalized
with equal weights over matching contexts: the predictive mean is the uniform mean of
per-context predictive probabilities. This is an explicit weighting assumption, not an
inferred distribution of signal contexts in the workload.

Its posterior is a weighted average of independent context probabilities, generally not a
single Dirichlet or Beta distribution. Intervals therefore use 16,384 seeded draws from
the matching Beta marginals and quantiles of their average. Averaging concentration
vectors would weight frequently observed contexts differently from the reported mean and
would attach an interval to another estimand. The reported mean concentration for a
partial query is descriptive; it is not an equivalent one-context ESS.

## Learning and persistence

Standalone `observe`/`fit` add counts. Counts files are validated and saved by atomic
replacement. The API instead installs a shared transactional count source from SQLite or
Redis. Feedback event recording and count increments commit together, and later posterior
queries refresh from that source. A final-answer verdict is mapped to `CERTAIN` or
`AMBIGUOUS` for every recorded step, which remains coarse credit assignment.

Service feedback does not merge per-process `.npy` files, and old files/Redis records are
not automatically migrated. New learning is not evidence that its confidence is calibrated;
that requires held-out labelled evaluation.

## Granularity and optional network view

`125 = 5^3` is a chosen grid size, not an optimum. More cells reduce observations per cell;
unvisited contexts retain their prior. `scaling/latent_bayes.py` projects continuous
features through standardization/PCA and quantile bins before applying conjugate updates.
Quantile bins balance individual axes; their Cartesian products need not all be populated.
Refitting a trained projection is rejected because it would change the meaning of stored
counts. Create a new pipeline for a new feature space.

`build_network()` provides a compatible pgmpy network for visualization/inference when
`requirements-analysis.txt` is installed. Partial-evidence queries are also available
without pgmpy. See [calibration](CALIBRATION.md) for the distinction between a coherent
posterior under these assumptions and empirically reliable answer confidence.
