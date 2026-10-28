# Bayes Execution Engine

A local Plan-and-Execute agent built with LangGraph. It samples each execution step,
measures agreement, maps three evidence signals into a Dirichlet posterior, and uses the
result to trigger a bounded re-check. A FastAPI service provides durable submit/poll jobs
and transactional feedback learning.

Answers come from the model's knowledge and earlier step results. There are no live data,
file, database or web tools. Agreement measures consistency, so a repeated wrong answer can
still score highly. The final `confidence_score` is a **weakest-step score**, not a calibrated
probability that the complete response is correct.

## Run locally

Python 3.10+ is declared; the current local verification used Python 3.14. Scientific
requirements use compatibility branches for older Python versions. Start with a virtual
environment, then follow [QUICKSTART.md](QUICKSTART.md).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env

# In a separate terminal, load your own model weights.
llama-server -m ./models/model.gguf --host 127.0.0.1 --port 8080 -c 8192

python main.py "Compare REST and gRPC and when to use each."
uvicorn service.api:app --host 127.0.0.1 --port 8000
# Optional synchronous local UI, from this source checkout:
python app.py
```

CLI, Flask and FastAPI load `.env`. The API defaults to SQLite at `./data/jobs.sqlite3` and
an in-process checkpoint saver. Jobs, verdicts and learned counts survive a restart;
resuming mid-graph requires Redis or Postgres checkpoints. With memory checkpoints, a
recovered job restarts the pure model workflow when its saved checkpoint is unavailable.

```bash
curl -s http://localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"question":"Explain RSA and its assumptions."}'
curl -s http://localhost:8000/jobs/JOB_ID
curl -s http://localhost:8000/jobs/JOB_ID/feedback \
  -H 'Content-Type: application/json' -d '{"correct":true}'
```

Set `API_KEY` to require `Authorization: Bearer TOKEN` on API routes other than `/health`.
This is a shared token for trusted callers, with shared learning state. There is no
per-user ownership or tenant authorization model; a `thread_id` is a session label.

## Execution and recovery

Planner → Executor → Replanner shares typed state. Each executor prompt includes bounded
prior results, so dependent steps can use earlier work. Planner and replanner JSON are
validated against explicit action schemas; repeated invalid output, unavailable inference
and expired deadlines return errors with zero confidence.

The executor samples four answers by default. Word-overlap cosine measures agreement;
`EMBEDDINGS_BASE_URL` enables embedding cosine. Evidence version 2 preserves numeric signs,
decimals, scientific notation, fractions and percentages, guards differing numeric claims
and units before consensus, and accepts short valid answers. Neither measure establishes
factual truth or detects every contradiction.

Steps below 0.5 get one forced re-check unless all samples refused. The loop stops after
`MAX_STEPS`, a job deadline, or a valid final response. Re-checks replace earlier attempts
when computing the weakest-step score. See [execution details](docs/EXECUTION_MODEL.md).

The API persists jobs before accepting them, claims them with expiring leases, and rejects
stale-worker completion. Each claimed attempt writes isolated checkpoints and publishes an
exact recovery reference only while it owns the job. Recovery uses that committed state;
model calls can repeat across failures. This is not an exactly-once external-tools engine.

Defaults are 128 pending jobs, 8 worker threads, a 300-second submission-to-completion
budget, 60-second inference requests and 24-hour terminal-job retention. Capacity returns
HTTP 429. Retention removes terminal jobs and their checkpoint threads while retaining
aggregate feedback counts. See [system design](docs/SYSTEM_DESIGN.md).

## Bayesian core and feedback

For each of 125 evidence contexts, outcome probabilities have a Dirichlet prior. Counts
update it by `alpha_posterior = alpha_prior + counts`. The reported single-step score is
`P(CERTAIN) + P(HIGH)` with its own credible interval; the MAP state's confidence and
interval are separate quantities.

Fully observed contexts have exact Beta-marginal intervals when SciPy is installed.
Partially observed queries average matching context predictions with equal weights and
use reproducibly seeded posterior draws for their intervals. Mean concentration for such
a query is descriptive, not the concentration of an equivalent Dirichlet posterior.
See [Bayesian design](docs/BAYESIAN_DESIGN.md).

A finished job accepts one strict boolean verdict. Its verdict and per-step count updates
commit together in SQLite or Redis, and other API workers read that shared count source.
Run-level feedback still assigns the same verdict to every recorded step: credit assignment
is coarse. API feedback no longer saves per-process `.npy` snapshots. Legacy `.npy` counts
and old Redis `job:*` records are not migrated automatically.

## Calibration: historical evidence and current scope

The committed real-model rows predate evidence version 2, versioned acquisition manifests
and the revised grader. Their reports were regenerated **from saved rows only**: original
answers, evidence, confidence and correctness labels are unchanged; tied-score selective
prediction is now invariant to input order. Missing acquisition settings remain unknown.
These historical results do not establish calibration of the current executor or of a
multi-step final response.

| Historical single-step signal, 500 questions | ECE ↓ | Brier ↓ | AUROC ↑ |
|---|---:|---:|---:|
| Engine, embedding agreement | 0.144 | 0.246 | 0.654 |
| Engine, word overlap | 0.145 | 0.259 | 0.601 |
| Raw embedding agreement | 0.434 | 0.424 | 0.617 |
| Mean token logprob baseline | 0.424 | 0.406 | 0.654 |

On those stored embedding scores, TriviaQA accuracy is 0.432 at full coverage and **0.611**
at 50% coverage; GSM8K is 0.392 → **0.444**. Intermediate coverage assumes uniform selection
within a tied group; a confidence threshold alone cannot separate equal scores.
Embedding GSM8K AUROC is 0.554 with a confidence interval spanning chance. Consistency alone
is insufficient for reliable arithmetic verification.

Reports: [embeddings](eval/results/embeddings/REPORT.md),
[word overlap](eval/results/word-overlap/REPORT.md),
[first historical run](eval/results/REPORT.md). The first run's held-out learning ECE change
is −0.055 [−0.107, 0.012], so that run does not demonstrate an improvement statistically.
The [simulated example](eval/results/example-simulated/REPORT.md) is explicitly told the
gold answer and is a pipeline demonstration only.

```bash
pip install -r requirements-eval.txt
# Fresh collection; completed rows are fsynced and acquisition metadata is versioned.
python -m eval.run_eval --model llama --dataset hf:triviaqa+hf:gsm8k \
  --limit 250 --keep-samples --out eval/results/new-run
# Recover an interrupted collection using identical acquisition arguments plus --resume.
# Reanalyse saved scores without model or embedding requests:
python -m eval.run_eval --rows eval/results/embeddings/rows.jsonl --out /tmp/bayes-analysis
# Offline synthetic demonstration:
python -m eval.run_eval --model sim --out /tmp/bayes-sim
```

`--model replay:PATH` re-scores stored samples under the current evidence implementation;
`--rows PATH` preserves stored scores and labels. These are different experiments. A fresh
live-model evaluation and an audited end-to-end benchmark remain necessary. See
[calibration methodology](docs/CALIBRATION.md).

## Deployment and optional integrations

```bash
# Default SQLite service + local llama.cpp container; mount ./models/model.gguf first.
docker compose up --build
# Modules-capable Redis + installed Redis saver/client + persistent Redis volume:
docker compose --profile redis up --build llama redis engine-redis
```

Both API variants bind localhost port 8000, so run one at a time. SQLite lives on the named
`engine-data` volume; Redis uses `redis-data` with AOF and `appendfsync always`.
A volume is persistence, not a backup.
The Redis saver requires JSON and Search support; the profile uses Redis Stack rather
than a plain Redis 7 image. See the [maintainer requirements](https://pypi.org/project/langgraph-checkpoint-redis/).

Optional dependencies are separate: `requirements-redis.txt`, `requirements-postgres.txt`,
`requirements-telemetry.txt`, `requirements-analysis.txt`, and `requirements-eval.txt`.
Setting a requested persistent backend without its package/service is a startup error.
Postgres is a checkpoint backend; jobs and feedback still use SQLite or Redis.

Structured JSON logs work by default. Install the telemetry requirements before setting
`OTEL_EXPORTER_OTLP_ENDPOINT` (OTLP/gRPC). Container builds enable optional packages through
`INSTALL_REDIS`, `INSTALL_POSTGRES` or `INSTALL_TELEMETRY` build arguments. Pin model weights
and image digests in an actual deployment; default image tags remain mutable.

## Verification and remaining work

```bash
ruff check .
pytest
```

Offline tests cover the graph, schema failures, evidence, Bayesian updates, uncertainty,
leased recovery, feedback transactions, HTTP validation, evaluation integrity and scaling.
Live backend tests are opt-in through `TEST_REDIS_URL` and `TEST_POSTGRES_DSN`. Local review
verification did not exercise a real model, Docker image build, Redis/Postgres server or
OTLP collector; the CI matrix is configured separately and is not a claim of local execution.

Runtime direct dependencies are pinned where inspected locally. Older-Python scientific
branches are bounded ranges, and transitive dependencies are not a complete platform lock.
The wheel includes the engine, API, CLI modules and bundled evaluation questions; the Flask
UI is run from the source checkout. Remaining work includes authenticated ownership for
multi-user hosting, real typed tools, independent correctness verifiers, an end-to-end
confidence evaluation, and live infrastructure validation.

[Original strict review](docs/CODE_REVIEW.md) records the issues found before remediation;
[remediation status](docs/REMEDIATION.md) describes the implemented fixes and validation limits.
