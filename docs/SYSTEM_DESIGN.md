# System design

The implemented system is a local-model workflow with durable jobs, leased dispatch,
checkpoint recovery and shared feedback counts. Kafka, RabbitMQ and Celery are possible
future integrations; none is part of the current runtime.

## API and admission

`POST /jobs {question, thread_id?}` persists a queued job and returns HTTP 202 with its ID.
`GET /jobs/{id}` returns its lifecycle/result, and `POST /jobs/{id}/feedback {correct}`
records one verdict on a successful finished job. Invalid bodies are rejected; a full
pending capacity returns 429 with `Retry-After`. `/health` reports API availability and
does not probe model readiness, checkpoint health or queue progress.

`API_KEY`, when set, requires a shared bearer token except on `/health`. This is intended
for trusted callers. There is no authenticated principal, per-user job ownership, tenant
quota or isolated learning state. Client `thread_id` values label sessions and confer no
access rights. Expose this service to multiple independent users only after adding that
ownership model.

## Persistence choices

| State | Default | Alternatives | Restart behavior |
|---|---|---|---|
| Jobs, leases, feedback events and counts | SQLite `./data/jobs.sqlite3` | Redis; explicit memory demo | SQLite/Redis survive process loss with durable storage |
| Graph checkpoints | MemorySaver | RedisSaver, PostgresSaver, none | Memory is lost; persistent savers retain committed progress |
| Worker threads | 8 per API process | `JOB_WORKERS` | Workers discover persisted pending/expired claims |

SQLite is intended for processes on one host sharing a reliable local database file. Do
not assume a network-mounted SQLite file gives a distributed database. Redis supports
shared dispatch/counts across replicas. Postgres currently stores checkpoints only; it is
not a job-store backend. A shared checkpointer alone does not make job admission durable.

Factories open stores/savers during FastAPI lifespan and keep connections alive until
bounded workers finish during shutdown. Requested backends fail startup on missing
packages, invalid configuration or setup failure; they never silently fall back to memory.
Importing routes does not start workers or create persistent files.

## Recovery and stale workers

A worker claims a job transactionally using a fresh attempt token and an expiring lease.
Heartbeat renewal requires ownership; result publication and checkpoint-reference updates
also verify the owner and unexpired lease. Failed renewals cancel further useful work and
leave recovery to a later claimant. A model call already in flight may still finish.

Each attempt writes to an isolated physical checkpoint thread. The saver publishes an
exact full-checkpoint reference through the job store's ownership check. Recovery copies
only that committed reference into a new attempt thread, ignoring stale/unpublished
checkpoints and pending writes. This prevents an old worker from changing the state used
by its successor. A session ID can be reused for a new question because each submission
has its own run identity.

With memory checkpoints, process loss can remove the published state. Recovery then
restarts this pure model workflow from the original prompt. A missing published Redis or
Postgres checkpoint is an error, rather than silently selecting another state. Do not add
side-effecting tools under an assumption of exactly-once execution: inference and other
work before a failure can repeat. Typed external tools, idempotency keys and transactional
outbox semantics remain future work.

Terminal results commit to the job store before memory checkpoint threads are cleared.
Persistent checkpoints remain until retention cleanup. Cleanup deletes every recorded
attempt thread before pruning a terminal job, preserving aggregate learned counts. A
checkpoint deletion failure leaves the terminal job available for cleanup retry.

## Feedback transactions

A successful job accepts one strict boolean verdict. Recording the verdict and all
per-step count increments is one SQLite transaction or Redis optimistic transaction;
competing or repeated verdicts return 409. Every API process uses that shared count source
for posterior queries. Counts remain after terminal job retention expires.

One final-answer verdict is assigned to all recorded step contexts. Re-checks and repeated
contexts can each contribute observations, so these are coarse correlated labels, not
independent adjudications of every step. Per-user learning isolation and detailed credit
assignment are not implemented.

`ENGINE_STATE_PATH` remains a standalone engine counts-file option. The API's transactional
count source takes precedence and does not write `.npy` snapshots. Existing `.npy` counts
and the earlier Redis `job:*` layout are not migrated automatically. Back up old state and
write an explicit migration before changing a deployed store.

## Limits

| Setting | Default | Meaning |
|---|---:|---|
| `MAX_PROMPT_CHARS` | 16000 | Maximum input question length |
| `JOB_MAX_PENDING` | 128 | Admission bound for active queued/running work |
| `JOB_WORKERS` | 8 | Worker threads per process, at most 64 |
| `JOB_TIMEOUT_SECONDS` | 300 | Absolute deadline set on submission, including queue time |
| `JOB_LEASE_SECONDS` | 30 | Claim duration, renewed during work |
| `JOB_RETENTION_SECONDS` | 86400 | Terminal job/checkpoint retention |
| `LLM_TIMEOUT_SECONDS` | 60 | Per-request timeout, reduced to remaining run budget |
| `LLM_MAX_TOKENS` | 2048 | Maximum generated tokens per model response |
| `MAX_STEPS` | 10 | Executed step budget, at most 100 |
| `EXECUTOR_SAMPLES` | 4 | Samples per step, at most 64 |
| `EXECUTOR_CONTEXT_CHARS` | 12000 | Prior-result prompt budget, at most 64000 |
| `MODEL_OUTPUT_ATTEMPTS` | 2 | Bounded schema-format attempts, at most 3 |

Limits bound resource use; they do not guarantee useful or correct answers. Choose worker
counts against the local model server's real capacity. Cleanup runs periodically, so
expiration is not an instantaneous TTL. API and inference limits are not a substitute for
external ingress/body-size controls on a public deployment.

## Containers and optional backends

The default Compose path runs a llama.cpp server and an engine with a named SQLite volume.
`docker compose --profile redis up --build llama redis engine-redis` selects a separate
engine build with Redis dependencies, the Redis URL, both Redis backends and an AOF volume
configured with `appendfsync always`.
Run only one API variant on the configured localhost port.

RedisSaver requires JSON and Search capabilities. The profile uses the official Redis
Stack server image, whose data directory is mounted on `redis-data`; plain Redis 7 is
insufficient. These requirements are documented by the
[saver maintainers](https://pypi.org/project/langgraph-checkpoint-redis/) and
[Redis Stack Docker documentation](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-stack/docker/).

For Postgres checkpoints, install `requirements-postgres.txt` and configure
`CHECKPOINTER=postgres` plus `POSTGRES_DSN`. The file includes Psycopg binary/pool support,
which the [Postgres saver package](https://pypi.org/project/langgraph-checkpoint-postgres/3.2.0/)
otherwise leaves to deployment choice. Keep a supported job store as well.

Model weights and image tags are external inputs. Set verified `LLAMA_IMAGE` and
`REDIS_IMAGE` digests in a real deployment, record loaded model weights, and back up volumes.
The supplied base/runtime pins are not a complete hash-locked deployment bill of materials.
Docker/model/backend integration was not executed in the offline review environment.

## Observability and validation

Structured JSON logs include correlated run/step events and both Bayesian confidence/CI
pairs. To export spans, install `requirements-telemetry.txt`, then configure an OTLP/gRPC
endpoint and optional service name. The SDK and gRPC exporter are both required; explicit
configuration with missing dependencies fails visibly. Localhost plaintext collectors
also need `OTEL_EXPORTER_OTLP_INSECURE=true`.

CI installs runtime/development requirements, runs Ruff and pytest on Python
3.10/3.11/3.12/3.14, then builds the default image. Local verification uses installed
Python 3.14 dependencies. Real-backend tests require `TEST_REDIS_URL` or
`TEST_POSTGRES_DSN`; without those services they are skipped, not certified passing.
