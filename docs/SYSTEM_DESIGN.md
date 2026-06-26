# System Design: From Experiment to Scalable Infrastructure

This document describes how the engine is structured for concurrent, multi-tenant,
fault-tolerant operation, and how the in-repo implementation maps onto a full
production deployment. Parts marked **[implemented]** exist in this repo; parts marked
**[blueprint]** are documented designs that the implemented pieces are shaped to accept.

## 1. Decoupled, event-driven execution **[implemented + blueprint]**

A Plan-and-Execute run makes several LLM calls and can take tens of seconds. Running
that inside a synchronous request causes load-balancer/gateway timeouts and pins one
worker per in-flight prompt.

**Contract (implemented, `service/api.py`):**

```
POST /jobs   {question, thread_id?}  -> 202 {job_id}     # returns immediately
GET  /jobs/{job_id}                  -> {status, result} # client polls
```

The request thread only enqueues work and returns; execution happens off-thread. In
this repo the worker is a `ThreadPoolExecutor` — sufficient to demonstrate the
decoupling and the API contract.

**Production topology (blueprint):** replace the thread pool with a broker and workers.
The API contract above is unchanged.

![img_1.png](img_1.png)

- **API replicas** are stateless and sit behind the load balancer; any replica serves
  any client.
- **Workers** are a separately scaled consumer group on the requests topic. Throughput
  scales by adding workers; back-pressure is handled by the broker, not by dropping
  requests.
- **Results** flow back on a second topic (or are read from the shared store by the
  polling endpoint).

Celery (or Faust/Kafka-streams) is the natural worker framework; the `_run_job`
function in `service/api.py` is already the unit a Celery task would wrap.

## 2. External state persistence & fault tolerance **[implemented]**

LangGraph keeps `PlanExecuteState` in memory by default, so a crash mid-DAG loses all
progress. `persistence/checkpointer.py` provides a **checkpointer factory** that
persists state after every node transition, keyed by `thread_id`:

- `CHECKPOINTER=memory` → `MemorySaver` (dev default).
- `CHECKPOINTER=redis` → `RedisSaver` (shared, durable).
- `CHECKPOINTER=postgres` → `PostgresSaver` (durable, queryable, transactional).

This buys three properties:

1. **Fault tolerance** — a restarted worker resumes a run from its last checkpoint
   instead of restarting the DAG.
2. **Concurrency / multi-tenancy** — each user session is a distinct `thread_id`, so
   many runs share one process with **zero state bleed**. `run_execution_engine(...,
   thread_id=...)` threads this through.
3. **Horizontal scaling** — with a shared saver, any stateless worker can resume any
   thread, so workers scale out freely.

The async job lifecycle itself is also persisted via the **job store**
(`persistence/job_store.py`): `InMemoryJobStore` for a single process, `RedisJobStore`
for a shared, restart-surviving store. `build_job_store()` selects via `JOB_STORE`.

## 3. Concurrency & isolation model **[implemented]**

| Concern | Mechanism |
|---|---|
| Per-user state isolation | LangGraph `thread_id` per session |
| Job state across restarts | `RedisJobStore` (TTL'd job records) |
| DAG progress across restarts | Redis/Postgres checkpointer |
| Thread-safe in-proc store | `InMemoryJobStore` guarded by a lock |

Because both the job store and the checkpointer are external and keyed, **N** workers
can process **M** concurrent sessions with no shared mutable memory.

## 4. Observability & telemetry **[implemented]**

`core/telemetry.py` emits **structured JSON** events (never free-text prints) and
optional OpenTelemetry spans (enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` is set). Key
events:

- `engine.run.start` / `.end` with duration and `thread_id`.
- `bayes.conflict_resolved` with `confidence`, resolved `state`, `credible_interval`,
  `effective_sample_size`, and the full evidence coordinates.
- `job.submitted` / `job.done` / `job.error`.

**Alerting strategy (blueprint):** these structured fields are designed to drive SLOs
without parsing text logs:

- Page when the **rate of low-confidence resolutions** (`confidence < τ`) rises — a
  proxy for degraded upstream data quality.
- Page when a specific tool's `ToolReliability` signal collapses across many
  resolutions — a failing tool schema or outage.
- Track p95 `engine.run` duration for latency regressions.
- Track posterior **effective sample size** per context to know when the model is still
  prior-dominated (needs more data) versus data-driven.

## 5. CI/CD & containerisation **[implemented]**

- **`Dockerfile`** — slim, layer-cached, non-root, with a `/health` healthcheck. The
  GGUF weights are deliberately *not* baked in; the app talks to a separate
  `llama-server` container (`docker-compose.yml`), keeping the image small.
- **`.github/workflows/ci.yml`** — on every push/PR: `ruff` lint → `pytest` on a
  Python 3.10/3.11/3.12 matrix → Docker image build. This mirrors an enterprise
  pipeline (e.g. Azure DevOps multi-stage) with quality gates before packaging.

## 6. Infrastructure as Code **[blueprint]**

A minimal Terraform module would provision: a container service (ECS Fargate / Cloud
Run / AKS) for the API and worker images, a managed Redis (ElastiCache / Memorystore)
for the job store and checkpoints, and an optional managed Postgres for durable
checkpoints. Sketch:

```hcl
# illustrative only
module "engine" {
  source        = "./modules/container_service"
  image         = var.engine_image
  desired_count = var.worker_count        # scale workers horizontally
  env = {
    JOB_STORE    = "redis"
    CHECKPOINTER = "redis"
    REDIS_URL    = module.cache.connection_url
  }
}

module "cache" {
  source     = "./modules/redis"
  node_type  = "cache.t4g.small"
}
```

The application is already 12-factor (all config via env vars), so it drops into such a
module without code changes.

## 7. Cross-stack interoperability **[blueprint]**

The submit/poll API and the broker contract are language-agnostic. A Java Spring Boot
microservice integrates by `POST`ing to `/jobs` and polling, or by producing to the
requests topic and consuming results — the Python engine and a JVM ecosystem
communicate purely through HTTP/JSON and the message broker, never through in-process
calls.
