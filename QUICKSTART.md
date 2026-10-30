# Quickstart

Use Python 3.10+; Python 3.14 was used for local review verification. Requirements for older
Python versions are compatibility branches selected by the CI matrix, not a locally reproduced matrix.

## Install and verify offline

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
pytest
ruff check .
python demo.py --sim
```

On Windows, activate `.venv\Scripts\Activate.ps1` in PowerShell. The offline demo uses a
scripted sampler and proves the mechanism runs; it does not establish model calibration.
The normal CLI/API returns an error when the model is unavailable.

## Start inference

Place your GGUF weights under `./models/`, then run:

```bash
llama-server -m ./models/model.gguf --host 127.0.0.1 --port 8080 -c 8192
python main.py "Compare REST and gRPC and when to use each."
uvicorn service.api:app --host 127.0.0.1 --port 8000
```

Wait for the model to finish loading before submitting jobs. `.env` is loaded by the API,
CLI and Flask entrypoints. `LLAMA_MODEL` is the request label; the loaded server weights
must be recorded separately for a reproducible experiment.

Keep downloaded weights in `./models/` or a persistent Docker mount. A container's
`--rm` option removes the container, not the files in its mounted model directory.
An existing Ollama container can also be reused without downloading its models again:

```bash
docker start ollama
docker exec ollama ollama list
# Set these in .env for the engine, or export them before running evaluation:
export LLAMA_CPP_BASE_URL=http://127.0.0.1:11434/v1
export LLAMA_MODEL=qwen2.5:3b-instruct
```

The Ollama model name must match `ollama list`. This repository's default llama.cpp
Compose setup expects a host GGUF under `./models`; the local Ollama container keeps
its weights in its own Docker volume. Docker on macOS uses CPU inference for this
Ollama image; native llama.cpp can use Metal with a GGUF copied from that cache.

```bash
curl -s http://localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"question":"Explain RSA and its assumptions.","thread_id":"demo-session"}'
curl -s http://localhost:8000/jobs/JOB_ID
curl -s http://localhost:8000/jobs/JOB_ID/feedback \
  -H 'Content-Type: application/json' -d '{"correct":true}'
```

Feedback requires a successful finished job and accepts one strict boolean verdict.
If `API_KEY` is configured, add `-H 'Authorization: Bearer TOKEN'`. Callers share jobs and
learning state; this token is not per-user authorization. The Flask UI (`python app.py`,
localhost:5000) is a synchronous source-checkout demo and does not use the async job queue.

## Persistence

Defaults: `JOB_STORE=sqlite`, `JOB_DB_PATH=./data/jobs.sqlite3`, `CHECKPOINTER=memory`.
SQLite persists jobs and aggregate feedback counts. Lost in-memory checkpoints cause a
recovered unfinished job to restart from its original prompt. Redis/Postgres checkpoints
allow recovery from a committed graph state. Missing published persistent checkpoints are
errors; stale attempts cannot publish a new recovery reference or completion.

For Redis, install `requirements-redis.txt` and configure both `JOB_STORE=redis` and
`CHECKPOINTER=redis` against Redis with JSON/Search support. For Postgres checkpoints,
install `requirements-postgres.txt`, set `CHECKPOINTER=postgres` and `POSTGRES_DSN`, and
keep a supported job store. Old Redis `job:*` records and `.npy` feedback counts require an
explicit migration; startup does not import them into the new transactional store.

## Containers

```bash
docker compose up --build
# Or, stop the default engine and use the Redis profile:
docker compose --profile redis up --build llama redis engine-redis
```

Set `LLAMA_MODEL_FILE` to the filename under `./models/`. The API is published on localhost
only. SQLite and Redis each have named data volumes. Use verified image digests for a
repeatable deployment and back up data separately. Container startup does not establish
model readiness; `/health` only reports that the API is running.

## Evaluate

```bash
pip install -r requirements-eval.txt
python -m eval.run_eval --model sim --out /tmp/bayes-sim
python -m eval.run_eval --model llama --dataset hf:triviaqa+hf:gsm8k \
  --limit 250 --keep-samples --out eval/results/new-run
# Continue the same interrupted collection with identical arguments and --resume.
python -m eval.run_eval --rows eval/results/embeddings/rows.jsonl --out /tmp/bayes-analysis
```

Collection refuses existing output without `--resume` or explicit `--overwrite`.
Historical real-model rows validate their saved single-step scores only. They predate the
current evidence/grading logic and do not validate final-response confidence.

## Troubleshooting

| Symptom | Action |
|---|---|
| `llm_unavailable` | Start the configured model server and check its URL/readiness. |
| `invalid_model_output` | Inspect model JSON/schema failures; try a model that follows the required format. |
| `execution_timeout` | Reduce the plan/sample budget or explicitly adjust the job/request limits. |
| HTTP 429 | The durable pending-job capacity is full; retry after the supplied delay. |
| Saver import/setup error | Install the matching optional requirements and verify the backend service/modules. |
| Lost mid-run progress after restart | Use a persistent checkpointer; memory checkpoints cannot survive process loss. |
| OTLP startup error | Install `requirements-telemetry.txt` and configure an OTLP/gRPC collector. |

See [system design](docs/SYSTEM_DESIGN.md) for limits, recovery and deployment boundaries.
