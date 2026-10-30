# Live validation — 2026-10-28/29

The hardening implementation was committed as `c534f11`. The following checks then
exercised real infrastructure locally, beyond the deterministic offline suite.

## Infrastructure and recovery

Final verification passed **342 tests with no skips**, plus Ruff, `git diff --check`
and the virtual environment's `pip check`. An earlier final rerun found Docker Desktop
stopped (five connection failures); after restoring the services the full suite passed.
The optional Redis/Postgres round trips and three process-crash recovery cases passed.
Each crash case uses real HTTP requests and the real OpenAI-compatible client against
a scripted model endpoint. The API is killed after an execution step commits, then a
new process waits out the original lease and recovers the job. Redis and Postgres resume
without repeating the committed planner/executor work. Memory checkpoints restart the
workflow, as documented. Two independent API processes race feedback; one succeeds and
the other receives HTTP 409, with one aggregate observation retained.

The production Dockerfile built successfully with all optional integrations enabled.
`pip check` passed both in that image and in the project virtual environment. A real
OTLP/gRPC collector received the `validation.collector` span and its run identifier.

A separate smoke test used the built Docker API, SQLite jobs, Postgres checkpoints and
the actual Qwen model. The container was killed after a real execution step committed.
The recovered job completed on its second attempt with “Paris” and “323”, recorded five
step observations through feedback, rejected duplicate feedback, and preserved its result
and verdict through another container restart. This is one successful operational smoke
test, not a confidence-calibration study. The captured evidence is
[live-model-recovery.json](validation/live-model-recovery.json).

Validated components:

| Component | Identity |
|---|---|
| Host Python | 3.14.7, macOS arm64 |
| API image Python | 3.12, Linux arm64 |
| Redis server | 7.4.7, Redis Stack image with JSON/Search |
| Postgres | 17.11 |
| OpenTelemetry collector | 0.164.0 |
| Engine image | `sha256:7021b1dbaf19a87a4fb719ddfeb1d9a99724132772c6ea67e1915e0c37c23069` |
| Image base | `python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1` |

The isolated service images are pinned by digest in
[`scripts/validation/compose.yaml`](../scripts/validation/compose.yaml).
These checks do not establish cross-host failover, backup/restore, load capacity or
multi-user authorization. Python 3.10/3.11 compatibility remains a CI responsibility.

## Reproduce infrastructure checks

From the repository root:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt -r requirements-redis.txt \
  -r requirements-postgres.txt -r requirements-telemetry.txt
docker compose -p bayes-validation -f scripts/validation/compose.yaml up -d --wait

TEST_SERVICE_RECOVERY=1 \
TEST_REDIS_URL=redis://127.0.0.1:16379/0 \
TEST_POSTGRES_DSN=postgresql://bayes_test:local-validation-only@127.0.0.1:15432/bayes_test \
  .venv/bin/python -m pytest

OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:14317 \
OTEL_EXPORTER_OTLP_INSECURE=true \
  .venv/bin/python -m scripts.validation.trace_smoke
docker compose -p bayes-validation -f scripts/validation/compose.yaml logs collector

docker build --build-arg INSTALL_REDIS=true --build-arg INSTALL_POSTGRES=true \
  --build-arg INSTALL_TELEMETRY=true -t bayes-engine:validation .
docker run --rm bayes-engine:validation python -m pip check
```

The Docker model smoke uses a dedicated container and volume, never a production
instance. Start the local model server first, then:

```bash
docker run -d --name bayes-engine-model-validation \
  --network bayes-validation_default -p 127.0.0.1:18000:8000 \
  -v bayes-model-validation:/app/data \
  -e LLAMA_CPP_BASE_URL=http://host.docker.internal:8080/v1 \
  -e LLAMA_MODEL=qwen2.5-3b-instruct-q4_k_m \
  -e CHECKPOINTER=postgres \
  -e POSTGRES_DSN=postgresql://bayes_test:local-validation-only@postgres:5432/bayes_test \
  -e JOB_WORKERS=1 -e JOB_LEASE_SECONDS=2 -e JOB_TIMEOUT_SECONDS=180 \
  -e LLM_MAX_TOKENS=512 -e API_KEY=local-validation-only bayes-engine:validation
.venv/bin/python -m scripts.validation.live_model_smoke
```

After review, remove only these disposable validation resources:

```bash
docker rm -f bayes-engine-model-validation
docker volume rm bayes-model-validation
docker compose -p bayes-validation -f scripts/validation/compose.yaml down --volumes
```

## Cached model and fresh acquisition

The existing stopped `ollama` container and named `ollama` volume already contained
`qwen2.5:3b-instruct`, Q4_K_M. No new model weights were downloaded. The model blob's
SHA-256 was independently verified:

`5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6`

The Ollama manifest digest is
`357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`.
Ollama 0.34.1 served a ten-question preflight successfully. The same cached GGUF was
copied into the ignored `models/` directory for native llama.cpp, avoiding Docker's
CPU-only inference on this Mac. The ignored `.env` selects that existing model filename
for subsequent CLI/Compose runs.

The native server is llama.cpp 0.7.0, build 11669, commit `620d2225c`:

```bash
llama-server -m models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 --port 8080 -c 8192 -np 1 -ngl all --seed -1 \
  --alias qwen2.5-3b-instruct-q4_k_m
```

An initial partial collection used `--seed 0` on the server, causing repeated samples.
It was stopped and preserved under ignored `data/validation-fixed-seed-diagnostic`;
it is excluded from the published benchmark. The replacement uses random server seeds
(`--seed -1`), temperature 0.7, four samples per question and a 512-token response limit.

[`eval/data/calibration-500.jsonl`](../eval/data/calibration-500.jsonl) freezes the 250
TriviaQA and 250 GSM8K questions/gold answers already present in historical word-overlap
rows. It contains no historical predictions or confidence scores. Its SHA-256 is
`c877ec0eb6954847364b744004dee3753877b7407ef110636dfc52aa1a36157e`.
Source historical rows SHA-256:
`93bb7e3dc615f06a770988e980f0b5387a0d9dd0577d830499c8fd25f0683fa7`.
This repeats the question set with fresh model responses; it is not a newly held-out
question set or a controlled comparison of inference-server versions.

Reproduce acquisition, or add `--resume` with identical settings after interruption:

```bash
export LLAMA_CPP_BASE_URL=http://127.0.0.1:8080/v1
export LLAMA_MODEL=qwen2.5-3b-instruct-q4_k_m
export LLAMA_MODEL_REVISION=ollama-manifest:357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b
export LLAMA_WEIGHTS_SHA256=5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6
export LLAMA_SERVER_VERSION=llama.cpp-0.7.0-build11669-620d2225c
export LLM_MAX_TOKENS=512 LLM_TIMEOUT_SECONDS=120
.venv/bin/python -m eval.run_eval --model llama \
  --dataset jsonl:eval/data/calibration-500.jsonl --samples 4 --temperature 0.7 \
  --seed 0 --keep-samples --out eval/results/reproduced-qwen3b-v3
```

The completed original acquisition is in
[`2026-10-28-qwen3b-v2-acquisition`](../eval/results/2026-10-28-qwen3b-v2-acquisition/README.md).
Its rows retain the original grading-v2 labels. A 20-label spot-check found two false
negatives; grading version 3 corrects eight labels across the full run. Every changed
label was reviewed. Corrected outputs are in
[`2026-10-28-qwen3b-v2`](../eval/results/2026-10-28-qwen3b-v2/README.md).
The directory name refers to evidence version 2; its corrected grading version is 3.
Answers, raw samples, evidence and confidence scores match the acquisition exactly.

Reproduce corrected scoring without inference:

```bash
.venv/bin/python -m eval.run_eval \
  --model replay:eval/results/2026-10-28-qwen3b-v2-acquisition/rows.jsonl \
  --samples 4 --keep-samples --out /tmp/bayes-corrected-scoring
```

The resulting engine accuracy is 0.478, ECE 0.177 and AUROC 0.556. Token logprob ranks
answers better (AUROC 0.770); engine ECE/Brier are better than the raw logprob score's.
This operational milestone is complete, but confidence quality and end-to-end validation
remain research work. See [calibration methodology](CALIBRATION.md) for scope and the
limitations of the spot-check.
