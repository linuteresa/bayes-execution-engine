# Quickstart: Run & Verify

The Bayesian core is deterministic, so you can prove it works with **no model running**.

## 1. Start the local model

```powershell
docker run --rm -it -e LLAMA_CACHE=/models -p 127.0.0.1:8080:8080 -v C:\llama_models:/models ^
  ghcr.io/ggml-org/llama.cpp:server -hf Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M ^
  --host 0.0.0.0 --port 8080 -c 8192
```

The app defaults to `http://localhost:8080/v1`, so no extra config.

## 2. Run

```powershell
pip install -r requirements.txt
python main.py "Compare REST and gRPC and when to use each."   # CLI
python app.py                                                  # web UI :5000
uvicorn service.api:app --port 8000                            # async API
```

Async API: `POST /jobs {question}` → `{job_id}` → poll `GET /jobs/{job_id}`.

## 3. Verify without a model (the real proof)

```powershell
pytest                                                          # 60 passed (or 58 + 2 skipped)

python demo.py --sim                                            # confident vs ambiguous, side by side

python -c "from bayesian_engine.bayes_engine import resolve_conflict; import json; print(json.dumps(resolve_conflict({'TaskStatus':4,'DataQuality':4,'ToolReliability':4})))"
```

Expected from the one-liner: `state AMBIGUOUS`, `confidence ≈ 0.43`, plus a credible interval
and effective sample size. Run it twice — **identical** (the v0 code returned random numbers;
making it deterministic and data-driven is the headline fix). All-zeros evidence ⇒ `CERTAIN`.

`python scaling/latent_bayes.py` → 390,625 naive states compress to 125 (`3,125×`), latent
accuracy 0.64 vs 0.22 baseline.

## 4. A full run

`python main.py "..."` runs Planner → Executor (samples the model and measures agreement) →
Replanner, then prints the final answer, steps executed, and a confidence score. Confidence is
earned: consistent answers score high; ambiguous prompts scatter, log a `bayes.conflict_resolved`
event, and score low. Costs ~`EXECUTOR_SAMPLES`× tokens per step.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused :8080` | The model container (step 1) isn't running. |
| `Steps Executed: 0` | Model returned malformed planner JSON; the parser recovers in most cases — retry or use a larger GGUF. Engine logic is fine (verify via §3). |
| `pytest` shows 2 skipped | Optional `pgmpy`/`langgraph` not installed — expected. |
| Want durable state | `JOB_STORE=redis`, `CHECKPOINTER=redis` (see `.env.example`). |
