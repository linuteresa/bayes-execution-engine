# Quickstart: Run & Verify

This guide gets the engine running and — more importantly — shows how to **prove the
Bayesian core works without any model running**, since that logic is fully deterministic.

## Prerequisites

- Python 3.10+
- Docker (for the local LLM) — or any llama.cpp `llama-server` install
- A GGUF model from Hugging Face

---

## 1. Start the local model

The engine talks to llama.cpp's OpenAI-compatible server at `http://localhost:8080/v1`.
Start it with your usual command (nothing here contacts a cloud LLM):

```powershell
docker run --rm -it -e LLAMA_CACHE=/models -p 127.0.0.1:8080:8080 -v C:\llama_models:/models ^
  ghcr.io/ggml-org/llama.cpp:server -hf Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M ^
  --host 0.0.0.0 --port 8080 -c 8192
```

Leave this running in its own terminal. The app's default `LLAMA_CPP_BASE_URL` already
matches this address, so no extra configuration is required.

---

## 2. Install & run the engine (second terminal)

```powershell
cd C:\Users\lAntEr\Desktop\bayes-execution-engine
pip install -r requirements.txt
```

Three ways to run it:

```powershell
# (a) CLI
python main.py "Which high-priority tasks are assigned and to whom?"

# (b) Simple web UI
python app.py                      # http://127.0.0.1:5000

# (c) Async submit/poll API (production-shaped)
uvicorn service.api:app --port 8000
```

For the async API (third terminal):

```powershell
curl -X POST localhost:8000/jobs -H "content-type: application/json" -d "{\"question\":\"list active users\"}"
# -> {"job_id":"<id>","status":"queued"}

curl localhost:8000/jobs/<id>
# -> {"status":"done","result":{...}}
```

Or bring up model + engine together: `docker compose up` (see `docker-compose.yml`).

---

## 3. Verify — no model required (this is the real proof)

The Bayesian engine is deterministic, so its correctness can be demonstrated with zero
LLM involvement.

### 3a. Run the test suite

```powershell
pytest
```

**Expected:** `40 passed` — or `38 passed, 2 skipped` if `pgmpy` / `langgraph` aren't
installed (those two tests auto-skip via `importorskip`).

### 3b. Prove the engine is real Bayesian inference, not randomness

```powershell
python -c "from bayesian_engine.bayes_engine import resolve_conflict; import json; print(json.dumps(resolve_conflict({'TaskStatus':4,'DataQuality':4,'ToolReliability':4}), indent=2))"
```

**Expected** (degraded signals → confident *low* outcome, with calibrated uncertainty):

```json
{
  "state": "AMBIGUOUS",
  "state_index": 4,
  "confidence": 0.428,
  "distribution": {"CERTAIN": 0.077, "HIGH": 0.081, "MEDIUM": 0.124, "LOW": 0.290, "AMBIGUOUS": 0.428},
  "credible_interval": [0.18, 0.69],
  "effective_sample_size": 13.0
}
```

Run it **twice — the numbers are identical.** (The previous implementation produced
*different random* numbers on every call; making this deterministic and data-driven is
the central fix.) Flip the evidence to all-zeros
(`{'TaskStatus':0,'DataQuality':0,'ToolReliability':0}`) and it resolves to `CERTAIN`.

Watch the posterior move with data (conjugate update):

```powershell
python -c "from bayesian_engine.bayes_engine import DirichletBayesianEngine as E; e=E(); ctx={'TaskStatus':2,'DataQuality':2,'ToolReliability':2}; b=e.resolve(ctx).distribution['CERTAIN']; [e.observe(2,2,2,0) for _ in range(50)]; print('P(CERTAIN) before/after 50 obs:', round(b,3), '->', round(e.resolve(ctx).distribution['CERTAIN'],3))"
```

**Expected:** `P(CERTAIN) before/after 50 obs: 0.11 -> 0.816`

### 3c. Prove the 10,000-state scaling story

```powershell
python scaling/latent_bayes.py
```

**Expected:**

```json
{
  "raw_dim": 8,
  "naive_raw_states": 390625,
  "latent_states": 125,
  "compression_ratio": 3125.0,
  "explained_variance": 0.891,
  "test_accuracy": 0.636,
  "majority_baseline": 0.216
}
```

The key line: a classifier built on the **125-state latent grid** scores **0.636** vs a
**0.216** majority baseline — PCA collapsed 390,625 naive states to 125 while keeping the
decision-relevant signal.

---

## 4. What a full end-to-end run looks like

`python main.py "..."` runs Planner (LLM) → Executor (one DAG step at a time) →
Replanner (LLM), and finishes with:

```
============================================================
EXECUTION COMPLETE
============================================================
Final Response: <natural-language answer from the model>
Steps Executed: 3
Confidence Score: 0.31
```

What to expect, honestly:

- **`Confidence Score` and `Steps Executed` are deterministic and meaningful.** When an
  executed step hits ambiguity (e.g. a "query" / "validate" step, which the mock executor
  returns as conflicting), the Bayesian engine fires and confidence drops to ~0.31. You'll
  also see a structured `bayes.conflict_resolved` JSON log line containing the confidence,
  credible interval, and the evidence-matrix coordinates.
- **The natural-language `Final Response` depends on the model.** Qwen2.5-3B-Instruct
  handles the planner's JSON well; if a small model ever returns malformed JSON you may see
  `Steps Executed: 0` and a fallback response. That's a model limitation, not an engine bug
  — which is exactly why the deterministic checks in section 3 are the real proof of the
  engineering.

---

## 5. Observability

Logs are structured JSON (see `core/telemetry.py`). To get more detail:

```powershell
set LOG_LEVEL=DEBUG       # PowerShell:  $env:LOG_LEVEL="DEBUG"
```

Conflict-resolution events look like:

```json
{"event": "bayes.conflict_resolved", "confidence": 0.31,
 "evidence": {"TaskStatus": 3, "DataQuality": 4, "ToolReliability": 0},
 "credible_interval": [0.19, 0.45], "effective_sample_size": 13.0}
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Connection refused` to `:8080` | The llama.cpp container in step 1 isn't running. |
| `Steps Executed: 0` | The model returned non-JSON to the planner; try a larger GGUF or rerun. Engine logic is unaffected (verify via section 3). |
| `pytest` shows 2 skipped | Optional `pgmpy` / `langgraph` not installed — expected; install them to run all 40. |
| Want durable state | Set `JOB_STORE=redis` and `CHECKPOINTER=redis` (see `.env.example`). |
