# Bayes Execution Engine

A **Plan-and-Execute** multi-agent orchestration engine (LangGraph) with a **conjugate
Bayesian conflict resolver**, served behind an **async API**, running entirely on a
**local LLM** via [llama.cpp](https://github.com/ggml-org/llama.cpp).

Standard ReAct agents are brittle: when a tool returns conflicting or uncertain data,
they let the LLM "guess again" and frequently loop or hallucinate. This engine instead
decouples *thinking* (a planned DAG of steps) from *doing*, and when execution hits
ambiguity it **pauses and runs a real Bayesian update** to decide how confident the
result is — with calibrated uncertainty, not a vibe.


## Table of contents

- [Architecture](#architecture)
- [The Bayesian core ](#the-bayesian-core-the-interesting-part)
- [Scaling to 10,000+ states](#scaling-to-10000-states)
- [Running it](#running-it)
- [System design & production concerns](#system-design--production-concerns)
- [Monitoring & observability](#monitoring--observability)
- [Testing & CI](#testing--ci)
- [Repository layout](#repository-layout)

---

## Architecture

The orchestration is a LangGraph `StateGraph`. A shared `PlanExecuteState` flows
between three nodes; the executor calls the Bayesian engine only when it detects a
conflict.

![img.png](img.png)

For long-running prompts the engine runs as an **event-driven service** so the LLM
work never blocks (or times out) an HTTP request:
![img_1.png](img_1.png)
```mermaid


In this repo the "queue/worker pool" is a `ThreadPoolExecutor`, which is enough to
prove the decoupling. The contract (`POST /jobs` → poll `GET /jobs/{id}`) is identical
to a production broker-backed deployment — see
[docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) for the Kafka/RabbitMQ + Celery topology.

---

## Execution model: self-consistency → real evidence

Full detail in **[docs/EXECUTION_MODEL.md](docs/EXECUTION_MODEL.md)**.

The executor doesn't run a step once and trust it. When an LLM is configured, it runs
each step by **sampling the model several times** (temperature > 0) and measuring how
much the samples agree (`nodes/llm_executor.py`). Disagreement among a model's own
samples is a well-established proxy for uncertainty/hallucination (self-consistency,
Wang et al. 2022). From those samples it derives the three ordinal signals the Bayesian
engine consumes — from **real measurements**, not keywords:

- `DataQuality` ← semantic agreement across samples,
- `TaskStatus` ← answerability (fraction of non-refusal samples),
- `ToolReliability` ← answer dispersion (how many distinct answers).

The consensus (medoid) answer flows to the replanner; the agreement signal flows to the
Bayesian engine. So the conflict resolver now fires on **genuine model uncertainty**, and
the per-step confidence is the posterior probability the answer is high-quality. When no
LLM is supplied (unit tests / offline), the executor falls back to a deterministic mock
that simulates conflicting enterprise tools.

### See it in one screen

```bash
python demo.py --sim     # no model needed; --sim forces the simulation
```

prints a confident vs. an ambiguous prompt side by side:

```
CONFIDENT  "What is the capital of France?"    self-consistency 1.00  CONFIDENCE 0.72  (CERTAIN)
AMBIGUOUS  "...Bitcoin price next Tuesday?"    self-consistency 0.15  CONFIDENCE 0.34  (conflict)
```

Drop `--sim` to run it against your live llama.cpp model.

## What you can ask it

The executor answers each step from the **local model's own knowledge** — no web access,
no private data. So it suits questions the model can reason about, ideally multi-step ones:

- Explanations & how-tos: *"Explain how RSA encryption works and why it's secure."*
- Comparisons & trade-offs: *"Compare REST and gRPC and when to use each."*
- Decomposable reasoning: *"Outline the steps to containerize a Python app and why each matters."*

The point worth demoing is the **confidence contrast**: ask something the model is solid on
and you get high confidence; ask something obscure or genuinely ambiguous and the samples
scatter, a conflict is logged, and confidence drops.

Poor fits, by design: real-time / current information, anything needing private data or
external tools, and hard trivia/math where a small model guesses (self-consistency will
correctly report low confidence, but the answer itself won't be reliable). With a 3B model
the value is the orchestration + calibrated confidence, not raw answer quality.

## The Bayesian core 

Full derivation in **[docs/BAYESIAN_DESIGN.md](docs/BAYESIAN_DESIGN.md)**. The short
version:

The engine maps an execution observation into three ordinal signals — `TaskStatus`,
`DataQuality`, `ToolReliability`, each in `{CERTAIN, HIGH, MEDIUM, LOW, AMBIGUOUS}` — and
asks: *given these signals, what is the distribution over the true outcome quality?*

For each of the `5 × 5 × 5 = 125` signal contexts, `P(Outcome | context)` is a
**Categorical** distribution. The conjugate prior of a Categorical is the
**Dirichlet** — that is the entire reason it is the right tool, and the property the
v0 code never actually used:

- **Conjugacy** ⇒ the posterior is Dirichlet too, updated in closed form as
  `α_posterior = α_prior + counts`. No sampling, no gradient steps, O(1) online
  updates, fully reproducible.
- The Dirichlet lives on the **probability simplex**, so every posterior-predictive
  column is a valid distribution *by construction* — not by a normalisation hack.
- The concentration sum `α₀ = Σα` **is** the model's effective sample size, giving us
  free, principled **uncertainty quantification**: credible intervals shrink as the
  engine observes more data.

```python
from bayesian_engine.bayes_engine import resolve_conflict

resolve_conflict({"TaskStatus": 4, "DataQuality": 4, "ToolReliability": 4})
# {
#   "state": "AMBIGUOUS",
#   "confidence": 0.43,
#   "distribution": {"CERTAIN": .02, "HIGH": .06, "MEDIUM": .17, "LOW": .32, "AMBIGUOUS": .43},
#   "credible_interval": [0.18, 0.69],   # 95% CI on the MAP probability
#   "effective_sample_size": 13.0        # prior + observed counts
# }
```

The prior is **informative and monotonic**: the more degraded the three input signals,
the more prior mass the outcome places on degraded states. It is learnable — call
`engine.observe(...)` / `engine.fit(...)` and the posterior moves toward the data.

### An honest note on "why 125"

125 is **not a magic or mathematically optimal number** — it is `5³`: three signals at
five ordinal levels. It is a deliberate **granularity choice** governed by a
bias–variance trade-off:

- Too few states → the model is too coarse to separate "uncertain" from "contradictory".
- Too many states → each context is visited so rarely that its posterior never leaves
  the prior (the curse of dimensionality).

Five ordinal levels is the standard psychometric granularity (think Likert scales), and
three signals keep every cell dense enough to actually learn. When the *input* needs to
be richer, we don't grow this table — we reduce dimensionality first (next section).
Defending the choice this way is the point; claiming 125 is "optimal" would be hand-waving.

---

## Scaling to 10,000+ states

A real agent might expose a dozen noisy telemetry channels. Discretising 8 raw signals
into 5 bins each is already `5⁸ = 390,625` contexts — statistically hopeless to
populate. The strategy is **reduce, then reason** (`scaling/latent_bayes.py`):

1. Standardise the high-dimensional, correlated signal vector.
2. Project onto a low-dimensional latent manifold with **PCA** (a drop-in stand-in for
   UMAP / an autoencoder — same interface).
3. Discretise each latent axis into quantile bins → a small, **dense** grid.
4. Run the same conjugate Dirichlet update in latent space.

Running `python scaling/latent_bayes.py` on synthetic 8-D agent signals:

| Metric | Value |
|---|---|
| Naive raw state space | **390,625** (`5⁸`) |
| Latent state space | **125** (`5³`) |
| Compression ratio | **3,125×** |
| Variance retained (PCA, 3 comp.) | **0.89** |
| Latent Bayesian test accuracy | **0.64** |
| Majority-class baseline | **0.22** |

The reduced representation keeps the statistical strength of conjugacy while letting
input dimensionality grow arbitrarily. Details in [docs/SCALING.md](docs/SCALING.md).

---

## Running it

> Full run-and-verify walkthrough (with expected outputs) in **[QUICKSTART.md](QUICKSTART.md)**.

This project runs a **local** GGUF model from Hugging Face via llama.cpp — no cloud
LLM, no API keys. `langchain_openai.ChatOpenAI` is used only because `llama-server`
exposes an OpenAI-*compatible* endpoint; nothing ever leaves the machine.

```bash
# 1. Start a local model (any GGUF you downloaded from Hugging Face)
llama-server -m ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8080

# 2. Install
pip install -r requirements.txt

# 3a. CLI
python main.py "Which high-priority tasks are assigned and to whom?"

# 3b. Simple web UI (synchronous)
python app.py            # http://127.0.0.1:5000

# 3c. Async API (production-shaped)
uvicorn service.api:app --port 8000
curl -X POST localhost:8000/jobs -H 'content-type: application/json' \
     -d '{"question":"..."}'        # -> {"job_id": "..."}
curl localhost:8000/jobs/<job_id>   # poll for the result
```

Or bring up the whole topology (engine + model server) with `docker compose up`.

Configuration is via environment variables (`.env` supported): `LLAMA_CPP_BASE_URL`,
`LLAMA_MODEL`, `JOB_STORE` (`memory`|`redis`), `CHECKPOINTER`
(`memory`|`redis`|`postgres`), `LOG_LEVEL`.

---

## System design & production concerns

Detailed in **[docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md)**. Highlights:

- **Decoupled async execution** — submit/poll API so long LLM runs never time out a
  request; swap the in-process worker pool for Kafka/RabbitMQ + Celery without changing
  the API contract.
- **External state persistence** — a LangGraph checkpointer (`persistence/checkpointer.py`)
  writes `PlanExecuteState` to Redis/Postgres after every node, so a crashed worker
  resumes mid-DAG instead of losing progress.
- **Concurrency / multi-tenancy** — each session is a distinct `thread_id`; many users'
  runs share one stateless process with zero state bleed, enabling horizontal scaling.
- **Fault tolerance & load balancing** — stateless workers behind a load balancer, all
  reading shared state; any worker can pick up any thread.

---

## Monitoring & observability

`core/telemetry.py` emits **structured JSON logs** (not free text) and optional
OpenTelemetry spans. Every conflict resolution logs the confidence score, the resolved
state, the credible interval, the effective sample size, and the full 125-state
evidence coordinates:

```json
{"event": "bayes.conflict_resolved", "confidence": 0.31,
 "evidence": {"TaskStatus": 3, "DataQuality": 4, "ToolReliability": 0},
 "credible_interval": [0.19, 0.45], "effective_sample_size": 13.0}
```

These feed dashboards and alerts — e.g. page when the rate of low-confidence
resolutions rises (degraded upstream data) or when a tool's reliability signal
collapses (a failing tool schema). See [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md).

---

## Testing & CI

- **60 unit tests** covering conjugate-update correctness, prior monotonicity, CPT
  normalisation, self-consistency evidence extraction, tolerant JSON parsing, DAG routing
  and replanner fallbacks, the scaling pipeline, and the job store. Run: `pytest`.
- **GitHub Actions** (`.github/workflows/ci.yml`) runs `ruff` lint + the full test
  suite on a Python 3.10/3.11/3.12 matrix, then builds the Docker image — on every push
  and PR.

---

## Roadmap / future scope

Natural extensions, roughly in order of value:

- **Semantic agreement.** Replace bag-of-words cosine with sentence embeddings or an NLI
  model so self-consistency credits paraphrases and penalises same-words/different-meaning
  answers. The `llm_executor` interface is built to swap this in without touching the rest.
- **Online learning of the CPT.** Feed observed `(evidence, outcome)` pairs back through the
  conjugate update (`engine.observe`) — e.g. using user feedback or a verifier as the
  outcome label — so the Dirichlet posterior adapts to a deployment instead of staying at
  its prior.
- **Real tools behind the executor.** Swap the single-LLM step for typed tools/MCP calls
  (DB, retrieval, APIs); run a step against *multiple* sources and let genuine cross-source
  disagreement drive the Bayesian update.
- **Confidence-driven control flow.** Use the credible interval (not just the point
  estimate) to decide when to re-plan, escalate to a larger model, or ask the user — turning
  confidence into an actual routing policy.
- **Broker-backed scaling.** Replace the in-process worker pool with Kafka/RabbitMQ + Celery
  and a shared Redis/Postgres checkpointer (designed for in `docs/SYSTEM_DESIGN.md`).
- **Calibration evaluation.** Add reliability diagrams / ECE on a labelled set to show the
  reported confidences are actually calibrated, not just monotonic.

## Repository layout

```
bayesian_engine/bayes_engine.py   Dirichlet–Multinomial conjugate engine
core/signals.py                   raw tool output  -> ordinal evidence
core/json_utils.py                tolerant JSON extraction for small-LLM output
core/telemetry.py                 structured JSON logging + tracing
core/graph.py                     LangGraph wiring (shared by CLI/UI/API)
core/llm.py                       llama.cpp client (OpenAI-compatible, local)
nodes/llm_executor.py             self-consistency execution + evidence extraction
nodes/                            planner / executor / replanner
scaling/latent_bayes.py           PCA latent-space scaling proof-of-concept
service/api.py                    async FastAPI submit/poll service
persistence/                      job store + LangGraph checkpointer factory
demo.py                           side-by-side confident vs ambiguous confidence demo
tests/                            pytest suite (60 tests)
docs/                             design deep-dives (Bayesian, execution, scaling, system)
```

<img width="1894" height="930" alt="UI screenshot" src="https://github.com/user-attachments/assets/88a1bca7-52a8-45f5-aacf-e2bda2c3879f" />
