# Eval results

`python -m eval.run_eval` writes `rows.jsonl`, `manifest.json`, `metrics.json`, `REPORT.md`
and the figures. Use a fresh `--out` for each experiment. Collection refuses to replace an
existing run unless `--overwrite` is explicit. Each completed row is flushed and fsynced,
so `--resume` can continue a failed collection with the same dataset/configuration; it
validates the experiment fingerprint, saved identities and checksummed durable prefix.

- **`2026-10-28-qwen3b-v2/`** — fresh 500-question word-overlap collection on the cached
  Qwen2.5-3B model, with recorded weights hash and server version. Evidence version 2,
  corrected grading version 3. A replay of the preserved acquisition corrects eight
  labels without changing model responses or confidence scores. See its label audit.
- **`2026-10-28-qwen3b-v2-acquisition/`** — original fresh rows with grading-v2 labels,
  retained to make the label correction traceable. Use the corrected report above for
  current conclusions. These are fresh samples on the existing historical question set.
- **`embeddings/`** and **`word-overlap/`** — historical evidence: 500 TriviaQA + GSM8K items
  sampled once (`--keep-samples`) under word overlap, then the *same samples* re-scored with
  embedding agreement via `--model replay:word-overlap/rows.jsonl`. Their reports were
  reanalysed with order-independent handling of confidence ties, preserving original
  evidence, confidences, samples and correctness labels. These rows predate versioned
  acquisition manifests and the current evidence/grading logic; they do not validate
  calibration of the current executor. Missing provenance stays unknown.
- **Top level** (`REPORT.md`, `rows.jsonl`, …) — the first run, before the evidence changes
  (different model build and step prompt). Kept for history; not directly comparable.
- **`example-simulated/`** — current-format output of `--model sim`, the offline stand-in that is *told the
  gold answer*. It is committed to show the report format and to prove the pipeline runs
  with no model. It is **not** a calibration result and is banner-stamped as such.

A reportable run needs a real model:

```bash
llama-server -m ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8080
python -m eval.run_eval --model llama --dataset hf:triviaqa+hf:gsm8k --limit 250 \
  --keep-samples --out eval/results/new-word-overlap
# same samples, embedding agreement (second llama-server with --embeddings on :8081)
EMBEDDINGS_BASE_URL=http://localhost:8081/v1 EMBEDDINGS_MODEL=all-minilm-l6-v2 \
  python -m eval.run_eval --model replay:eval/results/new-word-overlap/rows.jsonl \
  --keep-samples --out eval/results/new-embeddings
```

For analysis without sampling or embedding requests:

```bash
python -m eval.run_eval --rows eval/results/embeddings/rows.jsonl \
  --out eval/results/embeddings --bootstrap 2000
```

The metrics characterize **single-step answer confidence**. They do not establish the
confidence of a multi-step final response or validate the replanner/stopping policy.
Risk-coverage uses the expected accuracy of uniform selection within a boundary tie;
a confidence threshold alone cannot divide equally scored questions. New rows carry a
versioned experiment fingerprint with dataset identity, sampling and agreement settings,
code/dependency versions, acquisition source and grading version. Reanalysis records its
own configuration and source-row checksum separately from acquisition provenance.

See [docs/CALIBRATION.md](../../docs/CALIBRATION.md).
