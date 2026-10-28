# Eval results

`python -m eval.run_eval` writes `rows.jsonl`, `metrics.json`, `REPORT.md` and the figures
here. Point `--out` somewhere else to keep runs side by side.

- **`embeddings/`** and **`word-overlap/`** — the current run: 500 TriviaQA + GSM8K items
  sampled once (`--keep-samples`) under word overlap, then the *same samples* re-scored with
  embedding agreement via `--model replay:word-overlap/rows.jsonl`. Compare these two.
- **Top level** (`REPORT.md`, `rows.jsonl`, …) — the first run, before the evidence changes
  (different model build and step prompt). Kept for history; not directly comparable.
- **`example-simulated/`** — output of `--model sim`, the offline stand-in that is *told the
  gold answer*. It is committed to show the report format and to prove the pipeline runs
  with no model. It is **not** a calibration result and is banner-stamped as such.

A reportable run needs a real model:

```bash
llama-server -m ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8080
python -m eval.run_eval --model llama --dataset hf:triviaqa+hf:gsm8k --limit 250 \
  --keep-samples --out eval/results/word-overlap
# same samples, embedding agreement (second llama-server with --embeddings on :8081)
EMBEDDINGS_BASE_URL=http://localhost:8081/v1 EMBEDDINGS_MODEL=all-minilm-l6-v2 \
  python -m eval.run_eval --model replay:eval/results/word-overlap/rows.jsonl \
  --keep-samples --out eval/results/embeddings
```

See [docs/CALIBRATION.md](../../docs/CALIBRATION.md).
