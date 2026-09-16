# Eval results

`python -m eval.run_eval` writes `rows.jsonl`, `metrics.json`, `REPORT.md` and the figures
here. Point `--out` somewhere else to keep runs side by side.

- **`example-simulated/`** — output of `--model sim`, the offline stand-in that is *told the
  gold answer*. It is committed to show the report format and to prove the pipeline runs
  with no model. It is **not** a calibration result and is banner-stamped as such.

A reportable run needs a real model:

```bash
llama-server -m ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8080
python -m eval.run_eval --model llama --dataset hf:triviaqa+hf:gsm8k --limit 250
```

See [docs/CALIBRATION.md](../../docs/CALIBRATION.md).
