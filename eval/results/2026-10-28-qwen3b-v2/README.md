# Fresh samples, corrected grading

The 500 questions were sampled four times each using the cached Qwen2.5-3B Q4_K_M model, evidence version 2 and word-overlap agreement. [Original acquisition](../2026-10-28-qwen3b-v2-acquisition/README.md) retains grading-v2 labels. This directory replays those exact samples with grading version 3. All answers, samples, evidence and confidence scores are unchanged; eight correctness labels change from false to true. No additional model calls were made for the correction.

Use [REPORT.md](REPORT.md), [LABEL_AUDIT.json](LABEL_AUDIT.json), and [the validation record](../../../docs/VALIDATION.md). The spot-check covers 20 stratified labels and all eight changes, not all 500 labels. These questions were previously used in the historical run, so this is fresh inference on an existing benchmark rather than a new held-out question set.
