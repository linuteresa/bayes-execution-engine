# Review history

Chronological record of code reviews on this repository. Each entry states what was found,
what was fixed, and what was deliberately left. They are kept because the findings explain
why several design choices look the way they do.

| Date | Document | Scope | Outcome |
|---|---|---|---|
| 2026-10-28 | [Strict review](2026-10-28-review.md) | Pre-remediation snapshot | 19 findings |
| 2026-10-28 | [Remediation](2026-10-28-remediation.md) | Fixes for those 19 | Behaviour + regression coverage |
| 2026-10-29 | [PR #11 review](2026-10-29-review.md) | 104 files, +11,807/-1,913 | 15 findings: 7 fixed, 8 deferred |

Line numbers in a review describe the snapshot it was written against and will drift.

## Open items

The two carried forward from the 2026-10-29 review, both of which affect published
calibration numbers and should be settled before more are published:

- **Two numeric grammars disagree about equality.** `core/numbers.py` is Fraction-exact and
  unit-aware; `eval/grading.py` keeps a float parser with a tolerance. The confidence signal
  and the correctness label therefore use different definitions of "same number".
- **The grader accumulates per-benchmark heuristics.** Tuning it against observed misses
  moves headline numbers without the model changing.
