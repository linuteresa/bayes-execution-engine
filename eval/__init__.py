"""Calibration evaluation harness for the Bayes Execution Engine.

The README claims the engine produces *calibrated* confidence. This package is the
machinery that tests that claim instead of asserting it:

* :mod:`eval.harness`   -- run the real execution path over gold-answer questions and
  log one row per item (evidence signals, agreement, engine confidence, credible
  interval, ESS, medoid answer, correctness).
* :mod:`eval.metrics`   -- ECE / reliability bins, Brier score with bootstrap CIs,
  AUROC of confidence as a correctness detector, risk-coverage curves.
* :mod:`eval.learning`  -- close the loop: fit the Dirichlet posterior on a train split
  via ``engine.observe`` and measure prior-only vs posterior on held-out data.
* :mod:`eval.report`    -- render the numbers as a markdown report (+ optional figures).

Entry point: ``python -m eval.run_eval --help``.
"""

__all__ = ["datasets", "grading", "answerers", "harness", "metrics", "learning", "report"]
