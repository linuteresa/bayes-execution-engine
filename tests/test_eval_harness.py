"""End-to-end tests for the eval harness, using the offline simulated answerer."""

import json

import pytest

from bayesian_engine.bayes_engine import STATE_MAP
from eval.answerers import SimulatedAnswerer, is_simulated
from eval.datasets import EvalItem, describe, load_bundled, load_items
from eval.harness import outcome_label, read_rows, run_harness, run_metadata, write_rows


def _items(n=12):
    return load_bundled()[:n]


def test_bundled_dataset_loads_with_a_difficulty_spread():
    items = load_bundled()
    summary = describe(items)
    assert summary["n_items"] == 80
    # A calibration set needs items the model gets wrong, so it must not be all easy.
    assert set(summary["by_difficulty"]) >= {"easy", "medium", "hard", "arithmetic"}
    assert all(item.answers for item in items)


def test_load_items_is_deterministic_under_seed():
    a = [i.id for i in load_items("bundled", limit=10, seed=3)]
    b = [i.id for i in load_items("bundled", limit=10, seed=3)]
    assert a == b
    assert a != [i.id for i in load_items("bundled", limit=10, seed=4)]


def test_unknown_dataset_spec_is_rejected():
    with pytest.raises(ValueError):
        load_items("hf:does-not-exist")


def test_harness_produces_one_complete_row_per_item():
    rows = run_harness(_items(), SimulatedAnswerer(seed=0), n_samples=4)
    assert len(rows) == 12
    for row in rows:
        assert 0.0 <= row.confidence <= 1.0
        assert row.credible_low <= row.credible_high
        assert row.correct in (0, 1)
        assert all(0 <= v <= 4 for v in row.context)
        assert 0.0 <= row.agreement <= 1.0
        assert set(row.signals) >= {"agreement_mean", "answerability", "distinct_ratio"}


def test_harness_is_reproducible():
    a = run_harness(_items(), SimulatedAnswerer(seed=0), n_samples=4)
    b = run_harness(_items(), SimulatedAnswerer(seed=0), n_samples=4)
    assert [r.confidence for r in a] == [r.confidence for r in b]
    assert [r.correct for r in a] == [r.correct for r in b]


def test_agreement_tracks_confidence():
    """Higher measured agreement must not produce lower engine confidence."""
    rows = run_harness(load_bundled(), SimulatedAnswerer(seed=0), n_samples=4)
    ordered = sorted(rows, key=lambda r: r.agreement)
    assert ordered[0].confidence <= ordered[-1].confidence


def test_rows_roundtrip_through_jsonl(tmp_path):
    rows = run_harness(_items(6), SimulatedAnswerer(seed=1), n_samples=3)
    path = write_rows(rows, tmp_path / "rows.jsonl")
    again = read_rows(path)
    assert [r.as_dict() for r in again] == [r.as_dict() for r in rows]
    # Plain JSONL: every downstream stage can read it without this package.
    assert json.loads(path.read_text().splitlines()[0])["item_id"] == rows[0].item_id


def test_simulated_runs_are_flagged_in_metadata():
    """The guard that stops a smoke test being mistaken for a calibration result."""
    answerer = SimulatedAnswerer(seed=0)
    meta = run_metadata(answerer, _items(3), n_samples=4, dataset_spec="bundled")
    assert meta["is_simulated"] is True
    assert is_simulated(answerer)


def test_outcome_label_collapses_onto_the_ordinal_endpoints():
    assert outcome_label(1) == STATE_MAP["CERTAIN"]
    assert outcome_label(0) == STATE_MAP["AMBIGUOUS"]


def test_custom_jsonl_items_are_supported(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        json.dumps({"question": "What is 2+2?", "answer": "4", "grader": "numeric"}) + "\n"
    )
    items = load_items(f"jsonl:{path}")
    assert len(items) == 1
    assert isinstance(items[0], EvalItem)
    assert items[0].answers == ["4"]


def test_logprobs_rejection_downgrades_instead_of_failing(monkeypatch):
    """A server that refuses the logprobs flag must cost the baseline, not the run."""
    from eval import answerers

    class _Boom:
        def __init__(self, logprobs):
            self.logprobs = logprobs

        def invoke(self, _prompt):
            if self.logprobs:
                raise RuntimeError("400 Bad Request: unknown parameter 'logprobs'")
            return type("M", (), {"content": "an answer", "response_metadata": {}})()

    monkeypatch.setattr(answerers, "build_llm", None, raising=False)
    monkeypatch.setitem(
        __import__("sys").modules,
        "core.llm",
        type("M", (), {"build_llm": staticmethod(lambda temperature=0.0, logprobs=False: _Boom(logprobs))}),
    )

    answerer = answerers.LlamaAnswerer()
    assert answerer.request_logprobs is True
    assert answerer.invoke("hi").content == "an answer"
    assert answerer.request_logprobs is False       # permanently downgraded
    assert answerer.mean_logprob() is None


def test_unrelated_errors_still_propagate(monkeypatch):
    """A connection failure must not be mistaken for a logprobs rejection."""
    from eval import answerers

    class _Down:
        def __init__(self, logprobs):
            self.logprobs = logprobs

        def invoke(self, _prompt):
            raise ConnectionError("Connection refused")

    monkeypatch.setitem(
        __import__("sys").modules,
        "core.llm",
        type("M", (), {"build_llm": staticmethod(lambda temperature=0.0, logprobs=False: _Down(logprobs))}),
    )
    with pytest.raises(ConnectionError):
        answerers.LlamaAnswerer().invoke("hi")
