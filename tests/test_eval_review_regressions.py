"""Regression cases from the strict review: metric integrity and recoverable collection."""

import json
from dataclasses import replace

import numpy as np
import pytest

from eval.answerers import LlamaAnswerer, ReplayAnswerer, SimulatedAnswerer
from eval.datasets import EvalItem, load_items, load_jsonl
from eval.grading import contains_match, extract_numbers, numeric_match
from eval.harness import metadata_from_rows, read_rows, run_harness, run_metadata, write_rows
from eval.learning import split_rows
from eval.metrics import risk_coverage
from eval.provenance import file_digest
from eval.report import render_report
from eval.run_eval import main


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("EMBEDDINGS_BASE_URL", raising=False)
    for name in ("reliability_diagram", "risk_coverage_plot", "learning_panels"):
        monkeypatch.setattr(f"eval.run_eval.figures_mod.{name}", lambda *_a, **_k: None)


def _items():
    return [
        EvalItem(f"q-{i}", f"What is the result of {i}+1?", [str(i + 1)], "numeric")
        for i in range(8)
    ]


def _args(out, *extra):
    return ["--model", "sim", "--limit", "6", "--samples", "2", "--seed", "3",
            "--bootstrap", "5", "--no-learning", "--quiet", "--keep-samples",
            "--out", str(out), *extra]


def test_constant_confidence_cannot_gain_from_reordering_correctness():
    for labels in ([1] * 50 + [0] * 50, [0] * 50 + [1] * 50):
        curve = risk_coverage([0.5] * 100, labels)
        assert curve.accuracy_at_coverage(0.5) == pytest.approx(0.5)
        assert curve.aurc == pytest.approx(0.5)


def test_partial_tie_curve_is_permutation_invariant_and_interpolates_counts():
    scores = [0.9, 0.8, 0.8, 0.8, 0.1]
    correct = [1, 1, 0, 0, 0]
    expected = risk_coverage(scores, correct)
    # At two observations, keep the certain first success and 1/3 of a success
    # from the three-way tie. At 1.5 observations, keep 1 + (0.5 / 3) successes.
    assert expected.accuracy_at_coverage(0.4) == pytest.approx(2 / 3)
    assert expected.accuracy_at_coverage(0.3) == pytest.approx(7 / 9)
    for seed in range(8):
        perm = np.random.default_rng(seed).permutation(5)
        actual = risk_coverage(np.array(scores)[perm], np.array(correct)[perm])
        assert actual.aurc == pytest.approx(expected.aurc)
        assert [p.accuracy for p in actual.points] == pytest.approx(
            [p.accuracy for p in expected.points]
        )


def test_duplicate_ids_are_rejected_before_sampling_and_replay(tmp_path):
    duplicate = [replace(_items()[0], question="first"), replace(_items()[0], question="second")]
    with pytest.raises(ValueError, match="duplicate"):
        run_harness(duplicate, SimulatedAnswerer())
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(item.as_dict()) for item in duplicate))
    with pytest.raises(ValueError, match="duplicate"):
        load_jsonl(path)
    with pytest.raises(ValueError, match="duplicate"):
        load_items("bundled+bundled")
    row = run_harness(_items()[:1], SimulatedAnswerer(), keep_samples=True)[0]
    path.write_text((json.dumps(row.as_dict()) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        ReplayAnswerer.from_rows(path)


def test_repeated_questions_with_distinct_ids_never_cross_train_test():
    rows = run_harness(_items(), SimulatedAnswerer())
    rows += [replace(row, item_id=f"repeat-{row.item_id}",
                     question="  " + row.question.upper() + "  ", correct=1 - row.correct)
             for row in rows]
    split = split_rows(rows, seed=7)
    train_q = {" ".join(row.question.casefold().split()) for row in split.train}
    test_q = {" ".join(row.question.casefold().split()) for row in split.test}
    assert not train_q & test_q
    assert len(split.train) + len(split.test) == len(rows)


def test_rows_preserve_versioned_collection_config_and_replay_inherits_it(tmp_path):
    rows = run_harness(_items(), SimulatedAnswerer(seed=17), keep_samples=True)
    experiment = rows[0].experiment
    assert experiment["schema_version"] == 2
    assert experiment["confidence_scope"] == "single_step"
    assert experiment["sampling"]["seed"] == 17
    assert experiment["execution"]["evidence_version"] == 2
    assert experiment["execution"]["code_sha256"]
    assert experiment["dataset_digest"]
    path = write_rows(rows, tmp_path / "rows.jsonl")
    replay = ReplayAnswerer.from_rows(path)
    rescored = run_harness(replay.items, replay, keep_samples=True)
    provenance = rescored[0].experiment
    assert provenance["sampling"] == experiment["sampling"]
    assert provenance["dataset_spec"] == experiment["dataset_spec"]
    assert provenance["sample_source"]["sha256"] == file_digest(path)
    assert provenance["sample_source"]["experiment_id"] == experiment["experiment_id"]


def test_recorded_weights_identity_changes_the_resumable_experiment(monkeypatch):
    monkeypatch.setenv("LLAMA_MODEL_REVISION", "model-manifest-digest")
    monkeypatch.setenv("LLAMA_WEIGHTS_SHA256", "a" * 64)
    monkeypatch.setenv("LLAMA_SERVER_VERSION", "ollama-test")
    # Constructing the client and acquisition metadata makes no inference requests.
    answerer = LlamaAnswerer()
    first = run_metadata(answerer, _items(), n_samples=4)
    assert first["sampling"]["server_weights_revision"] == "model-manifest-digest"
    assert first["sampling"]["weights_sha256"] == "a" * 64
    assert first["sampling"]["server_version"] == "ollama-test"
    monkeypatch.setenv("LLAMA_WEIGHTS_SHA256", "b" * 64)
    second = run_metadata(answerer, _items(), n_samples=4)
    assert first["experiment_id"] != second["experiment_id"]


def test_legacy_provenance_is_partial_and_cannot_be_relabelled_by_cli_default():
    rows = run_harness(_items(), SimulatedAnswerer())
    rows = [replace(row, experiment={}, source="gsm8k", is_simulated=False) for row in rows]
    meta = metadata_from_rows(rows)
    assert meta["dataset_spec"] == "hf:gsm8k"
    assert meta["provenance_status"] == "legacy_partial"
    assert meta["historical_scores"]
    assert meta["similarity"] == "unknown (legacy rows)"
    report = render_report({"metadata": {**meta, "analysis_only": True}})
    assert "single-step answer confidence" in report
    assert "do not establish calibration of the current executor" in report
    assert "no model or embedding requests" in report


class _FailingAnswerer(SimulatedAnswerer):
    def __post_init__(self):
        super().__post_init__()
        self.calls = 0
        self.fail_at = None

    def invoke(self, prompt):
        self.calls += 1
        if self.calls == self.fail_at:
            raise ConnectionError("model disconnected")
        return super().invoke(prompt)


def test_collection_survives_disconnect_and_resume_skips_durable_items(tmp_path, monkeypatch):
    first = _FailingAnswerer(seed=3)
    first.fail_at = 3  # one two-sample item was completed
    monkeypatch.setattr("eval.run_eval.build_answerer", lambda *_a, **_k: first)
    with pytest.raises(ConnectionError):
        main(_args(tmp_path))
    saved = read_rows(tmp_path / "rows.jsonl")
    assert len(saved) == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["collection"]["status"] == "collecting"
    assert manifest["collection"]["completed_items"] == 1

    second = _FailingAnswerer(seed=3)
    monkeypatch.setattr("eval.run_eval.build_answerer", lambda *_a, **_k: second)
    assert main(_args(tmp_path, "--resume")) == 0
    resumed = read_rows(tmp_path / "rows.jsonl")
    assert len(resumed) == 6
    assert resumed[0].as_dict() == saved[0].as_dict()
    assert second.calls == 10
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["collection"]["status"] == "complete"
    assert manifest["collection"]["rows_sha256"] == file_digest(tmp_path / "rows.jsonl")


@pytest.mark.parametrize("changed", [
    ("--temperature", "0.2"), ("--samples", "3"), ("--seed", "9"), ("--limit", "5"),
])
def test_resume_rejects_changed_experiment_without_touching_rows(tmp_path, changed):
    main(_args(tmp_path))
    before = (tmp_path / "rows.jsonl").read_bytes()
    with pytest.raises(ValueError, match="configuration differs"):
        main(_args(tmp_path, "--resume", *changed))
    assert (tmp_path / "rows.jsonl").read_bytes() == before


def test_resume_checks_saved_question_identity_and_recovers_only_a_torn_tail(tmp_path):
    main(_args(tmp_path))
    path = tmp_path / "rows.jsonl"
    before = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"item_id":"torn')
    assert main(_args(tmp_path, "--resume")) == 0
    assert path.read_bytes() == before
    objects = [json.loads(line) for line in before.splitlines()]
    objects[0]["question"] = "a different question"
    path.write_text("\n".join(json.dumps(row) for row in objects) + "\n")
    with pytest.raises(ValueError, match="differs from the requested dataset"):
        main(_args(tmp_path, "--resume"))


def test_collection_does_not_silently_overwrite_existing_results(tmp_path):
    main(_args(tmp_path))
    before = (tmp_path / "rows.jsonl").read_bytes()
    with pytest.raises(ValueError, match="output already contains"):
        main(_args(tmp_path))
    assert (tmp_path / "rows.jsonl").read_bytes() == before


def test_analysis_reuses_frozen_rows_without_any_inference(tmp_path, monkeypatch):
    main(_args(tmp_path / "original"))
    path = tmp_path / "original" / "rows.jsonl"
    before = path.read_bytes()

    def no_inference(*_a, **_k):
        raise AssertionError("analysis called a sampler")

    monkeypatch.setattr("eval.run_eval.build_answerer", no_inference)
    monkeypatch.setenv("EMBEDDINGS_BASE_URL", "http://unavailable.example/v1")
    assert main(["--rows", str(path), "--out", str(tmp_path / "analysis"),
                 "--bootstrap", "5", "--no-learning", "--quiet"]) == 0
    assert path.read_bytes() == before
    metrics = json.loads((tmp_path / "analysis" / "metrics.json").read_text())
    assert metrics["metadata"]["similarity"] == "word-overlap"
    assert metrics["analysis"]["stored_confidence_evidence_and_labels_preserved"]
    assert metrics["analysis"]["source_rows_sha256"] == file_digest(path)
    report = (tmp_path / "analysis" / "REPORT.md").read_text()
    assert f"--rows {path}" in report


@pytest.mark.parametrize("prediction,target", [
    ("1e3", "1000"), ("-1.25E-2", "-0.0125"), ("3/4", "0.75"),
    (r"\boxed{\frac{3}{4}}", "0.75"),
    ("Working: 1000 items. Final answer: 12.\nCheck used 1000.", "12"),
])
def test_numeric_grading_handles_final_answer_and_complete_number_forms(prediction, target):
    assert numeric_match(prediction, [target])


@pytest.mark.parametrize("prediction,target", [
    ("1e3", "3"), ("3/4", "4"), ("1/0", "0"), ("not 12", "12"),
    ("Final answer: 3 or 4", "4"),
    ("12, then 1/0", "12"), ("12, then 1e999", "12"),
])
def test_numeric_grading_rejects_misparsed_or_ambiguous_answers(prediction, target):
    assert not numeric_match(prediction, [target])


def test_alias_grading_does_not_reward_rejected_or_ambiguous_candidates():
    assert not contains_match("Paris is not the answer; the answer is London.", ["Paris"])
    assert not contains_match("Paris is incorrect.", ["Paris"])
    assert not contains_match("Paris or London.", ["Paris"])
    assert contains_match("Paris was considered. Final answer: London.", ["London"])
    assert not contains_match("Paris was considered. Final answer: London.", ["Paris"])
    assert not contains_match("The answer is Paris. Final answer: London.", ["Paris"])
    assert contains_match("The answer is Paris. Final answer: London.", ["London"])
    assert extract_numbers("1e3, 3/4, -2.5e-2") == [1000.0, 0.75, -0.025]


def test_resume_rejects_modified_saved_scores(tmp_path):
    main(_args(tmp_path))
    path = tmp_path / "rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["confidence"] = 0.123456789
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        main(_args(tmp_path, "--resume"))
