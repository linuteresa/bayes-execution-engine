"""Unit tests for evidence extraction from raw tool output."""

from core.signals import EvidenceVector, extract_evidence, is_conflict


def test_clean_result_is_high_quality():
    ev = extract_evidence("search docs", "search_result: Found 5 relevant documents")
    assert ev.data_quality == 0          # no conflict language
    assert ev.task_status <= 1           # clean completion
    assert isinstance(ev, EvidenceVector)


def test_conflict_degrades_data_quality():
    ev = extract_evidence("query", "query_result: 3 records with conflicting timestamps")
    assert ev.data_quality == 4          # 'conflict' is maximally degrading
    assert is_conflict("3 records with conflicting timestamps")


def test_disagreement_flagged():
    assert is_conflict("validation uncertain - 2 sources disagree")
    ev = extract_evidence("validate", "Data validation uncertain - 2 sources disagree")
    assert ev.data_quality >= 3


def test_error_terms_lower_reliability():
    ev = extract_evidence("call api", "tool error: request failed after retry")
    assert ev.tool_reliability >= 3


def test_tool_failure_triggers_conflict_resolution():
    assert is_conflict("search_error: failed to retrieve current web results")


def test_no_conflict_for_plain_result():
    assert not is_conflict("executed: summarise the report")


def test_evidence_indices_in_range():
    ev = extract_evidence("x", "stale partial missing conflicting data error timeout")
    for v in (ev.task_status, ev.data_quality, ev.tool_reliability):
        assert 0 <= v <= 4


def test_as_evidence_keys():
    ev = extract_evidence("q", "conflict")
    keys = set(ev.as_evidence())
    assert keys == {"TaskStatus", "DataQuality", "ToolReliability"}
