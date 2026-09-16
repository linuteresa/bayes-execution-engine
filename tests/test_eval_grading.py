"""Unit tests for answer grading.

Grading errors show up downstream as fake miscalibration -- a correct answer scored
wrong looks exactly like overconfidence -- so the edge cases are worth pinning down.
"""

import pytest

from eval.grading import (
    contains_match,
    exact_match,
    extract_numbers,
    grade,
    normalize_answer,
    numeric_match,
)


def test_normalization_strips_articles_punctuation_and_case():
    assert normalize_answer("  The Eiffel Tower!  ") == "eiffel tower"
    assert normalize_answer("Brasília") == "brasilia"
    assert normalize_answer(None) == ""


def test_contains_match_finds_the_answer_in_prose():
    assert contains_match("The capital of France is Paris.", ["Paris"])
    assert contains_match("It is, I believe, Neil Armstrong.", ["Neil Armstrong"])


def test_contains_match_respects_token_boundaries():
    """The bug a naive substring check would have: prefixes must not count."""
    assert not contains_match("The answer is Austria.", ["Austr"])
    assert not contains_match("There is nothing to report.", ["no"])
    assert contains_match("There is no record of it.", ["no"])


def test_contains_match_is_alias_aware():
    assert contains_match("Written by Eric Blair.", ["George Orwell", "Eric Blair"])


def test_exact_match_requires_the_whole_answer():
    assert exact_match("Paris", ["Paris"])
    assert not exact_match("The capital of France is Paris.", ["Paris"])


def test_extract_numbers_handles_separators_and_signs():
    assert extract_numbers("about 1,234.5 and -12 and .5") == [1234.5, -12.0, 0.5]
    assert extract_numbers("no digits here") == []


def test_numeric_match_uses_the_final_number():
    """Working shown before the result must not be graded instead of the result."""
    assert numeric_match("48 in April plus 24 in May, so 72 clips.", ["72"])
    assert not numeric_match("48 in April plus 24 in May.", ["72"])


def test_numeric_match_tolerates_formatting():
    assert numeric_match("The gross pay is $675.00", ["675"])
    assert numeric_match("Total: 1,024", ["1024"])


def test_numeric_match_needs_a_number():
    assert not numeric_match("I am not sure.", ["72"])


def test_grade_dispatches_and_validates():
    assert grade("Paris is the capital.", ["Paris"], "contains")
    assert grade("The result is 12.", ["12"], "numeric")
    assert not grade("The result is 13.", ["12"], "numeric")
    with pytest.raises(ValueError):
        grade("x", ["y"], "llm-judge")
