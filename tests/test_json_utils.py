"""Unit tests for tolerant JSON extraction."""

from core.json_utils import extract_json


def test_plain_object():
    assert extract_json('{"steps": ["a", "b"]}') == {"steps": ["a", "b"]}


def test_fenced_json_block():
    text = 'Sure, here you go:\n```json\n{"action": "response", "response": "hi"}\n```'
    assert extract_json(text) == {"action": "response", "response": "hi"}


def test_prose_around_json():
    text = 'The plan is below.\n{"steps": ["x"]}\nLet me know if that works!'
    assert extract_json(text) == {"steps": ["x"]}


def test_nested_braces():
    assert extract_json('{"a": {"b": 1}, "c": [2, 3]}') == {"a": {"b": 1}, "c": [2, 3]}


def test_braces_inside_strings_do_not_break_scan():
    assert extract_json('{"response": "use {curly} braces"}') == {"response": "use {curly} braces"}


def test_first_balanced_object_wins():
    assert extract_json('{"steps": ["a"]} then {"junk": 1}') == {"steps": ["a"]}


def test_unparseable_returns_none():
    assert extract_json("no json here at all") is None
    assert extract_json("") is None
    assert extract_json("{not valid json}") is None


def test_skips_unparseable_then_finds_valid():
    text = '{broken json} and later {"ok": true}'
    assert extract_json(text) == {"ok": True}
