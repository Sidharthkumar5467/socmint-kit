from datetime import date

import pytest

from socmint.config import load_config, redact, save_config
from socmint.pipeline import RunInputError, validate_inputs
from socmint.schema import Edge, dump
from socmint.util.timeutil import day_end, day_start, iso, parse_ts
from socmint.verify import analysable, dedupe, gate_timestamps


def test_parse_ts_formats():
    assert iso(parse_ts("2026-11-01T10:00:00Z")) == "2026-11-01T10:00:00Z"
    assert iso(parse_ts("2026-11-01T12:00:00+02:00")) == "2026-11-01T10:00:00Z"
    assert iso(parse_ts("Sun Nov 01 10:00:00 +0000 2026")) == "2026-11-01T10:00:00Z"
    assert iso(parse_ts(1793527200)) == "2026-11-01T10:00:00Z"
    # naive timestamps are refused rather than guessed
    assert parse_ts("2026-11-01T10:00:00") is None
    assert parse_ts("2d") is None


def test_gate_marks_unverified_and_window(post_factory):
    start, end = day_start(date(2026, 11, 1)), day_end(date(2026, 11, 3))
    ok = post_factory("1")
    late = post_factory("2", published_at="2026-11-05T00:00:00Z")
    none = post_factory("3", published_at=None)
    future = post_factory("4", published_at="2026-12-30T00:00:00Z")
    gate_timestamps([ok, late, none, future], start, end)
    assert ok.timestamp_verified and ok.in_window
    assert late.timestamp_verified and not late.in_window
    assert not none.timestamp_verified and none.timestamp_note
    assert not future.timestamp_verified
    assert [p.post_id for p in analysable([ok, late, none, future])] == ["x:1"]


def test_dedupe_same_post_id_only(post_factory):
    a = post_factory("1", "same text", query="q1")
    b = post_factory("1", "same text", query="q2")
    c = post_factory("2", "same text")
    out = dedupe([a, b, c])
    assert [p.post_id for p in out] == ["x:1", "x:2"]
    assert out[0].query == "q1 | q2"


def test_validate_inputs():
    validate_inputs("acme", date(2026, 11, 1), date(2026, 11, 30), 30)
    with pytest.raises(RunInputError):
        validate_inputs("acme", date(2026, 11, 1), date(2026, 12, 1), 30)
    with pytest.raises(RunInputError):
        validate_inputs(" ", date(2026, 11, 1), date(2026, 11, 2), 30)


def test_config_save_and_redact():
    save_config({"YOUTUBE_API_KEY": "AIzaSyFAKEFAKEFAKEFAKEFAKE1234", "BOGUS": "x"})
    cfg = load_config()
    assert cfg["YOUTUBE_API_KEY"].startswith("AIza")
    msg = redact("error calling ?key=AIzaSyFAKEFAKEFAKEFAKEFAKE1234 failed", cfg)
    assert "FAKE" not in msg
    assert "sk-ant-" not in redact("token sk-ant-abcdefghijklmnop")


def test_edge_serializes_from_alias():
    e = Edge(**{"from": "x:1", "to": "x:2", "type": "direct", "basis": "quote"})
    assert dump(e)["from"] == "x:1"


def test_read_jsonl_keeps_unicode_line_separators(tmp_path, post_factory):
    """U+2028 inside post text must not split a JSONL record (live bug, 2026-11-18)."""
    from socmint.runstore import read_jsonl, write_jsonl

    posts = [post_factory("1", "first line\u2028second\u2029third\x85end"), post_factory("2", "ok")]
    p = tmp_path / "posts.jsonl"
    write_jsonl(p, posts)
    rows = read_jsonl(p)
    assert [r["post_id"] for r in rows] == ["x:1", "x:2"]
    assert rows[0]["text"] == "first line\u2028second\u2029third\x85end"
