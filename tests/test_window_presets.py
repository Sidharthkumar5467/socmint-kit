"""Window presets and LLM savings (owner decision 2026-11-20, docs/DECISIONS.md).

New Run offers "last 1 / 3 / 7 days" windows ending now, each with its own per-source
cap; a custom start/end range stays available. Small narratives get keyword labels and
claims are extracted only for the preliminary main list. No network, no real LLM.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from conftest import make_post
from fastapi.testclient import TestClient

from socmint import cli, pipeline
from socmint.analyze import brief as B
from socmint.analyze import narratives
from socmint.analyze.claims import MAX_NARRATIVES, assess_claims
from socmint.interfaces import CollectResult
from socmint.runstore import read_json
from socmint.schema import Narrative, RunMeta, SourceState, SourceStatus
from socmint.server.app import create_app
from socmint.server.jobs import Job, JobManager

NOW = datetime(2026, 11, 20, 9, 30, 15, 123456, tzinfo=UTC)
WEB_JS = (Path(__file__).resolve().parent.parent / "socmint" / "web" / "app.js").read_text(
    encoding="utf-8")


# --------------------------------------------------------------- window computation
def test_preset_window_ends_now_and_is_exact():
    s, e = pipeline.preset_window("1d", NOW)
    assert e == datetime(2026, 11, 20, 9, 30, 15, tzinfo=UTC)  # whole seconds
    assert s == datetime(2026, 11, 19, 9, 30, 15, tzinfo=UTC)  # last 24 hours, not 2 days
    s3, e3 = pipeline.preset_window("3d", NOW)
    assert e3 - s3 == timedelta(days=3)
    s7, e7 = pipeline.preset_window("7d", NOW)
    assert e7 - s7 == timedelta(days=7)
    with pytest.raises(pipeline.RunInputError):
        pipeline.preset_window("2d", NOW)


def test_preset_window_defaults_to_current_utc_time():
    before = datetime.now(UTC).replace(microsecond=0)
    _, e = pipeline.preset_window("3d")
    assert before <= e <= datetime.now(UTC) and e.tzinfo is not None


def test_window_days_and_validation_for_dates_and_datetimes():
    assert pipeline.window_days(date(2026, 11, 1), date(2026, 11, 7)) == 7
    s, e = pipeline.preset_window("7d", NOW)
    assert pipeline.window_days(s, e) == 7  # touches 8 UTC days, but is 7 days long
    pipeline.validate_inputs("heatwave", s, e, 7)  # allowed when the maximum is 7
    with pytest.raises(pipeline.RunInputError, match="maximum is 3"):
        pipeline.validate_inputs("heatwave", s, e, 3)
    with pytest.raises(pipeline.RunInputError, match="before start"):
        pipeline.validate_inputs("heatwave", e, s, 30)


def test_presets_and_caps_are_the_owner_values():
    assert pipeline.WINDOW_PRESETS == {"1d": 1, "3d": 3, "7d": 7}
    assert pipeline.PRESET_LIMITS == {"1d": 150, "3d": 300, "7d": 500}
    assert pipeline.DEFAULT_PRESET == "3d" and pipeline.CUSTOM_LIMIT == 500
    assert tuple(pipeline.WINDOW_PRESETS) == cli.WINDOW_PRESET_NAMES


def test_run_with_preset_bounds_collects_exact_window(monkeypatch, tmp_path):
    """Collectors get the exact datetime bounds; run.json keeps the UTC dates touched and
    records the exact bounds in options; posts outside the 24 h are out of window."""
    seen = {}

    def fake_collect_one(name, query, ctx, prog):
        seen.update(start=query.start, end=query.end, limit=query.limit)
        posts = [make_post("in", "heatwave alert for the region", source=name,
                           published_at="2026-11-20T08:00:00Z"),
                 make_post("old", "heatwave alert yesterday morning", source=name,
                           published_at="2026-11-19T08:00:00Z")]
        return CollectResult(posts=posts, status=[SourceStatus(
            source=name, query=query.topic, state=SourceState.ok, count=2)])

    monkeypatch.setattr(pipeline, "_collect_one", fake_collect_one)
    s, e = pipeline.preset_window("1d", NOW)
    run_dir = pipeline.run("heatwave", s, e, ["youtube"], limit=150, out_root=tmp_path,
                           options={"window_preset": "1d"}, cfg={})
    assert seen == {"start": s, "end": e, "limit": 150}
    meta = read_json(run_dir / "run.json")
    assert (meta["start"], meta["end"]) == ("2026-11-19", "2026-11-20")
    assert meta["options"]["window_preset"] == "1d" and meta["options"]["limit"] == 150
    assert meta["options"]["window_start_at"] == "2026-11-19T09:30:15Z"
    assert meta["options"]["window_end_at"] == "2026-11-20T09:30:15Z"
    assert meta["counts"]["posts_out_of_window"] == 1  # 08:00 on the 19th is > 24 h ago


def test_custom_date_run_keeps_whole_days(monkeypatch, tmp_path):
    seen = {}

    def fake_collect_one(name, query, ctx, prog):
        seen.update(start=query.start, end=query.end)
        return CollectResult(posts=[], status=[SourceStatus(
            source=name, query=query.topic, state=SourceState.completed_zero)])

    monkeypatch.setattr(pipeline, "_collect_one", fake_collect_one)
    run_dir = pipeline.run("heatwave", date(2026, 11, 1), date(2026, 11, 3), ["x"],
                           out_root=tmp_path, cfg={})
    assert seen["start"] == datetime(2026, 11, 1, tzinfo=UTC)
    assert seen["end"] == datetime(2026, 11, 3, 23, 59, 59, tzinfo=UTC)
    meta = read_json(run_dir / "run.json")
    assert (meta["start"], meta["end"]) == ("2026-11-01", "2026-11-03")


# ---------------------------------------------------------------------------- CLI
def _cli_run(monkeypatch, tmp_path, argv):
    seen = {}

    def fake_run(topic, start, end, sources, *, limit, options, progress, out_root):
        seen.update(start=start, end=end, limit=limit, options=options)
        d = tmp_path / "r"
        d.mkdir(exist_ok=True)
        (d / "run.json").write_text('{"status": "complete"}', encoding="utf-8")
        return d

    monkeypatch.setattr(pipeline, "run", fake_run)
    return cli.main(["run", "--topic", "heatwave", *argv]), seen


def test_cli_parses_window_presets():
    a = cli.build_parser().parse_args(["run", "--topic", "t", "--window", "1d"])
    assert a.window == "1d" and a.start is None and a.end is None and a.limit is None
    with pytest.raises(SystemExit):  # unknown preset
        cli.build_parser().parse_args(["run", "--topic", "t", "--window", "2d"])
    with pytest.raises(SystemExit):  # mutually exclusive with --start
        cli.build_parser().parse_args(["run", "--topic", "t", "--window", "3d",
                                       "--start", "2026-11-01", "--end", "2026-11-02"])
    with pytest.raises(SystemExit):  # one of them is required
        cli.build_parser().parse_args(["run", "--topic", "t"])


@pytest.mark.parametrize("preset,cap,days", [("1d", 150, 1), ("3d", 300, 3), ("7d", 500, 7)])
def test_cli_window_applies_preset_cap(monkeypatch, tmp_path, preset, cap, days):
    rc, seen = _cli_run(monkeypatch, tmp_path, ["--window", preset])
    assert rc == 0 and seen["limit"] == cap
    assert seen["end"] - seen["start"] == timedelta(days=days)
    assert seen["options"] == {"match": "phrase", "window_preset": preset}


def test_cli_custom_range_default_and_explicit_limit(monkeypatch, tmp_path):
    rc, seen = _cli_run(monkeypatch, tmp_path, ["--start", "2026-11-01", "--end", "2026-11-05"])
    assert rc == 0 and seen["limit"] == 500 and seen["options"]["window_preset"] == "custom"
    rc, seen = _cli_run(monkeypatch, tmp_path, ["--start", "2026-11-01", "--end", "2026-11-05",
                                                "--limit", "80"])
    assert rc == 0 and seen["limit"] == 80


@pytest.mark.parametrize("argv,err", [
    (["--window", "3d", "--limit", "50"], "--limit cannot be combined"),
    (["--window", "3d", "--end", "2026-11-05"], "--end cannot be combined"),
    (["--start", "2026-11-01"], "--start needs --end"),
])
def test_cli_window_bad_combinations_exit_2(monkeypatch, tmp_path, capsys, argv, err):
    rc, seen = _cli_run(monkeypatch, tmp_path, argv)
    assert rc == 2 and seen == {} and err in capsys.readouterr().err


# ---------------------------------------------------------------------------- API
class _RecJobs(JobManager):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def start(self, *, topic, start, end, sources, limit, out_root, cfg, options=None,
              timeout=60.0):
        self.calls.append({"start": start, "end": end, "limit": limit, "options": options})
        return Job(run_id="2026-11-20-heatwave-aaaaaa", done=True)


@pytest.fixture
def api(tmp_path):
    jobs = _RecJobs()
    c = TestClient(create_app(initial_run=None, jobs=jobs), base_url="http://127.0.0.1")
    return c, jobs


def _post(c, **body):
    return c.post("/api/runs", json={"topic": "heatwave", "sources": ["youtube"], **body})


@pytest.mark.parametrize("preset,cap,days", [("1d", 150, 1), ("3d", 300, 3), ("7d", 500, 7)])
def test_api_window_preset_starts_run_with_cap(api, preset, cap, days):
    c, jobs = api
    r = _post(c, window=preset)
    assert r.status_code == 200, r.text
    call = jobs.calls[-1]
    assert call["limit"] == cap and call["options"]["window_preset"] == preset
    assert call["end"] - call["start"] == timedelta(days=days)
    assert abs((datetime.now(UTC) - call["end"]).total_seconds()) < 60


def test_api_custom_range_still_works(api):
    c, jobs = api
    assert _post(c, start="2026-11-01", end="2026-11-03").status_code == 200
    assert jobs.calls[-1]["limit"] == 500
    assert jobs.calls[-1]["options"]["window_preset"] == "custom"
    assert jobs.calls[-1]["start"] == date(2026, 11, 1)
    assert _post(c, start="2026-11-01", end="2026-11-03", limit=40).status_code == 200
    assert jobs.calls[-1]["limit"] == 40


@pytest.mark.parametrize("body,detail", [
    ({"window": "2d"}, "window must be one of"),
    ({"window": "3d", "start": "2026-11-01", "end": "2026-11-03"}, "not both"),
    ({"window": "3d", "limit": 50}, "limit cannot be combined"),
    ({}, "window preset (1d, 3d, 7d) or both"),
    ({"start": "2026-11-01"}, "window preset (1d, 3d, 7d) or both"),
])
def test_api_window_validation(api, body, detail):
    c, jobs = api
    r = _post(c, **body)
    assert r.status_code == 400 and detail in r.json()["detail"]
    assert jobs.calls == []


def test_api_meta_lists_presets(api):
    c, _ = api
    m = c.get("/api/meta").json()
    assert m["window_presets"] == [{"id": "1d", "days": 1, "limit": 150},
                                   {"id": "3d", "days": 3, "limit": 300},
                                   {"id": "7d", "days": 7, "limit": 500}]
    assert m["default_window"] == "3d" and m["custom_limit"] == 500


# ---------------------------------------------------------- narrative label threshold
class _LabelLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def parse(self, schema, prompt, *, system="", purpose="", max_tokens=4096):
        self.calls.append(purpose)
        return schema(label="Reports of a dam failure upstream",
                      claim_summary="Posts claim the dam failed.", relevance="related",
                      severity="medium", severity_reason="Public safety rumour.")


def test_small_narratives_get_keyword_labels_without_llm_call():
    posts = [
        make_post("a", "The dam upstream of the valley burst overnight, villages flooded",
                  author="u1"),
        make_post("b", "Dam burst overnight upstream, the valley villages are flooded",
                  author="u2", source="telegram"),
        make_post("c", "Villages in the valley flooded after the upstream dam burst",
                  author="u3", source="youtube"),
        make_post("d", "New vaccine schedule for children announced by the ministry",
                  author="u4"),
        make_post("e", "Ministry announces children's vaccine schedule changes",
                  author="u5"),
    ]
    llm = _LabelLLM()
    stats: dict = {}
    narrs = narratives.build_narratives(posts, topic="flood", llm=llm, stats=stats)
    sizes = {n.id: len(n.member_post_ids) for n in narrs}
    assert sorted(sizes.values()) == [2, 3]
    big = next(n for n in narrs if len(n.member_post_ids) == 3)
    small = next(n for n in narrs if len(n.member_post_ids) == 2)
    assert llm.calls == [f"narrative_label:{big.id}"]  # one call, for the 3-post one
    assert big.label_method == "llm" and big.severity == "medium"
    assert small.label_method == "keyword" and small.severity is None
    assert stats["narratives_label_llm"] == 1 and stats["narratives_label_keyword"] == 1
    assert stats["narratives_label_small"] == 1
    assert narratives.LLM_LABEL_MIN_POSTS == 3


def test_label_stats_without_llm_count_keyword_only():
    posts = [make_post(str(i), t, author=f"u{i}") for i, t in enumerate([
        "Wildfire smoke closes schools across the county today",
        "Schools across the county closed today due to wildfire smoke",
        "County schools shut because of wildfire smoke"])]
    stats: dict = {}
    narrs = narratives.build_narratives(posts, topic="wildfire", llm=None, stats=stats)
    assert stats["narratives_label_llm"] == 0
    assert stats["narratives_label_keyword"] == len(narrs)
    assert stats["narratives_label_small"] == 0  # no LLM: nothing was skipped for size


# ------------------------------------------------------------ claims narrative choice
META = RunMeta(run_id="r", topic="recall", start="2026-11-13", end="2026-11-19",
               sources=["x"], created_at="2026-11-19T00:00:00Z")


def _n(nid, k, severity=None, relevance=None, start=0):
    posts = [make_post(f"{nid}-{i}", f"{nid} recall claim {i}", author=f"{nid}{i}",
                       published_at=f"2026-11-{13 + (start + i) % 7:02d}T12:00:00Z")
             for i in range(k)]
    vol: dict[str, int] = {}
    for p in posts:
        vol[p.published_at[:10]] = vol.get(p.published_at[:10], 0) + 1
    n = Narrative(id=nid, label=f"label {nid}", claim_summary=f"{nid} claims a recall",
                  member_post_ids=[p.post_id for p in posts],
                  example_post_ids=[posts[0].post_id], volume_by_day=vol,
                  volume_by_platform={"x": k}, severity=severity, relevance=relevance,
                  label_method="llm" if severity else "keyword")
    return n, posts


def test_claim_candidates_follow_preliminary_ranking_not_size():
    """A 12-post high-severity narrative outranks a 40-post low one; off-topic is left out;
    n = min(MAX_NARRATIVES, max(main list size, CLAIMS_MIN_NARRATIVES))."""
    big_low, p1 = _n("N1", 40, "low")
    high, p2 = _n("N2", 12, "high")
    off, p3 = _n("N3", 30, "medium", relevance="off_topic")
    tiny, p4 = _n("N4", 2)
    posts = p1 + p2 + p3 + p4
    order, n = B.claim_candidates(posts, [big_low, high, off, tiny], [], META, MAX_NARRATIVES)
    assert order[:2] == ["N2", "N1"] and "N3" not in order and order[-1] == "N4"
    assert n == B.CLAIMS_MIN_NARRATIVES  # main list has 2 (N2, N1); floor of 3 applies


def test_claim_candidates_put_small_severe_narratives_before_low_main_ones():
    """Review 2026-11-20 (live re-run: N82 'urging people to pull money', 5 posts, high,
    got no claim slot while low main narratives did). Non-bank: a 5-post high narrative
    about a product recall cover-up outranks a 40-post low one for claims, though only
    the low one is on the main list (5 posts, not coordinated, no claims yet)."""
    big_low, p1 = _n("N1", 40, "low")
    small_high, p2 = _n("N2", 5, "high")
    four_high, p3 = _n("N3", 4, "critical")  # below MAIN_MIN_POSTS_SEVERE: not promoted
    order, n = B.claim_candidates(p1 + p2 + p3, [big_low, small_high, four_high], [], META,
                                  MAX_NARRATIVES)
    ranks = B.rank_narratives([big_low, small_high, four_high], [], [], p1 + p2 + p3, META)
    assert [r.narrative_id for r in ranks if r.main] == ["N1"]
    assert order[:2] == ["N2", "N1"] and order[-1] == "N3"
    assert n == B.CLAIMS_MIN_NARRATIVES


def test_claim_candidates_capped_by_main_list_size():
    narrs, posts = [], []
    for i in range(12):
        nn, pp = _n(f"N{i + 1}", 10 + i, "medium", start=i)
        narrs.append(nn)
        posts += pp
    order, n = B.claim_candidates(posts, narrs, [], META, MAX_NARRATIVES)
    assert n == MAX_NARRATIVES  # 12 main narratives, capped at MAX_NARRATIVES (10)
    order, n = B.claim_candidates(posts, narrs[:5], [], META, MAX_NARRATIVES)
    assert n == 5  # main list (5) smaller than 8: only those


def test_assess_claims_uses_given_order_and_limit(tmp_path):
    a, pa = _n("N1", 30)
    b, pb = _n("N2", 6)
    c, pc = _n("N3", 4)
    posts = pa + pb + pc
    out = assess_claims(posts, [a, b, c], topic="recall", llm=None, cfg={}, run_dir=tmp_path,
                        narrative_order=["N3", "N2", "N1"], max_narratives=2)
    assert [x.narrative_id for x in out] == ["N3", "N2"]
    # without an order, the largest are taken (older callers)
    out = assess_claims(posts, [a, b, c], topic="recall", llm=None, cfg={}, run_dir=tmp_path,
                        max_narratives=1)
    assert [x.narrative_id for x in out] == ["N1"]


def test_pipeline_records_label_and_claims_counts(monkeypatch, tmp_path):
    def fake_collect_one(name, query, ctx, prog):
        posts = [make_post(f"{i}", f"Product recall of baby strollers announced, batch {i}",
                           source=name, author=f"u{i}",
                           published_at=f"2026-11-0{1 + i % 3}T12:00:00Z") for i in range(6)]
        return CollectResult(posts=posts, status=[SourceStatus(
            source=name, query=query.topic, state=SourceState.ok, count=6)])

    monkeypatch.setattr(pipeline, "_collect_one", fake_collect_one)
    run_dir = pipeline.run("stroller recall", date(2026, 11, 1), date(2026, 11, 3), ["x"],
                           out_root=tmp_path, cfg={})
    counts = read_json(run_dir / "run.json")["counts"]
    assert {"narratives_label_llm", "narratives_label_keyword", "narratives_label_small",
            "narratives_claims_assessed"} <= set(counts)
    assert counts["narratives_label_llm"] == 0  # no LLM configured
    assert counts["narratives_label_keyword"] == counts["narratives"]


# ---------------------------------------------------------------------- UI wiring
def test_new_run_page_wires_presets():
    m = re.search(r"async function pageNew\(main, seq\) \{(.*?)\n  function windowDays", WEB_JS, re.S)
    assert m, "pageNew missing"
    src = m.group(1)
    assert "role: 'radiogroup'" in src and "role: 'radio'" in src
    assert "'aria-checked'" in src and "ArrowRight" in src and "tabIndex" in src
    assert "Custom range" in src and "'aria-expanded'" in src and "custom-range hidden" in src
    assert "window: win" in src  # a preset run posts the preset, not dates
    assert "start: start.value, end: end.value" in src  # the custom range still posts dates
    assert "meta.window_presets" in src and "DEFAULT_WINDOW" in src


def test_results_header_shows_preset_label():
    assert "windowPresetLabel(r)" in WEB_JS and "windowBounds(r)" in WEB_JS
