"""No default run time limit, analyst stop, scaled custom limits and the LinkedIn Ads
lookback (owner decision 2026-11-20, docs/DECISIONS.md)."""

from __future__ import annotations

import math
import shutil
import threading
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from socmint import pipeline
from socmint.collect import linkedin_ads as la
from socmint.config import run_time_budget
from socmint.interfaces import CollectContext, CollectResult, Query
from socmint.runstore import read_json, write_json
from socmint.schema import AdMeta, Post, SourceState, SourceStatus
from socmint.server.app import create_app
from socmint.server.jobs import JobManager
from socmint.util.http import HttpFailure, request
from socmint.verify import analysable, gate_timestamps, in_scope

FIXTURES = Path(__file__).parent / "fixtures" / "linkedin_ads"


# ------------------------------------------------------------------ run time limit
@pytest.mark.parametrize("value,expected", [
    ("0", None), ("", None), (None, None), ("-5", None), ("abc", None), ("nan", None),
    ("inf", None), ("600", 600.0), ("30", 60.0), ("3600", 3600.0),
])
def test_run_time_budget(value, expected):
    cfg = {} if value is None else {"RUN_TIME_BUDGET_SECONDS": value}
    assert run_time_budget(cfg) == expected


def test_default_config_has_no_time_limit():
    from socmint.config import DEFAULTS

    assert run_time_budget(dict(DEFAULTS)) is None


@pytest.mark.parametrize("value,expected", [
    (None, 12), ("12", 12), ("16", 16), ("1", 1), ("0", 1), ("-4", 1), ("999", 32),
    ("", 12), ("nope", 12),
])
def test_analysis_workers_clamped(value, expected):
    from socmint.config import analysis_workers

    cfg = {} if value is None else {"ANALYSIS_LLM_WORKERS": value}
    assert analysis_workers(cfg) == expected


def _ctx(tmp_path, deadline=math.inf) -> CollectContext:
    return CollectContext(run_dir=tmp_path, raw_dir=tmp_path / "raw", config={},
                          deadline=deadline)


def test_unlimited_context_never_runs_out_until_stopped(tmp_path):
    ctx = _ctx(tmp_path)
    assert not ctx.out_of_time() and ctx.time_left() == math.inf
    ctx.stop.set()
    assert ctx.out_of_time() and ctx.time_left() == 0.0


def test_stop_interrupts_http_backoff():
    stop = threading.Event()
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        stop.set()  # the analyst presses Stop while the source is rate limited
        return httpx.Response(429, headers={"Retry-After": "600"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    t0 = time.monotonic()
    with pytest.raises(HttpFailure, match="stopped by the analyst"):
        request(client, "GET", "https://example.test/x", deadline=math.inf, stop=stop)
    assert time.monotonic() - t0 < 5 and calls["n"] == 1


# ------------------------------------------------------------------ custom limits
@pytest.mark.parametrize("days,limit", [(1, 500), (7, 500), (8, 560), (10, 700),
                                        (28, 1960), (30, 2000), (90, 2000)])
def test_custom_limit_scales_with_days(days, limit):
    assert pipeline.custom_limit(days) == limit


# ------------------------------------------------------------------ ads lookback
END = datetime(2026, 11, 19, 23, 59, 59, tzinfo=UTC)
START = datetime(2026, 11, 13, tzinfo=UTC)


def test_ads_window_start():
    assert pipeline.ads_window_start(START, END, 0) == START
    assert pipeline.ads_window_start(START, END, None) == START
    assert pipeline.ads_window_start(START, END, 90) == END - timedelta(days=90)
    assert pipeline.ads_window_start(START, END, 5000) == END - timedelta(days=365)
    # a lookback shorter than the window never narrows it
    assert pipeline.ads_window_start(START, END, 3) == START


def _ad(pid: str, run_start: str, run_end: str) -> Post:
    return Post(post_id=pid, source="linkedin_ads", kind="ad", text="ad", url="https://x.test",
                published_at=run_start, collected_at="2026-11-19T12:00:00Z",
                timestamp_verified=True, ad_meta=AdMeta(run_start=run_start, run_end=run_end))


def test_lookback_ads_are_in_scope_but_never_on_timelines():
    old = _ad("linkedin_ads:1", "2026-09-01T00:00:00Z", "2026-09-30T23:59:59Z")
    gate_timestamps([old], START, END)
    assert old.active_in_window is False and old not in in_scope([old])
    old = _ad("linkedin_ads:1", "2026-09-01T00:00:00Z", "2026-09-30T23:59:59Z")
    gate_timestamps([old], START, END, ads_start=END - timedelta(days=90))
    assert old.active_in_window is True and old.in_window is False
    assert in_scope([old]) == [old] and analysable([old]) == []
    # outside even the lookback: still out of scope
    older = _ad("linkedin_ads:2", "2026-03-01T00:00:00Z", "2026-03-30T23:59:59Z")
    gate_timestamps([older], START, END, ads_start=END - timedelta(days=90))
    assert older.active_in_window is False and in_scope([older]) == []


def test_ads_collector_searches_the_lookback_span(tmp_path):
    q = Query(topic="bank", start=datetime(2026, 11, 10, tzinfo=UTC),
              end=datetime(2026, 11, 17, 23, 59, 59, tzinfo=UTC),
              options={"ads_start_at": "2026-08-19T23:59:59Z"})
    col = la.LinkedInAdsCollector(sleep=lambda s: None, today=lambda: date(2026, 11, 18))
    with respx.mock(assert_all_called=False) as router:
        route = router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(200, text=(FIXTURES / "live_search_bank_noresults.html")
                                        .read_text(encoding="utf-8")))
        res = col.collect(q, _ctx(tmp_path, time.monotonic() + 600))
    (st,) = res.status
    assert st.state == SourceState.completed_zero
    assert "custom-date-range 2026-08-19..2026-11-17" in st.reason
    assert "ads lookback from 2026-08-19" in st.reason
    assert route.calls[0].request.url.params["startdate"] == "2026-08-19"


# ------------------------------------------------------------------ pipeline
def _fake_collect(seen: dict, block: threading.Event | None = None):
    def fake_collect_one(name, query, ctx, prog):
        seen[name] = {"options": dict(query.options), "deadline": ctx.deadline}
        if block is not None:  # a source still working when the analyst stops the run
            assert block.wait(10)
            assert ctx.out_of_time()
            return CollectResult(posts=[], status=[SourceStatus(
                source=name, query=query.topic, state=SourceState.partial,
                reason="time budget exhausted after 3 page(s)")])
        return CollectResult(posts=[], status=[SourceStatus(
            source=name, query=query.topic, state=SourceState.completed_zero)])
    return fake_collect_one


def test_pipeline_no_time_limit_and_ads_lookback(monkeypatch, tmp_path):
    seen: dict = {}
    monkeypatch.setattr(pipeline, "_collect_one", _fake_collect(seen))
    run_dir = pipeline.run("topic", date(2026, 11, 13), date(2026, 11, 19), ["linkedin_ads"],
                           out_root=tmp_path, cfg={},
                           options={"ads_lookback_days": 90, "window_preset": "custom"})
    assert seen["linkedin_ads"]["deadline"] == math.inf
    assert seen["linkedin_ads"]["options"]["ads_start_at"] == "2026-08-21T23:59:59Z"
    opts = read_json(run_dir / "run.json")["options"]
    assert opts["ads_lookback_days"] == 90 and opts["time_limit_s"] is None
    assert opts["ads_window_start_at"] == "2026-08-21T23:59:59Z"


def test_pipeline_lookback_not_longer_than_window_is_dropped(monkeypatch, tmp_path):
    seen: dict = {}
    monkeypatch.setattr(pipeline, "_collect_one", _fake_collect(seen))
    run_dir = pipeline.run("topic", date(2026, 10, 1), date(2026, 10, 30), ["linkedin_ads"],
                           out_root=tmp_path, cfg={}, options={"ads_lookback_days": 0})
    assert "ads_start_at" not in seen["linkedin_ads"]["options"]
    assert "ads_lookback_days" not in read_json(run_dir / "run.json")["options"]


def test_pipeline_time_limit_when_configured(monkeypatch, tmp_path):
    seen: dict = {}
    monkeypatch.setattr(pipeline, "_collect_one", _fake_collect(seen))
    pipeline.run("topic", date(2026, 11, 13), date(2026, 11, 19), ["youtube"],
                 out_root=tmp_path, cfg={"RUN_TIME_BUDGET_SECONDS": "600"})
    assert seen["youtube"]["deadline"] < math.inf


def test_pipeline_analyst_stop_discards_the_whole_run(monkeypatch, tmp_path):
    """A hard Stop abandons the run: run() returns None and the partial run directory is
    deleted, so no posts, status, export or report survive (owner decision 2026-11-20)."""
    seen: dict = {}
    stop = threading.Event()
    run_id: dict = {}
    monkeypatch.setattr(pipeline, "_collect_one", _fake_collect(seen, block=stop))

    def on_progress(ev):
        if ev.get("run_id"):
            run_id["id"] = ev["run_id"]

    out: dict = {}
    t = threading.Thread(target=lambda: out.update(d=pipeline.run(
        "topic", date(2026, 11, 13), date(2026, 11, 19), ["youtube"], out_root=tmp_path,
        cfg={}, stop=stop, progress=on_progress)))
    t.start()
    time.sleep(0.2)
    stop.set()
    t.join(30)
    assert out["d"] is None                       # cancelled run returns no directory
    assert not (tmp_path / run_id["id"]).exists()  # partial run was deleted, no report left


def test_mark_analyst_stop_leaves_other_reasons():
    sts = [SourceStatus(source="x", query="t", state=SourceState.partial,
                        reason="slice 1: time budget exhausted before this slice"),
           SourceStatus(source="x", query="t", state=SourceState.ok, reason="all good")]
    assert pipeline.mark_analyst_stop(sts) == 1
    assert sts[0].reason == "slice 1: stopped by the analyst before this slice"
    assert sts[1].reason == "all good"


# ------------------------------------------------------------------ server
def _client(jobs: JobManager, run_dir: Path) -> TestClient:
    # Host must be 127.0.0.1 (the app rejects other hosts); see tests/test_server.py.
    return TestClient(create_app(initial_run=run_dir, jobs=jobs), base_url="http://127.0.0.1")


@pytest.fixture
def run_dir() -> Path:
    from test_server import build_run

    from socmint.config import runs_dir

    return build_run(runs_dir())


def test_server_custom_limit_lookback_and_stop(run_dir):
    release = threading.Event()
    seen: dict = {}

    def fake_run(topic, start, end, sources, *, limit, options, progress, out_root, cfg,
                 stop=None):
        d = out_root / "fake-run-2"
        d.mkdir(parents=True)
        write_json(d / "run.json", {"run_id": "fake-run-2", "topic": topic,
                                    "status": "running", "created_at": "2026-11-18T00:00:00Z"})
        seen.update(limit=limit, options=options, stop=stop)
        progress({"stage": "collect", "run_id": "fake-run-2"})
        write_json(d / "progress.json", {"stage": "collect", "done": False, "sources": {},
                                         "log": [], "pid": __import__("os").getpid()})
        release.wait(10)
        return d

    jobs = JobManager(runner=fake_run)
    c = _client(jobs, run_dir)
    bad = c.post("/api/runs", json={"topic": "t", "start": "2026-10-01", "end": "2026-10-30",
                                    "sources": ["linkedin_ads"]})
    assert bad.status_code == 400  # linkedin_ads was removed from the active source set
    r = c.post("/api/runs", json={"topic": "t", "start": "2026-10-01", "end": "2026-10-30",
                                  "sources": ["linkedin"]})
    assert r.status_code == 200, r.text
    assert seen["limit"] == 2000  # 30-day custom range, no limit typed
    prog = c.get("/api/runs/fake-run-2/progress").json()
    assert prog["can_stop"] is True and prog["stopping"] is False
    assert c.post("/api/runs/other-run/stop").status_code == 409
    assert c.post("/api/runs/fake-run-2/stop").json() == {"run_id": "fake-run-2",
                                                          "stopping": True, "cancel": True}
    assert seen["stop"].is_set()
    prog = c.get("/api/runs/fake-run-2/progress").json()
    assert prog["stopping"] is True and prog["can_stop"] is False
    release.set()
    jobs.wait(10)
    assert c.post("/api/runs/fake-run-2/stop").status_code == 409


def test_server_reports_cancelled_after_run_deleted(run_dir):
    """After a hard Stop, pipeline.run returns None and deletes the run dir; the server
    still reports a clean cancelled/done state (no 404 spin) so the GUI can navigate away."""
    release = threading.Event()

    def fake_run(topic, start, end, sources, *, limit, options, progress, out_root, cfg,
                 stop=None):
        d = out_root / "fake-cancel"
        d.mkdir(parents=True)
        progress({"stage": "collect", "run_id": "fake-cancel"})
        write_json(d / "progress.json", {"stage": "collect", "done": False, "sources": {},
                                         "log": [], "pid": __import__("os").getpid()})
        release.wait(10)
        stop.set()          # emulate the analyst's Stop arriving during collection
        shutil.rmtree(d)    # pipeline.run deletes the partial run on cancel
        return None         # cancelled run returns no directory

    jobs = JobManager(runner=fake_run)
    c = _client(jobs, run_dir)
    r = c.post("/api/runs", json={"topic": "t", "start": "2026-11-18", "end": "2026-11-19",
                                  "sources": ["youtube"]})
    assert r.status_code == 200, r.text
    release.set()
    jobs.wait(10)
    assert jobs.current.cancelled is True
    prog = c.get("/api/runs/fake-cancel/progress").json()
    assert prog["done"] is True and prog["cancelled"] is True
    assert "can_stop" not in prog or prog["can_stop"] is not True
    # the discarded run must not resurface as a phantom "running" row in the run list
    runs = c.get("/api/runs").json()
    rows = runs["runs"] if isinstance(runs, dict) else runs
    assert "fake-cancel" not in [r["run_id"] for r in rows]


def test_meta_reports_limits(run_dir):
    c = _client(JobManager(), run_dir)
    m = c.get("/api/meta").json()
    assert "ads_lookback_choices" not in m  # LinkedIn Ads removed from the active source set
    assert m["custom_limit_per_day"] == 70 and m["custom_limit_max"] == 2000
    assert m["time_limit_s"] is None


def test_settings_accept_zero_time_limit(run_dir):
    c = _client(JobManager(), run_dir)
    assert c.post("/api/settings", json={"RUN_TIME_BUDGET_SECONDS": "0"}).status_code == 200
    assert c.post("/api/settings", json={"RUN_TIME_BUDGET_SECONDS": "90000"}).status_code == 400


# ------------------------------------------------------------------ review fixes
def test_future_start_is_rejected():
    future = datetime.now(UTC).date() + timedelta(days=2)
    with pytest.raises(pipeline.RunInputError, match="in the future"):
        pipeline.validate_inputs("t", future, future, 30)


def test_future_end_is_clamped_to_now(monkeypatch, tmp_path):
    # Seen live 2026-11-20: a 1-30 Nov range on 20 Nov made YouTube fail (HTTP 400 for a
    # slice entirely in the future) and X search empty future slices.
    seen: dict = {}

    def fake(name, query, ctx, prog):
        seen["end"] = query.end
        return CollectResult(posts=[], status=[SourceStatus(
            source=name, query=query.topic, state=SourceState.completed_zero)])
    monkeypatch.setattr(pipeline, "_collect_one", fake)
    today = datetime.now(UTC).date()
    run_dir = pipeline.run("t", today - timedelta(days=19), today + timedelta(days=10),
                           ["youtube"], out_root=tmp_path, cfg={})
    meta = read_json(run_dir / "run.json")
    assert seen["end"] <= datetime.now(UTC)
    assert meta["end"] == today.isoformat()
    assert meta["options"]["window_end_requested"] == (today + timedelta(days=10)).isoformat()


def test_end_today_is_not_flagged(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "_collect_one", _fake_collect({}))
    today = datetime.now(UTC).date()
    run_dir = pipeline.run("t", today - timedelta(days=2), today, ["youtube"],
                           out_root=tmp_path, cfg={})
    assert "window_end_requested" not in read_json(run_dir / "run.json")["options"]


def test_lookback_ignored_without_linkedin_ads(monkeypatch, tmp_path):
    seen: dict = {}
    monkeypatch.setattr(pipeline, "_collect_one", _fake_collect(seen))
    run_dir = pipeline.run("t", date(2026, 11, 13), date(2026, 11, 19), ["youtube"],
                           out_root=tmp_path, cfg={}, options={"ads_lookback_days": 365})
    assert "ads_start_at" not in seen["youtube"]["options"]
    opts = read_json(run_dir / "run.json")["options"]
    assert "ads_lookback_days" not in opts and "ads_window_start_at" not in opts


def test_stop_during_backoff_is_partial_not_failed():
    sts = [SourceStatus(source="x", query="t", state=SourceState.failed,
                        reason="HTTP failure on page 1: stopped by the analyst while backing off"),
           SourceStatus(source="youtube", query="t", state=SourceState.failed,
                        reason="HTTP 403: key invalid")]
    pipeline.mark_analyst_stop(sts)
    assert sts[0].state == SourceState.partial
    assert sts[1].state == SourceState.failed  # a real failure stays a failure


def test_undated_lookback_ad_is_not_in_window():
    ad = Post(post_id="linkedin_ads:9", source="linkedin_ads", kind="ad", text="ad",
              url="https://x.test", collected_at="2026-11-19T12:00:00Z",
              timestamp_verified=False, ad_meta=AdMeta())
    gate_timestamps([ad], START, END, ads_start=END - timedelta(days=365))
    assert ad.in_window is False and "may have run only before" in ad.timestamp_note
    assert in_scope([ad]) == [ad]  # still grouped into narratives
    plain = Post(post_id="linkedin_ads:8", source="linkedin_ads", kind="ad", text="ad",
                 url="https://x.test", collected_at="2026-11-19T12:00:00Z",
                 timestamp_verified=False, ad_meta=AdMeta())
    gate_timestamps([plain], START, END)  # no lookback: unchanged behaviour
    assert plain.in_window is True


def test_stop_refused_after_collection(run_dir):
    release = threading.Event()

    def fake_run(topic, start, end, sources, *, limit, options, progress, out_root, cfg,
                 stop=None):
        d = out_root / "fake-run-3"
        d.mkdir(parents=True)
        write_json(d / "run.json", {"run_id": "fake-run-3", "topic": topic,
                                    "status": "running", "created_at": "2026-11-18T00:00:00Z"})
        progress({"stage": "collect", "run_id": "fake-run-3"})
        write_json(d / "progress.json", {"stage": "claims", "done": False, "sources": {},
                                         "log": [], "pid": __import__("os").getpid()})
        release.wait(10)
        return d

    jobs = JobManager(runner=fake_run)
    c = _client(jobs, run_dir)
    assert c.post("/api/runs", json={"topic": "t", "window": "1d",
                                     "sources": ["youtube"]}).status_code == 200
    prog = c.get("/api/runs/fake-run-3/progress").json()
    assert prog["can_stop"] is False and prog["stopping"] is False
    assert c.post("/api/runs/fake-run-3/stop").status_code == 409
    release.set()
    jobs.wait(10)


def test_settings_reject_sub_minute_limit(run_dir):
    c = _client(JobManager(), run_dir)
    assert c.post("/api/settings", json={"RUN_TIME_BUDGET_SECONDS": "30"}).status_code == 400
    assert c.post("/api/settings", json={"RUN_TIME_BUDGET_SECONDS": "60"}).status_code == 200


def test_all_time_ads_lookback(monkeypatch, tmp_path):
    # Owner's LinkedIn Ads tool: no date filter, every ad the Ad Library lists.
    seen: dict = {}
    monkeypatch.setattr(pipeline, "_collect_one", _fake_collect(seen))
    run_dir = pipeline.run("t", date(2026, 10, 1), date(2026, 10, 30), ["linkedin_ads"],
                           out_root=tmp_path, cfg={},
                           options={"ads_lookback_days": pipeline.ADS_LOOKBACK_ALL})
    opts = seen["linkedin_ads"]["options"]
    assert opts["ads_all_time"] is True and opts["ads_start_at"].startswith("2000-01-01")
    meta = read_json(run_dir / "run.json")["options"]
    assert meta["ads_all_time"] is True and meta["ads_lookback_days"] == 3650
    assert "ads_window_start_at" not in meta


def test_ads_collector_all_time_sends_no_date_filter(tmp_path):
    q = Query(topic="bank", start=datetime(2026, 10, 1, tzinfo=UTC),
              end=datetime(2026, 10, 30, 23, 59, 59, tzinfo=UTC),
              options={"ads_all_time": True, "ads_start_at": "2000-01-01T00:00:00Z"})
    col = la.LinkedInAdsCollector(sleep=lambda s: None, today=lambda: date(2026, 11, 18))
    with respx.mock(assert_all_called=False) as router:
        route = router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(200, text=(FIXTURES / "live_search_bank_noresults.html")
                                        .read_text(encoding="utf-8")))
        res = col.collect(q, _ctx(tmp_path, time.monotonic() + 600))
    params = route.calls[0].request.url.params
    assert "dateOption" not in params and "startdate" not in params
    assert "no date filter (ads lookback: all time" in res.status[0].reason
