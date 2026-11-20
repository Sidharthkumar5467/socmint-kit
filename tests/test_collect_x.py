"""Tests for the X collector. No network: respx mocks every HTTP call.

Fixtures are doc_example_* (built from provider documentation, synthetic values);
no live X response has been captured yet.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from socmint.collect import x as xmod
from socmint.collect.x import (
    OFFICIAL_URL,
    TWITTERAPI_IO_URL,
    XCollector,
    build_twitterapi_io_query,
    parse_official_page,
    parse_twitterapi_io_page,
)
from socmint.interfaces import CollectContext, Query
from socmint.schema import Post, SourceState
from socmint.verify import analysable, gate_timestamps, in_scope

FX = Path(__file__).parent / "fixtures" / "x"
START = datetime(2026, 11, 1, tzinfo=UTC)
END = datetime(2026, 11, 18, 23, 59, 59, tzinfo=UTC)
# Collect-flow tests default to a one-day window = one time slice (one paginated search);
# multi-slice behaviour has its own tests below.
DAY = datetime(2026, 11, 18, tzinfo=UTC)


def load(name: str) -> dict:
    return json.loads((FX / name).read_text(encoding="utf-8"))


def make_ctx(tmp_path: Path, **cfg: str) -> CollectContext:
    conf = {"X_PROVIDER": "twitterapi_io", "TWITTERAPI_IO_KEY": "test-key-123456", **cfg}
    return CollectContext(run_dir=tmp_path, raw_dir=tmp_path / "raw", config=conf,
                          deadline=time.monotonic() + 60)


def q(limit: int = 500, start: datetime = DAY, end: datetime = END, **opts) -> Query:
    return Query(topic="example topic", start=start, end=end, limit=limit, options=opts)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("socmint.util.http.time.sleep", lambda s: None)


# ----------------------------------------------------------------------- parsing


def test_query_uses_documented_unix_operators():
    s = build_twitterapi_io_query(" example topic ", START, END)
    assert s == (f"(example topic) since_time:{int(START.timestamp())} "
                 f"until_time:{int(END.timestamp()) + 1}")
    assert "since:" not in s  # documented as unsupported


def test_query_parenthesises_or_topics():
    # Without parentheses the time bound could bind to only one side of the OR.
    s = build_twitterapi_io_query("vaccine OR vaccination", START, END)
    assert s.startswith("(vaccine OR vaccination) since_time:")


def test_parse_page_fields_and_relations():
    main, emb = parse_twitterapi_io_page(load("doc_example_advanced_search.json"),
                                         collected_at="2026-11-18T00:00:00Z",
                                         raw_ref="raw/x/page_1.json", query="t")
    assert [p.post_id for p in main] == ["x:1900000000000000001", "x:1900000000000000002"]
    p = main[0]
    assert p.url == "https://x.com/example_user/status/1900000000000000001"
    assert p.author_handle == "@example_user" and p.author_id == "111"
    assert p.author_created_at == "2007-12-19T20:20:32Z"
    assert p.author_avatar_url.endswith("example_normal.jpg")
    assert p.published_at == "2026-11-10T07:00:30Z" and p.timestamp_verified
    m = p.metrics
    assert (m.views, m.likes, m.replies, m.reposts, m.quotes) == (500, 10, 1, 3, 0)
    assert [md.type for md in p.media] == ["photo", "video"]
    assert p.media[1].url.endswith("example2_high.mp4")  # highest-bitrate mp4
    assert p.raw_ref == "raw/x/page_1.json"
    rels = {(r.type, r.target_post_id, r.target_url) for r in p.relations}
    assert ("quote", "x:1899999999999999999",
            "https://x.com/source_acct/status/1899999999999999999") in rels
    assert ("link", None, "https://example.org/report") in rels
    # the permalink of the quoted tweet is not duplicated as a 'link'
    assert not any(r.type == "link" and "status/1899" in (r.target_url or "")
                   for r in p.relations)
    assert all(r.evidence == "direct" for r in p.relations)

    reply = main[1]
    assert reply.lang == "ar" and "رد" in reply.text  # non-English text kept as-is
    assert [(r.type, r.target_post_id, r.target_handle) for r in reply.relations] == [
        ("reply", "x:1900000000000000001", "@example_user"),
        ("mention", None, "@example_user"),
    ]
    # the quoted tweet is emitted as its own post so the quote edge has both endpoints
    assert [(e.post_id, e.query) for e in emb] == [("x:1899999999999999999", "embedded:quoted")]
    assert emb[0].timestamp_verified and emb[0].metrics.likes == 120


def test_retweet_and_missing_timestamp():
    main, emb = parse_twitterapi_io_page(load("doc_example_advanced_search_page2.json"),
                                         collected_at="2026-11-18T00:00:00Z", raw_ref=None,
                                         query="t")
    rt = main[0]
    assert ("repost", "x:1899999999999999999") in {(r.type, r.target_post_id)
                                                     for r in rt.relations}
    assert emb[0].query == "embedded:retweeted"
    no_time = main[1]
    assert no_time.published_at is None and not no_time.timestamp_verified
    assert "createdAt" in (no_time.timestamp_note or "")
    assert no_time.lang is None
    assert no_time.url == "https://x.com/no_time/status/1900000000000000004"
    assert no_time.metrics.views is None  # absent metric is None, not 0
    gate_timestamps(main, START, END)
    assert not no_time.timestamp_verified and rt.in_window


def test_parse_official_page():
    main, emb = parse_official_page(load("doc_example_official_recent_search.json"),
                                    collected_at="2026-11-18T00:00:00Z", raw_ref="r",
                                    query="t")
    p = main[0]
    assert p.post_id == "x:1900000000000000010" and p.author_handle == "@official_user"
    assert p.published_at == "2026-11-17T10:00:00Z" and p.timestamp_verified
    assert p.author_created_at == "2020-01-01T00:00:00Z"
    assert (p.metrics.views, p.metrics.quotes) == (100, 4)
    assert p.media[0].type == "photo"
    rels = {(r.type, r.target_post_id, r.target_url, r.target_handle) for r in p.relations}
    assert ("quote", "x:1900000000000000011",
            "https://x.com/quoted_user/status/1900000000000000011", "@quoted_user") in rels
    assert ("link", None, "https://news.example.com/a", None) in rels
    assert ("mention", None, None, "@other") in rels
    assert [(e.post_id, e.query) for e in emb] == [("x:1900000000000000011", "embedded:quoted")]


# ----------------------------------------------------------------- collect flows


@respx.mock
def test_pagination_two_pages(tmp_path):
    route = respx.get(TWITTERAPI_IO_URL).mock(side_effect=[
        httpx.Response(200, json=load("doc_example_advanced_search.json")),
        httpx.Response(200, json=load("doc_example_advanced_search_page2.json")),
    ])
    ctx = make_ctx(tmp_path)
    res = XCollector().collect(q(), ctx)
    st = res.status[0]
    assert st.state == SourceState.ok and st.count == 4 and st.requests == 2
    assert st.cost_usd == pytest.approx(4 * 0.15 / 1000)
    # Search matches are emitted as normal posts; embedded (unmatched) tweets come back
    # flagged context_only so verify.analysable()/in_scope() keep them out of analysis.
    ids = [p.post_id for p in res.posts if not p.context_only]
    assert ids == ["x:1900000000000000001", "x:1900000000000000002",
                   "x:1900000000000000003", "x:1900000000000000004"]
    ctx_posts = [p for p in res.posts if p.context_only]
    assert all(p.context_only for p in ctx_posts)
    assert not analysable(ctx_posts) and not in_scope(ctx_posts)
    # ...but the direct edge is still resolvable from the relation itself
    q_rel = next(r for r in res.posts[0].relations if r.type == "quote")
    assert q_rel.target_post_id == "x:1899999999999999999" and q_rel.target_url
    # auth header, query params and cursor passed correctly
    r1, r2 = route.calls[0].request, route.calls[1].request
    assert r1.headers["X-API-Key"] == "test-key-123456"
    assert r1.url.params["queryType"] == "Latest" and r1.url.params["cursor"] == ""
    assert r1.url.params["query"] == (f'("example topic") since_time:{int(DAY.timestamp())} '
                                      f"until_time:{int(END.timestamp()) + 1}")
    assert r2.url.params["cursor"] == "DOC_EXAMPLE_CURSOR_PAGE_2"
    # raw pages saved and referenced
    assert (tmp_path / "raw" / "x" / "page_1.json").is_file()
    assert (tmp_path / "raw" / "x" / "page_2.json").is_file()
    assert res.posts[0].raw_ref == "raw/x/page_1.json"


@respx.mock
def test_partial_on_deadline_keeps_cursor(tmp_path):
    ctx = make_ctx(tmp_path)

    def first_page(request):
        ctx.deadline = time.monotonic() - 1  # budget runs out after page 1
        return httpx.Response(200, json=load("doc_example_advanced_search.json"))

    respx.get(TWITTERAPI_IO_URL).mock(side_effect=first_page)
    res = XCollector().collect(q(), ctx)
    st = res.status[0]
    assert st.state == SourceState.partial
    assert st.cursor == "2026-11-18:DOC_EXAMPLE_CURSOR_PAGE_2"  # slice first date + cursor
    assert st.count == 2 and "time budget" in st.reason


@respx.mock
def test_failed_on_401(tmp_path):
    respx.get(TWITTERAPI_IO_URL).mock(
        return_value=httpx.Response(401, json={"error": 401, "message": "Unauthorized"}))
    res = XCollector().collect(q(), make_ctx(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.failed and res.posts == []
    assert "401" in st.reason and "credential" in st.reason
    assert "test-key-123456" not in st.reason


@respx.mock
def test_partial_when_later_page_fails(tmp_path):
    respx.get(TWITTERAPI_IO_URL).mock(side_effect=[
        httpx.Response(200, json=load("doc_example_advanced_search.json")),
        httpx.Response(400, json={"error": 400, "message": "bad"}),
    ])
    res = XCollector().collect(q(), make_ctx(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.partial and st.count == 2
    assert st.cursor == "2026-11-18:DOC_EXAMPLE_CURSOR_PAGE_2"


@respx.mock
def test_completed_zero_names_query(tmp_path):
    respx.get(TWITTERAPI_IO_URL).mock(return_value=httpx.Response(
        200, json={"tweets": [], "has_next_page": False, "next_cursor": ""}))
    res = XCollector().collect(q(), make_ctx(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.completed_zero and st.count == 0
    assert '("example topic") since_time:' in st.reason
    assert st.cost_usd == pytest.approx(0.15 / 1000)  # documented per-request minimum


@respx.mock
def test_limit_at_page_boundary_saves_next_cursor(tmp_path):
    route = respx.get(TWITTERAPI_IO_URL).mock(
        return_value=httpx.Response(200, json=load("doc_example_advanced_search.json")))
    res = XCollector().collect(q(limit=2), make_ctx(tmp_path))
    st = res.status[0]
    assert route.call_count == 1 and st.state == SourceState.ok and st.count == 2
    # Updated for slicing: ok carries no cursor; the slice note names the next one.
    assert st.cursor is None and "share of 2 post(s)" in st.reason
    assert "DOC_EXAMPLE_CURSOR_PAGE_2" in st.reason


@respx.mock
def test_limit_mid_page_saves_cursor_of_that_page(tmp_path):
    # limit=3 cuts page 2 (2 + 2 posts): the next-page cursor would skip the cut post,
    # so the cursor that FETCHED page 2 is saved.
    respx.get(TWITTERAPI_IO_URL).mock(side_effect=[
        httpx.Response(200, json=load("doc_example_advanced_search.json")),
        httpx.Response(200, json=load("doc_example_advanced_search_page2.json")),
    ])
    res = XCollector().collect(q(limit=3), make_ctx(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.ok and st.count == 3
    assert "1 post(s) on the last page were not kept" in st.reason
    assert "resume from its cursor 'DOC_EXAMPLE_CURSOR_PAGE_2'" in st.reason


@respx.mock
def test_limit_mid_first_page_says_restart(tmp_path):
    respx.get(TWITTERAPI_IO_URL).mock(
        return_value=httpx.Response(200, json=load("doc_example_advanced_search.json")))
    st = XCollector().collect(q(limit=1), make_ctx(tmp_path)).status[0]
    assert st.count == 1 and st.cursor is None and "first page" in st.reason


def test_context_posts_emitted_only_with_flag(monkeypatch):
    """When the foundation adds Post.context_only, embedded tweets come back flagged."""
    class FlaggedPost(Post):
        context_only: bool = False

    monkeypatch.setattr(xmod, "Post", FlaggedPost)
    main, emb = parse_twitterapi_io_page(load("doc_example_advanced_search.json"),
                                         collected_at="2026-11-18T00:00:00Z",
                                         raw_ref=None, query="t")
    extra = xmod._context_posts(main, emb + emb)  # duplicates collapse
    assert [(p.post_id, p.context_only) for p in extra] == [("x:1899999999999999999", True)]
    assert not any(getattr(p, "context_only", False) for p in main)
    # an embedded post that no kept post references is not emitted
    assert xmod._context_posts(main[1:], emb) == []


@respx.mock
def test_empty_pages_with_cursor_do_not_loop_forever(tmp_path):
    n = {"i": 0}

    def page(request):
        n["i"] += 1
        return httpx.Response(200, json={"tweets": [], "has_next_page": True,
                                         "next_cursor": f"c{n['i']}"})

    respx.get(TWITTERAPI_IO_URL).mock(side_effect=page)
    res = XCollector().collect(q(), make_ctx(tmp_path))
    assert n["i"] == xmod.MAX_EMPTY_PAGES
    st = res.status[0]
    # The provider still reported more pages: this must NOT look like "no posts".
    assert st.state == SourceState.partial
    assert st.cursor == f"2026-11-18:c{xmod.MAX_EMPTY_PAGES}"
    assert "incomplete" in st.reason and "0 matching" not in st.reason


def test_unknown_provider_fails(tmp_path):
    res = XCollector().collect(q(), make_ctx(tmp_path, X_PROVIDER="bogus"))
    assert res.status[0].state == SourceState.failed


def test_capabilities_follow_provider(monkeypatch):
    # Patch load_config so the machine's .env / config.json cannot affect the result.
    monkeypatch.setattr(xmod, "load_config", lambda: {})
    assert XCollector().capabilities().credential_keys == ("TWITTERAPI_IO_KEY",)
    monkeypatch.setattr(xmod, "load_config", lambda: {"X_PROVIDER": "official"})
    assert XCollector().capabilities().credential_keys == ("X_BEARER_TOKEN",)


# ---------------------------------------------------------------- official path


@respx.mock
def test_official_reports_out_of_coverage_for_old_part(tmp_path):
    now = datetime.now(UTC)
    start = now - timedelta(days=20)
    route = respx.get(OFFICIAL_URL).mock(return_value=httpx.Response(
        200, json=load("doc_example_official_recent_search.json")))
    ctx = make_ctx(tmp_path, X_PROVIDER="official", X_BEARER_TOKEN="bearer-abcdef")
    res = XCollector().collect(q(start=start, end=now), ctx)
    states = [s.state for s in res.status]
    assert states == [SourceState.ok, SourceState.out_of_coverage]
    assert "7 days" in res.status[1].reason
    req = route.calls[0].request
    assert req.headers["Authorization"] == "Bearer bearer-abcdef"
    sent_start = datetime.fromisoformat(req.url.params["start_time"].replace("Z", "+00:00"))
    assert sent_start > now - timedelta(days=7)
    assert "referenced_tweets.id" in req.url.params["expansions"]
    # the included (quoted) tweet did not match the search: emitted only as context_only
    assert {p.post_id for p in res.posts if not p.context_only} == {"x:1900000000000000010"}
    assert all(p.context_only for p in res.posts if p.post_id != "x:1900000000000000010")


@respx.mock
def test_official_start_time_reclamped_per_request(tmp_path, monkeypatch):
    """A slow collection (e.g. 429 backoff) must not send a start_time > 7 days old.

    Updated for time slicing: one search per day, newest first, so the oldest (clamped)
    slice is requested last, after the clock has moved on.
    """
    t0 = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)
    clock = {"now": t0}
    monkeypatch.setattr(xmod, "now_utc", lambda: clock["now"])
    seen_at: list[datetime] = []

    def responder(request):
        seen_at.append(clock["now"])
        clock["now"] = t0 + timedelta(minutes=20)  # long backoff after the first request
        return httpx.Response(200, json={"meta": {"result_count": 0}})

    route = respx.get(OFFICIAL_URL).mock(side_effect=responder)
    ctx = make_ctx(tmp_path, X_PROVIDER="official", X_BEARER_TOKEN="bearer-abcdef")
    res = XCollector().collect(q(start=t0 - timedelta(days=20), end=t0), ctx)
    assert res.status[0].state == SourceState.completed_zero
    assert route.call_count == 8  # 2026-11-11 (clamped) .. 2026-11-18

    def sent(i, key="start_time"):
        v = route.calls[i].request.url.params[key]
        return datetime.fromisoformat(v.replace("Z", "+00:00"))

    for i in range(route.call_count):
        assert sent(i) > seen_at[i] - timedelta(days=7)  # within 7 days at request time
    assert sent(0) == datetime(2026, 11, 18, tzinfo=UTC)
    assert sent(0, "end_time") == t0 - xmod.OFFICIAL_END_MARGIN
    assert sent(1, "end_time") == datetime(2026, 11, 18, tzinfo=UTC)  # exclusive day end
    oldest = sent(route.call_count - 1)
    assert oldest >= t0 + timedelta(minutes=20) - timedelta(days=7)


@respx.mock
def test_official_window_entirely_too_old(tmp_path):
    route = respx.get(OFFICIAL_URL)
    ctx = make_ctx(tmp_path, X_PROVIDER="official", X_BEARER_TOKEN="bearer-abcdef")
    old = datetime(2025, 1, 1, tzinfo=UTC)
    res = XCollector().collect(q(start=old, end=old + timedelta(days=3)), ctx)
    assert [s.state for s in res.status] == [SourceState.out_of_coverage]
    assert route.call_count == 0


# ------------------------------------------------- official: wrong / mis-pasted tokens
WEB_APP_TOKEN = xmod.X_WEB_APP_TOKEN_PREFIX + "zzWEBAPPTOKENzz%3Dfake"  # public prefix only


@respx.mock
def test_official_refuses_xcom_website_token_without_request(tmp_path):
    route = respx.get(OFFICIAL_URL).mock(return_value=httpx.Response(401))
    now = datetime.now(UTC)
    for pasted in (WEB_APP_TOKEN, f"Bearer {WEB_APP_TOKEN}"):
        ctx = make_ctx(tmp_path, X_PROVIDER="official", X_BEARER_TOKEN=pasted)
        res = XCollector().collect(q(start=now - timedelta(days=2), end=now), ctx)
        assert [s.state for s in res.status] == [SourceState.failed]
        assert "WEBSITE token" in res.status[0].reason and "developer.x.com" in res.status[0].reason
        assert WEB_APP_TOKEN not in res.status[0].reason  # never echo the token
    assert not route.called


@respx.mock
def test_official_strips_pasted_bearer_prefix_and_quotes(tmp_path):
    route = respx.get(OFFICIAL_URL).mock(return_value=httpx.Response(
        200, json=load("doc_example_official_recent_search.json")))
    now = datetime.now(UTC)
    ctx = make_ctx(tmp_path, X_PROVIDER="official", X_BEARER_TOKEN=' "Bearer real-token%3Dx" ')
    XCollector().collect(q(start=now - timedelta(days=2), end=now), ctx)
    assert route.calls[0].request.headers["Authorization"] == "Bearer real-token%3Dx"


def test_official_bearer_problem_cases():
    assert xmod.official_bearer_problem("real-token") is None
    assert xmod.official_bearer_problem("bearer real-token") is None
    assert xmod.official_bearer_problem("abc def") is not None
    assert "WEBSITE" in xmod.official_bearer_problem(WEB_APP_TOKEN)


def test_doctor_flags_website_token_as_will_fail():
    from socmint.doctor import source_readiness

    rows = {r["name"]: r for r in source_readiness(
        {"X_PROVIDER": "official", "X_BEARER_TOKEN": f"Bearer {WEB_APP_TOKEN}"})}
    assert rows["x"]["ready"] is False and rows["x"]["outcome"] == "fail"
    assert rows["x"]["reason"].startswith("will fail:")
    rows = {r["name"]: r for r in source_readiness(
        {"X_PROVIDER": "official", "X_BEARER_TOKEN": "real-token"})}
    assert rows["x"]["ready"] is True


# ------------------------------------------------- live TwitterAPI.io response (2026-11-19)
LIVE_TAPI = "live_twitterapi_io_advanced_search_2026-11-19.json"


def test_live_twitterapi_io_page_parses_every_tweet():
    # A real advanced_search page ("bank of america", queryType=Latest), saved unedited.
    data = load(LIVE_TAPI)
    main, emb = parse_twitterapi_io_page(data, collected_at="2026-11-19T12:00:00Z",
                                         raw_ref="raw/x/page_12.json", query="bank of america")
    assert data["has_next_page"] is True and isinstance(data["next_cursor"], str)
    assert [p.post_id for p in main] == [f"x:{t['id']}" for t in data["tweets"]]
    assert all(p.timestamp_verified and p.published_at for p in main)
    assert all(p.author_handle and p.author_created_at for p in main)
    assert all(p.metrics.views is not None and p.metrics.likes is not None for p in main)
    by_id = {t["id"]: t for t in data["tweets"]}
    for p in main:
        t = by_id[p.post_id.split(":", 1)[1]]
        types = {r.type for r in p.relations}
        assert ("reply" in types) == bool(t.get("isReply") and t.get("inReplyToId"))
        assert ("quote" in types) == bool(t.get("quoted_tweet"))


def test_live_twitterapi_io_omits_leading_reply_mentions():
    # Seen live: the @handles that open a reply are in `text` (before displayTextRange[0])
    # but NOT in entities.user_mentions, so no mention relation is produced for them.
    t = next(t for t in load(LIVE_TAPI)["tweets"] if t["id"] == "2123238855851532544")
    assert t["text"].startswith("@Lennytweets702 @realericmoutsos")
    assert t["entities"]["user_mentions"] == [] and t["displayTextRange"][0] == 33


# ------------------------------------------------------------ time slices (2026-11-19)
# Seen live: newest-first search filled a 300 cap from the last 13 h of a 7-day window.
# Each UTC day is now its own paginated search, newest first, sharing the limit.

D16 = datetime(2026, 11, 16, tzinfo=UTC)


def day_of(request) -> int:
    """UTC day of the since_time bound in a TwitterAPI.io request."""
    since = int(request.url.params["query"].split("since_time:")[1].split()[0])
    return datetime.fromtimestamp(since, UTC).day


def shifted(name: str, suffix: str) -> dict:
    """A fixture page with tweet ids changed, so another slice returns new posts."""
    return json.loads(json.dumps(load(name)).replace("190000000000000000",
                                                     f"1900000000000{suffix}000"))


EMPTY_TIO = {"tweets": [], "has_next_page": False, "next_cursor": ""}


@respx.mock
def test_tio_one_search_per_day_newest_first_with_phrase(tmp_path):
    route = respx.get(TWITTERAPI_IO_URL).mock(return_value=httpx.Response(200, json=EMPTY_TIO))
    res = XCollector().collect(q(start=D16), make_ctx(tmp_path))
    sent = [c.request.url.params["query"] for c in route.calls]
    days = [datetime(2026, 11, d, tzinfo=UTC) for d in (18, 17, 16)]
    assert sent == [f'("example topic") since_time:{int(d.timestamp())} '
                    f"until_time:{int(d.timestamp()) + 86400}" for d in days]
    st = res.status[0]
    assert st.state == SourceState.completed_zero and st.requests == 3
    assert st.cost_usd == pytest.approx(3 * 0.15 / 1000)  # minimum charge per request
    assert ("per-day: 2026-11-18 0 (fewer than its share of 167), 2026-11-17 0 (fewer "
            "than its share of 250), 2026-11-16 0 (fewer than its share of 500)") in st.reason


@respx.mock
def test_tio_words_mode_sends_bare_words(tmp_path):
    route = respx.get(TWITTERAPI_IO_URL).mock(return_value=httpx.Response(200, json=EMPTY_TIO))
    XCollector().collect(q(match="words"), make_ctx(tmp_path))
    assert route.calls[0].request.url.params["query"].startswith("(example topic) since_time:")


@respx.mock
def test_tio_cap_shared_with_rollover_after_quiet_day(tmp_path):
    # limit 6 over 3 days: day 18 takes its share of 2 in one page; day 17 is quiet;
    # day 16 may take the remaining 4 (two pages).
    def responder(request):
        d, cur = day_of(request), request.url.params["cursor"]
        if d == 18:
            return httpx.Response(200, json=load("doc_example_advanced_search.json"))
        if d == 17:
            return httpx.Response(200, json=EMPTY_TIO)
        name = ("doc_example_advanced_search_page2.json" if cur else
                "doc_example_advanced_search.json")
        return httpx.Response(200, json=shifted(name, "16"))

    route = respx.get(TWITTERAPI_IO_URL).mock(side_effect=responder)
    res = XCollector().collect(q(limit=6, start=D16), make_ctx(tmp_path))
    st = res.status[0]
    matched = [p for p in res.posts if not p.context_only]
    assert st.state == SourceState.ok and st.count == len(matched) == 6, st.reason
    assert [day_of(c.request) for c in route.calls] == [18, 17, 16, 16]
    assert "per-day: 2026-11-18 2, 2026-11-17 0 (fewer than its share of 2), " \
           "2026-11-16 4" in st.reason
    assert st.cost_usd == pytest.approx((2 + 1 + 2 + 2) * 0.15 / 1000)  # summed over slices
    assert st.cursor is None


@respx.mock
def test_tio_time_budget_mid_way_names_unreached_days(tmp_path):
    ctx = make_ctx(tmp_path)

    def responder(request):
        ctx.deadline = time.monotonic() - 1  # budget runs out after day 18's first page
        return httpx.Response(200, json=load("doc_example_advanced_search.json"))

    route = respx.get(TWITTERAPI_IO_URL).mock(side_effect=responder)
    st = XCollector().collect(q(start=D16), ctx).status[0]
    assert route.call_count == 1
    assert st.state == SourceState.partial and st.count == 2 and "time budget" in st.reason
    assert "2026-11-17 not reached, 2026-11-16 not reached" in st.reason
    assert st.cursor == "2026-11-18:DOC_EXAMPLE_CURSOR_PAGE_2"


@respx.mock
def test_tio_http_failure_on_later_day_is_partial_and_stops(tmp_path):
    def responder(request):
        if day_of(request) == 18:
            return httpx.Response(200, json=load("doc_example_advanced_search_page2.json"))
        return httpx.Response(401, json={"error": 401, "message": "Unauthorized"})

    route = respx.get(TWITTERAPI_IO_URL).mock(side_effect=responder)
    st = XCollector().collect(q(start=D16), make_ctx(tmp_path)).status[0]
    assert [day_of(c.request) for c in route.calls] == [18, 17]  # day 16 never requested
    assert st.state == SourceState.partial and st.count == 2  # not failed: posts exist
    assert "credential" in st.reason and "2026-11-16 not reached" in st.reason
    assert st.cursor == "2026-11-17:"


@respx.mock
def test_official_sends_phrase_per_day(tmp_path, monkeypatch):
    now = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(xmod, "now_utc", lambda: now)
    route = respx.get(OFFICIAL_URL).mock(return_value=httpx.Response(
        200, json={"meta": {"result_count": 0}}))
    ctx = make_ctx(tmp_path, X_PROVIDER="official", X_BEARER_TOKEN="bearer-abcdef")
    start = (now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    res = XCollector().collect(q(start=start, end=now), ctx)
    assert route.call_count == 3
    assert {c.request.url.params["query"] for c in route.calls} == {'"example topic"'}
    starts = [c.request.url.params["start_time"][:10] for c in route.calls]
    assert starts == sorted(starts, reverse=True)  # newest day first
    assert res.status[0].state == SourceState.completed_zero


@respx.mock
def test_tio_budget_after_cut_day_does_not_reuse_that_days_cursor(tmp_path):
    # Review regression: day 18 is cut (empty pages with more reported), then the budget
    # runs out before day 17. The resume cursor must not pair day 17 with day 18's cursor.
    ctx = make_ctx(tmp_path)
    n = {"i": 0}

    def responder(request):
        n["i"] += 1
        if n["i"] == xmod.MAX_EMPTY_PAGES:
            ctx.deadline = time.monotonic() - 1
        return httpx.Response(200, json={"tweets": [], "has_next_page": True,
                                         "next_cursor": f"c{n['i']}"})

    respx.get(TWITTERAPI_IO_URL).mock(side_effect=responder)
    st = XCollector().collect(q(start=D16), ctx).status[0]
    assert st.state == SourceState.partial
    assert st.cursor == "2026-11-17:"


@respx.mock
def test_tio_limit_zero_is_not_reported_as_searched_zero(tmp_path):
    # Review regression: with no allowance nothing is requested, so the status must not
    # claim the search completed with 0 matching posts.
    route = respx.get(TWITTERAPI_IO_URL).mock(return_value=httpx.Response(200, json=EMPTY_TIO))
    st = XCollector().collect(q(limit=0, start=D16), make_ctx(tmp_path)).status[0]
    assert route.call_count == 0
    assert st.state != SourceState.completed_zero
    assert "0 matching" not in st.reason and "not searched" in st.reason


@respx.mock
def test_tio_limit_smaller_than_days_never_exceeds_and_skips_requests(tmp_path):
    route = respx.get(TWITTERAPI_IO_URL).mock(side_effect=lambda r: httpx.Response(
        200, json=shifted("doc_example_advanced_search.json", f"{day_of(r):02d}")))
    st = XCollector().collect(q(limit=2, start=datetime(2026, 11, 12, tzinfo=UTC)),
                              make_ctx(tmp_path)).status[0]
    assert st.count == 2 and route.call_count == 2
    assert st.state == SourceState.ok and "not searched" in st.reason
