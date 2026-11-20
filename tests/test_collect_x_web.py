"""Tests for the X web_session collector. No browser and no network: a fake page /
context replaces Playwright via x_web._open_browser.

Fixtures are doc_example_* (hand-built from the documented / observed GraphQL shape,
synthetic values); no live SearchTimeline response has been captured yet.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import pytest

from socmint.collect import x_web
from socmint.collect.x import XCollector
from socmint.interfaces import CollectContext, Query
from socmint.schema import SourceState
from socmint.verify import analysable

FX = Path(__file__).parent / "fixtures" / "x"
START = datetime(2026, 11, 1, tzinfo=UTC)
END = datetime(2026, 11, 18, 23, 59, 59, tzinfo=UTC)
# Control-flow tests use a one-day window = one time slice, so a fake page's steps map to
# a single X search; multi-slice behaviour has its own tests below.
DAY = datetime(2026, 11, 18, tzinfo=UTC)
TS = {d: int(datetime(2026, 11, d, tzinfo=UTC).timestamp()) for d in range(1, 21)}
RAW = f'("example topic") since_time:{TS[18]} until_time:{TS[19]}'  # the DAY window
TOKEN = "authtokensecretvalue0123456789"
CT0 = "ct0secretvalue0123456789abcdef"
SEARCH_PATH = "https://x.com/i/api/graphql/AbCdEf/SearchTimeline"
B1 = "DAADDAABCgABEXAMPLEBOTTOM1"
B2 = "DAADDAABCgABEXAMPLEBOTTOM2"
TOP1 = "DAADDAABCgABEXAMPLETOP"


def search_url(cursor: str | None = None, raw: str = RAW) -> str:
    """SearchTimeline GET URL as the x.com web app builds it (variables JSON in the
    query string; doc example, UNVERIFIED against a live capture)."""
    variables = {"rawQuery": raw, "count": 20, "querySource": "typed_query",
                 "product": "Latest"}
    if cursor:
        variables["cursor"] = cursor
    return f"{SEARCH_PATH}?{urlencode({'variables': json.dumps(variables)})}"


def load(name: str) -> dict:
    return json.loads((FX / name).read_text(encoding="utf-8"))


P1 = "doc_example_search_timeline.json"
P2 = "doc_example_search_timeline_page2.json"


def q(limit: int = 500, start: datetime = DAY, end: datetime = END, **opts) -> Query:
    return Query(topic="example topic", start=start, end=end, limit=limit, options=opts)


def make_ctx(tmp_path: Path, **cfg: str) -> CollectContext:
    conf = {"X_PROVIDER": "web_session", "X_AUTH_TOKEN": TOKEN, "X_CT0": CT0, **cfg}
    return CollectContext(run_dir=tmp_path, raw_dir=tmp_path / "raw", config=conf,
                          deadline=time.monotonic() + 60)


def parse(name: str):
    return x_web.parse_search_timeline(load(name), collected_at="2026-11-18T00:00:00Z",
                                       raw_ref="raw/x/search_page_1.json", query_str="s")


# ------------------------------------------------------------------ parser tests


def test_parse_matched_ids_skip_tombstone_and_include_module():
    main, emb, cur = parse(P1)
    ids = [p.post_id for p in main]
    assert ids == ["x:1900000000000001001", "x:1900000000000001002", "x:1900000000000001003",
                   "x:1900000000000001004", "x:1900000000000001005", "x:1900000000000001007"]
    assert "x:1900000000000001006" not in ids  # TweetTombstone skipped
    assert cur == "DAADDAABCgABEXAMPLEBOTTOM1"
    assert x_web.count_tweet_entries(load(P1)) == 7  # tombstone is still a search hit


def test_parse_skips_promoted_tweet():
    main, emb, _ = parse(P1)
    ids = {p.post_id for p in main + emb}
    assert "x:1900000000000009001" not in ids  # promoted-tweet-* with promotedMetadata
    assert all(p.author_handle != "@example_advertiser" for p in main + emb)
    assert x_web.count_promoted(load(P1)) == 1
    assert x_web.count_tweet_entries(load(P1)) == 7  # ad not counted as a search hit


def test_parse_skips_promoted_by_metadata_or_module_entry_id():
    data = load(P1)
    entries = data["data"]["search_by_raw_query"]["search_timeline"]["timeline"][
        "instructions"][0]["entries"]
    organic = entries[0]
    organic["content"]["itemContent"]["promotedMetadata"] = {"impressionId": "x"}
    module = next(e for e in entries if e["entryId"].startswith("conversationthread"))
    module["entryId"] = "promoted-conversationthread-1"
    main, _, _ = x_web.parse_search_timeline(data, collected_at="t", raw_ref=None,
                                             query_str="s")
    ids = {p.post_id for p in main}
    assert "x:1900000000000001001" not in ids and "x:1900000000000001007" not in ids
    assert x_web.count_promoted(data) == 3


def test_parse_new_user_shape_photo_metrics_relations():
    p = parse(P1)[0][0]
    assert p.url == "https://x.com/example_newshape/status/1900000000000001001"
    assert p.author_handle == "@example_newshape" and p.author_id == "2001"
    assert p.author_name == "Example New Shape"
    assert p.author_created_at == "2015-01-05T08:00:00Z"
    assert p.author_avatar_url.endswith("/2001/new_normal.jpg")
    assert p.published_at == "2026-11-02T10:00:00Z" and p.timestamp_verified
    assert p.collected_at == "2026-11-18T00:00:00Z"
    m = p.metrics
    assert (m.views, m.likes, m.replies, m.reposts, m.quotes) == (1234, 12, 1, 3, 2)
    assert [(x.type, x.url) for x in p.media] == [
        ("photo", "https://pbs.twimg.com/media/example_photo.jpg")]
    rels = {(r.type, r.target_url or r.target_handle) for r in p.relations}
    assert ("link", "https://news.example.org/story") in rels
    assert ("mention", "@example_mention") in rels
    assert all(r.evidence == "direct" for r in p.relations)
    assert p.lang == "en" and p.query == "s" and p.raw_ref == "raw/x/search_page_1.json"
    assert not p.context_only


def test_parse_visibility_wrapper_legacy_user_video_reply():
    p = parse(P1)[0][1]
    assert p.post_id == "x:1900000000000001002"
    assert p.author_handle == "@example_legacy"
    assert p.author_created_at == "2007-12-19T20:20:32Z"
    assert p.author_avatar_url.endswith("/2002/legacy_normal.jpg")
    assert p.media[0].type == "video"
    assert p.media[0].url.endswith("/1280x720/high.mp4")  # highest-bitrate mp4
    assert p.media[0].thumbnail.endswith("thumb.jpg")
    reply = [r for r in p.relations if r.type == "reply"]
    assert reply[0].target_post_id == "x:1900000000000000900"
    assert reply[0].target_handle == "@someone_else"
    assert p.metrics.views == 88


def test_parse_quote_and_retweet_are_direct_and_embedded_context_only():
    main, emb, _ = parse(P1)
    quote = main[2]
    qrel = [r for r in quote.relations if r.type == "quote"][0]
    assert qrel.target_post_id == "x:1900000000000000800"
    assert qrel.target_url == "https://x.com/example_quoted/status/1900000000000000800"
    assert qrel.target_handle == "@example_quoted"
    # the permalink to the quoted tweet is not duplicated as a link relation
    assert not [r for r in quote.relations if r.type == "link"]
    rt = main[3]
    rrel = [r for r in rt.relations if r.type == "repost"][0]
    assert rrel.target_post_id == "x:1900000000000000700"
    assert rrel.target_handle == "@example_original"
    by_id = {p.post_id: p for p in emb}
    assert set(by_id) == {"x:1900000000000000800", "x:1900000000000000700"}
    assert all(p.context_only for p in emb)
    assert by_id["x:1900000000000000800"].query == "embedded:quoted"
    assert by_id["x:1900000000000000700"].query == "embedded:retweeted"
    assert by_id["x:1900000000000000700"].published_at == "2026-10-29T07:00:00Z"
    assert analysable(emb) == []


def test_parse_note_tweet_preferred_over_truncated_full_text():
    p = parse(P1)[0][4]
    assert p.lang == "ar"
    assert not p.text.endswith("https://t.co/n") and len(p.text) > 300
    assert ("link", "https://long.example.net/article") in {
        (r.type, r.target_url) for r in p.relations}


def test_parse_page2_replace_entry_cursor_and_dedupe():
    main, emb, cur = parse(P2)
    assert cur == "DAADDAABCgABEXAMPLEBOTTOM2"  # from TimelineReplaceEntry
    assert [p.post_id for p in main] == ["x:1900000000000001008", "x:1900000000000001001"]
    assert emb == []


def test_parse_missing_created_at_is_unverified():
    data = load(P2)
    entry = data["data"]["search_by_raw_query"]["search_timeline"]["timeline"][
        "instructions"][0]["entries"][0]
    del entry["content"]["itemContent"]["tweet_results"]["result"]["legacy"]["created_at"]
    main, _, _ = x_web.parse_search_timeline(data, collected_at="2026-11-18T00:00:00Z",
                                             raw_ref=None, query_str="s")
    assert main[0].published_at is None and not main[0].timestamp_verified
    assert "missing" in main[0].timestamp_note


def test_parse_empty_and_garbage():
    assert x_web.parse_search_timeline({}, collected_at="t", raw_ref=None,
                                       query_str="s") == ([], [], None)
    assert x_web.count_tweet_entries({"data": None}) == 0


def test_build_search_uses_unix_bounds_until_exclusive():
    # Seen live: since:/until: dates are not UTC days; since_time/until_time are exact.
    # Plain multi-word topic -> exact phrase by default (window.search_topic).
    assert x_web.build_search(q(start=START)) == \
        f'("example topic") since_time:{TS[1]} until_time:{TS[19]}'


def test_build_search_words_mode_and_expressions_unchanged():
    assert x_web.build_search(q(match="words")) == \
        f"(example topic) since_time:{TS[18]} until_time:{TS[19]}"
    expr = Query(topic="vaccine OR vaccination", start=DAY, end=END, limit=1)
    assert x_web.build_search(expr) == \
        f"(vaccine OR vaccination) since_time:{TS[18]} until_time:{TS[19]}"


# ------------------------------------------------------------------ fake browser


class FakeResponse:
    def __init__(self, body=None, status=200, headers=None, url=None, cursor=None, raw=RAW):
        self.url = url or search_url(cursor, raw)
        self.status, self.headers, self._body = status, headers or {}, body

    def json(self):
        return self._body


class FakeContext:
    def __init__(self):
        self.cookies_added = []
        self._cookies = []

    def add_cookies(self, cookies):
        self.cookies_added.extend(cookies)

    def cookies(self, urls=None):
        return list(self._cookies)

    def storage_state(self):
        return {"cookies": self._cookies, "origins": []}


class FakePage:
    """steps[0] is delivered on goto, steps[i] on the i-th scroll. A step is a list of
    responses, a str (new page URL) or a callable(page)."""

    def __init__(self, steps, per_goto=None):
        # per_goto: one steps list per page.goto (one per time slice); else `steps` for all.
        self.steps = list(steps)
        self.per_goto = per_goto
        self.gotos: list[str] = []
        self.url = "about:blank"
        self.handlers = []
        self.scrolls = 0
        self.closed = False

    def on(self, event, handler):
        assert event == "response"
        self.handlers.append(handler)

    def _deliver(self, i):
        if i >= len(self.steps):
            return
        step = self.steps[i]
        if isinstance(step, str):
            self.url = step
        elif callable(step):
            step(self)
        else:
            for r in step:
                for h in self.handlers:
                    h(r)

    def goto(self, url, wait_until=None):
        self.url = url
        self.gotos.append(url)
        if self.per_goto is not None:
            i = len(self.gotos) - 1
            self.steps = list(self.per_goto[i]) if i < len(self.per_goto) else []
            self.scrolls = 0
        self._deliver(0)

    def evaluate(self, js):
        self.scrolls += 1
        self._deliver(self.scrolls)

    def wait_for_timeout(self, ms):
        pass

    def is_closed(self):
        return self.closed


@pytest.fixture
def fake_browser(monkeypatch):
    monkeypatch.setattr(x_web, "FIRST_RESPONSE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(x_web, "RESPONSE_TIMEOUT_S", 0.05)
    calls: dict = {"closed": 0}

    def install(steps, raise_on_open: Exception | None = None, per_goto=None):
        page = FakePage(steps, per_goto)
        ctx = FakeContext()

        def opener(**kw):
            calls["kwargs"] = kw
            if raise_on_open:
                raise raise_on_open

            def close():
                calls["closed"] += 1
            return x_web.BrowserHandle("chrome", ctx, page, close)
        monkeypatch.setattr(x_web, "_open_browser", opener)
        calls["page"] = page
        return calls
    return install


def resp(name, **kw):
    return FakeResponse(load(name), **kw)


def empty_page():
    return {"data": {"search_by_raw_query": {"search_timeline": {"timeline": {
        "instructions": [{"type": "TimelineAddEntries", "entries": [
            {"entryId": "cursor-bottom-0", "content": {
                "entryType": "TimelineTimelineCursor", "cursorType": "Bottom",
                "value": "C0"}}]}]}}}}}


# --------------------------------------------------------------- collect control flow


def test_unauthorized_is_skipped_no_credentials(tmp_path, fake_browser):
    calls = fake_browser([])
    res = x_web.collect_web(q(), make_ctx(tmp_path, X_AUTH_TOKEN="", X_CT0=""))
    st = res.status[0]
    assert st.state == SourceState.skipped_no_credentials
    assert st.reason == x_web.NO_SESSION
    assert "kwargs" not in calls  # no browser launched


def test_dispatch_from_xcollector(tmp_path, fake_browser):
    fake_browser([])
    res = XCollector().collect(q(), make_ctx(tmp_path, X_AUTH_TOKEN=""))
    assert res.status[0].state == SourceState.skipped_no_credentials


def test_capabilities_web_session(monkeypatch):
    from socmint.collect import x as xmod
    monkeypatch.setattr(xmod, "load_config", lambda: {"X_PROVIDER": "web_session"})
    caps = XCollector().capabilities()
    assert caps.needs_credentials is False and caps.credential_keys == ()


def test_pages_ok_with_raw_saved_and_context_posts(tmp_path, fake_browser):
    p3 = empty_page()
    calls = fake_browser([[resp(P1)], [resp(P2, cursor=B1)], [FakeResponse(p3, cursor=B2)]])
    ctx = make_ctx(tmp_path)
    progress = []
    ctx.progress = lambda s, info: progress.append(info)
    res = x_web.collect_web(q(), ctx)
    st = res.status[0]
    assert st.state == SourceState.ok, st.reason
    assert st.requests == 3 and st.count == 7 and st.cost_usd == 0.0 and st.date_filtered
    assert st.query == RAW
    assert st.reason.startswith("web_session (owner's logged-in account)")
    assert "1 promoted (ad) tweet(s) skipped" in st.reason
    assert "x:1900000000000009001" not in {p.post_id for p in res.posts}
    matched = [p for p in res.posts if not p.context_only]
    assert len(matched) == len({p.post_id for p in matched}) == 7
    ctxonly = {p.post_id for p in res.posts if p.context_only}
    assert ctxonly == {"x:1900000000000000800", "x:1900000000000000700"}
    assert (tmp_path / "raw" / "x" / "search_page_1.json").is_file()
    assert (tmp_path / "raw" / "x" / "search_page_3.json").is_file()
    assert matched[0].raw_ref == "raw/x/search_page_1.json"
    assert progress[-1]["state"] == "collecting" and progress[-1]["count"] == 7
    assert calls["closed"] == 1


def test_cookies_from_config_path(tmp_path, fake_browser):
    calls = fake_browser([[FakeResponse(empty_page())]])
    x_web.collect_web(q(), make_ctx(tmp_path))
    kw = calls["kwargs"]
    assert kw["storage_state"] is None and kw["headless"] is True
    cookies = {c["name"]: c for c in kw["cookies"]}
    assert cookies["auth_token"]["value"] == TOKEN and cookies["auth_token"]["httpOnly"]
    assert cookies["ct0"]["value"] == CT0 and not cookies["ct0"]["httpOnly"]
    for c in cookies.values():
        assert c["domain"] == ".x.com" and c["path"] == "/" and c["secure"]


def test_storage_state_preferred_when_saved(tmp_path, fake_browser):
    _save_session(expires=time.time() + 3600)
    calls = fake_browser([[FakeResponse(empty_page())]])
    x_web.collect_web(q(), make_ctx(tmp_path, X_AUTH_TOKEN="", X_CT0="",
                                    X_BROWSER_HEADLESS="false"))
    assert calls["kwargs"]["storage_state"] == str(x_web.session_file())
    assert calls["kwargs"]["cookies"] is None
    assert calls["kwargs"]["headless"] is False


def test_first_page_empty_is_completed_zero(tmp_path, fake_browser):
    fake_browser([[FakeResponse(empty_page())]])
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.completed_zero and st.count == 0
    assert "0 matching posts" in st.reason


def test_429_is_partial_with_reset(tmp_path, fake_browser):
    fake_browser([[resp(P1)], [FakeResponse(None, status=429,
                                            headers={"x-rate-limit-reset": "1795000000"})]])
    res = x_web.collect_web(q(), make_ctx(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.partial
    assert "429" in st.reason and "resets at" in st.reason
    assert st.cursor == f"2026-11-18:{B1}"  # slice first date + where it stopped
    assert st.count == 6


def test_429_on_first_page_is_partial_not_zero(tmp_path, fake_browser):
    fake_browser([[FakeResponse(None, status=429)]])
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.partial and "429" in st.reason


def test_login_redirect_is_failed(tmp_path, fake_browser):
    fake_browser(["https://x.com/i/flow/login?redirect_after_login=%2Fsearch"])
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.failed and "session expired" in st.reason


def test_401_is_failed(tmp_path, fake_browser):
    fake_browser([[FakeResponse(None, status=401)]])
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.failed and "log in again" in st.reason


def test_account_access_page_is_failed(tmp_path, fake_browser):
    fake_browser(["https://x.com/account/access"])
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.failed and "account" in st.reason


def test_deadline_is_partial_with_cursor(tmp_path, fake_browser):
    ctx = make_ctx(tmp_path)

    def expire(page):
        ctx.deadline = time.monotonic() - 1
        for h in page.handlers:
            h(resp(P1))
    fake_browser([expire])
    st = x_web.collect_web(q(), ctx).status[0]
    assert st.state == SourceState.partial and "time budget" in st.reason
    assert st.cursor == f"2026-11-18:{B1}" and st.count == 6


def test_limit_stops_ok(tmp_path, fake_browser):
    calls = fake_browser([[resp(P1)], [resp(P2, cursor=B1)]])
    res = x_web.collect_web(q(limit=3), make_ctx(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.ok and "share of 3 post(s)" in st.reason
    assert len([p for p in res.posts if not p.context_only]) == 3
    assert calls["page"].scrolls == 0
    # 3 posts of page 1 were cut: resume must re-fetch the slice's first page.
    assert st.cursor is None  # ok: no resume cursor (sliced runs keep one only when partial)
    assert "3 post(s) on the last page were not kept" in st.reason
    assert "slice's first page" in st.reason


def test_limit_mid_page_keeps_cursor_that_fetched_the_page(tmp_path, fake_browser):
    # Regression: the NEXT Bottom cursor was saved although posts of this page were cut.
    p2 = json.loads(json.dumps(load(P2)).replace("1900000000000001001", "1900000000000001009"))
    fake_browser([[resp(P1)], [FakeResponse(p2, cursor=B1)]])
    res = x_web.collect_web(q(limit=7), make_ctx(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.ok and st.count == 7  # 6 + 2 new, 1 dropped
    # Updated for slicing: an ok status carries no cursor; the note says to resume from
    # the cursor that fetched page 2 (not B2).
    assert st.cursor is None
    assert "1 post(s) on the last page were not kept" in st.reason
    assert "slice's saved cursor" in st.reason


def test_limit_exactly_at_page_end_saves_next_cursor(tmp_path, fake_browser):
    fake_browser([[resp(P1)], [resp(P2, cursor=B1)]])
    st = x_web.collect_web(q(limit=6), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.ok and "not kept" not in st.reason
    assert "more results were available" in st.reason


def test_max_scrolls_is_partial(tmp_path, fake_browser):
    fake_browser([[resp(P1)], [resp(P2, cursor=B1)]])
    st = x_web.collect_web(q(), make_ctx(tmp_path, X_MAX_SCROLLS="1")).status[0]
    assert st.state == SourceState.partial and "X_MAX_SCROLLS=1" in st.reason
    assert st.cursor == f"2026-11-18:{B2}"


def test_no_response_after_scrolls_is_partial(tmp_path, fake_browser):
    fake_browser([[resp(P1)]])
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.partial and "no SearchTimeline response" in st.reason


def test_no_first_response_is_failed(tmp_path, fake_browser):
    fake_browser([])
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.failed and st.count == 0


def test_no_new_tweets_twice_completes(tmp_path, fake_browser):
    p2 = load(P1)
    cur = p2["data"]["search_by_raw_query"]["search_timeline"]["timeline"]["instructions"][0][
        "entries"][-1]["content"]
    p3 = json.loads(json.dumps(p2))
    cur["value"] = "OTHER1"
    p3["data"]["search_by_raw_query"]["search_timeline"]["timeline"]["instructions"][0][
        "entries"][-1]["content"]["value"] = "OTHER2"
    fake_browser([[resp(P1)], [FakeResponse(p2, cursor=B1)],
                  [FakeResponse(p3, cursor="OTHER1")]])
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.ok and "no new tweets" in st.reason


def test_unchanged_cursor_completes(tmp_path, fake_browser):
    fake_browser([[resp(P1)], [resp(P1, cursor=B1)]])
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.ok and "unchanged" in st.reason


def top_poll_page():
    """What a Top-cursor 'new posts' poll can return: no tweets, only a Top cursor."""
    return {"data": {"search_by_raw_query": {"search_timeline": {"timeline": {
        "instructions": [{"type": "TimelineAddEntries", "entries": [
            {"entryId": "cursor-top-9", "content": {
                "entryType": "TimelineTimelineCursor", "cursorType": "Top",
                "value": "TOPNEW"}}]}]}}}}}


def test_top_poll_mid_run_is_ignored_not_exhausted(tmp_path, fake_browser):
    # Regression: an empty Top-cursor poll used to end the run as ok "search exhausted".
    fake_browser([[resp(P1)],
                  [FakeResponse(top_poll_page(), cursor=TOP1), resp(P2, cursor=B1)],
                  [FakeResponse(empty_page(), cursor=B2)]])
    res = x_web.collect_web(q(), make_ctx(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.ok and st.count == 7, st.reason
    assert "page 3 had no tweet entries" in st.reason
    assert "1 ignored" in st.reason and "4 SearchTimeline response(s), 3 used" in st.reason
    assert (tmp_path / "raw" / "x" / "search_ignored_1.json").is_file()
    assert (tmp_path / "raw" / "x" / "search_page_3.json").is_file()


def test_top_poll_alone_after_scroll_is_not_completion(tmp_path, fake_browser):
    # Only a Top poll answers the scroll; the Bottom page never comes -> partial.
    fake_browser([[resp(P1)], [FakeResponse(top_poll_page(), cursor=TOP1)],
                  [FakeResponse(load(P1))]])  # first-page refetch (no cursor): ignored too
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.partial and "exhausted" not in st.reason
    assert st.cursor == f"2026-11-18:{B1}" and st.count == 6
    assert "2 ignored" in st.reason


def test_top_poll_before_first_page_is_ignored(tmp_path, fake_browser):
    fake_browser([[FakeResponse(top_poll_page(), cursor=TOP1), resp(P1)],
                  [FakeResponse(empty_page(), cursor=B1)]])
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.ok and st.count == 6


def test_unreadable_request_variables_cannot_prove_exhaustion(tmp_path, fake_browser):
    fake_browser([[resp(P1, url=SEARCH_PATH)], [FakeResponse(empty_page(), url=SEARCH_PATH)]])
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.partial and "could not be matched" in st.reason
    assert st.count == 6


def test_unreadable_variables_on_empty_first_page_is_not_zero(tmp_path, fake_browser):
    fake_browser([[FakeResponse(empty_page(), url=SEARCH_PATH)]])
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.partial


def test_request_cursor_reads_variables():
    assert x_web.request_cursor(FakeResponse(cursor=B1)) == B1
    assert x_web.request_cursor(FakeResponse()) is None
    assert x_web.request_cursor(FakeResponse(url=SEARCH_PATH)) is x_web._UNKNOWN
    assert x_web.request_cursor(FakeResponse(url=SEARCH_PATH + "?variables=%7Bbad")) \
        is x_web._UNKNOWN


def test_promoted_only_first_page_is_completed_zero(tmp_path, fake_browser):
    ad = load(P1)["data"]["search_by_raw_query"]["search_timeline"]["timeline"][
        "instructions"][0]["entries"][1]
    assert ad["entryId"].startswith("promoted")
    page = empty_page()
    page["data"]["search_by_raw_query"]["search_timeline"]["timeline"]["instructions"][0][
        "entries"].insert(0, ad)
    fake_browser([[FakeResponse(page)]])
    res = x_web.collect_web(q(), make_ctx(tmp_path))
    st = res.status[0]
    assert res.posts == [] and st.state == SourceState.completed_zero
    assert "1 promoted (ad) tweet(s) skipped" in st.reason


def test_no_browser_is_failed(tmp_path, fake_browser):
    fake_browser([], raise_on_open=x_web.NoBrowserError("chrome: missing"))
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.failed
    assert "or install Google Chrome; 0 SearchTimeline response(s)" in st.reason
    assert "socmint doctor --prefetch" in st.reason


def test_error_redacts_auth_token(tmp_path, fake_browser):
    fake_browser([], raise_on_open=RuntimeError(f"boom cookie auth_token={TOKEN} ct0 {CT0}"))
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.failed
    assert TOKEN not in st.reason and CT0 not in st.reason and "boom" in st.reason


def test_page_exception_after_posts_is_partial(tmp_path, fake_browser):
    def explode(page):
        raise RuntimeError(f"Target closed {TOKEN}")
    fake_browser([[resp(P1)], explode])
    res = x_web.collect_web(q(), make_ctx(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.partial and st.count == 6 and TOKEN not in st.reason


# ------------------------------------------------------------------ session status


def _save_session(expires: float, name: str = "auth_token", user: str | None = "@owner"):
    state = {"cookies": [{"name": name, "value": "savedtokenvalue123456", "domain": ".x.com",
                          "path": "/", "expires": expires, "httpOnly": True, "secure": True,
                          "sameSite": "None"}], "origins": []}
    x_web._write_private(x_web.session_file(), json.dumps(state))
    x_web._write_private(x_web._meta_file(), json.dumps({"user": user}))


def test_status_nothing_configured():
    st = x_web.session_status({})
    assert st["configured"] is False and st["authorized"] is False
    assert st["source"] is None and st["reason"] == x_web.NO_SESSION
    assert st["session_path"].endswith("x_storage_state.json")
    assert st["login_state"] in ("idle", "waiting", "done", "failed")


def test_status_saved_browser_session(tmp_path):
    _save_session(expires=time.time() + 3600)
    assert str(x_web.SESSION_FILE).startswith(str(tmp_path))  # follows $SOCMINT_HOME
    assert oct(x_web.session_file().stat().st_mode & 0o777) == "0o600"
    st = x_web.session_status({})
    assert st["authorized"] and st["source"] == "browser_login" and st["user"] == "@owner"


def test_status_session_cookie_without_expiry_is_valid():
    _save_session(expires=-1)
    assert x_web.session_status({})["authorized"]


def test_status_expired_session_falls_back_to_pasted():
    _save_session(expires=time.time() - 10)
    st = x_web.session_status({})
    assert st["configured"] and not st["authorized"] and "expired" in st["reason"]
    st = x_web.session_status({"X_AUTH_TOKEN": TOKEN, "X_CT0": CT0})
    assert st["authorized"] and st["source"] == "pasted_cookies"


def test_status_needs_both_pasted_cookies():
    assert not x_web.session_status({"X_AUTH_TOKEN": TOKEN})["authorized"]


def test_logout_removes_file_and_mentions_pasted():
    _save_session(expires=time.time() + 3600)
    out = x_web.logout({"X_AUTH_TOKEN": TOKEN, "X_CT0": CT0})
    assert out["ok"] and not x_web.session_file().exists()
    assert "remain in config" in out["message"]
    assert x_web.logout({})["ok"]


def test_saved_cookie_values_redacted(tmp_path, fake_browser):
    _save_session(expires=time.time() + 3600)
    fake_browser([], raise_on_open=RuntimeError("failed with savedtokenvalue123456"))
    st = x_web.collect_web(q(), make_ctx(tmp_path, X_AUTH_TOKEN="", X_CT0="")).status[0]
    assert "savedtokenvalue123456" not in st.reason


# --------------------------------------------------------------------------- login


def test_start_login_refuses_second_concurrent_login(monkeypatch):
    release = threading.Event()
    started = threading.Event()

    def worker(cfg, timeout_s):
        started.set()
        release.wait(5)
        x_web._set_login("done")
    monkeypatch.setattr(x_web, "_login_worker", worker)
    monkeypatch.setitem(x_web._login, "state", "idle")
    monkeypatch.setitem(x_web._login, "thread", None)
    first = x_web.start_login({})
    assert first == {"ok": True, "error": None}
    assert started.wait(5)
    assert x_web.login_state()["state"] == "waiting"
    second = x_web.start_login({})
    assert second["ok"] is False and "already in progress" in second["error"]
    release.set()
    x_web._login["thread"].join(5)
    assert x_web.login_state()["state"] == "done"
    x_web._set_login("idle")


def test_login_worker_saves_storage_state(monkeypatch):
    """Worker logic with a fake headful browser: cookie appears on the 2nd poll."""
    ctx = FakeContext()
    page = FakePage([])
    polls = {"n": 0}

    def cookies(urls=None):
        polls["n"] += 1
        if polls["n"] >= 2:
            return [{"name": "auth_token", "value": "loggedintoken123456", "domain": ".x.com",
                     "expires": -1}, {"name": "twid", "value": "u%3D424242",
                                      "domain": ".x.com", "expires": -1}]
        return []
    ctx.cookies = cookies
    ctx.storage_state = lambda: {"cookies": cookies(), "origins": []}
    page.evaluate = lambda js: "owner_handle"
    closed = []
    monkeypatch.setattr(x_web, "_open_browser", lambda **kw: (
        closed.append(kw["headless"]) or x_web.BrowserHandle(
            "chrome", ctx, page, lambda: closed.append("closed"))))
    x_web._login_worker({}, 30)
    assert x_web.login_state()["state"] == "done"
    assert closed == [False, "closed"]  # headful, then closed
    meta = json.loads(x_web._meta_file().read_text())
    assert meta["user"] == "@owner_handle" and meta["user_id"] == "424242"
    assert x_web.session_status({})["authorized"]
    x_web._set_login("idle")


def test_login_worker_window_closed(monkeypatch):
    ctx = FakeContext()
    page = FakePage([])
    page.closed = True
    monkeypatch.setattr(x_web, "_open_browser",
                        lambda **kw: x_web.BrowserHandle("chrome", ctx, page, lambda: None))
    x_web._login_worker({}, 30)
    st = x_web.login_state()
    assert st["state"] == "failed" and "closed" in st["error"]
    assert not x_web.session_file().exists()
    x_web._set_login("idle")


def test_collect_refuses_while_login_window_open(tmp_path, fake_browser, monkeypatch):
    calls = fake_browser([[resp(P1)]])
    monkeypatch.setitem(x_web._login, "state", "waiting")
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.failed and "X login in progress" in st.reason
    assert "kwargs" not in calls  # no second browser on the account
    assert x_web._login["collecting"] is False


def test_collect_refuses_parallel_collection_and_login_waits(tmp_path, fake_browser,
                                                           monkeypatch):
    seen = {}

    def during(page):
        seen["second"] = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
        seen["login"] = x_web.start_login({})
        for h in page.handlers:
            h(FakeResponse(empty_page()))
    monkeypatch.setitem(x_web._login, "state", "idle")
    monkeypatch.setitem(x_web._login, "thread", None)
    monkeypatch.setattr(x_web, "_login_worker", lambda *a: pytest.fail("login started"))
    fake_browser([during])
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.completed_zero
    assert seen["second"].state == SourceState.failed
    assert "already running" in seen["second"].reason
    assert seen["login"]["ok"] is False and "collection is running" in seen["login"]["error"]
    assert x_web._login["collecting"] is False  # released after the run


def test_headless_noted_in_reason(tmp_path, fake_browser):
    fake_browser([[FakeResponse(empty_page())]])
    st = x_web.collect_web(q(), make_ctx(tmp_path)).status[0]
    assert "HeadlessChrome" in st.reason
    fake_browser([[FakeResponse(empty_page())]])
    st = x_web.collect_web(q(), make_ctx(tmp_path, X_BROWSER_HEADLESS="false")).status[0]
    assert "HeadlessChrome" not in st.reason


def test_ensure_browser_no_install_when_found(monkeypatch):
    monkeypatch.setattr(x_web, "_detect_browser", lambda: "chrome")
    monkeypatch.setattr(x_web.subprocess, "run", lambda *a, **k: pytest.fail("installed"))
    out = x_web.ensure_browser()
    assert out["ok"] is True and out["browser"] == "chrome" and "Chrome" in out["message"]


def test_ensure_browser_reports_when_missing_and_no_install(monkeypatch):
    monkeypatch.setattr(x_web, "_detect_browser", lambda: None)
    out = x_web.ensure_browser(install=False)
    assert out["ok"] is False and out["browser"] is None


# ------------------------------------------------------------ time slices (2026-11-19)
# Seen live: newest-first search filled a 300 cap from the last 13 h of a 7-day window.
# Each UTC day is now its own X search, newest first, sharing the per-source limit.

D16 = datetime(2026, 11, 16, tzinfo=UTC)


def raw_for(day: int, topic: str = '"example topic"') -> str:
    return f"({topic}) since_time:{TS[day]} until_time:{TS[day + 1]}"


def shifted(name: str, suffix: str) -> dict:
    """A fixture page with every tweet id changed, so another slice returns new posts."""
    return json.loads(json.dumps(load(name)).replace("19000000000000010",
                                                     f"19000000000000{suffix}"))


def sent_queries(page) -> list[str]:
    from urllib.parse import parse_qs, urlparse
    return [parse_qs(urlparse(u).query)["q"][0] for u in page.gotos]


def test_slices_search_each_day_newest_first_with_phrase(tmp_path, fake_browser):
    calls = fake_browser([], per_goto=[
        [[FakeResponse(empty_page(), raw=raw_for(18))]],
        [[FakeResponse(empty_page(), raw=raw_for(17))]],
        [[FakeResponse(empty_page(), raw=raw_for(16))]]])
    res = x_web.collect_web(q(start=D16), make_ctx(tmp_path))
    assert sent_queries(calls["page"]) == [raw_for(18), raw_for(17), raw_for(16)]
    st = res.status[0]
    assert st.query == f'("example topic") since_time:{TS[16]} until_time:{TS[19]}'
    assert st.state == SourceState.completed_zero, st.reason
    assert "per-day: 2026-11-18 0" in st.reason
    assert calls["closed"] == 1  # one browser session for every slice


def test_slices_words_mode_sends_bare_words(tmp_path, fake_browser):
    calls = fake_browser([[FakeResponse(empty_page(), raw=raw_for(18, "example topic"))]])
    st = x_web.collect_web(q(match="words"), make_ctx(tmp_path)).status[0]
    assert sent_queries(calls["page"]) == [raw_for(18, "example topic")]
    assert st.state == SourceState.completed_zero, st.reason


def test_slices_share_cap_with_rollover_after_quiet_day(tmp_path, fake_browser):
    # limit 9 over 3 days: day 18 gets ceil(9/3)=3 (page has 6 -> cut at 3), day 17 is
    # quiet (0 of its 3), so day 16 may take the remaining 6.
    fake_browser([], per_goto=[
        [[FakeResponse(load(P1), raw=raw_for(18))]],
        [[FakeResponse(empty_page(), raw=raw_for(17))]],
        [[FakeResponse(shifted(P1, "16"), raw=raw_for(16))]]])
    res = x_web.collect_web(q(limit=9, start=D16), make_ctx(tmp_path))
    st = res.status[0]
    matched = [p for p in res.posts if not p.context_only]
    assert st.state == SourceState.ok and st.count == len(matched) == 9, st.reason
    assert ("per-day: 2026-11-18 3, 2026-11-17 0 (fewer than its share of 3), "
            "2026-11-16 6") in st.reason
    assert st.cursor is None
    # each post records the slice search that found it
    assert {p.query for p in matched} == {raw_for(18), raw_for(16)}


def test_slices_never_exceed_limit_and_zero_share_sends_nothing(tmp_path, fake_browser):
    calls = fake_browser([], per_goto=[[[FakeResponse(load(P1), raw=raw_for(18))]]])
    res = x_web.collect_web(q(limit=1, start=D16), make_ctx(tmp_path))
    st = res.status[0]
    assert len(calls["page"].gotos) == 1  # days 17 and 16 have a share of 0: no search
    assert st.state == SourceState.ok and st.count == 1
    assert "2 older slice(s) not searched" in st.reason
    assert "2026-11-17 not reached, 2026-11-16 not reached" in st.reason


def test_slices_time_budget_names_unreached_days(tmp_path, fake_browser):
    ctx = make_ctx(tmp_path)

    def last_page_then_expire(page):
        ctx.deadline = time.monotonic() - 1
        for h in page.handlers:
            h(FakeResponse(empty_page(), cursor=B1, raw=raw_for(18)))
    calls = fake_browser([], per_goto=[
        [[FakeResponse(load(P1), raw=raw_for(18))], last_page_then_expire],
        [[FakeResponse(empty_page(), raw=raw_for(17))]]])
    st = x_web.collect_web(q(start=D16), ctx).status[0]
    assert len(calls["page"].gotos) == 1
    assert st.state == SourceState.partial and st.count == 6
    assert "time budget exhausted before slice 2026-11-17" in st.reason
    assert "2026-11-18 6 (fewer than its share of 167), 2026-11-17 not reached, " \
           "2026-11-16 not reached" in st.reason
    assert st.cursor == "2026-11-17:"  # resume at the first day not searched


def test_slices_late_response_of_previous_slice_is_ignored(tmp_path, fake_browser):
    # Day 17's goto first sees a straggler first-page response of day 18's search (no
    # cursor, like day 17's own first page): it must not be taken as day 17's page.
    fake_browser([], per_goto=[
        [[FakeResponse(empty_page(), raw=raw_for(18))]],
        [[FakeResponse(load(P1), raw=raw_for(18)),
          FakeResponse(empty_page(), raw=raw_for(17))]]])
    st = x_web.collect_web(q(start=datetime(2026, 11, 17, tzinfo=UTC)),
                           make_ctx(tmp_path)).status[0]
    assert st.state == SourceState.completed_zero and st.count == 0, st.reason
    assert "1 ignored" in st.reason


def test_slices_session_expiry_after_posts_is_partial_and_stops(tmp_path, fake_browser):
    calls = fake_browser([], per_goto=[
        [[FakeResponse(load(P1), raw=raw_for(18))],
         [FakeResponse(empty_page(), cursor=B1, raw=raw_for(18))]],
        [[FakeResponse(None, status=401, raw=raw_for(17))]],
        [[FakeResponse(empty_page(), raw=raw_for(16))]]])
    st = x_web.collect_web(q(start=D16), make_ctx(tmp_path)).status[0]
    assert len(calls["page"].gotos) == 2  # the run stops; day 16 is never opened
    assert st.state == SourceState.partial and st.count == 6  # not failed: posts exist
    assert "log in again" in st.reason and "2026-11-16 not reached" in st.reason
    assert st.cursor == "2026-11-17:"


def test_slices_soft_cut_moves_on_to_next_day(tmp_path, fake_browser):
    # Day 18 stops at X_MAX_SCROLLS (more results exist); day 17 is still searched.
    fake_browser([], per_goto=[
        [[FakeResponse(load(P1), raw=raw_for(18))]],
        [[FakeResponse(shifted(P1, "17"), raw=raw_for(17))],
         [FakeResponse(empty_page(), cursor=B1, raw=raw_for(17))]]])
    st = x_web.collect_web(q(start=datetime(2026, 11, 17, tzinfo=UTC)),
                           make_ctx(tmp_path, X_MAX_SCROLLS="1")).status[0]
    assert st.state == SourceState.partial and st.count == 12, st.reason
    assert "2026-11-18: stopped at X_MAX_SCROLLS=1" in st.reason
    assert st.cursor == f"2026-11-18:{B1}"


def test_slices_limit_zero_is_not_reported_as_searched_zero(tmp_path, fake_browser):
    # Review regression: no slice has an allowance, nothing is searched, so the status
    # must not claim "search completed with 0 matching posts".
    calls = fake_browser([], per_goto=[])
    st = x_web.collect_web(q(limit=0, start=D16), make_ctx(tmp_path)).status[0]
    assert calls["page"].gotos == []
    assert st.state != SourceState.completed_zero
    assert "0 matching" not in st.reason and "not searched" in st.reason


def test_slices_day_with_only_already_kept_posts_is_not_unparseable(tmp_path, fake_browser):
    # Review regression: a day whose tweets were all already collected for a newer day
    # (overlapping bounds) parsed fine; it must not be reported as unparseable entries
    # nor turn the run partial.
    fake_browser([], per_goto=[
        [[FakeResponse(load(P1), raw=raw_for(18))],
         [FakeResponse(empty_page(), cursor=B1, raw=raw_for(18))]],
        [[FakeResponse(load(P1), raw=raw_for(17))],
         [FakeResponse(empty_page(), cursor=B1, raw=raw_for(17))]],
        [[FakeResponse(empty_page(), raw=raw_for(16))]]])
    st = x_web.collect_web(q(start=D16), make_ctx(tmp_path)).status[0]
    assert "could be parsed" not in st.reason
    assert st.state == SourceState.ok and st.count == 6


class GoneResponse(FakeResponse):
    """A SearchTimeline response whose body the browser released (seen live 2026-11-20)."""

    def json(self):
        raise RuntimeError("Response.json: Protocol error (Network.getResponseBody): No "
                           "resource with given identifier found Response body is not "
                           "available for a response that was navigated away from.")


def test_released_response_body_is_skipped_not_fatal(tmp_path, fake_browser):
    # A late response whose body is gone must not fail the source: it is discarded and
    # the real page that follows is used.
    fake_browser([[GoneResponse(), resp(P1)], [FakeResponse(empty_page(), cursor=B1)]])
    res = x_web.collect_web(q(), make_ctx(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.ok, st.reason
    assert "1 discarded: the browser had already released the response body" in st.reason
    assert any(not p.context_only for p in res.posts)


def test_other_json_errors_still_fail_the_slice(tmp_path, fake_browser):
    class Broken(FakeResponse):
        def json(self):
            raise ValueError("Unexpected token < in JSON")
    fake_browser([[Broken()]])
    res = x_web.collect_web(q(), make_ctx(tmp_path))
    assert res.status[0].state in (SourceState.failed, SourceState.partial)
    assert "Unexpected token" in res.status[0].reason
