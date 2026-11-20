"""Tests for the LinkedIn posts collector (source "linkedin"). No browser, no network:
x_web._open_browser is replaced by a fake page.

Parser tests run on tests/fixtures/linkedin_posts/live_* (captured live 2026-11-19,
trimmed and sanitised, see the README there). Control-flow tests use small synthetic
flight streams built below in the same shape.
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from socmint import doctor
from socmint.collect import BUILTIN, DEFAULT_SOURCES, get_collector, x_web
from socmint.collect import linkedin_posts as lp
from socmint.interfaces import CollectContext, Query
from socmint.schema import SourceState
from socmint.server.app import create_app

FX = Path(__file__).parent / "fixtures" / "linkedin_posts"
COLLECTED = datetime(2026, 11, 19, 12, 52, 25, tzinfo=UTC)  # when the fixtures were captured
LI_AT = "AQEDAliatSECRETvalue0123456789abcdefXYZ"
JSID = "ajax:1234567890123456789"
COOKIE = f'bcookie="v=2&abc"; li_at={LI_AT}; JSESSIONID="{JSID}"; lang=v=2&lang=xx-test'
SEARCH = "https://www.linkedin.com/search/results/content/"
PAGINATION = ("https://www.linkedin.com/flagship-web/rsc-action/actions/pagination"
              "?sduiid=com.linkedin.sdui.search.contentSearchResults")


def fixture(name: str) -> str:
    return (FX / name).read_text(encoding="utf-8")


# ======================================================================= parser (live)


def test_parse_document_first_results_from_rehydration():
    items, payload, promoted, marker = lp.parse_document(fixture("live_search_page.html"))
    assert [i["native_id"] for i in items] == [
        "7529163328582156288", "7529162820656082944", "7529162861017776128"]
    assert marker and promoted == 0
    assert payload["startIndex"] == 3 and payload["keywords"] == '"bank of america"'
    assert payload["datePosted"] == ["past-24h"]
    quinn = items[0]
    assert quinn["author_handle"] == "in/quinnmi" and quinn["author_name"] == "Michael Quinn"
    assert quinn["author_url"] == "https://www.linkedin.com/in/quinnmi/"
    assert quinn["time_label"] == "5m"
    assert quinn["share_url"].startswith("https://www.linkedin.com/posts/quinnmi_")
    assert "Bank of America" in quinn["text"]


def test_parse_pagination_stream_incl_group_post():
    items, payload, _, marker = lp.parse_stream(fixture("live_search_pagination.rsc"))
    assert [i["native_id"] for i in items] == [
        "7529162869180129280", "7529162428027248641", "7529162074019622913"]
    assert marker and payload["startIndex"] == 6
    group = items[2]
    # A group post: the actor is the group; its own id is a groupPost URN, tied to the
    # activity that the card's "10m" label matches.
    assert group["author_handle"] == "groups/16179129"
    assert group["author_name"] == "Banca, Seguros & Fintech"
    assert group["content_urn"] == "urn:li:groupPost:16179129-7529113090341347328"
    assert group["activity_id"] == "7529162074019622913"
    assert group["text"].startswith("¿Quién responde cuando un agente de IA")  # original


def _posts(name: str, parse) -> list:
    items = parse(fixture(name))[0]
    return [lp.item_to_post(i, collected_at=COLLECTED, raw_ref="raw/linkedin/p.json",
                            query_str='"bank of america"') for i in items]


def test_posts_ids_urls_timestamps_verified_against_labels():
    posts = _posts("live_search_page.html", lp.parse_document) + \
        _posts("live_search_pagination.rsc", lp.parse_stream)
    assert len(posts) == 6
    for p in posts:
        nid = p.post_id.split(":", 1)[1]
        assert p.url == f"https://www.linkedin.com/feed/update/urn:li:activity:{nid}/"
        assert p.timestamp_verified, p.timestamp_note
        assert "decoded from the post ID" in p.timestamp_note
        assert p.collected_at == "2026-11-19T12:52:25Z"
    assert posts[0].published_at == "2026-11-19T12:47:02Z"  # label 5m
    assert posts[3].published_at == "2026-11-19T12:45:13Z"  # label 7m


def test_scheduled_post_uses_activity_not_older_ugcpost_id():
    kristin = _posts("live_search_page.html", lp.parse_document)[2]
    items = lp.parse_document(fixture("live_search_page.html"))[0]
    # The ugcPost id decodes to 03:52Z, nine hours before the "7m" card; the activity id
    # matches the label, so it is the post id and the publication time.
    assert items[2]["content_urn"] == "urn:li:ugcPost:7529028803810095105"
    assert lp.decode_id_time("7529028803810095105").hour == 3
    assert kristin.post_id == "linkedin:7529162861017776128"
    assert kristin.published_at == "2026-11-19T12:45:11Z" and kristin.timestamp_verified


def test_metrics_media_and_relations():
    doc = _posts("live_search_page.html", lp.parse_document)
    pag = _posts("live_search_pagination.rsc", lp.parse_stream)
    quinn, angel, kristin = doc
    lou = pag[0]
    assert quinn.metrics.reactions == 2 and quinn.metrics.comments == 0
    assert lou.metrics.reactions == 1 and lou.metrics.reposts == 0
    assert quinn.media and quinn.media[0].type == "photo"
    assert quinn.media[0].url.startswith("https://media.licdn.com/dms/image/")
    assert kristin.media[0].type == "video" and ".licdn.com/" in kristin.media[0].url
    assert kristin.media[0].thumbnail and "videocover" in kristin.media[0].thumbnail
    # External link behind LinkedIn's safety redirect is decoded; hashtags are not links.
    assert [(r.type, r.target_url) for r in angel.relations] == [
        ("link", "https://lnkd.in/gJSUcCKD")]
    mentions = [r.target_handle for r in kristin.relations if r.type == "mention"]
    assert mentions[:2] == ["company/fortune", "company/great-place-to-work-us"]
    assert all(r.evidence == "direct" for p in doc + pag for r in p.relations)
    assert not any(r.type in ("repost", "quote") for p in doc + pag for r in p.relations)
    assert quinn.author_avatar_url.startswith("https://media.licdn.com/")


def test_path_references_keep_the_space_before_a_mention():
    # Seen live: RSC deduplicates the " " before an inline mention into a path reference
    # ("$19f:props:children:...:0:0"); without resolving it the text read "produced
    # byGreat Place To Work" and the phrase check missed "withBank of America".
    kristin = lp.parse_document(fixture("live_search_page.html"))[0][2]
    assert "produced by Great Place To Work, understand" in kristin["text"]
    rows = {"1": ["$", "div", None, {"children": [["$", "span", None, {"children": " "}]]}],
            "0": ["$", "p", None, {"children": ["with", "$1:props:children:0:props:children",
                                                "Bank of America", "$9:props:x"]}]}
    assert lp._Tree(rows).text(rows["0"]) == "with Bank of America"


def test_mailto_links_are_not_web_links():
    nav = {"$type": "proto.sdui.actions.core.NavigateToUrl",
           "urlValue": {"url": "mailto:someone@example.test"}}
    web = {"$type": "proto.sdui.actions.core.NavigateToUrl",
           "urlValue": {"url": "https://example.test/a"}}
    node = {"viewTrackingSpecs": {"viewName": "feed-commentary"},
            "children": [{"triggers": [nav]}, {"triggers": [web]}]}
    rels = lp._relations(lp._Tree({}), {"children": [node]}, set())
    assert [(r.type, r.target_url) for r in rels] == [("link", "https://example.test/a")]


def test_fixtures_hold_no_account_data():
    for name in ("live_search_page.html", "live_search_pagination.rsc"):
        s = fixture(name).lower()
        # Structural checks only: the collecting account's real identifiers must never be
        # written into the repo, not even as a deny-list in a test.
        for bad in ("memberfirstname", "memberlastname", "memberheadline", "li_at",
                    "jsessionid", "nav-me-", "navigation-messaging", "commentbox"):
            assert bad not in s, (name, bad)
        vals = re.findall(r'viewerprofileid\\?":\s*\\?"([^"\\]*)', s)
        assert vals and set(vals) == {"[viewer-redacted]"}, (name, set(vals))


# ================================================================ flight + timestamps


def test_parse_flight_rows_text_rows_and_imports():
    text_row = "line one\nline two ü"
    raw = (f'0:["$","div",null,{{"children":"$1"}}]\n'
           f'1:T{len(text_row.encode()):x},{text_row}'
           '2:I["mod",[],"default"]\n'
           'zz garbage line\n'
           '3:{"a":1}\n')
    rows = lp.parse_flight(raw)
    assert rows["1"] == text_row and rows["2"] is None and rows["3"] == {"a": 1}
    assert lp.rehydration_stream("<html>no stream</html>") is None
    assert lp.parse_document("<html></html>") == ([], None, 0, False)


def test_decode_id_time_and_label_windows():
    assert lp.decode_id_time("7529162869180129280") == datetime(
        2026, 11, 19, 12, 45, 13, 94000, tzinfo=UTC)
    assert lp.decode_id_time("16179129-7529113090341347328").hour == 9
    assert lp.decode_id_time("abc") is None and lp.decode_id_time("5") is None
    lo, hi = lp.label_window("7m")
    assert lo <= 7.2 * 60 <= hi
    assert lp.label_window("2h")[0] <= 2.5 * 3600 <= lp.label_window("2h")[1]
    assert lp.label_window("3d")[1] < 5 * 86400
    assert lp.label_window("now")[1] < 600
    assert lp.label_window("Edited") is None


def test_check_timestamp_inconsistent_or_missing_label_is_unverified():
    pub = datetime(2026, 11, 19, 12, 45, tzinfo=UTC)
    ok, note = lp.check_timestamp(pub, "7m", pub + timedelta(minutes=7, seconds=20))
    assert ok and "consistent" in note
    ok, note = lp.check_timestamp(pub, "7m", pub + timedelta(hours=5))
    assert not ok and "INCONSISTENT" in note and "decoded from the post ID" in note
    ok, note = lp.check_timestamp(pub, None, pub)
    assert not ok and "not checked" in note
    ok, note = lp.check_timestamp(None, "7m", pub)
    assert not ok


# ========================================================================== cookies


def test_cookie_header_parsing_and_flags():
    pairs = lp.parse_cookie_header("Cookie: " + COOKIE)
    assert [n for n, _ in pairs] == ["bcookie", "li_at", "JSESSIONID", "lang"]
    cookies = lp.build_cookie_dicts(COOKIE)
    flags = {c["name"]: c["httpOnly"] for c in cookies}
    # Seen live: JSESSIONID marked httpOnly breaks LinkedIn's CSRF header (403).
    assert flags == {"bcookie": False, "li_at": True, "JSESSIONID": False, "lang": False}
    assert all(c["domain"] == ".linkedin.com" and c["secure"] and c["path"] == "/"
               for c in cookies)
    assert next(c for c in cookies if c["name"] == "JSESSIONID")["value"] == f'"{JSID}"'
    assert lp.has_li_at(COOKIE)


def test_bare_li_at_accepted_and_header_without_li_at_rejected():
    assert lp.build_cookie_dicts(LI_AT) == [{
        "name": "li_at", "value": LI_AT, "domain": ".linkedin.com", "path": "/",
        "secure": True, "httpOnly": True}]
    assert lp.has_li_at(LI_AT)
    assert not lp.has_li_at("JSESSIONID=abc; lang=en")
    assert not lp.has_li_at("")
    secrets = lp.cookie_secrets(COOKIE)
    assert LI_AT in secrets and JSID in secrets


# ==================================================================== search URL


def q(topic="bank of america", start=datetime(2026, 11, 18, tzinfo=UTC),
      end=datetime(2026, 11, 19, 23, 59, 59, tzinfo=UTC), limit=30, **opts) -> Query:
    return Query(topic=topic, start=start, end=end, limit=limit, options=opts)


def test_choose_date_preset_smallest_covering_window_start():
    now = datetime(2026, 11, 19, 13, 0, tzinfo=UTC)
    assert lp.choose_date_preset(now - timedelta(hours=10), now) == "past-24h"
    assert lp.choose_date_preset(now - timedelta(days=2), now) == "past-week"
    assert lp.choose_date_preset(now - timedelta(days=20), now) == "past-month"
    assert lp.choose_date_preset(now - timedelta(days=29), now) is None


def test_build_search_url_quotes_phrase_and_sorts_by_date():
    now = datetime(2026, 11, 19, 13, 0, tzinfo=UTC)
    url, kw, preset = lp.build_search_url(q(), now)
    assert kw == '"bank of america"' and preset == "past-week"
    assert url == (SEARCH + "?keywords=%22bank%20of%20america%22&origin=FACETED_SEARCH"
                   "&sortBy=%5B%22date_posted%22%5D&datePosted=%5B%22past-week%22%5D")
    url, kw, _ = lp.build_search_url(q(match="words"), now)
    assert kw == "bank of america"
    url, _, preset = lp.build_search_url(q(start=now - timedelta(days=29)), now)
    assert preset is None and "datePosted" not in url


# ============================================================= synthetic streams

NOW = datetime(2026, 11, 19, 13, 0, tzinfo=UTC)


def aid_at(dt: datetime, low: int = 1) -> str:
    return str((int(dt.timestamp() * 1000) << 22) | low)


def label_for(dt: datetime) -> str:
    age = (NOW - dt).total_seconds()
    if age < 3600:
        return f"{int(age // 60)}m"
    if age < 86400:
        return f"{int(age // 3600)}h"
    return f"{int(age // 86400)}d"


def item(dt: datetime, text: str, vanity: str = "alice", low: int = 1) -> dict:
    aid = aid_at(dt, low)
    nav = {"$type": "proto.sdui.actions.core.NavigateToScreen",
           "screenId": "com.linkedin.sdui.flagshipnav.profile.Profile", "url": f"/in/{vanity}/"}
    return {
        "viewTrackingSpecs": {"viewName": "feed-full-update"},
        "children": [
            ["$", "div", None, {"viewTrackingSpecs": {"viewName": "feed-actor"},
                                "triggers": [{"action": {
                                    "actions": [{"value": {"content": {"screen": nav}}}],
                                    "actionLabel": f"View {vanity.title()}’s profile"}}],
                                "children": [vanity.title()]}],
            ["$", "div", None, {"viewTrackingSpecs": {
                "viewName": "feed-actor-sub-description"}, "children": [f"{label_for(dt)} • "]}],
            ["$", "div", None, {"viewTrackingSpecs": {"viewName": "feed-commentary"},
                                "children": [text]}],
            {"key": {"id": f"reactionState-urn:li:activity:{aid}"}},
        ],
    }


def stream(items: list[dict], next_start: int | None = 3, kw: str = '"bank of america"',
           marker: bool = True) -> str:
    rows = ['0:' + json.dumps(["$", "div", None, {
        "observabilityIdentifier": lp.RESULTS_MARKER if marker else "other",
        "children": items}], ensure_ascii=False)]
    if next_start is not None:
        req = {"$type": "proto.sdui.actions.requests.PaginationRequest",
               "pagerId": lp.PAGER_ID, "requestedArguments": {"payload": {
                   "startIndex": next_start, "keywords": kw, "datePosted": ["past-week"]}}}
        rows.append('1:' + json.dumps([json.dumps(req, separators=(",", ":"))]))
    return "\n".join(rows) + "\n"


def document(items: list[dict], **kw) -> str:
    s = stream(items, **kw)
    return ("<html><body><script>window.__como_rehydration__ = "
            + json.dumps([s[: len(s) // 2], s[len(s) // 2:]]) + ";</script></body></html>")


class FakeRequest:
    def __init__(self, resource_type: str):
        self.resource_type = resource_type


class FakeResponse:
    def __init__(self, url: str, body: str = "", status: int = 200, rtype: str = "fetch"):
        self.url, self.status, self._body = url, status, body
        self.request = FakeRequest(rtype)

    def text(self) -> str:
        return self._body


def doc_resp(items: list[dict], **kw) -> FakeResponse:
    return FakeResponse(SEARCH + "?keywords=x", document(items, **kw), rtype="document")


def page_resp(items: list[dict], **kw) -> FakeResponse:
    return FakeResponse(PAGINATION + "&x=1", stream(items, **kw))


class FakePage:
    """steps[0] is delivered on goto, steps[i] on the i-th scroll (evaluate). A step is a
    list of responses or a str (the page navigates to that URL)."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.url = "about:blank"
        self.handlers = []
        self.scrolls = 0
        self.gotos: list[str] = []

    def on(self, event, handler):
        assert event == "response"
        self.handlers.append(handler)

    def _deliver(self, i):
        if i >= len(self.steps):
            return
        step = self.steps[i]
        if isinstance(step, str):
            self.url = step
            return
        for r in step:
            for h in self.handlers:
                h(r)

    def goto(self, url, wait_until=None):
        self.url = url
        self.gotos.append(url)
        self._deliver(0)

    def evaluate(self, js):
        self.scrolls += 1
        self._deliver(self.scrolls)

    def wait_for_timeout(self, ms):
        pass

    def title(self):
        return "search | LinkedIn"


@pytest.fixture
def fake_browser(monkeypatch):
    monkeypatch.setattr(lp, "FIRST_RESPONSE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(lp, "RESPONSE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(lp, "PAUSE_MS", (0, 0))
    monkeypatch.setattr(lp, "now_utc", lambda: NOW)
    calls: dict = {"closed": 0}

    def install(steps, raise_on_open: Exception | None = None):
        page = FakePage(steps)

        def opener(**kw):
            calls["kwargs"] = kw
            if raise_on_open:
                raise raise_on_open

            def close():
                calls["closed"] += 1
            return x_web.BrowserHandle("chrome", None, page, close)
        monkeypatch.setattr(x_web, "_open_browser", opener)
        calls["page"] = page
        return calls
    return install


def make_ctx(tmp_path: Path, deadline_s: float = 60, **cfg: str) -> CollectContext:
    conf = {"LINKEDIN_COOKIE": COOKIE, **cfg}
    return CollectContext(run_dir=tmp_path, raw_dir=tmp_path / "raw", config=conf,
                          deadline=time.monotonic() + deadline_s)


def collect(tmp_path, query=None, **cfg):
    return lp.LinkedInPostsCollector().collect(query or q(), make_ctx(tmp_path, **cfg))


T19 = [datetime(2026, 11, 19, h, 0, tzinfo=UTC) for h in (12, 10, 8)]
T18 = [datetime(2026, 11, 18, h, 0, tzinfo=UTC) for h in (20, 15, 9)]
OLD = datetime(2026, 11, 17, 22, 0, tzinfo=UTC)
TXT = "News about Bank of America today"


# ================================================================= collect: walk


def test_walk_stops_when_every_day_has_its_share(tmp_path, fake_browser):
    calls = fake_browser([
        [doc_resp([item(t, TXT, f"u{i}") for i, t in enumerate(T19)])],
        [page_resp([item(t, TXT, f"v{i}") for i, t in enumerate(T18)], next_start=6)],
        [page_resp([item(OLD, TXT, "old")], next_start=9)],
    ])
    res = collect(tmp_path, q(limit=4))
    st = res.status[0]
    assert st.state == SourceState.ok, st.reason
    assert st.date_filtered is False and st.query == '"bank of america"'
    # Cap split equally across the two days, newest kept within each day.
    assert [p.published_at for p in res.posts] == [
        "2026-11-19T12:00:00Z", "2026-11-19T10:00:00Z",
        "2026-11-18T20:00:00Z", "2026-11-18T15:00:00Z"]
    assert all(p.timestamp_verified for p in res.posts)
    assert "per-day: 2026-11-19 2, 2026-11-18 2" in st.reason
    # Every day already held its equal share: more results cannot change the split.
    assert "every day already had its share of candidates" in st.reason
    assert calls["page"].scrolls == 1 and calls["closed"] == 1
    assert "datePosted=%5B%22past-week%22%5D" in calls["page"].gotos[0]
    # Raw records are saved and referenced; they hold no cookie value.
    raw = sorted((tmp_path / "raw" / "linkedin").glob("search_page_*.json"))
    assert len(raw) == 2 and res.posts[0].raw_ref == "raw/linkedin/search_page_1.json"
    for f in raw:
        body = f.read_text(encoding="utf-8")
        assert LI_AT not in body and JSID not in body
        assert json.loads(body)["items"]


def test_walk_passes_window_start(tmp_path, fake_browser):
    calls = fake_browser([
        [doc_resp([item(t, TXT, f"u{i}") for i, t in enumerate(T19)])],
        [page_resp([item(t, TXT, f"v{i}") for i, t in enumerate(T18)], next_start=6)],
        [page_resp([item(OLD, TXT, "old")], next_start=9)],
        [page_resp([item(OLD - timedelta(hours=1), TXT, "older")], next_start=12)],
        [page_resp([item(OLD - timedelta(hours=2), TXT, "oldest")], next_start=15)],
        [page_resp([item(OLD - timedelta(hours=3), TXT, "never")], next_start=18)],
    ])
    res = collect(tmp_path, q(limit=30))
    st = res.status[0]
    assert st.state == SourceState.ok and len(res.posts) == 6
    assert "the walk passed the window start" in st.reason
    assert "3 result(s) older than the window start dropped" in st.reason
    assert ("per-day: 2026-11-19 3 (fewer than its share of 15), "
            "2026-11-18 3 (fewer than its share of 15)") in st.reason
    assert calls["page"].scrolls == 4  # the "never" step is not requested


def test_in_window_result_after_an_older_one_is_still_collected(tmp_path, fake_browser):
    # Seen live: date_posted order is not strict; one older result does not end the walk.
    late = datetime(2026, 11, 18, 2, 0, tzinfo=UTC)
    fake_browser([
        [doc_resp([item(T19[0], TXT), item(OLD, TXT, "old1")])],
        [page_resp([item(late, TXT, "late"), item(OLD - timedelta(hours=2), TXT, "o2")])],
        [page_resp([], next_start=None)],
    ])
    res = collect(tmp_path)
    assert {p.author_handle for p in res.posts} == {"in/alice", "in/late"}
    assert res.status[0].state == SourceState.ok


def test_phrase_filter_drops_loose_matches(tmp_path, fake_browser):
    fake_browser([
        [doc_resp([item(T19[0], TXT, "a"), item(T19[1], "Bank of the Americas", "b")],
                  next_start=None)],
    ])
    res = collect(tmp_path)
    st = res.status[0]
    assert [p.author_handle for p in res.posts] == ["in/a"]
    assert "1 result(s) without the exact phrase were not kept" in st.reason
    assert st.state == SourceState.ok
    assert "LinkedIn returned no further results page" in st.reason


def test_newer_than_window_end_dropped(tmp_path, fake_browser):
    fake_browser([[doc_resp([item(T19[0], TXT), item(T18[0], TXT, "b")], next_start=None)]])
    res = collect(tmp_path, q(start=datetime(2026, 11, 18, tzinfo=UTC),
                              end=datetime(2026, 11, 18, 23, 59, 59, tzinfo=UTC)))
    assert [p.author_handle for p in res.posts] == ["in/b"]
    assert "1 result(s) newer than the window end dropped" in res.status[0].reason


def test_end_of_results_with_nothing_is_completed_zero(tmp_path, fake_browser):
    fake_browser([[doc_resp([], next_start=None)]])
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.completed_zero, st.reason


def test_missing_results_structure_is_failed_not_zero(tmp_path, fake_browser):
    fake_browser([[doc_resp([], next_start=None, marker=False)]])
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.failed
    assert "expected results structure" in st.reason


def test_scroll_cap_is_partial_with_cursor(tmp_path, fake_browser):
    fake_browser([
        [doc_resp([item(T19[0], TXT)])],
        [page_resp([item(T19[1], TXT, "b")], next_start=6)],
        [page_resp([item(T19[2], TXT, "c")], next_start=9)],
    ])
    res = collect(tmp_path, LINKEDIN_MAX_SCROLLS="1")
    st = res.status[0]
    assert st.state == SourceState.partial
    assert "LINKEDIN_MAX_SCROLLS=1" in st.reason and len(res.posts) == 2
    assert st.cursor == "2026-11-19:startIndex=6"
    assert "2026-11-18 not reached" in st.reason


def test_one_stray_old_result_does_not_mark_unwalked_days_reached(tmp_path, fake_browser):
    # Seen live: date_posted order is not strict (an 11-06 result before an 11-18 one).
    # One stray result from 11-13 in the first page must not make the per-day summary
    # claim that 11-14..11-18 were searched and held nothing: the walk never got there.
    stray = datetime(2026, 11, 13, 9, 0, tzinfo=UTC)
    fake_browser([
        [doc_resp([item(T19[0], TXT), item(stray, TXT, "stray"), item(T19[1], TXT, "b")])],
        [page_resp([item(T19[2], TXT, "c")], next_start=6)],
    ])
    res = collect(tmp_path, q(start=datetime(2026, 11, 12, tzinfo=UTC), limit=30),
                  LINKEDIN_MAX_SCROLLS="1")
    st = res.status[0]
    assert st.state == SourceState.partial
    assert len(res.posts) == 4  # the stray in-window post is still kept as evidence
    for day in ("2026-11-18", "2026-11-17", "2026-11-16", "2026-11-15", "2026-11-14"):
        assert f"{day} not reached" in st.reason, st.reason
    assert "reached back to 2026-11-13 09:00 UTC" in st.reason  # the fact is still stated
    assert st.cursor.startswith("2026-11-19:")


def test_three_scrolls_without_new_results_end_the_walk(tmp_path, fake_browser):
    calls = fake_browser([[doc_resp([item(T19[0], TXT)])], [], [], [], []])
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.partial
    assert "3 consecutive scrolls brought no new results" in st.reason
    assert calls["page"].scrolls == 3


def test_repeated_results_count_as_no_new(tmp_path, fake_browser):
    same = [item(T19[0], TXT)]
    calls = fake_browser([[doc_resp(same)], [page_resp(same)], [page_resp(same)],
                          [page_resp(same)], [page_resp(same)]])
    st = collect(tmp_path).status[0]
    assert calls["page"].scrolls == 3 and "repeated result(s) ignored" in st.reason


def test_other_search_responses_are_ignored(tmp_path, fake_browser):
    fake_browser([[doc_resp([item(T19[0], TXT)], next_start=None),
                   page_resp([item(T19[1], TXT, "z")], kw='"something else"')]])
    res = collect(tmp_path)
    assert [p.author_handle for p in res.posts] == ["in/alice"]
    assert "1 response(s) for another search ignored" in res.status[0].reason


def test_echoed_keywords_compared_case_insensitively(tmp_path, fake_browser):
    # The analyst typed "Bank of America"; an echo in another case or spacing is the same
    # search and its results must not be thrown away as "another search".
    fake_browser([[doc_resp([item(T19[0], TXT)]),
                   page_resp([item(T19[1], TXT, "z")], kw='"bank  of america"',
                             next_start=None)]])
    res = collect(tmp_path, q(topic="Bank of America"))
    assert {p.author_handle for p in res.posts} == {"in/alice", "in/z"}
    assert "another search ignored" not in res.status[0].reason


# ============================================================ collect: stop rules


def test_login_redirect_is_failed(tmp_path, fake_browser):
    fake_browser(["https://www.linkedin.com/login?session_redirect=x"])
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.failed and "login page" in st.reason


@pytest.mark.parametrize("url,word", [
    ("https://www.linkedin.com/authwall?trk=x", "authwall"),
    ("https://www.linkedin.com/checkpoint/lg/login-submit", "checkpoint"),
    ("https://www.linkedin.com/checkpoint/challenge/AgF123", "CAPTCHA"),
    ("https://www.linkedin.com/uas/login?x=1", "login page"),
])
def test_checkpoint_after_posts_is_partial_and_stops(tmp_path, fake_browser, url, word):
    calls = fake_browser([[doc_resp([item(T19[0], TXT)])], url,
                          [page_resp([item(T19[1], TXT, "b")])]])
    res = collect(tmp_path)
    st = res.status[0]
    assert st.state == SourceState.partial and word in st.reason
    assert len(res.posts) == 1 and calls["page"].scrolls == 1  # stopped at once


def test_429_on_search_response_stops(tmp_path, fake_browser):
    calls = fake_browser([[doc_resp([item(T19[0], TXT)])],
                          [FakeResponse(PAGINATION, "", status=429)], [page_resp([])]])
    res = collect(tmp_path)
    st = res.status[0]
    assert st.state == SourceState.partial and "HTTP 429" in st.reason
    assert calls["page"].scrolls == 1


def test_429_on_any_linkedin_request_stops_before_results(tmp_path, fake_browser):
    fake_browser([[FakeResponse("https://www.linkedin.com/voyager/api/x", status=429)]])
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.failed and "HTTP 429" in st.reason


def test_403_from_pagination_mentions_session(tmp_path, fake_browser):
    fake_browser([[doc_resp([item(T19[0], TXT)])],
                  [FakeResponse(PAGINATION, "CSRF check failed.", status=403)]])
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.partial and "HTTP 403" in st.reason
    assert "LINKEDIN_COOKIE" in st.reason


def test_no_first_response_is_failed(tmp_path, fake_browser):
    fake_browser([[]])
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.failed and "no search results response" in st.reason


def test_deadline_is_partial(tmp_path, fake_browser):
    fake_browser([[doc_resp([item(T19[0], TXT)])]])
    res = lp.LinkedInPostsCollector().collect(q(), make_ctx(tmp_path, deadline_s=-1))
    assert res.status[0].state == SourceState.partial
    assert "time budget exhausted" in res.status[0].reason


def test_no_browser_is_failed(tmp_path, fake_browser):
    fake_browser([], raise_on_open=x_web.NoBrowserError("nope"))
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.failed and "no browser" in st.reason


def test_browser_error_redacts_cookie(tmp_path, fake_browser):
    def boom(page):
        raise RuntimeError(f"crash with li_at={LI_AT} and {JSID}")
    calls = fake_browser([[doc_resp([item(T19[0], TXT)])]])
    calls["page"].evaluate = lambda js: boom(calls["page"])
    res = collect(tmp_path)
    st = res.status[0]
    assert st.state == SourceState.partial and "RuntimeError" in st.reason
    assert LI_AT not in st.reason and JSID not in st.reason


# ======================================================== collect: credentials


def test_missing_cookie_is_skipped_without_browser(tmp_path, fake_browser):
    calls = fake_browser([])
    st = lp.LinkedInPostsCollector().collect(q(), make_ctx(tmp_path, LINKEDIN_COOKIE="")) \
        .status[0]
    assert st.state == SourceState.skipped_no_credentials and "kwargs" not in calls
    assert "per-day: 2026-11-19 not reached, 2026-11-18 not reached" in st.reason


def test_cookie_without_li_at_is_skipped(tmp_path, fake_browser):
    calls = fake_browser([])
    st = collect(tmp_path, LINKEDIN_COOKIE="JSESSIONID=abc; lang=en").status[0]
    assert st.state == SourceState.skipped_no_credentials and "li_at" in st.reason
    assert "kwargs" not in calls


def test_cookies_and_headless_passed_to_browser(tmp_path, fake_browser):
    calls = fake_browser([[doc_resp([], next_start=None)]])
    collect(tmp_path, LINKEDIN_COOKIE=LI_AT, LINKEDIN_BROWSER_HEADLESS="false")
    kw = calls["kwargs"]
    assert kw["headless"] is False
    assert kw["cookies"] == [{"name": "li_at", "value": LI_AT, "domain": ".linkedin.com",
                              "path": "/", "secure": True, "httpOnly": True}]


def test_one_collection_at_a_time(tmp_path, fake_browser):
    fake_browser([])
    assert lp._lock.acquire(blocking=False)
    try:
        st = collect(tmp_path).status[0]
    finally:
        lp._lock.release()
    assert st.state == SourceState.failed and "already running" in st.reason


# ============================================================== registry / doctor


def test_registry_and_capabilities():
    assert BUILTIN["linkedin"] == "socmint.collect.linkedin_posts:LinkedInPostsCollector"
    assert DEFAULT_SOURCES == ("x", "telegram", "youtube", "linkedin")
    caps = get_collector("linkedin").capabilities()
    assert caps.needs_credentials and caps.credential_keys == ("LINKEDIN_COOKIE",)
    assert caps.kind == "social" and caps.supports_date_filter is False
    assert "restricts accounts" in caps.description


def _row(cfg):
    return next(r for r in doctor.source_readiness(cfg) if r["name"] == "linkedin")


def test_doctor_missing_cookie_skips():
    row = _row({})
    assert row["outcome"] == "skip" and "skipped_no_credentials" in row["reason"]
    assert row["keys_set"] == {"LINKEDIN_COOKIE": False}


def test_doctor_cookie_without_li_at_warns(monkeypatch):
    monkeypatch.setattr(doctor, "x_browser_status", lambda: ("chrome", "Chrome"))
    row = _row({"LINKEDIN_COOKIE": "JSESSIONID=ajax:99999999; lang=en"})
    assert row["outcome"] == "skip" and "li_at" in row["warning"]
    assert "ajax:99999999" not in json.dumps(row)


def test_doctor_ready_and_no_browser(monkeypatch):
    monkeypatch.setattr(doctor, "x_browser_status", lambda: ("chrome", "Chrome"))
    row = _row({"LINKEDIN_COOKIE": COOKIE})
    assert row["ready"] and row["outcome"] == "run" and "4 cookie(s) incl. li_at" in \
        row["reason"]
    text = doctor.format_report({**doctor.collect_report(
        {"LINKEDIN_COOKIE": COOKIE}, check_telegram=False, check_model=False)})
    assert LI_AT not in text and JSID not in text and LI_AT not in json.dumps(row)
    monkeypatch.setattr(doctor, "x_browser_status", lambda: (None, "no browser here"))
    row = _row({"LINKEDIN_COOKIE": LI_AT})
    assert row["outcome"] == "fail" and "li_at only" in row["reason"]


# ===================================================================== settings


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def test_settings_validate_linkedin_keys(client):
    for bad in ("0", "201", "x"):
        assert client.post("/api/settings",
                           json={"LINKEDIN_MAX_SCROLLS": bad}).status_code == 400
    assert client.post("/api/settings",
                       json={"LINKEDIN_BROWSER_HEADLESS": "maybe"}).status_code == 400
    r = client.post("/api/settings", json={"LINKEDIN_MAX_SCROLLS": "40",
                                           "LINKEDIN_BROWSER_HEADLESS": "false",
                                           "LINKEDIN_COOKIE": COOKIE})
    assert r.status_code == 200
    cfg = r.json()["config"]
    assert cfg["LINKEDIN_MAX_SCROLLS"] == "40" and cfg["LINKEDIN_BROWSER_HEADLESS"] == "false"
    assert LI_AT not in r.text and JSID not in r.text  # the cookie is a masked secret


def test_gui_has_linkedin_card_without_external_urls():
    js = (Path(__file__).parent.parent / "socmint" / "web" / "app.js").read_text("utf-8")
    assert "'LinkedIn posts'" in js and "LINKEDIN_COOKIE" in js
    assert "LinkedIn restricts accounts that" in js
    assert not re.search(r"https?://(?!www\.w3\.org/2000/svg)", js)


def test_cookie_secrets_skip_preference_cookies_so_evidence_text_is_not_altered():
    raw = (f'li_at={LI_AT}; JSESSIONID="{JSID}"; lang=v=2&lang=en-us; '
           "timezone=Asia/Kolkata; li_theme=light; bcookie=\"v=2&short\"")
    secrets = lp.cookie_secrets(raw)
    assert LI_AT in secrets and JSID in secrets
    text = "Posted from Asia/Kolkata in en-us, light theme"
    assert lp.scrub(text, secrets) == text  # preferences never redacted from post text
    assert "[REDACTED]" in lp.scrub(f"token {LI_AT}", secrets)
