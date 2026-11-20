"""Tests for the LinkedIn posts "voyager" provider (socmint/collect/linkedin_voyager.py).
No network: make_http_client is replaced by a fake client.

Parser tests run on tests/fixtures/linkedin_posts/live_voyager_search_*.json (captured live
2026-11-19 by the lead, 2 calls of count 10, trimmed; the viewer's member id is replaced by
a placeholder, see the README there). Control-flow tests use small synthetic bodies in the
same shape.
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient

from socmint import doctor
from socmint.collect import linkedin_posts as lp
from socmint.collect import linkedin_voyager as lv
from socmint.config import DEFAULTS
from socmint.interfaces import CollectContext, Query
from socmint.schema import SourceState
from socmint.server.app import create_app

FX = Path(__file__).parent / "fixtures" / "linkedin_posts"
COLLECTED = datetime(2026, 11, 19, 13, 56, 40, tzinfo=UTC)  # when the captures were made
LI_AT = "AQEDAliatSECRETvalue0123456789abcdefXYZ"
JSID = "ajax:1234567890123456789"
COOKIE = f'bcookie="v=2&abc"; li_at={LI_AT}; JSESSIONID="{JSID}"; lang=v=2&lang=xx-test'
PLACEHOLDER = "VIEWER_PLACEHOLDER"


def load(name: str) -> dict:
    return json.loads((FX / name).read_text(encoding="utf-8"))


SORTED = "live_voyager_search_date_sorted.json"
RELEVANCE = "live_voyager_search_relevance.json"


# ===================================================================== request


def q(topic="bank of america", start=datetime(2026, 11, 18, tzinfo=UTC),
      end=datetime(2026, 11, 19, 23, 59, 59, tzinfo=UTC), limit=30, **opts) -> Query:
    return Query(topic=topic, start=start, end=end, limit=limit, options=opts)


def test_url_variables_exact_and_params_sorted_by_key():
    url = lv.build_url('"bank of america"', 20, 10, "past-week", lv.DEFAULT_QUERY_ID)
    assert url == (
        "https://www.linkedin.com/voyager/api/graphql?variables=(start:20,count:10,"
        "origin:SWITCH_SEARCH_VERTICAL,query:(keywords:%22bank%20of%20america%22,"
        "flagshipSearchIntent:SEARCH_SRP,queryParameters:List("
        "(key:datePosted,value:List(past-week)),(key:resultType,value:List(CONTENT)),"
        "(key:sortBy,value:List(date_posted))),includeFiltersInResponse:false))"
        "&queryId=voyagerSearchDashClusters.5ba32757c00b31aea747c8bebb92855c")
    # No preset (window start older than past-month): no datePosted parameter.
    assert lv.query_params(None) == [("resultType", "CONTENT"), ("sortBy", "date_posted")]
    # Rest.li reserved characters in keywords are percent-encoded.
    assert "keywords:a%2Cb%28c%29%3Ad," in lv.build_url("a,b(c):d", 0, 10, None, "q.1")


def test_headers_csrf_is_jsessionid_without_quotes():
    h = lv.build_headers("Cookie: " + COOKIE, {"LINKEDIN_USER_AGENT": "UA/1"})
    assert h == {
        "cookie": f'bcookie="v=2&abc"; li_at={LI_AT}; JSESSIONID="{JSID}"; lang=v=2&lang=xx-test',
        "csrf-token": JSID,
        "x-restli-protocol-version": "2.0.0",
        "accept": "application/vnd.linkedin.normalized+json+2.1",
        "accept-language": "en-US,en;q=0.9",
        "user-agent": "UA/1",
    }
    assert lv.build_headers(COOKIE, {})["user-agent"] == lv.DEFAULT_USER_AGENT
    assert lv.jsessionid(LI_AT) is None


def test_config_defaults_match_module():
    assert DEFAULTS["LINKEDIN_PROVIDER"] == "browser"
    assert DEFAULTS["LINKEDIN_VOYAGER_QUERY_ID"] == lv.DEFAULT_QUERY_ID
    assert DEFAULTS["LINKEDIN_USER_AGENT"] == lv.DEFAULT_USER_AGENT
    assert lv.provider({}) == "browser" and lv.provider({"LINKEDIN_PROVIDER": "Voyager"}) == \
        "voyager"
    assert lv.provider({"LINKEDIN_PROVIDER": "other"}) == "browser"
    assert lv.query_id({"LINKEDIN_VOYAGER_QUERY_ID": " "}) == lv.DEFAULT_QUERY_ID


# ================================================================ parser (live)


def _posts(name: str) -> list:
    items = lv.extract_page(load(name))[0]
    return [lp.item_to_post(i, collected_at=COLLECTED, raw_ref="raw/linkedin/v.json",
                            query_str='"bank of america"') for i in items]


def test_live_date_sorted_page_items_in_result_order():
    items, page, promoted = lv.extract_page(load(SORTED))
    assert page == {"start": 0, "count": 10, "total": 245} and promoted == 0
    assert len(items) == 10
    # Result order follows elements[].items[] (not the order of `included`).
    assert items[0]["native_id"] == "7529180776194625538" and items[0]["time_label"] == "now"
    cbi = next(i for i in items if i["native_id"] == "7529179043053772800")
    assert cbi["author_handle"] == "company/cbi-scpa"
    assert cbi["author_url"] == "https://www.linkedin.com/company/cbi-scpa/"
    assert cbi["author_name"] == "CBI S.c.p.a. Società Benefit"
    assert cbi["time_label"] == "7m"
    assert cbi["text"].startswith("🔄 L'evoluzione delle infrastrutture")  # original language
    assert cbi["content_urn"] == "urn:li:activity:7529179043053772800"
    assert "urn:li:share:7529179041677930496" in cbi["urns"]
    # shareUrl carries the viewer's member id as rcm=: the query string is dropped.
    assert cbi["share_url"] == ("https://www.linkedin.com/posts/cbi-scpa_cbi-salonedeipagamenti"
                                "-agoraeqdelfuturo-activity-7529179043053772800-P0ze")
    assert cbi["media"][0]["type"] == "photo"
    assert cbi["media"][0]["url"].startswith("https://media.licdn.com/dms/image/v2/")
    assert cbi["author_avatar_url"].startswith("https://media.licdn.com/")
    mentions = [r["target_handle"] for r in cbi["relations"] if r["type"] == "mention"]
    assert "company/cbi-scpa" not in mentions  # the author is not a mention of itself
    assert any(h.startswith("in/") for h in mentions)
    assert any(h.startswith("company/") for h in mentions)


def test_live_posts_ids_urls_timestamps_verified_against_labels():
    posts = _posts(SORTED) + _posts(RELEVANCE)
    assert len(posts) == 20
    for p in posts:
        nid = p.post_id.split(":", 1)[1]
        assert p.url == f"https://www.linkedin.com/feed/update/urn:li:activity:{nid}/"
        assert p.timestamp_verified, (p.post_id, p.timestamp_note)
        assert p.author_handle and p.author_handle.startswith(("in/", "company/"))
        assert p.text
    by_id = {p.post_id: p for p in posts}
    assert by_id["linkedin:7529179043053772800"].published_at == "2026-11-19T13:49:29Z"
    # "3w • Edited •" label: still parsed as 3w and consistent with the id.
    assert by_id["linkedin:7521566724303978496"].published_at == "2026-10-29T13:40:51Z"


def test_live_metrics_from_social_activity_counts():
    items = lv.extract_page(load(RELEVANCE))[0]
    sharon = next(i for i in items if i["native_id"] == "7528482349140008960")
    # SocialDetail -> *totalSocialActivityCounts (keyed by the ugcPost) -> counts.
    assert sharon["metrics"]["reactions"] == 86
    assert sharon["metrics"]["comments"] == 4 and sharon["metrics"]["reposts"] == 9
    assert all(i["metrics"]["reactions"] is not None for i in items)


def test_live_links_articles_jobs_and_lnkd_in():
    items = lv.extract_page(load(RELEVANCE))[0]
    by = {i["native_id"]: i for i in items}
    art = [r for r in by["7528492546617069568"]["relations"] if r["type"] == "link"]
    # linkedin.com article link without its trackingId query string.
    assert art[0]["target_url"].startswith("https://www.linkedin.com/pulse/world-lost-its-mind")
    assert "?" not in art[0]["target_url"]
    links = [r["target_url"] for i in items for r in i["relations"] if r["type"] == "link"]
    assert "https://lnkd.in/gYpuWbna" in links  # trailing ")" from the text stripped
    assert any(u.startswith("https://www.linkedin.com/jobs/view/") and "?" not in u
               for u in links)
    assert all(r["evidence"] == "direct" for i in items for r in i["relations"])
    assert not any(i["reshare"] for i in items)  # no reshare captured live


def test_extract_page_none_without_search_structure():
    assert lv.extract_page({"data": {"data": {}}, "included": []}) is None
    assert lv.extract_page({"errors": [{"message": "x"}]}) is None


# ============================================================ fixtures hygiene

_PROFILE_ID = re.compile(r"urn:li:fsd?_profile:\(?([A-Za-z0-9_-]+)")


def _viewer_ids(body: dict) -> set[str]:
    """Profile ids inside SocialPermissions urns (the viewer's own permissions)."""
    out: set[str] = set()
    for x in body["included"]:
        for k in ("entityUrn", "preDashEntityUrn", "*socialPermissions"):
            v = x.get(k)
            if isinstance(v, str) and "ocialPermissions" in v:
                out |= set(_PROFILE_ID.findall(v))
    return out


@pytest.mark.parametrize("name", [SORTED, RELEVANCE])
def test_fixtures_hold_only_the_viewer_placeholder(name):
    text = (FX / name).read_text(encoding="utf-8")
    body = json.loads(text)
    # Structural: the only viewer id in permission urns is the placeholder ...
    assert _viewer_ids(body) == {PLACEHOLDER}
    # ... every other profile id is an author / mentioned profile present in `included`
    # (a Profile entity, or an author's miniProfileUrn) ...
    known = {x["entityUrn"].rsplit(":", 1)[1] for x in body["included"]
             if x["$type"].endswith("Profile")}
    authors = set()
    for x in body["included"]:
        if x["$type"].endswith("feed.Update"):
            t = unquote((x["actor"].get("navigationContext") or {}).get("actionTarget", ""))
            authors |= set(_PROFILE_ID.findall(t))
    others = set(_PROFILE_ID.findall(unquote(text))) - {PLACEHOLDER}
    assert others <= known | authors, others - known - authors
    # ... and rcm= (the viewer id in share URLs) holds only the placeholder.
    assert set(re.findall(r"rcm=([A-Za-z0-9_-]+)", text)) == {PLACEHOLDER}
    low = text.lower()
    for bad in ("li_at", "jsessionid", "ajax:", "csrf"):
        assert bad not in low, bad


@pytest.mark.parametrize("name", [SORTED, RELEVANCE])
def test_saved_extract_has_no_viewer_or_permission_urns(name):
    items, page, _ = lv.extract_page(load(name))
    s = json.dumps({"paging": page, "items": items}, ensure_ascii=False)
    assert PLACEHOLDER not in s
    assert "fsd_profile" not in s and "ocialPermissions" not in s and "rcm=" not in s
    assert "miniProfileUrn" not in s


# ============================================================= synthetic bodies

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


def update(dt: datetime, text: str, vanity: str = "alice", low: int = 1,
           promoted: bool = False) -> dict:
    aid = aid_at(dt, low)
    sub = "Promoted" if promoted else f"{label_for(dt)} •  "
    return {
        "$type": "com.linkedin.voyager.dash.feed.Update",
        "entityUrn": f"urn:li:fsd_update:(urn:li:activity:{aid},BLENDED_SEARCH_FEED,EMPTY,"
                     "DEFAULT,false)",
        "metadata": {"backendUrn": f"urn:li:activity:{aid}"},
        "actor": {"name": {"text": vanity.title()}, "subDescription": {"text": sub},
                  "navigationContext": {"actionTarget":
                                        f"https://www.linkedin.com/in/{vanity}?miniProfileUrn=x"}},
        "commentary": {"text": {"text": text}},
    }


def body(updates: list[dict], start: int = 0, total: int | None = 100) -> dict:
    items = [{"item": {"searchFeedUpdate": {"*update": u["entityUrn"]}}} for u in updates]
    return {"data": {"data": {"searchDashClustersByAll": {
        "paging": {"start": start, "count": 25, "total": total},
        "elements": [{"items": [{"item": {"simpleTextV2": {"text": {"text": "n results"}}}}]},
                     {"items": items}]}}},
        "included": updates}


class FakeResp:
    def __init__(self, payload=None, status: int = 200, ctype: str = lv.ACCEPT,
                 headers: dict | None = None, text: str | None = None):
        self.status_code = status
        self.headers = {"content-type": ctype, **(headers or {})}
        self.text = text if text is not None else json.dumps(payload)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def get(self, url, headers=None):
        self.calls.append((url, dict(headers or {})))
        if not self.responses:
            raise AssertionError("unexpected extra request")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def close(self):
        self.closed = True


@pytest.fixture
def fake_http(monkeypatch):
    monkeypatch.setattr(lv, "PAUSE_S", (0.0, 0.0))
    monkeypatch.setattr(lv, "now_utc", lambda: NOW)
    monkeypatch.setattr(lp, "now_utc", lambda: NOW)
    box: dict = {}

    def install(responses):
        box["client"] = FakeClient(responses)
        monkeypatch.setattr(lv, "make_http_client", lambda: box["client"])
        return box["client"]
    return install


def make_ctx(tmp_path: Path, deadline_s: float = 60, **cfg: str) -> CollectContext:
    conf = {"LINKEDIN_COOKIE": COOKIE, "LINKEDIN_PROVIDER": "voyager", **cfg}
    return CollectContext(run_dir=tmp_path, raw_dir=tmp_path / "raw", config=conf,
                          deadline=time.monotonic() + deadline_s)


def collect(tmp_path, query=None, **cfg):
    return lp.LinkedInPostsCollector().collect(query or q(), make_ctx(tmp_path, **cfg))


T19 = [datetime(2026, 11, 19, h, 0, tzinfo=UTC) for h in (12, 10, 8)]
T18 = [datetime(2026, 11, 18, h, 0, tzinfo=UTC) for h in (20, 15, 9)]
OLD = datetime(2026, 11, 17, 22, 0, tzinfo=UTC)
TXT = "News about Bank of America today"


# ===================================================================== collect


def test_provider_switch_calls_voyager_with_exact_request(tmp_path, fake_http):
    client = fake_http([FakeResp(body([update(T19[0], TXT)], total=1))])
    res = collect(tmp_path)
    st = res.status[0]
    assert st.state == SourceState.ok, st.reason
    url, headers = client.calls[0]
    assert lv.DEFAULT_COUNT == 25  # verified live 2026-11-19
    assert url == lv.build_url('"bank of america"', 0, 25, "past-week", lv.DEFAULT_QUERY_ID)
    assert headers["csrf-token"] == JSID and headers["x-restli-protocol-version"] == "2.0.0"
    assert headers["accept"] == "application/vnd.linkedin.normalized+json+2.1"
    assert client.closed
    assert "paging.total" in st.reason and st.reason.startswith(lv.REASON_PREFIX)
    assert LI_AT not in st.reason and JSID not in st.reason


def test_browser_provider_is_default_and_never_calls_voyager(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(lv, "collect_voyager", lambda *a: called.append(a))
    monkeypatch.setattr(lp, "_collect", lambda *a: "browser")
    ctx = make_ctx(tmp_path, LINKEDIN_PROVIDER="")
    assert lp.LinkedInPostsCollector().collect(q(), ctx) == "browser" and not called


def test_walk_pages_split_cap_across_days_and_stop_when_full(tmp_path, fake_http):
    client = fake_http([
        FakeResp(body([update(t, TXT, f"u{i}") for i, t in enumerate(T19)])),
        FakeResp(body([update(t, TXT, f"v{i}") for i, t in enumerate(T18)], start=10)),
        FakeResp(body([update(OLD, TXT, "old")], start=20)),
    ])
    res = collect(tmp_path, q(limit=4))
    st = res.status[0]
    assert st.state == SourceState.ok, st.reason
    assert [p.published_at for p in res.posts] == [
        "2026-11-19T12:00:00Z", "2026-11-19T10:00:00Z",
        "2026-11-18T20:00:00Z", "2026-11-18T15:00:00Z"]
    assert all(p.timestamp_verified for p in res.posts)
    assert "per-day: 2026-11-19 2, 2026-11-18 2" in st.reason
    assert "every day already had its share" in st.reason
    assert len(client.calls) == 2
    assert "start:25,count:25" in client.calls[1][0]
    raw = sorted((tmp_path / "raw" / "linkedin").glob("voyager_page_*.json"))
    assert len(raw) == 2 and res.posts[0].raw_ref == "raw/linkedin/voyager_page_1.json"
    for f in raw:
        s = f.read_text(encoding="utf-8")
        assert LI_AT not in s and JSID not in s
        saved = json.loads(s)
        assert saved["items"] and saved["request"]["count"] == 25
        assert "included" not in saved and "data" not in saved


def test_walk_passes_window_start_after_older_results(tmp_path, fake_http):
    client = fake_http([
        FakeResp(body([update(T19[0], TXT), update(OLD, TXT, "o1"),
                       update(T18[0], TXT, "late")])),
        FakeResp(body([update(OLD - timedelta(hours=1), TXT, "o2"),
                       update(OLD - timedelta(hours=2), TXT, "o3")], start=10)),
        FakeResp(body([update(T18[1], TXT, "never")], start=20)),
    ])
    res = collect(tmp_path)
    st = res.status[0]
    assert {p.author_handle for p in res.posts} == {"in/alice", "in/late"}
    assert st.state == SourceState.ok and "the walk passed the window start" in st.reason
    assert len(client.calls) == 2


def test_phrase_filter_and_paging_total_end(tmp_path, fake_http):
    client = fake_http([FakeResp(body([update(T19[0], TXT, "a"),
                                       update(T19[1], "Bank of the Americas", "b")],
                                      total=2))])
    res = collect(tmp_path)
    st = res.status[0]
    assert [p.author_handle for p in res.posts] == ["in/a"]
    assert "1 result(s) without the exact phrase were not kept" in st.reason
    assert "all 2 result(s) LinkedIn reported (paging.total) were paged" in st.reason
    assert st.state == SourceState.ok and len(client.calls) == 1


def test_empty_page_ends_walk_completed_zero(tmp_path, fake_http):
    client = fake_http([FakeResp(body([], total=0))])
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.completed_zero, st.reason
    assert "page with no results" in st.reason and len(client.calls) == 1


def test_results_ending_early_on_a_big_search_are_linkedins_cap_not_empty_days(
        tmp_path, fake_http):
    # Live 2026-11-19: paging.total 245 for a week of a busy topic reached back only ~5 h.
    # LinkedIn reports 500 results but date order stops after one: the older day was NOT
    # searched by date order. The relevance pass (owner's tool) then runs; when it also
    # finds nothing older, the day must read "not reached" (partial), never empty (ok).
    client = fake_http([FakeResp(body([update(T19[0], TXT)], total=500)),
                        FakeResp(body([], start=25, total=500)),   # empty page 1 of 2
                        FakeResp(body([], start=50, total=500)),   # 2 in a row: date order ends
                        FakeResp(body([update(T19[0], TXT)], total=1200)),   # relevance p1
                        FakeResp(body([], start=25, total=1200)),
                        FakeResp(body([], start=50, total=1200))])
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.partial, st.reason
    assert "date order ended" in st.reason and len(client.calls) == 6
    assert "relevance" in st.reason and "2026-11-18 not reached" in st.reason
    rel_urls = [u for u, _ in client.calls[3:]]
    assert all("sortBy" not in u for u in rel_urls)          # relevance order: no sortBy
    assert all("date_posted" in u for u, _ in client.calls[:3])


def test_relevance_pass_finds_older_in_window_posts(tmp_path, fake_http):
    # Owner's tool: relevance order pages deeper than date order. Posts from the day date
    # order never reached are collected, and a complete relevance pass makes the run ok.
    fake_http([FakeResp(body([update(T19[0], TXT)], total=300)),
               FakeResp(body([], start=25, total=300)),
               FakeResp(body([], start=50, total=300)),
               FakeResp(body([update(T19[0], TXT), update(T18[0], TXT, low=2),
                              update(T18[1], TXT, low=3), update(OLD, TXT, low=4)],
                             total=4))])
    res = collect(tmp_path)
    st = res.status[0]
    assert st.state == SourceState.ok, st.reason
    days = sorted(p.published_at[:10] for p in res.posts)
    assert days == ["2026-11-18", "2026-11-18", "2026-11-19"]  # OLD is outside the window
    assert "relevance order: all 4 result(s) LinkedIn reported were paged (2 not already" \
           " seen" in st.reason or "relevance order: all 4 result(s)" in st.reason


def test_single_empty_page_does_not_end_the_walk(tmp_path, fake_http):
    # Owner's tool: LinkedIn interleaves non-post cards, so one page without posts is not
    # the end; only EMPTY_PAGE_TOLERANCE (2) in a row is.
    client = fake_http([FakeResp(body([update(T19[0], TXT)], total=60)),
                        FakeResp(body([], start=25, total=60)),
                        FakeResp(body([update(T18[0], TXT, low=2)], start=50, total=60))])
    res = collect(tmp_path)
    assert len(client.calls) == 3
    assert sorted(p.published_at[:10] for p in res.posts) == ["2026-11-18", "2026-11-19"]


def test_small_search_that_ends_is_complete_ok(tmp_path, fake_http):
    # A genuinely small search (well under LinkedIn's ~250 cap) that ends is complete.
    fake_http([FakeResp(body([update(T19[0], TXT)], total=1))])
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.ok, st.reason
    assert "LinkedIn caps" not in st.reason


def test_unresolved_result_refs_are_failed_not_zero(tmp_path, fake_http):
    # Review regression: results listed but no Update in `included` (shape change) used to
    # end as completed_zero, i.e. a parser failure looked like "no posts".
    b = body([update(T19[0], TXT)], total=245)
    b["included"] = []
    fake_http([FakeResp(b)])
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.failed, st.reason
    assert "none resolved to a post Update" in st.reason


def test_first_page_without_results_but_positive_total_is_failed(tmp_path, fake_http):
    # Review regression: a renamed result key left paging.total 245 with no parsed result;
    # this used to be completed_zero.
    b = body([update(T19[0], TXT)], total=245)
    for el in b["data"]["data"]["searchDashClustersByAll"]["elements"]:
        for it in el["items"]:
            if "searchFeedUpdate" in it["item"]:
                it["item"]["searchFeedUpdateV2"] = it["item"].pop("searchFeedUpdate")
    fake_http([FakeResp(b)])
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.failed, st.reason
    assert "paging.total 245" in st.reason and "not reported as zero posts" in st.reason


def test_promoted_items_skipped(tmp_path, fake_http):
    fake_http([FakeResp(body([update(T19[0], TXT), update(T19[1], TXT, "ad", promoted=True)],
                             total=2))])
    res = collect(tmp_path)
    assert [p.author_handle for p in res.posts] == ["in/alice"]
    assert "1 promoted item(s) skipped" in res.status[0].reason


def test_page_cap_is_partial_with_cursor(tmp_path, fake_http):
    client = fake_http([FakeResp(body([update(T19[0], TXT)]))])
    st = collect(tmp_path, LINKEDIN_MAX_SCROLLS="1").status[0]
    assert st.state == SourceState.partial and "LINKEDIN_MAX_SCROLLS=1" in st.reason
    assert st.cursor == "2026-11-19:start=25" and len(client.calls) == 1


@pytest.mark.parametrize("resp,word", [
    (FakeResp(status=429, text=""), "HTTP 429"),
    (FakeResp(status=999, text=""), "HTTP 999"),
    (FakeResp(status=401, text=""), "HTTP 401"),
    (FakeResp(status=403, text="CSRF check failed."), "CSRF"),
    (FakeResp(status=302, text="", headers={"location": "https://www.linkedin.com/login?x"}),
     "login page"),
    (FakeResp(status=303, text="",
              headers={"location": "https://www.linkedin.com/checkpoint/challenge/x"}),
     "CAPTCHA"),
    (FakeResp(status=302, text="", headers={"location": "https://www.linkedin.com/feed/"}),
     "redirect"),
    (FakeResp(status=200, ctype="text/html", text="<html>challenge</html>"), "not JSON"),
])
def test_stop_conditions_stop_at_once(tmp_path, fake_http, resp, word):
    client = fake_http([FakeResp(body([update(T19[0], TXT)])), resp,
                        FakeResp(body([update(T19[1], TXT, "b")], start=20))])
    res = collect(tmp_path)
    st = res.status[0]
    assert st.state == SourceState.partial and word in st.reason, st.reason
    assert len(res.posts) == 1 and len(client.calls) == 2  # never retried / continued


@pytest.mark.parametrize("code", [400, 404, 500])
def test_query_id_hint_on_400_404_500(tmp_path, fake_http, code):
    fake_http([FakeResp(status=code, text="{}", ctype="application/json")])
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.failed
    assert f"HTTP {code}" in st.reason and "queryId may be outdated" in st.reason
    assert "LINKEDIN_VOYAGER_QUERY_ID" in st.reason and "DevTools" in st.reason


def test_missing_structure_is_failed_not_zero(tmp_path, fake_http):
    fake_http([FakeResp({"data": {"data": {"other": {}}}, "included": []})])
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.failed and "searchDashClustersByAll" in st.reason


def test_network_error_redacted(tmp_path, fake_http):
    import httpx
    fake_http([httpx.ConnectError(f"boom {LI_AT}")])
    st = collect(tmp_path).status[0]
    assert st.state == SourceState.failed and "ConnectError" in st.reason
    assert LI_AT not in st.reason


def test_deadline_is_partial(tmp_path, fake_http):
    client = fake_http([])
    res = lp.LinkedInPostsCollector().collect(q(), make_ctx(tmp_path, deadline_s=-1))
    assert res.status[0].state == SourceState.partial
    assert "time budget exhausted" in res.status[0].reason and not client.calls


def test_cookie_without_jsessionid_is_skipped(tmp_path, fake_http):
    client = fake_http([])
    st = collect(tmp_path, LINKEDIN_COOKIE=LI_AT).status[0]
    assert st.state == SourceState.skipped_no_credentials and "JSESSIONID" in st.reason
    assert not client.calls


def test_custom_query_id_and_user_agent_sent(tmp_path, fake_http):
    client = fake_http([FakeResp(body([], total=0))])
    collect(tmp_path, LINKEDIN_VOYAGER_QUERY_ID="voyagerSearchDashClusters.abc123",
            LINKEDIN_USER_AGENT="MyUA/2")
    url, headers = client.calls[0]
    assert url.endswith("&queryId=voyagerSearchDashClusters.abc123")
    assert headers["user-agent"] == "MyUA/2"


# ===================================================================== doctor


def _row(cfg):
    return next(r for r in doctor.source_readiness(cfg) if r["name"] == "linkedin")


def test_doctor_voyager_ready_without_browser(monkeypatch):
    monkeypatch.setattr(doctor, "x_browser_status", lambda: (None, "no browser here"))
    row = _row({"LINKEDIN_COOKIE": COOKIE, "LINKEDIN_PROVIDER": "voyager"})
    assert row["ready"] and row["outcome"] == "run" and row["provider"] == "voyager"
    assert "provider voyager" in row["reason"] and lv.DEFAULT_QUERY_ID in row["reason"]
    assert LI_AT not in json.dumps(row) and JSID not in json.dumps(row)


def test_doctor_voyager_without_jsessionid_skips():
    row = _row({"LINKEDIN_COOKIE": LI_AT, "LINKEDIN_PROVIDER": "voyager"})
    assert row["outcome"] == "skip" and "JSESSIONID" in row["reason"]


def test_doctor_browser_provider_named(monkeypatch):
    monkeypatch.setattr(doctor, "x_browser_status", lambda: ("chrome", "Chrome"))
    row = _row({"LINKEDIN_COOKIE": COOKIE})
    assert row["provider"] == "browser" and "provider browser" in row["reason"]


# ==================================================================== settings


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def test_settings_validate_voyager_keys(client):
    assert client.post("/api/settings", json={"LINKEDIN_PROVIDER": "api"}).status_code == 400
    for bad in ("no dot here", "voyager.abc def", "a.b;c"):
        assert client.post("/api/settings",
                           json={"LINKEDIN_VOYAGER_QUERY_ID": bad}).status_code == 400
    assert client.post("/api/settings",
                       json={"LINKEDIN_USER_AGENT": "a\nb"}).status_code == 400
    r = client.post("/api/settings", json={
        "LINKEDIN_PROVIDER": "voyager",
        "LINKEDIN_VOYAGER_QUERY_ID": "voyagerSearchDashClusters.0123abcd",
        "LINKEDIN_USER_AGENT": "Mozilla/5.0 (X11) Chrome/142.0.0.0"})
    assert r.status_code == 200
    cfg = r.json()["config"]
    assert cfg["LINKEDIN_PROVIDER"] == "voyager"
    assert cfg["LINKEDIN_VOYAGER_QUERY_ID"] == "voyagerSearchDashClusters.0123abcd"
    # Empty restores the default.
    r = client.post("/api/settings", json={"LINKEDIN_VOYAGER_QUERY_ID": ""})
    assert r.json()["config"]["LINKEDIN_VOYAGER_QUERY_ID"] == lv.DEFAULT_QUERY_ID


def test_gui_card_has_provider_select_and_voyager_warning():
    js = (Path(__file__).parent.parent / "socmint" / "web" / "app.js").read_text("utf-8")
    assert "LINKEDIN_PROVIDER" in js and "LINKEDIN_VOYAGER_QUERY_ID" in js
    assert "LINKEDIN_USER_AGENT" in js
    assert "Voyager calls LinkedIn's API directly: faster, but no" in js
    assert "LinkedIn restricts accounts that" in js

