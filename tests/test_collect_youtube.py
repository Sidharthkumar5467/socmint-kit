"""YouTube collector tests. No network: respx mocks the Data API with doc_example fixtures."""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from socmint.collect.youtube import (
    API,
    COST_SEARCH,
    YouTubeCollector,
    description_relations,
    extract_urls,
    map_comment_thread,
    map_video,
    youtube_video_id,
)
from socmint.interfaces import CollectContext, Query
from socmint.schema import SourceState
from socmint.verify import gate_timestamps

FX = Path(__file__).parent / "fixtures" / "youtube"
START = datetime(2026, 11, 1, tzinfo=UTC)
END = datetime(2026, 11, 30, 23, 59, 59, tzinfo=UTC)
KEY = "test-key-not-real"


def fx(name: str) -> dict:
    data = json.loads((FX / name).read_text(encoding="utf-8"))
    data.pop("_fixture_note", None)  # provenance note, not part of the API shape
    return data


def ctx_for(tmp_path, quota: int | None = None, deadline: float | None = None,
            key: str = KEY) -> CollectContext:
    cfg = {"YOUTUBE_API_KEY": key}
    if quota is not None:
        cfg["YOUTUBE_QUOTA_LIMIT"] = str(quota)
    run_dir = tmp_path / "run"
    return CollectContext(run_dir=run_dir, raw_dir=run_dir / "raw", config=cfg,
                          deadline=deadline if deadline is not None else time.monotonic() + 60)


# Collector tests use a one-day window (one time slice) unless they test slicing itself.
DAY_START = datetime(2026, 11, 5, tzinfo=UTC)
DAY_END = datetime(2026, 11, 5, 23, 59, 59, tzinfo=UTC)


def query(**opts) -> Query:
    return Query(topic="flood barrier", start=DAY_START, end=DAY_END, limit=500, options=opts)


def search_side_effect(request: httpx.Request) -> httpx.Response:
    token = request.url.params.get("pageToken")
    if token is None:
        return httpx.Response(200, json=fx("doc_example_search_page1.json"))
    assert token == "CDIQAA"
    return httpx.Response(200, json=fx("doc_example_search_page2.json"))


def mock_all(router: respx.Router, comments: httpx.Response | None = None) -> dict:
    routes = {
        "search": router.get(f"{API}/search").mock(side_effect=search_side_effect),
        "videos": router.get(f"{API}/videos").mock(
            return_value=httpx.Response(200, json=fx("doc_example_videos.json"))),
        "channels": router.get(f"{API}/channels").mock(
            return_value=httpx.Response(200, json=fx("doc_example_channels.json"))),
        "comments": router.get(f"{API}/commentThreads").mock(
            return_value=comments or httpx.Response(
                200, json=fx("doc_example_comment_threads.json"))),
    }
    return routes


# ------------------------------------------------------------------------ pure parsers


def test_youtube_video_id_variants():
    assert youtube_video_id("https://youtu.be/abc123") == "abc123"
    assert youtube_video_id("https://www.youtube.com/watch?v=xyz&t=3") == "xyz"
    assert youtube_video_id("https://m.youtube.com/shorts/s1") == "s1"
    assert youtube_video_id("https://example.org/watch?v=no") is None


def test_extract_urls_strips_trailing_punctuation_and_dedupes():
    t = "See https://a.example/x, and (https://b.example/y). Again https://a.example/x"
    assert extract_urls(t) == ["https://a.example/x", "https://b.example/y"]


def test_description_relations_are_direct_links_and_skip_self():
    rels = description_relations(
        "https://example.org/r.pdf https://youtu.be/OTHER https://youtu.be/SELF", "SELF")
    assert [r.target_url for r in rels] == ["https://example.org/r.pdf", "https://youtu.be/OTHER"]
    assert all(r.type == "link" and r.evidence == "direct" for r in rels)
    assert rels[1].target_post_id == "youtube:OTHER"
    assert rels[0].target_post_id is None


def test_map_video_prefers_videos_list_timestamp_and_parses_stats():
    v = fx("doc_example_videos.json")["items"][0]
    s = fx("doc_example_search_page1.json")["items"][0]
    ch = fx("doc_example_channels.json")["items"][0]
    p = map_video(v, s, ch, collected_at="2026-11-18T00:00:00Z", topic="t", raw_ref="raw/x")
    assert p is not None
    assert p.post_id == "youtube:vidAAAAAAA1" and p.kind == "video"
    # videos.list publishedAt (…:07Z), not the search snippet (…:00Z)
    assert p.published_at == "2026-11-05T14:00:07Z"
    assert p.timestamp_verified is True and p.timestamp_note is None
    assert p.collected_at == "2026-11-18T00:00:00Z"
    assert p.metrics.views == 15000 and p.metrics.likes == 900 and p.metrics.comments == 42
    assert p.text.startswith("Flood barrier collapse claim explained\n\nFull description.")
    assert p.url == "https://www.youtube.com/watch?v=vidAAAAAAA1"
    assert p.media[0].type == "video"
    assert p.media[0].thumbnail == "https://i.ytimg.com/vi/vidAAAAAAA1/hqdefault.jpg"
    assert p.author_id == "UCchannelOne0000000000"
    assert p.author_handle == "@channelone"
    assert p.author_created_at == "2026-10-20T12:00:00Z"
    assert p.author_avatar_url == "https://yt3.ggpht.com/c1-high.jpg"
    targets = {r.target_url: r.target_post_id for r in p.relations}
    assert targets == {"https://example.org/report.pdf": None,
                       "https://youtu.be/vidCCCCCCC3": "youtube:vidCCCCCCC3"}


def test_map_video_hidden_likes_are_none_not_zero():
    v = fx("doc_example_videos.json")["items"][1]
    p = map_video(v, None, None, collected_at="2026-11-18T00:00:00Z")
    assert p is not None
    assert p.metrics.likes is None and p.metrics.views == 300
    assert p.text == "سد الفيضان - ما الذي حدث؟"  # non-English kept verbatim


def test_map_video_falls_back_to_search_snippet_timestamp():
    s = fx("doc_example_search_page1.json")["items"][0]
    p = map_video(None, s, None, collected_at="2026-11-18T00:00:00Z")
    assert p is not None
    assert p.published_at == "2026-11-05T14:00:00Z" and p.timestamp_verified
    assert "search.list" in (p.timestamp_note or "")
    assert p.metrics.views is None


def test_map_video_without_timestamp_is_unverified():
    s = {"id": {"videoId": "x"}, "snippet": {"title": "t"}}
    p = map_video(None, s, None, collected_at="2026-11-18T00:00:00Z")
    assert p is not None and p.timestamp_verified is False and p.published_at is None


def test_map_comment_thread_reply_relation_and_window_gate():
    items = fx("doc_example_comment_threads.json")["items"]
    posts = [map_comment_thread(t, collected_at="2027-01-01T00:00:00Z") for t in items]
    a, b = posts
    assert a is not None and b is not None
    assert a.post_id == "youtube:comment:UgxThreadOne" and a.kind == "comment"
    assert a.published_at == "2026-11-06T10:00:00Z" and a.timestamp_verified
    assert a.author_id == "UCcommenter1" and a.author_handle == "@commenter1"
    assert a.metrics.likes == 7 and a.metrics.replies == 2
    assert a.relations[0].type == "reply"
    assert a.relations[0].target_post_id == "youtube:vidAAAAAAA1"
    assert a.url == "https://www.youtube.com/watch?v=vidAAAAAAA1&lc=UgxThreadOne"
    assert b.author_handle is None and b.author_name == "Plain Name"
    # Out-of-window comment is kept; the verify gate marks it.
    gate_timestamps(posts, START, END)
    assert a.in_window is True and b.in_window is False


# --------------------------------------------------------------------------- collector


def test_missing_key_is_skipped(tmp_path):
    res = YouTubeCollector().collect(query(), ctx_for(tmp_path, key=""))
    assert res.posts == []
    assert res.status[0].state == SourceState.skipped_no_credentials


def test_full_run_paginates_and_maps(tmp_path):
    ctx = ctx_for(tmp_path)
    with respx.mock(assert_all_called=True) as router:
        routes = mock_all(router)
        res = YouTubeCollector().collect(query(youtube_comments=5), ctx)
    st = res.status[0]
    assert st.state == SourceState.ok, st.reason
    videos = [p for p in res.posts if p.kind == "video"]
    comments = [p for p in res.posts if p.kind == "comment"]
    assert [p.post_id for p in videos] == ["youtube:vidAAAAAAA1", "youtube:vidBBBBBBB2",
                                           "youtube:vidCCCCCCC3"]
    assert routes["search"].call_count == 2
    # Search params: date window in RFC3339 and the key sent as a param.
    first = routes["search"].calls[0].request.url.params
    assert first["publishedAfter"] == "2026-11-05T00:00:00Z"
    assert first["publishedBefore"] == "2026-11-05T23:59:59Z"
    assert first["order"] == "relevance" and first["type"] == "video" and first["key"] == KEY
    # Comments only for videos with commentCount>0 (A and C), ranked by views.
    assert routes["comments"].call_count == 2
    assert routes["comments"].calls[0].request.url.params["videoId"] == "vidAAAAAAA1"
    assert len(comments) == 4  # 2 threads per mocked call
    # Quota: 2 searches×100 + 1 videos + 1 channels + 2 comment calls.
    assert st.quota_units == 2 * COST_SEARCH + 1 + 1 + 2
    assert st.requests == 6 and st.cursor is None and st.count == len(res.posts)
    # Raw payloads were saved and posts reference them.
    assert (ctx.run_dir / videos[0].raw_ref).exists()
    assert (ctx.raw_dir / "youtube" / "search_002.json").exists()
    # The API key never ends up in the saved payloads or post data.
    for f in (ctx.raw_dir / "youtube").iterdir():
        assert KEY not in f.read_text(encoding="utf-8")


def test_zero_results_is_completed_zero(tmp_path):
    with respx.mock as router:
        router.get(f"{API}/search").mock(
            return_value=httpx.Response(200, json=fx("doc_example_search_empty.json")))
        res = YouTubeCollector().collect(query(), ctx_for(tmp_path))
    st = res.status[0]
    assert res.posts == []
    assert st.state == SourceState.completed_zero
    assert st.reason and st.quota_units == COST_SEARCH


def test_quota_ceiling_stops_cleanly_with_cursor(tmp_path):
    # 150 units: one search page (100) fits, the second does not.
    with respx.mock as router:
        routes = mock_all(router)
        res = YouTubeCollector().collect(query(youtube_comments=0), ctx_for(tmp_path, 150))
    st = res.status[0]
    assert routes["search"].call_count == 1
    assert st.state == SourceState.partial
    assert "quota ceiling 150 units" in st.reason
    assert st.cursor == "2026-11-05:CDIQAA"
    assert st.quota_units == COST_SEARCH + 2  # + videos + channels still enriched
    assert {p.post_id for p in res.posts} == {"youtube:vidAAAAAAA1", "youtube:vidBBBBBBB2"}


def test_quota_ceiling_below_one_search_is_partial_not_zero(tmp_path):
    with respx.mock as router:
        route = router.get(f"{API}/search")
        res = YouTubeCollector().collect(query(), ctx_for(tmp_path, 50))
    st = res.status[0]
    assert route.call_count == 0
    assert st.state == SourceState.partial and "quota ceiling 50" in st.reason


def test_quota_exceeded_on_first_search_is_reported(tmp_path):
    with respx.mock as router:
        router.get(f"{API}/search").mock(
            return_value=httpx.Response(403, json=fx("doc_example_error_quota.json")))
        res = YouTubeCollector().collect(query(), ctx_for(tmp_path))
    st = res.status[0]
    assert res.posts == []
    assert st.state == SourceState.partial
    assert "quotaExceeded" in st.reason and st.state != SourceState.completed_zero


def test_quota_exceeded_with_truncated_body_still_detected(tmp_path):
    # Google repeats a long message before errors[].reason; HttpFailure truncates bodies.
    long_msg = ("The request cannot be completed because you have exceeded your "
                "<a href=\"/youtube/v3/getting-started#quota\">quota</a>. " * 4)
    body = {"error": {"code": 403, "message": long_msg,
                      "errors": [{"message": long_msg, "domain": "youtube.quota",
                                  "reason": "quotaExceeded"}]}}
    with respx.mock as router:
        mock_all(router)
        router.get(f"{API}/search").mock(side_effect=[
            httpx.Response(200, json=fx("doc_example_search_page1.json")),
            httpx.Response(403, json=body)])
        res = YouTubeCollector().collect(query(youtube_comments=0), ctx_for(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.partial
    assert "quotaExceeded" in st.reason and st.cursor == "2026-11-05:CDIQAA"
    assert len(res.posts) == 2


def test_quota_exceeded_during_comments_keeps_videos(tmp_path):
    with respx.mock as router:
        mock_all(router, comments=httpx.Response(403, json=fx("doc_example_error_quota.json")))
        res = YouTubeCollector().collect(query(youtube_comments=5), ctx_for(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.partial and "comments stopped" in st.reason
    assert len([p for p in res.posts if p.kind == "video"]) == 3


def test_comments_disabled_is_noted_not_failed(tmp_path):
    with respx.mock as router:
        mock_all(router, comments=httpx.Response(
            403, json=fx("doc_example_error_comments_disabled.json")))
        res = YouTubeCollector().collect(query(youtube_comments=5), ctx_for(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.ok
    assert "comments disabled on 2 video(s)" in st.reason


def test_non_quota_http_error_first_page_is_failed(tmp_path):
    with respx.mock as router:
        router.get(f"{API}/search").mock(return_value=httpx.Response(
            400, json={"error": {"code": 400, "message": "API key not valid."}}))
        res = YouTubeCollector().collect(query(), ctx_for(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.failed and "search.list failed" in st.reason
    assert KEY not in st.reason


def test_deadline_is_partial(tmp_path):
    with respx.mock as router:
        route = router.get(f"{API}/search")
        res = YouTubeCollector().collect(
            query(), ctx_for(tmp_path, deadline=time.monotonic() - 1))
    st = res.status[0]
    assert route.call_count == 0
    assert st.state == SourceState.partial and "time budget" in st.reason


def test_post_limit_caps_search(tmp_path):
    q = Query(topic="flood barrier", start=DAY_START, end=DAY_END, limit=2, options={
        "youtube_comments": 0})
    with respx.mock as router:
        routes = mock_all(router)
        res = YouTubeCollector().collect(q, ctx_for(tmp_path))
    assert routes["search"].call_count == 1
    assert routes["search"].calls[0].request.url.params["maxResults"] == "2"
    st = res.status[0]
    # Changed with per-slice search: the cursor is only kept when the source is partial
    # (every slice was searched, so ok has no resume point); the note names the slice.
    assert st.state == SourceState.ok and st.cursor is None
    assert "post limit" in st.reason and "more results exist for 2026-11-05" in st.reason


@pytest.mark.parametrize("opt", ["bad", None])
def test_bad_comment_option_falls_back_to_default(tmp_path, opt):
    with respx.mock as router:
        routes = mock_all(router)
        YouTubeCollector().collect(query(youtube_comments=opt), ctx_for(tmp_path))
    assert routes["comments"].call_count == 2


# ------------------------------------------------------------- review-finding regressions


def test_extract_urls_keeps_balanced_parentheses():
    t = ("Ref https://en.wikipedia.org/wiki/Flood_(disambiguation) and "
         "(see https://en.wikipedia.org/wiki/Dam_(structure)).")
    assert extract_urls(t) == ["https://en.wikipedia.org/wiki/Flood_(disambiguation)",
                               "https://en.wikipedia.org/wiki/Dam_(structure)"]


def test_videos_and_channels_list_do_not_send_max_results_with_id(tmp_path):
    # The docs say maxResults is not supported together with `id`.
    with respx.mock as router:
        routes = mock_all(router)
        YouTubeCollector().collect(query(youtube_comments=0), ctx_for(tmp_path))
    for name in ("videos", "channels"):
        params = routes[name].calls[0].request.url.params
        assert "id" in params and "maxResults" not in params


def test_zero_comment_videos_do_not_use_comment_slots(tmp_path):
    # Make B (commentCount "0") the 2nd most-viewed: A=15000, B=10000, C=5000.
    # With N=2, B must be dropped before slicing so both A and C are fetched.
    vids_payload = fx("doc_example_videos.json")
    vids_payload["items"][1]["statistics"]["viewCount"] = "10000"
    with respx.mock as router:
        routes = mock_all(router)
        routes["videos"].mock(return_value=httpx.Response(200, json=vids_payload))
        YouTubeCollector().collect(query(youtube_comments=2), ctx_for(tmp_path))
    vids = [c.request.url.params["videoId"] for c in routes["comments"].calls]
    assert vids == ["vidAAAAAAA1", "vidCCCCCCC3"]


def test_retried_attempts_are_charged_quota(tmp_path, monkeypatch):
    monkeypatch.setattr("socmint.util.http.time.sleep", lambda s: None)
    with respx.mock as router:
        mock_all(router)
        router.get(f"{API}/search").mock(side_effect=[
            httpx.Response(503), httpx.Response(503),
            httpx.Response(200, json=fx("doc_example_search_empty.json"))])
        res = YouTubeCollector().collect(query(), ctx_for(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.completed_zero
    assert st.requests == 3 and st.quota_units == 3 * COST_SEARCH


def test_retries_stop_at_quota_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr("socmint.util.http.time.sleep", lambda s: None)
    with respx.mock as router:
        route = router.get(f"{API}/search").mock(return_value=httpx.Response(503))
        res = YouTubeCollector().collect(query(), ctx_for(tmp_path, 250))
    st = res.status[0]
    assert route.call_count == 2 and st.quota_units == 2 * COST_SEARCH
    assert st.state == SourceState.partial and "quota ceiling 250" in st.reason


def test_comment_http_failures_make_source_partial(tmp_path, monkeypatch):
    monkeypatch.setattr("socmint.util.http.time.sleep", lambda s: None)
    with respx.mock as router:
        mock_all(router, comments=httpx.Response(500, text="backend error"))
        res = YouTubeCollector().collect(query(youtube_comments=5), ctx_for(tmp_path))
    st = res.status[0]
    assert st.state == SourceState.partial
    assert "commentThreads.list failed" in st.reason
    assert "comment collection failed for 2 of 2" in st.reason
    assert len([p for p in res.posts if p.kind == "video"]) == 3


# ------------------------------------------------------------- per-day slices + phrase


W_START = datetime(2026, 11, 3, tzinfo=UTC)
W_END = datetime(2026, 11, 5, 23, 59, 59, tzinfo=UTC)


def window_query(limit: int, topic: str = "bank of america", **opts) -> Query:
    opts.setdefault("youtube_comments", 0)
    return Query(topic=topic, start=W_START, end=W_END, limit=limit, options=opts)


def _item(vid: str, day: str) -> dict:
    return {"id": {"kind": "youtube#video", "videoId": vid},
            "snippet": {"publishedAt": f"{day}T12:00:00Z", "channelId": "UCsynthetic",
                        "title": f"title {vid}", "description": "", "channelTitle": "Ch"}}


def by_day(per_day: dict[str, int], page_size: int | None = None):
    """search.list fake: per_day[YYYY-MM-DD] synthetic videos, in pages of page_size."""
    def side_effect(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        day = p["publishedAfter"][:10]
        assert p["publishedBefore"][:10] == day  # one-day slices in these tests
        n = per_day.get(day, 0)
        size = min(int(p["maxResults"]), page_size or 50)
        offset = int(p.get("pageToken") or 0)
        items = [_item(f"{day}-{k}", day) for k in range(offset, min(n, offset + size))]
        body: dict = {"items": items}
        if offset + size < n:
            body["nextPageToken"] = str(offset + size)
        return httpx.Response(200, json=body)
    return side_effect


def videos_echo(request: httpx.Request) -> httpx.Response:
    ids = request.url.params["id"].split(",")
    return httpx.Response(200, json={"items": [
        {"id": v, "snippet": {"publishedAt": f"{v[:10]}T12:00:00Z", "channelId": "UCsynthetic",
                              "title": f"title {v}", "description": ""},
         "statistics": {"viewCount": "1"}} for v in ids]})


def _with_topic_tags(request: httpx.Request) -> httpx.Response:
    """videos_echo, but every video is tagged with the test topics (YouTube results are
    filtered client-side to videos that mention the topic, window.mentions_topic)."""
    resp = videos_echo(request)
    body = json.loads(resp.content)
    for v in body["items"]:
        v["snippet"]["tags"] = ["bank of america", "flood barrier", "x"]
    return httpx.Response(200, json=body)


def mock_window(router: respx.Router, per_day: dict[str, int], page_size: int | None = None):
    search = router.get(f"{API}/search").mock(side_effect=by_day(per_day, page_size))
    router.get(f"{API}/videos").mock(side_effect=_with_topic_tags)
    router.get(f"{API}/channels").mock(return_value=httpx.Response(200, json={"items": []}))
    return search


def _days(route) -> list[tuple[str, str]]:
    return [(c.request.url.params["publishedAfter"], c.request.url.params["publishedBefore"])
            for c in route.calls]


def test_one_search_per_day_newest_first_with_day_bounds(tmp_path):
    with respx.mock as router:
        search = mock_window(router, {"2026-11-05": 5, "2026-11-04": 5, "2026-11-03": 5})
        res = YouTubeCollector().collect(window_query(6), ctx_for(tmp_path))
    st = res.status[0]
    assert _days(search) == [("2026-11-05T00:00:00Z", "2026-11-05T23:59:59Z"),
                             ("2026-11-04T00:00:00Z", "2026-11-04T23:59:59Z"),
                             ("2026-11-03T00:00:00Z", "2026-11-03T23:59:59Z")]
    assert [c.request.url.params["maxResults"] for c in search.calls] == ["2", "2", "2"]
    assert all(c.request.url.params["order"] == "relevance" for c in search.calls)
    assert len(res.posts) == 6
    assert st.state == SourceState.ok and st.cursor is None
    assert "per-day: 2026-11-05 2, 2026-11-04 2, 2026-11-03 2" in st.reason
    assert "more results exist for 2026-11-05, 2026-11-04, 2026-11-03" in st.reason


def test_cap_shared_with_rollover_from_quiet_day(tmp_path):
    with respx.mock as router:
        search = mock_window(router, {"2026-11-05": 1, "2026-11-04": 9, "2026-11-03": 9})
        res = YouTubeCollector().collect(window_query(6), ctx_for(tmp_path))
    st = res.status[0]
    # 6/3 -> 2 for the newest day; it has 1, so 5 left over 2 days -> 3, then 2.
    assert [c.request.url.params["maxResults"] for c in search.calls] == ["2", "3", "2"]
    assert len(res.posts) == 6 and st.state == SourceState.ok
    assert ("per-day: 2026-11-05 1 (fewer than its share of 2), 2026-11-04 3, "
            "2026-11-03 2") in st.reason


def test_quiet_slice_is_searched_and_reported(tmp_path):
    with respx.mock as router:
        search = mock_window(router, {"2026-11-05": 2, "2026-11-03": 2})
        res = YouTubeCollector().collect(window_query(30), ctx_for(tmp_path))
    st = res.status[0]
    assert search.call_count == 3 and len(res.posts) == 4
    assert st.state == SourceState.ok
    assert "2026-11-04 0 (fewer than its share of 14)" in st.reason


def test_all_slices_empty_is_completed_zero(tmp_path):
    with respx.mock as router:
        search = mock_window(router, {})
        res = YouTubeCollector().collect(window_query(30), ctx_for(tmp_path))
    st = res.status[0]
    assert search.call_count == 3 and res.posts == []
    assert st.state == SourceState.completed_zero
    assert st.reason.startswith("search.list returned no videos")
    assert "per-day: 2026-11-05 0" in st.reason


def test_zero_allowance_slice_sends_no_request(tmp_path):
    with respx.mock as router:
        search = mock_window(router, {"2026-11-05": 5, "2026-11-04": 5, "2026-11-03": 5})
        res = YouTubeCollector().collect(window_query(2), ctx_for(tmp_path))
    st = res.status[0]
    # 2 over 3 days: 1, 1, then nothing left -> the oldest day sends no request.
    assert search.call_count == 2 and len(res.posts) == 2
    assert st.state == SourceState.ok and "2026-11-03 0" in st.reason
    # Review fix: the unsearched day must not read as a quiet day.
    assert "not searched, post limit 2 already used by newer slices: 2026-11-03" in st.reason


def test_limit_smaller_than_slices_names_every_unsearched_day(tmp_path):
    # Review regression: limit 5 over 7 days -> 5 days get 1 each, 2 oldest are never
    # searched; they must be named as not searched, not shown as plain zeros only.
    q = Query(topic="bank of america", start=datetime(2026, 11, 13, tzinfo=UTC),
              end=datetime(2026, 11, 19, 23, 59, 59, tzinfo=UTC), limit=5,
              options={"youtube_comments": 0})
    per_day = {f"2026-11-{d}": 3 for d in range(13, 20)}
    with respx.mock as router:
        search = mock_window(router, per_day)
        res = YouTubeCollector().collect(q, ctx_for(tmp_path))
    st = res.status[0]
    assert search.call_count == 5 and len(res.posts) == 5
    assert {c.request.url.params["publishedAfter"][:10] for c in search.calls} == {
        "2026-11-19", "2026-11-18", "2026-11-17", "2026-11-16", "2026-11-15"}
    assert st.state == SourceState.ok
    assert "not searched, post limit 5 already used by newer slices: 2026-11-14, " \
           "2026-11-13" in st.reason


def test_zero_limit_does_not_claim_search_returned_nothing(tmp_path):
    with respx.mock as router:
        search = mock_window(router, {"2026-11-05": 5})
        res = YouTubeCollector().collect(window_query(0), ctx_for(tmp_path))
    st = res.status[0]
    assert search.call_count == 0
    assert "search.list returned no videos" not in st.reason
    assert "not searched, post limit 0" in st.reason


def test_total_never_exceeds_limit_with_pagination(tmp_path):
    with respx.mock as router:
        search = mock_window(router, {"2026-11-05": 9, "2026-11-04": 9, "2026-11-03": 9},
                             page_size=2)
        res = YouTubeCollector().collect(window_query(7), ctx_for(tmp_path))
    assert len([p for p in res.posts if p.kind == "video"]) == 7
    # 3+2+2 posts in pages of 2: 2 + 1 + 1 pages.
    assert search.call_count == 4
    assert search.calls[1].request.url.params["pageToken"] == "2"


def test_time_budget_mid_way_is_partial_naming_unreached_days(tmp_path):
    ctx = ctx_for(tmp_path)
    inner = by_day({"2026-11-05": 1, "2026-11-04": 1, "2026-11-03": 1})

    def expire_after_first(request: httpx.Request) -> httpx.Response:
        ctx.deadline = time.monotonic() - 1  # budget runs out once day one is collected
        return inner(request)

    with respx.mock as router:
        search = router.get(f"{API}/search").mock(side_effect=expire_after_first)
        res = YouTubeCollector().collect(window_query(30), ctx)
    st = res.status[0]
    assert search.call_count == 1 and len(res.posts) == 1
    assert st.state == SourceState.partial
    assert "search stopped at 2026-11-04: time budget exhausted" in st.reason
    assert "2026-11-04 not reached, 2026-11-03 not reached" in st.reason
    assert st.cursor == "2026-11-04:"


def test_quota_ceiling_mid_slices_is_partial_with_slice_cursor(tmp_path):
    # Ceiling 400 -> at most (400 // 2) // 100 = 2 slices: Nov 4-5 and Nov 3.
    # The first slice paginates 1 video per page; the 5th search page breaks the ceiling.
    with respx.mock as router:
        search = mock_window(router, {})
        n = iter(range(1, 100))

        def one_per_page(request: httpx.Request) -> httpx.Response:
            k = next(n)
            return httpx.Response(200, json={"items": [_item(f"v{k}", "2026-11-05")],
                                             "nextPageToken": f"tok{k}"})
        search.mock(side_effect=one_per_page)
        res = YouTubeCollector().collect(window_query(100), ctx_for(tmp_path, 400))
    st = res.status[0]
    assert search.call_count == 4
    first = search.calls[0].request.url.params
    assert (first["publishedAfter"], first["publishedBefore"]) == (
        "2026-11-04T00:00:00Z", "2026-11-05T23:59:59Z")
    assert st.state == SourceState.partial
    assert "search stopped at 2026-11-04: quota ceiling 400 units" in st.reason
    assert ("per-slice: 2026-11-04..2026-11-05 4 (fewer than its share of 50), "
            "2026-11-03 not reached") in st.reason
    assert st.cursor == "2026-11-04:tok4"
    assert len(res.posts) == 4


def test_max_slices_bounded_by_half_the_quota(tmp_path):
    # 30-day window, default 2000 units -> 10 slices; 250 units -> 1 slice.
    q30 = Query(topic="x", start=START, end=END, limit=50, options={"youtube_comments": 0})
    with respx.mock as router:
        search = router.get(f"{API}/search").mock(
            return_value=httpx.Response(200, json={"items": []}))
        YouTubeCollector().collect(q30, ctx_for(tmp_path))
        assert search.call_count == 10
    with respx.mock as router:
        search = router.get(f"{API}/search").mock(
            return_value=httpx.Response(200, json={"items": []}))
        YouTubeCollector().collect(q30, ctx_for(tmp_path, 250))
        assert search.call_count == 1
        p = search.calls[0].request.url.params
        assert (p["publishedAfter"], p["publishedBefore"]) == (
            "2026-11-01T00:00:00Z", "2026-11-30T23:59:59Z")


@pytest.mark.parametrize("topic,opts,expected", [
    ("bank of america", {}, '"bank of america"'),
    ("bank of america", {"match": "words"}, "bank of america"),
    ("flood", {}, "flood"),
    ('"bank of america" OR bofa', {}, '"bank of america" OR bofa'),
])
def test_search_q_phrase_quoting(tmp_path, topic, opts, expected):
    with respx.mock as router:
        search = mock_window(router, {})
        YouTubeCollector().collect(window_query(5, topic=topic, **opts), ctx_for(tmp_path))
    assert {c.request.url.params["q"] for c in search.calls} == {expected}


def test_analyst_stop_keeps_metrics_for_videos_already_found(tmp_path):
    # "Stop collecting" during search: the search stage and comments stop, but the
    # videos already found still get videos.list / channels.list (metrics, channel data).
    ctx = ctx_for(tmp_path, deadline=math.inf)

    def search_then_stop(request):
        ctx.stop.set()  # the analyst presses Stop while page 1 is being fetched
        return search_side_effect(request)

    with respx.mock(assert_all_called=False) as router:
        routes = mock_all(router)
        routes["search"].mock(side_effect=search_then_stop)
        res = YouTubeCollector().collect(query(youtube_comments=5), ctx)
    st = res.status[0]
    assert routes["search"].call_count == 1  # page 2 was not requested
    assert routes["videos"].call_count == 1 and routes["channels"].call_count == 1
    assert routes["comments"].call_count == 0
    videos = [p for p in res.posts if p.kind == "video"]
    assert videos and all(p.metrics.views is not None for p in videos)
    assert st.state == SourceState.partial and "time budget exhausted" in st.reason


def test_off_topic_videos_are_dropped_and_counted(tmp_path):
    # Relevance search also matches captions / related terms: a video whose title,
    # description and tags never name the topic is not kept, and the status says how many.
    with respx.mock(assert_all_called=False) as router:
        mock_window(router, {"2026-11-05": 3})
        router.get(f"{API}/videos").mock(side_effect=videos_echo)  # no topic anywhere
        res = YouTubeCollector().collect(window_query(10), ctx_for(tmp_path))
    st = res.status[0]
    assert [p for p in res.posts if p.kind == "video"] == []
    assert "3 video(s) returned by the search did not mention the topic" in st.reason


@pytest.mark.parametrize("topic,match,text,ok", [
    ("sahibinden", "phrase", "SAHİBİNDEN SATILIK Penthouse", True),     # Turkish capital İ
    ("sahibinden", "phrase", "ilan sahibinden.com'da", True),           # suffix / domain
    ("sahibinden", "phrase", "#sahibinden", True),
    ("sahibinden", "phrase", "Beşiktaş vs Fenerbahçe", False),
    ("sahibinden", "phrase", "evsahibinden", False),                    # not a word start
    ("bank of america", "phrase", "Bank-of-America outage", True),
    ("bank of america", "phrase", "america's bank of choice", False),
    ("bank of america", "words", "america's bank of choice", True),
    ("vaccine OR vaccination", "phrase", "nothing here", True),         # expression: platform's job
])
def test_mentions_topic(topic, match, text, ok):
    from socmint.collect.window import mentions_topic
    q = Query(topic=topic, start=W_START, end=W_END, options={"match": match})
    assert mentions_topic(q, text) is ok


def test_search_queries_per_day_429_is_a_quota_stop():
    from socmint.collect.youtube import quota_reason
    from socmint.util.http import HttpFailure
    e = HttpFailure("HTTP 429 from www.googleapis.com: { \"error\": { \"code\": 429, "
                    "\"message\": \"Quota exceeded for quota metric 'Search Queries' and "
                    "limit 'Search Queries per day' of service", status=429)
    assert quota_reason(e) == "quotaExceeded (Search Queries per day)"
