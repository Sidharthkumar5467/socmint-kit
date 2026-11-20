"""LinkedIn Ad Library collector tests.

Fixtures in tests/fixtures/linkedin_ads/live_*.html are real public pages fetched on
2026-11-18 WITHOUT any cookie:
  live_search_bank.html          /ad-library/search?keyword=bank&dateOption=last-30-days
  live_search_bank_page2.html    /ad-library/searchPaginationFragment (page 2 of the above)
  live_search_bank_noresults.html custom-date-range ending today ("No results found")
  live_detail_1620739093.html    detail page WITH run dates, impressions and targeting
  live_detail_1690960776.html    detail page WITHOUT run dates
Short inline HTML strings below are synthetic and labelled as such.
"""

from __future__ import annotations

import re
import time
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
import respx

from socmint.collect import linkedin_ads as la
from socmint.interfaces import CollectContext, Query
from socmint.schema import SourceState
from socmint.verify import analysable, gate_timestamps, in_scope

FIX = Path(__file__).parent / "fixtures" / "linkedin_ads"
TODAY = date(2026, 11, 18)
PAGE1_TOKEN = "1690807156-1794868284327"
PAGE2_TOKEN = "1620795743-1794865730712"


def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    # util.http backs off with time.sleep on 429/5xx; keep tests fast.
    monkeypatch.setattr("socmint.util.http.time.sleep", lambda s: None)


def _ctx(tmp_path, *, config=None, budget=600.0) -> CollectContext:
    return CollectContext(run_dir=tmp_path, raw_dir=tmp_path / "raw", config=config or {},
                          deadline=time.monotonic() + budget)


def _query(start=date(2026, 10, 20), end=date(2026, 11, 18), **opts) -> Query:
    return Query(topic="bank",
                 start=datetime(start.year, start.month, start.day, tzinfo=UTC),
                 end=datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=UTC),
                 options=opts)


def _collector() -> la.LinkedInAdsCollector:
    return la.LinkedInAdsCollector(sleep=lambda s: None, today=lambda: TODAY)


# ------------------------------------------------------------------------ listing


def test_listing_extracts_every_card_from_live_page():
    rows = la.extract_listing(_read("live_search_bank.html"))
    assert len(rows) == 24
    assert len({r.detail_id for r in rows}) == 24
    assert all(r.detail_url.startswith("https://www.linkedin.com/ad-library/detail/")
               for r in rows)
    assert all(r.advertiser and r.commentary for r in rows)


def test_listing_is_per_card_not_zipped():
    """12 of 24 live cards have no headline; a zipped extraction would shift them."""
    rows = {r.detail_id: r for r in la.extract_listing(_read("live_search_bank.html"))}
    assert sum(1 for r in rows.values() if r.headline) == 12
    first = rows["1690960776"]
    assert first.advertiser == "IM Capital"
    assert first.headline == "16% p.a. second mortgage, Sydney and Gold Coast"
    # Card without headline stays None and does not steal the next card's headline.
    assert rows["1643931884"].headline is None
    assert rows["1643931884"].advertiser.startswith("AZIZ DIL")
    feedzai = rows["1690807156"]
    assert feedzai.advertiser == "Feedzai"
    assert feedzai.headline == "Securing banking in the AI agent era."
    assert feedzai.commentary.startswith("Your next digital banking customer")
    assert feedzai.advertiser_logo and feedzai.advertiser_logo.startswith("https://media.licdn")


def test_listing_creative_types_carousel_and_thought_leader():
    rows = {r.detail_id: r for r in la.extract_listing(_read("live_search_bank.html"))}
    video = rows["1690953036"]
    assert video.creative_type == "SPONSORED_VIDEO"
    assert video.advertiser == "VoltBackstage"
    assert "videocover" in video.images[0]
    assert len(rows["1690759706"].images) == 5  # multi-image card keeps every image
    assert rows["1643799834"].images == []
    # Thought-leader ad: member post "Promoted by Guru" -> advertiser is the payer.
    guru = rows["1690877236"]
    assert guru.advertiser == "Guru"
    assert guru.advertiser_logo  # member photo used as avatar


def test_listing_pagination_fragment_and_empty_page():
    rows = la.extract_listing(_read("live_search_bank_page2.html"))
    assert len(rows) == 24
    assert rows[0].advertiser == "Guaranty Bank & Trust"
    assert la.extract_listing(_read("live_search_bank_noresults.html")) == []


def test_next_token_patterns():
    assert la.extract_next_token(_read("live_search_bank.html")) == PAGE1_TOKEN
    assert la.extract_next_token(_read("live_search_bank_page2.html")) == PAGE2_TOKEN
    assert la.extract_next_token(_read("live_search_bank_noresults.html")) is None
    # Synthetic strings for the owner's two fallback patterns.
    assert la.extract_next_token('<div data-pagination-token="12-34">') == "12-34"
    assert la.extract_next_token('<a href="/x?a=1&paginationToken=56-78">') == "56-78"


# ------------------------------------------------------------------------- detail


def test_detail_with_run_dates_impressions_targeting():
    d = la.extract_detail(_read("live_detail_1620739093.html"))
    assert d.run_text == "Ran from Nov 16, 2026 to Nov 18, 2026"
    assert d.run_start == date(2026, 11, 16)
    assert d.run_end == date(2026, 11, 18)
    assert d.advertiser_id == "company/108834576"
    assert d.advertiser_url == "https://www.linkedin.com/company/108834576"
    assert d.advertiser == "Decade"
    assert d.paid_for_by == "Decade"
    assert d.impressions_range == "1k-5k"
    assert d.impression_countries == ["Brazil"]
    assert d.targeted_locations == ["Greater Belo Horizonte", "Greater São Paulo Area",
                                    "Greater Rio de Janeiro"]
    assert d.landing_url is None  # live CTA is a <button> without href
    assert d.commentary.startswith("Rodrigo Guedes Xavier")


def test_detail_without_run_dates():
    d = la.extract_detail(_read("live_detail_1690960776.html"))
    assert d.run_start is None and d.run_end is None
    assert d.advertiser_id == "company/88276905"
    assert d.paid_for_by == "IM Capital Pty Ltd"
    assert d.impressions_range is None
    assert d.impression_countries == []
    assert "Image for illustration purposes only." in d.commentary  # full, untruncated


@pytest.mark.parametrize("text,expected", [
    ("Ran from Nov 16, 2026 to Nov 18, 2026", (date(2026, 11, 16), date(2026, 11, 18))),
    ("Ran from September 3, 2026 to Oct 1, 2026", (date(2026, 9, 3), date(2026, 10, 1))),
    ("Ran on Nov 2, 2026", (date(2026, 11, 2), date(2026, 11, 2))),
    ("Ran from Nov 2, 2026", (date(2026, 11, 2), None)),
    ("Paid for by Someone", (None, None)),
    ("Ran from sometime to later", (None, None)),
])
def test_parse_run_dates(text, expected):
    assert la.parse_run_dates(text) == expected


# ------------------------------------------------------------------- date window


def test_date_params_window_reaching_today_uses_its_own_range_and_today_supplement():
    # Owner request 2026-11-20: the window's own dates ("Select date range"), ending
    # yesterday (live: an end date of today returns "No results found"), plus today's ads.
    dp = la.choose_date_params(date(2026, 10, 20), TODAY, TODAY)
    assert dp.params == {"dateOption": "custom-date-range", "startdate": "2026-10-20",
                         "enddate": "2026-11-17"}
    assert dp.today_supplement and "end capped at 2026-11-17" in dp.note


def test_date_params_window_of_only_today():
    dp = la.choose_date_params(TODAY, TODAY, TODAY)
    assert dp.params == {"dateOption": "last-30-days"} and dp.today_supplement


def test_date_params_past_window_has_no_supplement():
    dp = la.choose_date_params(date(2026, 11, 1), date(2026, 11, 10), TODAY)
    assert not dp.today_supplement


def test_date_params_past_window_uses_custom_range():
    dp = la.choose_date_params(date(2026, 11, 1), date(2026, 11, 10), TODAY)
    assert dp.params == {"dateOption": "custom-date-range", "startdate": "2026-11-01",
                         "enddate": "2026-11-10"}


def test_date_params_caps_end_at_yesterday_for_old_start():
    # Window reaches today but starts > 30 days ago: custom range capped at yesterday.
    dp = la.choose_date_params(date(2026, 10, 1), TODAY, TODAY)
    assert dp.params["enddate"] == "2026-11-17"
    assert "capped" in dp.note


def test_date_params_out_of_coverage():
    dp = la.choose_date_params(date(2025, 1, 1), date(2025, 1, 20), TODAY)
    assert dp.out_of_coverage


# ---------------------------------------------------------------------- blocking


def test_looks_blocked():
    for name in ("live_search_bank.html", "live_search_bank_page2.html",
                 "live_detail_1620739093.html", "live_search_bank_noresults.html"):
        assert not la.looks_blocked(_read(name))
    # Synthetic challenge body (no ad-library markup).
    assert la.looks_blocked('<html><a href="https://www.linkedin.com/authwall?x">Join</a>')
    assert la.looks_blocked("<html>ok</html>", "https://www.linkedin.com/authwall?trk=x")


# ------------------------------------------------------------------- post mapping


def test_to_post_with_run_dates():
    ad = {r.detail_id: r for r in la.extract_listing(_read("live_search_bank.html"))}[
        "1620739093"]
    d = la.extract_detail(_read("live_detail_1620739093.html"))
    p = la.to_post(ad, d, topic="bank", collected_at="2026-11-18T12:00:00Z",
                   raw_ref="raw/linkedin_ads/detail_1620739093.html")
    assert p.post_id == "linkedin_ads:1620739093"
    assert p.kind == "ad" and p.source == "linkedin_ads"
    assert p.published_at == "2026-11-16T00:00:00Z"
    assert p.timestamp_verified is True
    assert p.timestamp_note == "date-only precision (ad run start)"
    assert p.author_id == "company/108834576" and p.author_name == "Decade"
    assert p.url == "https://www.linkedin.com/ad-library/detail/1620739093"
    assert p.text.startswith("Rodrigo Guedes Xavier")
    assert "…" not in p.text  # detail page's full commentary replaced the truncated one
    assert p.ad_meta.run_start == "2026-11-16T00:00:00Z"
    assert p.ad_meta.run_end == "2026-11-18T23:59:59Z"
    assert p.ad_meta.impressions_range == "1k-5k"
    assert p.ad_meta.regions == ["Brazil"]
    assert p.media and p.media[0].type == "image"
    assert all(v is None for v in p.metrics.model_dump().values())
    assert p.relations == []  # no landing URL exposed


def test_to_post_without_detail_is_unverified():
    ad = la.extract_listing(_read("live_search_bank.html"))[0]
    p = la.to_post(ad, None, topic="bank", collected_at="2026-11-18T12:00:00Z",
                   raw_ref=None, detail_note="ad detail page not fetched (limit)")
    assert p.published_at is None
    assert p.timestamp_verified is False
    assert p.timestamp_note == "ad detail page not fetched (limit)"
    assert p.author_id is None and p.author_name == "IM Capital"
    assert p.text.endswith("16% p.a. second mortgage, Sydney and Gold Coast")


def test_to_post_detail_without_dates_and_video_media():
    rows = {r.detail_id: r for r in la.extract_listing(_read("live_search_bank.html"))}
    p = la.to_post(rows["1690960776"], la.extract_detail(_read("live_detail_1690960776.html")),
                   topic="bank", collected_at="2026-11-18T12:00:00Z", raw_ref=None)
    assert p.published_at is None and not p.timestamp_verified
    assert p.timestamp_note == "ad detail page shows no run dates"
    v = la.to_post(rows["1690953036"], None, topic="bank",
                   collected_at="2026-11-18T12:00:00Z", raw_ref=None)
    assert v.media[0].type == "video" and v.media[0].url is None and v.media[0].thumbnail


def test_member_advertiser_id_synthetic():
    # Synthetic snippet mirroring the live member-advertiser link shape (/in/<slug>).
    html = ('<a href="https://www.linkedin.com/in/some-member?trk=ad_library_about_ad_advertiser"'
            ' data-tracking-control-name="ad_library_about_ad_advertiser">Some Member</a>')
    d = la.extract_detail(html)
    assert d.advertiser_id == "in/some-member"
    assert d.advertiser == "Some Member"


def test_landing_url_from_synthetic_anchor_cta():
    # Synthetic: an <a> CTA has not been seen live; checks the defensive branch only.
    html = ('<div class="ad-preview"><p class="commentary__content">Hi '
            '<a href="https://example.org/x">x</a></p>'
            '<a data-tracking-control-name="ad_library_ad_detail_cta" '
            'href="https://landing.example/p">Learn</a></div>')
    d = la.extract_detail(html)
    assert d.landing_url == "https://landing.example/p"
    assert d.commentary_links == ["https://example.org/x"]


def test_verify_keeps_dated_ads_analysable():
    rows = {r.detail_id: r for r in la.extract_listing(_read("live_search_bank.html"))}
    dated = la.to_post(rows["1620739093"], la.extract_detail(_read("live_detail_1620739093.html")),
                       topic="bank", collected_at="2026-11-18T23:00:00Z", raw_ref=None)
    undated = la.to_post(rows["1690960776"], None, topic="bank",
                         collected_at="2026-11-18T23:00:00Z", raw_ref=None)
    q = _query()
    posts = gate_timestamps([dated, undated], q.start, q.end)
    assert [p.post_id for p in analysable(posts)] == ["linkedin_ads:1620739093"]


# --------------------------------------------------------------- collector (respx)

DETAIL_RE = re.compile(r"https://www\.linkedin\.com/ad-library/detail/(\d+)")


def _mock_detail(router: respx.MockRouter) -> respx.Route:
    def handler(request: httpx.Request) -> httpx.Response:
        did = DETAIL_RE.match(str(request.url)).group(1)
        if did == "1620739093":
            return httpx.Response(200, text=_read("live_detail_1620739093.html"))
        return httpx.Response(404, text="not found")

    return router.get(DETAIL_RE).mock(side_effect=handler)


def test_collect_full_flow(tmp_path):
    with respx.mock(assert_all_called=False) as router:
        search = router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(200, text=_read("live_search_bank.html")))
        frag = router.get("https://www.linkedin.com/ad-library/searchPaginationFragment")
        frag.side_effect = [
            httpx.Response(200, text=_read("live_search_bank_page2.html")),
            httpx.Response(200, text=_read("live_search_bank_noresults.html")),
        ]
        _mock_detail(router)
        res = _collector().collect(_query(end=date(2026, 11, 17), linkedin_detail_limit=25),
                                   _ctx(tmp_path))

    first = search.calls[0].request
    assert first.url.params["keyword"] == "bank"
    assert first.url.params["dateOption"] == "custom-date-range"
    assert first.url.params["startdate"] == "2026-10-20"
    assert first.url.params["enddate"] == "2026-11-17"
    assert "cookie" not in first.headers  # no cookie unless configured
    p2 = frag.calls[0].request.url.params
    assert p2["paginationToken"] == PAGE1_TOKEN and p2["countries"] == "ALL"
    assert frag.calls[1].request.url.params["paginationToken"] == PAGE2_TOKEN

    (st,) = res.status
    assert st.state == SourceState.ok, st.reason
    assert st.date_filtered is False
    assert st.count == len(res.posts) == 48
    assert "3 page(s)" in st.reason
    assert "detail pages fetched for 1/48 ads, 1 with run dates" in st.reason
    assert "24 detail page(s) failed" in st.reason
    assert st.requests == 3 + 25

    by_id = {p.post_id: p for p in res.posts}
    dated = by_id["linkedin_ads:1620739093"]
    assert dated.timestamp_verified and dated.published_at == "2026-11-16T00:00:00Z"
    assert dated.raw_ref == "raw/linkedin_ads/detail_1620739093.html"
    assert (tmp_path / dated.raw_ref).exists()
    beyond = [p for p in res.posts if "linkedin_detail_limit=25" in (p.timestamp_note or "")]
    assert len(beyond) == 48 - 25
    assert all(not p.timestamp_verified for p in beyond)
    assert all(p.raw_ref and p.raw_ref.startswith("raw/linkedin_ads/search_p")
               for p in beyond)


def test_collect_sends_cookie_only_when_configured(tmp_path):
    with respx.mock(assert_all_called=False) as router:
        search = router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(200, text=_read("live_search_bank_noresults.html")))
        _collector().collect(_query(), _ctx(tmp_path, config={"LINKEDIN_COOKIE": "abc=1"}))
    assert search.calls[0].request.headers["cookie"] == "abc=1"


def test_collect_completed_zero(tmp_path):
    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(200, text=_read("live_search_bank_noresults.html")))
        res = _collector().collect(_query(date(2026, 11, 1), date(2026, 11, 10)),
                                   _ctx(tmp_path))
    (st,) = res.status
    assert st.state == SourceState.completed_zero
    assert res.posts == []
    assert "custom-date-range 2026-11-01..2026-11-10" in st.reason
    assert st.date_filtered is False


def test_collect_authwall_fails(tmp_path):
    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(302, headers={
                "location": "https://www.linkedin.com/authwall?trk=ad_library"}))
        router.get("https://www.linkedin.com/authwall").mock(
            return_value=httpx.Response(200, text="<html>Sign in to continue</html>"))
        res = _collector().collect(_query(), _ctx(tmp_path))
    (st,) = res.status
    assert st.state == SourceState.failed
    assert st.reason.startswith("LinkedIn returned an authwall/challenge")


def test_collect_999_is_challenge(tmp_path):
    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(999, text=""))
        res = _collector().collect(_query(), _ctx(tmp_path))
    assert res.status[0].state == SourceState.failed
    assert "authwall/challenge" in res.status[0].reason


def test_collect_429_on_page2_is_partial_with_cursor(tmp_path):
    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(200, text=_read("live_search_bank.html")))
        router.get("https://www.linkedin.com/ad-library/searchPaginationFragment").mock(
            return_value=httpx.Response(429, text="slow down"))
        _mock_detail(router)
        res = _collector().collect(_query(linkedin_detail_limit=0), _ctx(tmp_path))
    (st,) = res.status
    assert st.state == SourceState.partial
    assert st.cursor == PAGE1_TOKEN
    assert "rate limited (429)" in st.reason
    assert len(res.posts) == 24


def test_collect_deadline_is_partial(tmp_path):
    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(200, text=_read("live_search_bank.html")))
        res = _collector().collect(_query(), _ctx(tmp_path, budget=3.0))
    (st,) = res.status
    assert st.state == SourceState.partial
    assert "time budget" in st.reason
    assert st.cursor == PAGE1_TOKEN
    assert len(res.posts) == 24
    assert all("not fetched" in (p.timestamp_note or "") for p in res.posts)


def test_collect_limit_is_partial(tmp_path):
    q = _query(linkedin_detail_limit=0)
    q.limit = 10
    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(200, text=_read("live_search_bank.html")))
        res = _collector().collect(q, _ctx(tmp_path))
    assert len(res.posts) == 10
    assert res.status[0].state == SourceState.partial
    assert res.status[0].cursor == PAGE1_TOKEN


def test_collect_out_of_coverage(tmp_path):
    res = _collector().collect(_query(date(2025, 1, 1), date(2025, 1, 20)), _ctx(tmp_path))
    assert res.status[0].state == SourceState.out_of_coverage
    assert res.posts == []


def test_capabilities():
    caps = la.LinkedInAdsCollector().capabilities()
    assert caps.kind == "ads" and not caps.needs_credentials
    assert caps.supports_date_filter is False


# ------------------------------------------------------- review regression tests

def _no_token(html: str) -> str:
    """Live page with its pagination token removed (simulates the final page)."""
    out = re.sub(r'"paginationToken"\s*:\s*"[0-9-]+"', '"x":"y"', html)
    out = re.sub(r'data-pagination-token="[0-9-]+"', "", out)
    out = re.sub(r"[?&]paginationToken=[0-9-]+", "", out)
    assert la.extract_next_token(out) is None
    return out


def test_collect_limit_on_final_page_is_partial(tmp_path):
    """Ads dropped by the limit on a page without a next token must not look 'ok'."""
    q = _query(linkedin_detail_limit=0)
    q.limit = 5
    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(200, text=_no_token(_read("live_search_bank.html"))))
        res = _collector().collect(q, _ctx(tmp_path))
    (st,) = res.status
    assert len(res.posts) == 5
    assert st.state == SourceState.partial
    assert st.reason.startswith("stopped at post limit 5")
    assert "19 retrieved ad(s) dropped by the post limit" in st.reason
    assert st.cursor is None


def test_collect_limit_within_last_pagination_page_is_partial(tmp_path):
    q = _query(linkedin_detail_limit=0)
    q.limit = 30  # 24 on page 1, only 6 of page 2's 24 fit; page 2 is the last page
    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(200, text=_read("live_search_bank.html")))
        router.get("https://www.linkedin.com/ad-library/searchPaginationFragment").mock(
            return_value=httpx.Response(
                200, text=_no_token(_read("live_search_bank_page2.html"))))
        res = _collector().collect(q, _ctx(tmp_path))
    (st,) = res.status
    assert len(res.posts) == 30
    assert st.state == SourceState.partial
    assert "18 retrieved ad(s) dropped" in st.reason


def test_collect_zero_cards_without_empty_marker_fails(tmp_path):
    """A markup change must not be reported as 'no ads' (completed_zero)."""
    html = _no_token(_read("live_search_bank.html")).replace(
        "ad_library_view_ad_detail", "renamed_control")
    assert la.extract_listing(html) == []
    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(200, text=html))
        res = _collector().collect(_query(), _ctx(tmp_path))
    (st,) = res.status
    assert st.state == SourceState.failed
    assert "no empty-results marker" in st.reason
    assert res.posts == []


def test_shows_no_results_marker():
    assert la.shows_no_results(_read("live_search_bank_noresults.html"))
    assert not la.shows_no_results(_read("live_search_bank.html"))
    assert not la.shows_no_results(_read("live_search_bank_page2.html"))


def test_collect_empty_fragment_without_marker_is_partial(tmp_path):
    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(200, text=_read("live_search_bank.html")))
        router.get("https://www.linkedin.com/ad-library/searchPaginationFragment").mock(
            # Synthetic: a fragment with no cards and no "No results found" heading.
            return_value=httpx.Response(200, text="<div><span>unexpected</span></div>"))
        res = _collector().collect(_query(linkedin_detail_limit=0), _ctx(tmp_path))
    (st,) = res.status
    assert st.state == SourceState.partial
    assert "page 2 parsed 0 ads" in st.reason
    assert st.cursor == PAGE1_TOKEN
    assert len(res.posts) == 24


def test_looks_blocked_inline_authwall_embedding_ad_library_url():
    # Synthetic: authwall served at the original URL, embedding the requested path.
    html = ('<a href="/authwall?sessionRedirect=https%3A%2F%2Fwww.linkedin.com'
            '%2Fad-library%2Fsearch">Sign in</a>')
    assert la.looks_blocked(html, "https://www.linkedin.com/ad-library/search")
    assert not la.has_library_markup(html)
    for name in ("live_search_bank.html", "live_search_bank_page2.html",
                 "live_detail_1620739093.html", "live_detail_1690960776.html",
                 "live_search_bank_noresults.html"):
        assert la.has_library_markup(_read(name)), name


def test_collect_inline_authwall_fails(tmp_path):
    html = ('<html><a href="/authwall?sessionRedirect=https%3A%2F%2Fwww.linkedin.com'
            '%2Fad-library%2Fsearch">Sign in</a></html>')
    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(200, text=html))
        res = _collector().collect(_query(), _ctx(tmp_path))
    (st,) = res.status
    assert st.state == SourceState.failed
    assert st.reason.startswith("LinkedIn returned an authwall/challenge")


def test_ad_started_before_window_is_not_analysable():
    """A pre-window run start must not become 'first observed' in the window."""
    rows = {r.detail_id: r for r in la.extract_listing(_read("live_search_bank.html"))}
    d = la.extract_detail(_read("live_detail_1620739093.html"))
    d.run_start = date(2026, 6, 1)  # modified copy of the live detail
    q = _query()
    p = la.to_post(rows["1620739093"], d, topic="bank", collected_at="2026-11-18T23:00:00Z",
                   raw_ref=None, window_start=q.start.date())
    assert p.published_at == "2026-06-01T00:00:00Z"  # true platform date is kept
    # The date is a real platform date, so it stays verified; the foundation keeps it
    # off timelines because published_at (run start) is outside the window.
    assert p.timestamp_verified is True
    assert "before the window start 2026-10-20" in p.timestamp_note
    assert p.ad_meta.run_start == "2026-06-01T00:00:00Z"
    posts = gate_timestamps([p], q.start, q.end)
    assert analysable(posts) == []
    assert p.in_window is False
    # Its run (no end date -> treated as a single day at run start) does not overlap.
    assert in_scope(posts) == ([p] if p.active_in_window else [])
    # Same ad with an in-window start stays verified.
    d.run_start = date(2026, 11, 16)
    p2 = la.to_post(rows["1620739093"], d, topic="bank", collected_at="2026-11-18T23:00:00Z",
                    raw_ref=None, window_start=q.start.date())
    assert p2.timestamp_verified is True


def test_deadline_backoff_failure_is_not_labelled_429(tmp_path):
    """util.http gives up a 5xx backoff at the deadline with rate_limited=True."""
    e =la.HttpFailure("time budget exhausted while backing off", rate_limited=True)
    assert la._http_kind(e) == "time budget exhausted while backing off"
    assert la._http_kind(la.HttpFailure("HTTP 429", status=429, rate_limited=True)) == \
        "rate limited (429)"
    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(200, text=_read("live_search_bank.html")))
        router.get("https://www.linkedin.com/ad-library/searchPaginationFragment").mock(
            return_value=httpx.Response(503, text="unavailable"))
        res = _collector().collect(_query(linkedin_detail_limit=0), _ctx(tmp_path))
    (st,) = res.status
    assert st.state == SourceState.partial
    assert "429" not in st.reason
    assert "pagination stopped at page 2" in st.reason


def test_cookie_not_sent_when_linkedin_posts_shares_the_run(tmp_path):
    # CLAUDE.md section 5: the posts source drives the same account in this run.
    q = _query()
    q.options = {**(q.options or {}), "sources": ("x", "linkedin", "linkedin_ads")}
    with respx.mock(assert_all_called=False) as router:
        search = router.get("https://www.linkedin.com/ad-library/search").mock(
            return_value=httpx.Response(200, text=_read("live_search_bank_noresults.html")))
        res = _collector().collect(q, _ctx(tmp_path, config={"LINKEDIN_COOKIE": "abc=1"}))
    assert "cookie" not in search.calls[0].request.headers
    assert "fetched without LINKEDIN_COOKIE" in res.status[0].reason


def test_today_supplement_keeps_only_ads_that_ran_in_the_window(tmp_path):
    # The date range (ending yesterday) finds nothing; the last-30-days listing has 24 ads.
    # Only the one whose detail page dates its run inside the window is kept; ads without
    # run dates cannot be placed and are not kept.
    with respx.mock(assert_all_called=False) as router:
        def search(request):
            if request.url.params.get("dateOption") == "custom-date-range":
                return httpx.Response(200, text=_read("live_search_bank_noresults.html"))
            assert request.url.params.get("dateOption") == "last-30-days"
            return httpx.Response(200, text=_read("live_search_bank.html"))
        route = router.get("https://www.linkedin.com/ad-library/search").mock(side_effect=search)
        router.get("https://www.linkedin.com/ad-library/searchPaginationFragment").mock(
            return_value=httpx.Response(200, text=_read("live_search_bank_noresults.html")))
        _mock_detail(router)
        res = _collector().collect(_query(start=date(2026, 11, 15), end=TODAY),
                                   _ctx(tmp_path))
    st = res.status[0]
    assert [c.request.url.params["dateOption"] for c in route.calls] == [
        "custom-date-range", "last-30-days"]
    assert [p.post_id for p in res.posts] == ["linkedin_ads:1620739093"]
    assert "window includes today: last-30-days listing read" in st.reason
    assert "1 ad(s) added from last-30-days (ran in the window)" in st.reason
    assert "without run dates not kept" in st.reason
