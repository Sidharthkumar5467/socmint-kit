"""Plain-language status headlines (socmint.status_text). Reasons are the collectors' real
wording, several copied from live runs on 2026-11-20."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from socmint import status_text as stx
from socmint.schema import SourceState, SourceStatus

NOW = datetime(2026, 11, 20, 3, 0, tzinfo=UTC)


def st(source, state, reason="", count=0, finished="2026-11-20T03:00:00Z"):
    return SourceStatus(source=source, query="q", state=SourceState(state), reason=reason,
                        count=count, finished_at=finished)


def test_youtube_daily_quota_with_reset_time():
    h = stx.headline(st("youtube", "partial",
                        "search stopped at 2026-11-19: YouTube API quotaExceeded (HTTP 403) "
                        "after 9800 units counted this run; API-side quota/limit reached"), NOW)
    assert h.kind == "quota"
    assert h.text == "YouTube's daily quota is used up, so YouTube results are limited."
    # midnight Pacific (PST, UTC-8) after 03:00 UTC on 20 Nov = 08:00 UTC on 20 Nov
    assert h.resets_at == "2026-11-20T08:00:00Z"
    assert h.action == "It resets at 08:00 UTC, in about 5 h (midnight Pacific time)."


def test_youtube_rate_limit_is_not_called_daily_quota():
    h = stx.headline(st("youtube", "partial", "YouTube API rateLimitExceeded (HTTP 403) after "
                                              "300 units counted this run; API-side quota/limit "
                                              "reached"), NOW)
    assert h.kind == "rate_limit" and "daily quota" not in h.text


def test_youtube_tool_quota_ceiling():
    h = stx.headline(st("youtube", "partial", "search stopped at 2026-11-18: quota ceiling "
                                              "6000 units"), NOW)
    assert h.kind == "limit" and "6,000 quota units" in h.text
    assert "YOUTUBE_QUOTA_LIMIT" in h.action


def test_youtube_future_dates_400():
    # live 2026-11-20 (1-30 Nov range started on 20 Nov)
    h = stx.headline(st("youtube", "failed", 'search.list failed at 2026-11-28: HTTP 400 from '
                        'www.googleapis.com: { "error": { "code": 400, "message": "Request '
                        'contains an invalid argument." } }'), NOW)
    assert h.kind == "error" and h.text == "YouTube rejected the search request."


def test_x_account_challenge():
    # live 2026-11-20
    h = stx.headline(st("x", "failed", "web_session (owner's logged-in account): X shows an "
                        "account access / challenge page (/account/access): open x.com in a "
                        "normal browser, resolve what X asks for, then log in again"), NOW)
    assert h.kind == "account"
    assert h.text == "X is asking your account to verify itself, so X was not searched."
    assert "log in to X again in Settings" in h.action


def test_x_rate_limit_reset_from_header():
    h = stx.headline(st("x", "partial", "web_session (owner's logged-in account): X rate limit "
                        "(429) — the account's search quota is exhausted; try again later (X "
                        "reports the limit resets at 2026-11-20T03:15:00Z)"), NOW)
    assert h.kind == "rate_limit" and h.resets_at == "2026-11-20T03:15:00Z"
    assert h.action == "It resets at 03:15 UTC, in about 15 min."


def test_telegram_flood_wait_reset():
    h = stx.headline(st("telegram", "partial", "API mode; joined groups: stopped: flood wait "
                                               "600s"), NOW)
    assert h.kind == "rate_limit" and "about 10 min" in h.text
    assert h.resets_at == "2026-11-20T03:10:00Z"


def test_telegram_premium_quota():
    h = stx.headline(st("telegram", "skipped_capability", "API mode; global search: free daily "
                        "quota exhausted (0/10); paid search (Stars) is never used"), NOW)
    assert h.kind == "quota" and "Premium" in h.text


@pytest.mark.parametrize("reason,kind,needle", [
    ("LinkedIn Voyager content search API (owner's logged-in account): stopped: HTTP 429 from "
     "LinkedIn: rate limited / blocked; stopped at once", "rate_limit", "slow down"),
    ("linkedin.com content search (owner's logged-in account): stopped: LinkedIn showed a "
     "security checkpoint: open linkedin.com in your normal browser", "account", "verify"),
    ("stopped: LinkedIn redirected to its login page: the session in LINKEDIN_COOKIE is expired "
     "or invalid", "account", "expired"),
    ("LinkedIn's results ended (paging.total 244) at 2026-10-22 13:20 UTC, before the window "
     "start: LinkedIn caps search results, so older days were not reached", "coverage",
     "limited number"),
])
def test_linkedin(reason, kind, needle):
    h = stx.headline(st("linkedin", "partial", reason), NOW)
    assert h.kind == kind and needle in h.text


def test_shared_rules():
    assert stx.headline(st("youtube", "skipped_no_credentials",
                           "missing credential(s): YOUTUBE_API_KEY"), NOW).text == \
        "YouTube is not set up: YOUTUBE_API_KEY is missing."
    assert stx.headline(st("x", "partial", "stopped by the analyst after 3 page(s)"),
                        NOW).kind == "stopped"
    assert stx.headline(st("youtube", "completed_zero", "search.list returned no videos"),
                        NOW).text.startswith("YouTube was searched; no posts were found")
    ok = stx.headline(st("telegram", "ok", "API mode; global search ...", count=1234), NOW)
    assert ok.kind == "ok" and ok.text == "Telegram: 1,234 post(s) collected."
    gen = stx.headline(st("linkedin_ads", "failed", "search page: something odd happened; "
                                                    "more detail"), NOW)
    assert gen.kind == "error" and gen.text == "LinkedIn Ads failed: Search page: something " \
                                               "odd happened."


def test_source_headline_picks_the_most_important():
    sts = [st("telegram", "ok", "global search", count=89),
           st("telegram", "partial", "API mode; joined groups: stopped: flood wait 120s"),
           st("telegram", "completed_zero", "joined channels")]
    assert stx.source_headline(sts, NOW).kind == "rate_limit"
    both_ok = [st("telegram", "ok", count=5), st("telegram", "ok", count=7)]
    assert stx.source_headline(both_ok, NOW).text == "Telegram: 12 post(s) collected."


def test_apply_fills_fields_and_legacy_bundle_gets_headlines():
    from socmint.legacy import upgrade_bundle

    sts = [st("x", "failed", "X session expired or invalid: log in again from Settings")]
    stx.apply(sts, NOW)
    assert sts[0].headline == "Your X login has expired." and sts[0].headline_kind == "account"
    old = {"run": {"status": "failed"}, "source_status": [
        {"source": "youtube", "query": "q", "state": "skipped_no_credentials",
         "reason": "missing credential(s): YOUTUBE_API_KEY"}]}
    out = upgrade_bundle(old)
    assert out["source_status"][0]["headline_kind"] == "setup"
    assert "headline" not in old["source_status"][0]  # input not mutated


def test_youtube_search_queries_per_day_429():
    # live 2026-11-20 (run 2026-11-20-sahibinden-2e935f), project number removed
    h = stx.headline(st("youtube", "partial", 'search.list failed at 2026-11-10: HTTP 429 from '
                        'www.googleapis.com: { "error": { "code": 429, "message": "Quota '
                        "exceeded for quota metric 'Search Queries' and limit 'Search Queries "
                        "per day' of service 'youtube.googleapis.com' for consumer "
                        "'project_number:0'.\", ..."), NOW)
    assert h.kind == "quota"
    assert h.text == ("YouTube's daily search limit is used up (Google's \"Search Queries per "
                      "day\"), so YouTube results are limited.")
    assert h.resets_at == "2026-11-20T08:00:00Z" and "midnight Pacific" in h.action


def test_telegram_global_search_did_not_reach_a_past_window():
    # live 2026-11-20 ("saudi aramco", window 14 Oct - 12 Nov)
    h = stx.headline(st("telegram", "partial",
                        "API mode; global search (channels.searchPosts, results walked newest "
                        "first and spread across days by platform timestamps); global search "
                        "has no date filter; reached back to 2026-11-19 15:17 UTC; 998 result(s) "
                        "outside the window dropped; page cap (10) reached before the window "
                        "start; per-slice: ..."), NOW)
    assert h.kind == "coverage"
    assert h.text == ("Telegram's global search cannot filter by date: it reads the newest "
                      "posts first, and its 1,000-result limit ran out on posts newer than your "
                      "window (998 of them), reaching back only to 2026-11-19 15:17 UTC, so "
                      "your window was not reached.")
    assert "TELEGRAM_SEED_CHANNELS" in h.action


def test_channel_discovery_rows():
    ok = stx.headline(st("telegram", "ok", 'API mode; channel discovery: 4 public channel(s) '
                         'found for "saudi aramco" (3 by name, 1 similar), each searched by date '
                         'below: @a, @b, @c, @d'), NOW)
    assert ok.kind == "ok" and ok.text.startswith("Found 4 public Telegram channel(s)")
    none = stx.headline(st("telegram", "completed_zero", "API mode; channel discovery: "
                           "Telegram's directory listed no public channel or group for \"x\""), NOW)
    assert none.kind == "none" and "TELEGRAM_SEED_CHANNELS" in none.action


def test_underground_discovery_headline():
    ok = stx.headline(st("telegram", "ok", 'API mode; underground discovery: 4 leak/underground '
                         'channel(s) to search for "aramco" (1 known + in global results, 2 by '
                         'leak vocabulary, 1 from the known list), each searched by date below: '
                         '@a, @b, @c, @d'), NOW)
    assert ok.kind == "ok" and ok.text.startswith("Found 4 data-leak / underground Telegram")
    none = stx.headline(st("telegram", "completed_zero", "API mode; underground discovery: no "
                           "public leak/underground channel found for this topic"), NOW)
    assert none.kind == "none"
