"""Plain-language headlines for source statuses (owner request 2026-11-20).

Collectors write precise, technical reasons ("HTTP 403 from www.googleapis.com: ...
quotaExceeded ..."). The analyst needs the main reason first, in simple words, with what to
do and, when the platform says so, when it resets. `headline()` turns one SourceStatus into

    Headline(text="YouTube's daily quota is used up, so YouTube results are limited.",
             kind="quota", action="It resets at 08:00 UTC (midnight Pacific time), in about
             4 h.", resets_at="2026-11-21T08:00:00Z")

and `source_headline()` picks the most important one for a source with several statuses
(Telegram: global search, joined groups, joined channels, seed channels).

Rules are ordered pattern matches on the technical reason; the first match wins. Nothing
here invents a fact: a reset time is given only when it is known (YouTube's documented
midnight-Pacific reset, X's x-rate-limit-reset header, Telegram's flood-wait seconds).
The full technical reason always stays available next to the headline.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from socmint.schema import SourceStatus
from socmint.util.timeutil import iso, parse_ts

PLATFORM = {"x": "X", "youtube": "YouTube", "telegram": "Telegram", "linkedin": "LinkedIn",
            "linkedin_ads": "LinkedIn Ads"}

# Most important first: what source_headline() shows for a source with several statuses.
KIND_ORDER = ("account", "quota", "rate_limit", "error", "setup", "stopped", "limit",
              "coverage", "none", "ok")


@dataclass
class Headline:
    text: str
    kind: str  # one of KIND_ORDER
    action: str | None = None
    resets_at: str | None = None  # ISO UTC, only when the platform tells us


def _name(source: str) -> str:
    return PLATFORM.get(source, source or "This source")


def _when(dt: datetime, now: datetime) -> str:
    """'08:00 UTC, in about 4 h' / '21 Nov 08:00 UTC, in about 2 days'."""
    secs = max(0, int((dt - now).total_seconds()))
    if secs < 3600:
        rel = f"in about {max(1, round(secs / 60))} min"
    elif secs < 48 * 3600:
        rel = f"in about {round(secs / 3600)} h"
    else:
        rel = f"in about {round(secs / 86400)} days"
    same_day = dt.date() == now.date()
    stamp = dt.strftime("%H:%M UTC") if same_day else dt.strftime("%d %b %H:%M UTC")
    return f"{stamp}, {rel}"


def youtube_quota_reset(now: datetime) -> datetime:
    """Next midnight Pacific time (YouTube Data API quotas reset then, per Google's docs)."""
    try:
        from zoneinfo import ZoneInfo

        pt = ZoneInfo("America/Los_Angeles")
        local = now.astimezone(pt)
        nxt = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return nxt.astimezone(UTC)
    except Exception:  # noqa: BLE001 - no tz database (e.g. bare Windows): PST, UTC-8
        local = now - timedelta(hours=8)
        nxt = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return (nxt + timedelta(hours=8)).replace(tzinfo=UTC)


def _first_clause(reason: str, limit: int = 160) -> str:
    """The first meaningful clause of a technical reason, shortened."""
    r = re.sub(r"^(web_session \(owner's logged-in account\)|LinkedIn Voyager content search "
               r"API \(owner's logged-in account\)|linkedin\.com content search \(owner's "
               r"logged-in account\)|API mode(; [^:;]+)?|web-preview mode \([^)]*\)):\s*", "",
               (reason or "").strip())
    first = re.split(r";\s+", r, maxsplit=1)[0].strip()
    if len(first) > limit:
        first = first[: limit - 1].rstrip() + "…"
    return first[:1].upper() + first[1:] if first else ""


def headline(st: SourceStatus | dict, now: datetime | None = None) -> Headline:
    """Plain-language headline for one source status."""
    if isinstance(st, dict):
        st = SourceStatus.model_validate(st)
    now = now or datetime.now(UTC)
    ref = parse_ts(st.finished_at) or now  # reset times are relative to when it stopped
    name = _name(st.source)
    state = st.state.value if hasattr(st.state, "value") else str(st.state)
    reason = st.reason or ""
    low = reason.lower()

    # ---- the analyst's own stop and run-level limits
    if "stopped by the analyst" in low:
        return Headline(f"You stopped collection; {name} kept the posts it had collected.",
                        "stopped")

    # ---- YouTube
    if st.source == "youtube":
        if "ratelimitexceeded" in low:
            return Headline("YouTube is limiting requests right now, so YouTube results are "
                            "limited.", "rate_limit", "Try again in a few minutes.")
        m = re.search(r"quota exceeded for quota metric '[^']+' and limit '([^']+)'|"
                      r"quotaexceeded \((?!http)([^)]+)\)", reason, re.I)
        if m:
            limit = m.group(1) or m.group(2)
            reset = youtube_quota_reset(ref)
            return Headline(f"YouTube's daily search limit is used up (Google's \"{limit}\"), "
                            "so YouTube results are limited.", "quota",
                            f"It resets at {_when(reset, now)} (midnight Pacific time). Each "
                            "YouTube search page uses one of these daily searches.", iso(reset))
        if re.search(r"quotaexceeded|dailylimitexceeded|api-side quota", low):
            reset = youtube_quota_reset(ref)
            return Headline("YouTube's daily quota is used up, so YouTube results are limited.",
                            "quota", f"It resets at {_when(reset, now)} (midnight Pacific "
                                     "time).", iso(reset))
        m = re.search(r"quota ceiling (\d+) units", low)
        if m:
            return Headline(f"Stopped at this tool's YouTube limit of {int(m.group(1)):,} quota "
                            "units per run.", "limit",
                            "Raise YOUTUBE_QUOTA_LIMIT in Settings to collect more (Google's "
                            "default allowance is 10,000 units a day).")
        if "http 400" in low and "invalid argument" in low:
            return Headline("YouTube rejected the search request.", "error",
                            "Check that the dates are not in the future and try again.")
        if "http 403" in low and ("api key" in low or "keyinvalid" in low or "forbidden" in low):
            return Headline("YouTube rejected the API key.", "setup",
                            "Check YOUTUBE_API_KEY in Settings.")

    # ---- X
    if st.source == "x":
        if "account access / challenge page" in low:
            return Headline("X is asking your account to verify itself, so X was not searched.",
                            "account", "Open x.com in your browser, complete X's check, then "
                                       "log in to X again in Settings.")
        if "x rate limit" in low or ("http 429" in low and "x" == st.source):
            m = re.search(r"resets at ([0-9T:\-Z+]+)", reason)
            reset = parse_ts(m.group(1)) if m else None
            return Headline("X's search limit for your account is used up, so X results are "
                            "limited.", "rate_limit",
                            f"It resets at {_when(reset, now)}." if reset
                            else "Try again later; X did not say when the limit resets.",
                            iso(reset) if reset else None)
        if "x session expired" in low:
            return Headline("Your X login has expired.", "account",
                            "Log in to X again in Settings.")
        if "no x session" in low:
            return Headline("X is not set up: no login.", "setup", "Log in to X in Settings.")
        if "browser error" in low:
            return Headline("The browser used for X stopped with an error.", "error",
                            "Run again; if it repeats, restart the tool.")

    # ---- LinkedIn (posts and ads)
    if st.source in ("linkedin", "linkedin_ads"):
        if re.search(r"checkpoint|captcha|challenge page|security check", low) or \
                "authwall/challenge" in low:
            if st.source == "linkedin_ads":
                return Headline("LinkedIn blocked the Ad Library page with a security check.",
                                "account", "Try again later (no login is needed for ads).")
            return Headline("LinkedIn is asking your account to verify itself.", "account",
                            "Open linkedin.com in your browser, complete the check, then run "
                            "again.")
        if re.search(r"redirected to its login|authwall|session or its csrf|expired or invalid"
                     r"|not accepted", low):
            return Headline("Your LinkedIn login has expired or was rejected.", "account",
                            "Paste a fresh Cookie header into LINKEDIN_COOKIE in Settings.")
        if re.search(r"rate limited|http (429|999)", low):
            return Headline(f"{name} asked the tool to slow down (rate limit).", "rate_limit",
                            "Wait before running again; LinkedIn does not say for how long.")
        if "queryid may be outdated" in low:
            return Headline("LinkedIn changed its search, so the tool's search ID is out of date.",
                            "error", "Update LINKEDIN_VOYAGER_QUERY_ID in Settings (the steps "
                                     "are in the details).")
        if "no linkedin session" in low or "has no li_at" in low or "has no jsessionid" in low:
            return Headline("LinkedIn is not set up: the login cookie is missing or incomplete.",
                            "setup", "Paste the full Cookie header into LINKEDIN_COOKIE in "
                                     "Settings.")
        if "linkedin caps" in low or ("date order ended" in low and state == "partial"):
            return Headline("LinkedIn only returns a limited number of results per search, so "
                            "some older days in the window were not reached.", "coverage",
                            "Shorter windows, or windows ending today, get fuller coverage.")

    # ---- Telegram
    if st.source == "telegram":
        m = re.search(r"flood wait (\d+)\s*s", low)
        if m:
            secs = int(m.group(1))
            reset = ref + timedelta(seconds=secs)
            mins = max(1, round(secs / 60))
            return Headline(f"Telegram asked your account to pause searches for about {mins} "
                            "min (flood wait).", "rate_limit",
                            f"Searches can resume at {_when(reset, now)}.", iso(reset))
        if "underground discovery:" in low and state in ("ok", "completed_zero"):
            m = re.search(r"(\d+) leak/underground channel\(s\) to search", low)
            if m and state == "ok":
                return Headline(f"Found {int(m.group(1))} data-leak / underground Telegram "
                                "channel(s); each was searched for this topic by date (see "
                                "their rows).", "ok")
            return Headline("No public data-leak or underground channel was found for this "
                            "topic.", "none")
        if "channel discovery:" in low and state in ("ok", "completed_zero"):
            m = re.search(r"(\d+) public channel\(s\) found", low)
            if m and state == "ok":
                return Headline(f"Found {int(m.group(1))} public Telegram channel(s) for this "
                                "topic; each was searched by date (see their rows).", "ok")
            return Headline("Telegram's directory listed no public channel or group for this "
                            "topic.", "none", "Add channels you know to TELEGRAM_SEED_CHANNELS "
                                              "in Settings.")
        m = re.search(r"page cap \((\d+)\) reached before the window start", low)
        if m and "global search" in low:
            back = re.search(r"reached back to (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) utc", low)
            newer = re.search(r"(\d+) result\(s\) outside the window", low)
            return Headline(
                "Telegram's global search cannot filter by date: it reads the newest posts "
                f"first, and its {int(m.group(1)) * 100:,}-result limit ran out"
                + (f" on posts newer than your window ({int(newer.group(1)):,} of them)"
                   if newer else "")
                + (f", reaching back only to {back.group(1)} UTC" if back else "")
                + ", so your window was not reached.", "coverage",
                "Choose a window that ends today, or add relevant channels to "
                "TELEGRAM_SEED_CHANNELS (they are searched by date). Your joined groups and "
                "channels are searched by date already.")
        if "free daily quota exhausted" in low:
            return Headline("Telegram Premium's free daily global-search quota is used up.",
                            "quota", "It resets daily; searches of your groups, channels and "
                                     "seed channels still ran.")
        if "not authorized" in low or "log in from settings" in low:
            return Headline("Telegram is not logged in.", "setup",
                            "Log in to Telegram in Settings.")
        if "needs telegram premium" in low:
            return Headline("Telegram global search needs Telegram Premium, and no channels "
                            "were set up to search instead.", "setup",
                            "Set TELEGRAM_HAS_PREMIUM, or add TELEGRAM_SEED_CHANNELS in "
                            "Settings.")

    # ---- shared
    m = re.search(r"missing credential\(s\): ([^;]+)", reason)
    if m or state == "skipped_no_credentials":
        keys = m.group(1).strip() if m else ""
        return Headline(f"{name} is not set up" + (f": {keys} is missing." if keys else "."),
                        "setup", "Add it in Settings to include this source.")
    if "no browser" in low:
        return Headline(f"No browser is available for {name}.", "setup",
                        "Run `socmint doctor --prefetch` or install Google Chrome.")
    if "time budget exhausted" in low:
        return Headline("Stopped at the run time limit.", "limit",
                        "Set RUN_TIME_BUDGET_SECONDS to 0 in Settings for no limit.")
    m = re.search(r"(x_max_scrolls|linkedin_max_scrolls)=(\d+)", low)
    if m and state == "partial":
        key = m.group(1).upper()
        return Headline(f"Stopped at the page limit ({key}={m.group(2)}).", "limit",
                        f"Raise {key} in Settings to go further back.")
    m = re.search(r"post limit (\d+)|share of the post limit (\d+)", low)
    if (m or "post limit reached" in low) and state == "partial":
        n = next((g for g in (m.groups() if m else ()) if g), None)
        return Headline("Reached this source's post limit"
                        + (f" ({int(n):,})." if n else "."), "limit",
                        "Use a custom range with a higher per-source limit to collect more.")
    if re.search(r"network error|connecterror|readtimeout|connecttimeout|timed out", low):
        return Headline(f"Could not reach {name} (network problem).", "error",
                        "Check the internet connection and run again.")
    if state == "out_of_coverage":
        return Headline(f"{name} does not cover these dates.", "coverage",
                        _first_clause(reason) or None)
    if state == "skipped_capability":
        return Headline(f"{name} was not searched: " + (_first_clause(reason) or "not supported "
                        "for this search") + ".", "setup")
    if state == "completed_zero":
        return Headline(f"{name} was searched; no posts were found for this topic in this "
                        "window.", "none")
    if state == "ok":
        return Headline(f"{name}: {st.count:,} post(s) collected.", "ok")
    if state == "failed":
        return Headline(f"{name} failed: " + (_first_clause(reason) or "unknown error") + ".",
                        "error")
    return Headline(f"{name} stopped early: " + (_first_clause(reason) or "see details") + ".",
                    "limit")


def source_headline(statuses: Iterable[SourceStatus | dict],
                    now: datetime | None = None) -> Headline | None:
    """The most important headline among one source's statuses (KIND_ORDER); for a source
    whose statuses are all fine, one "N posts collected" line with the total."""
    sts = [SourceStatus.model_validate(s) if isinstance(s, dict) else s for s in statuses]
    if not sts:
        return None
    heads = [headline(s, now) for s in sts]
    if all(h.kind == "ok" for h in heads):
        total = sum(s.count for s in sts)
        return Headline(f"{_name(sts[0].source)}: {total:,} post(s) collected.", "ok")
    return min(heads, key=lambda h: KIND_ORDER.index(h.kind))


def apply(statuses: list[SourceStatus], now: datetime | None = None) -> None:
    """Fill headline / headline_kind / action / resets_at on each status in place."""
    for st in statuses:
        h = headline(st, now)
        st.headline, st.headline_kind, st.action, st.resets_at = (h.text, h.kind, h.action,
                                                                  h.resets_at)
