"""Timestamp gate, window filter, dedupe and language detection (docs/SPEC.md §7).

Rules:
- A post is `timestamp_verified` only if the collector took `published_at` from the
  platform's structured field (collector sets it) AND the value parses as tz-aware UTC
  AND it is not in the future relative to `collected_at` (5 min tolerance).
- Unverified posts are kept and shown, but excluded from timelines, counts and origin.
- `in_window` marks whether a verified timestamp falls inside [start, end]. Posts outside
  the window are kept for transparency (sources that cannot filter by date) but every
  analysis uses `analysable()` which requires in_window.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from socmint.schema import Post
from socmint.util.timeutil import iso, parse_ts

FUTURE_TOLERANCE = timedelta(minutes=5)


def gate_timestamps(posts: list[Post], start: datetime, end: datetime,
                    ads_start: datetime | None = None) -> list[Post]:
    """Normalise platform timestamps and mark in_window against [start, end].

    ads_start: start of the LinkedIn Ads search span when the run uses a longer ads
    lookback (pipeline.ads_window_start). An ad is active_in_window when its run overlaps
    [ads_start, end]; that keeps it in scope (narratives, coordination) while in_window,
    which timelines, counts and first-observed use, stays tied to the post window."""
    ad_from = min(start, ads_start) if ads_start is not None else start
    for p in posts:
        pub = parse_ts(p.published_at)
        col = parse_ts(p.collected_at)
        if pub is None:
            if p.timestamp_verified or not p.timestamp_note:
                p.timestamp_note = p.timestamp_note or "no parseable platform timestamp"
            p.timestamp_verified = False
            p.published_at = p.published_at or None
        else:
            p.published_at = iso(pub)
            if col is not None and pub > col + FUTURE_TOLERANCE:
                p.timestamp_verified = False
                p.timestamp_note = "platform timestamp is in the future relative to collection"
        # in_window: the platform timestamp itself lies inside the window. For an ad that
        # is its run START, so an ad that began before the window is not in_window and
        # never enters timelines/counts/origin, while its timestamp stays verified.
        if pub is not None:
            p.in_window = start <= pub <= end
        else:
            p.in_window = True  # unknown; kept, but excluded from timelines by verified flag
        if p.ad_meta is not None:
            # active_in_window: the ad's run overlaps the window at all (scope, not timeline).
            rs = parse_ts(p.ad_meta.run_start) or pub
            re_ = parse_ts(p.ad_meta.run_end) or rs
            p.active_in_window = bool(rs and re_ and rs <= end and re_ >= ad_from)
            if rs is None and ad_from < start:
                # Undated ad (no detail page) from a lookback run: it may have run only
                # before the post window, so it is not treated as in-window (kept for
                # narratives; never coordination evidence, timelines or counts).
                p.in_window = False
                note = (f"run dates unknown; the ads search reached back to "
                        f"{ad_from.date().isoformat()}, so this ad may have run only before "
                        "the post window")
                if note not in (p.timestamp_note or ""):
                    p.timestamp_note = "; ".join(x for x in (p.timestamp_note, note) if x)
    return posts


def dedupe(posts: list[Post]) -> list[Post]:
    """Collapse re-collections of the same platform post (same post_id).

    Keeps the first occurrence, merges relations and records extra queries. Distinct
    posts with identical text are NOT merged here: that is a coordination signal.
    """
    seen: dict[str, Post] = {}
    out: list[Post] = []
    for p in posts:
        prev = seen.get(p.post_id)
        if prev is None:
            seen[p.post_id] = p
            out.append(p)
            continue
        have = {(r.type, r.target_post_id, r.target_url, r.target_handle) for r in prev.relations}
        for r in p.relations:
            if (r.type, r.target_post_id, r.target_url, r.target_handle) not in have:
                prev.relations.append(r)
        if p.query and prev.query and p.query not in prev.query.split(" | "):
            prev.query = f"{prev.query} | {p.query}"
    return out


def detect_languages(posts: list[Post]) -> None:
    try:
        from langdetect import DetectorFactory, detect
        from langdetect.lang_detect_exception import LangDetectException
    except ImportError:
        return
    DetectorFactory.seed = 0
    for p in posts:
        if p.lang or len(p.text.strip()) < 12:
            continue
        try:
            p.lang = detect(p.text)
        except LangDetectException:
            p.lang = None


def analysable(posts: list[Post]) -> list[Post]:
    """Posts allowed into timelines, volume counts and origin analysis."""
    return [
        p for p in posts
        if p.timestamp_verified and p.in_window and not p.duplicate_of and not p.context_only
    ]


def in_scope(posts: list[Post]) -> list[Post]:
    """Posts that belong to the analysed sample (narratives, coordination, claims).

    Excludes context-only posts (embedded, not matched by the search), duplicates, and
    posts whose VERIFIED timestamp falls outside the window — except ads whose run was
    active during the window. Posts with unverified timestamps stay in scope (they are
    labelled, and timelines exclude them via analysable()).
    """
    out = []
    for p in posts:
        if p.duplicate_of or p.context_only:
            continue
        if p.timestamp_verified and not p.in_window and not p.active_in_window:
            continue
        out.append(p)
    return out


def run_verification(posts: list[Post], start: datetime, end: datetime,
                     ads_start: datetime | None = None) -> list[Post]:
    posts = dedupe(posts)
    gate_timestamps(posts, start, end, ads_start=ads_start)
    detect_languages(posts)
    return posts
