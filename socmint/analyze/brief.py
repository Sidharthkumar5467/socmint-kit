"""Run summary (brief.json): narrative ranking, key findings and coverage, built by rule.

No LLM is used here, so every number in brief.json can be recomputed from the other
analysis files (narratives.json, clusters.json, origin.json, claims.json,
source_status.json, posts.jsonl). Each finding carries the narrative / cluster / claim /
post IDs behind its sentence (CLAUDE.md §3).

Ranking order (owner decision 2026-11-19: by SEVERITY of the discussion, not size/surge):
  1. off_topic narratives always last (and never main);
  2. severity: critical > high > medium > low > None (not assessed: older runs, keyword
     labels, failed label calls);
  3. within a severity level: narratives pushed by coordinated clusters and narratives
     with a disputed/contradicted claim first (both > one > neither);
  4. then size (posts); 5. surge ratio only as the final tie-break; 6. narrative id.
  Narratives with severity None keep the earlier order (by activity score, below), so
  runs saved before severity existed rank exactly as before.
Main list (lead's decision 2026-11-19): up to MAIN_MAX narratives, never off_topic,
filled bucket by bucket (each bucket in the ranking order above, a narrative counted once):
  a. critical and high narratives with >= MAIN_MIN_POSTS posts;
  b. the top narrative of each main coordinated cluster (>= MAIN_CLUSTER_MIN_ACCOUNTS
     accounts), whatever its severity: coordination is itself a finding;
  c. critical/high narratives with MAIN_MIN_POSTS_SEVERE..MAIN_MIN_POSTS-1 posts, only
     when coordinated or with a disputed/contradicted claim;
  d. medium narratives with >= MAIN_MIN_POSTS posts;
  e. low narratives with >= MAIN_MIN_POSTS posts.
Narratives without a severity (older runs, keyword labels) are filled like bucket e,
after it. The main list is shown in the global ranking order.

Activity score (stored as NarrativeRank.score; orders only the severity-None level):
  log(1 + posts)
  + PLATFORM_BONUS per platform beyond the first
  + surge bonus when the peak day is SURGE_RATIO x the mean of the other window days
  + COORD_BONUS when supporting posts of a coordination cluster are members
  + CLAIM_BONUS_RISK per disputed/contradicted claim (capped), CLAIM_BONUS_SUPPORTED once
  then x RELEVANCE_MULT[relevance] (threat up, off_topic heavily down, None neutral).
Relevance and severity never drop a narrative; they only move it in the ranking.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from pydantic import BaseModel

from socmint.schema import (
    Brief,
    Claim,
    Cluster,
    CoverageNote,
    Finding,
    Narrative,
    NarrativeRank,
    OriginTrace,
    Post,
    RunMeta,
    SourceStatus,
)
from socmint.util.timeutil import parse_ts

# ------------------------------------------------------------------------ constants
MAIN_MAX = 10  # narratives shown in the main list
MAIN_MIN_POSTS = 10  # a main narrative has at least this many posts ...
MAIN_MIN_POSTS_SEVERE = 5  # ... or this many when its severity is in SEVERE_LEVELS
SEVERITY_LEVELS = ("critical", "high", "medium", "low")  # most severe first
SEVERE_LEVELS = ("critical", "high")
PLATFORM_BONUS = 0.5
MULTI_PLATFORM_MIN = 3  # platforms for the multi_platform tag
SURGE_RATIO = 3.0  # peak day >= this x mean of the other days
SURGE_MIN_PEAK = 5  # ... and at least this many posts on the peak day
SURGE_BONUS_MAX = 1.5
# A window day with less than this share inside the exact window bounds (preset runs end
# 'now', so their first and last UTC days are partial) is too short to compare: it is
# left out of surge and is not required for "covered the whole window".
MIN_DAY_FRACTION = 0.25
COORD_BONUS = 1.0
CLAIM_BONUS_RISK = 0.75  # per disputed / contradicted claim
CLAIM_BONUS_RISK_MAX = 1.5
CLAIM_BONUS_SUPPORTED = 0.25
RELEVANCE_MULT: dict[str | None, float] = {
    "threat": 1.5, "related": 1.0, None: 1.0, "off_topic": 0.3}
MAIN_CLUSTER_MIN_ACCOUNTS = 5
MAX_FINDINGS = 5
# Claims are extracted for the preliminary main list (claim_candidates), at least this many
# narratives when the main list is shorter (small runs, e.g. a 1-day window, may have no
# narrative of MAIN_MIN_POSTS posts but still need a misinformation assessment).
CLAIMS_MIN_NARRATIVES = 3
FINDING_POST_IDS_MAX = 50  # cap on post IDs attached to one finding (the rest via ids)

PLATFORM_NAMES = {"x": "X", "youtube": "YouTube", "telegram": "Telegram",
                  "linkedin": "LinkedIn", "linkedin_ads": "LinkedIn Ads"}
SIGNAL_PHRASES = {
    "near_duplicate_text": "near-identical text",
    "near_simultaneous_posting": "near-simultaneous posting",
    "similar_creation_dates": "accounts created close together",
    "reused_image": "reused images",
    "amplification_loop": "amplifying each other",
}
VERDICTS = ("supported", "contradicted", "disputed", "unverified")
# worst first: the state shown for a source with several queries
STATE_SEVERITY = ("failed", "partial", "out_of_coverage", "skipped_no_credentials",
                  "skipped_capability", "completed_zero", "ok")


# -------------------------------------------------------------------------- helpers
def _as(model: type[BaseModel], items: Iterable[Any] | None) -> list[Any]:
    """Accept models or plain dicts (e.g. read from disk)."""
    return [x if isinstance(x, model) else model.model_validate(x) for x in items or []]


def platform_name(source: str) -> str:
    return PLATFORM_NAMES.get(source, source.replace("_", " ").title())


def short_day(day: str | date | None) -> str:
    """'2026-11-16' -> '16 Nov'. Unparseable input is returned unchanged."""
    if day is None:
        return ""
    try:
        d = day if isinstance(day, date) else date.fromisoformat(str(day)[:10])
    except ValueError:
        return str(day)
    return f"{d.day} {d.strftime('%b')}"


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n:,} {word if n == 1 else (plural or word + 's')}"


def _window_days(meta: RunMeta) -> list[date]:
    try:
        start = date.fromisoformat(meta.start[:10])
        end = date.fromisoformat(meta.end[:10])
    except ValueError:
        return []
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def day_fractions(meta: RunMeta) -> dict[str, float]:
    """Share (0..1] of each window day (ISO date) inside the exact window bounds
    (options window_start_at / window_end_at). A window preset ends 'now', so its first
    and last UTC days are partial (a 1-day preset from 09:30 touches 2 UTC days, ~14 h
    and ~10 h). Days missing from the result are whole (custom ranges, older runs)."""
    o = meta.options or {}
    a, b = parse_ts(o.get("window_start_at")), parse_ts(o.get("window_end_at"))
    if a is None or b is None:
        return {}
    out: dict[str, float] = {}
    for d in _window_days(meta):
        d0 = datetime(d.year, d.month, d.day, tzinfo=UTC)
        lo, hi = max(a, d0), min(b, d0 + timedelta(days=1))
        frac = max(0.0, (hi - lo).total_seconds()) / 86400
        if frac < 0.999:  # end bounds like 23:59:59 count as whole
            out[d.isoformat()] = frac
    return out


def window_phrase(meta: RunMeta) -> str | None:
    """'last 24 hours: 2 partial UTC days' for a window with partial days, else None."""
    frac = day_fractions(meta)
    if not frac:
        return None
    n = len(_window_days(meta))
    o = meta.options or {}
    a, b = parse_ts(o.get("window_start_at")), parse_ts(o.get("window_end_at"))
    hours = round((b - a).total_seconds() / 3600) if a and b else 0
    span = ("last 24 hours" if hours == 24 else
            f"last {hours // 24} days" if hours % 24 == 0 and hours else f"{hours} hours")
    if len(frac) == n:
        return f"{span}: {_plural(n, 'partial UTC day')}"
    return f"{span}: {_plural(n, 'UTC day')}, first and last partial"


def _required_days(window: list[date], frac: dict[str, float]) -> list[date]:
    """Window days a source must reach to count as covering the whole window: a day with
    less than MIN_DAY_FRACTION of it inside the window is not required (a 1-hour sliver
    of the first day would otherwise make every source look incomplete)."""
    req = [d for d in window if frac.get(d.isoformat(), 1.0) >= MIN_DAY_FRACTION]
    return req or window


def _partial_note(day: str | date | None, frac: dict[str, float]) -> str:
    """' (partial day, 10 h in the window)' for a partial window day, else ''."""
    f = frac.get(str(day)[:10]) if day else None
    if f is None:
        return ""
    return f" (partial day, {max(1, round(f * 24))} h in the window)"


def peak(volume_by_day: dict[str, int]) -> tuple[str | None, int]:
    """Highest day (earliest on ties) and its count; (None, 0) for empty volume."""
    if not volume_by_day:
        return None, 0
    day = min(volume_by_day, key=lambda d: (-int(volume_by_day[d] or 0), d))
    return day, int(volume_by_day[day] or 0)


def surge_peak(volume_by_day: dict[str, int], window: list[date],
          fractions: dict[str, float] | None = None) -> tuple[str | None, int, float]:
    """(surge day, its post count, ratio): the day with the highest posts-per-day RATE /
    mean rate of the other window days (days without posts count as 0).

    fractions (day_fractions): a partial window day's count is divided by the share of it
    inside the window, so a 10-hour last day is compared per hour with whole days; a day
    with less than MIN_DAY_FRACTION inside the window is left out (too short to compare).
    The denominator is floored at 1 post/day so a narrative that exists on a single day
    is not called surging only because the other days are empty. Ratio 0.0 with fewer
    than 2 comparable days; (None, 0, 0.0) for empty volume."""
    frac = fractions or {}
    days = [d.isoformat() for d in window] or sorted(volume_by_day)
    days = [d for d in days if frac.get(d, 1.0) >= MIN_DAY_FRACTION]
    rate = {d: int(volume_by_day.get(d, 0) or 0) / frac.get(d, 1.0) for d in days}
    if not any(rate.values()):
        return None, 0, 0.0
    day = min(rate, key=lambda d: (-rate[d], d))
    if len(rate) < 2:  # one comparable day (e.g. a 1-day window): nothing to surge over
        return day, int(volume_by_day.get(day, 0) or 0), 0.0
    others = [v for d, v in rate.items() if d != day]
    mean_others = sum(others) / len(others) if others else 0.0
    return day, int(volume_by_day.get(day, 0) or 0), rate[day] / max(mean_others, 1.0)


def surge_ratio(volume_by_day: dict[str, int], window: list[date],
                fractions: dict[str, float] | None = None) -> float:
    """surge_peak()'s ratio only."""
    return surge_peak(volume_by_day, window, fractions)[2]


def covered_days(posts: list[Post], window: list[date]) -> dict[str, set[date]]:
    """source -> window days with at least one verified, in-window, non-context post."""
    days = set(window)
    out: dict[str, set[date]] = {}
    for p in posts:
        if p.context_only or not (p.timestamp_verified and p.in_window):
            continue
        ts = parse_ts(p.published_at)
        if ts is None:
            continue
        d = ts.astimezone(UTC).date()
        if d in days:
            out.setdefault(p.source, set()).add(d)
    return out


def full_sources(cov: dict[str, set[date]], window: list[date]) -> set[str]:
    """Sources with verified posts on every day of `window` (callers pass
    _required_days for windows with partial days)."""
    return {s for s, ds in cov.items() if window and ds >= set(window)}


def _day_counts(n: Narrative, by_id: dict[str, Post], sources: set[str]) -> dict[str, int]:
    """Per-day counts of verified, in-window members from `sources` only."""
    out: Counter[str] = Counter()
    for pid in n.member_post_ids:
        p = by_id.get(pid)
        if p is None or p.source not in sources or not (p.timestamp_verified and p.in_window):
            continue
        ts = parse_ts(p.published_at)
        if ts is not None:
            out[ts.astimezone(UTC).date().isoformat()] += 1
    return dict(out)


def _platform_counts(n: Narrative, by_id: dict[str, Post]) -> dict[str, int]:
    vol = {k: v for k, v in (n.volume_by_platform or {}).items() if v}
    if vol:
        return vol
    # empty volume (all members unverified): count members by source instead
    return dict(Counter(by_id[p].source for p in n.member_post_ids if p in by_id))


def _member_source_counts(n: Narrative, by_id: dict[str, Post]) -> dict[str, int]:
    """All members by source (verified or not), so the counts add up to the post total."""
    return dict(Counter(by_id[p].source for p in n.member_post_ids if p in by_id))


def _platform_text(counts: dict[str, int]) -> str:
    return ", ".join(f"{platform_name(s)} {c}" for s, c in
                     sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


# -------------------------------------------------------------------------- ranking
def verified_count(n: Narrative) -> int:
    """Members with a verified, in-window timestamp: the ones counted by day and platform
    (CLAUDE.md §3: unverified posts are excluded from counts)."""
    return max(0, len(n.member_post_ids) - int(n.unverified_member_count or 0))


def volume_phrase(n: Narrative) -> str:
    """'12 posts' when every member is counted, else '3 verified posts of 12 members':
    a volume is always the verified count, and a member total is labelled as such."""
    size, verified = len(n.member_post_ids), verified_count(n)
    if verified == size:
        return _plural(size, "post")
    return f"{_plural(verified, 'verified post')} of {_plural(size, 'member')}"


def _main_bucket(nr: NarrativeRank, size: int, cluster_narrs: set[str],
                 verified: int | None = None) -> int | None:
    """Main-list bucket (0 = a ... 4 = e, 5 = not rated), None = never main.

    A narrative with no timestamp-verified member is main only when rated medium or
    above (live re-run 2026-11-20: a low vendor ad narrative with 0 verified posts
    entered the main list)."""
    if "off_topic" in nr.tags:
        return None
    sev = nr.severity
    if verified == 0 and sev not in ("critical", "high", "medium"):
        return None
    risky = bool(nr.cluster_ids) or any(t in nr.tags for t in
                                        ("disputed_claim", "contradicted_claim"))
    if sev in SEVERE_LEVELS and size >= MAIN_MIN_POSTS:
        return 0
    if nr.narrative_id in cluster_narrs:
        return 1
    if sev in SEVERE_LEVELS and size >= MAIN_MIN_POSTS_SEVERE and risky:
        return 2
    if size >= MAIN_MIN_POSTS:
        return {"medium": 3, "low": 4}.get(sev or "", 5)
    return None


def rank_narratives(narratives: list[Narrative], clusters: list[Cluster],
                    claims: list[Claim], posts: list[Post],
                    meta: RunMeta) -> list[NarrativeRank]:
    """Score, tag and rank every narrative (rank 1 = most important)."""
    by_id = {p.post_id: p for p in posts}
    window = _window_days(meta)
    frac = day_fractions(meta)
    full = full_sources(covered_days(posts, window), _required_days(window, frac))
    cluster_posts = {c.id: set(c.supporting_post_ids) for c in clusters}
    cluster_size = {c.id: len(c.member_author_ids) for c in clusters}
    claims_by_n: dict[str, list[Claim]] = {}
    for c in claims:
        if c.narrative_id:
            claims_by_n.setdefault(c.narrative_id, []).append(c)

    scored: list[tuple[tuple, int, int, NarrativeRank]] = []
    for n in narratives:
        size = len(n.member_post_ids)
        relevance = getattr(n, "relevance", None)
        tags: list[str] = []
        reasons: list[str] = [volume_phrase(n)]
        score = math.log1p(size)

        plats = _platform_counts(n, by_id)
        if len(plats) > 1:
            score += PLATFORM_BONUS * (len(plats) - 1)
            reasons.append(f"{len(plats)} platforms")
        if len(plats) >= MULTI_PLATFORM_MIN:
            tags.append("multi_platform")

        pday, pcount = peak(n.volume_by_day or {})
        # Surge is judged only on sources that reached every window day, so a source that
        # only covered the newest day cannot fake a spike. Without any such source (or
        # without verified volume per post) the narrative's own volume is used.
        restricted = bool(full) and bool(by_id)
        svol = _day_counts(n, by_id, full) if restricted else (n.volume_by_day or {})
        sday, scount, ratio = surge_peak(svol, window, frac)
        if scount >= SURGE_MIN_PEAK and ratio >= SURGE_RATIO:
            score += min(math.log2(ratio) * 0.5, SURGE_BONUS_MAX)
            tags.append("surging")
            reasons.append(f"surge on {short_day(sday)} ({ratio:.1f}x other days"
                           + (", per hour" if frac else "")
                           + (", fully covered sources" if restricted else "") + ")")

        members = set(n.member_post_ids)
        overlaps = [(len(members & cp), cid) for cid, cp in cluster_posts.items()
                    if members & cp]
        cluster_ids = [cid for _, cid in sorted(overlaps, key=lambda t: (-t[0], t[1]))]
        if cluster_ids:
            big = any(cluster_size[c] >= MAIN_CLUSTER_MIN_ACCOUNTS for c in cluster_ids)
            score += COORD_BONUS if big else COORD_BONUS / 2
            tags.append("coordinated")
            reasons.append(f"coordinated group{'s' if len(cluster_ids) > 1 else ''} "
                           + ", ".join(cluster_ids))

        ncl = claims_by_n.get(n.id, [])
        vc = Counter(c.verdict for c in ncl)
        risk = vc["disputed"] + vc["contradicted"]
        if risk:
            score += min(CLAIM_BONUS_RISK * risk, CLAIM_BONUS_RISK_MAX)
        if vc["contradicted"]:
            tags.append("contradicted_claim")
            reasons.append(_plural(vc["contradicted"], "contradicted claim"))
        if vc["disputed"]:
            tags.append("disputed_claim")
            reasons.append(_plural(vc["disputed"], "disputed claim"))
        if vc["supported"]:
            score += CLAIM_BONUS_SUPPORTED
            tags.append("supported_claim")
            reasons.append(_plural(vc["supported"], "supported claim"))

        score *= RELEVANCE_MULT.get(relevance, 1.0)
        if relevance == "threat":
            tags.insert(0, "threat")
            reasons.insert(0, "threat-relevant")
        elif relevance == "off_topic":
            tags.insert(0, "off_topic")
            reasons.insert(0, "off-topic (ranked down)")

        severity = getattr(n, "severity", None)
        if relevance == "off_topic" or severity not in SEVERITY_LEVELS:
            severity = None
        if severity:
            tags.insert(0, f"severity:{severity}")
            reasons.insert(0, f"{severity} severity")

        nr = NarrativeRank(narrative_id=n.id, rank=0, score=round(score, 3), main=False,
                           tags=tags, reasons=reasons, cluster_ids=cluster_ids,
                           claim_ids=[c.id for c in ncl], peak_day=pday, peak_count=pcount,
                           severity=severity)
        pushed = int(bool(cluster_ids)) + int(bool(risk))
        surge = ratio if "surging" in tags else 0.0
        if severity:
            within = (-pushed, -size, -surge)
        else:  # not assessed: the earlier activity-score order
            within = (-score, -size, 0.0)
        key = (int(relevance == "off_topic"),
               SEVERITY_LEVELS.index(severity) if severity else len(SEVERITY_LEVELS),
               *within, n.id)
        scored.append((key, size, verified_count(n), nr))

    scored.sort(key=lambda t: t[0])
    # the top narrative of every main coordinated cluster (bucket b)
    big = [c for c in clusters if len(c.member_author_ids) >= MAIN_CLUSTER_MIN_ACCOUNTS]
    cluster_narrs = {ids[0] for ids in map_cluster_narratives(big, narratives).values() if ids}
    buckets: dict[int, list[NarrativeRank]] = {}
    for _, size, verified, nr in scored:
        b = _main_bucket(nr, size, cluster_narrs, verified)
        if b is not None:
            buckets.setdefault(b, []).append(nr)
    n_main = 0
    for b in sorted(buckets):
        for nr in buckets[b]:
            if n_main >= MAIN_MAX:
                break
            nr.main = True
            n_main += 1
    # Main narratives take ranks 1..n_main, in the global ranking order, so the numbers an
    # analyst sees in the main list have no gaps; the rest follow in ranking order (a
    # 4-post critical narrative ranks above a 40-post low one, but is not shown as main).
    out = [nr for *_, nr in scored if nr.main] + [nr for *_, nr in scored if not nr.main]
    for i, nr in enumerate(out, start=1):
        nr.rank = i
    return out


def claim_candidates(posts: list[Post], narratives: list[Narrative],
                     clusters: list[Cluster], meta: RunMeta | dict,
                     max_narratives: int) -> tuple[list[str], int]:
    """Which narratives get claim extraction (owner decision 2026-11-20: LLM savings).

    A preliminary ranking with the same rules as the brief but WITHOUT claim inputs
    (claims do not exist yet: severity, coordination, size and surge decide). Returns
    (order, n): narrative IDs in this order (off-topic narratives left out):
      1. critical/high narratives with >= MAIN_MIN_POSTS_SEVERE verified posts, main or
         not (review 2026-11-20: a 5-post "pull your money out" narrative must not lose
         its claim slot to a 40-post low one; claims are the misinformation finding);
      2. the rest of the preliminary main list;
      3. everything else;
    each part in the ranking order. n = how many to assess: min(max_narratives,
    max(size of parts 1+2, CLAIMS_MIN_NARRATIVES)). The caller takes the first n of
    `order` that can carry a claim."""
    narratives = _as(Narrative, narratives)
    meta = meta if isinstance(meta, RunMeta) else RunMeta.model_validate(meta)
    ranks = rank_narratives(narratives, _as(Cluster, clusters), [], _as(Post, posts), meta)
    vc = {n.id: verified_count(n) for n in narratives}
    ranks = [r for r in ranks if "off_topic" not in r.tags]
    ranks.sort(key=lambda r: (r.main is False, r.rank))  # ranking order (main first)
    severe = [r.narrative_id for r in ranks if r.severity in SEVERE_LEVELS
              and vc.get(r.narrative_id, 0) >= MAIN_MIN_POSTS_SEVERE]
    first = set(severe)
    main = [r.narrative_id for r in ranks if r.main and r.narrative_id not in first]
    rest = [r.narrative_id for r in ranks if not r.main and r.narrative_id not in first]
    n = min(int(max_narratives), max(len(severe) + len(main), CLAIMS_MIN_NARRATIVES))
    return severe + main + rest, max(0, n)


# ------------------------------------------------------------------------- clusters
def _cluster_strength(c: Cluster) -> tuple[int, int, float]:
    types = {s.type for s in c.signals}
    mean = sum(s.score for s in c.signals) / len(c.signals) if c.signals else 0.0
    return len(types), len(c.member_author_ids), mean


def rank_clusters(clusters: list[Cluster]) -> list[str]:
    """IDs of clusters with >= MAIN_CLUSTER_MIN_ACCOUNTS accounts, strongest first."""
    big = [c for c in clusters if len(c.member_author_ids) >= MAIN_CLUSTER_MIN_ACCOUNTS]
    return rank_clusters_all(big)


def map_cluster_narratives(clusters: list[Cluster],
                           narratives: list[Narrative]) -> dict[str, list[str]]:
    """cluster id -> narrative ids containing its supporting posts, largest overlap first."""
    out: dict[str, list[str]] = {}
    for c in clusters:
        sup = set(c.supporting_post_ids)
        hits = [(len(sup & set(n.member_post_ids)), n.id) for n in narratives]
        out[c.id] = [nid for k, nid in sorted(hits, key=lambda t: (-t[0], t[1])) if k]
    return out


# ------------------------------------------------------------------------- coverage
def _worst_state(states: list[str]) -> str:
    known = [s for s in states if s in STATE_SEVERITY]
    if not known:
        return states[0] if states else "unknown"
    return min(known, key=STATE_SEVERITY.index)


def coverage_notes(posts: list[Post], statuses: list[SourceStatus],
                   meta: RunMeta) -> list[CoverageNote]:
    """One note per source: on how many window days it has verified posts, and why not."""
    window = _window_days(meta)
    total = len(window)
    frac = day_fractions(meta)
    required = set(_required_days(window, frac))
    phrase = window_phrase(meta)
    cov = covered_days(posts, window)
    by_source: dict[str, list[SourceStatus]] = {}
    for st in statuses:
        by_source.setdefault(st.source, []).append(st)
    for s in meta.sources:
        by_source.setdefault(s, [])

    notes: list[CoverageNote] = []
    for source, sts in by_source.items():
        states = [st.state.value if hasattr(st.state, "value") else str(st.state)
                  for st in sts]
        state = _worst_state(states) if states else "unknown"
        own = [p for p in posts if p.source == source and not p.context_only]
        seen = cov.get(source, set())
        dated = sum(1 for p in own if p.timestamp_verified and p.in_window
                    and parse_ts(p.published_at) is not None)
        days_with = len(seen)

        if not sts:
            note = "no status recorded"
        elif state == "skipped_no_credentials":
            note = "skipped: no credentials"
        elif state == "skipped_capability":
            note = "skipped: not supported for this search"
        elif state == "failed" and not own:
            note = "failed; see source details"
        elif state == "out_of_coverage" and not own:
            note = "outside this source's coverage"
        elif not own:
            note = "searched; no posts found"
        elif all(p.kind == "ad" for p in own):
            note = f"ads: {dated} of {len(own)} with a verified date in the window"
            ads_from = str((meta.options or {}).get("ads_window_start_at") or "")[:10]
            if (meta.options or {}).get("ads_all_time"):
                note += "; ads searched over all time"
            elif ads_from:  # pipeline.ads_window_start: a longer ads search span
                note += f"; ads searched from {ads_from}"
        elif total and days_with == total:
            note = (f"whole window covered ({phrase})" if phrase
                    else f"all {total} days covered")
        elif total and phrase and seen >= required:
            note = (f"whole window covered ({phrase}; no posts in the "
                    + ", ".join(short_day(d) for d in window if d not in seen)
                    + " sliver)")
        elif days_with == 0:
            note = "no posts with a verified date in the window"
        else:
            newest = window[-days_with:] if total else []
            suffix = bool(newest) and seen == set(newest)
            if suffix and state == "partial":
                note = ("newest day only; older days not reached" if days_with == 1 else
                        f"newest {days_with} days only; older days not reached")
            else:
                missing = [d for d in window if d not in seen]
                note = (f"posts on {days_with} of {total} "
                        + ("UTC days" if phrase else "days") + " (none on "
                        + ", ".join(short_day(d) + (" (partial)" if d.isoformat() in frac
                                                    else "") for d in missing[:6])
                        + (f" +{len(missing) - 6} more" if len(missing) > 6 else "") + ")")
        notes.append(CoverageNote(source=source, state=state, days_with_posts=days_with,
                                  days_total=total, note=note))
    return notes


# ------------------------------------------------------------------------- findings
def _claim_notability(c: Claim) -> tuple:
    order = {"contradicted": 0, "disputed": 1, "supported": 2, "unverified": 3}
    asserts = int((c.stances or {}).get("assert", 0) or 0)
    return order.get(c.verdict, 9), -asserts, -c.confidence, c.id


def _peak_coverage(n: Narrative, nr: NarrativeRank, by_id: dict[str, Post],
                   cov: dict[str, set[date]], full: set[str]
                   ) -> tuple[list[str], bool, str | None, int] | None:
    """When the peak day is inflated by sources that did not cover the whole window:
    (those sources, True if the peak day is the ONLY day they covered, peak day and count
    on the fully covered sources). None when the peak is comparable across days."""
    if not (nr.peak_day and full):
        return None
    try:
        pd = date.fromisoformat(nr.peak_day[:10])
    except ValueError:
        return None
    plats = _platform_counts(n, by_id)
    part = [s for s in plats if s not in full and pd in cov.get(s, set())]
    if not part:
        return None
    only = all(cov.get(s, set()) == {pd} for s in part)
    fday, fcount = peak(_day_counts(n, by_id, full))
    return part, only, fday, fcount


def _peak_caveat(n: Narrative, nr: NarrativeRank, by_id: dict[str, Post],
                 cov: dict[str, set[date]], full: set[str]) -> list[str]:
    """Sentences warning that the peak day is inflated by sources that did not cover the
    whole window, plus the peak on the fully covered sources when that differs."""
    pc = _peak_coverage(n, nr, by_id, cov, full)
    if pc is None:
        return []
    part, _only, fday, fcount = pc
    out = [" and ".join(platform_name(s) for s in part)
           + f" did not cover every day, so {short_day(nr.peak_day)} counts are not "
           "comparable across days."]
    names = ", ".join(platform_name(s) for s in sorted(full))
    if fday:
        out.append(f"On the fully covered sources ({names}) it peaked on {short_day(fday)} "
                   f"({_plural(fcount, 'post')}).")
    else:
        out.append(f"None of its posts came from the fully covered sources ({names}).")
    return out


def _peak_headline(n: Narrative, nr: NarrativeRank, by_id: dict[str, Post],
                   cov: dict[str, set[date]], full: set[str],
                   frac: dict[str, float] | None = None) -> tuple[str, bool]:
    """(headline clause about the peak, True when it already carries the coverage caveat).

    Many analysts read only the headline, so a peak caused by collection coverage is
    named as such there, with the peak on the fully covered sources beside it."""
    if not nr.peak_day:
        return "", False
    part_txt = _partial_note(nr.peak_day, frac or {})
    raw = (f"{short_day(nr.peak_day)} ({_plural(nr.peak_count, 'post')}"
           + (f"; {part_txt.strip()[1:-1]}" if part_txt else "") + ")")
    pc = _peak_coverage(n, nr, by_id, cov, full)
    if pc is None:
        return f", peaking on {raw}", False
    part, only, fday, fcount = pc
    names = " and ".join(platform_name(s) for s in part)
    verb = "was" if len(part) == 1 else "were"
    why = (f"the only day {names} {verb} collected" if only
           else f"but {names} did not cover every day")
    full_txt = (f"peak on fully covered sources: {short_day(fday)}, {_plural(fcount, 'post')}"
                if fday else "none on the fully covered sources")
    return f"; most on {raw}, {why} ({full_txt})", True


def _unverified_note(n: Narrative) -> list[str]:
    if not n.unverified_member_count:
        return []
    return [f"{n.unverified_member_count} member post(s) without a verified "
            "timestamp are not counted by day or platform."]


def _sentence(text: str) -> str:
    t = (text or "").strip()
    return t + "." if t and t[-1] not in ".!?…" else t


def _severity_why(n: Narrative, sev: str) -> list[str]:
    """'Rated critical by the labeller: <reason>.' (the reason may be empty)."""
    reason = _sentence(getattr(n, "severity_reason", "") or "")
    return [f"Rated {sev} by the labeller" + (f": {reason}" if reason else ".")]


def _finding_narrative(fid: str, n: Narrative, nr: NarrativeRank, by_id: dict[str, Post],
                       claims: dict[str, Claim], trace: OriginTrace | None,
                       lead: str, cov: dict[str, set[date]], full: set[str],
                       frac: dict[str, float] | None = None) -> Finding:
    size = len(n.member_post_ids)
    plats = _platform_counts(n, by_id)
    sev = nr.severity
    if sev:
        # posts per platform as counted, no "on K platforms" emphasis: a severe narrative
        # can hold a few unrelated members, and the count must not imply they all are
        # the severe activity
        srcs = _member_source_counts(n, by_id)
        head = (f"Most severe narrative “{n.label}” ({sev} severity): "
                + (_plural(size, "post") if verified_count(n) == size
                   else _plural(size, "member post"))
                + (f" ({_platform_text(srcs)})" if srcs else "")
                + (f", {verified_count(n)} with a verified in-window date"
                   if verified_count(n) != size else ""))
    else:
        head = (f"{lead}“{n.label}”: {volume_phrase(n)} on "
                f"{_plural(len(plats), 'platform')}")
    peak_txt, in_head = _peak_headline(n, nr, by_id, cov, full, frac)
    head += peak_txt + "."
    detail = _severity_why(n, sev) if sev else []
    if plats and not (sev and plats == _member_source_counts(n, by_id)):
        detail.append(("Verified, in-window posts by platform: " if sev else "By platform: ")
                      + f"{_platform_text(plats)}.")
    # the headline already gives the fully covered peak; keep the comparability note
    detail += _peak_caveat(n, nr, by_id, cov, full)[:1 if in_head else None]
    ncl = [claims[c] for c in nr.claim_ids if c in claims]
    if ncl:
        vc = Counter(c.verdict for c in ncl)
        detail.append("Claims: " + ", ".join(f"{vc[v]} {v}" for v in VERDICTS if vc[v]) + ".")
    detail += _unverified_note(n)
    lead_ids = trace.earliest_post_ids[:1] if trace else []
    if trace and trace.earliest_post_ids:
        strong = by_id.get(trace.first_strong_post_id or "")
        if trace.earliest_match == "weak" and strong is not None:
            # The earliest post only loosely matches: cite the first close match instead,
            # and say so, so the date is not read as the narrative's start.
            detail.append("First observed in collected data: "
                          f"{short_day(n.first_seen_at) or 'see origin'}, but that post only "
                          "loosely matches the narrative; the first close match is from "
                          f"{short_day(strong.published_at) or 'an unknown day'} "
                          f"({strong.post_id}).")
            lead_ids = [strong.post_id]
        elif trace.earliest_match == "weak":
            detail.append("First observed in collected data: "
                          f"{short_day(n.first_seen_at) or 'see origin'} (a loose match; "
                          "no member post closely matches the narrative).")
        else:
            detail.append("First observed in collected data: "
                          f"{short_day(n.first_seen_at) or 'see origin'}.")
    post_ids = list(dict.fromkeys(lead_ids + list(n.example_post_ids)))
    return Finding(id=fid, kind="narrative", headline=head, detail=" ".join(detail),
                   narrative_ids=[n.id], cluster_ids=list(nr.cluster_ids),
                   claim_ids=list(nr.claim_ids), post_ids=post_ids[:FINDING_POST_IDS_MAX])


def _finding_threat(fid: str, n: Narrative, nr: NarrativeRank, by_id: dict[str, Post],
                    cov: dict[str, set[date]] | None = None,
                    full: set[str] | None = None, lead_severity: str | None = None,
                    frac: dict[str, float] | None = None) -> Finding:
    """Second narrative finding: the next critical/high narrative (severity-rated runs) or,
    for runs without severity, the first threat-relevant one. "Also <level>" only when the
    lead finding has the same level; otherwise "Next most severe (<level>)"."""
    plats = _platform_counts(n, by_id)
    sev = nr.severity
    if sev:
        lead_in = (f"Also {sev} severity" if sev == lead_severity
                   else f"Next most severe ({sev})")
        head = f"{lead_in}: narrative “{n.label}”: {volume_phrase(n)}"
    else:
        head = f"Threat-relevant narrative “{n.label}”: {volume_phrase(n)}"
    if plats:
        head += f" ({_platform_text(plats)})"
    if nr.peak_day:
        head += (f"; {nr.peak_count} of them on {short_day(nr.peak_day)}"
                 + _partial_note(nr.peak_day, frac or {}))
    head += "."
    if sev:
        detail = _severity_why(n, sev)
    else:
        reason = _sentence(getattr(n, "relevance_reason", "") or "")
        detail = [f"Why threat-relevant (labeller): {reason}"] if reason else []
    detail += _peak_caveat(n, nr, by_id, cov or {}, full or set())
    detail += _unverified_note(n)
    return Finding(id=fid, kind="narrative", headline=head, detail=" ".join(detail),
                   narrative_ids=[n.id], cluster_ids=list(nr.cluster_ids),
                   claim_ids=list(nr.claim_ids),
                   post_ids=list(n.example_post_ids)[:FINDING_POST_IDS_MAX])


def _finding_cluster(fid: str, c: Cluster, n_big: int, narr_ids: list[str],
                     narr_by_id: dict[str, Narrative], siblings: list[Cluster]) -> Finding:
    """`siblings`: other main clusters whose top narrative is the same as c's."""
    accounts = len(c.member_author_ids)
    plats = ", ".join(platform_name(s) for s in c.sources) or "unknown platform"
    types = list(dict.fromkeys(s.type for s in c.signals))
    sig_text = ", ".join(SIGNAL_PHRASES.get(t, t) for t in types)
    head = (f"Strongest coordinated group {c.id}: {accounts} {plats} accounts with "
            f"{_plural(len(types), 'signal type')} ({sig_text}) over "
            f"{_plural(len(c.supporting_post_ids), 'supporting post')}")
    sup = set(c.supporting_post_ids)
    if narr_ids:
        top = narr_by_id[narr_ids[0]]
        k = len(sup & set(top.member_post_ids))
        head += f"; {k} of those posts are in narrative “{top.label}”"
    head += "."
    detail = []
    summary = (getattr(c, "summary", "") or "").strip()
    if summary:
        detail.append(summary)
    if siblings and narr_ids:
        more = sum(len(x.member_author_ids) for x in siblings)
        detail.append(f"Group{'s' if len(siblings) > 1 else ''} "
                      + ", ".join(x.id for x in siblings)
                      + f" ({_plural(more, 'more account')}) also post mainly into "
                      f"“{narr_by_id[narr_ids[0]].label}”.")
    if n_big > 1:
        detail.append(f"{n_big} groups have {MAIN_CLUSTER_MIN_ACCOUNTS}+ accounts.")
    detail.append("Coordination is not proof of inauthentic behaviour; check the benign "
                  "explanations.")
    return Finding(id=fid, kind="coordination", headline=head, detail=" ".join(detail),
                   narrative_ids=narr_ids[:3], cluster_ids=[c.id, *(x.id for x in siblings)],
                   post_ids=list(c.supporting_post_ids)[:FINDING_POST_IDS_MAX])


def _finding_claims(fid: str, claims: list[Claim], vc: dict[str, int]) -> Finding:
    head = (f"{_plural(len(claims), 'claim')} checked: "
            + ", ".join(f"{vc.get(v, 0)} {v}" for v in VERDICTS) + ".")
    notable = sorted(claims, key=_claim_notability)[0]
    post_ids: list[str] = []
    detail = ""
    if notable.verdict in ("disputed", "contradicted"):
        detail = (f"Most notable: “{notable.text}” is {notable.verdict} "
                  f"(confidence {notable.confidence:.2f}).")
        bl = (getattr(notable, "bottom_line", "") or "").strip()
        if bl:
            detail += f" {bl}"
        post_ids = list((notable.stance_post_ids or {}).get("assert") or [])
        post_ids += [p for p in notable.related_post_ids if p not in post_ids]
    elif claims:
        # Since "disputed" needs a contradicting item or a stated conflict, a run can end
        # with none: say so instead of leaving the finding without a detail line.
        detail = "No retrieved source contradicts a claim, and none is disputed."
    ordered = [notable.id] + [c.id for c in claims if c.id != notable.id]
    return Finding(id=fid, kind="claims", headline=head, detail=detail,
                   narrative_ids=[notable.narrative_id] if notable.narrative_id else [],
                   claim_ids=ordered, post_ids=post_ids[:FINDING_POST_IDS_MAX])


def _finding_coverage(fid: str, notes: list[CoverageNote]) -> Finding | None:
    def whole(c: CoverageNote) -> bool:  # incl. a window whose only gap is a short sliver
        return bool(c.days_total) and (c.days_with_posts == c.days_total
                                       or c.note.startswith("whole window covered"))
    gaps = [c for c in notes if c.days_total and not whole(c)
            and not c.note.startswith("ads:")]
    full = [c for c in notes if whole(c)]
    if not gaps:
        return None
    parts = []
    for c in gaps:
        if c.days_with_posts:
            parts.append(f"{platform_name(c.source)} {c.days_with_posts} of {c.days_total} "
                         f"days ({c.note})")
        else:
            parts.append(f"{platform_name(c.source)} none ({c.note})")
    head = "Not every source covered the whole window: " + "; ".join(parts) + "."
    detail = ""
    if full:
        detail = ("Full coverage: " + ", ".join(platform_name(c.source) for c in full)
                  + ". Days a source did not reach are not evidence of silence.")
    return Finding(id=fid, kind="coverage", headline=head, detail=detail)


# ---------------------------------------------------------------------------- build
def build_brief(posts: list[Post], narratives: list[Narrative], clusters: list[Cluster],
                traces: list[OriginTrace], claims: list[Claim],
                statuses: list[SourceStatus], meta: RunMeta | dict) -> Brief:
    """Rule-based run summary. Accepts models or plain dicts for every input."""
    posts = _as(Post, posts)
    narratives = _as(Narrative, narratives)
    clusters = _as(Cluster, clusters)
    traces = _as(OriginTrace, traces)
    claims = _as(Claim, claims)
    statuses = _as(SourceStatus, statuses)
    meta = meta if isinstance(meta, RunMeta) else RunMeta.model_validate(meta)

    by_id = {p.post_id: p for p in posts}
    narr_by_id = {n.id: n for n in narratives}
    claim_by_id = {c.id: c for c in claims}
    trace_by_n = {t.narrative_id: t for t in traces}

    ranks = rank_narratives(narratives, clusters, claims, posts, meta)
    main_clusters = rank_clusters(clusters)
    cl_narr = map_cluster_narratives(clusters, narratives)
    coverage = coverage_notes(posts, statuses, meta)
    window = _window_days(meta)
    frac = day_fractions(meta)
    cov_days = covered_days(posts, window)
    full = full_sources(cov_days, _required_days(window, frac))
    vc = Counter(c.verdict for c in claims)
    verdict_counts = {v: vc.get(v, 0) for v in VERDICTS}

    findings: list[Finding] = []

    def fid() -> str:
        return f"F{len(findings) + 1}"

    lead = next((r for r in ranks if r.main), ranks[0] if ranks else None)
    if lead:
        findings.append(_finding_narrative(
            fid(), narr_by_id[lead.narrative_id], lead, by_id, claim_by_id,
            trace_by_n.get(lead.narrative_id), "Top-ranked narrative ", cov_days, full, frac))
    not_lead = [r for r in ranks if lead is None or r.narrative_id != lead.narrative_id]
    if any(r.severity for r in ranks):
        # severity-rated run: the next critical/high narrative, if any
        threat = next((r for r in not_lead if r.severity in SEVERE_LEVELS), None)
    else:  # older run: the first threat-relevant narrative
        threat = next((r for r in not_lead if "threat" in r.tags), None)
    if threat:
        findings.append(_finding_threat(fid(), narr_by_id[threat.narrative_id], threat,
                                        by_id, cov_days, full,
                                        lead.severity if lead else None, frac))
    strongest = main_clusters[0] if main_clusters else (
        rank_clusters_all(clusters)[0] if clusters else None)
    if strongest:
        c = next(x for x in clusters if x.id == strongest)
        top_n = (cl_narr.get(c.id) or [None])[0]
        siblings = [x for x in clusters if x.id in main_clusters and x.id != c.id
                    and top_n is not None and (cl_narr.get(x.id) or [None])[0] == top_n]
        findings.append(_finding_cluster(fid(), c, len(main_clusters), cl_narr.get(c.id, []),
                                         narr_by_id, siblings))
    if claims:
        findings.append(_finding_claims(fid(), claims, verdict_counts))
    cov = _finding_coverage(fid(), coverage)
    if cov:
        findings.append(cov)

    return Brief(narrative_ranks=ranks, main_cluster_ids=main_clusters,
                 cluster_narratives=cl_narr, findings=findings[:MAX_FINDINGS],
                 coverage=coverage, verdict_counts=verdict_counts)


def rank_clusters_all(clusters: list[Cluster]) -> list[str]:
    """All cluster IDs, strongest first (used when no cluster reaches the main size)."""
    return [c.id for c in sorted(clusters, key=lambda c: (
        *(-x for x in _cluster_strength(c)), c.id))]


__all__ = ["build_brief", "claim_candidates", "rank_narratives", "rank_clusters", "map_cluster_narratives",
           "coverage_notes", "peak", "surge_ratio", "short_day", "platform_name"]

