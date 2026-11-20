"""YouTube collector — official YouTube Data API v3 over plain httpx.

Flow for one topic:
  1. search.list   (type=video, q=search_topic(query), order=relevance) — one paginated pass per
                   time slice (window.time_slices, newest first), publishedAfter/Before =
                   slice bounds; the video cap is shared across slices (window.SliceShare).
  2. videos.list   (≤50 ids per call) — exact publishedAt, statistics, tags, description.
  3. channels.list (≤50 ids per call) — channel creation date, handle, avatar.
  4. commentThreads.list (optional) — top-level comments of the N most-viewed videos.

Evidence rules encoded here:
  * `published_at` is the platform's `snippet.publishedAt` (videos.list first, search.list
    snippet as a fallback). `collected_at` is our fetch time and is never substituted.
  * Links in a video description are DIRECT relations (the uploader wrote them).
  * Comments are `reply` relations to their video (direct, from the API).
  * Comments outside the window are kept; `verify.gate_timestamps` sets `in_window`.
  * Every stop condition (quota ceiling, quotaExceeded, deadline, HTTP error) is reported
    in the SourceStatus with a reason and the last search cursor — an early stop never
    looks like "no posts".

Response shapes follow developers.google.com/youtube/v3/docs (search, videos, channels,
commentThreads, comments resources); they have not yet been confirmed against a live
response in this repo — see the UNVERIFIED markers.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from socmint.collect.window import (
    DEFAULT_MAX_SLICES,
    SliceShare,
    mentions_topic,
    search_topic,
    time_slices,
)
from socmint.interfaces import Capabilities, CollectContext, CollectResult, Query, skipped
from socmint.schema import Media, Metrics, Post, Relation, SourceState, SourceStatus
from socmint.util.http import HttpFailure, make_client, request_json
from socmint.util.log import get_logger
from socmint.util.timeutil import iso, now_iso, parse_ts

log = get_logger(__name__)

API = "https://www.googleapis.com/youtube/v3"

# Quota costs in units.
# UNVERIFIED: the long-standing documented cost of search.list is 100 units. As of
# 2026-11 the determine_quota_cost page says "1 unit, from a separate Search Queries
# bucket limited to 100 calls/day". We keep 100 (conservative: it caps us at
# ceiling/100 search pages, which also stays inside a 100-calls/day bucket).
COST_SEARCH = 100
COST_VIDEOS = 1
COST_CHANNELS = 1
COST_COMMENTS = 1

# Raised from 2000 with relevance ordering (owner request 2026-11-20: results were too few):
# a 30-day run can use ~50 search pages. Google's default project quota is 10,000/day.
DEFAULT_QUOTA_LIMIT = 6000
# search.list order. Live check 2026-11-20 ("sahibinden", 28-30 Oct 2026): order=date
# returned 30 videos and no nextPageToken (a 1-day or 12-hour slice returned the same
# videos, so narrower slices do not help); order=relevance returned 211 unique videos over
# 5 pages, including all 30. Page 1: 45/50 had the topic in title, description or tags,
# so results are filtered client-side (window.mentions_topic) after videos.list.
SEARCH_ORDER = "relevance"
DEFAULT_COMMENT_VIDEOS = 5

# Error reasons that mean "no more quota" (403). quotaExceeded is documented on
# developers.google.com/youtube/v3/docs/errors; the other two are standard Google API
# reasons. UNVERIFIED: dailyLimitExceeded / rateLimitExceeded for YouTube specifically.
QUOTA_REASONS = ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded")

# ')' is allowed inside a URL (e.g. Wikipedia's ".../Flood_(disambiguation)"); a trailing
# ')' is only stripped when it is unbalanced, as URL linkifiers do (see _trim_url).
_URL_RE = re.compile(r"https?://[^\s<>\"'\]]+", re.IGNORECASE)
_TRAILING = ".,;:!?…»”’'\""


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _int(v: Any) -> int | None:
    """videos.list statistics are documented as strings ("1234"); hidden => absent."""
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def best_thumbnail(thumbs: dict | None, prefer: tuple[str, ...] = ("high", "medium",
                                                                   "default")) -> str | None:
    if not isinstance(thumbs, dict):
        return None
    for k in prefer + ("standard", "maxres"):
        t = thumbs.get(k)
        if isinstance(t, dict) and t.get("url"):
            return t["url"]
    return None


def youtube_video_id(url: str) -> str | None:
    """Video id from a youtube.com/watch, youtu.be, /shorts/ or /live/ URL."""
    try:
        u = urlparse(url)
    except ValueError:
        return None
    host = (u.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        vid = u.path.strip("/").split("/")[0]
        return vid or None
    if host in ("youtube.com", "music.youtube.com"):
        if u.path == "/watch":
            return (parse_qs(u.query).get("v") or [None])[0]
        parts = u.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in ("shorts", "live", "embed"):
            return parts[1] or None
    return None


def _trim_url(u: str) -> str:
    """Strip trailing punctuation and an unbalanced trailing ')' (repeat until stable).

    "(see https://x/y)." -> "https://x/y", but "https://w/Flood_(disambiguation)" is kept:
    a wrong target_url would break cross-platform URL matching in origin/spread.
    """
    while True:
        before = u
        u = u.rstrip(_TRAILING)
        if u.endswith(")") and u.count(")") > u.count("("):
            u = u[:-1]
        if u == before:
            return u


def extract_urls(text: str) -> list[str]:
    """URLs written in free text, de-duplicated in order, trailing punctuation stripped."""
    out: list[str] = []
    for m in _URL_RE.finditer(text or ""):
        u = _trim_url(m.group(0))
        if u and u not in out:
            out.append(u)
    return out


def description_relations(description: str, self_id: str | None = None) -> list[Relation]:
    """Direct `link` relations for URLs the uploader put in the description.

    A link to another YouTube video also gets `target_post_id` so origin tracing can draw
    a direct edge when that video is in the collected set.
    """
    rels: list[Relation] = []
    for u in extract_urls(description):
        vid = youtube_video_id(u)
        if vid and vid == self_id:
            continue
        rels.append(Relation(type="link", target_url=u,
                             target_post_id=f"youtube:{vid}" if vid else None))
    return rels


def quota_reason(err: HttpFailure) -> str | None:
    """Return the quota error reason if this failure is a quota/limit error."""
    msg = str(err)
    for r in QUOTA_REASONS:
        if r in msg:
            return r
    # HttpFailure keeps only the first 300 chars of the body, and Google's error envelope
    # repeats the human message before `errors[].reason`, so the reason can be cut off.
    # Fall back to the documented message wording ("exceeded your quota").
    if err.status == 403 and "quota" in msg.lower():
        return "quotaExceeded"
    # Seen live 2026-11-20: search.list answered HTTP 429 "Quota exceeded for quota metric
    # 'Search Queries' and limit 'Search Queries per day'" (a separate per-day bucket for
    # searches, as the quota-cost page says), after ~100 searches that day.
    if err.status == 429 and "quota exceeded" in msg.lower():
        m = re.search(r"limit '([^']+)'", msg)
        return f"quotaExceeded ({m.group(1)})" if m else "quotaExceeded"
    return None


# ----------------------------------------------------------------------------- mapping


def map_video(
    video: dict | None,
    search_item: dict | None,
    channel: dict | None,
    *,
    collected_at: str,
    topic: str | None = None,
    raw_ref: str | None = None,
) -> Post | None:
    """Map a videos.list item (preferred) or a bare search.list item to a Post."""
    vid = None
    if video:
        vid = video.get("id")
    if not vid and search_item:
        vid = (search_item.get("id") or {}).get("videoId")
    if not vid:
        return None
    vsn = (video or {}).get("snippet") or {}
    ssn = (search_item or {}).get("snippet") or {}
    stats = (video or {}).get("statistics") or {}
    csn = (channel or {}).get("snippet") or {}

    title = vsn.get("title") or ssn.get("title") or ""
    # search.list descriptions are truncated; videos.list gives the full text.
    description = vsn.get("description") if video else ssn.get("description")
    description = description or ""
    text = f"{title}\n\n{description}".strip() if description else title

    # Platform timestamp: videos.list first, search.list snippet as a fallback.
    raw_ts = vsn.get("publishedAt") or ssn.get("publishedAt")
    ts_source = "videos.list" if vsn.get("publishedAt") else "search.list"
    pub = parse_ts(raw_ts)
    note = None
    if pub is None:
        note = "no parseable snippet.publishedAt"
    elif ts_source == "search.list":
        note = "timestamp from search.list snippet (videos.list not fetched)"

    channel_id = vsn.get("channelId") or ssn.get("channelId")
    url = watch_url(vid)
    return Post(
        post_id=f"youtube:{vid}",
        source="youtube",
        kind="video",
        author_id=channel_id,
        author_handle=csn.get("customUrl") or None,  # e.g. "@handle"
        author_name=vsn.get("channelTitle") or ssn.get("channelTitle") or csn.get("title"),
        author_created_at=iso(parse_ts(csn.get("publishedAt"))),
        author_avatar_url=best_thumbnail(csn.get("thumbnails")),
        text=text,
        published_at=iso(pub) if pub else None,
        collected_at=collected_at,
        timestamp_verified=pub is not None,
        timestamp_note=note,
        metrics=Metrics(
            views=_int(stats.get("viewCount")),
            likes=_int(stats.get("likeCount")),
            comments=_int(stats.get("commentCount")),
        ),
        media=[Media(type="video", url=url,
                     thumbnail=best_thumbnail(vsn.get("thumbnails") or ssn.get("thumbnails")))],
        url=url,
        relations=description_relations(description, vid),
        raw_ref=raw_ref,
        query=topic,
    )


def map_comment_thread(
    thread: dict, *, collected_at: str, topic: str | None = None, raw_ref: str | None = None
) -> Post | None:
    """Map a commentThreads.list item (its top-level comment) to a Post kind=comment."""
    tsn = thread.get("snippet") or {}
    top = tsn.get("topLevelComment") or {}
    csn = top.get("snippet") or {}
    cid = top.get("id") or thread.get("id")
    # videoId is documented on the commentThread snippet; the comment snippet may carry it
    # too. UNVERIFIED: comment.snippet.videoId (not listed on the comments resource page).
    vid = tsn.get("videoId") or csn.get("videoId")
    if not cid or not vid:
        return None
    pub = parse_ts(csn.get("publishedAt"))
    name = csn.get("authorDisplayName") or None
    # UNVERIFIED: authorDisplayName now often shows the "@handle"; only use it as a handle
    # when it actually looks like one.
    handle = name if name and name.startswith("@") else None
    author_id = (csn.get("authorChannelId") or {}).get("value")
    text = csn.get("textDisplay") or csn.get("textOriginal") or ""
    return Post(
        post_id=f"youtube:comment:{cid}",
        source="youtube",
        kind="comment",
        author_id=author_id,
        author_handle=handle,
        author_name=name,
        author_avatar_url=csn.get("authorProfileImageUrl") or None,
        text=text,
        published_at=iso(pub) if pub else None,
        collected_at=collected_at,
        timestamp_verified=pub is not None,
        timestamp_note=None if pub else "no parseable comment publishedAt",
        metrics=Metrics(likes=_int(csn.get("likeCount")),
                        replies=_int(tsn.get("totalReplyCount"))),
        # UNVERIFIED: the `lc=` deep-link parameter is a web convention, not API-documented.
        url=f"{watch_url(vid)}&lc={cid}",
        relations=[Relation(type="reply", target_post_id=f"youtube:{vid}",
                            target_url=watch_url(vid))],
        raw_ref=raw_ref,
        query=topic,
    )


# --------------------------------------------------------------------------- collector


@dataclass
class _Run:
    """Mutable book-keeping for one collect() call."""

    limit: int
    used: int = 0
    requests: int = 0
    stop_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    hard_error: str | None = None  # an error that stopped the search before any results
    search_pages: int = 0  # search.list pages actually received
    cost: int = 0  # unit cost of the logical call in flight (charged per HTTP attempt)

    def charge_attempt(self, request: httpx.Request) -> None:
        """httpx request hook: charge every HTTP attempt, including request_json retries.

        Google charges failed calls too, so a search page retried 3 times on 503 costs
        4 x 100 units. Raising _Stop here (before the request is sent) also keeps retries
        inside the YOUTUBE_QUOTA_LIMIT ceiling.
        """
        if not self.can_spend(self.cost):
            raise _Stop(f"quota ceiling {self.limit} units")
        self.used += self.cost
        self.requests += 1

    def can_spend(self, units: int) -> bool:
        return self.used + units <= self.limit


class _Stop(Exception):
    """Internal: a quota/deadline condition that ends the current stage cleanly."""


class YouTubeCollector:
    name = "youtube"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_date_filter=True,
            needs_credentials=True,
            kind="video",
            credential_keys=("YOUTUBE_API_KEY",),
            description="YouTube Data API v3: search.list by date window, videos/channels "
                        "enrichment, optional top-level comments. Quota-limited.",
        )

    # ------------------------------------------------------------------ http helpers

    def _call(self, client: httpx.Client, ctx: CollectContext, st: _Run, endpoint: str,
              params: dict[str, Any], cost: int, *, enrich: bool = False) -> dict:
        """One API call with quota and deadline guards. Raises _Stop or HttpFailure.

        enrich=True (videos.list / channels.list): an analyst stop does not cancel these
        few 1-unit batches, so the videos already found keep their metrics and channel
        data ("each source keeps what it has"); only the search stage and comments stop.
        A real time limit (RUN_TIME_BUDGET_SECONDS) still applies."""
        analyst_stop = ctx.stop.is_set()
        if enrich and analyst_stop:
            if time.monotonic() >= ctx.deadline:
                raise _Stop("time budget exhausted")
        elif ctx.out_of_time():
            raise _Stop("time budget exhausted")
        if not st.can_spend(cost):
            raise _Stop(f"quota ceiling {st.limit} units")
        # Units are charged per HTTP attempt by the client's request hook (_Run.charge_attempt),
        # so request_json retries on 429/5xx are counted too.
        st.cost = cost
        params = {**params, "key": ctx.config.get("YOUTUBE_API_KEY", "")}
        try:
            data = request_json(client, "GET", f"{API}/{endpoint}", params=params,
                                deadline=ctx.deadline,
                                stop=None if enrich else ctx.stop, max_retries=3)
        except HttpFailure as e:
            q = quota_reason(e)
            if q:
                raise _Stop(f"YouTube API {q} (HTTP {e.status}) after {st.used} units "
                            f"counted this run; API-side quota/limit reached") from e
            if "budget exhausted" in str(e) or "stopped by the analyst" in str(e):
                raise _Stop("time budget exhausted") from e
            raise
        return data if isinstance(data, dict) else {}

    # ---------------------------------------------------------------------- collect

    def collect(self, query: Query, ctx: CollectContext) -> CollectResult:
        key = str(ctx.config.get("YOUTUBE_API_KEY", "") or "").strip()
        if not key:
            return skipped(self.name, query.topic, SourceState.skipped_no_credentials,
                           "missing credential(s): YOUTUBE_API_KEY")
        try:
            limit = int(ctx.config.get("YOUTUBE_QUOTA_LIMIT") or DEFAULT_QUOTA_LIMIT)
        except ValueError:
            limit = DEFAULT_QUOTA_LIMIT
        st = _Run(limit=limit)
        started = now_iso()
        client = make_client()
        client.event_hooks = {"request": [st.charge_attempt], "response": []}
        try:
            posts, cursor, search_complete = self._run(client, query, ctx, st)
        finally:
            client.close()

        videos = sum(1 for p in posts if p.kind == "video")
        if st.hard_error and not posts:
            state = SourceState.failed
        elif st.stop_reasons or not search_complete:
            state = SourceState.partial
        elif videos == 0:
            state = SourceState.completed_zero
        else:
            state = SourceState.ok
        reasons = st.stop_reasons + ([st.hard_error] if st.hard_error else []) + st.notes
        if state is SourceState.completed_zero and st.search_pages:
            reasons = ["search.list returned no videos for this topic and window", *reasons]
        status = SourceStatus(
            source=self.name, query=query.topic, state=state, count=len(posts),
            reason="; ".join(dict.fromkeys(reasons)), cursor=cursor, date_filtered=True,
            requests=st.requests, quota_units=st.used, started_at=started,
            finished_at=now_iso(),
        )
        return CollectResult(posts=posts, status=[status])

    def _run(self, client: httpx.Client, query: Query, ctx: CollectContext,
             st: _Run) -> tuple[list[Post], str | None, bool]:
        """Returns (posts, last search cursor if search stopped early, search_complete)."""
        collected_at = now_iso()
        topic = query.topic

        # ---- 1. search.list, one paginated pass per time slice ------------------------
        order, search_items, search_refs, cursor, search_complete = self._search(
            client, query, ctx, st)

        # ---- 2. videos.list (exact timestamps + stats) ---------------------------------
        # No maxResults here or in channels.list: the docs say it is not supported together
        # with `id`; the ≤50-id batch is the page size.
        videos: dict[str, dict] = {}
        video_refs: dict[str, str] = {}
        for i in range(0, len(order), 50):
            batch = order[i:i + 50]
            try:
                data = self._call(client, ctx, st, "videos", {
                    "part": "snippet,statistics,contentDetails", "id": ",".join(batch)},
                    COST_VIDEOS, enrich=True)
            except (_Stop, HttpFailure) as e:
                st.stop_reasons.append(
                    f"videos.list stopped ({e}); {len(order) - i} videos kept with "
                    f"search.list snippet only (no metrics)")
                break
            ref = ctx.save_raw(self.name, f"videos_{i // 50 + 1:03d}.json", data)
            for v in data.get("items") or []:
                if v.get("id"):
                    videos[v["id"]] = v
                    video_refs[v["id"]] = ref

        # ---- 3. channels.list (author metadata) ----------------------------------------
        channel_ids: list[str] = []
        for vid in order:
            cid = ((videos.get(vid) or {}).get("snippet") or {}).get("channelId") or \
                ((search_items[vid].get("snippet") or {}).get("channelId"))
            if cid and cid not in channel_ids:
                channel_ids.append(cid)
        channels: dict[str, dict] = {}
        for i in range(0, len(channel_ids), 50):
            batch = channel_ids[i:i + 50]
            try:
                data = self._call(client, ctx, st, "channels", {
                    "part": "snippet,statistics", "id": ",".join(batch)},
                    COST_CHANNELS, enrich=True)
            except (_Stop, HttpFailure) as e:
                st.stop_reasons.append(f"channels.list stopped ({e}); channel metadata "
                                       f"missing for {len(channel_ids) - i} channels")
                break
            ctx.save_raw(self.name, f"channels_{i // 50 + 1:03d}.json", data)
            for c in data.get("items") or []:
                if c.get("id"):
                    channels[c["id"]] = c

        posts: list[Post] = []
        off_topic = 0
        for vid in order:
            v = videos.get(vid)
            sitem = search_items.get(vid)
            if v is not None:
                vs = v.get("snippet") or {}
                text = " ".join([vs.get("title") or "", vs.get("description") or "",
                                 " ".join(vs.get("tags") or [])])
                if not mentions_topic(query, text):
                    # YouTube also matches captions and related terms; only videos whose
                    # own title, description or tags carry the topic are evidence.
                    off_topic += 1
                    continue
            cid = ((v or {}).get("snippet") or {}).get("channelId") or \
                ((sitem or {}).get("snippet") or {}).get("channelId")
            p = map_video(v, sitem, channels.get(cid) if cid else None,
                          collected_at=collected_at, topic=topic,
                          raw_ref=video_refs.get(vid) or search_refs.get(vid))
            if p:
                posts.append(p)
        if off_topic:
            st.notes.append(f"{off_topic} video(s) returned by the search did not mention the "
                            "topic in their title, description or tags and were not kept")
        if order and len(videos) < len(order) and not any(
                r.startswith("videos.list") for r in st.stop_reasons):
            st.notes.append(f"videos.list returned no data for {len(order) - len(videos)} "
                            f"video(s) (deleted or private since search)")

        # ---- 4. commentThreads.list (optional) -----------------------------------------
        try:
            n_comment_videos = int(query.options.get("youtube_comments", DEFAULT_COMMENT_VIDEOS))
        except (TypeError, ValueError):
            n_comment_videos = DEFAULT_COMMENT_VIDEOS
        if n_comment_videos > 0 and posts:
            posts.extend(self._comments(client, ctx, st, posts, n_comment_videos,
                                        collected_at, topic))
        return posts, cursor, search_complete

    def _search(self, client: httpx.Client, query: Query, ctx: CollectContext, st: _Run
                ) -> tuple[list[str], dict[str, dict], dict[str, str], str | None, bool]:
        """search.list over the window, one slice (UTC day or group of days) at a time.

        Search returns newest first, so a single pass over the whole window fills the cap
        from the last hours (seen live 2026-11-19). Each slice gets a share of the video
        cap (unused share rolls to later slices, window.SliceShare); slices run
        sequentially, newest first.

        Returns (video ids in order, search items, raw refs, cursor, search_complete).
        cursor is "<slice first date>:<nextPageToken or empty>" for the slice where the
        search stopped early, None when every slice was searched.
        """
        # Max slices: each slice costs at least one search page (COST_SEARCH units), so
        # cap the slice count at half the quota ceiling's worth of search pages. The other
        # half stays for extra pages in busy slices and for videos/channels/comments.
        # With the default 2000-unit ceiling: min(10, 1000 // 100) = 10 slices.
        max_slices = min(DEFAULT_MAX_SLICES, max(1, (st.limit // 2) // COST_SEARCH))
        slices = time_slices(query.start, query.end, max_slices)
        share = SliceShare(query.limit, len(slices))
        # Phrase mode quotes a plain multi-word topic. Live check 2026-11-19 ("bank of
        # america", 2026-11-18, 25 results each): bare 18/25 and quoted 17/25 had the phrase
        # in the (truncated) search title+description; totalResults estimate 52200 vs 27708;
        # 13 ids in common. Quoting narrows the result set but did not visibly raise
        # precision on the first page; kept because it is harmless and consistent with X.
        q = search_topic(query)

        order: list[str] = []
        search_items: dict[str, dict] = {}
        search_refs: dict[str, str] = {}
        cursor: str | None = None
        stopped = False
        capped_with_more: list[str] = []
        unsearched: list[str] = []  # zero-allowance slices: no request, not a quiet day
        page = 0  # global page counter for raw file names
        for i, (s, e) in enumerate(slices):
            cap = share.allowance(i)
            if cap <= 0:
                share.record(i, 0)  # nothing left to share: no request for this slice
                unsearched.append(f"{s:%Y-%m-%d}" if s.date() == e.date()
                                  else f"{s:%Y-%m-%d}..{e:%Y-%m-%d}")
                continue
            got = 0
            requested = False
            page_token: str | None = None
            while got < cap:
                params: dict[str, Any] = {
                    "part": "snippet", "type": "video", "q": q, "order": SEARCH_ORDER,
                    "publishedAfter": iso(s), "publishedBefore": iso(e),
                    "maxResults": min(50, max(1, cap - got)),
                }
                if page_token:
                    params["pageToken"] = page_token
                try:
                    data = self._call(client, ctx, st, "search", params, COST_SEARCH)
                except _Stop as stop:
                    st.stop_reasons.append(f"search stopped at {s:%Y-%m-%d}: {stop}")
                    stopped = True
                    break
                except HttpFailure as err:
                    msg = f"search.list failed at {s:%Y-%m-%d}: {err}"
                    if order:
                        st.stop_reasons.append(msg)
                    else:
                        st.hard_error = msg
                    stopped = True
                    break
                requested = True
                st.search_pages += 1
                page += 1
                ref = ctx.save_raw(self.name, f"search_{page:03d}.json", data)
                for item in data.get("items") or []:
                    if got >= cap:
                        break
                    vid = (item.get("id") or {}).get("videoId")
                    if vid and vid not in search_items:
                        order.append(vid)
                        search_items[vid] = item
                        search_refs[vid] = ref
                        got += 1
                ctx.progress(self.name, {"count": len(order), "quota_units": st.used})
                page_token = data.get("nextPageToken") or None
                if not page_token:
                    break
            else:
                if page_token:
                    capped_with_more.append(f"{s:%Y-%m-%d}")
            if requested or got:
                share.record(i, got)
            if stopped:
                cursor = f"{s:%Y-%m-%d}:{page_token or ''}"
                break

        if capped_with_more:
            st.notes.append(f"stopped at the per-slice share of the post limit {query.limit}; "
                            f"more results exist for {', '.join(capped_with_more)}")
        if unsearched:
            # The summary shows these as "0"; say that they were never searched, so a
            # zero there is not read as a quiet day (CLAUDE.md: zero results explained).
            st.notes.append(f"not searched, post limit {query.limit} already used by newer "
                            f"slices: {', '.join(unsearched)}")
        st.notes.append(share.summary(slices))
        return order[: query.limit], search_items, search_refs, cursor, not stopped

    def _comments(self, client: httpx.Client, ctx: CollectContext, st: _Run,
                  video_posts: list[Post], n: int, collected_at: str,
                  topic: str) -> list[Post]:
        """Top-level comments (one page, newest first) for the N most-viewed videos."""
        ranked = sorted((p for p in video_posts if p.kind == "video"),
                        key=lambda p: p.metrics.views or 0, reverse=True)
        # Drop videos that report commentCount == 0 BEFORE taking the top N, so they do not
        # use up a slot. Unknown counts (None: stats not fetched/hidden) are still tried.
        ranked = [p for p in ranked if p.metrics.comments != 0][:n]
        out: list[Post] = []
        disabled = 0
        failed = 0
        attempted = 0
        for p in ranked:
            vid = p.post_id.split(":", 1)[1]
            attempted += 1
            try:
                data = self._call(client, ctx, st, "commentThreads", {
                    "part": "snippet", "videoId": vid, "maxResults": 100, "order": "time",
                    "textFormat": "plainText"}, COST_COMMENTS)
            except _Stop as e:
                st.stop_reasons.append(f"comments stopped: {e}")
                break
            except HttpFailure as e:
                # commentsDisabled is a documented 403 for commentThreads.list.
                # (Matched on reason or message wording: the body may be truncated.)
                if "commentsDisabled" in str(e) or "disabled comments" in str(e):
                    disabled += 1
                else:
                    # A real failure (5xx after retries, 404, network): comment coverage is
                    # incomplete, so the source is partial, not ok.
                    failed += 1
                    st.stop_reasons.append(f"commentThreads.list failed for {vid}: {e}")
                continue
            ref = ctx.save_raw(self.name, f"comments_{vid}.json", data)
            for t in data.get("items") or []:
                c = map_comment_thread(t, collected_at=collected_at, topic=topic, raw_ref=ref)
                if c:
                    out.append(c)
            if data.get("nextPageToken"):
                st.notes.append(f"comments for {vid}: first 100 threads only")
        if disabled:
            st.notes.append(f"comments disabled on {disabled} video(s)")
        if failed:
            st.stop_reasons.append(f"comment collection failed for {failed} of {attempted} "
                                   f"video(s); missing comments are not evidence of none")
        return out
