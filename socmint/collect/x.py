"""X collector: owner's logged-in web session (default), TwitterAPI.io or official X API.

Provider is chosen by config `X_PROVIDER`:
  - `web_session` (default, owner decision 2026-11-18): the owner's own x.com login driven
    in a local browser; implemented in socmint.collect.x_web.
  - `twitterapi_io`: GET https://api.twitterapi.io/twitter/tweet/advanced_search
    with header `X-API-Key`. Documented in the public OpenAPI spec
    (https://docs.twitterapi.io/api-reference/openapi.json); not yet verified live.
  - `official`: GET https://api.x.com/2/tweets/search/recent with a bearer token. Recent
    search only covers the last 7 days; the older part of a window is reported as
    `out_of_coverage` instead of silently returning fewer posts.

Evidence rules encoded here (CLAUDE.md §3):
  - `published_at` comes only from the platform `createdAt` / `created_at` field;
    `timestamp_verified=True` only if that value parses to tz-aware UTC.
  - Relations are DIRECT only and come only from platform fields (reply id, quoted /
    retweeted tweet, entity URLs, entity mentions). Nothing is inferred here.
  - Quoted / retweeted tweets embedded in a response were NOT matched by the topic
    search. They must never enter narrative clustering, volume counts or "first observed"
    analysis. They are emitted as their own posts only when the schema has a
    `context_only` flag that verify.analysable() can exclude (foundation request);
    until then they are dropped and the direct edge survives through the relation's
    target_post_id / target_url / target_handle.
  - Every page is saved raw and referenced from `Post.raw_ref`.
  - Every stop reason (no more pages, limit, deadline, HTTP error) is reported in the
    SourceStatus so an empty or short result is never mistaken for "no posts".
  - Search returns newest first, so the window is split into whole-day time slices
    (socmint.collect.window) searched one after another, newest first, sharing the
    per-source limit; the status reason lists the posts kept per day. A plain multi-word
    topic is searched as an exact phrase unless the run's match option is 'words'.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from socmint.collect.window import (
    DEFAULT_MAX_SLICES,
    SliceShare,
    search_topic,
    time_slices,
)
from socmint.config import load_config
from socmint.interfaces import Capabilities, CollectContext, CollectResult, Query
from socmint.schema import Media, Metrics, Post, Relation, SourceState, SourceStatus
from socmint.util.http import HttpFailure, make_client, request_json
from socmint.util.log import get_logger
from socmint.util.timeutil import iso, now_iso, now_utc, parse_ts

log = get_logger(__name__)

SOURCE = "x"

TWITTERAPI_IO_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
# Published rate (docs.twitterapi.io/pricing): 15 credits per tweet returned
# ($0.15 / 1k tweets), minimum 15 credits per request even when it returns nothing;
# 100,000 credits = $1. UNVERIFIED: not reconciled against a real invoice.
USD_PER_TWEET = 0.15 / 1000
MAX_QUERY_LEN = 512  # documented: longer queries return HTTP 400
# TwitterAPI.io can return pages with no tweets but has_next_page=true (pages are
# filtered for ads). Stop after this many consecutive empty pages to avoid paying for
# an endless cursor walk.
MAX_EMPTY_PAGES = 3

# Marker for posts that were embedded in a response but not matched by the search.
# Post has no such field yet (foundation request); see _context_posts().
CONTEXT_FLAG = "context_only"

OFFICIAL_URL = "https://api.x.com/2/tweets/search/recent"
# x.com's own website ships one public app token in its JavaScript (identical for every
# visitor; it is what DevTools shows in the `authorization` header). It only works on
# x.com's internal endpoints together with a logged-in session, never on the official
# X API, which answers 401. Only its fixed, public prefix is kept here, to recognise it.
X_WEB_APP_TOKEN_PREFIX = "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAA"
X_WEB_APP_TOKEN_PROBLEM = (
    "X_BEARER_TOKEN is the x.com WEBSITE token (copied from the browser's DevTools), not "
    "an X API token, so the official API rejects it (HTTP 401). Use a bearer token from "
    "your app at developer.x.com, or switch X_PROVIDER to twitterapi_io or web_session")
OFFICIAL_WINDOW = timedelta(days=7)
# Recent search rejects a start_time older than 7 days *at request time*. Pagination
# with 429 backoff can take minutes, so the coverage start keeps a 15 min margin and
# each request re-clamps start_time to at least now - 7 days + OFFICIAL_REQ_MARGIN.
OFFICIAL_START_MARGIN = timedelta(minutes=15)
OFFICIAL_REQ_MARGIN = timedelta(minutes=1)
# UNVERIFIED: the classic v2 docs required end_time to be at least 10 s before the
# request; we keep a 30 s margin. Field/expansion names below are the classic v2
# names (tweet.fields, referenced_tweets.id); the newer docs.x.com pages show
# post.fields / referenced_posts aliases. Not verified live.
OFFICIAL_END_MARGIN = timedelta(seconds=30)
OFFICIAL_PARAMS = {
    "max_results": "100",
    "expansions": "author_id,attachments.media_keys,referenced_tweets.id",
    "tweet.fields": "created_at,public_metrics,referenced_tweets,entities,lang,"
                    "in_reply_to_user_id,author_id,attachments",
    "user.fields": "created_at,profile_image_url,username,name",
    "media.fields": "type,url,preview_image_url,variants",
}

_LANG_UNKNOWN = {"", "und", "zxx"}

# Callback types for the shared pagination loop.
FetchFn = Callable[[str | None], dict]
ParseFn = Callable[[dict, str, str], tuple[list[Post], list[Post], str | None]]


# ------------------------------------------------------------------- small helpers


def _int(v: Any) -> int | None:
    """Metric value or None when absent (None means 'not exposed', never 0)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _handle(username: str | None) -> str | None:
    return f"@{username.lstrip('@')}" if username else None


def _status_url(username: str | None, tweet_id: str, fallback: str | None = None) -> str:
    if username:
        return f"https://x.com/{username.lstrip('@')}/status/{tweet_id}"
    return fallback or f"https://x.com/i/status/{tweet_id}"


def _lang(v: Any) -> str | None:
    s = str(v or "").strip()
    return None if s in _LANG_UNKNOWN else s


def _platform_time(value: Any, field: str) -> tuple[str | None, bool, str | None]:
    """(published_at, timestamp_verified, note) from the platform's own time field."""
    dt = parse_ts(value) if value else None
    if dt is None:
        note = f"platform field {field} missing" if not value else \
            f"platform field {field} not parseable: {str(value)[:60]!r}"
        return None, False, note
    return iso(dt), True, None


def _is_self_status_link(url: str, tweet_ids: set[str]) -> bool:
    """True for x.com/twitter.com permalinks to a tweet already linked as quote/repost."""
    m = re.search(r"(?:x|twitter)\.com/[^/]+/status(?:es)?/(\d+)", url or "")
    return bool(m and m.group(1) in tweet_ids)


# --------------------------------------------------------- TwitterAPI.io parsing


def _tio_media(tweet: dict) -> list[Media]:
    # UNVERIFIED: media is not in the TwitterAPI.io OpenAPI schema. We read the classic
    # X shape (`extendedEntities.media[]` / `entities.media[]` with `type`,
    # `media_url_https`, `video_info.variants[]`) defensively; replace with the live
    # shape once a live_*.json fixture exists.
    items = ((tweet.get("extendedEntities") or {}).get("media")
             or (tweet.get("entities") or {}).get("media") or [])
    out: list[Media] = []
    for m in items:
        if not isinstance(m, dict):
            continue
        mtype = m.get("type")
        thumb = m.get("media_url_https") or m.get("media_url")
        if mtype in ("video", "animated_gif"):
            variants = [v for v in ((m.get("video_info") or {}).get("variants") or [])
                        if isinstance(v, dict) and v.get("url")]
            mp4 = [v for v in variants if v.get("content_type") == "video/mp4"] or variants
            best = max(mp4, key=lambda v: v.get("bitrate") or 0, default=None)
            out.append(Media(type=mtype, url=(best or {}).get("url") or thumb, thumbnail=thumb))
        elif mtype == "photo":
            out.append(Media(type="photo", url=thumb, thumbnail=thumb))
        elif thumb:
            out.append(Media(type="other", url=thumb, thumbnail=thumb))
    return out


def _tio_relations(tweet: dict) -> list[Relation]:
    """DIRECT relations only, each taken from a named platform field."""
    rels: list[Relation] = []
    linked: set[str] = set()
    reply_to = str(tweet.get("inReplyToId") or "").strip()
    if reply_to:
        rels.append(Relation(type="reply", target_post_id=f"x:{reply_to}",
                             target_handle=_handle(tweet.get("inReplyToUsername"))))
    for field, rtype in (("quoted_tweet", "quote"), ("retweeted_tweet", "repost")):
        ref = tweet.get(field)
        if isinstance(ref, dict) and ref.get("id"):
            rid = str(ref["id"])
            linked.add(rid)
            user = (ref.get("author") or {}).get("userName")
            rels.append(Relation(type=rtype, target_post_id=f"x:{rid}",
                                 target_url=_status_url(user, rid, ref.get("url")),
                                 target_handle=_handle(user)))
    ent = tweet.get("entities") or {}
    for u in ent.get("urls") or []:
        url = (u or {}).get("expanded_url") or (u or {}).get("url")
        if url and not _is_self_status_link(url, linked):
            rels.append(Relation(type="link", target_url=url))
    for m in ent.get("user_mentions") or []:
        name = (m or {}).get("screen_name")
        if name:
            rels.append(Relation(type="mention", target_handle=_handle(name)))
    return rels


def parse_twitterapi_io_tweet(tweet: dict, *, collected_at: str, raw_ref: str | None,
                              query: str | None) -> Post | None:
    """Map one TwitterAPI.io Tweet object to a Post (None if it has no id)."""
    tid = str(tweet.get("id") or "").strip()
    if not tid:
        return None
    author = tweet.get("author") or {}
    user = author.get("userName")
    published, verified, note = _platform_time(tweet.get("createdAt"), "createdAt")
    author_created = parse_ts(author.get("createdAt")) if author.get("createdAt") else None
    return Post(
        post_id=f"x:{tid}",
        source=SOURCE,
        kind="post",
        author_id=str(author["id"]) if author.get("id") else None,
        author_handle=_handle(user),
        author_name=author.get("name") or None,
        author_created_at=iso(author_created),
        author_avatar_url=author.get("profilePicture") or None,
        text=tweet.get("text") or "",
        lang=_lang(tweet.get("lang")),
        published_at=published,
        collected_at=collected_at,
        timestamp_verified=verified,
        timestamp_note=note,
        metrics=Metrics(
            views=_int(tweet.get("viewCount")),
            likes=_int(tweet.get("likeCount")),
            replies=_int(tweet.get("replyCount")),
            reposts=_int(tweet.get("retweetCount")),
            quotes=_int(tweet.get("quoteCount")),
        ),
        media=_tio_media(tweet),
        url=_status_url(user, tid, tweet.get("url")),
        relations=_tio_relations(tweet),
        raw_ref=raw_ref,
        query=query,
    )


def parse_twitterapi_io_page(data: dict, *, collected_at: str, raw_ref: str | None,
                             query: str) -> tuple[list[Post], list[Post]]:
    """(matched posts, embedded quoted/retweeted posts) from one advanced_search page."""
    main: list[Post] = []
    embedded: list[Post] = []
    for t in data.get("tweets") or []:
        if not isinstance(t, dict):
            continue
        p = parse_twitterapi_io_tweet(t, collected_at=collected_at, raw_ref=raw_ref,
                                      query=query)
        if p is None:
            continue
        main.append(p)
        for field, label in (("quoted_tweet", "embedded:quoted"),
                             ("retweeted_tweet", "embedded:retweeted")):
            ref = t.get(field)
            # Only emit an embedded tweet as evidence when it carries its own platform
            # timestamp; otherwise the edge still exists via the relation's target_url.
            if isinstance(ref, dict) and ref.get("createdAt"):
                ep = parse_twitterapi_io_tweet(ref, collected_at=collected_at,
                                               raw_ref=raw_ref, query=label)
                if ep is not None:
                    embedded.append(ep)
    return main, embedded


def build_twitterapi_io_query(topic: str, start: datetime, end: datetime) -> str:
    """Topic plus the documented Unix-seconds window operators.

    TwitterAPI.io documents `since_time:` / `until_time:` (Unix seconds) and states the
    `since:YYYY-MM-DD_HH:MM:SS_UTC` form is NOT supported. `end` is the inclusive last
    second, so until_time is end+1. verify.gate_timestamps still enforces the window.

    The topic is wrapped in parentheses so a user-supplied `OR` cannot bind the time
    operators to only one side of it (topics pass through so users may add operators).
    """
    since = int(start.timestamp())
    until = int(end.timestamp()) + 1
    return f"({topic.strip()}) since_time:{since} until_time:{until}"


# ------------------------------------------------------------- official v2 parsing


def _official_media(tweet: dict, media_by_key: dict[str, dict]) -> list[Media]:
    out: list[Media] = []
    for key in (tweet.get("attachments") or {}).get("media_keys") or []:
        m = media_by_key.get(key)
        if not m:
            continue
        mtype = m.get("type")
        if mtype == "photo":
            out.append(Media(type="photo", url=m.get("url"), thumbnail=m.get("url")))
        elif mtype in ("video", "animated_gif"):
            variants = [v for v in m.get("variants") or [] if isinstance(v, dict) and v.get("url")]
            best = max(variants, key=lambda v: v.get("bit_rate") or 0, default=None)
            out.append(Media(type=mtype, url=(best or {}).get("url"),
                             thumbnail=m.get("preview_image_url")))
    return out


def parse_official_page(data: dict, *, collected_at: str, raw_ref: str | None,
                        query: str) -> tuple[list[Post], list[Post]]:
    """(matched posts, embedded referenced posts) from one v2 recent-search page."""
    inc = data.get("includes") or {}
    users = {u.get("id"): u for u in inc.get("users") or [] if isinstance(u, dict)}
    media = {m.get("media_key"): m for m in inc.get("media") or [] if isinstance(m, dict)}
    included = {t.get("id"): t for t in inc.get("tweets") or [] if isinstance(t, dict)}
    rtype_map = {"replied_to": "reply", "quoted": "quote", "retweeted": "repost"}

    def to_post(t: dict, q: str) -> Post | None:
        tid = str(t.get("id") or "")
        if not tid:
            return None
        u = users.get(t.get("author_id")) or {}
        uname = u.get("username")
        published, verified, note = _platform_time(t.get("created_at"), "created_at")
        rels: list[Relation] = []
        linked: set[str] = set()
        for ref in t.get("referenced_tweets") or []:
            rt = rtype_map.get((ref or {}).get("type"))
            rid = str((ref or {}).get("id") or "")
            if not rt or not rid:
                continue
            linked.add(rid)
            ru = users.get((included.get(rid) or {}).get("author_id")) or {}
            rels.append(Relation(type=rt, target_post_id=f"x:{rid}",
                                 target_url=_status_url(ru.get("username"), rid),
                                 target_handle=_handle(ru.get("username"))))
        ent = t.get("entities") or {}
        for e in ent.get("urls") or []:
            url = (e or {}).get("expanded_url") or (e or {}).get("url")
            if url and not _is_self_status_link(url, linked):
                rels.append(Relation(type="link", target_url=url))
        for m in ent.get("mentions") or []:
            if (m or {}).get("username"):
                rels.append(Relation(type="mention", target_handle=_handle(m["username"])))
        pm = t.get("public_metrics") or {}
        created = parse_ts(u.get("created_at")) if u.get("created_at") else None
        return Post(
            post_id=f"x:{tid}", source=SOURCE, kind="post",
            author_id=t.get("author_id"), author_handle=_handle(uname),
            author_name=u.get("name") or None, author_created_at=iso(created),
            author_avatar_url=u.get("profile_image_url") or None,
            text=t.get("text") or "", lang=_lang(t.get("lang")),
            published_at=published, collected_at=collected_at,
            timestamp_verified=verified, timestamp_note=note,
            metrics=Metrics(
                views=_int(pm.get("impression_count")),  # UNVERIFIED: field availability
                likes=_int(pm.get("like_count")), replies=_int(pm.get("reply_count")),
                reposts=_int(pm.get("retweet_count")), quotes=_int(pm.get("quote_count"))),
            media=_official_media(t, media), url=_status_url(uname, tid),
            relations=rels, raw_ref=raw_ref, query=q,
        )

    main = [p for p in (to_post(t, query) for t in data.get("data") or []
                        if isinstance(t, dict)) if p]
    main_ids = {p.post_id for p in main}
    embedded: list[Post] = []
    for t in included.values():
        if not t.get("created_at") or f"x:{t.get('id')}" in main_ids:
            continue
        kinds = {r.type for p in main for r in p.relations
                 if r.target_post_id == f"x:{t.get('id')}"}
        label = "embedded:" + ("quoted" if "quote" in kinds else
                               "retweeted" if "repost" in kinds else "replied_to")
        p = to_post(t, label)
        if p:
            embedded.append(p)
    return main, embedded


# ----------------------------------------------------------------- the collector


def official_bearer(raw: Any) -> str:
    """The X API bearer token as entered, without quotes or a pasted 'Bearer ' prefix."""
    t = str(raw or "").strip().strip("'\"").strip()
    if t[:7].lower() == "bearer ":
        t = t[7:].strip()
    return t


def official_bearer_problem(raw: Any) -> str | None:
    """Why a configured X_BEARER_TOKEN cannot work, or None (checked without a request)."""
    t = official_bearer(raw)
    if t.startswith(X_WEB_APP_TOKEN_PREFIX):
        return X_WEB_APP_TOKEN_PROBLEM
    if any(c.isspace() for c in t):
        return "X_BEARER_TOKEN contains spaces or line breaks; paste the token only"
    return None


def _context_posts(main: list[Post], embedded: list[Post]) -> list[Post]:
    """Embedded quoted/retweeted posts to emit alongside the matched posts.

    EVIDENCE RULE: these tweets were not matched by the topic search, so they must not
    reach clustering, volume counts or "first observed" analysis. They are emitted only
    when Post has a `context_only` field (requested from the foundation, to be excluded
    by verify.analysable); otherwise they are dropped and the direct edge remains
    resolvable through the relation's target_post_id / target_url / target_handle.
    Kept: only endpoints of a kept post's direct relation, once each, never duplicating
    a matched post.
    """
    if CONTEXT_FLAG not in Post.model_fields:
        return []
    keep = {p.post_id for p in main}
    referenced = {r.target_post_id for p in main for r in p.relations}
    extra: dict[str, Post] = {}
    for p in embedded:
        if p.post_id in referenced and p.post_id not in keep:
            extra.setdefault(p.post_id, p.model_copy(update={CONTEXT_FLAG: True}))
    return list(extra.values())


def _slice_label(s: datetime, e: datetime) -> str:
    return f"{s:%Y-%m-%d}" if s.date() == e.date() else f"{s:%Y-%m-%d}..{e:%Y-%m-%d}"


@dataclass
class _SliceOutcome:
    """How one time slice ended: 'done' (share reached or search exhausted), 'cut'
    (stopped early, the run continues with the next slice) or 'halt' (the run stops)."""
    kind: str = "done"
    state: SourceState = SourceState.ok
    note: str = ""
    cursor: str | None = None
    got: int = 0
    exhausted: bool = False


class XCollector:
    name = SOURCE

    def capabilities(self) -> Capabilities:
        provider = (load_config().get("X_PROVIDER") or "twitterapi_io").strip().lower()
        if provider == "web_session":
            # No config key is required: a saved browser login OR pasted cookies both
            # work, so collect_web itself reports skipped_no_credentials when neither is set.
            return Capabilities(
                supports_date_filter=True,
                needs_credentials=False,
                kind="social",
                credential_keys=(),
                description="X search through the owner's logged-in x.com session in a "
                            "local browser (date-bounded with since_time:/until_time:).",
            )
        key = "X_BEARER_TOKEN" if provider == "official" else "TWITTERAPI_IO_KEY"
        return Capabilities(
            supports_date_filter=True,
            needs_credentials=True,
            kind="social",
            credential_keys=(key,),
            description=f"X search via {provider} (date-bounded server-side).",
        )

    def collect(self, query: Query, ctx: CollectContext) -> CollectResult:
        provider = (ctx.config.get("X_PROVIDER") or "twitterapi_io").strip().lower()
        if provider == "web_session":
            # Lazy import: x_web imports helpers from this module.
            from socmint.collect import x_web
            return x_web.collect_web(query, ctx)
        if provider == "official":
            return self._collect_official(query, ctx)
        if provider != "twitterapi_io":
            return CollectResult(posts=[], status=[SourceStatus(
                source=SOURCE, query=query.topic, state=SourceState.failed,
                reason=f"unknown X_PROVIDER {provider!r}; use web_session, twitterapi_io "
                       "or official")])
        return self._collect_twitterapi_io(query, ctx)

    # ---------------------------------------------------------------- shared loop

    def _paginate_slice(self, cap: int, ctx: CollectContext, status: SourceStatus,
                        fetch: FetchFn, parse: ParseFn, seen: dict[str, Post],
                        embedded: list[Post], counter: dict[str, int]) -> _SliceOutcome:
        """Walk the pages of ONE time slice until exhausted / its share / deadline / error.

        New posts go into `seen` (deduplicated across slices); at most `cap` per slice.
        """
        out = _SliceOutcome()
        cursor: str | None = None
        pages = empty_streak = 0
        while True:
            if ctx.out_of_time():
                out.kind, out.state, out.cursor = "halt", SourceState.partial, cursor
                out.note = f"time budget exhausted after {pages} page(s) of this slice"
                return out
            try:
                data = fetch(cursor)
            except HttpFailure as e:
                hint = " (check the X credential)" if e.status in (401, 403) else ""
                out.kind, out.state, out.cursor = "halt", SourceState.failed, cursor
                out.note = f"HTTP failure on page {pages + 1} of this slice{hint}: {e}"
                return out
            pages += 1
            counter["pages"] += 1
            status.requests += 1
            raw_ref = ctx.save_raw(SOURCE, f"page_{counter['pages']}.json", data)
            got, emb, next_cursor = parse(data, raw_ref, now_iso())
            new: dict[str, Post] = {}
            for p in got:
                if p.post_id not in seen:
                    new.setdefault(p.post_id, p)
            new_posts = list(new.values())
            keep = new_posts[: max(0, cap - out.got)]
            dropped = len(new_posts) - len(keep)
            for p in keep:
                seen[p.post_id] = p
            out.got += len(keep)
            embedded.extend(emb)
            counter["billed"] += max(len(got), 1)  # documented minimum per request
            ctx.progress(SOURCE, {"count": len(seen), "pages": counter["pages"],
                                  "cost_usd": round(counter["billed"] * USD_PER_TWEET, 5)})
            empty_streak = 0 if got else empty_streak + 1
            if out.got >= cap:
                out.note = f"reached its share of {cap} post(s)"
                if dropped:
                    # Posts were cut from THIS page: the next-page cursor would skip them.
                    where = ("its cursor " + repr(cursor) if cursor else
                             "its first page (the cut was on page 1)")
                    out.note += (f"; {dropped} post(s) on the last page were not kept; "
                                 f"resume from {where}, which re-fetches that page, so up "
                                 f"to {len(got)} post(s) repeat")
                elif next_cursor:
                    out.note += f"; more results were available (next cursor {next_cursor!r})"
                return out
            if not next_cursor or next_cursor == cursor:
                out.exhausted = True
                out.note = f"search exhausted after {pages} page(s)"
                return out
            if empty_streak >= MAX_EMPTY_PAGES:
                # The provider still reports more pages: this is NOT "no posts".
                out.kind, out.cursor = "cut", next_cursor
                out.note = (f"stopped after {MAX_EMPTY_PAGES} consecutive empty pages although "
                            "the API reported more (has_next_page); results may be incomplete")
                return out
            cursor = next_cursor

    def _collect_slices(self, query: Query, ctx: CollectContext, status: SourceStatus,
                        slices: list[tuple[datetime, datetime]], whole_search: str,
                        fetch_for: Callable[[datetime, datetime], FetchFn],
                        parse: ParseFn) -> tuple[list[Post], int]:
        """One paginated search per time slice (newest first), sharing `query.limit`.

        Fills `status` (state, reason incl. per-day counts, count, cursor). Returns
        (posts incl. context-only embedded posts, number of tweets billed).
        """
        share = SliceShare(query.limit, len(slices))
        seen: dict[str, Post] = {}
        embedded: list[Post] = []
        counter = {"pages": 0, "billed": 0}
        notes: list[str] = []
        halt: _SliceOutcome | None = None
        stop_date: str | None = None
        cut_cursor: str | None = None
        any_cut, all_exhausted = False, True
        capless = 0
        for i, (s, e) in enumerate(slices):
            label = _slice_label(s, e)
            if ctx.out_of_time():
                halt = _SliceOutcome(kind="halt", state=SourceState.partial,
                                     note="time budget exhausted before this slice")
                notes.append(f"{label}: {halt.note}")
                # This slice was never started: no cursor (not an earlier cut slice's).
                stop_date, cut_cursor = f"{s:%Y-%m-%d}", None
                break
            cap = share.allowance(i)
            if cap <= 0:  # the cap is used up: no request for this slice
                capless += 1
                continue
            out = self._paginate_slice(cap, ctx, status, fetch_for(s, e), parse, seen,
                                       embedded, counter)
            share.record(i, out.got)
            notes.append(f"{label}: {out.note}")
            if not (out.kind == "done" and out.exhausted):
                all_exhausted = False
            if out.kind == "halt":
                halt, stop_date = out, f"{s:%Y-%m-%d}"
                cut_cursor = out.cursor
                break
            if out.kind == "cut":
                any_cut = True
                stop_date, cut_cursor = f"{s:%Y-%m-%d}", out.cursor
        main = list(seen.values())
        if halt is not None:
            state = halt.state
            if state == SourceState.failed and main:
                state = SourceState.partial
            head = halt.note
        elif any_cut:
            state, head = SourceState.partial, "some slices were cut short (see slice notes)"
        elif main:
            state = SourceState.ok
            head = (f"collected {len(main)} post(s) across {len(slices)} time slice(s), "
                    f"newest first, sharing the per-source limit of {query.limit}")
        elif capless == len(slices):
            # Nothing was requested (per-source limit 0): never report "0 matching posts".
            state, head = SourceState.partial, "no slice searched"
            stop_date = f"{slices[0][0]:%Y-%m-%d}" if slices else None
        elif all_exhausted:
            state = SourceState.completed_zero
            head = (f"search completed with 0 matching posts in every time slice for query "
                    f"{whole_search!r}")
        else:  # defensive: nothing collected and not every slice proven empty
            state, head = SourceState.partial, "not every slice was searched to its end"
        if capless:
            head += (f"; {capless} older slice(s) not searched: the per-source limit of "
                     f"{query.limit} was used up by newer slices")
        if state == SourceState.partial and stop_date:
            status.cursor = f"{stop_date}:{cut_cursor or ''}"
        status.state, status.count = state, len(main)
        status.reason = f"{head}; {share.summary(slices)}; slices: " + " | ".join(notes)
        return main + _context_posts(main, embedded), counter["billed"]

    # -------------------------------------------------------------- TwitterAPI.io

    def _collect_twitterapi_io(self, query: Query, ctx: CollectContext) -> CollectResult:
        topic = search_topic(query)
        whole = build_twitterapi_io_query(topic, query.start, query.end)
        status = SourceStatus(source=SOURCE, query=query.topic, state=SourceState.ok,
                              started_at=now_iso())
        if len(whole) > MAX_QUERY_LEN:
            status.state = SourceState.failed
            status.reason = (f"search string is {len(whole)} chars; TwitterAPI.io accepts "
                             f"at most {MAX_QUERY_LEN}. Shorten the topic.")
            return CollectResult(posts=[], status=[status])
        headers = {"X-API-Key": ctx.config.get("TWITTERAPI_IO_KEY", "")}
        slices = time_slices(query.start, query.end, DEFAULT_MAX_SLICES)

        with make_client(headers=headers) as client:
            def fetch_for(s: datetime, e: datetime) -> FetchFn:
                search = build_twitterapi_io_query(topic, s, e)

                def fetch(cursor: str | None) -> dict:
                    params = {"query": search, "queryType": "Latest", "cursor": cursor or ""}
                    data = request_json(client, "GET", TWITTERAPI_IO_URL, params=params,
                                        deadline=ctx.deadline, stop=ctx.stop)
                    if not isinstance(data, dict):
                        raise HttpFailure("unexpected non-object JSON from TwitterAPI.io")
                    if "tweets" not in data and "error" in data:
                        # Documented error shape: {"error": int, "message": str}
                        raise HttpFailure(f"TwitterAPI.io error {data.get('error')}: "
                                          f"{data.get('message')}")
                    return data
                return fetch

            def parse(data: dict, raw_ref: str, collected_at: str):
                main, emb = parse_twitterapi_io_page(data, collected_at=collected_at,
                                                     raw_ref=raw_ref, query=query.topic)
                nxt = data.get("next_cursor") if data.get("has_next_page") else None
                return main, emb, nxt or None

            posts, billed = self._collect_slices(query, ctx, status, slices, whole,
                                                 fetch_for, parse)
        status.cost_usd = round(billed * USD_PER_TWEET, 6)
        status.finished_at = now_iso()
        return CollectResult(posts=posts, status=[status])

    # ---------------------------------------------------------------- official v2

    def _collect_official(self, query: Query, ctx: CollectContext) -> CollectResult:
        now = now_utc()
        cov_start = now - OFFICIAL_WINDOW + OFFICIAL_START_MARGIN
        start = max(query.start, cov_start)
        end = min(query.end, now - OFFICIAL_END_MARGIN)
        statuses: list[SourceStatus] = []
        if query.start < cov_start:
            gap_end = min(query.end, cov_start)
            statuses.append(SourceStatus(
                source=SOURCE, query=query.topic, state=SourceState.out_of_coverage,
                date_filtered=True, started_at=now_iso(), finished_at=now_iso(),
                reason=(f"official X API recent search covers only the last 7 days; "
                        f"{iso(query.start)} to {iso(gap_end)} was not searched "
                        "(full-archive search needs a higher API tier; or use "
                        "X_PROVIDER=twitterapi_io)")))
        if end <= start:
            return CollectResult(posts=[], status=statuses)

        problem = official_bearer_problem(ctx.config.get("X_BEARER_TOKEN"))
        if problem:  # refuse before sending anything: the request could only fail
            statuses.insert(0, SourceStatus(
                source=SOURCE, query=query.topic, state=SourceState.failed,
                date_filtered=True, started_at=now_iso(), finished_at=now_iso(),
                reason=problem))
            return CollectResult(posts=[], status=statuses)
        status = SourceStatus(source=SOURCE, query=query.topic, state=SourceState.ok,
                              started_at=now_iso())
        search = search_topic(query)
        headers = {"Authorization":
                   f"Bearer {official_bearer(ctx.config.get('X_BEARER_TOKEN'))}"}
        # Slices cover only the searchable part [start, end]; the older part keeps its
        # own out_of_coverage status above.
        slices = time_slices(start, end, DEFAULT_MAX_SLICES)
        with make_client(headers=headers) as client:
            def fetch_for(s: datetime, e: datetime) -> FetchFn:
                # end_time is exclusive in v2; e is the slice's inclusive last second.
                slice_end = min(e + timedelta(seconds=1), end)

                def fetch(cursor: str | None) -> dict:
                    # Re-clamp per request: a long backoff must not push start_time past
                    # the 7-day limit (HTTP 400). Only bites after ~14 min of paging.
                    req_start = max(s, now_utc() - OFFICIAL_WINDOW + OFFICIAL_REQ_MARGIN)
                    params = dict(OFFICIAL_PARAMS, query=search,
                                  start_time=iso(req_start), end_time=iso(slice_end))
                    if cursor:
                        params["next_token"] = cursor
                    data = request_json(client, "GET", OFFICIAL_URL, params=params,
                                        deadline=ctx.deadline, stop=ctx.stop)
                    if not isinstance(data, dict):
                        raise HttpFailure("unexpected non-object JSON from X API")
                    if "data" not in data and data.get("errors") and not data.get("meta"):
                        raise HttpFailure(f"X API errors: {str(data['errors'])[:300]}")
                    return data
                return fetch

            def parse(data: dict, raw_ref: str, collected_at: str):
                main, emb = parse_official_page(data, collected_at=collected_at,
                                                raw_ref=raw_ref, query=query.topic)
                return main, emb, (data.get("meta") or {}).get("next_token") or None

            posts, _ = self._collect_slices(query, ctx, status, slices, search,
                                            fetch_for, parse)
        # Official API pricing depends on the account tier; no per-tweet meter.
        status.finished_at = now_iso()
        statuses.insert(0, status)
        return CollectResult(posts=posts, status=statuses)
