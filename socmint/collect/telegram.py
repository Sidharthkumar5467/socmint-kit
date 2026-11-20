"""Telegram collector (docs/SPEC.md §2, §5.1, §9).

Two transports, chosen automatically (the SourceStatus reason always says which ran):

A) API mode — Telethon user session. Needs TELEGRAM_API_ID + TELEGRAM_API_HASH and an
   authorized session file at ~/.socmint/sessions/<TELEGRAM_SESSION> (created by the GUI
   login, see socmint.collect.telegram_auth). The steps run in this order (owner decision
   2026-11-20): channel discovery and underground discovery find channels, the targeted
   per-channel date-bounded search collects from all the seed/discovered/underground
   channels, then the joined-chats search, and finally the global search with whatever cap
   is left — so the noisy, undated global search can no longer fill the whole post cap and
   starve the targeted channels (which is what happened when it ran first).
   * Channel discovery (owner request 2026-11-20) when TELEGRAM_DISCOVER_CHANNELS=true:
     contacts.search finds public channels/supergroups whose name or username matches the
     topic, plus channels.getChannelRecommendations for the best matches; each found
     public channel is then searched like a seed channel (date-bounded, below). Capped by
     TELEGRAM_DISCOVER_MAX. The discovery status lists every channel it picked.
   * Underground discovery (owner request 2026-11-20) when
     TELEGRAM_UNDERGROUND_DISCOVERY=true: known data-leak / infostealer / breach channels
     (collect.underground, bundled from deepdarkCTI) plus newly-found ones (leak vocabulary
     in the directory) are searched for the topic by date - a leak about the subject, not
     general chatter. Capped by TELEGRAM_UNDERGROUND_MAX. Passwords in the posts are masked
     (util.pan.mask_credentials).
   * Per-channel, date-bounded search over seed channels (TELEGRAM_SEED_CHANNELS plus
     query.options["telegram_channels"]) and every discovered/underground channel. Public
     channels are read without joining.
   * Search of the groups and channels the account has joined (messages.searchGlobal,
     owner request 2026-11-20) when TELEGRAM_SEARCH_JOINED_CHATS=true: server-side date
     bounds (min_date/max_date), one pass per time slice, newest slice first, run once
     with groups_only and once with broadcasts_only. One-to-one private chats are never
     searched. Private groups/channels get their members-only t.me/c/<id>/<msg> link.
   * Global post search (channels.searchPosts) LAST, when TELEGRAM_HAS_PREMIUM=true and the
     installed Telethon layer exposes a `query` parameter (layer 233 does). The free quota
     is checked first with channels.checkSearchPostsFlood; we NEVER pay Stars. It keeps at
     most the cap left after the targeted channels above.

B) Web-preview mode — no credentials. For each seed channel, fetch the public channel
   search preview https://t.me/s/<channel>?q=<topic> and page backwards with
   ?before=<oldest id>. Parsed from the HTML structure verified against a live page
   saved in tests/fixtures/telegram/live_preview_*.html.

C) Neither credentials nor seed channels -> skipped_no_credentials with an explanation.

Spreading the post cap across the window (socmint.collect.window; seen live 2026-11-19:
300 newest-first posts covered only the last 6 hours of a 7-day window):
* Per-channel API search is date-bounded (offset_date): one search per day slice, newest
  first, each slice limited to its share of the channel's cap; unused share rolls to older
  slices. A slice already known to be empty (the previous search ran past it) is skipped
  without a request. If cap is left after the last slice (older days were quiet), the
  slices that filled their share are searched again from where they stopped, sharing the
  leftover, so a quiet older day never wastes cap a busy newer day could use.
* Global search and the web preview have no date filter: results are walked newest first
  and every in-window candidate is bucketed by day slice (deduplicated by post id). Only
  after the walk is the cap split across the buckets (window.fill_slices: equal shares,
  unused share passes to days that have more), keeping the NEWEST posts inside each day.
  Seen live 2026-11-19: deciding a day's share while walking (300/7 = 43) kept 43 posts
  and wasted 257 slots on days the 10-page walk never reached. The walk stops early only
  when it passes the window start, the search ends, a page/time/flood limit hits, or every
  day already holds at least its equal share of candidates (then more results cannot
  change the split). The reason states how far back the walk reached and which days it
  never reached.
* Several seed channels share the source cap: each channel gets the remaining cap divided
  by the channels left (rounded up), so one busy channel cannot starve the others and a
  quiet channel's unused share rolls to the next one.
* Telegram has no documented phrase search: the plain topic is sent, and in phrase mode
  (the default) only posts containing the exact phrase are kept; the rest are counted in
  the reason.

Evidence rules encoded here:
* published_at comes only from the platform (Telethon `message.date`, or the
  `<time datetime>` attribute the web preview renders). A post without one is kept but
  flagged timestamp_verified=False.
* Forward/reply/link relations are DIRECT relations read from platform data.
* Every channel × transport gets its own SourceStatus so an empty or failed channel is
  never mistaken for "no posts".
"""

from __future__ import annotations

import asyncio
import inspect
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from socmint.collect import underground as ug
from socmint.collect.window import (
    MATCH_PHRASE,
    SliceShare,
    fill_slices,
    is_search_expression,
    match_mode,
    time_slices,
    topic_matches,
)
from socmint.config import has, home_dir, redact
from socmint.interfaces import Capabilities, CollectContext, CollectResult, Query, skipped
from socmint.schema import Media, Metrics, Post, Relation, SourceState, SourceStatus
from socmint.util.http import HttpFailure, make_client, request
from socmint.util.log import get_logger
from socmint.util.timeutil import iso, now_iso, parse_ts

log = get_logger(__name__)

SOURCE = "telegram"
WEB_BASE = "https://t.me/s/"
WEB_MAX_PAGES_PER_CHANNEL = 15  # ~20 messages per preview page
WEB_DELAY = (1.0, 2.5)  # polite pause between preview pages (seconds)
API_PAGE = 100
GLOBAL_MAX_PAGES = 10
# messages.searchGlobal (joined chats): pages per time slice and scope (groups / channels).
JOINED_MAX_PAGES_PER_SLICE = 5
# Channel discovery: public channels found per run (TELEGRAM_DISCOVER_MAX overrides),
# directory results asked for, and how many top matches get "similar channels" lookups.
DISCOVER_MAX = 15
DISCOVER_SEARCH_LIMIT = 50
DISCOVER_RECOMMEND_FROM = 2
# Underground discovery: known + newly-found leak channels searched for the topic per run.
UNDERGROUND_MAX = 20
FLOOD_RETRY_MAX_SECONDS = 60  # longer flood waits stop the account -> partial

# Telegram public usernames: letters, digits, underscore (5–32 chars; allow a little slack).
_USERNAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{2,63}$")
# t.me/<user>/<id> post links (also t.me/s/<user>/<id>); t.me/c/<id>/<n> is private.
_TME_POST_RE = re.compile(r"^/(?:s/)?([A-Za-z][A-Za-z0-9_]{2,63})/(\d+)/?$")
_TME_USER_RE = re.compile(r"^/(?:s/)?([A-Za-z][A-Za-z0-9_]{2,63})/?$")
_BG_URL_RE = re.compile(r"background-image:\s*url\(['\"]?([^'\")]+)['\"]?\)")
_TME_RESERVED = {"s", "c", "joinchat", "addstickers", "share", "proxy", "iv", "login"}

Mode = Literal["api", "web", "none"]


# ============================================================================ helpers


def parse_count(text: str | None) -> tuple[int | None, bool]:
    """Parse a Telegram display count ("987", "1,234", "12.6K", "7.24M").

    Returns (value, approximate). The web preview rounds large values, so a K/M/B
    suffix yields an approximate integer and approximate=True.
    """
    if not text:
        return None, False
    s = text.strip().replace(",", "").replace(" ", "").replace(" ", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KMB]?)", s, flags=re.IGNORECASE)
    if not m:
        return None, False
    num, suffix = float(m.group(1)), m.group(2).upper()
    mult = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
    return int(round(num * mult)), bool(suffix)


def normalize_channel(value: str | None) -> str | None:
    """'@Durov', 'durov', 'https://t.me/durov', 't.me/s/durov' -> 'durov' (lowercase)."""
    if not value:
        return None
    s = value.strip()
    if not s:
        return None
    if "t.me/" in s or "telegram.me/" in s:
        path = urlparse(s if "://" in s else "https://" + s).path
        parts = [p for p in path.split("/") if p]
        if parts and parts[0] == "s":
            parts = parts[1:]
        s = parts[0] if parts else ""
    s = s.lstrip("@")
    if not _USERNAME_RE.match(s) or s.lower() in _TME_RESERVED:
        return None
    return s.lower()


def seed_channels(cfg: dict[str, str], query: Query) -> list[str]:
    """Seed channels from config + per-run options, normalized and de-duplicated."""
    raw: list[str] = []
    raw.extend((cfg.get("TELEGRAM_SEED_CHANNELS") or "").replace("\n", ",").split(","))
    extra = query.options.get("telegram_channels") or []
    if isinstance(extra, str):
        extra = extra.replace("\n", ",").split(",")
    raw.extend(str(x) for x in extra)
    out: list[str] = []
    for item in raw:
        ch = normalize_channel(item)
        if ch and ch not in out:
            out.append(ch)
    return out


def session_path(cfg: dict[str, str]) -> Path:
    """The Telethon session file. Lives in ~/.socmint/sessions/, never in the repo."""
    name = Path(cfg.get("TELEGRAM_SESSION") or "socmint.session").name or "socmint.session"
    if not name.endswith(".session"):
        name += ".session"
    return home_dir() / "sessions" / name


def is_true(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def select_mode(cfg: dict[str, str], seeds: list[str]) -> tuple[Mode, str]:
    """Pick the transport. Returns (mode, human-readable reason)."""
    has_keys = has(cfg, "TELEGRAM_API_ID", "TELEGRAM_API_HASH")
    has_session = session_path(cfg).is_file()
    if has_keys and has_session:
        return "api", "API mode (Telethon user session)"
    if seeds:
        why = ("API keys set but no session file - log in from Settings"
               if has_keys else "no Telegram API credentials")
        return "web", f"web-preview mode ({why})"
    return "none", ("no Telegram credentials and no seed channels (TELEGRAM_SEED_CHANNELS); "
                    "topic search needs an authorized session (Premium for global search)")


def _plain_topic(query: Query) -> str:
    """Topic sent to Telegram: whitespace-normalised, never quoted (no phrase search)."""
    return " ".join(query.topic.split())


def _phrase_filtering(query: Query) -> bool:
    """True when posts must be checked client-side for the exact phrase."""
    t = _plain_topic(query)
    return match_mode(query) == MATCH_PHRASE and " " in t and not is_search_expression(t)


def _phrase_note(n: int) -> str:
    return f"{n} result(s) without the exact phrase were not kept (topic match: phrase)"


def _split(remaining: int, parts_left: int) -> int:
    """Share of a remaining cap for the next of `parts_left` parts (rounded up)."""
    return max(0, -(-max(0, remaining) // max(1, parts_left)))


def _unreached(query: Query) -> str:
    """Per-day summary for a status that never searched any slice (all 'not reached')."""
    slices = time_slices(query.start, query.end)
    return SliceShare(query.limit, len(slices)).summary(slices) if slices else ""


def _with_summary(reason: str, query: Query) -> str:
    summ = _unreached(query)
    return f"{reason}; {summ}" if summ else reason


def _flood_text(e: Any, state: Any) -> str:
    """Stop reason for a FloodWait that ended a search. A wait cut short by the analyst
    (_flood_retry marks it) is reported as such and does not block the account."""
    if getattr(e, "socmint_stopped", False):
        return "stopped by the analyst during a flood wait"
    state.flood_blocked = int(getattr(e, "seconds", 0) or 0)
    return f"flood wait {state.flood_blocked}s"


def _first_day(query: Query) -> str:
    """First date of the newest slice (prefix for 'not reached' cursors)."""
    slices = time_slices(query.start, query.end)
    return f"{slices[0][0]:%Y-%m-%d}" if slices else f"{query.start:%Y-%m-%d}"


class _SliceWalk:
    """Client-side cap balancing for a search with no date filter, walked newest first.

    During the walk every in-window, phrase-matching result is kept as a CANDIDATE in its
    day slice (deduplicated by post id; out-of-order results land in their own slice).
    After the walk, select() splits the cap across the slices with window.fill_slices and
    keeps the newest candidates inside each slice: newest because that is the order the
    platform returned them in, so a day the walk only partly covered is represented by the
    same posts a resumed walk would start from. Posts without a platform timestamp belong
    to no slice; they are kept as unverified only with cap the dated posts left unused.
    """

    def __init__(self, query: Query, total: int):
        self.query = query
        self.total = max(0, int(total))
        self.slices = time_slices(query.start, query.end)
        self.n = len(self.slices)
        # Per slice: (published_at, walk order, item).
        self.buckets: list[list[tuple[datetime, int, Any]]] = [[] for _ in self.slices]
        self.undated: list[tuple[int, Any]] = []
        self.seen: set[str] = set()
        self.order = 0
        self.duplicates = 0
        self.dropped_out = 0
        self.dropped_phrase = 0
        self.oldest: datetime | None = None
        self.passed_start = False
        self.exhausted = False
        # Filled by select().
        self.keep: list[int] = [0] * self.n
        self.unverified = 0
        self.undated_over_cap = 0

    def _index(self, dt: datetime) -> int | None:
        for i, (s, e) in enumerate(self.slices):
            if s <= dt <= e:
                return i
        return None

    @property
    def candidates(self) -> int:
        return sum(len(b) for b in self.buckets)

    def offer(self, key: str, dt: datetime | None, text: str | None, item: Any) -> None:
        """Record one search result (dt = platform timestamp or None)."""
        if key in self.seen:
            self.duplicates += 1
            return
        self.seen.add(key)
        self.order += 1
        if dt is None:
            if topic_matches(self.query, text):
                self.undated.append((self.order, item))
            else:
                self.dropped_phrase += 1
            return
        if self.oldest is None or dt < self.oldest:
            self.oldest = dt
        if dt < self.query.start:
            self.dropped_out += 1
            self.passed_start = True
            return
        j = self._index(dt)
        if j is None:  # newer than the window end
            self.dropped_out += 1
            return
        if not topic_matches(self.query, text):
            self.dropped_phrase += 1
            return
        self.buckets[j].append((dt, self.order, item))

    def full(self) -> bool:
        """More results cannot change what select() keeps: the cap is 0, or every slice
        already holds at least its equal share (ceil(cap / slices)) of candidates.

        Never "candidates >= cap": that would stop on the newest day and starve the rest.
        """
        if self.total <= 0:
            return True
        need = -(-self.total // max(1, self.n))
        return self.n > 0 and all(len(b) >= need for b in self.buckets)

    def finish(self, exhausted: bool) -> None:
        """The walk ended; `exhausted` = the search ran out or passed the window start, so
        every slice was covered."""
        self.exhausted = exhausted or self.passed_start or self.full()

    def reached(self, i: int) -> bool:
        if self.exhausted:
            return True
        return self.oldest is not None and self.oldest <= self.slices[i][1]

    def all_reached(self) -> bool:
        return all(self.reached(i) for i in range(self.n))

    def select(self) -> list[Any]:
        """Items to keep (sum <= cap), in walk order."""
        self.keep = fill_slices([len(b) for b in self.buckets], self.total)
        chosen: list[tuple[int, Any]] = []
        for i, b in enumerate(self.buckets):
            newest = sorted(b, key=lambda t: (-t[0].timestamp(), t[1]))[: self.keep[i]]
            chosen.extend((order, item) for _, order, item in newest)
        room = max(0, self.total - len(chosen))
        self.unverified = min(room, len(self.undated))
        self.undated_over_cap = len(self.undated) - self.unverified
        chosen.extend(self.undated[: self.unverified])
        return [item for _, item in sorted(chosen, key=lambda t: t[0])]

    def cursor(self, c: str) -> str:
        """Prefix a resume cursor with the first date of the oldest slice reached."""
        if not self.slices:
            return c
        idx = max([i for i in range(self.n) if self.reached(i)], default=0)
        return f"{self.slices[idx][0]:%Y-%m-%d}:{c}"

    def notes(self, what: str) -> list[str]:
        reach = (f"reached back to {self.oldest:%Y-%m-%d %H:%M} UTC" if self.oldest
                 else "reached no dated result")
        out = [f"{what} has no date filter; {reach}"]
        if self.dropped_out:
            out.append(f"{self.dropped_out} result(s) outside the window dropped")
        if self.dropped_phrase:
            out.append(_phrase_note(self.dropped_phrase))
        if self.duplicates:
            out.append(f"{self.duplicates} duplicate result(s) ignored")
        cands = self.candidates
        if cands > sum(self.keep):
            out.append(f"{cands} in-window candidate(s); the cap of {self.total} was split "
                       "across the days reached, keeping the newest posts within each day")
        if self.unverified:
            out.append(f"{self.unverified} without a platform timestamp kept as unverified")
        if self.undated_over_cap:
            out.append(f"{self.undated_over_cap} without a platform timestamp not kept "
                       "(cap used by dated posts)")
        return out

    def summary(self) -> str:
        """Per-day summary from the final keep counts; a day's share is its equal part of
        the cap among the days reached."""
        share = SliceShare(self.total, self.n)
        reached = [i for i in range(self.n) if self.reached(i)]
        fair = fill_slices([self.total if i in reached else 0 for i in range(self.n)],
                           self.total)
        for i in reached:
            share.cap[i] = fair[i]
            share.record(i, self.keep[i])
        return share.summary(self.slices)


def _tme_relation(href: str, rel_type: str) -> Relation | None:
    """Turn a t.me link into a direct relation; other http(s) links become 'link'."""
    if not href or not href.startswith(("http://", "https://")):
        return None
    u = urlparse(href)
    host = (u.hostname or "").lower()
    if host in {"t.me", "telegram.me", "www.t.me"}:
        m = _TME_POST_RE.match(u.path)
        if m and m.group(1).lower() not in _TME_RESERVED:
            ch, mid = m.group(1).lower(), m.group(2)
            return Relation(type=rel_type, target_post_id=f"{SOURCE}:{ch}:{mid}",  # type: ignore[arg-type]
                            target_url=f"https://t.me/{ch}/{mid}", target_handle=f"@{ch}")
        m = _TME_USER_RE.match(u.path)
        if m and m.group(1).lower() not in _TME_RESERVED:
            ch = m.group(1).lower()
            # A bare channel link is a mention when it is a plain link, but a forward
            # header pointing at a user/channel without a post id keeps its type.
            t = "mention" if rel_type == "link" else rel_type
            return Relation(type=t, target_url=href, target_handle=f"@{ch}")  # type: ignore[arg-type]
    return Relation(type=rel_type, target_url=href)  # type: ignore[arg-type]


def _dedupe_relations(rels: list[Relation]) -> list[Relation]:
    seen: set[tuple] = set()
    out: list[Relation] = []
    for r in rels:
        k = (r.type, r.target_post_id, r.target_url, r.target_handle)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


# ===================================================================== web preview


@dataclass
class PreviewPage:
    posts: list[Post] = field(default_factory=list)
    oldest_id: int | None = None
    channel_found: bool = False  # page rendered a channel header (channel is public)
    channel_title: str | None = None
    approximate_counts: bool = False  # views/reactions were rounded display values
    unverified: int = 0  # posts without a parseable platform timestamp


def _bg_url(tag: Tag | None) -> str | None:
    if tag is None:
        return None
    m = _BG_URL_RE.search(tag.get("style") or "")
    if not m:
        return None
    url = m.group(1)
    return "https:" + url if url.startswith("//") else url


def _message_text(msg: Tag) -> Tag | None:
    """The message's own text block. Reply previews contain a nested
    .tgme_widget_message_text (the quoted parent) that must not be used."""
    for tx in msg.select("div.tgme_widget_message_text"):
        if tx.find_parent(class_="tgme_widget_message_reply") is None:
            return tx
    return None


def _text_of(tx: Tag | None) -> str:
    if tx is None:
        return ""
    # Work on a copy so <br> substitution doesn't alter the soup used for links.
    tx = BeautifulSoup(str(tx), "html.parser")
    for br in tx.find_all("br"):
        br.replace_with("\n")
    return tx.get_text("").strip()


def parse_preview(
    html: str,
    *,
    channel: str,
    collected_at: str,
    topic: str | None = None,
    raw_ref: str | None = None,
) -> PreviewPage:
    """Parse one t.me/s/<channel> page (optionally a ?q= search page) into Posts."""
    soup = BeautifulSoup(html, "html.parser")
    page = PreviewPage()
    header = soup.select_one(".tgme_channel_info_header_title")
    page.channel_found = header is not None
    page.channel_title = header.get_text(" ", strip=True) if header else None

    for msg in soup.select("div.tgme_widget_message[data-post]"):
        data_post = msg.get("data-post") or ""
        m = re.match(r"^([A-Za-z0-9_]+)/(\d+)$", data_post)
        if not m:
            continue
        handle, mid = m.group(1).lower(), int(m.group(2))
        page.oldest_id = mid if page.oldest_id is None else min(page.oldest_id, mid)

        # Timestamp: ONLY the datetime attribute inside the message's date link. Other
        # <time> tags (e.g. video duration) carry no datetime and must be ignored.
        t = msg.select_one("a.tgme_widget_message_date time[datetime]")
        pub = parse_ts(t.get("datetime")) if t else None
        note = None
        if pub is None:
            page.unverified += 1
            note = "web preview rendered no timezone-aware datetime for this message"

        owner = msg.select_one(".tgme_widget_message_owner_name")
        author_name = owner.get_text(" ", strip=True) if owner else page.channel_title
        avatar = msg.select_one(".tgme_widget_message_user_photo img[src]")

        tx = _message_text(msg)
        text = _text_of(tx)

        # ---- metrics (display values; K/M suffixes are rounded)
        views, approx_v = parse_count(_txt(msg.select_one(".tgme_widget_message_views")))
        reactions_total: int | None = None
        approx_r = False
        for r in msg.select(".tgme_widget_message_reactions .tgme_reaction"):
            # Paid reactions (Stars) count Stars spent, not reacting users; summing them
            # would inflate engagement by orders of magnitude (live: durov/564 "2.08M").
            if "tgme_reaction_paid" in (r.get("class") or []):
                continue
            n, a = parse_count(r.get_text("", strip=True))
            if n is not None:
                reactions_total = (reactions_total or 0) + n
                approx_r = approx_r or a
        if approx_v or approx_r:
            page.approximate_counts = True

        # ---- media
        media: list[Media] = []
        for ph in msg.select("a.tgme_widget_message_photo_wrap"):
            u = _bg_url(ph)
            media.append(Media(type="photo", url=u, thumbnail=u))
        for vp in msg.select(".tgme_widget_message_video_player"):
            v = vp.select_one("video[src]")
            thumb = _bg_url(vp.select_one(".tgme_widget_message_video_thumb"))
            media.append(Media(type="video", url=v.get("src") if v else None, thumbnail=thumb))
        for _doc in msg.select(".tgme_widget_message_document"):
            media.append(Media(type="document"))

        # ---- direct relations
        rels: list[Relation] = []
        fwd = msg.select_one("a.tgme_widget_message_forwarded_from_name[href]")
        if fwd is not None:
            r = _tme_relation(fwd["href"], "forward")
            if r:
                rels.append(r)
        elif msg.select_one(".tgme_widget_message_forwarded_from") is not None:
            # Forwarded from a hidden/private sender: the fact of forwarding is still
            # direct evidence, the source just isn't linkable.
            rels.append(Relation(type="forward"))
        rep = msg.select_one("a.tgme_widget_message_reply[href]")
        if rep is not None:
            r = _tme_relation(rep["href"], "reply")
            if r:
                rels.append(r)
        if tx is not None:
            for a in tx.select("a[href]"):
                r = _tme_relation(a["href"], "link")
                if r:
                    rels.append(r)
        lp = msg.select_one("a.tgme_widget_message_link_preview[href]")
        if lp is not None:
            r = _tme_relation(lp["href"], "link")
            if r:
                rels.append(r)

        page.posts.append(Post(
            post_id=f"{SOURCE}:{handle}:{mid}",
            source=SOURCE,
            kind="post",
            # Web preview does not expose the numeric channel id; the public username
            # is the stable identifier in this mode (see DECISIONS).
            author_id=handle,
            author_handle=f"@{handle}",
            author_name=author_name,
            author_avatar_url=avatar.get("src") if avatar else None,
            text=text,
            published_at=iso(pub),
            collected_at=collected_at,
            timestamp_verified=pub is not None,
            timestamp_note=note,
            metrics=Metrics(views=views, reactions=reactions_total),
            media=media,
            url=f"https://t.me/{handle}/{mid}",
            relations=_dedupe_relations(rels),
            raw_ref=raw_ref,
            query=topic,
        ))
    return page


def _txt(tag: Tag | None) -> str | None:
    return tag.get_text("", strip=True) if tag is not None else None


# ===================================================================== API mapping


def _utf16_slice(text: str, offset: int, length: int) -> str:
    """Telegram entity offsets are in UTF-16 code units, not Python characters."""
    b = text.encode("utf-16-le")
    return b[offset * 2:(offset + length) * 2].decode("utf-16-le", errors="ignore")


def _peer(peer: Any) -> tuple[str, int] | None:
    """Duck-typed Telethon Peer -> (kind, id)."""
    if peer is None:
        return None
    for attr, kind in (("channel_id", "channel"), ("user_id", "user"), ("chat_id", "chat")):
        v = getattr(peer, attr, None)
        if v is not None:
            return kind, int(v)
    return None


def _entity_kind(e: Any) -> str:
    if hasattr(e, "first_name"):
        return "user"
    if hasattr(e, "broadcast") or hasattr(e, "megagroup"):
        return "channel"
    return "chat"


def entity_lookup(*entities: Any) -> dict[tuple[str, int], Any]:
    """Index chats/users (e.g. a response's .chats + .users) by (kind, id)."""
    out: dict[tuple[str, int], Any] = {}
    for e in entities:
        if e is not None and getattr(e, "id", None) is not None:
            out[(_entity_kind(e), int(e.id))] = e
    return out


def public_handle(e: Any) -> str | None:
    """The entity's public username (original case), or None if it has none.

    Channel/User.username is None when every public name is a collectible (Fragment)
    username or the main one lives only in `usernames`; fall back to the first active
    entry of `usernames` so such channels still get t.me links and username-keyed ids.
    """
    if e is None:
        return None
    u = getattr(e, "username", None)
    if u:
        return str(u)
    for un in getattr(e, "usernames", None) or []:
        name = getattr(un, "username", None)
        if name and getattr(un, "active", False):
            return str(name)
    return None


def _chat_key(chat: Any, handle: str | None = None) -> str:
    """The <channel> part of a post id: lowercase username if public, else numeric id."""
    u = handle or public_handle(chat)
    return u.lower() if u else str(getattr(chat, "id", "unknown"))


def _display_name(e: Any) -> str | None:
    title = getattr(e, "title", None)
    if title:
        return title
    parts = [getattr(e, "first_name", None), getattr(e, "last_name", None)]
    name = " ".join(p for p in parts if p)
    return name or None


def _creation_date(chat: Any) -> str | None:
    """Channel.date is the creation date ONLY when the session user is not a member
    (`left=True`); for members it is the join date. Use it only in the former case.
    # UNVERIFIED: semantics from the TL schema docs for Channel.date, not a live test.
    """
    d = getattr(chat, "date", None)
    if isinstance(d, datetime) and getattr(chat, "left", None) is True and d.tzinfo:
        return iso(d)
    return None


def _ref_target(kind_id: tuple[str, int] | None, msg_id: int | None,
                lookup: dict[tuple[str, int], Any]) -> dict[str, Any]:
    """Build target fields for a relation pointing at (peer, msg_id)."""
    if kind_id is None:
        return {}
    ent = lookup.get(kind_id)
    uname = public_handle(ent)
    key = uname.lower() if uname else str(kind_id[1])
    out: dict[str, Any] = {"target_handle": f"@{uname.lower()}" if uname else None}
    if msg_id and kind_id[0] in ("channel", "chat"):
        out["target_post_id"] = f"{SOURCE}:{key}:{msg_id}"
        if uname:
            out["target_url"] = f"https://t.me/{uname.lower()}/{msg_id}"
    return out


def message_to_post(
    m: Any,
    *,
    chat: Any,
    collected_at: str,
    topic: str | None = None,
    raw_ref: str | None = None,
    lookup: dict[tuple[str, int], Any] | None = None,
    handle: str | None = None,
) -> Post | None:
    """Map a Telethon Message (or a duck-typed equivalent) to a Post.

    `chat` is the channel/group the message lives in; `lookup` resolves forward/reply
    peers and senders from the response's chats/users (no extra API calls). `handle`
    is a public username already known to resolve to `chat` (the seed channel name),
    used only when the entity itself exposes no public username.
    """
    if m is None or getattr(m, "id", None) is None:
        return None
    lookup = dict(lookup or {})
    lookup.update(entity_lookup(chat))
    fwd_obj = getattr(m, "forward", None)  # Telethon's custom Forward (has resolved .chat)
    if fwd_obj is not None:
        lookup.update(entity_lookup(getattr(fwd_obj, "chat", None),
                                    getattr(fwd_obj, "sender", None)))

    uname = public_handle(chat) or handle
    key = _chat_key(chat, uname)
    text = getattr(m, "message", None) or ""

    # ---- author: channel posts are authored by the channel; group messages by the sender
    from_peer = _peer(getattr(m, "from_id", None))
    is_channel_post = bool(getattr(m, "post", False)) or from_peer is None
    if is_channel_post:
        author = chat
    else:
        author = getattr(m, "sender", None) or lookup.get(from_peer)  # type: ignore[arg-type]
    a_uname = uname if author is chat else public_handle(author)
    if author is not None and getattr(author, "id", None) is not None:
        author_id: str | None = str(author.id)
    else:
        author_id = str(from_peer[1]) if from_peer else None

    # ---- timestamp (platform field only)
    date = getattr(m, "date", None)
    pub = date if isinstance(date, datetime) and date.tzinfo else None

    # ---- metrics
    replies = getattr(m, "replies", None)
    reactions = getattr(m, "reactions", None)
    reaction_total = None
    if reactions is not None and getattr(reactions, "results", None) is not None:
        # Skip ReactionPaid counts: they are Stars spent, not reactions (see web parser).
        reaction_total = sum(int(getattr(r, "count", 0) or 0) for r in reactions.results
                             if type(getattr(r, "reaction", None)).__name__ != "ReactionPaid")
    metrics = Metrics(
        views=getattr(m, "views", None),
        forwards=getattr(m, "forwards", None),
        replies=getattr(replies, "replies", None) if replies is not None else None,
        reactions=reaction_total,
    )

    # ---- media (no public file URLs via MTProto; the post URL is the public link)
    media: list[Media] = []
    rels: list[Relation] = []
    md = getattr(m, "media", None)
    if md is not None:
        if getattr(md, "photo", None) is not None:
            media.append(Media(type="photo"))
        elif getattr(md, "document", None) is not None:
            mime = str(getattr(md.document, "mime_type", "") or "")
            mtype = "video" if mime.startswith("video/") else (
                "image" if mime.startswith("image/") else "document")
            media.append(Media(type=mtype))  # type: ignore[arg-type]
        elif getattr(md, "webpage", None) is not None:
            wurl = getattr(md.webpage, "url", None)
            if wurl:
                r = _tme_relation(wurl, "link")
                if r:
                    rels.append(r)
        else:
            media.append(Media(type="other"))

    # ---- forward (direct)
    fwd = getattr(m, "fwd_from", None)
    if fwd is not None:
        src = _peer(getattr(fwd, "from_id", None))
        # A hidden sender (fwd.from_name only) still yields a forward relation with no
        # target: the display name is not a verifiable handle.
        tgt = _ref_target(src, getattr(fwd, "channel_post", None), lookup)
        rels.append(Relation(type="forward", **{k: v for k, v in tgt.items() if v}))

    # ---- reply (direct)
    rt = getattr(m, "reply_to", None)
    if rt is not None and getattr(rt, "reply_to_msg_id", None):
        rpeer = _peer(getattr(rt, "reply_to_peer_id", None)) or _peer(getattr(m, "peer_id", None))
        if rpeer is None:
            rpeer = (_entity_kind(chat), int(getattr(chat, "id", 0)))
        tgt = _ref_target(rpeer, rt.reply_to_msg_id, lookup)
        rels.append(Relation(type="reply", **{k: v for k, v in tgt.items() if v}))

    # ---- links from entities (UTF-16 offsets)
    for ent in getattr(m, "entities", None) or []:
        cls = type(ent).__name__
        url = getattr(ent, "url", None)
        if url is None and cls == "MessageEntityUrl":
            url = _utf16_slice(text, int(ent.offset), int(ent.length))
        if url:
            if not url.startswith(("http://", "https://")) and "." in url:
                url = "https://" + url
            r = _tme_relation(url, "link")
            if r:
                rels.append(r)

    return Post(
        post_id=f"{SOURCE}:{key}:{m.id}",
        source=SOURCE,
        kind="post",
        author_id=author_id,
        author_handle=f"@{a_uname}" if a_uname else None,
        author_name=_display_name(author) if author is not None else None,
        author_created_at=_creation_date(author) if author is chat else None,
        text=text,
        published_at=iso(pub),
        collected_at=collected_at,
        timestamp_verified=pub is not None,
        timestamp_note=None if pub else "Telegram message carried no timezone-aware date",
        metrics=metrics,
        media=media,
        url=_message_url(chat, uname, m.id),
        relations=_dedupe_relations(rels),
        raw_ref=raw_ref,
        query=topic,
    )


def _message_url(chat: Any, uname: str | None, msg_id: int) -> str | None:
    """Public t.me/<username>/<id>; for a private channel or supergroup the members-only
    t.me/c/<channel id>/<id> link (opens for the analyst, who is a member). Legacy basic
    groups have no message links (None)."""
    if uname:
        return f"https://t.me/{uname.lower()}/{msg_id}"
    cid = getattr(chat, "id", None)
    if cid is not None and (getattr(chat, "megagroup", False) or getattr(chat, "broadcast", False)):
        return f"https://t.me/c/{int(cid)}/{msg_id}"
    return None


def _to_raw(m: Any) -> Any:
    to_dict = getattr(m, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # noqa: BLE001
            pass
    return {"id": getattr(m, "id", None), "date": str(getattr(m, "date", None)),
            "message": getattr(m, "message", None)}


# ======================================================================= collector


def _flood_error_cls() -> type[BaseException]:
    try:
        from telethon.errors import FloodWaitError
        return FloodWaitError
    except Exception:  # noqa: BLE001 — telethon missing: nothing can raise it
        class _Never(Exception):
            seconds = 0
        return _Never


@dataclass
class _ApiState:
    """Shared across one API run: the account is used strictly sequentially."""

    flood_blocked: int | None = None  # seconds of a flood wait that stopped the account
    requests: int = 0
    collected: int = 0
    # Public channels/supergroups seen in global-search results (in or out of the window)
    # whose post matched the topic: lower username -> [matching posts, username, entity].
    # Channel discovery searches the most frequent ones by date (live 2026-11-20).
    global_channels: dict[str, list] = field(default_factory=dict)


class TelegramCollector:
    name = SOURCE

    def __init__(self, client_factory: Callable[[dict[str, str]], Any] | None = None,
                 http_client_factory: Callable[[], Any] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        # Factories are injectable so tests run with fakes and no network.
        self._client_factory = client_factory or _make_telethon_client
        self._http_client_factory = http_client_factory or (lambda: make_client(
            headers={"accept-language": "en-US,en;q=0.9"}))
        self._sleep = sleep

    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_date_filter=True,
            needs_credentials=False,  # web-preview mode works without credentials
            kind="messaging",
            credential_keys=(),
            description=("Telethon user session (per-channel search; global search with "
                         "Premium) or public t.me/s web preview for seed channels"),
        )

    # ------------------------------------------------------------------ entry point
    def collect(self, query: Query, ctx: CollectContext) -> CollectResult:
        seeds = seed_channels(ctx.config, query)
        mode, why = select_mode(ctx.config, seeds)
        if mode == "none":
            return skipped(SOURCE, query.topic, SourceState.skipped_no_credentials, why)
        if mode == "api":
            try:
                # collect() runs on a pipeline worker thread: give Telethon its own loop.
                return asyncio.run(self._collect_api(query, ctx, seeds))
            except _ApiUnavailable as e:
                if not seeds:
                    return skipped(SOURCE, query.topic, SourceState.failed,
                                   f"API mode unavailable: {e}")
                why = f"web-preview mode (API mode unavailable: {e})"
        return self._collect_web(query, ctx, seeds, why)

    # ============================================================== web preview
    def _collect_web(self, query: Query, ctx: CollectContext, seeds: list[str],
                     why: str) -> CollectResult:
        posts: list[Post] = []
        statuses: list[SourceStatus] = []
        with self._http_client_factory() as client:
            for n, ch in enumerate(seeds):
                remaining = query.limit - len(posts)
                if remaining <= 0 or ctx.out_of_time():
                    statuses.append(SourceStatus(
                        source=SOURCE, query=query.topic, state=SourceState.partial,
                        reason=_with_summary(
                            f"{why}; @{ch}: not reached ("
                            f"{'time budget exhausted' if ctx.out_of_time() else 'post limit reached'})",
                            query),
                        cursor=f"{_first_day(query)}:{ch}:start", date_filtered=False,
                        started_at=now_iso()))
                    continue
                # Channels share the cap: remaining / channels left (see module docstring).
                budget = _split(remaining, len(seeds) - n)
                got, st = self._web_channel(client, ch, query, ctx, budget, why)
                posts.extend(got)
                statuses.append(st)
        return CollectResult(posts=posts, status=statuses)

    def _web_channel(self, client: Any, ch: str, query: Query, ctx: CollectContext,
                     cap: int, why: str) -> tuple[list[Post], SourceStatus]:
        st = SourceStatus(source=SOURCE, query=query.topic, state=SourceState.ok,
                          started_at=now_iso(),
                          # The preview has no server-side date filter: we page newest ->
                          # oldest and bound by the platform timestamps client-side.
                          date_filtered=False)
        kept: list[Post] = []
        walk = _SliceWalk(query, cap)
        topic = _plain_topic(query)
        before: int | None = None
        approx = False
        ended: str | None = None  # "exhausted" | "full" | "time budget exhausted" | None
        for page_no in range(1, WEB_MAX_PAGES_PER_CHANNEL + 1):
            if ctx.out_of_time():
                ended = "time budget exhausted"
                break
            params: dict[str, Any] = {"q": topic}
            if before:
                params["before"] = before
            try:
                resp = request(client, "GET", f"{WEB_BASE}{ch}", params=params,
                               deadline=ctx.deadline, stop=ctx.stop, max_retries=2)
            except HttpFailure as e:
                st.requests += 1
                walk.finish(False)
                kept = walk.select()
                st.state = SourceState.partial if kept else SourceState.failed
                st.reason = "; ".join([f"{why}; @{ch}: {e}", *walk.notes("web preview search"),
                                       walk.summary()])
                st.cursor = walk.cursor(f"{ch}:before={before}" if before else f"{ch}:start")
                st.count = len(kept)
                return kept, st
            st.requests += 1
            # A missing/private channel (or one with the preview disabled) redirects
            # away from /s/ to the plain t.me/<name> page.
            if "/s/" not in str(resp.url):
                st.state = SourceState.out_of_coverage
                st.reason = (f"{why}; @{ch}: no public web preview (channel missing, "
                             "private, or preview disabled)")
                return [], st
            collected_at = now_iso()
            raw_ref = ctx.save_raw(SOURCE, f"web_{ch}_{page_no}.html", resp.text)
            page = parse_preview(resp.text, channel=ch, collected_at=collected_at,
                                 topic=query.topic, raw_ref=raw_ref)
            if page_no == 1 and not page.channel_found and not page.posts:
                st.state = SourceState.out_of_coverage
                st.reason = f"{why}; @{ch}: page has no channel preview"
                return [], st
            approx = approx or page.approximate_counts
            last_seen: int | None = None  # id of the last post actually examined
            for p in sorted(page.posts, key=lambda x: -int(x.post_id.rsplit(":", 1)[1])):
                if walk.full():
                    break
                last_seen = int(p.post_id.rsplit(":", 1)[1])
                # Candidates only; select() decides after the walk. A post without a
                # timestamp may be kept as unverified (excluded from timelines downstream).
                walk.offer(p.post_id, parse_ts(p.published_at), p.text, p)
            ctx.progress(SOURCE, {"message": f"telegram: @{ch} page {page_no}",
                                  "count": min(walk.candidates, walk.total)})
            if walk.full():
                ended = "full"
                # Resume just below the last post examined, NOT the page's oldest id:
                # the rest of this page was never looked at.
                before = last_seen if last_seen is not None else before
                break
            if (not page.posts or page.oldest_id is None or page.oldest_id <= 1
                    or walk.passed_start):
                ended = "exhausted"
                break
            before = page.oldest_id
            self._sleep(random.uniform(*WEB_DELAY))

        walk.finish(ended == "exhausted")
        kept = walk.select()
        st.count = len(kept)
        notes = [f"{why}; @{ch}: t.me/s search preview, paged newest->oldest, "
                 "spread across days by platform timestamps",
                 *walk.notes("web preview search")]
        if approx:
            notes.append("views/reactions are rounded display values (e.g. 1.2K)")
        if ended in ("exhausted", "full"):
            st.state = SourceState.ok if kept else SourceState.completed_zero
            if ended == "full":
                notes.append("stopped early: every day already had its share of candidates")
        else:
            st.state = SourceState.partial
            st.cursor = walk.cursor(f"{ch}:before={before}" if before else f"{ch}:start")
            stop = ended or f"page cap ({WEB_MAX_PAGES_PER_CHANNEL}) reached"
            notes.append(f"{stop} before the window start")
        notes.append(walk.summary())
        st.reason = "; ".join(notes)
        return kept, st

    # ===================================================================== API
    async def _collect_api(self, query: Query, ctx: CollectContext,
                           seeds: list[str]) -> CollectResult:
        try:
            # A malformed TELEGRAM_API_ID (int() fails) must fall back to web mode too.
            client = self._client_factory(ctx.config)
        except Exception as e:  # noqa: BLE001
            raise _ApiUnavailable(redact(f"client setup failed: {type(e).__name__}: {e}")) from e
        try:
            try:
                await client.connect()
                authorized = await client.is_user_authorized()
            except Exception as e:  # noqa: BLE001
                raise _ApiUnavailable(redact(f"connect failed: {type(e).__name__}: {e}")) from e
            if not authorized:
                raise _ApiUnavailable("session file is not authorized - log in from Settings")

            state = _ApiState()
            posts: list[Post] = []
            statuses: list[SourceStatus] = []
            premium = is_true(ctx.config.get("TELEGRAM_HAS_PREMIUM"))
            joined = is_true(ctx.config.get("TELEGRAM_SEARCH_JOINED_CHATS"))

            # Order (owner decision 2026-11-20): the TARGETED per-channel searches — seed
            # channels, discovered channels and underground/leak channels — run FIRST and
            # get budget priority; the noisy, undated global search runs LAST with only the
            # leftover cap, so it can no longer starve the targeted channels (which is what
            # happened when global search ran first and filled the whole post cap).

            # ---- channel discovery (owner request 2026-11-20): public channels whose name
            # matches the topic, searched by date like seed channels. Collector default off;
            # config.DEFAULTS turns it on for the app. Global-search results are no longer a
            # discovery seed (global runs last now); discovery uses the directory + the known
            # list, which degrade gracefully when state.global_channels is empty.
            discovered: list[tuple[str, Any]] = []
            if is_true(ctx.config.get("TELEGRAM_DISCOVER_CHANNELS")) and \
                    state.flood_blocked is None and not ctx.out_of_time():
                try:
                    cap_n = int(ctx.config.get("TELEGRAM_DISCOVER_MAX") or DISCOVER_MAX)
                except ValueError:
                    cap_n = DISCOVER_MAX
                discovered, dst = await self._discover(
                    client, query, ctx, state, {c.lower() for c in seeds},
                    max(1, min(cap_n, 50)), skip_joined=joined)
                statuses.append(dst)
            entities = {u.lower(): e for u, e in discovered}
            seeds = seeds + [u for u, _ in discovered]

            # ---- underground / leak channel discovery (owner request 2026-11-20): known
            # and newly-found data-leak / infostealer / breach channels, searched for the
            # topic by date. A hit is a leak about the subject, not general chatter.
            underground: list[tuple[str, Any, str]] = []
            if is_true(ctx.config.get("TELEGRAM_UNDERGROUND_DISCOVERY")) and \
                    state.flood_blocked is None and not ctx.out_of_time():
                try:
                    ug_cap = int(ctx.config.get("TELEGRAM_UNDERGROUND_MAX") or UNDERGROUND_MAX)
                except ValueError:
                    ug_cap = UNDERGROUND_MAX
                underground, ust = await self._discover_underground(
                    client, query, ctx, state, {c.lower() for c in seeds},
                    max(1, min(ug_cap, 60)), skip_joined=joined)
                statuses.append(ust)
                for u, e, _how in underground:
                    entities[u.lower()] = e
                found_how = {u.lower(): how for u, _e, how in underground}
                seeds = seeds + [u for u, _e, _how in underground]
            else:
                found_how = {}

            if not seeds and not premium and not joined:
                statuses.append(SourceStatus(
                    source=SOURCE, query=query.topic,
                    state=SourceState.skipped_capability, started_at=now_iso(),
                    reason=("API mode: global post search needs Telegram Premium "
                            "(TELEGRAM_HAS_PREMIUM=false) and no seed channels are "
                            "configured (TELEGRAM_SEED_CHANNELS)")))
                return CollectResult(posts=posts, status=statuses)

            # ---- per-channel date-bounded search over the seeds, strictly sequential on one
            # account. Runs before global search so these targeted channels get the budget.
            seen = {p.post_id for p in posts}
            for n, ch in enumerate(seeds):
                remaining = query.limit - len(posts)
                if state.flood_blocked is not None or remaining <= 0 or ctx.out_of_time():
                    why = (f"account in flood wait {state.flood_blocked}s"
                           if state.flood_blocked is not None else
                           "post limit reached" if remaining <= 0 else "time budget exhausted")
                    statuses.append(SourceStatus(
                        source=SOURCE, query=query.topic, state=SourceState.partial,
                        reason=_with_summary(f"API mode; @{ch}: not reached ({why})", query),
                        cursor=f"{_first_day(query)}:{ch}:start", started_at=now_iso()))
                    continue
                # Channels share the cap: remaining / channels left (see module docstring).
                budget = _split(remaining, len(seeds) - n)
                fb = found_how.get(ch.lower()) or (
                    "channel discovery" if ch.lower() in entities else None)
                got, st = await self._api_channel(
                    client, ch, query, ctx, state, budget, entity=entities.get(ch.lower()),
                    found_by=fb)
                for p in got:
                    if p.post_id not in seen:
                        seen.add(p.post_id)
                        posts.append(p)
                statuses.append(st)

            # ---- groups and channels this account has joined (owner request 2026-11-20).
            # The collector default is off; config.DEFAULTS turns it on for the app.
            if joined:
                for scope in ("groups", "channels"):
                    remaining = query.limit - len(posts)
                    if state.flood_blocked is not None or remaining <= 0 or ctx.out_of_time():
                        why = (f"account in flood wait {state.flood_blocked}s"
                               if state.flood_blocked is not None else
                               "post limit reached" if remaining <= 0 else "time budget exhausted")
                        statuses.append(SourceStatus(
                            source=SOURCE, query=query.topic, state=SourceState.partial,
                            date_filtered=True, started_at=now_iso(),
                            reason=_with_summary(f"API mode; joined {scope}: not reached ({why})",
                                                 query)))
                        continue
                    # _api_joined dedups against `seen` and adds the ids it keeps to it,
                    # so `got` is already new; extend without re-checking.
                    got, st = await self._api_joined(client, query, ctx, state, scope,
                                                     remaining, seen)
                    posts.extend(got)
                    statuses.append(st)

            # ---- global search LAST (Premium), capped at the remaining budget so it never
            # starves the targeted channels above.
            if premium:
                remaining = query.limit - len(posts)
                if state.flood_blocked is None and remaining > 0 and not ctx.out_of_time():
                    got, st = await self._api_global(client, query, ctx, state, cap=remaining)
                    for p in got:
                        if p.post_id not in seen:
                            seen.add(p.post_id)
                            posts.append(p)
                    statuses.append(st)
                else:
                    why = (f"account in flood wait {state.flood_blocked}s"
                           if state.flood_blocked is not None else
                           "post limit reached" if remaining <= 0 else "time budget exhausted")
                    statuses.append(SourceStatus(
                        source=SOURCE, query=query.topic, state=SourceState.partial,
                        date_filtered=False, started_at=now_iso(),
                        reason=_with_summary(f"API mode; global search: not reached ({why})",
                                             query)))
            return CollectResult(posts=posts, status=statuses)
        finally:
            try:
                await client.disconnect()
            except Exception:  # noqa: BLE001
                pass

    async def _flood_retry(self, e: Any, ctx: CollectContext, retried: bool) -> bool:
        """Sleep through a short flood wait once; return False to stop the account.

        An analyst stop (before or during the wait) ends it at once; the exception is
        then marked so the caller reports the stop, not an account flood block."""
        secs = int(getattr(e, "seconds", 0) or 0)
        if ctx.stop.is_set():
            e.socmint_stopped = True
            return False
        if not retried and secs <= FLOOD_RETRY_MAX_SECONDS and ctx.time_left() > secs + 5:
            log.warning("telegram flood wait %ss; sleeping once", secs)
            waited = 0.0
            while waited < secs:  # in short steps, so "Stop scan" is not delayed
                if ctx.stop.is_set():
                    e.socmint_stopped = True
                    return False
                step = min(0.5, secs - waited)
                await asyncio.sleep(step)
                waited += step
            return True
        return False

    async def _api_channel(self, client: Any, ch: str, query: Query, ctx: CollectContext,
                           state: _ApiState, cap: int, entity: Any = None,
                           found_by: str | None = None) -> tuple[list[Post], SourceStatus]:
        """Date-bounded search of one channel. `entity` (from channel discovery) skips the
        flood-prone username resolution; `found_by` labels how the channel was chosen."""
        FloodWait = _flood_error_cls()
        st = SourceStatus(source=SOURCE, query=query.topic, state=SourceState.ok,
                          started_at=now_iso(), date_filtered=True)
        prefix = f"API mode; @{ch}" + (f" (found by {found_by})" if found_by else "")

        # ---- resolve the channel (ResolveUsername is the flood-prone call)
        retried = False
        while entity is None:
            try:
                st.requests += 1
                entity = await client.get_entity(ch)
            except FloodWait as e:  # type: ignore[misc]
                if await self._flood_retry(e, ctx, retried):
                    retried = True
                    continue
                why = _flood_text(e, state)
                st.state = SourceState.partial
                st.reason = _with_summary(f"{prefix}: {why} while resolving", query)
                st.cursor = f"{_first_day(query)}:{ch}:start"
                return [], st
            except Exception as e:  # noqa: BLE001
                name = type(e).__name__
                st.state = (SourceState.out_of_coverage if "Private" in name
                            else SourceState.failed)
                st.reason = _with_summary(
                    redact(f"{prefix}: cannot resolve channel ({name}: {e})"), query)
                return [], st

        # ---- date-bounded search, one pass per day slice (newest first). offset_date =
        # slice end + 1 s returns messages older than the slice end; stop once
        # message.date < slice start. Reading a public channel needs no join.
        topic = _plain_topic(query)
        filtering = _phrase_filtering(query)
        slices = time_slices(query.start, query.end)
        share = SliceShare(cap, len(slices))
        out: list[Post] = []
        raw: list[Any] = []
        offset_id = 0
        stop_i = 0
        retried = False
        stop: str | None = None
        error: str | None = None
        dropped_phrase = 0
        skipped_empty = 0
        history_end = False
        # Date of the newest message seen older than the previous slice's start: there is
        # no matching message between it and that start, so slices in between are empty.
        older_seen: datetime | None = None
        async def search_slice(i: int, allowance: int,
                               start_id: int) -> tuple[int, int, datetime | None, bool]:
            """Search slice i from offset_id `start_id` (0 = from the slice end), keeping up
            to `allowance` posts. Returns (kept, last offset_id, the first message seen
            older than the slice start, True if the channel's history ran out)."""
            nonlocal retried, stop, error, dropped_phrase
            s, e = slices[i]
            offset_date = e + timedelta(seconds=1)
            oid = start_id
            got = 0
            older: datetime | None = None
            ran_out = False
            while True:
                # Words mode: ask for exactly the allowance. Phrase mode: some results
                # will be dropped, so page until the allowance is met or the slice ends.
                lim = None if filtering else allowance - got
                yielded = 0
                hit_start = False
                try:
                    st.requests += 1
                    async for m in client.iter_messages(entity, search=topic,
                                                        offset_date=offset_date,
                                                        offset_id=oid, limit=lim):
                        yielded += 1
                        d = getattr(m, "date", None)
                        if isinstance(d, datetime) and d.tzinfo and d < s:
                            older = d
                            hit_start = True
                            break
                        oid = m.id
                        raw.append(_to_raw(m))
                        # `ch` resolved to this entity, so it is a valid public handle
                        # even when the entity's own `username` is empty (collectible).
                        p = message_to_post(m, chat=entity, collected_at=now_iso(),
                                            topic=query.topic, handle=ch)
                        if p is None:
                            continue
                        dt = parse_ts(p.published_at)
                        if p.timestamp_verified and (dt is None or not s <= dt <= e):
                            pass
                        elif not topic_matches(query, p.text):
                            dropped_phrase += 1
                        else:
                            out.append(p)
                            got += 1
                            if got >= allowance:
                                break
                        # Checked after every examined message, kept or not: in phrase
                        # mode a long run of dropped results must not overrun the budget.
                        if ctx.out_of_time():
                            stop = "time budget exhausted"
                            break
                    if (not hit_start and got < allowance and stop is None
                            and (lim is None or yielded < lim)):
                        ran_out = True  # nothing older exists in this channel
                    break
                except FloodWait as fe:  # type: ignore[misc]
                    if await self._flood_retry(fe, ctx, retried):
                        retried = True
                        continue
                    stop = _flood_text(fe, state)
                    break
                except Exception as ex:  # noqa: BLE001
                    error = redact(f"{type(ex).__name__}: {ex}")
                    stop = "error"
                    break
            return got, oid, older, ran_out

        # Slices that stopped because they filled their allowance -> offset_id to resume.
        saturated: dict[int, int] = {}
        for i, (s, _e) in enumerate(slices):
            stop_i = i
            offset_id = 0
            allowance = share.allowance(i)
            if allowance <= 0:
                stop = "post limit reached before the older days"
                break
            if history_end or (older_seen is not None and older_seen < s):
                share.record(i, 0)  # known to be empty: no request
                skipped_empty += 1
                continue
            if ctx.out_of_time():
                stop = "time budget exhausted"
                break
            got, offset_id, older_seen, ran_out = await search_slice(i, allowance, 0)
            history_end = history_end or ran_out
            if got >= allowance:
                saturated[i] = offset_id
            share.record(i, got)
            if stop:
                break

        # ---- top-up: every slice was searched but cap is left (quieter older days rolled
        # their share forward with nowhere to go). Busy slices that filled their allowance
        # are searched again from where they stopped, newest first, sharing the leftover.
        # Seen live 2026-11-19 in the global walk (300/7 = 43 kept, 257 wasted); the same
        # waste existed here whenever the older days of the window were quiet.
        topped = 0
        topup_stop: str | None = None
        left = cap - share.used
        if stop is None and saturated and left > 0:
            # Repeated rounds: a busy slice that filled its top-up share stays in play, so
            # cap a slice could not use (it ran out) goes to the others in the next round
            # (review 2026-11-19: one round wasted it). Ends when the cap is used, every
            # slice ran out, or a stop hits. Each round makes progress or empties `order`.
            order = sorted(saturated)  # slice index 0 = newest
            while order and left > 0 and topup_stop is None:
                still: list[int] = []
                for k, i in enumerate(order):
                    if left <= 0:
                        break
                    if ctx.out_of_time():
                        topup_stop = "time budget exhausted"
                        break
                    want = _split(left, len(order) - k)
                    got, oid, _older, _ran_out = await search_slice(i, want, saturated[i])
                    saturated[i] = oid
                    share.record(i, share.got.get(i, 0) + got)
                    left -= got
                    topped += got
                    if stop:
                        # The main pass already covered every slice, so the status stays
                        # complete; the account-level flood flag (state) is still honoured.
                        topup_stop = f"error: {error}" if stop == "error" else stop
                        stop = None
                        break
                    if got >= want:
                        still.append(i)
                order = still
            if topped:
                out.sort(key=lambda p: p.published_at or "", reverse=True)


        if raw:
            ref = ctx.save_raw(SOURCE, f"api_{ch}.json", raw)
            for p in out:
                p.raw_ref = ref
        state.collected += len(out)
        st.count = len(out)
        ctx.progress(SOURCE, {"message": f"telegram: @{ch} {len(out)} posts",
                              "count": state.collected})
        notes = [f"{prefix}: date-bounded channel search, one search per day slice "
                 "(offset_date), cap shared across days"]
        if skipped_empty:
            notes.append(f"{skipped_empty} slice(s) known empty from the previous search "
                         "(no request)")
        if dropped_phrase:
            notes.append(_phrase_note(dropped_phrase))
        if topped:
            notes.append(f"{topped} more post(s) taken from busier days with the cap that "
                         "quieter older days left unused")
        if topup_stop:
            notes.append(f"top-up of busier days stopped ({topup_stop})")
        if stop == "error":
            st.state = SourceState.partial if out else SourceState.failed
            notes.insert(1, f"error: {error}")
        elif stop:
            st.state = SourceState.partial
            notes.append(stop)
        else:
            st.state = SourceState.ok if out else SourceState.completed_zero
            notes.append("history exhausted" if history_end else "reached start date")
        if stop and slices:
            st.cursor = f"{slices[stop_i][0]:%Y-%m-%d}:{ch}:offset_id={offset_id}"
        notes.append(share.summary(slices))
        st.reason = "; ".join(notes)
        return out, st

    async def _discover_underground(self, client: Any, query: Query, ctx: CollectContext,
                                    state: _ApiState, exclude: set[str], cap: int, *,
                                    skip_joined: bool) -> tuple[list[tuple[str, Any, str]],
                                                                SourceStatus]:
        """Find data-leak / infostealer / breach channels and return them to be searched for
        the topic by date (owner request 2026-11-20). Prioritised, cheapest signal first:

        (a) the bundled known-channel list (collect.underground, from deepdarkCTI),
            highest-signal category first, resolving usernames until the cap is met or the
            account is asked to pause (resolution is the flood-prone call, so it stops at the
            first flood wait);
        (b) Telegram directory matches for leak vocabulary (collect.underground.VOCAB) -
            one request per term, entities returned with the results (no per-channel resolve).

        (Global search runs AFTER this discovery now, so it is no longer a seed source.)

        Only public channels/supergroups are kept (readable without joining); joined ones
        are left to the joined-chats search. Returns ([(username, entity, how)], status)."""
        st = SourceStatus(source=SOURCE, query=query.topic, state=SourceState.ok,
                          started_at=now_iso(), date_filtered=True)
        prefix = "API mode; underground discovery"
        try:
            from telethon.tl import functions
        except Exception as e:  # noqa: BLE001
            st.state = SourceState.skipped_capability
            st.reason = f"{prefix}: Telethon not importable ({type(e).__name__})"
            return [], st
        FloodWait = _flood_error_cls()
        curated = ug.curated_usernames()
        picked: list[tuple[str, Any, str]] = []
        seen = set(exclude)
        notes: list[str] = []
        n_known = n_vocab = 0

        def add(uname: str | None, entity: Any, how: str) -> bool:
            nonlocal n_known
            if not uname or uname.lower() in seen or len(picked) >= cap:
                return False
            if skip_joined and getattr(entity, "left", True) is False:
                return False  # joined: covered by the joined-chats search
            seen.add(uname.lower())
            picked.append((uname, entity, how))
            if uname.lower() in curated:
                n_known += 1
            return True

        # (a) the bundled known-channel list, highest-signal category first; resolution is
        # the flood-prone call, so it stops at the first flood wait rather than hammering
        # the account.
        resolved = failed = 0
        for row in ug.curated_channels():
            if len(picked) >= cap or ctx.out_of_time() or state.flood_blocked is not None:
                break
            uname = row["username"]
            if uname.lower() in seen:
                continue
            try:
                st.requests += 1
                ent = await client.get_entity(uname)
            except FloodWait as fe:  # type: ignore[misc]
                notes.append(f"resolving known channels stopped: {_flood_text(fe, state)}")
                break
            except Exception:  # noqa: BLE001 - dead/renamed channel, never fatal
                failed += 1
                continue
            if add(uname, ent, f"known {(row.get('categories') or ['leak'])[0]} channel"):
                resolved += 1
        if failed:
            notes.append(f"{failed} known channel(s) could not be resolved (dead or renamed)")

        # (b) fresh channels from leak vocabulary in Telegram's directory, kept only when
        # the name confirms a leak/underground purpose (ug.looks_underground).
        retried = False
        for term in ug.VOCAB:
            if len(picked) >= cap or ctx.out_of_time() or state.flood_blocked is not None:
                break
            try:
                st.requests += 1
                found = await client(functions.contacts.SearchRequest(
                    q=term, limit=DISCOVER_SEARCH_LIMIT))
            except FloodWait as fe:  # type: ignore[misc]
                if await self._flood_retry(fe, ctx, retried):
                    retried = True
                    continue
                notes.append(f"vocabulary search stopped: {_flood_text(fe, state)}")
                break
            except Exception as ex:  # noqa: BLE001
                notes.append(redact(f"vocabulary search \"{term}\" failed ({type(ex).__name__})"))
                continue
            for c in getattr(found, "chats", None) or []:
                if not (getattr(c, "broadcast", False) or getattr(c, "megagroup", False)):
                    continue
                uname = public_handle(c)
                if not uname:
                    continue
                known = uname.lower() in curated
                if not known and not ug.looks_underground(getattr(c, "title", None), uname):
                    continue  # name does not confirm a leak channel: skip the noise
                if add(uname, c, f'leak vocabulary "{term}"'
                       + (" (also in the known list)" if known else "")):
                    n_vocab += 1

        names = ", ".join(f"@{u}" for u, _e, _h in picked)
        if picked:
            st.reason = (f"{prefix}: {len(picked)} leak/underground channel(s) to search for "
                         f'"{_plain_topic(query)}" ({resolved} from the known list, '
                         f"{n_vocab} by leak vocabulary), each searched by date below: {names}")
        else:
            st.state = SourceState.completed_zero
            st.reason = (f"{prefix}: no public leak/underground channel found for this topic "
                         "(known list and leak-vocabulary directory search)")
        if notes:
            st.reason += "; " + "; ".join(notes)
        return picked, st

    async def _discover(self, client: Any, query: Query, ctx: CollectContext,
                        state: _ApiState, exclude: set[str], cap: int, *,
                        skip_joined: bool) -> tuple[list[tuple[str, Any]], SourceStatus]:
        """Find public channels / supergroups for the topic (owner request 2026-11-20).

        contacts.search (Telegram's directory search over chat names and usernames) gives
        the best matches; channels.getChannelRecommendations adds Telegram's "similar
        channels" for the first DISCOVER_RECOMMEND_FROM of them. Only public chats (with a
        username) are kept, since they can be searched without joining. Chats the account
        has joined are skipped when the joined-chats search already covers them, and seed
        channels are never repeated. Returns ([(username, entity)], status).
        UNVERIFIED: contacts.Found / recommendation shapes are from the TL schema
        (Telethon 1.46, layer 233), not yet confirmed live."""
        st = SourceStatus(source=SOURCE, query=query.topic, state=SourceState.ok,
                          started_at=now_iso(), date_filtered=True)
        prefix = "API mode; channel discovery"
        try:
            from telethon.tl import functions
        except Exception as e:  # noqa: BLE001
            st.state = SourceState.skipped_capability
            st.reason = f"{prefix}: Telethon not importable ({type(e).__name__})"
            return [], st
        FloodWait = _flood_error_cls()
        topic = _plain_topic(query)
        picked: list[tuple[str, Any]] = []
        seen: set[str] = set(exclude)
        skipped_joined = 0
        notes: list[str] = []

        def take(chats: list[Any], joined_ids: set[int], how: str) -> int:
            nonlocal skipped_joined
            n = 0
            for c in chats:
                if len(picked) >= cap:
                    break
                if not (getattr(c, "broadcast", False) or getattr(c, "megagroup", False)):
                    continue  # bots, users and legacy groups are not channels
                uname = public_handle(c)
                if not uname or uname.lower() in seen:
                    continue
                if skip_joined and (getattr(c, "id", None) in joined_ids
                                    or getattr(c, "left", True) is False):
                    skipped_joined += 1
                    continue
                seen.add(uname.lower())
                picked.append((uname, c))
                n += 1
            return n

        # Live 2026-11-20: the directory returns ~10 name matches per query, so the topic's
        # most distinctive word is searched too ("saudi aramco" -> also "aramco": 10 more,
        # incl. Arabic-titled channels). Candidates are then ranked by subscriber count so
        # active channels come first under the cap.
        queries = [topic]
        words = [w for w in re.findall(r"\w+", topic) if len(w) >= 4]
        if len(topic.split()) > 1 and words and not is_search_expression(topic):
            key = max(words, key=len)
            if key.lower() != topic.lower():
                queries.append(key)
        chats: list[Any] = []
        mine: set[int] = set()
        for qi, qtext in enumerate(queries):
            retried = False
            found = None
            while found is None:
                try:
                    st.requests += 1
                    found = await client(functions.contacts.SearchRequest(
                        q=qtext, limit=DISCOVER_SEARCH_LIMIT))
                except FloodWait as e:  # type: ignore[misc]
                    if await self._flood_retry(e, ctx, retried):
                        retried = True
                        continue
                    if qi == 0:
                        st.state = SourceState.partial
                        st.reason = f"{prefix}: {_flood_text(e, state)}; no channels discovered"
                        return [], st
                    notes.append(f"directory search for \"{qtext}\" stopped: "
                                 f"{_flood_text(e, state)}")
                    break
                except Exception as e:  # noqa: BLE001
                    if qi == 0:
                        st.state = SourceState.failed
                        st.reason = redact(f"{prefix}: directory search failed "
                                           f"({type(e).__name__}: {e})")
                        return [], st
                    notes.append(redact(f"directory search for \"{qtext}\" failed "
                                        f"({type(e).__name__})"))
                    break
            if found is None:
                break
            mine |= {p[1] for p in (_peer(x) for x in getattr(found, "my_results", None) or [])
                     if p}
            have = {getattr(c, "id", None) for c in chats}
            chats += [c for c in (getattr(found, "chats", None) or [])
                      if getattr(c, "id", None) not in have]
        # Directory name matches: bigger audiences first (unknown counts last). (Global
        # search now runs AFTER discovery, so its result channels are no longer a seed here;
        # discovery relies on the directory and Telegram's "similar channels".)
        ranked = sorted(chats, key=lambda c: -(getattr(c, "participants_count", None) or -1))
        from_search = take(ranked, mine, "search")

        # Telegram's "similar channels" for the best matches.
        from_recs = 0
        Recs = getattr(functions.channels, "GetChannelRecommendationsRequest", None)
        if Recs is not None and state.flood_blocked is None:
            for _u, c in list(picked[:DISCOVER_RECOMMEND_FROM]):
                if len(picked) >= cap or ctx.out_of_time() or not getattr(c, "broadcast", False):
                    continue
                try:
                    from telethon import utils as tl_utils
                    st.requests += 1
                    rec = await client(Recs(channel=tl_utils.get_input_channel(c)))
                except FloodWait as e:  # type: ignore[misc]
                    notes.append(f"similar-channel lookup stopped: {_flood_text(e, state)}")
                    break
                except Exception as e:  # noqa: BLE001
                    notes.append(redact(f"similar-channel lookup failed for @{_u} "
                                        f"({type(e).__name__})"))
                    continue
                from_recs += take(list(getattr(rec, "chats", None) or []), mine, "similar")

        names = ", ".join(f"@{u}" for u, _ in picked)
        if picked:
            st.reason = (f"{prefix}: {len(picked)} public channel(s) found for "
                         + " / ".join(f'"{x}"' for x in queries)
                         + f" ({from_search} by name, {from_recs} similar), each searched by "
                         f"date below: {names}")
        else:
            st.state = SourceState.completed_zero
            st.reason = (f"{prefix}: Telegram's directory listed no public channel or group "
                         f"for \"{topic}\"")
        if skipped_joined:
            notes.append(f"{skipped_joined} channel(s) you have joined left to the joined-chats "
                         "search")
        if len(picked) >= cap:
            notes.append(f"stopped at TELEGRAM_DISCOVER_MAX={cap}")
        if notes:
            st.reason += "; " + "; ".join(notes)
        st.count = 0  # posts are counted on each channel's own status
        return picked, st

    async def _api_joined(self, client: Any, query: Query, ctx: CollectContext,
                          state: _ApiState, scope: str, cap: int,
                          seen: set[str]) -> tuple[list[Post], SourceStatus]:
        """Search the groups (scope="groups") or channels ("channels") this account has
        joined, via messages.searchGlobal with server-side date bounds.

        One pass per time slice (newest first), sharing `cap` across slices (SliceShare).
        Private one-to-one chats are never searched (groups_only / broadcasts_only), and a
        result whose peer is a user is dropped defensively. Phrase mode is checked
        client-side (topic_matches), as for the other Telegram searches.
        UNVERIFIED: searchGlobal's paging (next_rate + offset peer/id of the last message)
        follows the TL schema and Telethon's iter_messages; not yet confirmed live."""
        st = SourceStatus(source=SOURCE, query=query.topic, state=SourceState.ok,
                          started_at=now_iso(), date_filtered=True)
        prefix = f"API mode; joined {scope}"
        try:
            from telethon import utils as tl_utils
            from telethon.tl import functions, types
        except Exception as e:  # noqa: BLE001
            st.state = SourceState.skipped_capability
            st.reason = f"{prefix}: Telethon not importable ({type(e).__name__})"
            return [], st
        SearchGlobal = getattr(functions.messages, "SearchGlobalRequest", None)
        if SearchGlobal is None:
            st.state = SourceState.skipped_capability
            st.reason = f"{prefix}: installed Telethon has no messages.searchGlobal"
            return [], st
        params = inspect.signature(SearchGlobal).parameters
        flag = "groups_only" if scope == "groups" else "broadcasts_only"
        if flag not in params:
            st.state = SourceState.skipped_capability
            st.reason = (f"{prefix}: installed Telethon layer has no {flag} filter, so private "
                         "chats could not be excluded; not searched")
            return [], st

        FloodWait = _flood_error_cls()
        topic = _plain_topic(query)
        slices = time_slices(query.start, query.end)
        share = SliceShare(cap, len(slices))
        out: list[Post] = []
        raw: list[Any] = []
        dropped_phrase = dropped_private = unresolved = 0
        stop: str | None = None
        error: str | None = None
        retried = False
        cursor: str | None = None
        for i, (s, e) in enumerate(slices):
            allowance = share.allowance(i)
            if allowance <= 0:
                share.record(i, 0)
                continue
            if stop is not None:
                break
            got = 0
            offset_rate, offset_peer, offset_id = 0, types.InputPeerEmpty(), 0
            pages = 0
            while got < allowance and pages < JOINED_MAX_PAGES_PER_SLICE:
                if ctx.out_of_time():
                    stop = "time budget exhausted"
                    break
                try:
                    st.requests += 1
                    res = await client(SearchGlobal(
                        q=topic, filter=types.InputMessagesFilterEmpty(),
                        min_date=s, max_date=e, offset_rate=offset_rate,
                        offset_peer=offset_peer, offset_id=offset_id,
                        limit=min(API_PAGE, max(1, allowance - got)), **{flag: True}))
                except FloodWait as fe:  # type: ignore[misc]
                    if await self._flood_retry(fe, ctx, retried):
                        retried = True
                        continue
                    stop = _flood_text(fe, state)
                    break
                except Exception as ex:  # noqa: BLE001
                    error = redact(f"{type(ex).__name__}: {ex}")
                    stop = "error"
                    break
                pages += 1
                msgs = list(getattr(res, "messages", None) or [])
                lookup = entity_lookup(*(getattr(res, "chats", None) or []),
                                       *(getattr(res, "users", None) or []))
                collected_at = now_iso()
                for m in msgs:
                    if got >= allowance:
                        break
                    raw.append(_to_raw(m))
                    peer = _peer(getattr(m, "peer_id", None))
                    if peer is None or peer[0] == "user":
                        dropped_private += 1  # never a one-to-one chat
                        continue
                    chat = lookup.get(peer)
                    if chat is None:
                        unresolved += 1
                        continue
                    p = message_to_post(m, chat=chat, collected_at=collected_at,
                                        topic=query.topic, lookup=lookup)
                    if p is None or p.post_id in seen:
                        continue
                    if not topic_matches(query, p.text):
                        dropped_phrase += 1
                        continue
                    seen.add(p.post_id)
                    out.append(p)
                    got += 1
                ctx.progress(SOURCE, {"message": f"telegram: joined {scope} {len(out)} posts "
                                                 f"(slice {i + 1}/{len(slices)})",
                                      "count": state.collected + len(out)})
                next_rate = getattr(res, "next_rate", None)
                if not msgs or got >= allowance:
                    break
                last = msgs[-1]
                lp = _peer(getattr(last, "peer_id", None))
                last_chat = lookup.get(lp) if lp else None
                if next_rate is None or last_chat is None:
                    break
                offset_rate, offset_id = int(next_rate), int(last.id)
                offset_peer = tl_utils.get_input_peer(last_chat)
            share.record(i, got)
            if stop is not None:
                cursor = f"{s:%Y-%m-%d}:joined-{scope}:offset_id={offset_id}"
        if raw:
            ref = ctx.save_raw(SOURCE, f"api_joined_{scope}.json", raw)
            for p in out:
                p.raw_ref = ref
        state.collected += len(out)
        st.count = len(out)
        notes = [f"{prefix} (messages.searchGlobal over chats this account has joined, "
                 f"{'groups' if scope == 'groups' else 'channels'} only, date-bounded per "
                 "slice; one-to-one chats are never searched)"]
        if dropped_phrase:
            notes.append(_phrase_note(dropped_phrase))
        if dropped_private:
            notes.append(f"{dropped_private} result(s) from one-to-one chats not kept")
        if unresolved:
            notes.append(f"{unresolved} result(s) skipped: their chat was missing from the "
                         "response")
        if stop == "error":
            st.state = SourceState.partial if out else SourceState.failed
            notes.append(f"error: {error}")
        elif stop is not None:
            st.state = SourceState.partial
            notes.append(f"stopped: {stop}")
            st.cursor = cursor
        else:
            st.state = SourceState.ok if out else SourceState.completed_zero
        notes.append(share.summary(slices))
        st.reason = "; ".join(notes)
        return out, st

    async def _api_global(self, client: Any, query: Query, ctx: CollectContext,
                          state: _ApiState,
                          cap: int | None = None) -> tuple[list[Post], SourceStatus]:
        """Premium global post search via channels.searchPosts (no server date filter).

        `cap` bounds how many posts this search may keep. Global search runs LAST (after the
        targeted per-channel searches), so `cap` is the budget left over — the noisy global
        search can never starve the targeted seed/underground channels (owner decision
        2026-11-20)."""
        st = SourceStatus(source=SOURCE, query=query.topic, state=SourceState.ok,
                          started_at=now_iso(), date_filtered=False)
        prefix = "API mode; global search"
        try:
            from telethon.tl import functions, types
        except Exception as e:  # noqa: BLE001
            st.state = SourceState.skipped_capability
            st.reason = f"{prefix}: Telethon not importable ({type(e).__name__})"
            return [], st
        SearchPosts = getattr(functions.channels, "SearchPostsRequest", None)
        if SearchPosts is None or "query" not in inspect.signature(SearchPosts).parameters:
            st.state = SourceState.skipped_capability
            st.reason = (f"{prefix}: installed Telethon layer has no channels.searchPosts "
                         "with a text query; falling back to per-channel search")
            return [], st

        FloodWait = _flood_error_cls()
        topic = _plain_topic(query)
        # Check the free search quota first; never spend Stars.
        # UNVERIFIED: SearchPostsFlood semantics (query_is_free/remains) from TL schema only.
        Check = getattr(functions.channels, "CheckSearchPostsFloodRequest", None)
        if Check is not None:
            try:
                st.requests += 1
                fl = await client(Check(query=topic))
                if not getattr(fl, "query_is_free", False) and getattr(fl, "remains", 1) == 0:
                    st.state = SourceState.skipped_capability
                    st.reason = (f"{prefix}: free daily quota exhausted "
                                 f"(0/{getattr(fl, 'total_daily', '?')}); paid search "
                                 "(Stars) is never used")
                    return [], st
            except FloodWait as e:  # type: ignore[misc]
                state.flood_blocked = int(getattr(e, "seconds", 0) or 0)
                st.state = SourceState.partial
                st.reason = _with_summary(f"{prefix}: flood wait {state.flood_blocked}s", query)
                return [], st
            except Exception as e:  # noqa: BLE001
                log.warning("checkSearchPostsFlood failed: %s", redact(str(e)))

        # Verified live 2026-11-19 (lead): channels.searchPosts has no date filter and
        # offset_rate set to a unix timestamp returned 0 messages, so it cannot seek to a
        # date. Results are walked newest first and spread across day slices client-side.
        walk = _SliceWalk(query, query.limit if cap is None else max(0, cap))
        raw: list[Any] = []
        # Offsets of the NEXT request; when the walk stops mid-page they still point at
        # that page, so the cursor resumes where collection stopped.
        offset_rate, offset_peer, offset_id = 0, types.InputPeerEmpty(), 0
        retried = False
        ended: str | None = None  # exhausted | full | error | <stop reason> | None=page cap
        error: str | None = None
        unresolved = 0  # messages whose peer was absent from the response's chats
        pages = 0
        while pages < GLOBAL_MAX_PAGES:
            if walk.full():
                ended = "full"
                break
            if ctx.out_of_time():
                ended = "time budget exhausted"
                break
            try:
                st.requests += 1
                res = await client(SearchPosts(offset_rate=offset_rate, offset_peer=offset_peer,
                                               offset_id=offset_id, limit=API_PAGE,
                                               query=topic))
            except FloodWait as e:  # type: ignore[misc]
                if await self._flood_retry(e, ctx, retried):
                    retried = True
                    continue
                ended = _flood_text(e, state)
                break
            except Exception as e:  # noqa: BLE001
                error = redact(f"{type(e).__name__}: {e}")
                ended = "error"
                break
            pages += 1
            msgs = list(getattr(res, "messages", None) or [])
            lookup = entity_lookup(*(getattr(res, "chats", None) or []),
                                   *(getattr(res, "users", None) or []))
            collected_at = now_iso()
            broke_mid_page = False
            for m in msgs:
                if walk.full():
                    broke_mid_page = True
                    break
                raw.append(_to_raw(m))
                peer = _peer(getattr(m, "peer_id", None))
                chat = lookup.get(peer) if peer else None
                if chat is None:
                    unresolved += 1  # counted and reported, never silently dropped
                    continue
                p = message_to_post(m, chat=chat, collected_at=collected_at,
                                    topic=query.topic, lookup=lookup)
                if p is None:
                    continue
                uname = public_handle(chat)
                if uname and (getattr(chat, "broadcast", False) or getattr(chat, "megagroup",
                                                                             False)) \
                        and topic_matches(query, p.text):
                    e = state.global_channels.setdefault(uname.lower(), [0, uname, chat])
                    e[0] += 1
                # Candidates only; the cap is split across days after the walk.
                walk.offer(p.post_id, parse_ts(p.published_at), p.text, p)
            got = min(walk.candidates, walk.total)
            ctx.progress(SOURCE, {"message": f"telegram: global search {walk.candidates} "
                                             f"candidate posts (page {pages})",
                                  "count": state.collected + got})
            next_rate = getattr(res, "next_rate", None)
            if broke_mid_page:
                ended = "full"
                break
            # UNVERIFIED: assumes results come newest-first, so once the walk has seen a
            # result older than the window start, later pages are older too.
            if not msgs or not next_rate or walk.passed_start:
                ended = "exhausted"
                break
            if walk.full():
                ended = "full"
                break
            last = msgs[-1]
            lp = _peer(getattr(last, "peer_id", None))
            last_chat = lookup.get(lp) if lp else None
            if last_chat is None:
                ended = "cannot page further (last result's chat missing from response)"
                break
            from telethon import utils as tl_utils
            offset_rate, offset_id = int(next_rate), int(last.id)
            offset_peer = tl_utils.get_input_peer(last_chat)

        walk.finish(ended == "exhausted")
        out: list[Post] = walk.select()
        if raw:
            ref = ctx.save_raw(SOURCE, "api_global_search.json", raw)
            for p in out:
                p.raw_ref = ref
        state.collected += len(out)
        st.count = len(out)
        notes = [f"{prefix} (channels.searchPosts, results walked newest first and spread "
                 "across days by platform timestamps)", *walk.notes("global search")]
        if unresolved:
            notes.append(f"{unresolved} result(s) skipped: their chat was missing from "
                         "the response")
        if ended == "error":
            st.state = SourceState.partial if out else SourceState.failed
            notes.insert(1, f"error: {error}")
        elif walk.total <= 0:
            # Nothing was requested (limit 0): never report "0 matching posts".
            st.state = SourceState.partial
            notes.append("post limit is 0; nothing searched")
        elif ended in ("exhausted", "full"):
            st.state = SourceState.ok if out else SourceState.completed_zero
            if ended == "full":
                notes.append("stopped early: every day already had its share of candidates")
        else:
            st.state = SourceState.partial
            stop = ended or f"page cap ({GLOBAL_MAX_PAGES}) reached"
            notes.append(f"{stop} before the window start")
        if st.state == SourceState.partial:
            st.cursor = walk.cursor(f"global:offset_rate={offset_rate}:offset_id={offset_id}")
        notes.append(walk.summary())
        st.reason = "; ".join(notes)
        return out, st


class _ApiUnavailable(Exception):
    """API mode cannot run (connect failed / not authorized); caller may fall back."""


def _make_telethon_client(cfg: dict[str, str]) -> Any:
    from telethon import TelegramClient

    path = session_path(cfg)
    # flood_sleep_threshold=0: every FloodWait reaches our handler, which decides
    # whether to sleep (<=60s, once, within the deadline) or stop with `partial`.
    return TelegramClient(str(path), int(cfg["TELEGRAM_API_ID"]), cfg["TELEGRAM_API_HASH"],
                          flood_sleep_threshold=0, receive_updates=False)


__all__ = [
    "TelegramCollector", "parse_preview", "parse_count", "message_to_post",
    "normalize_channel", "seed_channels", "select_mode", "session_path", "entity_lookup",
    "public_handle",
]
