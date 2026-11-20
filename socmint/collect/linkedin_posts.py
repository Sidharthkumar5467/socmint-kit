"""LinkedIn POSTS through the owner's OWN logged-in linkedin.com session (source "linkedin").

Two providers (config LINKEDIN_PROVIDER): "browser" (this module, the default) and
"voyager" (linkedin_voyager.py: direct calls to the Voyager search endpoint with the same
session; owner decision 2026-11-19). Both share the walk, Post mapping and lock below.

Owner decision 2026-11-19 (CLAUDE.md §2): LinkedIn posts are read through the owner's own
account in a local browser. LinkedIn Ads stays a separate, public source ("linkedin_ads").

What this module does and does NOT do:
  - It opens linkedin.com's OWN content search page
    (/search/results/content/?keywords=...&sortBy=["date_posted"]&datePosted=[...]) in a
    local browser with the owner's cookies (config LINKEDIN_COOKIE) and scrolls it.
    LinkedIn's web app builds every request; we only OBSERVE the responses it receives.
  - This (browser) provider makes no direct calls to LinkedIn internal APIs and uses no
    stealth / anti-detection or user-agent override; the separate `voyager` provider
    (linkedin_voyager.py, owner decision 2026-11-19) calls the search endpoint directly. One browser, one page, one scroll at a time,
    2-4 s human-scale pauses, never in parallel (module lock).
  - It stops at once on a login / authwall / checkpoint / challenge page, on HTTP 429 (or
    LinkedIn's 999 block code), and ends the walk after 3 scrolls with no new results.

Where the results come from (seen live 2026-11-19, headless Chrome, owner's session):
  - The page is server-driven UI: React Server Components "flight" rows. The FIRST
    results (3 posts) are embedded in the HTML document as `window.__como_rehydration__`
    (a JS array of string chunks that concatenate to a flight stream).
  - Further results arrive as POST /flagship-web/rsc-action/actions/pagination
    ?sduiid=com.linkedin.sdui.search.contentSearchResults (a flight stream, ~2.6 MB,
    3 posts per call). Each stream carries the NEXT request's payload
    (PaginationRequest pagerId=...contentSearchResults, startIndex, keywords, datePosted);
    a stream without one means the results ended.
  - Pagination is triggered by an item-distance trigger (preloadDistance 3,
    preloadLength 1500 px): the page's results column must actually scroll. Seen live: a
    mouse wheel at the default pointer position (0,0, over the fixed nav bar) scrolled
    nothing, so only the first 6 results loaded. We scroll the last result card into view.
  - Each post is a node with viewTrackingSpecs.viewName "feed-full-update"; inside it:
    "feed-actor" (author link + label), "feed-actor-sub-description" (relative time
    label "7m •"), "feed-commentary" (the post text, original language), "feed-update-image"
    (images); videos are <source src type="video/mp4">. Counts live in the stream's
    modelStates keyed by the post URN (ReactionType_*_<urn>, commentCount-<urn>,
    repostCount-<urn>). The share payload ("Repost with thoughts") names the activity id.
  - The search stream ALSO carries the logged-in account's own data (name, headline,
    avatar, viewerProfileId in comment boxes). Raw streams are therefore NEVER saved:
    raw/linkedin/search_page_<n>.json holds only the fields extracted from each result
    item (post content and its author), plus the next-page request payload.

Evidence rules (CLAUDE.md §3):
  - post_id "linkedin:<activity id>"; url https://www.linkedin.com/feed/update/urn:li:
    activity:<id>/. A post whose stream ties a ugcPost/share/groupPost to an activity is
    stored once, under the activity.
  - published_at is DECODED FROM THE POST ID: LinkedIn ids embed creation time,
    (id >> 22) = Unix ms. The ACTIVITY id is used because seen live a ugcPost/share id
    can predate publication by hours or weeks (scheduled / approved posts) while the
    activity id matched the card's "7m" label. timestamp_verified=True only when the
    decoded time agrees with the card's relative label; otherwise False with a note.
  - Relations are DIRECT only: links and mentions in the post text; a reshare/quote
    relation only when the item embeds a second post by a different author (UNVERIFIED).
  - LinkedIn has no custom date range: results are walked newest first (date_posted)
    with the smallest datePosted preset covering the window start, every in-window
    candidate is kept, and only after the walk is the cap split across day slices
    (window.fill_slices, newest first within a day), exactly like Telegram global search.
    The date order is not strict (seen live), so the walk ends at the window start only
    after OLDER_TO_PASS results older than it.
  - Verified live 2026-11-19 (two collector runs, 49 results): the datePosted presets
    past-24h, past-week and past-month are echoed back in the next-page payload; 49/49
    ID-decoded times agreed with labels from "now" to "4w"; ~3 results per pagination
    call, ~5 per scroll, ~8 s per scroll. Quoted keywords are mostly but not strictly
    phrase matches (2 of 33 results lacked the phrase), hence the client-side check.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from socmint.collect.window import (
    SliceShare,
    fill_slices,
    search_topic,
    time_slices,
    topic_matches,
)
from socmint.config import redact
from socmint.interfaces import Capabilities, CollectContext, CollectResult, Query
from socmint.schema import Media, Metrics, Post, Relation, SourceState, SourceStatus
from socmint.util.log import get_logger
from socmint.util.timeutil import iso, now_iso, now_utc

log = get_logger(__name__)

SOURCE = "linkedin"
BASE = "https://www.linkedin.com"
SEARCH_PATH = "/search/results/content/"
PAGINATION_PATH = "/flagship-web/rsc-action/actions/pagination"
PAGER_ID = "com.linkedin.sdui.search.contentSearchResults"
RESULTS_MARKER = "com.linkedin.sdui.search.contentSearchResults"
COOKIE_DOMAIN = ".linkedin.com"

REASON_PREFIX = "linkedin.com content search (owner's logged-in account)"
NO_COOKIE = "no LinkedIn session: paste your linkedin.com Cookie header into LINKEDIN_COOKIE"
NO_LI_AT = ("LINKEDIN_COOKIE has no li_at cookie (the login cookie): paste the full Cookie "
            "header of a logged-in linkedin.com tab, or the li_at value alone")
NO_BROWSER = ("no browser: run socmint doctor --prefetch (installs Chromium) or install "
              "Google Chrome")
BUSY = "another LinkedIn collection is already running on this account"
ACCOUNT_RISK = ("Uses your own LinkedIn account. LinkedIn restricts accounts that automate "
                "searches; collection is paced and stops at any checkpoint.")

# Timing (module constants so tests can shrink them).
FIRST_RESPONSE_TIMEOUT_S = 30.0
RESPONSE_TIMEOUT_S = 12.0
POLL_MS = 250
PAUSE_MS = (2000, 4000)  # human-scale pause before every scroll
NO_NEW_LIMIT = 3  # consecutive scrolls without a new result => end of the walk
# Seen live 2026-11-19: sortBy=date_posted is NOT strictly ordered (a 2026-10-25 post
# arrived after two posts from 2026-10-22; an 11-06 post before an 11-18 one). The walk
# counts as past the window start only after this many results older than the start.
OLDER_TO_PASS = 3
DEFAULT_MAX_SCROLLS = 25
MAX_SCROLLS_CAP = 200
VIEWPORT = {"width": 1280, "height": 900}

# datePosted presets, smallest first, with the age they are assumed to cover.
# Verified live 2026-11-19: past-24h (echoed in the next-page payload) and past-week.
# UNVERIFIED: past-month's exact span; 28 days is used so a 29-30 day old start falls back
# to no preset (a longer walk, never a silently truncated window).
DATE_PRESETS: tuple[tuple[str, timedelta], ...] = (
    ("past-24h", timedelta(hours=24)),
    ("past-week", timedelta(days=7)),
    ("past-month", timedelta(days=28)),
)

_lock = threading.Lock()

# UNVERIFIED (one live observation, 2026-11-19): LinkedIn content search stops around 250
# results (paging.total 245 for a week of a topic that produces ~75 posts per 90 minutes).
# When at least this many results were walked and they end before the window start, the
# end is treated as LinkedIn's cap (older days "not reached"), not as "no older posts".
RESULT_CAP_SUSPECT = 200


# ===================================================================== cookies


def parse_cookie_header(raw: str | None) -> list[tuple[str, str]]:
    """(name, value) pairs from a Cookie header string ("a=1; b=2"), or a bare li_at."""
    s = str(raw or "").strip()
    if s.lower().startswith("cookie:"):
        s = s[7:].strip()
    if not s:
        return []
    if "=" not in s:
        return [("li_at", s.strip('"'))]
    out: list[tuple[str, str]] = []
    for part in s.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name, value = name.strip(), value.strip()
        if name:
            out.append((name, value))
    return out


def has_li_at(raw: str | None) -> bool:
    return any(n == "li_at" and v.strip('"') for n, v in parse_cookie_header(raw))


def build_cookie_dicts(raw: str | None) -> list[dict]:
    """Playwright cookies for .linkedin.com.

    Seen live 2026-11-19: only li_at may be httpOnly. LinkedIn's JS reads JSESSIONID to
    build its CSRF header; marking it httpOnly made every page XHR fail with 403
    "CSRF check failed.". Values are passed exactly as pasted (JSESSIONID keeps its quotes).
    """
    return [{"name": n, "value": v, "domain": COOKIE_DOMAIN, "path": "/", "secure": True,
             "httpOnly": n == "li_at"} for n, v in parse_cookie_header(raw)]


# Preference/analytics cookies whose values are not secrets and can legitimately occur in
# post text ("en-us", a timezone); scrubbing them would alter evidence.
NON_SECRET_COOKIES = frozenset({
    "lang", "timezone", "li_theme", "li_theme_set", "sdui_ver", "g_state", "visit",
    "AnalyticsSyncHistory", "UserMatchHistory", "lms_ads", "lms_analytics", "_uetvid",
    "_guid", "li_sugr", "dfpfpt", "fptctx2", "__cf_bm", "lidc",
})
SECRET_MIN_LEN = 16


def cookie_secrets(raw: str | None) -> list[str]:
    """Values of the session cookies (li_at, JSESSIONID, li_rm, ...) to scrub from text.

    Skips known preference/analytics cookies and short values, so ordinary words in a post
    are never redacted.
    """
    out: set[str] = set()
    for n, v in parse_cookie_header(raw):
        if n in NON_SECRET_COOKIES or n.startswith(("AMCV", "AMCVS")):
            continue
        for x in (v, v.strip('"')):
            if len(x) >= SECRET_MIN_LEN:
                out.add(x)
                out.add(quote(x, safe=""))
    whole = str(raw or "").strip()
    if len(whole) >= 6:
        out.add(whole)
    return sorted(out, key=len, reverse=True)


def scrub(text: str, secrets: list[str]) -> str:
    for v in secrets:
        text = text.replace(v, "[REDACTED]")
    return text


def _redact(text: str, cfg: dict) -> str:
    return redact(scrub(str(text), cookie_secrets(cfg.get("LINKEDIN_COOKIE"))), cfg)


# ============================================================ RSC flight stream

_REF = re.compile(r"^\$[LQ@WK]?([0-9a-f]+)$")
_PATH_REF = re.compile(r"^\$([0-9a-f]+):([A-Za-z0-9_$-]+(?::[A-Za-z0-9_$-]+)*)$")
_MISSING = object()
_ROW_ID = re.compile(rb"[0-9a-f]+")


def parse_flight(data: str | bytes) -> dict[str, Any]:
    """Rows of a React Server Components flight stream: {row id: value}.

    Row forms seen live: `<hex id>:<json>`, `<id>:I[...]` (client module import),
    `<id>:T<hex byte length>,<text>` (text row that may contain newlines). Unparseable
    rows are skipped. Import / hint rows are kept as None (never followed).
    """
    b = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    rows: dict[str, Any] = {}
    i, n = 0, len(b)
    while i < n:
        if b[i:i + 1] in (b"\n", b"\r"):
            i += 1
            continue
        c = b.find(b":", i, i + 12)
        if c < 0 or not _ROW_ID.fullmatch(b[i:c]):
            j = b.find(b"\n", i)
            i = n if j < 0 else j + 1
            continue
        rid = b[i:c].decode("ascii")
        j = c + 1
        if b[j:j + 1] == b"T":
            k = b.find(b",", j)
            try:
                length = int(b[j + 1:k], 16)
            except ValueError:
                length = -1
            if k > 0 and length >= 0:
                rows[rid] = b[k + 1:k + 1 + length].decode("utf-8", "replace")
                i = k + 1 + length
                continue
        e = b.find(b"\n", j)
        e = n if e < 0 else e
        line = b[j:e].decode("utf-8", "replace")
        i = e + 1
        if line[:1] in ("I", "H", "E", "D", "W"):
            rows[rid] = None  # import / hint / error / debug row: not data
            continue
        try:
            rows[rid] = json.loads(line)
        except ValueError:
            continue
    return rows


def rehydration_stream(html: str) -> str | None:
    """The flight stream embedded in the search page as window.__como_rehydration__."""
    k = html.find("window.__como_rehydration__")
    if k < 0:
        return None
    s = html.find("[", k)
    if s < 0:
        return None
    try:
        arr, _ = json.JSONDecoder().raw_decode(html, s)
    except ValueError:
        return None
    if not isinstance(arr, list):
        return None
    return "".join(x for x in arr if isinstance(x, str))


class _Tree:
    """Read-only helpers over flight rows (follows $L/$@/$Q references once per walk)."""

    def __init__(self, rows: dict[str, Any]):
        self.rows = rows

    def _at_path(self, rid: str, path: str) -> Any:
        """Value at a path inside a row ("props" = an element's props), or _MISSING."""
        cur = self.rows.get(rid, _MISSING)
        hops = 0
        for seg in path.split(":"):
            while isinstance(cur, str) and (m := _REF.match(cur)) and hops < 50:
                cur, hops = self.rows.get(m.group(1), _MISSING), hops + 1
            if isinstance(cur, list):
                if seg == "props" and len(cur) == 4 and cur[0] == "$":
                    cur = cur[3]
                elif seg.isdigit() and int(seg) < len(cur):
                    cur = cur[int(seg)]
                else:
                    return _MISSING
            elif isinstance(cur, dict) and seg in cur:
                cur = cur[seg]
            else:
                return _MISSING
        return cur

    def deref(self, v: Any, seen: set[str]) -> Any:
        """Row reference ("$L1a", "$@3", "$Q9") or path reference ("$19f:props:children:0",
        seen live: RSC deduplicates repeated values, e.g. the space before a mention, by
        pointing into an earlier row). Each reference is followed once per walk."""
        if isinstance(v, str) and v.startswith("$"):
            m = _REF.match(v)
            if m:
                if m.group(1) in self.rows and m.group(1) not in seen:
                    seen.add(m.group(1))
                    return self.rows[m.group(1)]
                return v
            m = _PATH_REF.match(v)
            if m and v not in seen:
                seen.add(v)
                found = self._at_path(m.group(1), m.group(2))
                if found is not _MISSING:
                    return found
        return v

    def iter_nodes(self, root: Any, seen: set[str] | None = None, follow: bool = True):
        """Every dict and string under root; references followed once each (follow=True)
        or not at all (follow=False: scan each row on its own)."""
        seen = set() if seen is None else seen
        stack = [root]
        while stack:
            x = stack.pop()
            if isinstance(x, str):
                y = self.deref(x, seen) if follow else x
                if y is not x:
                    stack.append(y)
                else:
                    yield x
            elif isinstance(x, list):
                if len(x) == 4 and x[0] == "$" and isinstance(x[1], str):
                    stack.append(x[3])  # React element: skip type / key
                else:
                    stack.extend(reversed(x))
            elif isinstance(x, dict):
                yield x
                stack.extend(reversed(list(x.values())))

    def blocks(self, root: Any, view_name: str) -> list[dict]:
        out = []
        for x in self.iter_nodes(root):
            if isinstance(x, dict):
                vts = x.get("viewTrackingSpecs")
                if isinstance(vts, dict) and vts.get("viewName") == view_name:
                    out.append(x)
        return out

    def text(self, node: Any) -> str:
        """Visible text of a rendered subtree; <br> becomes a newline."""
        parts: list[str] = []
        seen: set[str] = set()

        def go(v: Any, depth: int) -> None:
            if depth > 400:
                return
            v = self.deref(v, seen)
            if isinstance(v, str):
                if not v.startswith("$"):
                    parts.append(v)
            elif isinstance(v, list):
                if len(v) == 4 and v[0] == "$" and isinstance(v[1], str):
                    if v[1] == "br":
                        parts.append("\n")
                    elif isinstance(v[3], dict):
                        go(v[3], depth + 1)
                    return
                for x in v:
                    go(x, depth + 1)
            elif isinstance(v, dict):
                for k in ("children", "textProps"):
                    if k in v:
                        go(v[k], depth + 1)

        go(node, 0)
        return "".join(parts)


def _navs(tree: _Tree, node: Any) -> list[tuple[str, str]]:
    """(kind, url) of every navigation under node: kind = screenId or 'url'."""
    out = []
    for x in tree.iter_nodes(node):
        if not isinstance(x, dict):
            continue
        t = x.get("$type")
        if t == "proto.sdui.actions.core.NavigateToScreen" and isinstance(x.get("url"), str) \
                and x["url"]:
            out.append((str(x.get("screenId") or "screen"), x["url"]))
        elif t == "proto.sdui.actions.core.NavigateToUrl":
            u = (x.get("urlValue") or {}).get("url")
            if isinstance(u, str) and u:
                out.append(("url", u))
    return out


def _labels(tree: _Tree, node: Any) -> list[str]:
    out = []
    for x in tree.iter_nodes(node):
        if isinstance(x, dict):
            for k in ("actionLabel", "aria-label", "a11yText"):
                v = x.get(k)
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
    return out


def _image(tree: _Tree, node: Any) -> str | None:
    """Largest rendition of the first image asset under node."""
    for x in tree.iter_nodes(node):
        if isinstance(x, dict) and isinstance(x.get("rootUrl"), str) \
                and isinstance(x.get("imageRenditions"), list) and x["imageRenditions"]:
            r = max((r for r in x["imageRenditions"] if isinstance(r, dict)),
                    key=lambda r: int(r.get("width") or 0), default=None)
            if r and r.get("suffixUrl"):
                return x["rootUrl"] + str(r["suffixUrl"])
    return None


_URN = re.compile(r"urn:li:(activity|ugcPost|share|groupPost):(\d+(?:-\d+)?)")
_BOUND = re.compile(r"^(?:reactionState|confirmedReactionState|commentCount|repostCount|"
                    r"reactionsCount)-(urn:li:(?:activity|ugcPost|share|groupPost):"
                    r"\d+(?:-\d+)?)$")
_ACTIVITY_KEYS = ("repostingPostUpdateKey", "updateKey")


def _model_states(rows: dict[str, Any]) -> dict[str, Any]:
    """id -> scalar value of every modelStates entry in the stream (counts live here)."""
    out: dict[str, Any] = {}
    tree = _Tree(rows)
    for v in rows.values():
        for x in tree.iter_nodes(v, follow=False):  # no reference following
            if not isinstance(x, dict) or not isinstance(x.get("modelStates"), list):
                continue
            for m in x["modelStates"]:
                try:
                    kid = m["key"]["key"]["value"]["id"]
                    val = m["value"]
                except (KeyError, TypeError):
                    continue
                case = val.get("$case") if isinstance(val, dict) else None
                if case in ("intValue", "stringValue", "booleanValue"):
                    out[str(kid)] = val.get(case)
    return out


def _pagination_payload(rows: dict[str, Any]) -> dict | None:
    """requestedArguments.payload of the next contentSearchResults page request."""
    tree = _Tree(rows)
    for v in rows.values():
        for x in tree.iter_nodes(v, follow=False):
            req = None
            if isinstance(x, dict) and x.get("pagerId") == PAGER_ID:
                req = x
            elif (isinstance(x, str) and x.startswith("{") and "PaginationRequest" in x
                  and PAGER_ID in x):
                try:
                    req = json.loads(x)
                except ValueError:
                    req = None
            if isinstance(req, dict) and req.get("pagerId") == PAGER_ID:
                payload = (req.get("requestedArguments") or {}).get("payload")
                if isinstance(payload, dict):
                    return payload
    return None


def _activity_ids(tree: _Tree, node: Any) -> list[str]:
    """Activity ids named by the item's own update keys (share payload, refresh request)."""
    out: list[str] = []
    for x in tree.iter_nodes(node):
        if not isinstance(x, dict):
            continue
        for k in _ACTIVITY_KEYS:
            key = x.get(k)
            if isinstance(key, dict):
                for it in key.get("items") or []:
                    try:
                        aid = str(it["feedUpdateUrn"]["updateUrnActivityUrn"]["activityUrn"]
                                  ["activityId"])
                    except (KeyError, TypeError):
                        continue
                    if aid.isdigit() and aid not in out:
                        out.append(aid)
    return out


def _share_url(tree: _Tree, node: Any) -> str | None:
    for x in tree.iter_nodes(node):
        if isinstance(x, dict) and isinstance(x.get("repostingPostUpdateKey"), dict):
            u = x.get("url")
            if isinstance(u, str) and u.startswith(BASE + "/posts/"):
                return u
    return None


_HANDLE_RE = re.compile(r"^/(in|company|school|showcase)/([^/?#]+)/?")
_GROUP_RE = re.compile(r"^/groups/(\d+)/?")


def _profile_handle(url: str) -> tuple[str, str] | None:
    """('in/vanity' | 'company/slug' | 'groups/<id>', canonical URL) for a LinkedIn link."""
    p = urlparse(url if "://" in url else BASE + url)
    if p.netloc and not p.netloc.endswith("linkedin.com"):
        return None
    m = _HANDLE_RE.match(p.path)
    if m:
        handle = f"{m.group(1)}/{unquote(m.group(2))}"
        return handle, f"{BASE}/{m.group(1)}/{m.group(2)}/"
    m = _GROUP_RE.match(p.path)
    if m:
        return f"groups/{m.group(1)}", f"{BASE}/groups/{m.group(1)}/"
    return None


_NAME_PATTERNS = (
    re.compile(r"^View (?:group|company|page|showcase page|school): (.+)$"),
    re.compile(r"^View (.+?)(?:’s|'s|’|')\s+profile"),
    re.compile(r"^Open control menu for post by (.+)$"),
)
_TIME_LABEL = re.compile(
    r"^\s*(now|just now|\d+\s*(?:mo|yrs?|s|m|h|d|w|y)\b)", re.IGNORECASE)


def _author(tree: _Tree, item: Any) -> dict:
    actors = tree.blocks(item, "feed-actor")
    targets: list[tuple[str, str]] = []
    for a in actors:
        for _, url in _navs(tree, a):
            ph = _profile_handle(url)
            if ph and ph not in targets:
                targets.append(ph)
                break
    name = None
    # The actor block's own labels first ("View X’s profile", "View group: X"); the
    # post's control-menu label ("Open control menu for post by X") only as a fallback.
    labels = [lab for a in actors[:1] for lab in _labels(tree, a)]
    labels += [lab for lab in _labels(tree, item) if lab.startswith("Open control menu")]
    for lab in labels:
        m = next((m for pat in _NAME_PATTERNS if (m := pat.match(lab))), None)
        if m:
            name = m.group(1).strip()
            break
    avatar = None
    imgs = tree.blocks(item, "feed-actor-image")
    if imgs:
        avatar = _image(tree, imgs[0])
    return {"targets": targets, "name": name, "avatar": avatar}


def _commentaries(tree: _Tree, item: Any) -> list[str]:
    """Texts of the item's commentary blocks, longest first, nested duplicates dropped."""
    texts = [tree.text(b).strip() for b in tree.blocks(item, "feed-commentary")]
    texts = sorted({t for t in texts if t}, key=len, reverse=True)
    out: list[str] = []
    for t in texts:
        if not any(t in o for o in out):
            out.append(t)
    return out


def _real_url(url: str) -> str | None:
    """External target of a LinkedIn link (safety/go redirect decoded), or None."""
    p = urlparse(url)
    if p.netloc.endswith("linkedin.com") and p.path.startswith("/safety/go"):
        target = (parse_qs(p.query).get("url") or [None])[0]
        return target or None
    return url


def _relations(tree: _Tree, item: Any, own: set[str]) -> list[Relation]:
    rels: list[Relation] = []
    seen_urls: set[str] = set()
    seen_handles: set[str] = set()
    blocks = tree.blocks(item, "feed-commentary")
    for b in blocks:
        for kind, url in _navs(tree, b):
            if kind != "url" and not kind.endswith(".Profile"):
                continue
            if not (url.startswith("/") or url.lower().startswith(("http://", "https://"))):
                continue  # mailto:, tel: and other non-web links (seen live: mailto)
            p = urlparse(url if "://" in url else BASE + url)
            if p.netloc.endswith("linkedin.com") and p.path.startswith("/search/"):
                continue  # hashtag / keyword search link
            ph = _profile_handle(url)
            if ph:
                if ph[0] not in seen_handles and ph[0] not in own:
                    seen_handles.add(ph[0])
                    rels.append(Relation(type="mention", target_handle=ph[0],
                                         target_url=ph[1]))
                continue
            real = _real_url(url if "://" in url else BASE + url)
            if real and real not in seen_urls:
                seen_urls.add(real)
                rels.append(Relation(type="link", target_url=real))
    return rels


def _media(tree: _Tree, item: Any) -> list[Media]:
    out: list[Media] = []
    for b in tree.blocks(item, "feed-update-image"):
        u = _image(tree, b)
        if u and all(m.url != u for m in out):
            out.append(Media(type="photo", url=u))
    videos = []
    cover = None
    for x in tree.iter_nodes(item):
        if not isinstance(x, dict):
            continue
        if isinstance(x.get("src"), str) and str(x.get("type") or "").startswith("video/"):
            videos.append(x["src"])
        root = x.get("rootUrl")
        if cover is None and isinstance(root, str) and "/videocover-" in root:
            cover = _image(tree, x)
    if videos:
        out.append(Media(type="video", url=videos[0], thumbnail=cover))
    return out


def _metrics(states: dict[str, Any], urns: list[str]) -> Metrics:
    """Counts from modelStates for the first URN that has them (UNVERIFIED: ReactionType_*
    values are per-type reaction counts; seen live only as small integers)."""
    for urn in urns:
        rx = [v for k, v in states.items()
              if k.startswith("ReactionType_") and k.endswith("_" + urn) and isinstance(v, int)]
        comments = states.get(f"commentCount-{urn}")
        reposts = states.get(f"repostCount-{urn}")
        if rx or isinstance(comments, int) or isinstance(reposts, int):
            return Metrics(reactions=sum(rx) if rx else None,
                           comments=comments if isinstance(comments, int) else None,
                           reposts=reposts if isinstance(reposts, int) else None)
    return Metrics()


def extract_items(rows: dict[str, Any]) -> tuple[list[dict], dict | None, int]:
    """(result items, next-page payload, promoted items skipped) from one flight stream.

    An item is a plain dict of the fields read from one "feed-full-update" node; it holds
    no data about the logged-in account and is what gets saved as the raw record.
    """
    tree = _Tree(rows)
    states = _model_states(rows)
    found: list[dict] = []
    for v in rows.values():
        for x in tree.iter_nodes(v, follow=False):
            if isinstance(x, dict):
                vts = x.get("viewTrackingSpecs")
                if isinstance(vts, dict) and vts.get("viewName") == "feed-full-update":
                    found.append(x)
    items: list[dict] = []
    seen_ids: set[str] = set()
    promoted = 0
    for node in found:
        urn_counts: dict[str, int] = {}
        for s in tree.iter_nodes(node):
            if isinstance(s, str):
                for m in _URN.finditer(s):
                    urn_counts[m.group(0)] = urn_counts.get(m.group(0), 0) + 1
        if not urn_counts:
            continue
        # The post's own URN is the one its social-state bindings are keyed by
        # ("reactionState-<urn>", "commentCount-<urn>"); seen live this is the activity, or
        # the ugcPost / groupPost for scheduled and group posts.
        bound: dict[str, int] = {}
        for s in tree.iter_nodes(node):
            if isinstance(s, str) and (m := _BOUND.match(s)):
                bound[m.group(1)] = bound.get(m.group(1), 0) + 1
        pool = bound or urn_counts
        content_urn = max(pool, key=lambda u: pool[u])
        acts = _activity_ids(tree, node)
        activity_id = acts[0] if acts else None
        if activity_id is None and content_urn.startswith("urn:li:activity:"):
            activity_id = content_urn.rsplit(":", 1)[1]
        if activity_id is None:
            activity_id = next((u.rsplit(":", 1)[1] for u in urn_counts
                                if u.startswith("urn:li:activity:")), None)
        sub = tree.blocks(node, "feed-actor-sub-description")
        sub_text = tree.text(sub[0]).strip() if sub else ""
        if re.search(r"\bPromoted\b", sub_text):
            promoted += 1
            continue
        m = _TIME_LABEL.match(sub_text)
        label = re.sub(r"\s+", "", m.group(1)).lower() if m else None
        native = activity_id or content_urn.rsplit(":", 1)[1]
        key = activity_id or content_urn
        if key in seen_ids:
            continue
        seen_ids.add(key)
        author = _author(tree, node)
        texts = _commentaries(tree, node)
        own = {author["targets"][0][0]} if author["targets"] else set()
        main_urns = {content_urn, *(f"urn:li:activity:{a}" for a in acts)}
        for x in tree.iter_nodes(node):
            if isinstance(x, dict) and isinstance(x.get("rootUrn"), str):
                main_urns.add(x["rootUrn"])
        others = [u for u in urn_counts if u not in main_urns
                  and not u.startswith("urn:li:groupPost:")]
        reshare = None
        # UNVERIFIED (no reshare captured live yet): an item that embeds a second post by a
        # DIFFERENT author is treated as a reshare (no own text) or quote (own text).
        if len(author["targets"]) > 1 and others:
            reshare = {"urn": others[0], "author": author["targets"][1][0],
                       "author_url": author["targets"][1][1],
                       "quote": len(texts) > 1}
        items.append({
            "native_id": native,
            "activity_id": activity_id,
            "content_urn": content_urn,
            "urns": sorted(urn_counts),
            "share_url": _share_url(tree, node),
            "author_handle": author["targets"][0][0] if author["targets"] else None,
            "author_url": author["targets"][0][1] if author["targets"] else None,
            "author_name": author["name"],
            "author_avatar_url": author["avatar"],
            "time_label": label,
            "time_line": sub_text[:80],
            "text": texts[0] if texts else "",
            "media": [md.model_dump() for md in _media(tree, node)],
            "relations": [r.model_dump() for r in _relations(tree, node, own)],
            "metrics": _metrics(states, [content_urn] + [f"urn:li:activity:{a}"
                                                         for a in acts]).model_dump(),
            "reshare": reshare,
        })
    return items, _pagination_payload(rows), promoted


def parse_stream(text: str) -> tuple[list[dict], dict | None, int, bool]:
    """Pagination response body -> (items, next payload, promoted, results marker)."""
    rows = parse_flight(text)
    items, nxt, promoted = extract_items(rows)
    return items, nxt, promoted, RESULTS_MARKER in text


def parse_document(html: str) -> tuple[list[dict], dict | None, int, bool]:
    """Search page HTML -> first results from window.__como_rehydration__."""
    stream = rehydration_stream(html)
    if stream is None:
        return [], None, 0, False
    return parse_stream(stream)


# ================================================================ timestamps


def decode_id_time(native_id: str | int | None) -> datetime | None:
    """Creation time embedded in a LinkedIn numeric id: (id >> 22) = Unix milliseconds.

    UNVERIFIED as a documented rule; seen live 2026-11-19 to agree with the cards'
    relative labels ("5m", "7m", "10m") for activity ids. Returns None for ids that do
    not decode to a plausible date (before 2003 or in the far future).
    """
    s = str(native_id or "").split("-")[-1]
    if not s.isdigit():
        return None
    ms = int(s) >> 22
    if ms <= 0:
        return None
    try:
        dt = datetime.fromtimestamp(ms / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None
    if dt.year < 2003 or dt > now_utc() + timedelta(days=2):
        return None
    return dt


_UNIT_S = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}
SLACK_S = 180  # page render vs response capture, and clock differences


def label_window(label: str | None) -> tuple[float, float] | None:
    """(min, max) post age in seconds that a relative label like '7m' / '2h' / '3d' /
    '1w' / '2mo' / '1yr' / 'now' stands for, including tolerance. None when unknown."""
    if not label:
        return None
    s = label.strip().lower().replace(" ", "")
    if s in ("now", "justnow"):
        return (-SLACK_S, 60 + SLACK_S)
    m = re.fullmatch(r"(\d+)(mo|yrs?|s|m|h|d|w|y)", s)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    if unit == "mo":
        return ((n - 1) * 28 * 86400, (n + 1) * 31 * 86400 + SLACK_S)
    if unit.startswith("y"):
        return ((n - 1) * 365 * 86400, (n + 1) * 366 * 86400 + SLACK_S)
    u = _UNIT_S[unit]
    return (max(0, n - 1) * u - SLACK_S, (n + 1) * u + SLACK_S)


def check_timestamp(published: datetime | None, label: str | None,
                    collected: datetime) -> tuple[bool, str]:
    """(timestamp_verified, timestamp_note) for an id-decoded publication time."""
    base = "decoded from the post ID ((id >> 22) = Unix ms; UNVERIFIED rule)"
    if published is None:
        return False, f"{base}: the ID did not decode to a plausible date"
    win = label_window(label)
    if win is None:
        return False, (f"{base}; not checked: the card showed no recognisable relative "
                       f"time label ({label or 'none'})")
    age = (collected - published).total_seconds()
    if win[0] <= age <= win[1]:
        return True, f"{base}; consistent with the card's relative label '{label}'"
    return False, (f"{base}; INCONSISTENT with the card's relative label '{label}' "
                   f"(decoded age {age / 3600:.1f} h)")


def item_to_post(item: dict, *, collected_at: datetime, raw_ref: str | None,
                 query_str: str) -> Post:
    published = decode_id_time(item.get("activity_id") or item.get("native_id"))
    verified, note = check_timestamp(published, item.get("time_label"), collected_at)
    if not item.get("activity_id"):
        note += (f"; no activity id in the stream, decoded from {item.get('content_urn')}, "
                 "which can predate publication")
        verified = False
    nid = str(item["native_id"])
    urn = (f"urn:li:activity:{nid}" if item.get("activity_id")
           else str(item.get("content_urn")))
    rels = [Relation(**r) for r in item.get("relations") or []]
    rs = item.get("reshare")
    if rs:
        tid = str(rs["urn"]).rsplit(":", 1)[1]
        rels.insert(0, Relation(type="quote" if rs.get("quote") else "repost",
                                target_post_id=f"{SOURCE}:{tid}",
                                target_url=f"{BASE}/feed/update/{rs['urn']}/",
                                target_handle=rs.get("author")))
    return Post(
        post_id=f"{SOURCE}:{nid}",
        source=SOURCE,
        kind="post",
        author_id=item.get("author_handle"),
        author_handle=item.get("author_handle"),
        author_name=item.get("author_name"),
        author_avatar_url=item.get("author_avatar_url"),
        text=item.get("text") or "",
        published_at=iso(published),
        collected_at=iso(collected_at) or now_iso(),
        timestamp_verified=verified,
        timestamp_note=note,
        metrics=Metrics(**(item.get("metrics") or {})),
        media=[Media(**m) for m in item.get("media") or []],
        url=f"{BASE}/feed/update/{urn}/",
        relations=rels,
        raw_ref=raw_ref,
        query=query_str,
    )


# ================================================================== search URL


def choose_date_preset(start: datetime, now: datetime) -> str | None:
    """Smallest datePosted preset whose span covers the window start (None = no filter)."""
    age = now - start
    for name, span in DATE_PRESETS:
        if age <= span:
            return name
    return None


def build_search_url(query: Query, now: datetime) -> tuple[str, str, str | None]:
    """(url, keywords, preset). Keywords go through window.search_topic (exact phrase)."""
    keywords = search_topic(query)
    preset = choose_date_preset(query.start, now)
    url = (f"{BASE}{SEARCH_PATH}?keywords={quote(keywords, safe='')}&origin=FACETED_SEARCH"
           f"&sortBy=%5B%22date_posted%22%5D")
    if preset:
        url += f"&datePosted=%5B%22{preset}%22%5D"
    return url, keywords, preset


# ============================================================== page helpers


def _url_problem(url: str) -> str | None:
    """'login' | 'authwall' | 'checkpoint' | 'challenge' for a page that is not results."""
    try:
        p = urlparse(url or "")
    except ValueError:
        return None
    path = p.path.lower()
    if p.netloc and not p.netloc.endswith("linkedin.com"):
        return None
    if "challenge" in path or "captcha" in path:
        return "challenge"
    if path.startswith("/checkpoint"):
        return "checkpoint"
    if path.startswith("/authwall"):
        return "authwall"
    if path.startswith(("/login", "/uas/", "/signup")) or path.startswith("/m/login"):
        return "login"
    return None


_PROBLEM_TEXT = {
    "login": "LinkedIn redirected to its login page: the session in LINKEDIN_COOKIE is "
             "expired or invalid; paste a fresh Cookie header in Settings",
    "authwall": "LinkedIn showed its authwall: the session in LINKEDIN_COOKIE is not "
                "accepted; paste a fresh Cookie header in Settings",
    "checkpoint": "LinkedIn showed a security checkpoint: open linkedin.com in your normal "
                  "browser and resolve it before collecting again",
    "challenge": "LinkedIn showed a challenge / CAPTCHA page: open linkedin.com in your "
                 "normal browser and resolve it before collecting again",
}

_SCROLL_JS = """() => {
  const hs = Array.from(document.querySelectorAll('h2'))
    .filter((e) => (e.textContent || '').trim() === 'Feed post');
  const last = hs.length ? hs[hs.length - 1] : null;
  if (last) last.scrollIntoView({ block: 'start' });
  const se = document.scrollingElement;
  if (se) se.scrollBy(0, window.innerHeight);
  const m = document.querySelector('main');
  if (m) m.scrollBy(0, m.clientHeight || 800);
  return hs.length;
}"""


def _title_problem(page: Any) -> str | None:
    """Challenge pages served without a redirect. Never checked on the search page itself:
    its title contains the analyst's keywords (a topic may well say "captcha")."""
    try:
        if urlparse(page.url or "").path.startswith(SEARCH_PATH):
            return None
        title = str(page.title() or "").lower()
    except Exception:  # noqa: BLE001
        return None
    if "security verification" in title or "captcha" in title or "challenge" in title:
        return "challenge"
    return None


def _req_type(resp: Any) -> str:
    try:
        return str(resp.request.resource_type or "")
    except Exception:  # noqa: BLE001
        return ""


def _kind(resp: Any) -> str | None:
    """'document' | 'pagination' | None for a captured response."""
    try:
        p = urlparse(resp.url or "")
    except Exception:  # noqa: BLE001
        return None
    if not p.netloc.endswith("linkedin.com"):
        return None
    if p.path.startswith(SEARCH_PATH) and _req_type(resp) in ("document", ""):
        return "document"
    if p.path == PAGINATION_PATH and f"sduiid={PAGER_ID}" in (p.query or ""):
        return "pagination"
    return None


def _kw(v: Any) -> str:
    """Keywords compared case- and spacing-insensitively (UNVERIFIED whether LinkedIn
    normalises the echoed keywords; a mismatch would silently drop every later page)."""
    return " ".join(str(v).split()).casefold()


def _truthy(v: Any, default: bool) -> bool:
    s = str(v if v is not None else "").strip().lower()
    if not s:
        return default
    return s in ("1", "true", "yes", "on")


def _int_cfg(v: Any, default: int) -> int:
    try:
        return max(1, min(MAX_SCROLLS_CAP, int(str(v).strip())))
    except (TypeError, ValueError):
        return default


# ================================================================ day walk


class _Walk:
    """Newest-first walk with no server date filter (mirrors Telegram's global search):
    every in-window, phrase-matching result is a candidate in its day slice; after the
    walk window.fill_slices splits the cap across slices, newest first within a slice."""

    def __init__(self, query: Query):
        self.query = query
        self.total = max(0, int(query.limit))
        self.slices = time_slices(query.start, query.end)
        self.n = len(self.slices)
        self.buckets: list[list[tuple[datetime, int, Post]]] = [[] for _ in self.slices]
        self.seen: set[str] = set()
        self.order = 0
        self.duplicates = 0
        self.newer = 0
        self.older = 0
        self.no_time = 0
        self.dropped_phrase = 0
        self.oldest: datetime | None = None
        self.dated: list[datetime] = []  # every dated result seen (frontier estimate)
        self.passed_start = False
        self.exhausted = False
        self.keep: list[int] = [0] * self.n
        # False while offering results that are NOT in date order (the Voyager relevance
        # pass): they never move the newest-first frontier or "passed the window start".
        self.ordered = True
        self.unordered = 0

    @property
    def candidates(self) -> int:
        return sum(len(b) for b in self.buckets)

    def offer(self, p: Post) -> bool:
        """True when the post was new to the walk (a result we had not seen)."""
        if p.post_id in self.seen:
            self.duplicates += 1
            return False
        self.seen.add(p.post_id)
        self.order += 1
        dt = datetime.fromisoformat(p.published_at.replace("Z", "+00:00")) \
            if p.published_at else None
        if dt is None:
            self.no_time += 1
            return True
        if self.ordered:
            if self.oldest is None or dt < self.oldest:
                self.oldest = dt
            self.dated.append(dt)
        else:
            self.unordered += 1
        if dt < self.query.start:
            self.older += 1
            if self.ordered:
                self.passed_start = self.older >= OLDER_TO_PASS
            return True
        j = next((i for i, (s, e) in enumerate(self.slices) if s <= dt <= e), None)
        if j is None:
            self.newer += 1
            return True
        if not topic_matches(self.query, p.text):
            self.dropped_phrase += 1
            return True
        self.buckets[j].append((dt, self.order, p))
        return True

    def full(self) -> bool:
        if self.total <= 0:
            return True
        need = -(-self.total // max(1, self.n))
        return self.n > 0 and all(len(b) >= need for b in self.buckets)

    def finish(self, exhausted: bool) -> None:
        self.exhausted = exhausted or self.passed_start or self.full()

    @property
    def frontier(self) -> datetime | None:
        """How far back the walk has really got. The date order is not strict (seen live:
        an 11-06 result before an 11-18 one), so a single stray old result must not mark
        the days between as searched: like OLDER_TO_PASS, it takes OLDER_TO_PASS results
        at or before a time to count it as reached."""
        if not self.dated:
            return None
        k = min(OLDER_TO_PASS, len(self.dated))
        return sorted(self.dated)[k - 1]

    def reached(self, i: int) -> bool:
        if self.exhausted:
            return True
        f = self.frontier
        return f is not None and f <= self.slices[i][1]

    def select(self) -> list[Post]:
        self.keep = fill_slices([len(b) for b in self.buckets], self.total)
        chosen: list[tuple[int, Post]] = []
        for i, b in enumerate(self.buckets):
            newest = sorted(b, key=lambda t: (-t[0].timestamp(), t[1]))[: self.keep[i]]
            chosen.extend((order, p) for _, order, p in newest)
        return [p for _, p in sorted(chosen, key=lambda t: t[0])]

    def summary(self) -> str:
        share = SliceShare(self.total, self.n)
        reached = [i for i in range(self.n) if self.reached(i)]
        fair = fill_slices([self.total if i in reached else 0 for i in range(self.n)],
                           self.total)
        for i in reached:
            share.cap[i] = fair[i]
            share.record(i, self.keep[i])
        return share.summary(self.slices)

    def notes(self) -> list[str]:
        reach = (f"reached back to {self.oldest:%Y-%m-%d %H:%M} UTC" if self.oldest
                 else "reached no dated result")
        out = [f"results walked newest first (no custom date range on LinkedIn), {reach}"]
        if self.newer:
            out.append(f"{self.newer} result(s) newer than the window end dropped")
        if self.older:
            out.append(f"{self.older} result(s) older than the window start dropped")
        if self.dropped_phrase:
            out.append(f"{self.dropped_phrase} result(s) without the exact phrase were not "
                       "kept (topic match: phrase; LinkedIn's own matching is looser)")
        if self.no_time:
            out.append(f"{self.no_time} result(s) whose ID did not decode to a date skipped")
        if self.duplicates:
            out.append(f"{self.duplicates} repeated result(s) ignored")
        if self.candidates > sum(self.keep):
            out.append(f"{self.candidates} in-window candidate(s); the cap of {self.total} "
                       "was split across the days reached, keeping the newest posts within "
                       "each day")
        return out


# ================================================================== collector


class _Stop(Exception):
    def __init__(self, why: str, reason: str):
        super().__init__(reason)
        self.why, self.reason = why, reason


@dataclass
class _Run:
    responses: int = 0
    pages: int = 0  # search responses parsed (document + pagination)
    scrolls: int = 0
    promoted: int = 0
    unparsed: int = 0
    first_items: int | None = None
    first_marker: bool = False
    payload: dict | None = None
    preset_echo: list = field(default_factory=list)
    ended: str | None = None  # exhausted | passed | full | cap | no_new | stop:<why>
    stop_reason: str | None = None
    ignored: int = 0


class LinkedInPostsCollector:
    name = SOURCE

    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_date_filter=False,
            needs_credentials=True,
            kind="social",
            credential_keys=("LINKEDIN_COOKIE",),
            description=("LinkedIn posts from linkedin.com's own content search with your "
                         "logged-in session (LINKEDIN_COOKIE), read in a local browser "
                         "(LINKEDIN_PROVIDER=browser) or through LinkedIn's Voyager search "
                         "API (voyager); no custom date range, results walked newest "
                         "first. " + ACCOUNT_RISK),
        )

    def collect(self, query: Query, ctx: CollectContext) -> CollectResult:
        cfg = ctx.config
        raw = str(cfg.get("LINKEDIN_COOKIE") or "").strip()
        keywords = search_topic(query)
        if not raw:
            return _result(keywords, SourceState.skipped_no_credentials, NO_COOKIE, query)
        if not has_li_at(raw):
            return _result(keywords, SourceState.skipped_no_credentials, NO_LI_AT, query)
        if not _lock.acquire(blocking=False):
            return _result(keywords, SourceState.failed, BUSY, query)
        try:
            from socmint.collect import linkedin_voyager

            if linkedin_voyager.provider(cfg) == "voyager":
                return linkedin_voyager.collect_voyager(query, ctx, raw)
            return _collect(query, ctx, raw)
        finally:
            _lock.release()


def _result(keywords: str, state: SourceState, reason: str, query: Query) -> CollectResult:
    slices = time_slices(query.start, query.end)
    summ = SliceShare(query.limit, len(slices)).summary(slices) if slices else ""
    return CollectResult(posts=[], status=[SourceStatus(
        source=SOURCE, query=keywords, state=state, date_filtered=False,
        started_at=now_iso(), finished_at=now_iso(),
        reason=f"{REASON_PREFIX}: {reason}" + (f"; {summ}" if summ else ""))])


def _collect(query: Query, ctx: CollectContext, raw_cookie: str) -> CollectResult:
    from socmint.collect import x_web  # generic browser launcher (no X specifics)

    cfg = ctx.config
    secrets = cookie_secrets(raw_cookie)
    headless = _truthy(cfg.get("LINKEDIN_BROWSER_HEADLESS"), True)
    max_scrolls = _int_cfg(cfg.get("LINKEDIN_MAX_SCROLLS"), DEFAULT_MAX_SCROLLS)
    url, keywords, preset = build_search_url(query, now_utc())
    status = SourceStatus(source=SOURCE, query=keywords, state=SourceState.ok,
                          date_filtered=False, started_at=now_iso(), cost_usd=0.0)
    walk = _Walk(query)
    run = _Run()
    inbox: list[Any] = []
    blocked: list[int] = []
    handle = None

    def on_response(r: Any) -> None:
        try:
            code = int(getattr(r, "status", 200) or 200)
            host = urlparse(r.url or "").netloc
        except Exception:  # noqa: BLE001
            return
        if host.endswith("linkedin.com") and code in (429, 999):
            blocked.append(code)
        if _kind(r):
            inbox.append(r)

    def save(n: int, kind: str, items: list[dict], payload: dict | None) -> str:
        body = json.dumps({"kind": kind, "note": "fields extracted from the search "
                           "response; the raw stream is not saved because it also carries "
                           "the logged-in account's own data",
                           "next_page_payload": payload, "items": items},
                          ensure_ascii=False, default=str)
        clean = scrub(body, secrets)
        return ctx.save_raw(SOURCE, f"search_page_{n}.json", clean)

    def handle_one(r: Any) -> int:
        """Parse one captured search response; returns how many results were new."""
        run.responses += 1
        status.requests += 1
        kind = _kind(r)
        code = int(getattr(r, "status", 200) or 200)
        if code in (429, 999):
            raise _Stop("rate", f"HTTP {code} from LinkedIn on a search response: rate "
                                "limited / blocked; stopped at once")
        if code in (301, 302, 303, 307, 308):
            return 0  # seen live: the search document first answers 307, then 200
        if code >= 400:
            raise _Stop("http", f"HTTP {code} from the search {kind} response"
                        + (" (session / CSRF problem: check LINKEDIN_COOKIE)"
                           if code in (401, 403) else ""))
        try:
            text = r.text()
        except Exception as e:  # noqa: BLE001
            run.unparsed += 1
            log.info("linkedin: unreadable %s response: %s", kind, type(e).__name__)
            return 0
        if kind == "document":
            items, payload, promoted, marker = parse_document(text)
        else:
            items, payload, promoted, marker = parse_stream(text)
        echo = payload.get("keywords") if payload else None
        if kind == "pagination" and echo is not None and _kw(echo) != _kw(keywords):
            run.ignored += 1  # a response for another search (e.g. a stale request)
            return 0
        run.pages += 1
        run.promoted += promoted
        if run.first_items is None:
            run.first_items, run.first_marker = len(items), marker
        run.payload = payload
        if payload and isinstance(payload.get("datePosted"), list):
            run.preset_echo = payload["datePosted"]
        collected = now_utc()
        ref = save(run.pages, kind, items, payload)
        new = 0
        for it in items:
            p = item_to_post(it, collected_at=collected, raw_ref=ref, query_str=keywords)
            if walk.offer(p):
                new += 1
        ctx.progress(SOURCE, {"state": "collecting", "count": min(walk.candidates,
                                                                  walk.total),
                              "message": f"linkedin: {walk.candidates} candidate posts "
                                         f"({run.pages} search responses)"})
        if not payload:
            run.ended = "exhausted"
        return new

    def check_page(page: Any) -> None:
        if blocked:
            raise _Stop("rate", f"HTTP {blocked[0]} from linkedin.com: rate limited / "
                                "blocked; stopped at once")
        problem = _url_problem(getattr(page, "url", "") or "") or _title_problem(page)
        if problem:
            raise _Stop(problem, _PROBLEM_TEXT[problem])

    def drain(page: Any, timeout_s: float) -> int | None:
        """Wait for search responses; None = none arrived before the timeout."""
        end = time.monotonic() + timeout_s
        got: int | None = None
        while True:
            check_page(page)
            if inbox:
                got = (got or 0)
                while inbox:
                    got += handle_one(inbox.pop(0))
                # give the second response of a burst (e.g. auto pagination) a moment
                end = min(end, time.monotonic() + 1.5)
            if ctx.out_of_time():
                raise _Stop("deadline", "time budget exhausted")
            if time.monotonic() >= end:
                return got
            page.wait_for_timeout(POLL_MS)

    try:
        try:
            handle = x_web._open_browser(headless=headless,
                                         cookies=build_cookie_dicts(raw_cookie))
        except x_web.NoBrowserError as e:
            log.warning("linkedin: no browser: %s", _redact(str(e), cfg))
            raise _Stop("browser", NO_BROWSER) from e
        page = handle.page
        page.on("response", on_response)
        page.goto(url, wait_until="domcontentloaded")
        got = drain(page, FIRST_RESPONSE_TIMEOUT_S)
        if got is None and run.pages == 0:
            raise _Stop("first", f"no search results response within "
                                 f"{FIRST_RESPONSE_TIMEOUT_S:.0f} s of opening the search "
                                 "page (layout or login state may have changed)")
        no_new = 0
        while run.ended is None:
            if walk.passed_start:
                run.ended = "passed"
                break
            if walk.full():
                run.ended = "full"
                break
            if run.scrolls >= max_scrolls:
                run.ended = "cap"
                break
            if ctx.out_of_time():
                raise _Stop("deadline", "time budget exhausted")
            page.wait_for_timeout(random.randint(*PAUSE_MS))
            mouse = getattr(page, "mouse", None)
            if mouse is not None:
                try:  # pointer over the results column, then a normal wheel scroll
                    mouse.move(VIEWPORT["width"] * 0.45, VIEWPORT["height"] * 0.6)
                    mouse.wheel(0, 2400)
                except Exception:  # noqa: BLE001
                    pass
            page.evaluate(_SCROLL_JS)
            run.scrolls += 1
            got = drain(page, RESPONSE_TIMEOUT_S)
            no_new = no_new + 1 if not got else 0
            if no_new >= NO_NEW_LIMIT and run.ended is None:
                run.ended = "no_new"
    except _Stop as s:
        run.ended, run.stop_reason = f"stop:{s.why}", s.reason
    except Exception as e:  # noqa: BLE001 — a browser failure must never look like zero
        run.ended = "stop:error"
        run.stop_reason = _redact(f"browser error after {run.pages} search response(s): "
                                  f"{type(e).__name__}: {e}", cfg)
        log.warning("linkedin collection error: %s", run.stop_reason)
    finally:
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass

    # LinkedIn caps search results (live 2026-11-19: ~245 for a week of a busy topic, which
    # reached back only ~5 h). Results ending before the walk passed the window start
    # means older days were NOT searched; they must not be shown as empty.
    capped = (run.ended == "exhausted" and len(walk.dated) >= RESULT_CAP_SUSPECT
              and not (walk.passed_start or walk.full()))
    walk.finish(run.ended in ("passed", "full") or (run.ended == "exhausted" and not capped))
    posts = walk.select()
    status.count = len(posts)
    notes: list[str] = []
    stop = run.ended or ""
    if capped:
        state = SourceState.partial
        f = walk.frontier
        notes.append("LinkedIn returned no further results page"
                     + (f" at {f:%Y-%m-%d %H:%M} UTC" if f else "")
                     + ", before the window start: LinkedIn caps search results, so older "
                       "days were not reached (not the same as no posts)")
    elif stop.startswith("stop:"):
        # failed only when nothing was collected because of the error; the time budget
        # running out is a cut, never a failure.
        state = (SourceState.partial if posts or stop == "stop:deadline"
                 else SourceState.failed)
        notes.append(f"stopped: {run.stop_reason}")
    elif walk.total <= 0:
        state = SourceState.partial
        notes.append("post limit is 0; nothing kept")
    elif stop in ("exhausted", "passed", "full"):
        if posts:
            state = SourceState.ok
        elif run.pages and (run.first_marker or run.first_items):
            state = SourceState.completed_zero
        else:
            state = SourceState.failed
            notes.append("the search page did not contain the expected results structure "
                         "(layout may have changed); not reported as zero posts")
        notes.append({"exhausted": "LinkedIn returned no further results page",
                      "passed": f"the walk passed the window start ({OLDER_TO_PASS} results "
                                "older than it; LinkedIn's date order is not strict)",
                      "full": "stopped early: every day already had its share of candidates"
                      }[stop])
    elif stop == "cap":
        state = SourceState.partial
        notes.append(f"stopped at LINKEDIN_MAX_SCROLLS={max_scrolls} before the window "
                     "start; older posts in the window may exist")
    else:  # no_new
        state = SourceState.partial
        notes.append(f"{NO_NEW_LIMIT} consecutive scrolls brought no new results before "
                     "the window start (LinkedIn stopped loading); older posts may exist")
    if state == SourceState.partial and run.payload:
        idx = max([i for i in range(walk.n) if walk.reached(i)], default=0)
        day = f"{walk.slices[idx][0]:%Y-%m-%d}" if walk.slices else ""
        status.cursor = f"{day}:startIndex={run.payload.get('startIndex')}"
    head = [f"datePosted={preset or 'any time'}, sortBy=date_posted, keywords {keywords}"]
    if preset and run.preset_echo and preset not in run.preset_echo:
        head.append(f"LinkedIn echoed datePosted={run.preset_echo}, not {preset}")
    extras = [f"{run.scrolls} scroll(s), {run.pages} search response(s) parsed"]
    if run.ignored:
        extras.append(f"{run.ignored} response(s) for another search ignored")
    if run.unparsed:
        extras.append(f"{run.unparsed} response(s) unreadable")
    if run.promoted:
        extras.append(f"{run.promoted} promoted item(s) skipped")
    extras.append("published_at decoded from post IDs, checked against each card's "
                  "relative time label")
    if headless:
        extras.append("headless browser (LINKEDIN_BROWSER_HEADLESS=true)")
    parts = head + notes + walk.notes() + extras + [walk.summary()]
    status.state = state
    status.reason = _redact(f"{REASON_PREFIX}: " + "; ".join(parts), cfg)
    status.finished_at = now_iso()
    return CollectResult(posts=posts, status=[status])


__all__ = [
    "LinkedInPostsCollector",
    "build_cookie_dicts",
    "build_search_url",
    "check_timestamp",
    "choose_date_preset",
    "decode_id_time",
    "extract_items",
    "has_li_at",
    "parse_cookie_header",
    "parse_document",
    "parse_flight",
    "parse_stream",
]
