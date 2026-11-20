"""LinkedIn posts, provider "voyager": LinkedIn's Voyager GraphQL content search, called
directly with the owner's OWN session (config LINKEDIN_PROVIDER=voyager).

Owner decision 2026-11-19 (CLAUDE.md §2): the same request linkedin.com's web app sends
for its content search, made with the owner's own session cookie and its CSRF token
(LinkedIn's standard double-submit CSRF: the csrf-token header repeats the JSESSIONID
cookie value without its quotes). The browser provider (linkedin_posts.py) stays the
default.

What this module does and does NOT do:
  - GET https://www.linkedin.com/voyager/api/graphql?variables=(start:N,count:N,
    origin:SWITCH_SEARCH_VERTICAL,query:(keywords:<urlencoded>,flagshipSearchIntent:
    SEARCH_SRP,queryParameters:List(<params>),includeFiltersInResponse:false))
    &queryId=<LINKEDIN_VOYAGER_QUERY_ID>
    with headers cookie, csrf-token, x-restli-protocol-version 2.0.0, accept
    application/vnd.linkedin.normalized+json+2.1, accept-language, user-agent
    (LINKEDIN_USER_AGENT, a desktop Chrome string matching the owner's browser).
  - No anti-bot, fingerprint or challenge tokens are generated or sent; no stealth.
    Sequential, one page at a time, a random 3-6 s pause between pages, module lock
    shared with the browser provider (one LinkedIn collection per account).
  - It stops at once on HTTP 429 / 999, 401 / 403, any redirect (login, authwall,
    checkpoint, challenge or other), a non-JSON answer, and on 400 / 404 / 5xx (the
    persisted queryId may be outdated: LinkedIn rotates it with deploys).

Seen live 2026-11-19 (lead's 2 captures, count 10, "bank of america" in quotes):
  - HTTP 200, application/vnd.linkedin.normalized+json+2.1; body {data:{data:
    {searchDashClustersByAll:{paging:{count,start,total}, elements:[...]}}}, included:[...]}.
    Result order = elements[].items[].item.searchFeedUpdate["*update"] (an fsd_update urn);
    each resolves to a com.linkedin.voyager.dash.feed.Update in `included`.
  - Update: metadata.backendUrn urn:li:activity:<id>, metadata.shareUrn; actor.name.text,
    actor.subDescription.text ("7m •", "3w • Edited •"), actor.navigationContext
    .actionTarget (/in/<vanity>?miniProfileUrn=... or /company/<slug>/posts);
    commentary.text.text (original language) with attributesV2 (*profileMention ->
    Profile.publicIdentifier, *companyName -> Company.url); content.imageComponent /
    articleComponent / entityComponent / pollComponent; *socialDetail -> SocialDetail
    -> *totalSocialActivityCounts -> SocialActivityCounts (numLikes = sum of
    reactionTypeCounts, numComments, numShares).
  - resultType CONTENT alone returns relevance order; adding datePosted + sortBy
    date_posted (params sorted by key, as the web app does) returned newest first.
  - The response ALSO carries the viewer's own member id (socialPermissions urns, the
    rcm= parameter of socialContent.shareUrl). The response is therefore NEVER saved:
    raw/linkedin/voyager_page_<n>.json holds only the fields read per item + paging.

Evidence rules: identical to the browser provider (item_to_post): post_id
linkedin:<activity id>, url /feed/update/urn:li:activity:<id>/, published_at decoded
from the activity id and verified against the card's relative label.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlparse, urlunparse

import httpx

from socmint.collect import linkedin_posts as lp
from socmint.collect.window import SliceShare, search_topic, time_slices
from socmint.interfaces import CollectContext, CollectResult, Query
from socmint.schema import Media, Metrics, Relation, SourceState, SourceStatus
from socmint.util.log import get_logger
from socmint.util.timeutil import now_iso, now_utc

log = get_logger(__name__)

SOURCE = lp.SOURCE
BASE = lp.BASE
GRAPHQL_URL = BASE + "/voyager/api/graphql"
DEFAULT_QUERY_ID = "voyagerSearchDashClusters.5ba32757c00b31aea747c8bebb92855c"
# A desktop Chrome user agent matching the owner's browser (macOS Chrome); change it in
# Settings to your own browser's string.
DEFAULT_USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36")
ACCEPT = "application/vnd.linkedin.normalized+json+2.1"
ACCEPT_LANGUAGE = "en-US,en;q=0.9"
PROVIDERS = ("browser", "voyager")
DEFAULT_PROVIDER = "browser"
QUERY_ID_RE = re.compile(r"^[A-Za-z0-9_-]+\.[0-9A-Za-z]+$")

REASON_PREFIX = "LinkedIn Voyager content search API (owner's logged-in account)"
NO_JSESSIONID = ("LINKEDIN_COOKIE has no JSESSIONID cookie: the voyager provider needs the "
                 "full Cookie header of a logged-in linkedin.com tab (li_at and JSESSIONID)")

# Timing / paging (module constants so tests can shrink them).
PAUSE_S = (3.0, 6.0)  # random pause between pages
TIMEOUT_S = 30.0
# Verified live 2026-11-19: count 10 (lead) and count 25 (paging.count 25, 25 Updates)
# are accepted. Larger counts were not tried.
DEFAULT_COUNT = 25
# Relevance pass (owner's LinkedIn tool, ../linkedin/lib/scrape.ts + voyager.ts, owner
# request 2026-11-20): no sortBy, so LinkedIn returns relevance order, which it pages much
# deeper than date order. Live 2026-11-20 ("sahibinden"): paging.total 494 by relevance vs
# 244 with sortBy=date_posted. The owner's tool pages to start >= min(total, 1000) and only
# gives up after EMPTY_PAGE_TOLERANCE consecutive pages without posts (LinkedIn interleaves
# non-post cards).
RESULT_CEILING = 1000
EMPTY_PAGE_TOLERANCE = 2
QUERY_ID_HINT = ("the persisted queryId may be outdated (LinkedIn rotates it with deploys): "
                 "open linkedin.com in your browser, DevTools > Network, run a posts search, "
                 "find the request to /voyager/api/graphql whose queryId starts with "
                 "voyagerSearchDashClusters. and paste that queryId into "
                 "LINKEDIN_VOYAGER_QUERY_ID in Settings")


def provider(cfg: dict) -> str:
    v = str(cfg.get("LINKEDIN_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    return v if v in PROVIDERS else DEFAULT_PROVIDER


def query_id(cfg: dict) -> str:
    return str(cfg.get("LINKEDIN_VOYAGER_QUERY_ID") or "").strip() or DEFAULT_QUERY_ID


def user_agent(cfg: dict) -> str:
    return str(cfg.get("LINKEDIN_USER_AGENT") or "").strip() or DEFAULT_USER_AGENT


# =================================================================== request


def jsessionid(raw: str | None) -> str | None:
    """JSESSIONID cookie value without its quotes (the csrf-token header), or None."""
    for n, v in lp.parse_cookie_header(raw):
        if n == "JSESSIONID" and v.strip('"'):
            return v.strip('"')
    return None


def cookie_header(raw: str | None) -> str:
    """The Cookie header as pasted (a leading 'Cookie:' removed; a bare li_at named)."""
    pairs = lp.parse_cookie_header(raw)
    return "; ".join(f"{n}={v}" for n, v in pairs)


def build_headers(raw_cookie: str, cfg: dict) -> dict[str, str]:
    return {
        "cookie": cookie_header(raw_cookie),
        "csrf-token": jsessionid(raw_cookie) or "",
        "x-restli-protocol-version": "2.0.0",
        "accept": ACCEPT,
        "accept-language": ACCEPT_LANGUAGE,
        "user-agent": user_agent(cfg),
    }


def query_params(preset: str | None, by_date: bool = True) -> list[tuple[str, str]]:
    """(key, value) search parameters, sorted by key as linkedin.com's web app sends them.
    by_date=False leaves out sortBy (relevance order, as the owner's LinkedIn tool does)."""
    params = [("resultType", "CONTENT")]
    if by_date:
        params.append(("sortBy", "date_posted"))
    if preset:
        params.append(("datePosted", preset))
    return sorted(params)


def build_url(keywords: str, start: int, count: int, preset: str | None, qid: str,
              by_date: bool = True) -> str:
    plist = ",".join(f"(key:{k},value:List({v}))" for k, v in query_params(preset, by_date))
    variables = (f"(start:{int(start)},count:{int(count)},origin:SWITCH_SEARCH_VERTICAL,"
                 f"query:(keywords:{quote(keywords, safe='')},flagshipSearchIntent:SEARCH_SRP,"
                 f"queryParameters:List({plist}),includeFiltersInResponse:false))")
    return f"{GRAPHQL_URL}?variables={variables}&queryId={qid}"


def make_http_client() -> httpx.Client:
    """Redirects are NOT followed: a redirect is a login / checkpoint signal and stops."""
    from socmint.util.http import _ssl_context

    return httpx.Client(timeout=TIMEOUT_S, follow_redirects=False, verify=_ssl_context())


# =================================================================== parsing


def _by_urn(included: list) -> dict[str, dict]:
    return {x["entityUrn"]: x for x in included
            if isinstance(x, dict) and isinstance(x.get("entityUrn"), str)}


def _search(body: Any) -> dict | None:
    try:
        s = body["data"]["data"]["searchDashClustersByAll"]
    except (KeyError, TypeError):
        return None
    return s if isinstance(s, dict) else None


def _txt(v: Any) -> str:
    return str(v.get("text") or "") if isinstance(v, dict) else ""


def _largest(vector: Any) -> str | None:
    if not isinstance(vector, dict) or not isinstance(vector.get("rootUrl"), str):
        return None
    arts = [a for a in vector.get("artifacts") or [] if isinstance(a, dict)
            and a.get("fileIdentifyingUrlPathSegment")]
    if not arts:
        return None
    a = max(arts, key=lambda a: int(a.get("width") or 0))
    return vector["rootUrl"] + str(a["fileIdentifyingUrlPathSegment"])


def _vectors(node: Any):
    """Every VectorImage under node (depth first)."""
    stack = [node]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            if isinstance(x.get("rootUrl"), str) and isinstance(x.get("artifacts"), list):
                yield x
                continue
            stack.extend(reversed(list(x.values())))
        elif isinstance(x, list):
            stack.extend(reversed(x))


def _first_image(node: Any) -> str | None:
    for v in _vectors(node):
        u = _largest(v)
        if u:
            return u
    return None


def _clean_li_url(url: str) -> str:
    """A linkedin.com URL without its query string (it can carry tracking ids or, seen
    live in shareUrl, the viewer's own member id as rcm=)."""
    p = urlparse(url)
    if p.netloc.endswith("linkedin.com"):
        return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
    return url


_URL_IN_TEXT = re.compile(r"https?://[^\s<>\"']+")


def _links_in_text(text: str) -> list[str]:
    out = []
    for m in _URL_IN_TEXT.finditer(text or ""):
        u = m.group(0).rstrip(".,;:!?)]}»”’")
        if u not in out:
            out.append(u)
    return out


def _activity_from(urn: Any) -> str | None:
    m = re.search(r"urn:li:activity:(\d+)", str(urn or ""))
    return m.group(1) if m else None


def _actor(update: dict) -> dict:
    actor = update.get("actor") if isinstance(update.get("actor"), dict) else {}
    target = (actor.get("navigationContext") or {}).get("actionTarget")
    ph = lp._profile_handle(target) if isinstance(target, str) and target else None
    sub = _txt(actor.get("subDescription")).strip()
    return {"handle": ph[0] if ph else None, "url": ph[1] if ph else None,
            "name": _txt(actor.get("name")).strip() or None,
            "avatar": _first_image(actor.get("image")), "sub": sub}


def _relations(update: dict, idx: dict[str, dict], own: str | None) -> list[Relation]:
    rels: list[Relation] = []
    handles: set[str] = set()
    urls: set[str] = set()

    def add_handle(ph: tuple[str, str] | None) -> None:
        if ph and ph[0] != own and ph[0] not in handles:
            handles.add(ph[0])
            rels.append(Relation(type="mention", target_handle=ph[0], target_url=ph[1]))

    def add_url(u: str | None) -> None:
        if not u or not u.lower().startswith(("http://", "https://")):
            return
        p = urlparse(u)
        if p.netloc.endswith("linkedin.com") and p.path.startswith("/search/"):
            return  # hashtag / keyword search link
        real = lp._real_url(u)
        if not real:
            return
        ph = lp._profile_handle(real)
        if ph:
            add_handle(ph)
            return
        real = _clean_li_url(real)
        if real not in urls:
            urls.add(real)
            rels.append(Relation(type="link", target_url=real))

    text_vm = (update.get("commentary") or {}).get("text") or {}
    for a in text_vm.get("attributesV2") or []:
        dd = (a or {}).get("detailData") or {}
        prof = idx.get(str(dd.get("*profileMention") or ""))
        if prof and prof.get("publicIdentifier"):
            vanity = str(prof["publicIdentifier"])
            add_handle((f"in/{vanity}", f"{BASE}/in/{quote(vanity, safe='')}/"))
        comp = idx.get(str(dd.get("*companyName") or ""))
        if comp and isinstance(comp.get("url"), str):
            add_handle(lp._profile_handle(comp["url"]))
        if isinstance(dd.get("hyperlink"), str):
            add_url(dd["hyperlink"])
    for u in _links_in_text(_txt(text_vm)):
        add_url(u)
    content = update.get("content") if isinstance(update.get("content"), dict) else {}
    for name in ("articleComponent", "entityComponent"):
        comp = content.get(name)
        if isinstance(comp, dict):
            add_url((comp.get("navigationContext") or {}).get("actionTarget"))
    return rels


def _media(update: dict) -> list[Media]:
    out: list[Media] = []
    content = update.get("content") if isinstance(update.get("content"), dict) else {}
    img = content.get("imageComponent")
    if isinstance(img, dict):  # seen live
        for im in img.get("images") or []:
            u = _first_image(im)
            if u and all(m.url != u for m in out):
                out.append(Media(type="photo", url=u))
    for name, comp in content.items():
        # UNVERIFIED (no video post captured live yet): a video component is recorded with
        # its cover image; the stream URL is not read.
        if isinstance(comp, dict) and "video" in name.lower():
            out.append(Media(type="video", url=None, thumbnail=_first_image(comp)))
    return out


def _metrics(update: dict, idx: dict[str, dict]) -> Metrics:
    sd = idx.get(str(update.get("*socialDetail") or ""))
    counts = idx.get(str((sd or {}).get("*totalSocialActivityCounts") or "")) if sd else None
    if not counts:
        return Metrics()

    def num(k: str) -> int | None:
        v = counts.get(k)
        return v if isinstance(v, int) else None
    return Metrics(reactions=num("numLikes"), comments=num("numComments"),
                   reposts=num("numShares"))


def _reshare(update: dict, idx: dict[str, dict], has_text: bool) -> dict | None:
    """An embedded resharedUpdate (inline or by urn) is a repost, or a quote when the outer
    post has its own text. Seen live 2026-11-19 (provider run, 3 quotes parsed from
    resharedUpdate); a pure repost without text is UNVERIFIED."""
    inner = update.get("resharedUpdate")
    if not isinstance(inner, dict):
        inner = idx.get(str(update.get("*resharedUpdate") or ""))
    if not isinstance(inner, dict):
        return None
    aid = _activity_from((inner.get("metadata") or {}).get("backendUrn")) \
        or _activity_from(inner.get("entityUrn"))
    if not aid:
        return None
    a = _actor(inner)
    return {"urn": f"urn:li:activity:{aid}", "author": a["handle"], "author_url": a["url"],
            "quote": has_text}


def result_refs(body: Any) -> list[str]:
    """The fsd_update urns of the post results, in result order
    (elements[].items[].item.searchFeedUpdate["*update"])."""
    s = _search(body) or {}
    refs: list[str] = []
    for el in s.get("elements") or []:
        for it in (el or {}).get("items") or []:
            sfu = ((it or {}).get("item") or {}).get("searchFeedUpdate")
            ref = (sfu or {}).get("*update") if isinstance(sfu, dict) else None
            if isinstance(ref, str) and ref not in refs:
                refs.append(ref)
    return refs


def extract_page(body: Any) -> tuple[list[dict], dict, int] | None:
    """(items in result order, paging {start,count,total}, promoted skipped) from one
    Voyager search response, or None when the expected structure is absent.

    An item is a plain dict with the same keys as the browser provider's items (so
    linkedin_posts.item_to_post applies unchanged); it holds no data about the viewer and
    is what gets saved as the raw record.
    """
    s = _search(body)
    if s is None:
        return None
    included = body.get("included") if isinstance(body.get("included"), list) else []
    idx = _by_urn(included)
    paging = s.get("paging") if isinstance(s.get("paging"), dict) else {}
    page = {k: paging.get(k) for k in ("start", "count", "total")}
    refs = result_refs(body)
    items: list[dict] = []
    promoted = 0
    seen: set[str] = set()
    for ref in refs:
        u = idx.get(ref)
        if not isinstance(u, dict):
            continue
        meta = u.get("metadata") if isinstance(u.get("metadata"), dict) else {}
        aid = _activity_from(meta.get("backendUrn")) or _activity_from(u.get("entityUrn"))
        if not aid or aid in seen:
            continue
        seen.add(aid)
        a = _actor(u)
        if re.search(r"\bPromoted\b", a["sub"]):
            promoted += 1
            continue
        m = lp._TIME_LABEL.match(a["sub"])
        label = re.sub(r"\s+", "", m.group(1)).lower() if m else None
        text = _txt((u.get("commentary") or {}).get("text"))
        share = (u.get("socialContent") or {}).get("shareUrl")
        backend = str(meta.get("backendUrn") or f"urn:li:activity:{aid}")
        urns = sorted({backend, *([meta["shareUrn"]] if meta.get("shareUrn") else [])})
        items.append({
            "native_id": aid,
            "activity_id": aid,
            "content_urn": backend,
            "urns": urns,
            "share_url": _clean_li_url(share) if isinstance(share, str) and share else None,
            "author_handle": a["handle"],
            "author_url": a["url"],
            "author_name": a["name"],
            "author_avatar_url": a["avatar"],
            "time_label": label,
            "time_line": a["sub"][:80],
            "text": text,
            "media": [md.model_dump() for md in _media(u)],
            "relations": [r.model_dump() for r in _relations(u, idx, a["handle"])],
            "metrics": _metrics(u, idx).model_dump(),
            "reshare": _reshare(u, idx, bool(text.strip())),
        })
    return items, page, promoted


# ================================================================== collector


class _Stop(Exception):
    def __init__(self, why: str, reason: str):
        super().__init__(reason)
        self.why, self.reason = why, reason


@dataclass
class _Run:
    pages: int = 0
    requests: int = 0
    promoted: int = 0
    start: int = 0  # next start index
    total: int | None = None
    ended: str | None = None  # exhausted | total | ceiling | passed | full | cap | stop:<why>
    stop_reason: str | None = None
    empty_streak: int = 0
    new: int = 0  # results this pass added to the walk (not seen by an earlier pass)


def _cut_by_linkedin_cap(run: _Run, walk: Any) -> bool:
    """Date-ordered results ended before the window start on a big search: LinkedIn's cap
    (seen live 2026-11-19: paging.total 245 for a week of a busy topic), not the end of
    the posts. Older days were then not searched."""
    walked = max(run.total or 0, len(walk.dated))
    return (run.ended in ("exhausted", "total") and bool(walk.dated)
            and walked >= lp.RESULT_CAP_SUSPECT
            and not (walk.passed_start or walk.full()))


def check_response(resp: Any) -> Any:
    """JSON body of a 200 answer; _Stop for every other outcome (never retried)."""
    code = int(getattr(resp, "status_code", 0) or 0)
    if code in (429, 999):
        raise _Stop("rate", f"HTTP {code} from LinkedIn: rate limited / blocked; stopped at once")
    if 300 <= code < 400:
        loc = str(resp.headers.get("location") or "")
        problem = lp._url_problem(loc)
        if problem:
            raise _Stop(problem, lp._PROBLEM_TEXT[problem])
        raise _Stop("redirect", f"HTTP {code} redirect from the Voyager search (to "
                                f"{urlparse(loc).path or 'an unknown page'}); stopped at once")
    if code in (401, 403):
        raise _Stop("auth", f"HTTP {code} from the Voyager search: the session or its CSRF "
                            "token was refused (check LINKEDIN_COOKIE: it needs li_at and "
                            "JSESSIONID from a logged-in tab); stopped at once")
    if code in (400, 404) or code >= 500:
        raise _Stop("query", f"HTTP {code} from the Voyager search: {QUERY_ID_HINT}")
    if code != 200:
        raise _Stop("http", f"HTTP {code} from the Voyager search; stopped at once")
    ctype = str(resp.headers.get("content-type") or "").lower()
    if "json" not in ctype:
        raise _Stop("challenge", f"the Voyager search answered {ctype or 'no content type'}, "
                                 "not JSON (possibly a challenge page); stopped at once")
    try:
        return json.loads(resp.text)
    except ValueError as e:
        raise _Stop("json", "the Voyager search answer was not valid JSON; stopped at once") \
            from e


def collect_voyager(query: Query, ctx: CollectContext, raw_cookie: str) -> CollectResult:
    """Called by LinkedInPostsCollector.collect with the module lock held."""
    cfg = ctx.config
    keywords = search_topic(query)
    if not jsessionid(raw_cookie):
        slices = time_slices(query.start, query.end)
        summ = SliceShare(query.limit, len(slices)).summary(slices) if slices else ""
        return CollectResult(posts=[], status=[SourceStatus(
            source=SOURCE, query=keywords, state=SourceState.skipped_no_credentials,
            date_filtered=False, started_at=now_iso(), finished_at=now_iso(),
            reason=f"{REASON_PREFIX}: {NO_JSESSIONID}" + (f"; {summ}" if summ else ""))])
    secrets = lp.cookie_secrets(raw_cookie)
    qid = query_id(cfg)
    count = DEFAULT_COUNT
    max_pages = lp._int_cfg(cfg.get("LINKEDIN_MAX_SCROLLS"), lp.DEFAULT_MAX_SCROLLS)
    preset = lp.choose_date_preset(query.start, now_utc())
    headers = build_headers(raw_cookie, cfg)
    status = SourceStatus(source=SOURCE, query=keywords, state=SourceState.ok,
                          date_filtered=False, started_at=now_iso(), cost_usd=0.0)
    walk = lp._Walk(query)
    run = _Run()

    def save(n: int, start: int, items: list[dict], page: dict, by_date: bool = True) -> str:
        body = json.dumps({
            "kind": "voyager", "note": "fields extracted from the Voyager search response; "
            "the response is not saved because it also carries the logged-in account's own "
            "member id", "request": {"start": start, "count": count, "keywords": keywords,
                                     "params": query_params(preset, by_date),
                                     "queryId": qid},
            "paging": page, "items": items}, ensure_ascii=False, default=str)
        return ctx.save_raw(SOURCE, f"voyager_page_{n}.json", lp.scrub(body, secrets))

    run2: _Run | None = None  # relevance pass (only when the date pass ran out early)
    saved = 0

    def page_pass(client: httpx.Client, r: _Run, by_date: bool, max_p: int) -> None:
        """Page one search order into the shared walk until it ends; sets r.ended."""
        nonlocal saved
        walk.ordered = by_date
        while r.ended is None:
            if by_date and walk.passed_start:
                r.ended = "passed"
                break
            if walk.full():
                r.ended = "full"
                break
            if r.total is not None and r.start >= r.total:
                r.ended = "total"
                break
            if not by_date and r.start >= RESULT_CEILING:
                r.ended = "ceiling"
                break
            if r.pages >= max_p:
                r.ended = "cap"
                break
            if ctx.out_of_time():
                raise _Stop("deadline", "time budget exhausted")
            if run.requests:
                time.sleep(random.uniform(*PAUSE_S))
                if ctx.out_of_time():
                    raise _Stop("deadline", "time budget exhausted")
            start = r.start
            url = build_url(keywords, start, count, preset, qid, by_date)
            run.requests += 1
            status.requests += 1
            body = check_response(client.get(url, headers=headers))
            parsed = extract_page(body)
            if parsed is None:
                raise _Stop("structure", "the Voyager answer had no searchDashClustersByAll "
                                         "(response shape changed, or " + QUERY_ID_HINT + ")")
            items, page, promoted = parsed
            r.pages += 1
            r.promoted += promoted
            if isinstance(page.get("total"), int):
                r.total = page["total"]
            r.start = start + count
            collected = now_utc()
            saved += 1
            ref = save(saved, start, items, page, by_date)
            for it in items:
                if walk.offer(lp.item_to_post(it, collected_at=collected, raw_ref=ref,
                                              query_str=keywords)):
                    r.new += 1
            ctx.progress(SOURCE, {"state": "collecting",
                                  "count": min(walk.candidates, walk.total),
                                  "message": f"linkedin: {walk.candidates} candidate posts "
                                             f"({run.requests} Voyager page(s))"})
            if not items and not promoted:
                refs = result_refs(body)
                if refs:
                    # Result references that resolve to no Update: the response shape
                    # changed. Never reported as zero posts.
                    raise _Stop("structure", f"the Voyager page listed {len(refs)} result(s) "
                                             "but none resolved to a post Update in "
                                             "`included` (response shape changed)")
                if start == 0 and isinstance(r.total, int) and r.total > 0:
                    raise _Stop("structure", f"LinkedIn reported paging.total {r.total} "
                                             "but the first page held no post results "
                                             "(response shape changed?); not reported as "
                                             "zero posts")
                r.empty_streak += 1
                if start == 0 or r.empty_streak >= EMPTY_PAGE_TOLERANCE:
                    r.ended = "exhausted"
            else:
                r.empty_streak = 0

    client = None
    try:
        client = make_http_client()
        page_pass(client, run, True, max_pages)
        # Date order hit LinkedIn's cap (~250 results) before the window start: page the
        # relevance order too, as the owner's tool does (deeper, different set). Not after
        # a small search that date order paged completely, nor after LINKEDIN_MAX_SCROLLS.
        if _cut_by_linkedin_cap(run, walk):
            run2 = _Run()
            page_pass(client, run2, False, -(-RESULT_CEILING // count))
    except _Stop as s:
        cur = run2 if run2 is not None else run
        cur.ended, cur.stop_reason = f"stop:{s.why}", s.reason
    except httpx.HTTPError as e:
        cur = run2 if run2 is not None else run
        cur.ended = "stop:network"
        cur.stop_reason = lp._redact(f"network error after {run.requests} page(s): "
                                     f"{type(e).__name__}: {e}", cfg)
    except Exception as e:  # noqa: BLE001 - an error must never look like zero posts
        cur = run2 if run2 is not None else run
        cur.ended = "stop:error"
        cur.stop_reason = lp._redact(f"error after {run.requests} page(s): "
                                     f"{type(e).__name__}: {e}", cfg)
        log.warning("linkedin voyager error: %s", cur.stop_reason)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

    # Seen live 2026-11-19: LinkedIn reported paging.total 245 for a week of "bank of
    # america" while the topic produced ~75 posts per 90 minutes, so the results ended
    # ~5 h back. LinkedIn caps search results: when they end before the walk passed the
    # window start, older days were NOT searched and must not be shown as empty.
    capped = _cut_by_linkedin_cap(run, walk)
    # The relevance pass paged every result LinkedIn reported for the query (below its
    # 1000 ceiling): nothing more can be had from this search, so every day counts as
    # searched even where date order did not reach.
    rel_done = run2 is not None and (
        run2.ended in ("total", "full")
        or (run2.ended == "exhausted" and (run2.total or 0) <= run2.start))
    walk.finish(run.ended in ("passed", "full")
                or (run.ended in ("exhausted", "total") and not capped) or rel_done)
    posts = walk.select()
    status.count = len(posts)
    notes: list[str] = []
    stop = run.ended or ""
    stop2 = (run2.ended or "") if run2 is not None else ""
    if stop.startswith("stop:") or stop2.startswith("stop:"):
        why = stop if stop.startswith("stop:") else stop2
        state = (SourceState.partial if posts or why == "stop:deadline"
                 else SourceState.failed)
        notes.append(f"stopped{' during the relevance pass' if why is stop2 else ''}: "
                     f"{(run.stop_reason if why is stop else run2.stop_reason)}")
    elif walk.total <= 0:
        state = SourceState.partial
        notes.append("post limit is 0; nothing kept")
    elif run2 is not None:
        f = walk.frontier
        notes.append(
            f"date order ended ({'paging.total ' + str(run.total) if stop == 'total' else stop})"
            + (f" at {f:%Y-%m-%d %H:%M} UTC" if f else "")
            + " before the window start (LinkedIn caps date-sorted results), so the "
              "relevance order was paged too, as the owner's LinkedIn tool does")
        if rel_done:
            state = SourceState.ok if posts else SourceState.completed_zero
            notes.append(f"relevance order: all {run2.total} result(s) LinkedIn reported were "
                         f"paged ({run2.new} not already seen by date order)")
        else:
            state = SourceState.partial
            limit_txt = ("LinkedIn's 1000-result ceiling" if stop2 == "ceiling"
                         else f"result {run2.start} (LinkedIn returned empty pages)"
                         if stop2 == "exhausted" else f"{run2.pages} page(s)")
            notes.append(f"relevance order: stopped at {limit_txt} of paging.total "
                         f"{run2.total} ({run2.new} not already seen by date order); older "
                         "posts in the window may exist")
    elif capped:
        state = SourceState.partial if posts or walk.dated else SourceState.completed_zero
        f = walk.frontier
        notes.append(
            f"LinkedIn's results ended ({'paging.total ' + str(run.total) if stop == 'total' else 'empty page'})"
            + (f" at {f:%Y-%m-%d %H:%M} UTC" if f else "")
            + ", before the window start: LinkedIn caps search results, so older days were "
              "not reached (not the same as no posts)")
    elif stop in ("exhausted", "total", "passed", "full"):
        state = SourceState.ok if posts else SourceState.completed_zero
        notes.append({
            "exhausted": "LinkedIn returned a page with no results (end of the results)",
            "total": f"all {run.total} result(s) LinkedIn reported (paging.total) were paged",
            "passed": f"the walk passed the window start ({lp.OLDER_TO_PASS} results older "
                      "than it; LinkedIn's date order is not strict)",
            "full": "stopped early: every day already had its share of candidates"}[stop])
    else:  # cap
        state = SourceState.partial
        notes.append(f"stopped at LINKEDIN_MAX_SCROLLS={max_pages} page(s) before the window "
                     "start; older posts in the window may exist")
    if state == SourceState.partial and run.pages:
        idx = max([i for i in range(walk.n) if walk.reached(i)], default=0)
        day = f"{walk.slices[idx][0]:%Y-%m-%d}" if walk.slices else ""
        status.cursor = f"{day}:start={run.start}" + (
            f":relevance_start={run2.start}" if run2 is not None else "")
    head = [f"datePosted={preset or 'any time'}, sortBy=date_posted"
            + (" then relevance order" if run2 is not None else "")
            + f", keywords {keywords}, queryId {qid}"]
    extras = [f"{run.pages} date-order page(s) of {count} parsed"
              + (f" of paging.total {run.total}" if run.total is not None else "")]
    if run2 is not None:
        extras.append(f"{run2.pages} relevance-order page(s)"
                      + (f" of paging.total {run2.total}" if run2.total is not None else ""))
    promoted = run.promoted + (run2.promoted if run2 is not None else 0)
    if promoted:
        extras.append(f"{promoted} promoted item(s) skipped")
    extras.append("published_at decoded from post IDs, checked against each card's "
                  "relative time label")
    parts = head + notes + walk.notes() + extras + [walk.summary()]
    status.state = state
    status.reason = lp._redact(f"{REASON_PREFIX}: " + "; ".join(parts), cfg)
    status.finished_at = now_iso()
    return CollectResult(posts=posts, status=[status])


__all__ = ["DEFAULT_QUERY_ID", "DEFAULT_USER_AGENT", "PROVIDERS", "build_headers",
           "build_url", "check_response", "collect_voyager", "extract_page", "jsessionid",
           "provider", "query_params", "result_refs"]
