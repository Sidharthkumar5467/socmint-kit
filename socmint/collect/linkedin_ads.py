"""LinkedIn Ad Library collector (public, no login).

Python port of the owner's TypeScript module (`linkedin_ads/lib/scraper.ts`, itself a
port of the n8n "CTI Engineering_17_Linkedin-Ads" workflow). Porting instead of
shelling out to Node keeps setup to one `pip install -e .` (docs/DECISIONS.md).

Flow (mirrors the owner's crawl, plus a detail-page step for run dates):
  1. GET /ad-library/search?keyword=<topic>&<date params>
  2. extract cards per ad (anchored on each ad's own detail link, NOT zipped arrays)
  3. regex the pagination token; GET /ad-library/searchPaginationFragment?... until no
     (new) token, MAX_PAGES, the post limit or the deadline; 1.5 s between pages
  4. GET /ad-library/detail/<id> for up to `linkedin_detail_limit` ads to read the
     run dates ("Ran from <date> to <date>"), advertiser id, impressions, regions

Evidence rules encoded here:
  - published_at is the ad's run START date shown by LinkedIn, only when the detail page
    shows it. It is date-only, stored as 00:00:00Z and annotated as such. Without a
    detail page, published_at stays None and timestamp_verified=False.
  - LinkedIn filters by when ads ran, not when they were published, and only in whole
    days relative to "today", so every status carries date_filtered=False.
  - The LINKEDIN_COOKIE is only attached when set: the owner verified that an invalid
    li_at makes LinkedIn challenge or 429 an otherwise public request.

All selectors and page texts below were checked against live pages saved in
tests/fixtures/linkedin_ads/live_*.html (fetched 2026-11-18 without any cookie).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
from urllib.parse import urlencode, urlparse

import httpx

from socmint.config import redact
from socmint.interfaces import Capabilities, CollectContext, CollectResult, Query
from socmint.schema import AdMeta, Media, Post, Relation, SourceState, SourceStatus
from socmint.util.http import HttpFailure, make_client, request
from socmint.util.log import get_logger
from socmint.util.timeutil import now_iso, now_utc, parse_ts

log = get_logger(__name__)

SOURCE = "linkedin_ads"
BASE = "https://www.linkedin.com"
SEARCH_URL = f"{BASE}/ad-library/search"
PAGINATION_URL = f"{BASE}/ad-library/searchPaginationFragment"
DETAIL_URL = f"{BASE}/ad-library/detail/"

# Owner's runaway guard and politeness delay (scraper.ts MAX_PAGES / PAGE_DELAY_MS).
MAX_PAGES = 100
PAGE_DELAY_S = 1.5
# Owner: LinkedIn stalls throttled connections instead of erroring -> hard timeout.
REQUEST_TIMEOUT_S = 12.0
# Ads whose detail page is fetched (run dates, regions, landing URL); the rest keep no
# run dates. Raised from 40 when runs lost their time limit (owner decision 2026-11-20):
# one paced request each, so 200 details add about 5 minutes.
DEFAULT_DETAIL_LIMIT = 200
# Stop starting new requests when less than this much of the budget is left.
DEADLINE_MARGIN_S = 5.0

# Owner's headers (session.ts DEFAULT_UA + accept headers). English is requested so
# the "Ran from Nov 16, 2026" date text is in a format we can parse.
HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
    ),
    "accept-language": "en-US,en;q=0.9",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Owner's challenge detection (session.ts BLOCKED): a challenge is served either as a
# redirect to one of these paths or as a 200 page that mentions them and has no
# ad-library markup.
BLOCKED_RE = re.compile(r"authwall|checkpoint/challenge|uas/login|/login\b", re.I)
AUTHWALL_REASON = "LinkedIn returned an authwall/challenge"

# Markup that only real Ad Library pages carry (all seen in the live fixtures): the
# page-instance URN of the search/detail page, the per-ad detail link of a listing or
# pagination fragment, and the empty-results heading. The bare substring "ad-library"
# is NOT a marker: authwall pages embed the requested URL (sessionRedirect=...).
LIBRARY_MARKER_RE = re.compile(
    r"urn:li:page:d_ad-library-(?:search|detail)"
    r"|data-tracking-control-name=\"ad_library_view_ad_detail\""
    r"|>\s*No results found\s*<", re.I)
# Live (live_search_bank_noresults.html): <h1 ...>No results found</h1>.
EMPTY_RESULTS_RE = re.compile(r">\s*No results found\s*<", re.I)

DATE_ONLY_NOTE = "date-only precision (ad run start)"

# --------------------------------------------------------------------------- mini DOM
# bs4 is declared in requirements.txt but this module only needs a handful of scoped
# lookups, so a tiny stdlib tree keeps it dependency-free.

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
         "source", "track", "wbr"}


@dataclass(eq=False)
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[Node | str] = field(default_factory=list)
    parent: Node | None = None

    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    def has_classes(self, *names: str) -> bool:
        return set(names) <= self.classes()

    def iter(self) -> Iterator[Node]:
        """Descendants in document order (excluding self)."""
        for c in self.children:
            if isinstance(c, Node):
                yield c
                yield from c.iter()

    def find_all(self, pred: Callable[[Node], bool]) -> list[Node]:
        return [n for n in self.iter() if pred(n)]

    def find(self, pred: Callable[[Node], bool]) -> Node | None:
        return next((n for n in self.iter() if pred(n)), None)

    def text(self) -> str:
        parts: list[str] = []
        for c in self.children:
            parts.append(c if isinstance(c, str) else c.text())
        return "".join(parts)

    def closest(self, pred: Callable[[Node], bool]) -> Node | None:
        n: Node | None = self
        while n is not None:
            if pred(n):
                return n
            n = n.parent
        return None


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("#root")
        self.stack: list[Node] = [self.root]
        self._skip = 0  # inside <script>/<style>: text is not page content

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag, {k: (v or "") for k, v in attrs}, parent=self.stack[-1])
        self.stack[-1].children.append(node)
        if tag in ("script", "style"):
            self._skip += 1
        if tag not in _VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag, {k: (v or "") for k, v in attrs}, parent=self.stack[-1])
        self.stack[-1].children.append(node)

    def handle_endtag(self, tag: str) -> None:
        # Pop to the matching open tag; ignore stray end tags.
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                if tag in ("script", "style"):
                    self._skip = max(0, self._skip - 1)
                del self.stack[i:]
                return

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.stack[-1].children.append(data)


def parse_html(html: str) -> Node:
    b = _TreeBuilder()
    b.feed(html)
    b.close()
    return b.root


def _clean(s: str | None) -> str | None:
    """Trim; collapse runs of spaces but keep line breaks (commentary is pre-wrap)."""
    if s is None:
        return None
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in s.strip().splitlines()]
    out = "\n".join(ln for ln in lines).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out or None


# ------------------------------------------------------------------ listing (search)

DETAIL_CTRL = "ad_library_view_ad_detail"
# Owner: ADVERTISER_SEL = ".block.text-md.text-color-text.font-bold"
ADVERTISER_CLASSES = ("block", "text-md", "text-color-text", "font-bold")


def _is_detail_link(n: Node) -> bool:
    return n.tag == "a" and n.attrs.get("data-tracking-control-name") == DETAIL_CTRL


def _is_advertiser(n: Node) -> bool:
    return n.has_classes(*ADVERTISER_CLASSES)


def _is_commentary(n: Node) -> bool:
    return "commentary__content" in n.classes()


def _headline(card: Node) -> str | None:
    # Owner: ".sponsored-content-headline h2"
    box = card.find(lambda n: "sponsored-content-headline" in n.classes())
    h2 = box.find(lambda n: n.tag == "h2") if box else None
    return _clean(h2.text()) if h2 else None


def _images(card: Node) -> list[str]:
    # Owner reads the first ".ad-preview__dynamic-dimensions-image"[data-delayed-url];
    # carousels hold several, so all are kept (first one == owner's value).
    out: list[str] = []
    for n in card.find_all(lambda n: "ad-preview__dynamic-dimensions-image" in n.classes()):
        u = (n.attrs.get("data-delayed-url") or "").strip()
        if u and u not in out:
            out.append(u)
    return out


def _logo(card: Node) -> str | None:
    # Owner: img[alt="advertiser logo"]. Thought-leader ads (a member's post promoted by
    # a company, seen live) show the member's photo with alt="member logo" instead.
    for alt in ("advertiser logo", "member logo"):
        n = card.find(lambda n, a=alt: n.tag == "img" and n.attrs.get("alt") == a)
        if n is not None:
            return (n.attrs.get("data-delayed-url") or "").strip() or None
    return None


def _fallback_advertiser(card: Node) -> str | None:
    """Advertiser when the owner's selector finds nothing (thought-leader ads).

    Live markup: '<p>Promoted by <strong>Guru</strong></p>' and the card's
    aria-label 'Guru, Single Image Ad, View details'. The paying company is the
    advertiser; the member whose post is promoted is not.
    """
    for p in card.find_all(lambda n: n.tag == "p"):
        if (_clean(p.text()) or "").startswith("Promoted by"):
            strong = p.find(lambda n: n.tag == "strong")
            name = _clean(strong.text()) if strong else None
            if name:
                return name
    labelled = card if "aria-label" in card.attrs else card.find(
        lambda n: "base-ad-preview-card" in n.classes())
    label = (labelled.attrs.get("aria-label") if labelled else "") or ""
    if label.endswith("View details") and label.count(",") >= 2:
        # rsplit keeps advertiser names that contain commas ("Lendgo, Inc.").
        return label.rsplit(",", 2)[0].strip() or None
    return None


def _creative_type(card: Node) -> str | None:
    ct = card.attrs.get("data-creative-type")
    if not ct:
        inner = card.find(lambda n: "data-creative-type" in n.attrs)
        ct = inner.attrs.get("data-creative-type") if inner else None
    if not ct:
        outer = card.closest(lambda n: "ad-preview" in n.classes())
        ct = outer.attrs.get("data-creative-type") if outer else None
    return (ct or "").strip() or None


_DETAIL_ID_RE = re.compile(r"/ad-library/detail/(\d+)")


@dataclass
class ListingAd:
    detail_id: str
    detail_url: str
    advertiser: str | None
    creative_type: str | None
    commentary: str | None
    headline: str | None
    images: list[str]
    advertiser_logo: str | None


def extract_listing(html: str) -> list[ListingAd]:
    """Owner's node 05, per-card.

    The original n8n flow collected each field into its own array and zipped them by
    index; a missing headline (12 of 24 cards in live_search_bank.html) then shifted
    every later row onto another advertiser's copy. Instead we anchor on each ad's
    detail link, climb to that ad's own card (never past a container holding a second
    ad) and read every field inside that card only.
    """
    root = parse_html(html)
    rows: list[ListingAd] = []
    seen_cards: set[int] = set()
    seen_ids: set[str] = set()
    for link in root.find_all(_is_detail_link):
        card = link
        for _ in range(10):
            parent = card.parent
            if parent is None or parent.tag == "#root":
                break
            if len(parent.find_all(_is_detail_link)) > 1:
                break  # parent would include another ad
            card = parent
            if card.find(_is_advertiser) and card.find(_is_commentary):
                break
        if id(card) in seen_cards:
            continue  # several links can share one card
        seen_cards.add(id(card))

        path = (link.attrs.get("href") or "").strip()
        m = _DETAIL_ID_RE.search(path)
        if not m:
            continue  # owner's node 06 skips rows without a detail path/id
        did = m.group(1)
        if did in seen_ids:
            continue
        seen_ids.add(did)
        adv = card.find(_is_advertiser)
        com = card.find(_is_commentary)
        rows.append(ListingAd(
            detail_id=did,
            detail_url=f"{DETAIL_URL}{did}",
            advertiser=(_clean(adv.text()) if adv else None) or _fallback_advertiser(card),
            creative_type=_creative_type(card),
            commentary=_clean(com.text()) if com else None,
            headline=_headline(card),
            images=_images(card),
            advertiser_logo=_logo(card),
        ))
    return rows


def extract_next_token(html: str) -> str | None:
    """Owner's node 12: pagination token, multi-pattern regex (verbatim)."""
    m = (re.search(r'"paginationToken"\s*:\s*"([0-9-]+)"', html, re.I)
         or re.search(r'data-pagination-token="([0-9-]+)"', html, re.I)
         or re.search(r"[?&]paginationToken=([0-9-]+)", html))
    return m.group(1) if m else None


def has_library_markup(html: str) -> bool:
    """True if the page carries markup only real Ad Library pages have."""
    return bool(LIBRARY_MARKER_RE.search(html))


def shows_no_results(html: str) -> bool:
    """True if the page shows LinkedIn's own empty-results heading."""
    return bool(EMPTY_RESULTS_RE.search(html))


def looks_blocked(html: str, final_url: str = "") -> bool:
    """Owner's challenge test: redirected to a login/authwall, or a page that mentions
    one and carries no real ad-library markup (see LIBRARY_MARKER_RE).

    UNVERIFIED: no live challenge page has been observed; the inline-authwall case is
    covered by a synthetic test only.
    """
    if final_url and BLOCKED_RE.search(urlparse(final_url).path or ""):
        return True
    return bool(BLOCKED_RE.search(html)) and not has_library_markup(html)


# ----------------------------------------------------------------------- detail page


@dataclass
class AdDetail:
    run_start: date | None = None
    run_end: date | None = None
    run_text: str | None = None
    advertiser_id: str | None = None
    advertiser_url: str | None = None
    advertiser: str | None = None
    paid_for_by: str | None = None
    commentary: str | None = None
    headline: str | None = None
    impressions_range: str | None = None
    impression_countries: list[str] = field(default_factory=list)
    targeted_locations: list[str] = field(default_factory=list)
    landing_url: str | None = None
    commentary_links: list[str] = field(default_factory=list)


_MONTH_FORMATS = ("%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%d %b %Y")


def parse_ad_date(s: str) -> date | None:
    s = re.sub(r"\s+", " ", s.strip().rstrip("."))
    for fmt in _MONTH_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


# Live: "Ran from Nov 16, 2026 to Nov 18, 2026".
_RAN_FROM_TO = re.compile(r"Ran from (.+?) to (.+)$", re.I)
# UNVERIFIED: single-day / open-ended variants ("Ran on <d>", "Ran from <d>") have not
# been seen live; handled defensively so they still yield a run start.
_RAN_ON = re.compile(r"Ran (?:on|from) (.+)$", re.I)
# Live: companies link to /company/<numeric id>, member advertisers to /in/<slug>.
# The id keeps the path kind ("company/123", "in/slug") so namespaces never collide.
_ADVERTISER_RE = re.compile(r"linkedin\.com/(company|in|showcase|school)/([^/?#]+)")


def parse_run_dates(text: str) -> tuple[date | None, date | None]:
    t = re.sub(r"\s+", " ", text).strip()
    m = _RAN_FROM_TO.search(t)
    if m:
        return parse_ad_date(m.group(1)), parse_ad_date(m.group(2))
    m = _RAN_ON.search(t)
    if m:
        d = parse_ad_date(m.group(1))
        return d, (d if t.lower().startswith("ran on") else None)
    return None, None


def _external_href(n: Node) -> str | None:
    href = (n.attrs.get("href") or "").strip()
    if not href.startswith("http"):
        return None
    host = (urlparse(href).hostname or "").lower()
    if host.endswith("linkedin.com") or host.endswith("licdn.com"):
        return None
    return href


def extract_detail(html: str) -> AdDetail:
    root = parse_html(html)
    d = AdDetail()

    dur = root.find(lambda n: "about-ad__availability-duration" in n.classes())
    if dur is not None:
        d.run_text = _clean(dur.text())
        if d.run_text:
            d.run_start, d.run_end = parse_run_dates(d.run_text)

    # Advertiser link: "About the ad" rail first, then the preview header.
    for ctrl in ("ad_library_about_ad_advertiser", "ad_library_ad_preview_advertiser"):
        a = root.find(lambda n, c=ctrl: n.tag == "a"
                      and n.attrs.get("data-tracking-control-name") == c)
        if a is not None:
            href = a.attrs.get("href", "")
            m = _ADVERTISER_RE.search(href)
            if m:
                d.advertiser_id = f"{m.group(1)}/{m.group(2)}"
                d.advertiser_url = f"{BASE}/{d.advertiser_id}"
            d.advertiser = d.advertiser or _clean(a.text())
            if d.advertiser_id:
                break

    paid = root.find(lambda n: "about-ad__paying-entity" in n.classes())
    if paid is not None:
        d.paid_for_by = _clean(re.sub(r"^\s*Paid for by\s*", "", paid.text()))

    # Ad copy: first variant only (detail pages may hold several versions).
    variant = root.find(lambda n: "ad-preview" in n.classes()) or root
    com = variant.find(_is_commentary)
    d.commentary = _clean(com.text()) if com else None
    d.headline = _headline(variant)

    # "Total Impressions" -> sibling <p> holds the range, e.g. "1k-5k".
    label = root.find(lambda n: n.tag == "p" and _clean(n.text()) == "Total Impressions")
    if label is not None and label.parent is not None:
        ps = [c for c in label.parent.children if isinstance(c, Node) and c.tag == "p"]
        vals = [_clean(p.text()) for p in ps if p is not label]
        d.impressions_range = next((v for v in vals if v), None)

    # Countries with impressions: aria-label "Brazil, impressions 100%".
    for n in root.find_all(lambda n: "ad-analytics__country-impressions" in n.classes()):
        name = n.find(lambda x: x.tag == "p")
        country = _clean(name.text()) if name else None
        if not country:
            country = (n.attrs.get("aria-label") or "").split(",")[0].strip() or None
        if country and country not in d.impression_countries:
            d.impression_countries.append(country)

    # Targeting "Location" block: "Targeting includes A, B and <button>1 others</button>
    # <span class=ad-targeting__other-segments>C</span>".
    loc_h3 = root.find(lambda n: n.tag == "h3" and _clean(n.text()) == "Location")
    if loc_h3 is not None and loc_h3.parent is not None:
        seg = loc_h3.parent.find(lambda n: "ad-targeting__segments" in n.classes())
        if seg is not None:
            head = "".join(c for c in seg.children if isinstance(c, str))
            head = re.sub(r"^\s*Targeting (?:includes|excludes)\s*", "", head.strip())
            names = re.split(r",\s*|\s+and\s*$|\s+and\s+", head)
            other = seg.find(lambda n: "ad-targeting__other-segments" in n.classes())
            if other is not None:
                names += re.split(r",\s*|\s+and\s+", other.text())
            for nm in names:
                nm = (_clean(nm) or "").strip()
                if nm and nm not in d.targeted_locations:
                    d.targeted_locations.append(nm)

    # Landing URL: live detail pages render the CTA as a <button> without href, so this
    # is usually None. UNVERIFIED: an <a> CTA (same tracking name) carrying an external
    # href has not been seen live; it is read defensively if LinkedIn ever renders one.
    cta = variant.find(lambda n: n.tag == "a"
                       and n.attrs.get("data-tracking-control-name") == "ad_library_ad_detail_cta")
    if cta is not None:
        d.landing_url = _external_href(cta)
    # External links written in the ad copy are kept as plain link relations, never
    # presented as the landing URL.
    if com is not None:
        for n in com.find_all(lambda n: n.tag == "a"):
            href = _external_href(n)
            if href and href != d.landing_url and href not in d.commentary_links:
                d.commentary_links.append(href)
    return d


# ---------------------------------------------------------------------- date window


@dataclass
class DateParams:
    params: dict[str, str]
    note: str
    out_of_coverage: bool = False
    # The window includes today, which a custom range cannot (it must end yesterday): the
    # collector also reads the last-30-days listing and keeps only extra ads whose detail
    # page shows they ran inside the window (ads that started today).
    today_supplement: bool = False


def choose_date_params(start: date, end: date, today: date) -> DateParams:
    """Narrowest LinkedIn date filter covering [start, end].

    Live form (2026-11-18) offers dateOption = last-30-days | current-month |
    current-year | last-year | custom-date-range (startdate/enddate, YYYY-MM-DD, min =
    today minus one year, max = yesterday). The owner's "last-24-hours"/"last-7-days"
    are not in the live form, and "last-24-hours" returned exactly the last-30-days
    results, so they are not used. A custom range whose enddate is today returned
    "No results found" live (again 2026-11-20), so enddate is capped at yesterday.

    Owner request 2026-11-20: always filter with the window's own dates ("Select date
    range") rather than last-30-days (live, "sahibinden": 17-19 Nov -> 2 ads, last 30 days
    -> 12). When the window includes today, today_supplement asks the collector to add
    ads that started today from the last-30-days listing (kept only if their run dates
    fall in the window). A window that is only today uses last-30-days + that filter.
    """
    yesterday = today - timedelta(days=1)
    try:
        earliest = today.replace(year=today.year - 1)
    except ValueError:  # Feb 29
        earliest = today.replace(year=today.year - 1, day=28)
    if end < earliest:
        return DateParams({}, "LinkedIn Ad Library only covers about the last 12 months",
                          out_of_coverage=True)
    s = max(start, earliest)
    e = min(end, yesterday)
    reaches_today = end >= today
    if s > e:
        # The whole window is today: no custom range is possible.
        return DateParams({"dateOption": "last-30-days"},
                          "dateOption=last-30-days (the window is only today; ads kept only "
                          "if their run dates fall in the window)", today_supplement=True)
    note = f"dateOption=custom-date-range {s.isoformat()}..{e.isoformat()}"
    if e < end:
        note += f" (end capped at {e.isoformat()}: LinkedIn allows ranges up to yesterday)"
    if s > start:
        note += f" (start raised to {s.isoformat()}: library depth)"
    return DateParams({"dateOption": "custom-date-range", "startdate": s.isoformat(),
                       "enddate": e.isoformat()}, note, today_supplement=reaches_today)


# ----------------------------------------------------------------------- post mapping


def _day_start_iso(d: date) -> str:
    return f"{d.isoformat()}T00:00:00Z"


def _day_end_iso(d: date) -> str:
    return f"{d.isoformat()}T23:59:59Z"


def to_post(ad: ListingAd, detail: AdDetail | None, *, topic: str, collected_at: str,
            raw_ref: str | None, detail_note: str | None = None,
            window_start: date | None = None) -> Post:
    """Map one ad into a Post. `detail_note` explains a missing/failed detail fetch.

    `window_start` is the query's start date. An ad whose run START is before it is
    kept (it ran during the window) but marked timestamp_verified=False, because its
    published_at lies outside the window: verify.gate_timestamps marks ads in_window on
    run overlap, so a verified pre-window start would enter analysable() and could be
    reported as "first observed in collected data" months before the investigation.
    """
    commentary = ad.commentary
    headline = ad.headline
    advertiser = ad.advertiser
    if detail is not None:
        # Listing copy is truncated with "…"; the detail page carries the full text.
        if detail.commentary and len(detail.commentary) >= len(commentary or ""):
            commentary = detail.commentary
        headline = headline or detail.headline
        advertiser = advertiser or detail.advertiser
    text = "\n\n".join(t for t in (commentary, headline) if t)

    media: list[Media] = []
    for i, url in enumerate(ad.images):
        if ad.creative_type == "SPONSORED_VIDEO" and i == 0:
            # Live: video cards only expose the video cover image, not the video URL.
            media.append(Media(type="video", url=None, thumbnail=url))
        else:
            media.append(Media(type="image", url=url))

    published_at: str | None = None
    verified = False
    note: str
    ad_meta = AdMeta()
    relations: list[Relation] = []
    if detail is not None:
        regions = detail.impression_countries or detail.targeted_locations
        ad_meta = AdMeta(
            # Full UTC timestamps (day start/end) so verify's overlap test can parse them.
            run_start=_day_start_iso(detail.run_start) if detail.run_start else None,
            run_end=_day_end_iso(detail.run_end) if detail.run_end else None,
            advertiser_id=detail.advertiser_id,
            landing_url=detail.landing_url,
            regions=list(regions),
            impressions_range=detail.impressions_range,
        )
        for link in [detail.landing_url, *detail.commentary_links]:
            if link:
                relations.append(Relation(type="link", target_url=link))
        if detail.run_start and window_start and detail.run_start < window_start:
            # Real platform date, but before the window. The date stays verified;
            # verify.gate_timestamps sets in_window=False (published_at = run start is
            # outside the window) and active_in_window=True if the run overlaps, so the
            # ad is analysed as a narrative member but kept off timelines, counts and
            # first-observed analysis.
            published_at = _day_start_iso(detail.run_start)
            verified = True
            note = (f"{DATE_ONLY_NOTE}; ad run started {detail.run_start.isoformat()}, "
                    f"before the window start {window_start.isoformat()}, so it is excluded "
                    "from timelines, counts and first-observed analysis")
        elif detail.run_start:
            # Platform-provided date without time: verified, with stated precision.
            published_at = _day_start_iso(detail.run_start)
            verified = True
            note = DATE_ONLY_NOTE
        else:
            note = "ad detail page shows no run dates"
    else:
        note = detail_note or "ad detail page not fetched; run dates unknown"

    return Post(
        post_id=f"{SOURCE}:{ad.detail_id}",
        source=SOURCE,
        kind="ad",
        author_id=detail.advertiser_id if detail else None,
        author_handle=None,
        author_name=advertiser,
        author_avatar_url=ad.advertiser_logo,
        text=text,
        published_at=published_at,
        collected_at=collected_at,
        timestamp_verified=verified,
        timestamp_note=note,
        media=media,
        url=ad.detail_url,
        relations=relations,
        raw_ref=raw_ref,
        query=topic,
        ad_meta=ad_meta,
    )


# -------------------------------------------------------------------------- collector


class _Blocked(Exception):
    pass


def _http_kind(e: HttpFailure) -> str:
    """Status label: only a real HTTP 429 is called a 429. util.http also raises
    rate_limited=True when the deadline cuts a 5xx/network backoff short."""
    if e.status == 429:
        return "rate limited (429)"
    return redact(str(e))


class LinkedInAdsCollector:
    name = SOURCE

    def __init__(self, sleep: Callable[[float], None] = time.sleep,
                 client_factory: Callable[[], httpx.Client] | None = None,
                 today: Callable[[], date] | None = None) -> None:
        self._sleep = sleep
        self._client_factory = client_factory
        self._today = today or (lambda: now_utc().date())

    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_date_filter=False,  # filters by run period, whole days only
            needs_credentials=False,
            kind="ads",
            credential_keys=(),
            description="LinkedIn Ad Library (public, no login); run dates from detail pages",
        )

    # ------------------------------------------------------------------ http helpers
    def _client(self, ctx: CollectContext, *, send_cookie: bool = True) -> httpx.Client:
        if self._client_factory is not None:
            return self._client_factory()
        headers = dict(HEADERS)
        cookie = str(ctx.config.get("LINKEDIN_COOKIE") or "").strip()
        if cookie and send_cookie:  # only when supplied: a stale li_at triggers challenges/429
            headers["cookie"] = cookie
        return make_client(timeout=REQUEST_TIMEOUT_S, headers=headers)

    def _get(self, client: httpx.Client, url: str, ctx: CollectContext) -> str:
        """GET a library page. Raises _Blocked on authwall/challenge, HttpFailure else."""
        try:
            resp = request(client, "GET", url, max_retries=2, deadline=ctx.deadline, stop=ctx.stop)
        except HttpFailure as e:
            if e.status in (403, 999):  # owner: 403/999 == challenged
                raise _Blocked(f"HTTP {e.status}") from e
            raise
        html = resp.text
        if looks_blocked(html, str(resp.url)):
            raise _Blocked(f"redirected/served {urlparse(str(resp.url)).path}")
        return html

    def _list_all(self, client: httpx.Client, ctx: CollectContext, topic: str,
                  params: dict[str, str], st: SourceStatus, tag: str
                  ) -> tuple[list[tuple[ListingAd, str]], int, str | None]:
        """Every listing page for `params` (search page, then pagination tokens), politely
        paced. Returns ([(ad, raw ref)], pages, stop reason or None)."""
        out: list[tuple[ListingAd, str]] = []
        pages = 0
        token: str | None = None
        while pages < MAX_PAGES:
            if not self._wait(ctx):
                return out, pages, "time budget exhausted"
            url = (f"{SEARCH_URL}?{urlencode({'keyword': topic, **params})}" if token is None
                   else f"{PAGINATION_URL}?" + urlencode({"keyword": topic, "countries": "ALL",
                                                          **params, "paginationToken": token}))
            st.requests += 1
            try:
                html = self._get(client, url, ctx)
            except _Blocked as e:
                return out, pages, f"{AUTHWALL_REASON} ({e})"
            except HttpFailure as e:
                return out, pages, _http_kind(e)
            pages += 1
            ref = ctx.save_raw(SOURCE, f"{tag}_p{pages:03d}.html", html)
            rows = extract_listing(html)
            out.extend((r, ref) for r in rows)
            nxt = extract_next_token(html)
            if not rows or not nxt or nxt == token:
                return out, pages, None
            token = nxt
        return out, pages, f"{MAX_PAGES}-page safety cap"

    def _wait(self, ctx: CollectContext) -> bool:
        """Politeness delay; False if it would run past the deadline."""
        if ctx.time_left() < PAGE_DELAY_S + DEADLINE_MARGIN_S:
            return False
        self._sleep(PAGE_DELAY_S)
        return True

    # ---------------------------------------------------------------------- collect
    def collect(self, query: Query, ctx: CollectContext) -> CollectResult:
        started = now_iso()
        topic = query.topic
        st = SourceStatus(source=SOURCE, query=topic, state=SourceState.ok,
                          date_filtered=False, started_at=started)
        # The pipeline may widen the ads search beyond the post window (ads lookback,
        # pipeline.ads_window_start): options["ads_start_at"] is then the earlier start.
        ads_from = parse_ts(str((query.options or {}).get("ads_start_at") or "")) or query.start
        ads_from = min(ads_from, query.start)
        all_time = bool((query.options or {}).get("ads_all_time"))
        if all_time:
            # As the owner's LinkedIn Ads tool does: no dateOption, every ad the Ad Library
            # lists for the keyword (live 2026-11-20: 54 ads for "sahibinden" vs 13 with a
            # 1-30 Oct custom range). Run dates come from the detail pages.
            dp = DateParams({}, "no date filter (ads lookback: all time; every ad the Ad "
                                "Library lists for the keyword)")
        else:
            dp = choose_date_params(ads_from.date(), query.end.date(), self._today())
        if dp.out_of_coverage:
            st.state = SourceState.out_of_coverage
            st.reason = dp.note
            return CollectResult(posts=[], status=[st])

        reasons: list[str] = [dp.note]
        if ads_from < query.start and not all_time:
            reasons.append(f"ads lookback from {ads_from.date().isoformat()} (longer than the "
                           "post window; ads that ran only before the window are analysed but "
                           "kept off timelines, counts and first-observed analysis)")
        partial_reason: str | None = None
        # CLAUDE.md section 5: never use one account in parallel. The LinkedIn posts source
        # drives the same account in a browser during the same run, so the (public) Ad
        # Library is then fetched without the cookie.
        share_account = "linkedin" in ((query.options or {}).get("sources") or ())
        if share_account and str(ctx.config.get("LINKEDIN_COOKIE") or "").strip():
            reasons.append("fetched without LINKEDIN_COOKIE: the LinkedIn posts source "
                           "uses that account in this run")
        ads: list[ListingAd] = []
        ad_raw: dict[str, str] = {}
        seen: set[str] = set()
        token: str | None = None
        limit = max(1, int(query.limit or 500))
        # Counts ads push() drops because the post limit was reached, so a limit
        # stop is reported as partial even when no further page token exists.
        dropped = 0

        client = self._client(ctx, send_cookie=not share_account)
        try:
            # ------------------------------------------------------- listing pages
            first = f"{SEARCH_URL}?{urlencode({'keyword': topic, **dp.params})}"
            try:
                html = self._get(client, first, ctx)
            except _Blocked as e:
                st.state = SourceState.failed
                st.reason = f"{AUTHWALL_REASON} ({e})"
                st.requests = 1
                return CollectResult(posts=[], status=[st])
            except HttpFailure as e:
                st.state = SourceState.failed
                st.reason = redact(f"search page: {e}")
                st.requests = 1
                return CollectResult(posts=[], status=[st])
            st.requests = 1
            pages = 1
            raw = ctx.save_raw(SOURCE, "search_p001.html", html)

            def push(rows: list[ListingAd], raw_ref: str) -> None:
                nonlocal dropped
                for r in rows:
                    if r.detail_id in seen:
                        continue
                    if len(ads) >= limit:
                        dropped += 1  # retrieved but dropped: must show in status
                        continue
                    seen.add(r.detail_id)
                    ads.append(r)
                    ad_raw[r.detail_id] = raw_ref

            first_rows = extract_listing(html)
            if not first_rows and not shows_no_results(html):
                # Zero cards and no "No results found" heading: a markup change or an
                # interstitial, not an empty result. Never report it as completed_zero.
                st.state = SourceState.failed
                st.reason = ("search page parsed 0 ads but shows no empty-results marker; "
                             "page structure may have changed or a challenge was served")
                return CollectResult(posts=[], status=[st])
            push(first_rows, raw)
            ctx.progress(SOURCE, {"count": len(ads), "message": f"{SOURCE}: page 1"})
            token = extract_next_token(html)

            while token:
                if len(ads) >= limit:
                    partial_reason = f"stopped at post limit {limit}"
                    break
                if pages >= MAX_PAGES:
                    partial_reason = f"stopped at the {MAX_PAGES}-page safety cap"
                    break
                if not self._wait(ctx):
                    partial_reason = "time budget exhausted during pagination"
                    break
                url = f"{PAGINATION_URL}?" + urlencode(
                    {"keyword": topic, "countries": "ALL", **dp.params,
                     "paginationToken": token})
                st.requests += 1
                try:
                    html = self._get(client, url, ctx)
                except _Blocked as e:
                    partial_reason = f"{AUTHWALL_REASON} on page {pages + 1} ({e})"
                    break
                except HttpFailure as e:
                    partial_reason = (f"pagination stopped at page {pages + 1}: "
                                      f"{_http_kind(e)}")
                    break
                pages += 1
                raw = ctx.save_raw(SOURCE, f"search_p{pages:03d}.html", html)
                rows = extract_listing(html)
                if not rows:
                    if shows_no_results(html):
                        token = None  # LinkedIn says the results are exhausted
                    else:
                        # UNVERIFIED: the shape of a real exhausted pagination fragment
                        # has not been observed live; without the marker we cannot tell
                        # an end from a markup change, so stop as partial with cursor.
                        partial_reason = (f"page {pages} parsed 0 ads but shows no "
                                          "empty-results marker; page structure may "
                                          "have changed")
                    break
                push(rows, raw)
                ctx.progress(SOURCE, {"count": len(ads),
                                      "message": f"{SOURCE}: page {pages}"})
                nxt = extract_next_token(html)
                if not nxt or nxt == token:  # owner's node 13: no (new) token => stop
                    token = None
                    break
                token = nxt
            if dropped and partial_reason is None:
                # The last page held more ads than the limit allowed (no token needed).
                partial_reason = f"stopped at post limit {limit}"
            if dropped:
                reasons.append(f"{dropped} retrieved ad(s) dropped by the post limit")
            reasons.append(f"{pages} page(s)")

            # ---------------------------------------------- today's ads (supplement)
            extras: list[ListingAd] = []
            if dp.today_supplement and partial_reason is None:
                if dp.params.get("dateOption") == "last-30-days":
                    extras, ads = ads, []  # the main listing IS last-30-days: all need dates
                else:
                    rows, sup_pages, sup_stop = self._list_all(
                        client, ctx, topic, {"dateOption": "last-30-days"}, st, "today")
                    for r, ref in rows:
                        if r.detail_id not in seen and len(ads) + len(extras) < limit:
                            seen.add(r.detail_id)
                            extras.append(r)
                            ad_raw[r.detail_id] = ref
                    reasons.append(f"window includes today: last-30-days listing read "
                                   f"({sup_pages} page(s), {len(extras)} ad(s) not in the date "
                                   "range; kept only if their run dates fall in the window)"
                                   + (f"; supplement stopped: {sup_stop}" if sup_stop else ""))

            # --------------------------------------------------------- detail pages
            main_ads = ads
            ads = main_ads + extras
            detail_limit = int(query.options.get("linkedin_detail_limit",
                                                 DEFAULT_DETAIL_LIMIT))
            details: dict[str, AdDetail] = {}
            notes: dict[str, str] = {}
            detail_stop: str | None = None
            failed_details = 0
            for i, ad in enumerate(ads):
                if i >= detail_limit:
                    notes[ad.detail_id] = (f"ad detail page not fetched (linkedin_detail_limit"
                                           f"={detail_limit}); run dates unknown")
                    continue
                if detail_stop:
                    notes[ad.detail_id] = f"ad detail page not fetched ({detail_stop})"
                    continue
                if not self._wait(ctx):
                    detail_stop = "time budget exhausted"
                    notes[ad.detail_id] = f"ad detail page not fetched ({detail_stop})"
                    continue
                st.requests += 1
                try:
                    dhtml = self._get(client, ad.detail_url, ctx)
                except _Blocked as e:
                    detail_stop = f"{AUTHWALL_REASON} on detail pages ({e})"
                    notes[ad.detail_id] = f"ad detail page not fetched ({detail_stop})"
                    continue
                except HttpFailure as e:
                    if e.rate_limited:
                        # 429, or util.http giving up a backoff at the deadline.
                        detail_stop = f"{_http_kind(e)} on detail pages"
                        notes[ad.detail_id] = f"ad detail page not fetched ({detail_stop})"
                    else:
                        failed_details += 1
                        notes[ad.detail_id] = redact(f"ad detail page fetch failed: {e}")
                    continue
                ad_raw[ad.detail_id] = ctx.save_raw(SOURCE, f"detail_{ad.detail_id}.html",
                                                    dhtml)
                details[ad.detail_id] = extract_detail(dhtml)

            fetched = len(details)
            dated = sum(1 for d in details.values() if d.run_start)
            reasons.append(f"detail pages fetched for {fetched}/{len(ads)} ads, "
                           f"{dated} with run dates")
            if failed_details:
                reasons.append(f"{failed_details} detail page(s) failed")
            if detail_stop:
                if partial_reason:
                    reasons.append(f"detail fetch stopped: {detail_stop}")
                else:
                    partial_reason = f"detail fetch stopped: {detail_stop}"
        finally:
            if self._client_factory is None:
                client.close()

        if extras:
            # Keep an extra (last-30-days) ad only when its detail page dates its run inside
            # the window; without run dates it cannot be placed and is not kept.
            w0, w1 = query.start.date(), query.end.date()
            kept_extra, no_dates, outside = [], 0, 0
            for ad in extras:
                d = details.get(ad.detail_id)
                if d is None or d.run_start is None:
                    no_dates += 1
                    continue
                if d.run_start <= w1 and (d.run_end or w1) >= w0:
                    kept_extra.append(ad)
                else:
                    outside += 1
            ads = main_ads + kept_extra
            reasons.append(f"{len(kept_extra)} ad(s) added from last-30-days (ran in the window)"
                           + (f", {outside} ran outside it" if outside else "")
                           + (f", {no_dates} without run dates not kept" if no_dates else ""))

        collected_at = now_iso()
        posts = [to_post(ad, details.get(ad.detail_id), topic=topic,
                         collected_at=collected_at, raw_ref=ad_raw.get(ad.detail_id),
                         detail_note=notes.get(ad.detail_id),
                         window_start=query.start.date())
                 for ad in ads]

        st.count = len(posts)
        if partial_reason:
            st.state = SourceState.partial
            st.cursor = token
            reasons.insert(0, partial_reason)
        elif not posts:
            st.state = SourceState.completed_zero
            reasons.insert(0, "LinkedIn Ad Library returned no ads for this keyword "
                              "and date filter")
        st.reason = "; ".join(reasons)
        st.finished_at = now_iso()
        return CollectResult(posts=posts, status=[st])


__all__ = [
    "LinkedInAdsCollector", "extract_listing", "extract_next_token", "extract_detail",
    "choose_date_params", "parse_run_dates", "looks_blocked", "has_library_markup",
    "shows_no_results", "to_post",
]
