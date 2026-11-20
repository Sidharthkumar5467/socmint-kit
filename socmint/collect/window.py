"""Shared collection helpers: spread a post cap across the date window, and phrase matching.

Why (seen live 2026-11-19, docs/DECISIONS.md): search APIs return newest first, so a
300-post cap on a 7-day window filled from the last 6-13 hours and the older days were
never reached. Collectors whose search takes a date range run one search per time slice
and share the cap across slices; collectors that cannot (Telegram global search) balance
client-side and report the days they reached.

Phrase matching (also seen live): `(bank of america)` matches all three words anywhere,
so ~37% of X posts did not contain the phrase. By default a plain multi-word topic is sent
as an exact phrase.

Platform date bounds per slice: X web search must use since_time:/until_time: (Unix
seconds). Seen live 2026-11-19: x.com's since:/until: dates do not follow UTC days
(until:2026-11-15 returned posts up to 18:21Z on 11-15).

Contract used by every collector:
  slices = time_slices(query.start, query.end, max_slices)   # newest first
  share = SliceShare(query.limit, len(slices))
  for i, (s, e) in enumerate(slices):
      cap = share.allowance(i)            # posts this slice may add (remaining / slices left)
      ... collect up to `cap` posts published in [s, e] ...
      share.record(i, got)                # actual posts kept for this slice
  share.summary(slices)                   # "per-day: 2026-11-19 43, 2026-11-18 43, ..."
  search_topic(query)                     # topic string to send to a platform search
  topic_matches(query, text)              # client-side phrase check (phrase mode only)
  mentions_topic(query, text)             # stricter: the topic itself occurs in the text
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta

from socmint.interfaces import Query

MATCH_PHRASE = "phrase"
MATCH_WORDS = "words"
MATCH_MODES = (MATCH_PHRASE, MATCH_WORDS)
DEFAULT_MATCH = MATCH_PHRASE

# Default upper bound on slices; a 30-day window becomes 10 slices of 3 days.
DEFAULT_MAX_SLICES = 10

# A topic containing any of these is treated as a hand-written search expression and sent
# unchanged (quotes, boolean operators, exclusions, platform operators like from:/lang:).
_OPERATOR_RE = re.compile(
    r'["“”()]|\bOR\b|\bAND\b|\bNOT\b|(^|\s)-\S|\|\S*|\b[a-z_]+:\S', re.IGNORECASE)


def match_mode(query: Query) -> str:
    """The run's topic-match mode ('phrase' or 'words'); unknown values fall back to phrase."""
    m = str((query.options or {}).get("match") or DEFAULT_MATCH).strip().lower()
    return m if m in MATCH_MODES else DEFAULT_MATCH


def is_search_expression(topic: str) -> bool:
    """True if the analyst wrote search syntax themselves (then it is never rewritten)."""
    return bool(_OPERATOR_RE.search(topic))


def search_topic(query: Query) -> str:
    """Topic text to send to a platform search.

    phrase mode + a plain multi-word topic -> "the phrase" in double quotes.
    Single words, hashtags, @handles and hand-written expressions are sent unchanged.
    """
    t = " ".join(query.topic.split())
    if match_mode(query) == MATCH_PHRASE and " " in t and not is_search_expression(t):
        return f'"{t}"'
    return t


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"[^\w#@]+", " ", s)  # punctuation/whitespace runs -> one space
    return f" {s.strip()} "


def topic_matches(query: Query, text: str | None) -> bool:
    """Client-side check for platforms with no phrase search (Telegram).

    Always True in words mode, for single-word topics, and for hand-written expressions
    (those are the platform's job). In phrase mode: the phrase occurs in the text with
    case, punctuation and spacing differences ignored ("Bank-of-America" matches).
    """
    t = " ".join(query.topic.split())
    if match_mode(query) != MATCH_PHRASE or " " not in t or is_search_expression(t):
        return True
    return _norm(t) in _norm(text or "")


def _fold(s: str) -> str:
    """Case-, accent- and punctuation-insensitive form, padded with spaces: "SAHİBİNDEN'den"
    -> " sahibinden den ". Dotted/dotless capitals and accents are folded (NFKD, combining
    marks dropped), so Turkish, French etc. capitalisation does not hide a match."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    s = re.sub(r"[^\w]+", " ", s)
    return f" {s.strip()} "


def mentions_topic(query: Query, text: str | None) -> bool:
    """True if `text` itself mentions the topic. Used to filter platforms whose search also
    matches on fields we cannot see (YouTube: captions, related terms).

    phrase mode: the topic's words in order, each word starting at a word boundary (a
    suffix is allowed, so agglutinative forms like "sahibinden'den" or "sahibinden.com"
    match). words mode: every word occurs (same rule). Hand-written expressions are left
    to the platform (always True)."""
    t = " ".join(query.topic.split())
    if not t or is_search_expression(t):
        return True
    hay = _fold(text or "")
    words = _fold(t).split()
    if not words:
        return True
    if match_mode(query) == MATCH_PHRASE:
        return (" " + " ".join(words)) in hay
    return all((" " + w) in hay for w in words)


def time_slices(start: datetime, end: datetime,
                max_slices: int = DEFAULT_MAX_SLICES) -> list[tuple[datetime, datetime]]:
    """Split [start, end] (inclusive, UTC) into whole-day slices, newest first.

    One slice per UTC day when the window has <= max_slices days; otherwise days are grouped
    into max_slices slices of (nearly) equal whole-day length, longer ones at the newest end.
    Each slice is (00:00:00 of its first day, 23:59:59 of its last day), clipped to the
    window. A window shorter than a day is a single slice.
    """
    if end < start:
        return []
    max_slices = max(1, int(max_slices))
    first, last = start.date(), end.date()
    n_days = (last - first).days + 1
    n = min(n_days, max_slices)
    base, extra = divmod(n_days, n)
    out: list[tuple[datetime, datetime]] = []
    day_hi = last
    for i in range(n):
        length = base + (1 if i < extra else 0)
        day_lo = day_hi - timedelta(days=length - 1)
        s = max(start, datetime.combine(day_lo, datetime.min.time(), tzinfo=start.tzinfo))
        e = min(end, datetime.combine(day_hi, datetime.max.time().replace(microsecond=0),
                                      tzinfo=end.tzinfo))
        out.append((s, e))
        day_hi = day_lo - timedelta(days=1)
    return out


def slice_query(query: Query, s: datetime, e: datetime, limit: int) -> Query:
    """A copy of `query` bounded to one slice with its own post allowance."""
    return Query(topic=query.topic, start=s, end=e, limit=max(0, int(limit)),
                 options=dict(query.options or {}))


class SliceShare:
    """Shares a per-source post cap across slices; unused allowance rolls to later slices."""

    def __init__(self, total: int, n_slices: int):
        self.total = max(0, int(total))
        self.n = max(1, int(n_slices))
        self.got: dict[int, int] = {}
        self.cap: dict[int, int] = {}

    @property
    def used(self) -> int:
        return sum(self.got.values())

    def allowance(self, i: int) -> int:
        """Posts slice i may add: remaining cap / slices not yet recorded (rounded up)."""
        left = self.total - self.used
        slices_left = max(1, self.n - len(self.got))
        cap = max(0, -(-left // slices_left))
        self.cap[i] = cap
        return cap

    def record(self, i: int, got: int) -> None:
        self.got[i] = max(0, int(got))

    def summary(self, slices: list[tuple[datetime, datetime]]) -> str:
        """'per-day: 2026-11-19 43, 2026-11-18 43 (fewer than its share of 43), ...'."""
        parts = []
        for i, (s, e) in enumerate(slices):
            label = f"{s:%Y-%m-%d}" if s.date() == e.date() else f"{s:%Y-%m-%d}..{e:%Y-%m-%d}"
            if i not in self.got:
                parts.append(f"{label} not reached")
                continue
            note = (f" (fewer than its share of {self.cap[i]})"
                    if self.got[i] < self.cap.get(i, 0) else "")
            parts.append(f"{label} {self.got[i]}{note}")
        unit = "per-day" if all(s.date() == e.date() for s, e in slices) else "per-slice"
        return f"{unit}: " + ", ".join(parts)


def fill_slices(counts: list[int], total: int) -> list[int]:
    """Water-fill `total` across slices that have `counts[i]` candidate posts each.

    For walks with no server-side date filter (Telegram global search, LinkedIn post
    search): collect every candidate first, then call this. Slices the walk never reached
    (count 0) take nothing, so their share goes to the slices that have posts. Every slice
    gets an equal share; any share a slice cannot use passes to the others. Returns how
    many to keep per slice (sum <= total, keep[i] <= counts[i]).
    """
    keep = [0] * len(counts)
    left = max(0, int(total))
    open_ = [i for i, c in enumerate(counts) if c > 0]
    while left > 0 and open_:
        share = max(1, left // len(open_))
        nxt = []
        for i in open_:
            take = min(share, counts[i] - keep[i], left)
            keep[i] += take
            left -= take
            if keep[i] < counts[i]:
                nxt.append(i)
            if left == 0:
                break
        open_ = nxt
    return keep
