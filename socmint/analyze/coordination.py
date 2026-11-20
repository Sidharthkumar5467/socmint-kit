"""Coordination detection (docs/SPEC.md §6 "Coordination"; CLAUDE.md §3).

Evidence rules encoded here:

* **Coordination is not inauthenticity.** Two signal types must AGREE on the SAME
  accounts: an account pair is "strongly linked" only when >= 2 different signal types
  connect that very pair, and a flagged group is a connected set of strongly linked
  pairs. So every flagged member is tied to another member by two independent kinds of
  evidence; a component chained together by one signal never gets flagged because two
  of its members (or two unrelated accounts at its edge) share a second signal. Every
  flagged cluster carries the signals that fired, the post IDs behind each signal, and
  benign explanations where they plausibly apply.
* **Only the requested window.** Posts with ``in_window=False`` (e.g. a channel crawl
  that returned older posts) are never used as signal evidence, so a cluster shown for
  the analyst's date range always rests on posts from that range. They are still used
  to resolve WHO a repost/forward points at, and for account identity.
* **Only fields that exist are used.** Each signal skips accounts/posts that lack the
  field it needs (no creation date -> no creation-date signal for that account; no
  verified timestamp -> no simultaneity for that post). Nothing is assumed.
* **Reposts/forwards are amplification, not copy-paste.** A post whose relations say it
  is a repost or forward of another is excluded from the near-duplicate and
  simultaneity signals; it is only used by the amplification-loop signal.
* **Timing uses platform timestamps only** (`timestamp_verified=True`).

Signals are computed between AUTHORS, keyed ``"<source>:<author_id or author_handle>"``.

Scores (0-1) per signal, as reported in ``Signal.score``:

* near_duplicate_text: mean exact shingle Jaccard of the post pairs in the group.
* near_simultaneous_posting: mean of ``1 - lag / window`` over the post pairs.
* similar_creation_dates: mean of ``1 - gap_days / 3`` over the account pairs.
* reused_image: 1.0 for an identical URL, ``1 - hamming / 64`` for a pHash match; mean.
* amplification_loop: 1.0 when a reciprocal/cyclic loop exists in the group; otherwise
  the MEAN of the per-edge scores, where each one-way edge amplified ``n`` (>= 3) times
  scores ``min(0.9, 0.5 + 0.1 * (n - 3))``.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from urllib.parse import urlsplit

from socmint.schema import Cluster, Post, Signal
from socmint.util.log import get_logger
from socmint.util.timeutil import parse_ts

log = get_logger(__name__)

# ----------------------------------------------------------------------- parameters

SHINGLE_WORDS = 5  # word shingle size for near-duplicate text
MIN_TOKENS = 8  # texts shorter than this (after normalisation) are too generic to compare
CHAR_SHINGLE = 5  # fallback for unspaced scripts (CJK, Thai): character shingles
MIN_CHARS_UNSPACED = 30
JACCARD_THRESHOLD = 0.8  # LSH threshold and exact-Jaccard confirmation threshold
NUM_PERM = 128
SIMULTANEITY_WINDOW_S = 600  # 10 minutes
CREATION_WINDOW_DAYS = 3
PHASH_MAX_HAMMING = 4
REPEAT_AMPLIFY_MIN = 3  # one account amplifying the same other account >= 3 times

GENERIC_CAVEAT = (
    "Coordination signals show accounts acting similarly; they do not by themselves "
    "prove inauthentic behaviour."
)

# Substrings of avatar/image URLs that denote a platform placeholder image. Many
# unrelated accounts share these, so they are never "reused image" evidence.
PLACEHOLDER_IMAGE_MARKERS = (
    "default_profile",
    "default_avatar",
    "defaultavatar",
    "default-avatar",
    "no_avatar",
    "no-avatar",
    "noavatar",
    "placeholder",
    "blank_profile",
    "anonymous_avatar",
    "ghost_person",  # LinkedIn's generic silhouette
)

# Platform hosts are not "news domains": sharing x.com links is not syndication.
PLATFORM_HOSTS = (
    "x.com", "twitter.com", "t.co", "t.me", "telegram.me", "youtube.com", "youtu.be",
    "linkedin.com", "lnkd.in", "facebook.com", "instagram.com", "tiktok.com",
)

AMPLIFY_RELATIONS = ("repost", "forward", "quote")
COPY_RELATIONS = ("repost", "forward")  # posts that ARE someone else's content

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_PUNCT_RE = re.compile(r"[^\w\s]|_")
_WS_RE = re.compile(r"\s+")

SIGNAL_ORDER = (
    "near_duplicate_text",
    "near_simultaneous_posting",
    "similar_creation_dates",
    "reused_image",
    "amplification_loop",
)


# --------------------------------------------------------------------------- helpers


def author_key(post: Post) -> str | None:
    """Stable author key; None when the post carries no author identity at all."""
    ident = post.author_id or post.author_handle
    if not ident:
        return None
    return f"{post.source}:{ident}"


def normalize_text(text: str) -> str:
    """Lowercase, strip URLs, @mentions and punctuation, collapse whitespace."""
    t = _URL_RE.sub(" ", text or "")
    t = _MENTION_RE.sub(" ", t)
    t = t.lower()
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def shingles(text: str) -> set[str] | None:
    """5-word shingles of normalised text; None when the text is too short to compare.

    Unspaced scripts (Chinese, Japanese, Thai) yield few "words", so for them we fall
    back to 5-character shingles over the text with spaces removed.
    """
    norm = normalize_text(text)
    tokens = norm.split()
    if len(tokens) >= MIN_TOKENS:
        return {
            " ".join(tokens[i : i + SHINGLE_WORDS])
            for i in range(len(tokens) - SHINGLE_WORDS + 1)
        }
    compact = norm.replace(" ", "")
    if tokens and max(len(t) for t in tokens) >= 20 and len(compact) >= MIN_CHARS_UNSPACED:
        return {compact[i : i + CHAR_SHINGLE] for i in range(len(compact) - CHAR_SHINGLE + 1)}
    return None


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_placeholder_image(url: str) -> bool:
    u = url.lower()
    return any(m in u for m in PLACEHOLDER_IMAGE_MARKERS)


def phash_distance(a: str, b: str) -> int | None:
    """Hamming distance between two hex pHash strings (as produced by imagehash)."""
    try:
        if len(a) != len(b):
            return None
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return None


def _is_copy(post: Post) -> bool:
    return any(r.type in COPY_RELATIONS for r in post.relations)


def _verified_ts(post: Post) -> datetime | None:
    if not post.timestamp_verified:
        return None
    return parse_ts(post.published_at)


def _link_urls(post: Post) -> list[str]:
    """Link targets the platform reported as direct `link` relations."""
    return [r.target_url for r in post.relations if r.type == "link" and r.target_url]


def _norm_url(url: str) -> str:
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip().lower()
    host = (parts.hostname or "").lower().removeprefix("www.")
    path = parts.path.rstrip("/")
    query = f"?{parts.query}" if parts.query else ""
    return f"{host}{path}{query}"


def _domain(url: str) -> str | None:
    try:
        host = (urlsplit(url.strip()).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return None
    return host or None


def _is_platform_host(host: str) -> bool:
    return any(host == h or host.endswith("." + h) for h in PLATFORM_HOSTS)


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


# ----------------------------------------------------------------- evidence records


@dataclass
class _Ev:
    """One piece of pairwise evidence between two authors for one signal type."""

    posts: tuple[str, ...]
    value: float  # per-pair score component (see module docstring)
    note: str = ""
    lag_s: float | None = None  # simultaneity only: seconds between the two posts


@dataclass
class _Author:
    key: str
    source: str
    handles: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    created_at: datetime | None = None
    created_post: str | None = None  # post that exposed the creation date
    posts: list[Post] = field(default_factory=list)


# Graph: author pair -> signal type -> list of evidence.
_Graph = dict[tuple[str, str], dict[str, list[_Ev]]]


def _add(graph: _Graph, a: str, b: str, sig: str, ev: _Ev) -> None:
    if a == b:
        return
    graph.setdefault(_pair(a, b), {}).setdefault(sig, []).append(ev)


# --------------------------------------------------------------------------- signals


def _near_duplicate_pairs(posts: list[Post], keys: dict[str, str]) -> list[tuple[Post, Post, float]]:
    """Pairs of posts by different authors whose text is a near-duplicate.

    MinHash LSH (datasketch) proposes candidates; every candidate is confirmed with the
    exact shingle Jaccard so the reported score is not an estimate. Reposts/forwards
    are excluded (they are amplification, not independent copy-paste).
    """
    sh: dict[str, set[str]] = {}
    by_id: dict[str, Post] = {}
    for p in posts:
        if _is_copy(p) or p.post_id not in keys:
            continue
        s = shingles(p.text)
        if s:
            sh[p.post_id] = s
            by_id[p.post_id] = p

    candidates: set[tuple[str, str]] = set()
    try:
        from datasketch import MinHash, MinHashLSH

        lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=NUM_PERM)
        mh: dict[str, MinHash] = {}
        for pid in sorted(sh):
            m = MinHash(num_perm=NUM_PERM)
            m.update_batch([s.encode("utf-8") for s in sh[pid]])
            mh[pid] = m
            lsh.insert(pid, m)
        for pid, m in mh.items():
            for other in lsh.query(m):
                if other != pid:
                    candidates.add(_pair(pid, other))
    except ImportError:
        # Fallback (same result, exact): inverted shingle index. Used only when
        # datasketch is not installed; slower on very large runs.
        log.warning("datasketch not installed; using exact shingle index for near-duplicates")
        index: dict[str, list[str]] = defaultdict(list)
        for pid in sorted(sh):
            for s in sh[pid]:
                index[s].append(pid)
        for ids in index.values():
            if len(ids) > 1:
                candidates.update(_pair(a, b) for a, b in combinations(ids, 2))

    out: list[tuple[Post, Post, float]] = []
    for a, b in sorted(candidates):
        if keys[a] == keys[b]:
            continue  # same author repeating themselves is not coordination
        j = jaccard(sh[a], sh[b])
        if j >= JACCARD_THRESHOLD:
            out.append((by_id[a], by_id[b], j))
    return out


def _simultaneous_pairs(
    dup_pairs: list[tuple[Post, Post, float]], posts: list[Post], keys: dict[str, str]
) -> list[tuple[Post, Post, float, str]]:
    """Same-content post pairs by different authors within SIMULTANEITY_WINDOW_S.

    "Same content" = a confirmed near-duplicate pair, or the same direct link target.
    Only posts with platform-verified timestamps are used.
    """
    out: dict[tuple[str, str], tuple[Post, Post, float, str]] = {}

    def consider(p: Post, q: Post, basis: str) -> None:
        tp, tq = _verified_ts(p), _verified_ts(q)
        if tp is None or tq is None or keys[p.post_id] == keys[q.post_id]:
            return
        lag = abs((tp - tq).total_seconds())
        if lag <= SIMULTANEITY_WINDOW_S:
            k = _pair(p.post_id, q.post_id)
            if k not in out:
                out[k] = (p, q, lag, basis)

    for p, q, _ in dup_pairs:
        consider(p, q, "near-duplicate text")

    by_url: dict[str, list[tuple[datetime, Post]]] = defaultdict(list)
    for p in posts:
        if _is_copy(p) or p.post_id not in keys:
            continue
        ts = _verified_ts(p)
        if ts is None:
            continue
        for u in {_norm_url(u) for u in _link_urls(p)}:
            by_url[u].append((ts, p))
    for u, items in by_url.items():
        items.sort(key=lambda x: (x[0], x[1].post_id))
        # Sliding window keeps this linear-ish even for widely shared URLs.
        for i, (ti, pi) in enumerate(items):
            for tj, pj in items[i + 1 :]:
                if (tj - ti).total_seconds() > SIMULTANEITY_WINDOW_S:
                    break
                consider(pi, pj, f"same link {u}")
    return list(out.values())


def _image_urls(post: Post) -> list[tuple[str, str]]:
    """(url, kind) pairs for images on a post: avatar and media/thumbnails."""
    out: list[tuple[str, str]] = []
    if post.author_avatar_url:
        out.append((post.author_avatar_url, "avatar"))
    if not _is_copy(post):  # a repost naturally carries the original's media
        for m in post.media:
            if m.type in ("photo", "image") and m.url:
                out.append((m.url, "image"))
            if m.thumbnail:
                out.append((m.thumbnail, "thumbnail"))
    return [(u, k) for u, k in out if not is_placeholder_image(u)]


def _reused_image_edges(
    posts: list[Post], keys: dict[str, str], graph: _Graph, image_hashes: dict[str, str] | None
) -> None:
    # url -> author -> posts carrying it
    holders: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    kinds: dict[str, str] = {}
    for p in posts:
        k = keys.get(p.post_id)
        if not k:
            continue
        for u, kind in _image_urls(p):
            holders[u][k].append(p.post_id)
            kinds.setdefault(u, kind)

    def ev_posts(*lists: list[str]) -> tuple[str, ...]:
        # one representative post per author/url is enough to show the image
        return tuple(sorted({lst[0] for lst in lists}))

    # 1) identical URL used by different authors
    for u, by_author in holders.items():
        authors = sorted(by_author)
        for a, b in combinations(authors, 2):
            _add(graph, a, b, "reused_image",
                 _Ev(ev_posts(by_author[a], by_author[b]), 1.0, f"identical {kinds[u]} URL"))

    # 2) perceptual hash, only when a precomputed url->phash mapping was supplied.
    # This function never downloads images.
    if not image_hashes:
        return
    hashed = sorted(u for u in holders if image_hashes.get(u))
    for u, v in combinations(hashed, 2):
        d = phash_distance(image_hashes[u], image_hashes[v])
        if d is None or d > PHASH_MAX_HAMMING:
            continue
        for a in holders[u]:
            for b in holders[v]:
                if a != b:
                    _add(graph, a, b, "reused_image",
                         _Ev(ev_posts(holders[u][a], holders[v][b]), 1.0 - d / 64,
                             f"perceptual hash distance {d}"))


@dataclass
class _AmpInfo:
    # (src, dst) -> post ids of the amplifying posts
    edges: dict[tuple[str, str], list[str]]
    cyclic: set[tuple[str, str]]  # directed edges that lie on a cycle (SCC >= 2)
    flagged: set[tuple[str, str]]  # directed edges counted as amplification evidence


def _amplification(posts: list[Post], keys: dict[str, str], authors: dict[str, _Author],
                   graph: _Graph, target_ids: Iterable[str] | None = None) -> _AmpInfo:
    """Directed author->author graph from DIRECT repost/forward/quote relations.

    Only `posts` (the amplifying posts) are evidence. `target_ids` may additionally
    list collected post IDs (e.g. out-of-window originals) used only to resolve which
    account a relation points at.
    """
    by_id = {p.post_id for p in posts} | set(target_ids or ())
    handle_map: dict[tuple[str, str], str] = {}
    for a in authors.values():
        for h in a.handles:
            handle_map.setdefault((a.source, h.lstrip("@").lower()), a.key)

    edges: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in posts:
        src = keys.get(p.post_id)
        if not src:
            continue
        for r in p.relations:
            if r.type not in AMPLIFY_RELATIONS:
                continue
            dst: str | None = None
            if r.target_post_id and r.target_post_id in by_id:
                dst = keys.get(r.target_post_id)
            if dst is None and r.target_handle:
                h = r.target_handle.lstrip("@").lower()
                if h:
                    dst = handle_map.get((p.source, h)) or f"{p.source}:{h}"
                    if dst not in authors:
                        # Amplified account not itself collected: record its handle so
                        # the cluster can still name it.
                        authors[dst] = _Author(key=dst, source=p.source,
                                               handles={r.target_handle})
            if dst and dst != src:
                if p.post_id not in edges[(src, dst)]:
                    edges[(src, dst)].append(p.post_id)

    cyclic: set[tuple[str, str]] = set()
    if edges:
        import networkx as nx

        g = nx.DiGraph()
        g.add_edges_from(edges)
        for comp in nx.strongly_connected_components(g):
            if len(comp) >= 2:
                cyclic.update(e for e in edges if e[0] in comp and e[1] in comp)

    flagged: set[tuple[str, str]] = set()
    for (s, d), pids in edges.items():
        if (s, d) in cyclic:
            flagged.add((s, d))
            _add(graph, s, d, "amplification_loop",
                 _Ev(tuple(sorted(pids)), 1.0, "reciprocal/cyclic amplification"))
        elif len(pids) >= REPEAT_AMPLIFY_MIN:
            flagged.add((s, d))
            score = min(0.9, 0.5 + 0.1 * (len(pids) - REPEAT_AMPLIFY_MIN))
            _add(graph, s, d, "amplification_loop",
                 _Ev(tuple(sorted(pids)), score, f"repeated one-way amplification x{len(pids)}"))
    return _AmpInfo(dict(edges), cyclic, flagged)


def _creation_pairs(members: Iterable[str], authors: dict[str, _Author]) -> list[tuple[str, str, float]]:
    """Account pairs whose creation dates are within CREATION_WINDOW_DAYS.

    Never used to FORM a group (shared creation dates are common by chance); only
    evaluated inside a group that another signal already connected.
    """
    dated = sorted(m for m in members if authors[m].created_at is not None)
    out = []
    for a, b in combinations(dated, 2):
        gap = abs((authors[a].created_at - authors[b].created_at).total_seconds()) / 86400  # type: ignore[operator]
        if gap <= CREATION_WINDOW_DAYS:
            out.append((a, b, gap))
    return out


# ------------------------------------------------------------------------- grouping


def _components(nodes: Iterable[tuple[str, str]]) -> list[set[str]]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in nodes:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, set[str]] = defaultdict(set)
    for x in parent:
        groups[find(x)].add(x)
    return [g for g in groups.values() if len(g) >= 2]


def _strong_pairs(graph: _Graph, authors: dict[str, _Author]) -> list[tuple[str, str]]:
    """Author pairs DIRECTLY linked by >= 2 different signal types.

    Similar creation dates count as a type here only when the same pair is already
    linked by another signal (they never link a pair on their own).
    """
    strong = []
    for (a, b), sigs in graph.items():
        types = {t for t, evs in sigs.items() if evs}
        if types and _creation_pairs((a, b), authors):
            types.add("similar_creation_dates")
        if len(types) >= 2:
            strong.append((a, b))
    return strong


def _fmt_lag(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.0f} s"


def _evaluate(group: set[str], graph: _Graph, authors: dict[str, _Author],
              amp: _AmpInfo) -> tuple[list[Signal], dict] | None:
    """Return the fired signals for a group of strongly linked accounts, plus the raw
    numbers behind them (used by `cluster_summary`; never parsed back from `detail`).

    The pairwise ">= 2 signal types on the same pair" rule is applied BEFORE this (see
    `_strong_pairs`); here every signal type observed between group members is
    reported with its evidence, so each member's inclusion is explained.
    """
    need = 2
    stats: dict = {}
    per_type: dict[str, list[tuple[str, str, _Ev]]] = defaultdict(list)
    for (a, b), sigs in graph.items():
        if a in group and b in group:
            for t, evs in sigs.items():
                per_type[t].extend((a, b, e) for e in evs)

    signals: list[Signal] = []
    for t in SIGNAL_ORDER:
        if t == "similar_creation_dates":
            continue  # handled below; never forms a group on its own
        items = per_type.get(t, [])
        involved = {x for a, b, _ in items for x in (a, b)}
        if len(involved) < need:
            continue
        evs = [e for _, _, e in items]
        post_ids = sorted({p for e in evs for p in e.posts})
        score = round(sum(e.value for e in evs) / len(evs), 3)
        if t == "near_duplicate_text":
            detail = (f"{len(post_ids)} posts by {len(involved)} accounts with near-identical "
                      f"text (shingle Jaccard >= {JACCARD_THRESHOLD}, mean {score:.2f})")
            stats[t] = {"accounts": len(involved), "posts": len(post_ids), "mean": score}
        elif t == "near_simultaneous_posting":
            lags = sorted(e.lag_s or 0.0 for e in evs)
            n_text = sum(1 for e in evs if e.note == "near-duplicate text")
            detail = (f"{len(evs)} same-content post pairs ({n_text} near-duplicate text, "
                      f"{len(evs) - n_text} same link) by {len(involved)} accounts posted "
                      f"within {SIMULTANEITY_WINDOW_S // 60} min of each other "
                      f"(median lag {_fmt_lag(lags[len(lags) // 2])}; platform timestamps)")
            stats[t] = {"accounts": len(involved), "pairs": len(evs), "text_pairs": n_text,
                        "median_lag_s": lags[len(lags) // 2]}
        elif t == "reused_image":
            notes = Counter(e.note for e in evs)
            detail = (f"{len(involved)} accounts share images: "
                      + ", ".join(f"{n} x {k}" for k, n in sorted(notes.items())))
            stats[t] = {"accounts": len(involved),
                        "avatar": any("avatar" in k for k in notes),
                        "image": any("avatar" not in k for k in notes)}
        else:  # amplification_loop
            dir_edges = [e for e in amp.flagged if e[0] in group and e[1] in group]
            loops = [e for e in dir_edges if e in amp.cyclic]
            parts = []
            if loops:
                parts.append(f"{len(loops)} directed amplification links form a loop/reciprocal "
                             f"pattern")
            repeats = [e for e in dir_edges if e not in amp.cyclic]
            for s, d in sorted(repeats):
                parts.append(f"{_label(s, authors)} amplified {_label(d, authors)} "
                             f"{len(amp.edges[(s, d)])} times")
            detail = "; ".join(parts) + " (direct repost/forward/quote relations)"
            score = 1.0 if loops else score
            targets: dict[str, set[str]] = defaultdict(set)
            for s_, d_ in repeats:
                targets[d_].add(s_)
            stats[t] = {"accounts": len(involved),
                        "loop_accounts": len({x for e in loops for x in e}),
                        "targets": {_label(d_, authors): len(srcs)
                                    for d_, srcs in sorted(targets.items())}}
        signals.append(Signal(type=t, score=score, evidence_post_ids=post_ids, detail=detail))

    if not signals:
        return None
    # Creation dates count only as a corroborating signal inside an already-linked group.
    cpairs = _creation_pairs(group, authors)
    involved = {x for a, b, _ in cpairs for x in (a, b)}
    if len(involved) >= need:
        post_ids = sorted({authors[m].created_post for m in involved if authors[m].created_post})
        max_gap = max(g for _, _, g in cpairs)
        score = round(sum(1 - g / CREATION_WINDOW_DAYS for _, _, g in cpairs) / len(cpairs), 3)
        dated = sum(1 for m in group if authors[m].created_at is not None)
        signals.append(Signal(
            type="similar_creation_dates", score=score, evidence_post_ids=post_ids,
            detail=(f"{len(involved)} of {dated} accounts with a platform-reported creation "
                    f"date were created within {CREATION_WINDOW_DAYS} days of each other "
                    f"(max gap {max_gap:.1f} days)")))
        stats["similar_creation_dates"] = {"accounts": len(involved), "dated": dated,
                                           "max_gap_days": max_gap}
        signals.sort(key=lambda s: SIGNAL_ORDER.index(s.type))

    if len({s.type for s in signals}) < 2:
        return None  # the core rule: one signal type alone is never a flag
    return signals, stats


# ---------------------------------------------------------------------- summaries

PLATFORM_NAMES = {"x": "X", "telegram": "Telegram", "youtube": "YouTube",
                  "linkedin": "LinkedIn", "linkedin_ads": "LinkedIn Ads"}


def _platform(source: str) -> str:
    return PLATFORM_NAMES.get(source, source.replace("_", " ").title())


def _join_and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _count(k: int, n: int) -> str:
    """'k' accounts of a group of n, phrased for a follow-on clause."""
    if k >= n:
        return "both" if n == 2 else f"all {n}"
    return str(k)


def cluster_summary(n_members: int, sources: list[str], stats: dict) -> str:
    """One plain sentence describing what a flagged group did, built from signal numbers.

    `stats` is the per-signal numbers `_evaluate` computed (account counts, median lag,
    creation gap ...), so the sentence never depends on parsing `Signal.detail`.
    Example: "69 Telegram accounts posted near-identical text, a median 15 s apart; 53
    were created within 3 days of each other."
    """
    n = n_members
    plats = [_platform(s) for s in sorted(set(sources))]
    subject = (f"{n} {plats[0]} accounts" if len(plats) == 1
               else f"{n} accounts across {_join_and(plats)}")
    clauses: list[tuple[int, str]] = []  # (accounts involved, verb phrase)

    dup = stats.get("near_duplicate_text")
    sim = stats.get("near_simultaneous_posting")
    if dup and sim and sim["text_pairs"] and sim["accounts"] == dup["accounts"]:
        lag = sim["median_lag_s"]
        when = ("within the same second" if lag < 1 else
                f"a median {_fmt_lag(lag)} apart")
        clauses.append((dup["accounts"], f"posted near-identical text, {when}"))
        sim = None
    elif dup:
        clauses.append((dup["accounts"], "posted near-identical text"))
    if sim:
        what = ("posted the same content" if sim["text_pairs"] and
                sim["text_pairs"] < sim["pairs"] else
                "posted the same text" if sim["text_pairs"] else "shared the same link")
        clauses.append((sim["accounts"], f"{what} within {SIMULTANEITY_WINDOW_S // 60} min "
                                         f"of each other (median gap "
                                         f"{_fmt_lag(sim['median_lag_s'])})"))
    img = stats.get("reused_image")
    if img:
        what = ("the same profile picture and images" if img["avatar"] and img["image"]
                else "the same profile picture" if img["avatar"] else "the same images")
        clauses.append((img["accounts"], f"used {what}"))
    amp = stats.get("amplification_loop")
    if amp:
        if amp["loop_accounts"]:
            clauses.append((amp["loop_accounts"], "reposted or forwarded each other in a loop"))
        for target, k in amp["targets"].items():
            clauses.append((k, f"repeatedly reposted or forwarded {target}"))
    cre = stats.get("similar_creation_dates")
    if cre:
        clauses.append((cre["accounts"], f"were created within {CREATION_WINDOW_DAYS} days "
                                         "of each other"))
    if not clauses:
        return f"{subject} acted alike on two or more signals."
    k0, lead = clauses[0]
    head = f"{subject} {lead}" if k0 >= n else f"{k0} of {subject} {lead}"
    rest = [f"{_count(k, n)} {phrase}" for k, phrase in clauses[1:]]
    return "; ".join([head, *rest]) + "."


def _label(key: str, authors: dict[str, _Author]) -> str:
    a = authors.get(key)
    if a and a.handles:
        return sorted(a.handles)[0]
    return key


# ------------------------------------------------------------------- benign reasons


def _benign(group: set[str], signals: list[Signal], authors: dict[str, _Author],
            by_id: dict[str, Post], amp: _AmpInfo) -> list[str]:
    notes: list[str] = []

    # Same display name across accounts/platforms: likely one organisation's accounts.
    name_to: dict[str, set[str]] = defaultdict(set)
    for m in group:
        for n in authors[m].names:
            nn = normalize_text(n)
            if nn:
                name_to[nn].add(m)
    for n, ms in sorted(name_to.items()):
        if len(ms) >= 2:
            srcs = sorted({authors[m].source for m in ms})
            notes.append(
                f"{len(ms)} accounts share the display name '{n}' (on {', '.join(srcs)}): "
                "possibly one organisation's official accounts cross-posting.")

    # Mostly links to one non-platform domain: syndication / news wire.
    support = {p for s in signals for p in s.evidence_post_ids}
    dom_count: Counter[str] = Counter()
    for pid in support:
        p = by_id.get(pid)
        if not p:
            continue
        urls = _link_urls(p) + _URL_RE.findall(p.text or "")
        doms = {d for d in (_domain(u) for u in urls) if d and not _is_platform_host(d)}
        dom_count.update(doms)
    if dom_count and support:
        dom, n = dom_count.most_common(1)[0]
        if n / len(support) >= 0.5:
            notes.append(
                f"{n} of {len(support)} supporting posts link to {dom}: news syndication or "
                "wire-copy sharing can produce identical text and timing.")

    # One-way repeated amplification of a single account: fans/staff sharing an
    # official account is normal.
    if any(s.type == "amplification_loop" for s in signals):
        targets = Counter(d for (s, d) in amp.flagged
                          if s in group and d in group and (s, d) not in amp.cyclic)
        for d, n in sorted(targets.items()):
            notes.append(
                f"{n} account(s) repeatedly reposted/forwarded {_label(d, authors)}: routine "
                "sharing of an official or popular account's content looks like this.")

    # LinkedIn / other ads from the same advertiser.
    ads = [by_id[p] for p in support if p in by_id and by_id[p].kind == "ad"]
    if ads:
        adv = Counter(
            (p.ad_meta.advertiser_id if p.ad_meta and p.ad_meta.advertiser_id else None)
            or (p.author_name or "").strip().lower() or None
            for p in ads)
        adv.pop(None, None)
        if adv and adv.most_common(1)[0][1] >= 2:
            notes.append(
                "Several ads come from the same advertiser: one advertiser running multiple "
                "creatives (or an agency running them) is normal ad practice.")
        else:
            notes.append(
                "Group includes ads: similar creatives across advertisers can reflect a shared "
                "agency or template rather than covert coordination.")

    notes.append(GENERIC_CAVEAT)
    return notes


# ----------------------------------------------------------------------------- main


def detect_coordination(
    posts: list[Post], image_hashes: dict[str, str] | None = None
) -> list[Cluster]:
    """Flag groups of accounts that act together on >= 2 independent signal types.

    `image_hashes` is an optional precomputed mapping image URL -> hex pHash (from a
    separate enrichment step); without it only identical image URLs are compared.
    """
    # Exact re-collections of the same post (dedupe) would duplicate evidence.
    collected = [p for p in posts if not p.duplicate_of]
    # Evidence rule: only posts inside the requested window may support a cluster.
    # (in_window is True for posts whose timestamp is unknown; verify.py keeps those.)
    usable = [p for p in collected if p.in_window]
    keys: dict[str, str] = {}
    authors: dict[str, _Author] = {}
    # Identity (handles, names, creation date) may come from any collected post, but
    # in-window posts go first so the creation-date evidence post is in the window.
    for p in sorted(collected, key=lambda q: not q.in_window):
        k = author_key(p)
        if not k:
            continue
        keys[p.post_id] = k
        a = authors.setdefault(k, _Author(key=k, source=p.source))
        a.posts.append(p)
        if p.author_handle:
            a.handles.add(p.author_handle)
        if p.author_name:
            a.names.add(p.author_name)
        ca = parse_ts(p.author_created_at) if p.author_created_at else None
        if ca is not None and a.created_at is None:
            a.created_at, a.created_post = ca, p.post_id
    by_id = {p.post_id: p for p in usable}

    graph: _Graph = {}

    dup = _near_duplicate_pairs(usable, keys)
    for p, q, j in dup:
        _add(graph, keys[p.post_id], keys[q.post_id], "near_duplicate_text",
             _Ev((p.post_id, q.post_id), j))

    for p, q, lag, basis in _simultaneous_pairs(dup, usable, keys):
        _add(graph, keys[p.post_id], keys[q.post_id], "near_simultaneous_posting",
             _Ev((p.post_id, q.post_id), 1 - lag / SIMULTANEITY_WINDOW_S, basis, lag))

    _reused_image_edges(usable, keys, graph, image_hashes)
    amp = _amplification(usable, keys, authors, graph,
                         target_ids=[p.post_id for p in collected if p.post_id in keys])

    # Core rule (CLAUDE.md §3): only pairs linked by >= 2 signal types form groups.
    flagged: list[tuple[set[str], list[Signal], dict]] = []
    for core in _components(_strong_pairs(graph, authors)):
        got = _evaluate(core, graph, authors, amp)
        if got:
            flagged.append((core, *got))

    clusters: list[Cluster] = []
    for members, sigs, stats in flagged:
        sources = sorted({authors[m].source for m in members})
        support = sorted({p for s in sigs for p in s.evidence_post_ids})
        clusters.append(Cluster(
            id="",
            member_author_ids=sorted(members),
            signals=sigs,
            supporting_post_ids=support,
            benign_explanations=_benign(members, sigs, authors, by_id, amp),
            member_handles=sorted({h for m in members for h in authors[m].handles}),
            sources=sources,
            summary=cluster_summary(len(members), sources, stats),
        ))
    clusters.sort(key=lambda c: (-len(c.member_author_ids), -len(c.supporting_post_ids),
                                 c.member_author_ids[0]))
    for i, c in enumerate(clusters, 1):
        c.id = f"C{i}"
    log.info("coordination: %d candidate author pairs, %d clusters flagged",
             len(graph), len(clusters))
    return clusters
