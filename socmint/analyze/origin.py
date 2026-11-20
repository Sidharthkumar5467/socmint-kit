"""Earliest-observed posts and cross-platform spread per narrative (docs/SPEC.md §6).

Evidence rules encoded here (CLAUDE.md §3):
- Only `verify.analysable(posts)` (platform-verified timestamp, inside the window, not a
  duplicate) feed earliest posts, platform sequence and edges. A post without a verified
  timestamp never appears in an edge, even if a platform relation points at it.
- We never claim a true starting point. Everything is "earliest observed in collected
  data"; the trace note says so explicitly.
- DIRECT edges come only from relations the platform itself reported (quote, reply,
  repost, forward, link) that resolve to another collected post, by post id or by URL.
- INFERRED edges come only from similar content + later timing within one narrative, are
  typed "inferred", carry the similarity, and are never created where a direct edge
  already connects the pair.
- The earliest post is also scored for how well it matches the narrative
  (earliest_match); a weak one points the analyst to first_strong_post_id. Hubs count
  the direct edges leaving each post. Neither changes the edges.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import numpy as np

from socmint.schema import Edge, Hub, Narrative, OriginTrace, Post
from socmint.util.log import get_logger
from socmint.util.timeutil import iso, parse_ts
from socmint.verify import analysable

log = get_logger(__name__)

TIE_SECONDS = 60  # posts within this of the earliest are all "earliest"
MAX_EARLIEST = 3
SIM_THRESHOLD = 0.75  # cosine needed for an inferred edge
MAX_INFER_LAG = timedelta(days=7)
MAX_INFERRED_EDGES = 200  # per narrative
# Minimum comparable content for a post to take part in an inferred edge, measured after
# URLs and @mentions are stripped. Media-only posts (no caption) and URL-only posts have
# nothing to compare: the embedder maps all of them to the same placeholder vector, so
# without this gate any two of them would score cosine 1.0 and fabricate "similar content".
MIN_CONTENT_CHARS = 15  # word characters, scripts written with spaces between words
MIN_CONTENT_TOKENS = 3
MIN_CONTENT_CHARS_NOSPACE = 8  # CJK / Thai / etc., where one "token" can be a sentence

# How well a member matches its narrative = cosine of its vector to the centroid of the
# OTHER comparable usable members (leave-one-out, so a post never vouches for itself).
# A member is a "weak" (loose) match only when BOTH hold:
#   - it is an outlier within its own narrative: below the narrative's
#     MATCH_TYPICAL_QUANTILE (10th percentile) of member match scores, i.e. among the
#     bottom tenth of members; how far a member sits from the centre is only meaningful
#     against how far the narrative's own members sit (tightness varies widely: median
#     member cosine 0.67-0.99 on the sample run); and
#   - it is below the absolute bar MATCH_STRONG_ABS, the same 0.75 dense cosine the module
#     already uses to call two posts "similar content" (SIM_THRESHOLD), so a post that
#     would count as similar content to the narrative's centre is never called loose,
#     however tight the narrative (near-copy clusters put their 10th percentile >0.9).
# Otherwise "strong". The bar was the 25th percentile until 2026-11-20: that called a
# quarter of every narrative "loose", including clearly on-narrative earliest posts that
# merely sat in its outer quarter (sample run N4: an outage round-up at 0.717 vs a 25th
# percentile of 0.728). 10th = "outlier", not "outer quarter". On the sample run the
# decision does not move for any absolute bar between 0.65 and 0.75, so the bar is the
# existing similarity bar, not a tuned value. TF-IDF char n-gram cosines run much lower;
# 0.5 is not tuned on real data. Below MIN_MATCH_MEMBERS comparable members there is no
# percentile to speak of and the centroid is noise: on the sample run 3-4 member
# narratives scored 0.50-0.83 whatever the content, and an absolute-bar-only fallback
# (tried: 0.70) called 57 of 67 of them loose, mostly on-narrative earliest posts, so
# those narratives stay unjudged (None) rather than get a misleading label.
MATCH_STRONG_ABS = {"dense": SIM_THRESHOLD, "tfidf": 0.5}
MATCH_TYPICAL_QUANTILE = 10  # percentile of the narrative's member match scores
MIN_MATCH_MEMBERS = 5
MAX_HUBS = 5

DIRECT_TYPES = {"quote", "reply", "repost", "forward", "link"}

BASE_NOTE = (
    "Earliest observed in collected data; not necessarily the true origin. "
    "Direct edges are platform-reported quote/reply/repost/forward/link relations; "
    "inferred edges are similar content posted later with no platform link observed."
)


# ----------------------------------------------------------------------- URL matching

_X_STATUS_RE = re.compile(r"^/(?:[^/]+|i/web|i)/status(?:es)?/(\d+)")
_YT_SHORT_RE = re.compile(r"^/(?:shorts|live|embed)/([^/?#]+)")


def normalize_url(url: str | None) -> str | None:
    """Canonical key for matching a referenced URL to a collected post's URL.

    Strips scheme, `www.`/`m.`, query, fragment and trailing slash; treats twitter.com
    and x.com as the same (keyed by status id); folds t.me/s/<chan>/<n> into
    t.me/<chan>/<n>. YouTube keeps the identifying `v` (video) and `lc` (comment) query
    parameters, because there the id lives in the query string.
    """
    if not url:
        return None
    u = url.strip()
    if "://" not in u:
        u = "https://" + u
    try:
        parts = urlsplit(u)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    for prefix in ("www.", "m.", "mobile."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    path = parts.path.rstrip("/")
    if not host:
        return None

    if host in {"twitter.com", "x.com"}:
        m = _X_STATUS_RE.match(path)
        if m:
            return f"x.com/status/{m.group(1)}"
        return f"x.com{path.lower()}"

    if host in {"t.me", "telegram.me"}:
        p = path.lower()
        if p.startswith("/s/"):
            p = p[2:]
        return f"t.me{p}"

    if host in {"youtube.com", "youtu.be", "music.youtube.com"}:
        q = parse_qs(parts.query)
        vid = None
        if host == "youtu.be":
            vid = path.lstrip("/") or None
        elif path == "/watch":
            vid = (q.get("v") or [None])[0]
        else:
            m = _YT_SHORT_RE.match(path)
            vid = m.group(1) if m else None
        if vid:
            key = f"youtube.com/watch?v={vid}"
            lc = (q.get("lc") or [None])[0]
            return f"{key}&lc={lc}" if lc else key
        return f"youtube.com{path}"

    return f"{host}{path}"


# ---------------------------------------------------------------------------- helpers


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<!\w)@\w+")
_WORD_RE = re.compile(r"\w+")
# Scripts written without spaces between words (kana, CJK ideographs, Thai, Myanmar).
_NOSPACE_SCRIPT_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿฀-๿က-႟]")


def comparable_text(text: str | None) -> str:
    """Text with URLs and @mentions removed: what an inferred edge may compare."""
    t = _URL_RE.sub(" ", text or "")
    t = _MENTION_RE.sub(" ", t)
    return " ".join(t.split())


def has_comparable_content(text: str | None) -> bool:
    """True when a post has enough own text to support a "similar content" claim."""
    t = comparable_text(text)
    tokens = _WORD_RE.findall(t)
    nchars = sum(len(w) for w in tokens)
    if _NOSPACE_SCRIPT_RE.search(t):
        return nchars >= MIN_CONTENT_CHARS_NOSPACE
    return nchars >= MIN_CONTENT_CHARS and len(tokens) >= MIN_CONTENT_TOKENS


def _fmt_lag(seconds: float) -> str:
    sign = "+" if seconds >= 0 else "-"
    s = int(abs(seconds))
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    if d:
        return f"{sign}{d}d{h}h"
    if h:
        return f"{sign}{h}h{m:02d}m"
    return f"{sign}{m}m"


def _embed(texts: list[str]) -> tuple[np.ndarray, str]:
    """L2-normalised vectors of member texts and their space ("dense" | "tfidf");
    reuses the narrative embedder when available."""
    space = "tfidf"
    try:
        from socmint.analyze.embed import embed_texts

        vecs, method = embed_texts(texts)
        vecs = np.asarray(vecs, dtype=np.float32)
        space = "tfidf" if method.startswith("tfidf") else "dense"
    except Exception as e:  # noqa: BLE001 - fall back to a local lexical similarity
        log.warning("embedder unavailable for spread inference (%s); using TF-IDF", e)
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = [(t or "").lower() or "_" for t in texts]
        x = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(docs)
        vecs = x.toarray().astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms, space


def _similarity_matrix(texts: list[str]) -> np.ndarray:
    """Cosine similarity of member texts."""
    vecs, _space = _embed(texts)
    return vecs @ vecs.T


def _direct_edges(usable: dict[str, Post], times: dict[str, datetime]) -> list[Edge]:
    """All direct edges between usable posts, built only from platform relations."""
    by_url: dict[str, list[str]] = {}
    for pid, p in usable.items():
        key = normalize_url(p.url)
        if key:
            by_url.setdefault(key, []).append(pid)

    edges: list[Edge] = []
    seen: set[tuple[str, str, str]] = set()
    for pid, p in usable.items():
        for r in p.relations:
            if r.type not in DIRECT_TYPES:
                continue  # mentions are not post-to-post links
            targets: list[tuple[str, str]] = []  # (target pid, basis)
            if r.target_post_id and r.target_post_id in usable and r.target_post_id != pid:
                targets.append((r.target_post_id, f"{r.type} of {r.target_post_id}"))
            if not targets and r.target_url:
                key = normalize_url(r.target_url)
                basis = (f"links to {r.target_url}" if r.type == "link"
                         else f"{r.type} of {r.target_url}")
                for tid in by_url.get(key or "", []):
                    if tid != pid:
                        targets.append((tid, basis))
            for tid, basis in targets:
                k = (tid, pid, r.type)
                if k in seen:
                    continue
                seen.add(k)
                # from = the referenced (earlier) post, to = the post referencing it.
                lag = (times[pid] - times[tid]).total_seconds()
                edges.append(Edge(**{"from": tid}, to=pid, type="direct", basis=basis,
                                  relation=r.type, lag_seconds=lag))
    return edges


def _earliest(members: list[Post], times: dict[str, datetime]) -> list[str]:
    ordered = sorted(members, key=lambda p: (times[p.post_id], p.post_id))
    t0 = times[ordered[0].post_id]
    tied = [p.post_id for p in ordered
            if (times[p.post_id] - t0).total_seconds() <= TIE_SECONDS]
    return tied[:MAX_EARLIEST]


def _comparable_sorted(members: list[Post], times: dict[str, datetime]) -> list[Post]:
    """Members with comparable text, in platform-time order (ties by post id)."""
    comparable = [p for p in members if has_comparable_content(p.text)]
    return sorted(comparable, key=lambda p: (times[p.post_id], p.post_id))


def _inferred_edges(members: list[Post], times: dict[str, datetime],
                    linked: set[frozenset[str]],
                    vecs: np.ndarray | None = None) -> tuple[list[Edge], int, int]:
    """For each member, one inferred edge from its most similar earlier member.

    Cross-platform candidates are preferred; a same-platform predecessor is used only
    when no cross-platform one passes the threshold. Members without comparable text
    (media-only, URL-only, a few words) are neither source nor target of an inferred
    edge. `vecs`, when given, are the normalised vectors of _comparable_sorted(members)
    in that order (shared with the match scores). Returns (edges, dropped, no_text).
    """
    ordered = _comparable_sorted(members, times)
    no_text = len(members) - len(ordered)
    if len(ordered) < 2:
        return [], 0, no_text
    if vecs is None:
        sim = _similarity_matrix([comparable_text(p.text) for p in ordered])
    else:
        sim = vecs @ vecs.T
    candidates: list[Edge] = []
    for j, later in enumerate(ordered):
        tj = times[later.post_id]
        best: dict[bool, tuple[float, int]] = {}  # cross_platform? -> (sim, idx)
        for i in range(j):
            earlier = ordered[i]
            ti = times[earlier.post_id]
            if ti >= tj or tj - ti > MAX_INFER_LAG:
                continue  # need strictly later, within the lag window
            if frozenset((earlier.post_id, later.post_id)) in linked:
                continue  # a direct edge already explains this pair
            s = float(sim[i, j])
            if s < SIM_THRESHOLD:
                continue
            cross = earlier.source != later.source
            if cross not in best or s > best[cross][0]:
                best[cross] = (s, i)
        pick = best.get(True) or best.get(False)
        if not pick:
            continue
        s, i = pick
        earlier = ordered[i]
        lag = (tj - times[earlier.post_id]).total_seconds()
        candidates.append(Edge(
            **{"from": earlier.post_id}, to=later.post_id, type="inferred",
            basis=(f"similar content (cosine={s:.2f}) and later ({_fmt_lag(lag)}); "
                   "no platform link observed"),
            similarity=round(s, 4), lag_seconds=lag,
        ))
    # Keep the strongest; cross-platform first since spread across platforms matters most.
    src = {p.post_id: p.source for p in members}
    candidates.sort(key=lambda e: (src[e.from_] == src[e.to], -(e.similarity or 0)))
    dropped = max(0, len(candidates) - MAX_INFERRED_EDGES)
    return candidates[:MAX_INFERRED_EDGES], dropped, no_text


def _match_scores(ordered: list[Post], vecs: np.ndarray | None,
                  space: str) -> tuple[dict[str, float], float] | None:
    """(post id -> leave-one-out cosine to the narrative centroid, strong-match bar), or
    None when there are fewer than MIN_MATCH_MEMBERS comparable members."""
    if vecs is None or len(ordered) < MIN_MATCH_MEMBERS:
        return None
    total = vecs.sum(axis=0)
    scores: dict[str, float] = {}
    for i, p in enumerate(ordered):
        others = total - vecs[i]
        norm = float(np.linalg.norm(others)) or 1.0
        scores[p.post_id] = float(vecs[i] @ others) / norm
    typical = float(np.percentile(list(scores.values()), MATCH_TYPICAL_QUANTILE))
    bar = min(MATCH_STRONG_ABS.get(space, MATCH_STRONG_ABS["dense"]), typical)
    return scores, bar


def _earliest_match(earliest: list[str], ordered: list[Post],
                    match: tuple[dict[str, float], float] | None
                    ) -> tuple[str | None, float | None, str | None]:
    """(earliest_match, earliest_similarity, first_strong_post_id)."""
    if match is None:
        return None, None, None
    scores, bar = match
    first_strong = next((p.post_id for p in ordered if scores[p.post_id] >= bar - 1e-9), None)
    e0 = earliest[0] if earliest else None
    if e0 is None or e0 not in scores:  # earliest has no comparable text: not judged
        return None, None, first_strong
    sim = scores[e0]
    return ("strong" if sim >= bar - 1e-9 else "weak"), round(sim, 4), first_strong


def _strong_first_seen(ordered: list[Post], times: dict[str, datetime],
                       match: tuple[dict[str, float], float] | None) -> dict[str, str]:
    """Per platform, the time of its first member post at or above the strong-match bar.

    {} when the match was not judged; a platform whose posts are all loose matches (or
    have no comparable text) is left out."""
    if match is None:
        return {}
    scores, bar = match
    out: dict[str, datetime] = {}
    for p in ordered:
        s = scores.get(p.post_id)
        if s is None or s < bar - 1e-9:
            continue
        t = times[p.post_id]
        if p.source not in out or t < out[p.source]:
            out[p.source] = t
    return {src: iso(t) for src, t in sorted(out.items(), key=lambda kv: (kv[1], kv[0]))}  # type: ignore[misc]


def _hubs(edges: list[Edge]) -> list[Hub]:
    """Top MAX_HUBS posts by direct edges starting from them (e.g. a video and the
    comments replying to it), with relation counts and their inferred out-degree."""
    direct: dict[str, dict[str, int]] = {}
    inferred: dict[str, int] = {}
    for e in edges:
        if e.type == "direct":
            rel = direct.setdefault(e.from_, {})
            key = e.relation or "other"
            rel[key] = rel.get(key, 0) + 1
        else:
            inferred[e.from_] = inferred.get(e.from_, 0) + 1
    ranked = sorted(direct.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
    return [Hub(post_id=pid, direct_out=sum(rel.values()),
                relations=dict(sorted(rel.items(), key=lambda kv: (-kv[1], kv[0]))),
                inferred_out=inferred.get(pid, 0))
            for pid, rel in ranked[:MAX_HUBS]]


def _exclusion_reason(p: Post | None) -> str:
    """Why a narrative member is not usable, mirroring verify.analysable() + parse check."""
    if p is None:
        return "not in the collected posts"
    if p.duplicate_of:
        return "duplicate of another collected post"
    if not p.timestamp_verified:
        return "no verified platform timestamp"
    if not p.in_window:
        return "outside the date window"
    return "unparseable platform timestamp"


def _exclusion_note(member_ids: list[str], usable: dict[str, Post],
                    by_id: dict[str, Post]) -> tuple[int, str]:
    """(excluded count, "N reason; M reason") for members not usable for ordering."""
    counts: dict[str, int] = {}
    for pid in member_ids:
        if pid in usable:
            continue
        reason = _exclusion_reason(by_id.get(pid))
        counts[reason] = counts.get(reason, 0) + 1
    total = sum(counts.values())
    detail = "; ".join(f"{n} {r}" for r, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return total, detail


# ------------------------------------------------------------------------------ main


def trace_origins(posts: list[Post], narratives: list[Narrative]) -> list[OriginTrace]:
    """Earliest observed posts, platform sequence and direct/inferred edges per narrative."""
    usable: dict[str, Post] = {}
    times: dict[str, datetime] = {}
    for p in analysable(posts):
        t = parse_ts(p.published_at)
        if t is None:
            continue  # verified flag without a parseable time: cannot be ordered
        usable[p.post_id] = p
        times[p.post_id] = t

    # Prefer the canonical (non-duplicate) record when a post_id appears more than once.
    by_id: dict[str, Post] = {}
    for p in posts:
        if p.post_id not in by_id or (by_id[p.post_id].duplicate_of and not p.duplicate_of):
            by_id[p.post_id] = p

    all_direct = _direct_edges(usable, times)
    traces: list[OriginTrace] = []
    for n in narratives:
        unique_ids = list(dict.fromkeys(n.member_post_ids))  # de-duplicate, keep order
        member_ids = set(unique_ids)
        members = [usable[pid] for pid in unique_ids if pid in usable]
        excluded, why = _exclusion_note(unique_ids, usable, by_id)
        if not members:
            traces.append(OriginTrace(
                narrative_id=n.id, earliest_post_ids=[], edges=[],
                note=(f"No member post is usable for ordering: it needs a verified, "
                      f"parseable platform timestamp inside the window and must not be a "
                      f"duplicate ({excluded} member post(s) excluded: {why}), so no "
                      "earliest observed post or spread can be shown. Earliest observed "
                      "in collected data; not necessarily the true origin."),
            ))
            continue

        first_seen: dict[str, datetime] = {}
        for p in members:
            t = times[p.post_id]
            if p.source not in first_seen or t < first_seen[p.source]:
                first_seen[p.source] = t
        sequence = sorted(first_seen, key=lambda s: (first_seen[s], s))

        direct = [e for e in all_direct if e.from_ in member_ids or e.to in member_ids]
        linked = {frozenset((e.from_, e.to)) for e in direct}
        ordered = _comparable_sorted(members, times)
        vecs, space = (_embed([comparable_text(p.text) for p in ordered])
                       if len(ordered) >= 2 else (None, ""))
        inferred, dropped, no_text = _inferred_edges(members, times, linked, vecs)
        earliest = _earliest(members, times)
        scored = _match_scores(ordered, vecs, space)
        match, sim, first_strong = _earliest_match(earliest, ordered, scored)

        notes = [BASE_NOTE]
        if excluded:
            notes.append(f"{excluded} member post(s) excluded from earliest posts and "
                         f"edges ({why}).")
        if no_text:
            notes.append(f"{no_text} member post(s) had too little text (media-only, "
                         "link-only or very short) to compare and were left out of "
                         "inferred edges.")
        if dropped:
            notes.append(f"Inferred edges capped at {MAX_INFERRED_EDGES}; {dropped} weaker "
                         "inferred edge(s) dropped.")
        if match == "weak":
            notes.append(f"The earliest observed post only loosely matches this narrative "
                         f"(similarity {sim:.2f} to the other members); "
                         + (f"the earliest close match is {first_strong}."
                            if first_strong else "no member is a close match."))
        edges = direct + inferred
        traces.append(OriginTrace(
            narrative_id=n.id,
            earliest_post_ids=earliest,
            edges=edges,
            platform_first_seen={s: iso(t) for s, t in first_seen.items()},  # type: ignore[misc]
            platform_sequence=sequence,
            note=" ".join(notes),
            earliest_match=match,
            earliest_similarity=sim,
            first_strong_post_id=first_strong,
            hubs=_hubs(edges),
            platform_first_seen_strong=_strong_first_seen(ordered, times, scored),
        ))
    return traces
