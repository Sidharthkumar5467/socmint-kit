"""Narrative clustering, labelling and volume (docs/SPEC.md §6, Narratives).

Pipeline:
  candidate posts -> multilingual embeddings (analyze.embed) -> HDBSCAN (agglomerative
  for tiny sets) -> per cluster: representative examples, c-TF-IDF keywords, label +
  claim summary (LLM from member texts only, else keyword fallback) -> volumes.

Evidence rules encoded here:
- Every Narrative carries member_post_ids and example_post_ids (its evidence).
- Posts with unverified timestamps MAY be members (what they say is still evidence of
  the narrative), but volume_by_day, volume_by_platform and first_seen_at are computed
  ONLY from verify.analysable() members (and, for ads, only when the run start is inside
  the window). The excluded count is unverified_member_count.
- HDBSCAN noise points are not forced into a narrative.
- The LLM sees only member posts (text plus platform, @account, kind and, for ads, the
  advertiser name, and how many distinct accounts posted) and is told not to add facts;
  its summary is framed as what the posts claim, never as a fact. On failure we fall
  back to keywords.
- The same label call rates relevance to the run topic (threat | related | off_topic);
  it is a reading aid only and never drops posts. Keyword labels leave it None.
- The same call also rates severity (critical | high | medium | low, owner decision
  2026-11-19; narratives are ranked by it in analyze/brief.py). Severity rates what the
  content does, not whether it is true: truth words are removed from severity_reason,
  and a reason saying the subject is not actually discussed caps it at medium. Invalid,
  keyword-only and off_topic -> None. No extra LLM call. The rubric is written for any
  kind of subject (company, person, event, policy, product) and the people connected to
  it; a label that describes jokes/satire/mockery (_MOCKERY_RE) or posts debunking,
  refuting or fact-checking a claim (_DISMISS_RE) caps high at medium.
- Mixed clusters: the same call returns the numbers of shown posts that do NOT fit the
  label ("outliers"). Members flagged by the model, and members whose embedding is closer
  (by PRUNE_MARGIN) to the outliers' centroid than to the fitting examples' centroid, are
  removed from the narrative, but never more than the flagged share: k of m shown flagged
  -> at most ceil(k/m * n) of n members, furthest toward the outliers first. A removed
  set that is coherent (_coherent_group) and at least the minimum narrative size becomes
  its own narrative (new ID after the others; LLM-labelled within what is left of the
  label budget, else keyword-labelled); other removed posts are left unassigned, like
  clustering noise: they stay in the run and in posts.jsonl. Examples, keywords,
  volumes and first-seen are computed after pruning. Never pruned with fewer than 2
  fitting examples or no outliers. The minimum split size is the smallest narrative the
  run labels (_split_min_size), not HDBSCAN's min_cluster_size. A narrative where the
  labeller flagged >= PRUNE_MAX_FLAGGED_SHARE of the shown posts, or where pruning would
  leave fewer than LLM_LABEL_MIN_POSTS members, is kept whole and recorded as mixed.
  Totals go to run.json counts via `stats`; the per-narrative removed post IDs, split
  targets and mixed narratives go to run.json options via `trace` (shown on the
  narrative page and in the report).
- Severity caps after the label call: "high" is capped at medium when the label or
  summary describes mockery, unless a harm (impersonation, scam, fraud, phishing, leak,
  breach, threat) is named; critical/high is capped at medium for a narrative made only
  of one advertiser's ads unless the label, summary or a reason says (unhedged) the ads
  impersonate, scam, defraud or phish.
- LLM savings (owner decision 2026-11-20): only narratives with at least
  LLM_LABEL_MIN_POSTS members get an LLM label call; smaller ones (shown collapsed in the
  UI) get the keyword label, so they carry no relevance/severity and are not pruned.
  Label counts go to run.json counts via `stats` (narratives_label_llm / _keyword /
  _small).
- Original-language text is kept; nothing is translated or dropped by language.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, date, datetime

import numpy as np
from pydantic import BaseModel, Field, field_validator

from socmint.analyze.embed import embed_texts
from socmint.config import redact
from socmint.llm import LLM, LLMError
from socmint.schema import Narrative, Post
from socmint.util.log import get_logger
from socmint.util.timeutil import day_start, iso, parse_ts
from socmint.verify import analysable

log = get_logger(__name__)

MIN_TEXT_CHARS = 10  # word characters after stripping URLs/handles
TINY_DATASET = 8  # below this, HDBSCAN is unreliable: use agglomerative clustering
# Thresholds depend on the embedding space: char n-gram TF-IDF similarities are much
# lower than sentence-embedding similarities for posts that say the same thing.
# AGGLO: max average cosine distance inside a tiny-set cluster.
# NEIGHBOR: a member must have at least one other member this similar, else it is
# returned to noise (HDBSCAN on small sets can attach far-away outliers to a cluster).
AGGLO_COSINE_THRESHOLD = {"tfidf": 0.8, "dense": 0.5}
MIN_NEIGHBOR_SIM = {"tfidf": 0.12, "dense": 0.4}
# Outlier pruning (see module docstring): a member leaves the narrative when its cosine
# similarity to the outliers' centroid exceeds its similarity to the fitting examples'
# centroid by more than this. Small on purpose: it only stops near-ties from being
# removed. It was NOT tuned to the sample: offline replay of sample N6 (dense space,
# model assumed to flag the one review video it is shown) gives +0.085 for an unseen
# review (removed), +0.013 for another (kept: a near-tie, residual in EVALUATION.md),
# -0.012 for a card-fraud offer that fits and -0.118 for the nearest card-dump post.
PRUNE_MARGIN = {"tfidf": 0.01, "dense": 0.02}
# No pruning when the labeller flags at least this share of the posts it was shown: the
# cluster is mixed rather than a narrative with a few strays, and the label may describe
# the minority (review 2026-11-20, live re-run N53: 3 of 5 removed, label kept). The
# narrative is kept whole and recorded as mixed (run.json options narratives_mixed).
PRUNE_MAX_FLAGGED_SHARE = 0.5
# Above this size the O(n^2) agglomerative cross-check of HDBSCAN is skipped.
AGGLO_CROSSCHECK_MAX_N = 3000
MAX_EXAMPLES = 5
MAX_LLM_TEXTS = 12
LLM_TEXT_CHARS = 500
N_KEYWORDS = 8
# LLM labelling runs concurrently: one call per narrative was ~1.7 s and, sequential,
# took 436 s of an 812 s live run (244 narratives). The Anthropic client is shared by
# the worker threads (httpx connection pool; LLM counters/log are lock-protected).
LABEL_WORKERS = 8
# Wall-clock budget for all LLM label calls. Once it is spent no new call starts; the
# remaining narratives (the smallest, since labelling goes biggest first) get keyword
# labels. Calls already in flight finish (each is bounded by the client timeout).
LABEL_TIME_BUDGET_S = 120.0
# Narratives smaller than this get the keyword label without an LLM call (cost; owner
# decision 2026-11-20). They are collapsed in the UI; in the brief's main list only as a
# coordinated cluster's top narrative (coordination, not severity, is that finding).
LLM_LABEL_MIN_POSTS = 3
# Keys build_narratives(stats=) fills; the pipeline copies them into run.json counts.
STATS_KEYS = ("posts_pruned", "posts_pruned_unassigned", "narratives_split",
              "narratives_mixed", "narratives_label_llm", "narratives_label_keyword",
              "narratives_label_small")
# Keys build_narratives(trace=) fills (per-narrative pruning evidence); the pipeline copies
# them into run.json options so the GUI and report can show which posts left a narrative.
#   pruned_post_ids:   {origin nid: [post_id, ...]} every post removed from that narrative
#   pruned_split_into: {origin nid: new nid} when the removed posts became a narrative
#   narratives_mixed:  {nid: {"flagged": k, "shown": m, "reason": "share" | "size",
#                      "flagged_post_ids": [...]}} kept
#                      whole: the labeller flagged >= PRUNE_MAX_FLAGGED_SHARE of the shown
#                      posts ("share"), or pruning would leave < LLM_LABEL_MIN_POSTS ("size")
TRACE_KEYS = ("pruned_post_ids", "pruned_split_into", "narratives_mixed")
_clock = time.monotonic  # patched in tests

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HANDLE_RE = re.compile(r"(?<!\w)@\w+")
_WORD_RE = re.compile(r"[^\W\d_](?:[\w'’-]*\w)?", re.UNICODE)
_SENT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")

# Small multilingual stopword list (function words only) on top of sklearn's English list.
_EXTRA_STOP = set(
    """
    rt via amp https http www com html co t.me youtu be youtube watch video live
    el la los las un una unos unas de del al y o que en por para con sin se su sus es
    son como más mas pero lo le les ya muy este esta esto estos estas ese esa hay
    le la les un une des du de et ou que qui dans pour par sur avec sans ce cette ces
    est sont pas plus ne au aux il elle ils elles nous vous on son sa ses leur
    der die das ein eine einen einem einer und oder dass den dem des zu mit von für
    auf ist sind nicht auch sich es im am bei aus wie wir ihr sie er
    o a os as um uma de do da dos das e ou que em por para com não se seu sua é são
    il lo gli la le un uno una di da del della che e o per con non si è sono
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
    только ее мне было вот от меня еще нет о из ему это этот эта эти для
    في من على إلى عن مع هذا هذه ذلك التي الذي أن إن كان و أو لا ما هو هي
    ve bir bu da de ile için ama gibi daha çok
    """.split()
)


def _stopwords(topic: str) -> set[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    topic_words = {w.lower() for w in _WORD_RE.findall(topic or "")}
    return set(ENGLISH_STOP_WORDS) | _EXTRA_STOP | topic_words


def _strip(text: str) -> str:
    return _HANDLE_RE.sub(" ", _URL_RE.sub(" ", text or ""))


def _tokens(text: str, stop: set[str]) -> list[str]:
    out = []
    for w in _WORD_RE.findall(_strip(text).replace("#", " ").lower()):
        w = w.strip("'’-")
        if len(w) >= 3 and w not in stop:
            out.append(w)
    return out


def _non_trivial(text: str) -> bool:
    return sum(ch.isalnum() for ch in _strip(text)) >= MIN_TEXT_CHARS


# --------------------------------------------------------------------------- clustering


def _space(method: str) -> str:
    return "tfidf" if method.startswith("tfidf") else "dense"


def _cluster(vecs: np.ndarray, method: str = "dense") -> np.ndarray:
    """Return a label per row; -1 = noise (not assigned to any narrative)."""
    space = _space(method)
    labels = _raw_cluster(vecs, space)
    return _drop_outliers(vecs, labels, MIN_NEIGHBOR_SIM[space])


def _drop_outliers(vecs: np.ndarray, labels: np.ndarray, min_sim: float) -> np.ndarray:
    """Return members with no sufficiently similar co-member to noise; dissolve
    clusters left with fewer than 2 members. Noise is never forced into a narrative."""
    labels = labels.copy()
    for lab in sorted(set(labels.tolist()) - {-1}):
        idx = np.where(labels == lab)[0]
        sims = vecs[idx] @ vecs[idx].T
        np.fill_diagonal(sims, -1.0)
        weak = idx[sims.max(axis=1) < min_sim]
        labels[weak] = -1
        if (labels == lab).sum() < 2:
            labels[labels == lab] = -1
    return labels


def _agglomerative(vecs: np.ndarray, space: str) -> np.ndarray:
    """Average-linkage clustering with a cosine distance ceiling; singletons -> noise."""
    from sklearn.cluster import AgglomerativeClustering

    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=AGGLO_COSINE_THRESHOLD[space],
        metric="cosine",
        linkage="average",
    ).fit_predict(vecs)
    # Singletons are not narratives.
    counts = Counter(labels.tolist())
    return np.array([lab if counts[lab] >= 2 else -1 for lab in labels])


def _summary(labels: np.ndarray) -> tuple[int, int]:
    """(number of clusters, number of points assigned to a cluster)."""
    return len(set(labels.tolist()) - {-1}), int((labels >= 0).sum())


def _rescue_exact_copies(vecs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Verbatim copies (identical vectors from distinct posts, which dedupe keeps on
    purpose) that the clusterer left entirely as noise become their own group: two or
    more posts saying exactly the same thing are a narrative, not noise. HDBSCAN cannot
    separate zero-distance points, so without this a copy-paste wave can vanish."""
    labels = labels.copy()
    _, inverse = np.unique(np.round(vecs, 6), axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    next_lab = int(labels.max()) + 1 if len(labels) else 0
    for g in np.unique(inverse):
        idx = np.where(inverse == g)[0]
        if len(idx) >= 2 and (labels[idx] == -1).all():
            labels[idx] = next_lab
            next_lab += 1
    return labels


def _raw_cluster(vecs: np.ndarray, space: str) -> np.ndarray:
    n = len(vecs)
    if n < 2:
        return np.full(n, -1)
    if n < TINY_DATASET:
        return _rescue_exact_copies(vecs, _agglomerative(vecs, space))

    from sklearn.cluster import HDBSCAN

    # Small adaptive minimum: 3 for small collections, growing slowly for large ones so
    # a 2,000-post run does not fragment into hundreds of micro-narratives.
    mcs = _min_cluster_size(n)
    # Vectors are L2-normalised, so euclidean distance is monotone in cosine distance.
    # allow_single_cluster stays False: with True, HDBSCAN merged clearly distinct
    # topics into one cluster in testing. Its known blind spots (one dominant topic,
    # identical vectors, fragmentation of a single topic) are covered by the
    # agglomerative cross-check below instead.
    hdb = HDBSCAN(min_cluster_size=mcs, min_samples=min(mcs, 3), metric="euclidean", copy=True)
    labels = hdb.fit_predict(vecs)

    if n <= AGGLO_CROSSCHECK_MAX_N:
        h_k, h_cov = _summary(labels)
        alt = _agglomerative(vecs, space)
        a_k, a_cov = _summary(alt)
        # Prefer the agglomerative result when HDBSCAN found at most one cluster or left
        # most posts as noise, or when agglomerative explains strictly more posts with no
        # more clusters (HDBSCAN fragmenting one topic and dropping the rest to noise).
        if h_k <= 1 or h_cov < n / 2 or (a_cov > h_cov and a_k <= h_k):
            if a_cov >= h_cov:
                labels = alt
    labels = _split_dominant(vecs, labels, space)
    return _rescue_exact_copies(vecs, labels)


# A single cluster holding most of the assigned posts is under-segmentation: a homogeneous
# topic (every post mentions the subject) where the clusterer merged distinct stories into
# one mega-narrative that swallows the run. Live 2026-11-20, "bank of america" on a quiet
# week: 1,278 of 1,314 assigned posts landed in one cluster, mislabelled, while the real
# sub-stories (outages, how-tos, sponsorships, support replies) were lost inside it. Re-
# cluster that one blob on its own with average-linkage agglomerative (a cosine-distance
# ceiling, no density merging), which separates the distinct stories; keep the split only
# when it actually finds >= 2 sub-clusters, so a genuinely single narrative is left whole.
DOMINANT_CLUSTER_SHARE = 0.4
DOMINANT_CLUSTER_MIN = 60


def _split_dominant(vecs: np.ndarray, labels: np.ndarray, space: str) -> np.ndarray:
    labels = labels.copy()
    assigned = int((labels >= 0).sum())
    if assigned < DOMINANT_CLUSTER_MIN:
        return labels
    for lab in sorted(set(labels.tolist()) - {-1}):
        idx = np.where(labels == lab)[0]
        if (len(idx) < DOMINANT_CLUSTER_MIN or len(idx) <= DOMINANT_CLUSTER_SHARE * assigned
                or len(idx) > AGGLO_CROSSCHECK_MAX_N):  # O(n^2); do not re-split huge blobs
            continue
        sub = _agglomerative(vecs[idx], space)
        if _summary(sub)[0] < 2:
            continue  # genuinely one story: leave it whole
        base = int(labels.max()) + 1
        for s in sorted(set(sub.tolist()) - {-1}):
            labels[idx[sub == s]] = base
            base += 1
        # agglomerative singletons (sub == -1) keep the original label, so they stay one
        # residual narrative rather than being dropped; _drop_outliers dissolves it if weak.
    return labels


def _pick_examples(idx: list[int], vecs: np.ndarray, posts: list[Post]) -> tuple[list[int], np.ndarray]:
    """Up to MAX_EXAMPLES members closest to the centroid, preferring distinct authors
    and sources. Returns (example indices, member indices ordered by centrality)."""
    centroid = vecs[idx].mean(axis=0)
    norm = np.linalg.norm(centroid) or 1.0
    sims = vecs[idx] @ (centroid / norm)
    order = [idx[i] for i in np.argsort(-sims, kind="stable")]

    chosen: list[int] = []
    authors: set[str] = set()
    sources: set[str] = set()

    def author_key(p: Post) -> str:
        return f"{p.source}:{p.author_id or p.author_handle or p.post_id}"

    passes = (
        lambda p: author_key(p) not in authors and p.source not in sources,
        lambda p: author_key(p) not in authors,
        lambda p: True,
    )
    for ok in passes:
        for i in order:
            if len(chosen) >= MAX_EXAMPLES:
                break
            if i in chosen or not ok(posts[i]):
                continue
            chosen.append(i)
            authors.add(author_key(posts[i]))
            sources.add(posts[i].source)
    return chosen, np.array(order)


def _ctfidf_keywords(
    cluster_docs: dict[int, list[str]], background: list[str], stop: set[str]
) -> dict[int, list[str]]:
    """Class-based TF-IDF: each cluster's concatenated text is one 'document'; noise
    posts form one extra background class so common-to-everything words score low."""
    classes: dict[int, Counter] = {
        k: Counter(t for d in docs for t in _tokens(d, stop)) for k, docs in cluster_docs.items()
    }
    bg = Counter(t for d in background for t in _tokens(d, stop))
    all_classes = list(classes.values()) + ([bg] if bg else [])
    total_freq: Counter = Counter()
    for c in all_classes:
        total_freq.update(c)
    avg_words = (sum(sum(c.values()) for c in all_classes) / max(1, len(all_classes))) or 1.0
    out: dict[int, list[str]] = {}
    for k, c in classes.items():
        size = sum(c.values()) or 1
        scored = [
            (tf / size * math.log(1 + avg_words / total_freq[t]), t) for t, tf in c.items()
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out[k] = [t for _, t in scored[:N_KEYWORDS]]
    return out


# ------------------------------------------------------------------------------ labels


# Labels are shown as headings in the GUI and report: a complete short phrase. The model
# is asked for <= LABEL_MAX_WORDS words and LABEL_TARGET_CHARS characters; the hard limit
# is LABEL_MAX_CHARS, and a label is cut (at a word boundary, never leaving a dangling
# "in"/"and"/"of") only when it is longer than that, as a last resort. A hard 8-word cap
# used to cut 84 of 272 sample-run labels mid-phrase ("Bank of America increasing Bitcoin
# ETF holdings in"; the model had written "... in Q3"). The longest of those 272 raw
# labels is 84 chars; 80 keeps all but that one whole while the prompt targets 60.
LABEL_MAX_WORDS = 10
LABEL_TARGET_CHARS = 60
LABEL_MAX_CHARS = 80
# Reasons: the model is asked for <= 15 words; the full reason is kept up to this hard
# cap and, only when longer, cut at the last clause boundary (never mid-clause).
REASON_MAX_CHARS = 160
RELEVANCE_REASON_MAX_CHARS = REASON_MAX_CHARS  # older name, kept for callers and tests
RELEVANCE_VALUES = ("threat", "related", "off_topic")
SEVERITY_VALUES = ("critical", "high", "medium", "low")  # schema.Narrative.severity
# Words a label must not end on after a cut (English function words; labels are English).
_DANGLING = set(
    """
    a an the and or but nor so yet of in on at to for from by with without into onto
    over under about after before between during against via vs versus as than per
    amid among across toward towards through while whose which that who whom its their
    his her our your my this these those is are was were be been being has have had
    """.split()
)


_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("`", "`"))


def _clean_label(text: str, limit: int = LABEL_MAX_CHARS) -> str:
    """A complete short label: whitespace collapsed, wrapping quotes and trailing
    punctuation removed. Longer than `limit`: cut at the last word boundary that fits,
    then drop trailing function words and separators so it never ends mid-phrase."""
    t = re.sub(r"\s+", " ", text or "").strip()
    for a, b in _QUOTE_PAIRS:  # only quotes that wrap the whole label
        if len(t) >= 2 and t[0] == a and t[-1] == b:
            t = t[1:-1].strip()
    t = t.rstrip(" ;:,-–—/|&")  # keep a final "." (e.g. "U.S.")
    if len(t) <= limit:
        return t
    words = t[: limit + 1].split(" ")
    if len(words) > 1:
        words = words[:-1]  # the last piece is partial (or ends exactly at the limit)
    while len(words) > 1 and (
        words[-1].lower().strip(".,;:()") in _DANGLING or not any(c.isalnum() for c in words[-1])
    ):
        words.pop()
    out = " ".join(words).rstrip(" .;:,-–—/|&(")
    return out[:limit]  # a single overlong word: hard cut as the last resort


_STRONG_END_RE = re.compile(r"[;:.!?)\]–—]+(?=\s)|\s[–—-]\s")
_COMMA_RE = re.compile(r",(?=\s)")


def _clean_reason(reason: str | None, limit: int = REASON_MAX_CHARS) -> str:
    """One short reason, whole when it fits `limit`. Longer: cut at the last clause
    boundary (comma, semicolon, dash, sentence end) that keeps at least a third of the
    text; without one, at the last word that is not a dangling function word, plus '…'.
    A sample-run reason cut at a fixed width used to end "... name, zip and phone raises…"."""
    r = re.sub(r"\s+", " ", reason or "").strip()
    if len(r) <= limit:
        return r
    head = r[: limit + 1]
    # a sentence/semicolon/dash boundary first; a comma only when there is none
    cuts = ([m.start() for m in _STRONG_END_RE.finditer(head) if m.start() >= limit // 3]
            or [m.start() for m in _COMMA_RE.finditer(head) if m.start() >= limit // 3])
    if cuts:
        out = head[: cuts[-1]].rstrip(" ,;:–—-")
        words = out.split(" ")
        while len(words) > 1 and words[-1].lower().strip(".,;:()") in _DANGLING:
            words.pop()
        return " ".join(words)
    words = head[:limit - 1].split(" ")[:-1] or [head[:limit - 1]]
    while len(words) > 1 and (words[-1].lower().strip(".,;:()") in _DANGLING
                              or not any(c.isalnum() for c in words[-1])):
        words.pop()
    return " ".join(words).rstrip(" ,;:–—-") + "…"


# Severity rates what the content DOES, not whether it is true: truth words are removed
# from severity_reason (the prompt forbids them too; the claims step judges truth).
_TRUTH_WORD_RE = re.compile(
    # compounds such as "false-flag" or "true-crime" name a kind of content, not a truth
    # judgement, and are kept (live re-run 2026-11-19 turned "false-flag" into "-flag")
    r"\b(?:un)?(?:confirmed|verified)(?:-\w+)?\b|(?<!-)\b(?:false(?:ly)?|true|untrue)\b(?!-)",
    re.IGNORECASE)
# A reason saying the subject is not actually discussed caps severity at medium.
# Phrasings found in live runs (2026-11-19/20): "no Bank of America-specific attack is
# named", "does not focus on Bank of America specifically", "Bank of America only appears
# in one unrelated map video", "not a direct attack on the bank". "not only" is excluded
# ("not only names the bank but ..." is the opposite). The window never crosses
# punctuation, so "no withdrawals possible, bank named" does not trigger.
_NEG = r"(?:\bnot\b(?!\s+only\b)|\bnever\b|\bno\b|\bwithout\b|n't\b)"
_SEP = r"[^\w.,;:!?]+"  # between two words, never across punctuation
_NOT_NAMED_RE = re.compile(
    # negation, up to 8 words, then a naming/discussing verb
    _NEG + r"(?:" + _SEP + r"\w+){0,8}?" + _SEP + r"(?:named|names|name|naming|mentioned|"
    r"mention|mentions|discussed|discuss|discusses|referenced|reference|references|identified|cited)\b"
    # negation, up to 4 words, then a focus/target word
    r"|" + _NEG + r"(?:" + _SEP + r"\w+){0,4}?" + _SEP + r"(?:focus|focuses|focused|"
    r"focussed|centre|centred|center|centered|target|targets|targeted|about|directed|aimed|specific|specifically|"
    r"direct|directly)\b"
    r"|\bonly\s+(?:in passing|mentioned|named|referenced|cited|tangentially|incidentally|"
    r"appears?|appearing|shows? up|surfaces?)\b"
    r"|\b(?:appears?|mentioned|named|referenced)\s+only\b"
    r"|\b(?:in passing|tangential(?:ly)?|incidental(?:ly)?|peripheral(?:ly)?)\b"
    r"|\bunrelated\s+to\s+(?:the\s+)?(?:bank|subject|topic|company|brand)\b",
    re.IGNORECASE)


def _strip_truth_words(reason: str) -> str:
    r = _TRUTH_WORD_RE.sub("", reason or "")
    r = re.sub(r"\s+([,;:.])", r"\1", re.sub(r"\s{2,}", " ", r)).strip(" ,;:-–—")
    r = re.sub(r"^(?:and|or|but)\s+", "", r, flags=re.IGNORECASE)
    return r[:1].upper() + r[1:] if r else r


def _clean_relevance(value: str | None, reason: str | None) -> tuple[str | None, str]:
    """Validate the model's relevance; anything outside RELEVANCE_VALUES -> (None, "")."""
    v = re.sub(r"[\s-]+", "_", (value or "").strip().lower())
    if v not in RELEVANCE_VALUES:
        return None, ""
    # same truth-word ban as severity_reason: relevance says what the posts are about
    return v, _clean_reason(_strip_truth_words(reason or ""))


def _clean_severity(value: str | None, reason: str | None, relevance: str | None,
                    relevance_reason: str | None = None) -> tuple[str | None, str]:
    """Validate the model's severity; anything outside SEVERITY_VALUES -> (None, "").
    An off_topic narrative carries no severity (it is not about the subject). Truth words
    are removed from the reason; when either reason says the subject is not actually
    named or discussed, critical/high is capped at medium."""
    v = (value or "").strip().lower()
    if relevance == "off_topic" or v not in SEVERITY_VALUES:
        return None, ""
    r = _clean_reason(_strip_truth_words(reason or ""))
    if v in ("critical", "high") and any(
            _NOT_NAMED_RE.search(x or "") for x in (reason, relevance_reason)):
        v = "medium"
    return v, r


# A label or summary saying the posts joke about, mock or satirise something describes
# content that does not itself spread alarm; the rubric makes it low (medium if spread
# widely), so "high" is capped at medium. Sample run: N55 "Commenters mock claims of a
# cyber attack" and N202 "Users joke about withdrawing money" were rated high. "critical"
# is left alone (mockery around leaked data is still the leaked data). Negated forms
# ("no joke", "not satire") do not count; "ridiculous" and "meme" are not matched.
# "spoof" and "parody" are NOT mockery words here: "spoofed calls", "spoofed emails" and
# "parody accounts impersonating X" are fraud/impersonation (review 2026-11-20).
_MOCKERY_RE = re.compile(
    r"\b(?:mock(?:s|ed|ing|ery)?|jok(?:e|es|ed|ing|ingly)|satir(?:e|es|ical|ise|ises|ize|izes|ising|izing)"
    r"|sarcas(?:m|tic|tically)|ridicul(?:e|es|ed|ing)|lampoon(?:s|ed|ing)?)\b", re.IGNORECASE)
_MOCKERY_NEG_RE = re.compile(r"\b(?:no|not|never|isn't|aren't|wasn't)\s+(?:\w+\s+)?$", re.IGNORECASE)
# Same principle for posts that debunk, refute or fact-check a claim (review 2026-11-20):
# they push back on the claim, they do not spread it. Checked in the LABEL only (a summary
# like "the ministry refuted the report" describes the subject's act, not the posters').
# "dismiss" is deliberately NOT included: "Commenters dismiss Bank of America software bug
# as cyberattack" (live re-run N86, the case the review cited) means the commenters reject
# the official explanation and ASSERT the attack (its summary: "Posts claim that a Bank of
# America outage was a cyber attack"), so it spreads the alarm and high was right.
# Same negation and harm-word exceptions as mockery.
_DISMISS_RE = re.compile(
    r"\b(?:debunk(?:s|ed|ing)?|refut(?:e|es|ed|ing|ation)|fact[- ]?check(?:s|ed|ing|ers?)?)\b",
    re.IGNORECASE)
# Harm words: when the label, summary or a reason names one, the posts are about the harm
# itself (mocking victims of a leak is still about the leak), so the mockery cap is
# skipped. _FRAUD_RE is the narrower set that lifts the single-advertiser ad cap.
# A product or service that fights the harm ("fraud detection", "anti-phishing training",
# "scam awareness") names a defence, not the harm: live re-run N8 "ThreatMark ad promoting
# behavioral fraud intelligence for banks".
_DEFENCE = (r"(?<!anti-)(?<!anti)(?![\w-]*[\s-]+(?:prevention|detection|protection|"
            r"intelligence|awareness|monitoring|training|defen[cs]e|solutions?|software|"
            r"platform|tools?|management|analytics)\b)")
_FRAUD_WORDS = r"impersonat\w*|scam\w*|fraud\w*|phish\w*"
_FRAUD_RE = re.compile(rf"\b{_DEFENCE}(?:{_FRAUD_WORDS})", re.IGNORECASE)
_HARM_RE = re.compile(rf"\b{_DEFENCE}(?:{_FRAUD_WORDS}|leak(?:s|ed|ing)?\b|"
                      r"breach(?:es|ed)?\b|threat(?:s|en\w*)?\b)", re.IGNORECASE)


def _describes_harm(*texts: str) -> bool:
    return any(_HARM_RE.search(t or "") for t in texts)


# "possible impersonation", "may be a scam": the labeller's hedge, not a description of
# what the ads do (live re-run N75: a dental clinic's ads, reason "possible
# impersonation", label and summary describing an ordinary offer).
_HEDGE_BEFORE_RE = re.compile(r"\b(?:possibl[ey]|potential(?:ly)?|may|might|could|perhaps|"
                              r"maybe|(?:appears?|seems?)\s+to)\b(?:\W+\w+){0,2}\W*$",
                              re.IGNORECASE)


def _describes_fraud(*texts: str) -> bool:
    """True when a text says the posts impersonate, scam, defraud or phish (a hedged
    mention, or a product that fights fraud, does not count)."""
    for t in texts:
        for m in _FRAUD_RE.finditer(t or ""):
            if not _HEDGE_BEFORE_RE.search((t or "")[max(0, m.start() - 40): m.start()]):
                return True
    return False


def _single_advertiser_ads(members: list[Post]) -> bool:
    """True when every member is an ad and all come from one advertiser. One ordinary
    advertiser's ads are capped at medium severity unless the label, summary or a reason
    describes impersonation, scam, fraud or phishing (live re-run 2026-11-20: a dental
    clinic's ad naming the bank was rated high)."""
    if not members or not all(p.kind == "ad" or p.ad_meta is not None for p in members):
        return False
    keys = {((p.ad_meta.advertiser_id if p.ad_meta else None) or p.author_id
             or p.author_name or p.author_handle or "").strip().lower() for p in members}
    return len(keys) == 1 and "" not in keys


def _describes_mockery(*texts: str) -> bool:
    for t in texts:
        for m in _MOCKERY_RE.finditer(t or ""):
            if not _MOCKERY_NEG_RE.search((t or "")[: m.start()]):
                return True
    return False


def _describes_dismissal(label: str) -> bool:
    """True when the label says the posts debunk, refute or fact-check a claim."""
    for m in _DISMISS_RE.finditer(label or ""):
        if not _MOCKERY_NEG_RE.search((label or "")[: m.start()]):
            return True
    return False


def _clean_outliers(values: object, n_shown: int) -> tuple[int, ...]:
    """1-based [n] numbers from the model -> sorted unique 0-based positions; numbers
    outside the shown range are dropped."""
    out: set[int] = set()
    for x in values or ():
        try:
            k = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= k <= n_shown:
            out.add(k - 1)
    return tuple(sorted(out))


def _prune_outliers(idx: list[int], shown: list[int], outliers: tuple[int, ...],
                    vecs: np.ndarray, margin: float) -> tuple[list[int], list[int]]:
    """Split narrative members `idx` into (kept, removed).

    shown: the member indices the labeller saw, in prompt order; outliers: positions in
    `shown` the model said do not fit. Flagged members are always removed. Other members
    are candidates when their similarity to the outliers' centroid exceeds their
    similarity to the fitting examples' centroid by more than `margin`.

    Removal is capped by the share the model flagged: k of m shown -> at most
    ceil(k / m * n) of the n members (never fewer than the k flagged), taking candidates
    furthest toward the outlier centroid first. The model judged a sample; the cap keeps
    a near-tie in embedding space from spreading one flagged post into most of the
    narrative (live re-run 2026-11-20: 2 of 12 flagged removed 72 of 119 members).
    Fitting examples are always kept. No pruning without outliers or with fewer than 2
    fitting examples. `removed` is ordered flagged first, then by distance."""
    out_set = {shown[k] for k in outliers if k < len(shown)}
    fit_idx = [i for k, i in enumerate(shown) if k not in set(outliers)]
    if not out_set or len(fit_idx) < 2:
        return list(idx), []
    out_idx = [i for i in idx if i in out_set]

    def centroid(rows: list[int]) -> np.ndarray:
        c = vecs[rows].mean(axis=0)
        return c / (np.linalg.norm(c) or 1.0)

    c_fit, c_out = centroid(fit_idx), centroid(out_idx or sorted(out_set))
    fit_set = set(fit_idx)
    cap = max(len(out_idx), math.ceil(len(out_set) / max(1, len(shown)) * len(idx)))
    scored = []
    for i in idx:
        if i in fit_set or i in out_set:
            continue
        d = float(vecs[i] @ c_out) - float(vecs[i] @ c_fit)
        if d > margin:
            scored.append((-d, i))
    scored.sort()
    removed = out_idx + [i for _, i in scored[: max(0, cap - len(out_idx))]]
    gone = set(removed)
    return [i for i in idx if i not in gone], removed


def _min_cluster_size(n: int) -> int:
    """Smallest group the clusterer keeps as a narrative for a set of n candidates:
    2 for tiny sets (agglomerative), else HDBSCAN's min_cluster_size."""
    return 2 if n < TINY_DATASET else max(3, min(15, n // 60))


def _split_min_size(n: int) -> int:
    """Smallest removed group that becomes its own narrative: the smallest narrative the
    run labels (LLM_LABEL_MIN_POSTS; 2 for tiny sets), NOT HDBSCAN's min_cluster_size,
    which grows to 15 on large runs while the run keeps narratives of 2-14 posts (review
    2026-11-20: a coherent 10-post story pruned from N3 went to no narrative)."""
    return min(_min_cluster_size(n), LLM_LABEL_MIN_POSTS)


def _coherent_group(rows: list[int], vecs: np.ndarray, space: str) -> list[int]:
    """The members of `rows` that form one coherent group by the clusterer's own
    standards: each keeps at least one co-member >= MIN_NEIGHBOR_SIM (the same check
    as _drop_outliers), and the group's mean pairwise similarity is >= that threshold
    too (a removed set mixing two unrelated themes is not one narrative). [] if not."""
    if len(rows) < 2:
        return []
    min_sim = MIN_NEIGHBOR_SIM[space]
    rows = list(rows)
    while len(rows) >= 2:
        sims = vecs[rows] @ vecs[rows].T
        np.fill_diagonal(sims, -1.0)
        ok = [r for r, best in zip(rows, sims.max(axis=1), strict=True) if best >= min_sim]
        if len(ok) == len(rows):
            break
        rows = ok
    if len(rows) < 2:
        return []
    sims = vecs[rows] @ vecs[rows].T
    mean = (float(sims.sum()) - float(np.trace(sims))) / (len(rows) * (len(rows) - 1))
    return rows if mean >= min_sim else []


class NarrativeLabel(BaseModel):
    label: str = Field(
        description=f"Short neutral label for the narrative: a complete phrase, at most "
        f"{LABEL_MAX_WORDS} words and {LABEL_TARGET_CHARS} characters."
    )
    claim_summary: str = Field(
        description="One sentence stating what the posts claim, framed as 'Posts claim that ...'."
    )
    # Plain str, not a Literal: the value is validated in _clean_relevance so a bad value
    # costs only the relevance (-> None), never the label and summary.
    relevance: str = Field(
        default="",
        description="One of: threat | related | off_topic (relative to the search topic).",
    )
    relevance_reason: str = Field(default="", description="One short clause, at most 15 words.")
    # Plain str for the same reason; validated in _clean_severity.
    severity: str = Field(
        default="",
        description="One of: critical | high | medium | low (severity for the topic's subject).",
    )
    severity_reason: str = Field(default="", description="One short clause, at most 15 words.")
    outliers: list[int] = Field(
        default_factory=list,
        description="Numbers [n] of shown posts that do not fit the label; empty if all fit.",
    )

    @field_validator("outliers", mode="before")
    @classmethod
    def _lenient_outliers(cls, v: object) -> list[int]:
        # A malformed list costs only the pruning (-> []), never the label.
        if not isinstance(v, list | tuple):
            return []
        out = []
        for x in v:
            try:
                out.append(int(str(x).strip().strip("[]")))
            except ValueError:
                continue
        return out


LLM_SYSTEM = (
    "You label clusters of social media posts for a threat-intelligence analyst. "
    "Use ONLY the posts provided. Do not add facts, names, numbers or context that are "
    "not in the posts. Do not judge whether the claims are true or false: severity rates "
    "what the content does, never its truth. Stay in the posts' own "
    "framing: the summary must describe what the posts claim (e.g. 'Posts claim that ...'), "
    "never state it as fact. Posts may be in any language; write the label and summary in "
    f"English. The label is a complete phrase of at most {LABEL_MAX_WORDS} words."
)

# Definitions match schema.Narrative.relevance.
RELEVANCE_PROMPT = (
    "Also rate relevance to the search topic:\n"
    "- threat: a risk to the topic's subject (security incident, fraud, outage, scam, "
    "impersonation, disinformation, reputational attack);\n"
    "- related: on-topic but routine (markets, products, news, opinion);\n"
    "- off_topic: the topic is only mentioned in passing.\n"
    "Give relevance_reason as one short clause (max 15 words).\n"
    + "Then rate severity: how harmful this discussion is to the search topic's subject "
    "(the company, person, product, event, policy or thing searched) and the people "
    "connected to it (customers, users, voters, patients, followers, the public), judged "
    "by what the content DOES, not by whether it is true:\n"
    "- critical: active harm: leaked, sold or exposed card, account or personal data; "
    "fraud or scam campaigns; impersonation of the subject; doxxing or credible threats of "
    "violence; credible attack or breach claims;\n"
    "- high: content that spreads alarm likely to cause panic, financial loss or harm to "
    "health or safety (rumours, attack, lockdown or contamination claims, calls to "
    "withdraw money (bank-run behaviour), stop a treatment or stay away from a vote), or "
    "misleading content pushed by coordinated accounts (many different accounts posting "
    "the same words);\n"
    "- medium: plain news reporting of an incident (an outage, a recall, a fine, a "
    "lawsuit, a scandal), criticism, complaints, controversy or allegations that could "
    "hurt the subject's reputation;\n"
    "- low: routine news, market chatter, product talk, opinion, neutral mentions.\n"
    "Rules: satire, jokes and mockery of a claim are low (medium only if they repeat the "
    "claim to a wide audience); they do not spread alarm. Personal appeals (donations, "
    "fundraising, prayers, messages of support) are low unless they impersonate the "
    "subject or solicit money fraudulently. Complaints about service, prices or support "
    "are medium at most unless they reveal harm (exposed data, fraud, stolen money). The "
    "subject's own official accounts (for example a company's verified customer support "
    "account replying to customers, or a campaign's own account) are not impersonation and "
    "not a lure; many posts from ONE account are not coordination. One ordinary "
    "advertiser's normal ad is low unless the ad itself impersonates the subject or is a "
    "scam. If the posts do not actually discuss the subject, give severity low. Judge "
    "severity relative to the search topic, not to the world in general. If relevance is "
    "off_topic, give severity low. Give severity_reason as one short clause (max 15 words) "
    "naming what the posts do; never use the words 'true', 'false', 'unconfirmed' or "
    "'unverified'."
)

OUTLIER_PROMPT = (
    "Finally, list in outliers the numbers [n] of any shown posts that do NOT fit your "
    "label: a different subject or purpose from the rest (for example a product review or "
    "a news story grouped with posts selling stolen data, or a recipe grouped with recall "
    "warnings). Label and rate what the posts that fit have in common, not the outliers. "
    "Leave outliers empty when all posts fit or when most of them do not share one theme."
)

PLATFORM_NAMES = {"x": "X", "youtube": "YouTube", "telegram": "Telegram",
                  "linkedin": "LinkedIn", "linkedin_ads": "LinkedIn Ads"}


def _account_key(p: Post) -> str:
    return f"{p.source}:{(p.author_id or p.author_handle or p.author_name or p.post_id)}".lower()


def _post_header(p: Post) -> str:
    """'(X, @BofA_Help, post)' / '(LinkedIn Ads, ad, advertiser: 212 Dental Care)': who
    posted, so the labeller can tell an official account or a lone advertiser from a
    campaign of many accounts."""
    parts = [PLATFORM_NAMES.get(p.source, p.source)]
    handle = (p.author_handle or "").strip()
    if handle:
        parts.append(handle if handle.startswith("@") else f"@{handle}")
    elif p.author_name and p.kind != "ad":
        parts.append(re.sub(r"\s+", " ", p.author_name).strip()[:60])
    kind = "ad" if (p.kind == "ad" or p.ad_meta is not None) else p.kind
    parts.append(kind)
    if kind == "ad" and p.author_name:
        parts.append("advertiser: " + re.sub(r"\s+", " ", p.author_name).strip()[:60])
    return "(" + ", ".join(parts) + ")"


def _first_sentence(text: str, limit: int = 240) -> str:
    t = re.sub(r"\s+", " ", _URL_RE.sub("", text or "")).strip()
    parts = [s for s in _SENT_RE.split((text or "").strip()) if s.strip()]
    if parts:
        t = re.sub(r"\s+", " ", _URL_RE.sub("", parts[0])).strip() or t
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


def _keyword_label(keywords: list[str], central: Post) -> tuple[str, str]:
    label = " / ".join(keywords[:4]) if keywords else _clean_label(_first_sentence(central.text))
    summary = f'Representative post: "{_first_sentence(central.text)}"'
    return label, summary


@dataclass
class _Label:
    label: str
    summary: str
    relevance: str | None = None
    relevance_reason: str = ""
    severity: str | None = None
    severity_reason: str = ""
    outliers: tuple[int, ...] = ()  # 0-based positions in the texts the model was shown


def _llm_label(llm: LLM, topic: str, texts: list[str], nid: str,
               headers: list[str] | None = None, n_posts: int | None = None,
               n_accounts: int | None = None) -> _Label:
    flat = [re.sub(r"\s+", " ", t).strip()[:LLM_TEXT_CHARS] for t in texts]
    heads = list(headers or [])
    numbered = "\n\n".join(
        f"[{i + 1}] {heads[i] + ' ' if i < len(heads) and heads[i] else ''}{t}"
        for i, t in enumerate(flat))
    total = n_posts if n_posts is not None else len(texts)
    who = (f" from {n_accounts} distinct account{'s' if n_accounts != 1 else ''}"
           if n_accounts is not None else "")
    shown = f" ({len(texts)} shown)" if total != len(texts) else ""
    prompt = (
        f"Search topic (do not add facts from it; use it only to rate relevance and severity): {topic}\n\n"
        f"These {total} posts{who} were grouped together by similarity{shown}. Each is "
        "prefixed with (platform, @account, kind) and, for ads, the advertiser:\n\n"
        f"{numbered}\n\n"
        f"Give a short label (a complete phrase, max {LABEL_MAX_WORDS} words and "
        f"{LABEL_TARGET_CHARS} characters) and one "
        "sentence stating what these posts claim.\n" + RELEVANCE_PROMPT + "\n"
        + OUTLIER_PROMPT
    )
    out = llm.parse(NarrativeLabel, prompt, system=LLM_SYSTEM, purpose=f"narrative_label:{nid}", max_tokens=2048)
    label = _clean_label(out.label)
    summary = re.sub(r"\s+", " ", out.claim_summary).strip()
    if not label or not summary:
        raise LLMError("empty label or summary")
    rel_reason_raw = getattr(out, "relevance_reason", None)
    relevance, reason = _clean_relevance(getattr(out, "relevance", None), rel_reason_raw)
    severity, sev_reason = _clean_severity(getattr(out, "severity", None),
                                           getattr(out, "severity_reason", None), relevance,
                                           rel_reason_raw)
    if severity == "high" and (_describes_mockery(label, summary) or _describes_dismissal(label)) \
            and not _describes_harm(
            label, summary, sev_reason, reason):
        severity = "medium"
    return _Label(label, summary, relevance, reason, severity, sev_reason,
                  _clean_outliers(getattr(out, "outliers", None), len(texts)))


@dataclass
class _LabelJob:
    nid: str
    texts: list[str]
    headers: list[str] | None = None  # "(platform, @account, kind)" per text
    n_posts: int | None = None  # narrative size (texts are a sample of it)
    n_accounts: int | None = None  # distinct accounts among ALL members


# Outcome per job: a _Label on success, or None plus a reason
# ("failed" | "budget") meaning the caller must use the keyword label.
_LabelResult = tuple[_Label | None, str]


def _label_all(llm: LLM, topic: str, jobs: list[_LabelJob],
               budget: float | None = None, workers: int | None = None) -> list[_LabelResult]:
    """LLM-label every job concurrently; results are returned in job order, so output
    is identical to sequential labelling whatever order the calls complete in.

    Jobs must be ordered biggest narrative first: the executor starts them in submission
    order, so when LABEL_TIME_BUDGET_S runs out it is the smallest that are skipped.
    `budget` overrides LABEL_TIME_BUDGET_S (narratives split off by pruning get what is
    left of the run's labelling budget)."""
    if not jobs:
        return []
    t0 = _clock()
    budget = LABEL_TIME_BUDGET_S if budget is None else budget

    def one(job: _LabelJob) -> _LabelResult:
        try:
            return _llm_label(llm, topic, job.texts, job.nid, job.headers, job.n_posts,
                              job.n_accounts), "llm"
        except Exception as e:  # noqa: BLE001 - any failure: keyword label for this one
            # LLM.parse wraps only API errors in LLMError; a validation error or an
            # unexpected response must not discard every other narrative.
            log.warning("narrative %s: LLM label failed (%s: %s); using keywords",
                        job.nid, type(e).__name__, redact(str(e))[:300])
            return None, "failed"

    # Dispatch from this thread, in job order, only when a worker is free, and check the
    # budget just before each submission. (Checking inside the workers raced: a worker
    # that dequeued a bigger narrative could read the clock after one that dequeued a
    # smaller one, so a bigger narrative was skipped.) Skipped jobs are always a suffix.
    results: list[_LabelResult] = [(None, "budget")] * len(jobs)
    workers = max(1, min(workers or LABEL_WORKERS, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="narr-label") as ex:
        futs: dict[Future[_LabelResult], int] = {}
        running: set[Future[_LabelResult]] = set()
        for i, job in enumerate(jobs):
            if len(running) >= workers:
                _, running = wait(running, return_when=FIRST_COMPLETED)
            if _clock() - t0 >= budget:
                break
            f = ex.submit(one, job)
            futs[f] = i
            running.add(f)
        for f, i in futs.items():
            results[i] = f.result()
    n_budget = sum(1 for _, why in results if why == "budget")
    n_failed = sum(1 for _, why in results if why == "failed")
    log.info("narratives: %d labelled by LLM in %.1f s (%d workers); %d keyword-labelled "
             "after LLM failure; %d keyword-labelled because the %.0f s label time budget "
             "was spent", len(results) - n_budget - n_failed, _clock() - t0, workers,
             n_failed, n_budget, budget)
    if n_budget:
        skipped = [j.nid for j, (_, why) in zip(jobs, results, strict=True) if why == "budget"]
        log.warning("narratives: label time budget spent; keyword labels for %d smallest "
                    "narratives (%s)", n_budget, ", ".join(skipped[:20]) + (" ..." if len(skipped) > 20 else ""))
    return results


def _label_eligible(llm: LLM, topic: str, jobs: list[_LabelJob],
                    budget: float | None = None,
                    workers: int | None = None) -> list[_LabelResult]:
    """_label_all for the jobs with >= LLM_LABEL_MIN_POSTS posts; the rest get
    (None, "small") without a call. Results stay in job order."""
    def big(j: _LabelJob) -> bool:
        return (j.n_posts or len(j.texts)) >= LLM_LABEL_MIN_POSTS

    got = iter(_label_all(llm, topic, [j for j in jobs if big(j)], budget=budget, workers=workers))
    out = [next(got) if big(j) else (None, "small") for j in jobs]
    n_small = sum(1 for _, why in out if why == "small")
    if n_small:
        log.info("narratives: %d narrative(s) under %d posts keyword-labelled without an "
                 "LLM call", n_small, LLM_LABEL_MIN_POSTS)
    return out


# ------------------------------------------------------------------------------- main


def _ts_key(p: Post) -> tuple[str, str]:
    dt = parse_ts(p.published_at) if p.published_at else None
    return (iso(dt) if dt else "9999", p.post_id)


def _window_start(value: str | date | datetime | None) -> datetime | None:
    """Accept RunMeta.start ('YYYY-MM-DD'), an ISO timestamp, a date or a datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return day_start(value)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        return day_start(date.fromisoformat(value.strip()))
    return parse_ts(value)


def _counted_in_timeline(p: Post, window_start: datetime | None) -> bool:
    """Whether an analysable member may enter volume_by_day / volume_by_platform /
    first_seen_at.

    Ads are analysable when their RUN overlaps the window, but their published_at is the
    run START, which can precede the window. Bucketing it would put a day outside the
    analyst's range on the timeline and could make a pre-window date the narrative's
    "first observed". So an ad counts only when its start date is known to be inside the
    window; when the window start is not supplied, ads are not counted at all (they stay
    members and are reported through unverified_member_count)."""
    if p.ad_meta is None and p.kind != "ad":
        return True
    if window_start is None:
        return False
    dt = parse_ts(p.published_at)
    return dt is not None and dt >= window_start


def build_narratives(
    posts: list[Post],
    *,
    topic: str,
    llm: LLM | None,
    window_start: str | date | datetime | None = None,
    stats: dict[str, int] | None = None,
    trace: dict[str, dict] | None = None,
    label_budget_s: float | None = None,
    workers: int | None = None,
) -> list[Narrative]:
    """Cluster posts into narratives.

    label_budget_s: wall-clock cap for all LLM label calls (None = LABEL_TIME_BUDGET_S;
    math.inf = no cap, used when the run has no time limit).

    stats: when given, filled with pruning totals (posts_pruned, posts_pruned_unassigned,
    narratives_split, narratives_mixed and label counts); the pipeline records them in
    run.json counts.

    trace: when given, filled with TRACE_KEYS (per-narrative pruned post IDs, split
    targets, mixed narratives); the pipeline records them in run.json options.

    window_start: the run's start (ISO string or datetime). Optional so existing callers
    keep working; without it, ad members are left out of volume and first-seen (see
    _counted_in_timeline)."""
    ws = _window_start(window_start)
    if stats is not None:
        stats.update(dict.fromkeys(STATS_KEYS, 0))
    if trace is not None:
        trace.update({k: {} for k in TRACE_KEYS})
    pruned_ids: dict[str, list[str]] = {}
    split_into: dict[str, str] = {}
    mixed: dict[str, dict] = {}
    # Candidates: one per post_id, not marked duplicate, with enough text to cluster.
    seen: set[str] = set()
    cand: list[Post] = []
    for p in posts:
        if p.duplicate_of or p.post_id in seen or not _non_trivial(p.text):
            continue
        seen.add(p.post_id)
        cand.append(p)
    if len(cand) < 2:
        return []

    vecs, method = embed_texts([p.text for p in cand])
    log.info("narratives: %d candidate posts embedded with %s", len(cand), method)
    labels = _cluster(vecs, method)

    groups: dict[int, list[int]] = {}
    for i, lab in enumerate(labels.tolist()):
        if lab >= 0:
            groups.setdefault(lab, []).append(i)
    if not groups:
        return []

    stop = _stopwords(topic)
    noise_docs = [cand[i].text for i, lab in enumerate(labels.tolist()) if lab < 0]
    keywords = _ctfidf_keywords(
        {k: [cand[i].text for i in idx] for k, idx in groups.items()}, noise_docs, stop
    )

    # Deterministic order: size desc, then earliest member post_id.
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), min(cand[i].post_id for i in kv[1])))
    analysable_ids = {p.post_id for p in analysable(cand)}

    picked = [_pick_examples(idx, vecs, cand) for _, idx in ordered]
    llm_results: list[_LabelResult] = []
    shown: list[list[int]] = []  # per narrative: member indices the labeller saw, in order
    n_pruned = n_unassigned = 0

    def make_job(nid: str, ex_idx: list[int], central_order: np.ndarray) -> _LabelJob:
        texts_idx = ex_idx + [int(i) for i in central_order if int(i) not in ex_idx]
        shown.append(texts_idx[:MAX_LLM_TEXTS])
        sample = [cand[i] for i in shown[-1]]
        members = [cand[int(i)] for i in central_order]
        return _LabelJob(nid, [p.text for p in sample], [_post_header(p) for p in sample],
                         len(members), len({_account_key(p) for p in members}))

    splits: list[tuple[int, list[int]]] = []  # (origin narrative number, members)
    if llm is not None:
        t_label = _clock()
        jobs = [make_job(f"N{n}", ex_idx, central_order)
                for n, (ex_idx, central_order) in enumerate(picked, start=1)]
        label_cap = LABEL_TIME_BUDGET_S if label_budget_s is None else label_budget_s
        llm_results = _label_eligible(llm, topic, jobs, budget=label_cap, workers=workers)

        # Prune members that do not fit the label (module docstring). Narrative IDs keep
        # the labelling order so they match the llm_log purposes. A coherent removed set
        # of at least the minimum narrative size becomes its own narrative (labelled
        # after the others, within what is left of the label budget); the rest of the
        # removed posts become unassigned, like clustering noise.
        space = _space(method)
        margin = PRUNE_MARGIN[space]
        min_size = _split_min_size(len(cand))

        def flagged_ids(n: int, outliers: tuple[int, ...]) -> list[str]:
            return [cand[shown[n][k]].post_id for k in outliers if k < len(shown[n])]

        for n, ((lab, idx), (got, _)) in enumerate(zip(ordered, llm_results, strict=True)):
            if got is None or not got.outliers:
                continue
            k_flag, m_shown = len(got.outliers), len(shown[n])
            if m_shown and k_flag / m_shown >= PRUNE_MAX_FLAGGED_SHARE:
                mixed[f"N{n + 1}"] = {"flagged": k_flag, "shown": m_shown, "reason": "share",
                                      "flagged_post_ids": flagged_ids(n, got.outliers)}
                log.info("narrative N%d: labeller flagged %d of %d shown posts as not "
                         "fitting; kept whole as a mixed narrative", n + 1, k_flag, m_shown)
                continue
            kept, removed = _prune_outliers(idx, shown[n], got.outliers, vecs, margin)
            if not removed:
                continue
            if len(kept) < LLM_LABEL_MIN_POSTS:
                # pruning would leave a narrative smaller than any the run labels
                mixed[f"N{n + 1}"] = {"flagged": k_flag, "shown": m_shown, "reason": "size",
                                      "flagged_post_ids": flagged_ids(n, got.outliers)}
                log.info("narrative N%d: pruning would leave %d member(s); kept whole as a "
                         "mixed narrative", n + 1, len(kept))
                continue
            ordered[n] = (lab, kept)
            picked[n] = _pick_examples(kept, vecs, cand)
            groups[lab] = kept
            split = _coherent_group(removed, vecs, space) if len(removed) >= min_size else []
            split = split if len(split) >= min_size else []
            left = [i for i in removed if i not in set(split)]
            n_pruned += len(removed)
            n_unassigned += len(left)
            pruned_ids[f"N{n + 1}"] = [cand[i].post_id for i in removed]
            if split:
                splits.append((n + 1, split))
            log.info("narrative N%d: %d of %d member(s) do not fit the label (labeller "
                     "flagged %d of %d shown); %d split into a new narrative, %d left "
                     "unassigned: %s", n + 1, len(removed), len(idx), len(got.outliers),
                     len(shown[n]), len(split), len(left),
                     ", ".join(cand[i].post_id for i in removed[:20])
                     + (" ..." if len(removed) > 20 else ""))
        if splits:
            next_lab = max(groups) + 1
            new_jobs = []
            for j, (origin_n, sidx) in enumerate(splits):
                lab = next_lab + j
                groups[lab] = sidx
                ordered.append((lab, sidx))
                picked.append(_pick_examples(sidx, vecs, cand))
                nid = f"N{len(ordered)}"
                split_into[f"N{origin_n}"] = nid
                new_jobs.append(make_job(nid, *picked[-1]))
                log.info("narrative %s: split from N%d (%d members: %s)", nid, origin_n,
                         len(sidx), ", ".join(cand[i].post_id for i in sidx[:20])
                         + (" ..." if len(sidx) > 20 else ""))
            # Split narratives are labelled only within the run's label budget; their own
            # outliers are not pruned again (one pruning pass).
            left_s = max(0.0, label_cap - (_clock() - t_label))
            llm_results += _label_eligible(llm, topic, new_jobs, budget=left_s, workers=workers)
        if n_pruned:  # keywords from the pruned groups; removed posts join the background
            assigned = {i for idx in groups.values() for i in idx}
            keywords = _ctfidf_keywords(
                {k: [cand[i].text for i in idx] for k, idx in groups.items()},
                [p.text for i, p in enumerate(cand) if i not in assigned], stop)
    if stats is not None:
        stats.update({"posts_pruned": n_pruned, "posts_pruned_unassigned": n_unassigned,
                      "narratives_split": len(splits), "narratives_mixed": len(mixed),
                      "narratives_label_small": sum(1 for _, why in llm_results
                                                    if why == "small")})

    out: list[Narrative] = []
    for n, (lab, idx) in enumerate(ordered, start=1):
        nid = f"N{n}"
        members = [cand[i] for i in idx]
        ex_idx, central_order = picked[n - 1]
        central = cand[int(central_order[0])]

        # Volume and first-seen: verified, in-window members only (evidence rule); ads
        # whose run began before the window are excluded (their date is pre-window).
        counted = [
            p for p in members if p.post_id in analysable_ids and _counted_in_timeline(p, ws)
        ]
        by_day: Counter = Counter()
        by_platform: Counter = Counter()
        first_dt = None
        for p in counted:
            dt = parse_ts(p.published_at)
            if dt is None:  # analysable() implies parseable; guard anyway
                continue
            by_day[dt.strftime("%Y-%m-%d")] += 1
            by_platform[p.source] += 1
            first_dt = dt if first_dt is None or dt < first_dt else first_dt

        kw = keywords.get(lab, [])
        got = llm_results[n - 1][0] if llm_results else None
        # Schema allows only "llm" | "keyword": budget-skipped narratives are "keyword"
        # (the count is in the log line from _label_all). Keyword labels carry no
        # relevance (None = not assessed).
        if got is not None:
            label, summary, method_used = got.label, got.summary, "llm"
            relevance, relevance_reason = got.relevance, got.relevance_reason
            severity, severity_reason = got.severity, got.severity_reason
            if severity in ("critical", "high") and _single_advertiser_ads(members) and \
                    not _describes_fraud(label, summary, severity_reason, relevance_reason):
                log.info("narrative %s: %s -> medium (ads from one advertiser, no "
                         "impersonation/scam/fraud/phishing described)", nid, severity)
                severity = "medium"
        else:
            (label, summary), method_used = _keyword_label(kw, central), "keyword"
            relevance, relevance_reason = None, ""
            severity, severity_reason = None, ""

        out.append(
            Narrative(
                id=nid,
                label=label,
                claim_summary=summary,
                member_post_ids=[p.post_id for p in sorted(members, key=_ts_key)],
                example_post_ids=[cand[i].post_id for i in ex_idx],
                volume_by_day=dict(sorted(by_day.items())),
                volume_by_platform=dict(sorted(by_platform.items(), key=lambda kv: (-kv[1], kv[0]))),
                keywords=kw,
                label_method=method_used,
                first_seen_at=iso(first_dt) if first_dt else None,
                unverified_member_count=len(members) - len(counted),
                relevance=relevance,
                relevance_reason=relevance_reason,
                severity=severity,
                severity_reason=severity_reason,
            )
        )
    if trace is not None:
        trace.update({"pruned_post_ids": pruned_ids, "pruned_split_into": split_into,
                      "narratives_mixed": mixed})
    if stats is not None:
        n_llm = sum(1 for n in out if n.label_method == "llm")
        stats.update({"narratives_label_llm": n_llm,
                      "narratives_label_keyword": len(out) - n_llm})
    return out
