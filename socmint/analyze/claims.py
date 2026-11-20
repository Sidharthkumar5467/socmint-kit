"""Misinformation assessment (docs/SPEC.md §6; CLAUDE.md §3).

extract atomic claims -> classify each related post's stance -> gather external
evidence -> assign a verdict with cited evidence, a confidence and a rationale.

Evidence rules encoded in this module:
- There is NO automated true/false. Verdicts are only
  `supported | contradicted | disputed | unverified`.
- Stance (assert / report / question / debunk) describes how a collected post presents
  a claim. Stance counts are context, never evidence: a post that reports or debunks a
  claim is never counted as support, and collected posts are never cited as evidence.
- The LLM may only cite evidence the pipeline actually retrieved. Any URL it cites that
  is not in the retrieved list is dropped; if no valid citation remains, the verdict is
  forced to `unverified`. A verdict must also be consistent with the per-URL relations
  the model gave (e.g. `supported` needs at least one cited `supports` item).
- `disputed` means the retrieved sources conflict (a contradicting item, a mixed
  fact-check rating, or supporting items the model says disagree); support plus
  context is `unverified`. Debunking posts never make a claim disputed.
- Confidence is capped by how much evidence there is (thin evidence -> low confidence).
- Evidence dates (asymmetric rule, enforced in code after the LLM and in the rule-based
  path): an item may count as `supports` only when its own publication date is known
  and falls in [window_start - SUPPORT_LOOKBACK_DAYS, window_end + SUPPORT_LOOKAHEAD_DAYS];
  older or later items describe a different event and are forced to `context`. An item
  with an unknown date may support only alongside an in-window dated supporting item.
  A `contradicts` item may be older (recurring hoaxes are debunked again), but the
  rationale then names its date. Found in testing: docs/EVALUATION.md (2024 articles
  about a different glitch rated as support for a 2026 claim).
- Relevance (code-enforced): an evidence item that shares none of the claim's salient
  terms is flagged `unrelated` (a search miss) and can never count as support or
  contradiction. Items dated before the support range are flagged `old_event`.
- Output split: `rationale` is analysis only; pipeline notes (method, fallbacks,
  overrides, providers, excluded posts) go to `method_notes`. `bottom_line` is one
  sentence an analyst reads first, written by the verdict LLM call (no extra call) or
  by rule, and replaced by the rule sentence whenever code overrode the model.
- Every LLM prompt/response is logged by socmint.llm to llm_log.jsonl; every evidence
  query and raw response is logged by socmint.analyze.evidence.
- Without an LLM everything degrades to keyword heuristics, labelled `method=heuristic`.
"""

from __future__ import annotations

import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal, NamedTuple
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from socmint.analyze import evidence as evidence_mod
from socmint.schema import STANCES, Claim, Evidence, Narrative, Post
from socmint.util.log import get_logger

log = get_logger(__name__)

MAX_NARRATIVES = 10  # top narratives (by rank) that get LLM claim extraction + stance
MAX_CLAIMS_PER_NARRATIVE = 3
EXTRACT_POSTS = 15
MAX_STANCE_POSTS = 40
STANCE_BATCH = 15
LLM_WORKERS = 8  # LLM calls only; evidence providers have one worker each (evidence.py)
EVIDENCE_BUDGET_S = 240.0  # wall-clock cap for all evidence lookups in one run
SUPPORT_LOOKBACK_DAYS = 30  # supporting evidence may predate the window by this much
SUPPORT_LOOKAHEAD_DAYS = 7  # ... or follow its end by this much
HEURISTIC_CONFIDENCE = 0.5
THIN_HEURISTIC_CONFIDENCE = 0.3

Verdict = Literal["supported", "contradicted", "disputed", "unverified"]


# ------------------------------------------------------------------- LLM schemas


class _ClaimText(BaseModel):
    text: str = Field(description="One atomic, checkable factual claim as asserted by the "
                                  "posts, at most 30 words, in the posts' original language "
                                  "or English.")


class ExtractedClaims(BaseModel):
    claims: list[_ClaimText]


class _StanceItem(BaseModel):
    post_id: str
    stance: Literal["assert", "report", "question", "debunk"]
    rationale: str = Field(description="At most 20 words.")


class StanceBatch(BaseModel):
    items: list[_StanceItem]


class _Cited(BaseModel):
    url: str
    relation: Literal["supports", "contradicts", "context"]


BOTTOM_LINE_MAX = 220


class VerdictOut(BaseModel):
    verdict: Verdict
    confidence: float = Field(description="0 to 1")
    rationale: str
    bottom_line: str = Field(
        description=f"ONE plain sentence, at most {BOTTOM_LINE_MAX} characters, saying what "
                    "the cited evidence shows about the claim (which outlets confirm or "
                    "contradict which part, and what nothing confirms). Mention only "
                    "what the numbered evidence list contains. No URLs, no [n] markers.")
    cited: list[_Cited] = Field(description="Only URLs from the numbered evidence list.")
    conflicting_items: list[int] = Field(
        default_factory=list,
        description="Only for verdict 'disputed' when no cited item contradicts the claim: "
                    "the [n] numbers of the supporting items that disagree with each "
                    "other about it (at least two). Empty otherwise.")


# --------------------------------------------------------------- small utilities


def _llm_name(llm: Any) -> str:
    return getattr(llm, "name", None) or f"llm:{getattr(llm, 'model', 'unknown')}"


_REP_WRAPPER = re.compile(r'^\s*Representative post:\s*["“]?(.*?)["”]?\s*$', re.S)


def heuristic_claim_text(narrative: Narrative) -> str:
    """The narrative's claim_summary with the 'Representative post: "..."' wrapper removed."""
    s = (narrative.claim_summary or "").strip()
    m = _REP_WRAPPER.match(s)
    if m:
        s = m.group(1).strip()
    return s or narrative.label


def _empty_stances() -> tuple[dict[str, int], dict[str, list[str]]]:
    return {s: 0 for s in STANCES}, {s: [] for s in STANCES}


def _stance_summary(counts: dict[str, int]) -> str:
    return ", ".join(f"{s}={counts.get(s, 0)}" for s in STANCES)


# ------------------------------------------------------------- heuristic stance

# Multilingual-light keyword rules. Order matters: debunk > question > report > assert.
# Latin-script terms use word boundaries; other scripts use plain substring matching.
_DEBUNK_TERMS = [
    "fake", "false", "hoax", "debunk", "debunked", "debunking", "misleading", "not true",
    "fact check", "fact-check", "factcheck", "no evidence", "misinformation",
    "disinformation", "fabricated", "untrue",
    # fr / es / pt / de / it
    "faux", "intox", "falso", "falsa", "bulo", "desinformación", "falsch", "fälschung",
    "notizia falsa",
]
_DEBUNK_NONLATIN = ["كاذب", "كاذبة", "مضلل", "مضللة", "شائعة", "إشاعة", "مفبرك", "غير صحيح",
                    "фейк", "ложь", "неправда"]
_QUESTION_TERMS = ["is it true", "really", "can anyone confirm", "is this real", "c'est vrai",
                   "es verdad", "stimmt das"]
_QUESTION_NONLATIN = ["؟", "¿", "هل صحيح", "правда ли"]
_REPORT_TERMS = ["reportedly", "according to", "claims that", "claim that", "alleged",
                 "allegedly", "says", "said", "selon", "según", "segundo", "laut",
                 "angeblich", "presunto", "supuestamente"]
_REPORT_NONLATIN = ["بحسب", "وفقا", "وفقاً", "حسب", "يزعم", "مزاعم", "по данным", "якобы"]


def _rx(terms: list[str]) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)",
                      re.IGNORECASE)


_DEBUNK_RX = _rx(_DEBUNK_TERMS)
_QUESTION_RX = _rx(_QUESTION_TERMS)
_REPORT_RX = _rx(_REPORT_TERMS)


def _has_any(text: str, rx: re.Pattern[str], nonlatin: list[str]) -> str | None:
    m = rx.search(text)
    if m:
        return m.group(0)
    for t in nonlatin:
        if t in text:
            return t
    return None


_URL_RX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def heuristic_stance(text: str) -> tuple[str, str]:
    """Keyword stance for one post: returns (stance, rationale).

    Note: this classifies the post's framing, not its relation to a specific claim; a
    post saying "the denial is fake" is labelled debunk. Recorded as a limitation.
    """
    t = _URL_RX.sub(" ", text or "")  # a "?" inside a URL is not a question
    hit = _has_any(t, _DEBUNK_RX, _DEBUNK_NONLATIN)
    if hit:
        return "debunk", f'keyword "{hit}"'
    if "?" in t:
        return "question", 'contains "?"'
    hit = _has_any(t, _QUESTION_RX, _QUESTION_NONLATIN)
    if hit:
        return "question", f'keyword "{hit}"'
    hit = _has_any(t, _REPORT_RX, _REPORT_NONLATIN)
    if hit:
        return "report", f'keyword "{hit}"'
    return "assert", "no debunk/question/report cue"


# -------------------------------------------------------------- rating mapping

_RATING_DISPUTED = _rx(["mixed", "mixture", "half true", "half-true", "half", "partly",
                        "partially", "partly false", "partly true", "unproven",
                        "missing context", "needs context", "lacks context",
                        "unsupported", "exaggerated", "engañoso", "trompeur", "impreciso",
                        "teilweise"])
_RATING_CONTRA = _rx(["false", "fake", "pants on fire", "incorrect", "inaccurate",
                      "misleading", "not true", "untrue", "wrong", "hoax", "fabricated",
                      "debunked", "scam", "satire", "faux", "falso", "falsa", "falsch",
                      "bulo", "mostly false", "four pinocchios"])
_RATING_SUPPORT = _rx(["true", "correct", "accurate", "mostly true", "verified",
                       "vrai", "verdadero", "verdadeiro", "richtig", "wahr"])
_RATING_AR_CONTRA = ["زائف", "كاذب", "مضلل", "غير صحيح", "مفبرك", "غير دقيق", "خاطئ"]
_RATING_AR_SUPPORT = ["صحيح"]

# Negation handling. A negated support word ("This isn't true.", "Not correct",
# "Pas vrai", "No es cierto") must NEVER map to "supports": a rating that denies a
# claim is a debunk, and debunks never count as support (CLAUDE.md §3).
_NEG = (r"(?:not|isn['’]?t|is\s+not|aren['’]?t|wasn['’]?t|no|never|pas|n['’]est\s+pas|"
        r"no\s+es|nicht|ist\s+nicht|não|nao|non|не)")
_SUPPORT_WORDS = (r"(?:true|correct|accurate|right|verified|confirmed|proven|vrai|vraie|"
                  r"exact|exacte|verdad|cierto|cierta|correcto|correcta|verdadero|verdadera|"
                  r"verdadeiro|verdadeira|verdade|richtig|wahr|vero|правда|верно)")
_HEDGE = r"(?:entirely|quite|totally|completely|necessarily|exactly|fully|wholly|all)"
# "Not entirely true" / "Not quite accurate" -> disputed (partly right, partly wrong).
_RATING_NEG_HEDGED = re.compile(rf"(?<!\w){_NEG}\W+{_HEDGE}\W+(?:\w+\W+)?{_SUPPORT_WORDS}(?!\w)",
                                re.IGNORECASE)
# "Not verified" / "not confirmed" / "not proven" -> disputed (same as "unproven"):
# the checker could not establish the claim, which is not a contradiction either.
_RATING_NEG_UNPROVEN = re.compile(
    rf"(?<!\w){_NEG}\W+(?:\w+\W+)?(?:verified|confirmed|proven)(?!\w)|"
    r"(?<!\w)(?:unverified|unconfirmed|unsubstantiated)(?!\w)", re.IGNORECASE)
# Negation followed (within one optional word) by a support word -> contradicts.
_RATING_NEG_SUPPORT = re.compile(rf"(?<!\w){_NEG}\W+(?:\w+\W+)?{_SUPPORT_WORDS}(?!\w)",
                                 re.IGNORECASE)
# Negated contradiction words ("Not false") are ambiguous -> not mappable.
_RATING_NEG_CONTRA = re.compile(rf"(?<!\w){_NEG}\W+(?:\w+\W+)?(?:false|fake|wrong)(?!\w)",
                                re.IGNORECASE)
# Arabic negators (ليس / ليست / غير / لا) in front of a support word.
_RATING_AR_NEG = re.compile(r"(?:ليس|ليست|غير|لا)\s+(?:\S+\s+)?(?:صحيح|دقيق|حقيقي)")
_RATING_RU_NEG = ["неправда", "неверно"]


def map_rating(rating: str) -> Literal["supports", "contradicts", "disputed"] | None:
    """Map a fact-checker's textualRating to a relation. None = not mappable.

    Order (most specific first; "supports" is only reachable when no negation and no
    contradicting or disputed cue is present):
      1. disputed cues ("partly false", "missing context") and hedged negations
         ("not entirely true") and unproven negations ("not verified");
      2. negated contradictions ("not false") -> None (ambiguous, never support);
      3. negated support words ("isn't true", "not correct", "pas vrai", Arabic
         "ليس صحيحا") and contradicting cues ("false", "incorrect") -> contradicts;
      4. support cues ("true", "correct", "accurate").
    """
    r = (rating or "").strip()
    if not r:
        return None
    if (_RATING_DISPUTED.search(r) or _RATING_NEG_HEDGED.search(r)
            or _RATING_NEG_UNPROVEN.search(r)):
        return "disputed"
    if _RATING_NEG_CONTRA.search(r):
        return None
    if (_RATING_NEG_SUPPORT.search(r) or _RATING_AR_NEG.search(r)
            or any(t in r.lower() for t in _RATING_RU_NEG)):
        return "contradicts"
    if _RATING_CONTRA.search(r) or any(t in r for t in _RATING_AR_CONTRA):
        return "contradicts"
    if _RATING_SUPPORT.search(r) or any(t in r for t in _RATING_AR_SUPPORT):
        return "supports"
    return None


def _rating_of(e: Evidence) -> str:
    m = re.match(r"rating:\s*([^;]*)", e.note or "")
    if not m:
        return ""
    v = m.group(1).strip()
    return "" if v == "(none given)" else v


# -------------------------------------------------------------- step 1: extract


def _countable(p: Post) -> bool:
    """Same gate as verify.analysable(): only these posts enter counts (CLAUDE.md §3)."""
    return bool(p.timestamp_verified and p.in_window and not p.duplicate_of)


def _member_posts(n: Narrative, by_id: dict[str, Post]) -> list[Post]:
    """Text-bearing, countable narrative members, earliest first.

    Evidence rule: posts without a verified platform timestamp, or published outside
    the window, are kept in the run but excluded from counts. Stance counts are counts,
    so such posts are not classified or counted here; they are reported separately by
    `_excluded_members` and named in the claim rationale.
    """
    ps = [by_id[i] for i in n.member_post_ids if i in by_id]
    ps = [p for p in ps if _countable(p) and (p.text or "").strip()]
    return sorted(ps, key=lambda p: (p.published_at or "9999", p.post_id))


def _excluded_members(n: Narrative, by_id: dict[str, Post]) -> list[str]:
    """Non-duplicate text-bearing members excluded from counts (unverified / out of window)."""
    return [i for i in n.member_post_ids
            if i in by_id and not by_id[i].duplicate_of and not _countable(by_id[i])
            and (by_id[i].text or "").strip()]


# "Post [2] asserts that ...", "Posts [1] and [3] claim ...", "According to post 4, ...":
# the prompt numbers the posts and the model sometimes copies the numbering into the claim.
_POST_REF = r"(?:the\s+)?posts?\s*\[?\d+\]?(?:\s*(?:,|and|&|-|–)\s*\[?\d+\]?)*|\[\d+\]"
_LEAD_REF_RE = re.compile(
    rf"^\s*(?:(?:according to|in|per)\s+(?:{_POST_REF})\s*,?\s*"
    rf"|(?:{_POST_REF})\s*(?:asserts?|claims?|says?|states?|reports?|alleges?|suggests?|"
    r"notes?|mentions?|argues?|writes?|indicates?|shows?)?\s*(?:that\s+)?[:,]?\s*)",
    re.IGNORECASE)
_INLINE_REF_RE = re.compile(rf"\s*\((?:see\s+)?(?:{_POST_REF})\)", re.IGNORECASE)


def clean_claim_text(text: str | None) -> str:
    """A claim as a standalone statement: whitespace collapsed, a leading reference to
    the numbered prompt posts ("Post [2] asserts that ...") and inline "(post [3])"
    removed, first letter capitalised."""
    t = re.sub(r"\s+", " ", text or "").strip()
    t2 = _INLINE_REF_RE.sub("", _LEAD_REF_RE.sub("", t, count=1)).strip()
    if t2 and t2 != t:
        t = t2[:1].upper() + t2[1:]
    return t


def _extract_llm(n: Narrative, members: list[Post], topic: str, llm: Any) -> list[str]:
    lines = [f"[{i + 1}] {p.text.strip()[:600]}" for i, p in enumerate(members[:EXTRACT_POSTS])]
    prompt = (
        f"Investigation topic: {topic}\n"
        f"Narrative label: {n.label}\nNarrative summary: {n.claim_summary}\n\n"
        "Posts in this narrative (original language kept):\n" + "\n".join(lines) + "\n\n"
        f"Extract at most {MAX_CLAIMS_PER_NARRATIVE} atomic, checkable FACTUAL claims that "
        "these posts assert (who did what, when, where, how many). Each claim at most 30 "
        "words, phrased neutrally as the posts assert it, as a standalone statement about "
        "the world: never refer to the posts or their numbers (write 'Bank X offers ...', "
        "not 'Post [2] asserts that Bank X offers ...'). Do not judge truth. Omit "
        "opinions, slogans and calls to action. Return an empty list if there are none."
    )
    out = llm.parse(ExtractedClaims, prompt, system="You extract factual claims for an "
                    "OSINT analyst. You never judge whether a claim is true.",
                    purpose=f"claims.extract:{n.id}", max_tokens=4096)
    texts: list[str] = []
    for c in out.claims:
        t = clean_claim_text(c.text)
        if t and t.lower() not in {x.lower() for x in texts}:
            texts.append(t)
    return texts[:MAX_CLAIMS_PER_NARRATIVE]


# --------------------------------------------------------------- step 2: stance


def _stance_llm_batch(claim_text: str, batch: list[Post], llm: Any,
                      purpose: str) -> dict[str, tuple[str, str]]:
    ids = {p.post_id for p in batch}
    lines = [f"- post_id={p.post_id} | {p.text.strip()[:500]}" for p in batch]
    prompt = (
        f'Claim: "{claim_text}"\n\nFor each post below, classify its stance toward the claim:\n'
        "- assert: presents the claim as true\n"
        "- report: relays that others claim it, without endorsing it\n"
        "- question: asks about or doubts the claim\n"
        "- debunk: says the claim is false or misleading\n"
        "Return one item per post_id exactly as given, with a rationale of at most 20 "
        "words.\n\nPosts:\n" + "\n".join(lines)
    )
    out = llm.parse(StanceBatch, prompt, system="You classify stance for an OSINT analyst.",
                    purpose=purpose, max_tokens=8192)
    res: dict[str, tuple[str, str]] = {}
    for it in out.items:
        # Only accept ids we actually sent; anything else is a model hallucination.
        if it.post_id in ids and it.post_id not in res:
            res[it.post_id] = (it.stance, (it.rationale or "")[:200])
    return res


# ------------------------------------------------------------- step 4: verdicts


def _date_label(e: Evidence) -> str:
    if e.published_at:
        raw = f" (given as '{e.published_raw}')" if e.published_raw else ""
        return f"{e.published_at[:10]}{raw}"
    if e.published_raw:
        return f"unknown (unparsed '{e.published_raw}')"
    return "unknown"


def _evidence_block(ev: list[Evidence]) -> str:
    rows = []
    for i, e in enumerate(ev, 1):
        flag = (" | FLAG: shares no key term with the claim (likely a search miss)"
                if e.flag == "unrelated" else "")
        rows.append(f"[{i}] url={e.url} | source={e.source} | publisher={e.publisher or '?'}"
                    f" | published={_date_label(e)} | title={e.title or '?'} | {e.note}{flag}")
    return "\n".join(rows)


# ------------------------------------------------------------- evidence date rule

Window = tuple[date, date]


def _as_date(v: Any) -> date | None:
    if v is None or v == "":
        return None
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v).strip()[:10])
    except ValueError:
        return None


def make_window(start: Any, end: Any) -> Window | None:
    """(start, end) as dates, or None when either is missing/unparseable."""
    s, e = _as_date(start), _as_date(end)
    if s is None or e is None:
        return None
    return (s, e) if s <= e else (e, s)


def window_from_posts(posts: list[Post]) -> Window | None:
    """Fallback window: the span of countable posts' platform timestamps."""
    days = sorted(d for d in (_as_date(p.published_at) for p in posts if _countable(p)) if d)
    return (days[0], days[-1]) if days else None


def _fmt_window(w: Window) -> str:
    return f"{w[0].isoformat()}..{w[1].isoformat()}"


def _support_range(w: Window) -> tuple[date, date]:
    return (w[0] - timedelta(days=SUPPORT_LOOKBACK_DAYS),
            w[1] + timedelta(days=SUPPORT_LOOKAHEAD_DAYS))


def _outside_window(e: Evidence, w: Window) -> bool:
    """True when the item is dated and its date is outside the claim window itself."""
    d = _as_date(e.published_at)
    return d is not None and not (w[0] <= d <= w[1])


def apply_date_rule(ev: list[Evidence], window: Window | None
                    ) -> tuple[list[Evidence], list[str], list[str]]:
    """Enforce the asymmetric evidence-date rule on already-classified evidence copies.

    Returns (new copies, rationale notes, method notes). `supports` items dated outside
    the support range, or undated without a dated in-range supporting item, become
    `context` (the reason is appended to the item's note; the re-classification is a
    method note). Items dated BEFORE the support range that are not `contradicts` get
    `flag="old_event"` (they describe an earlier event). Out-of-window `contradicts`
    items stay and are named with their date in the rationale notes. Without a window
    nothing is changed and a method note says so.
    """
    if window is None:
        if any(e.supports != "context" for e in ev):
            return ([e.model_copy() for e in ev], [],
                    ["The claim's time window is unknown, so evidence dates were not "
                     "checked."])
        return [e.model_copy() for e in ev], [], []
    lo, hi = _support_range(window)
    notes: list[str] = []
    method: list[str] = []

    def in_range(e: Evidence) -> bool | None:
        d = _as_date(e.published_at)
        return None if d is None else lo <= d <= hi

    dated_support = any(e.supports == "supports" and in_range(e) for e in ev)
    out: list[Evidence] = []
    demoted: list[str] = []
    for e in ev:
        e2 = e.model_copy()
        r = in_range(e)
        if e.supports == "supports":
            if r is False:
                why = f"published {e.published_at[:10]}, outside the claim's time window"  # type: ignore[index]
                e2.supports = "context"
                e2.note = f"{e2.note}; {why}" if e2.note else why
                demoted.append(f"{e.publisher or e.url} ({e.published_at[:10]})")  # type: ignore[index]
            elif r is None and not dated_support:
                why = ("publication date unknown, so it cannot be placed in the claim's "
                       "time window")
                e2.supports = "context"
                e2.note = f"{e2.note}; {why}" if e2.note else why
                demoted.append(f"{e.publisher or e.url} (date unknown)")
        elif e.supports == "contradicts" and _outside_window(e, window):
            notes.append(f"[{e.publisher or e.url} contradicts the claim but was published "
                         f"{e.published_at[:10]}, outside the claim window "  # type: ignore[index]
                         f"{_fmt_window(window)}; it may address a different instance of "
                         "the claim.]")
        d = _as_date(e.published_at)
        if (d is not None and d < lo and e2.supports != "contradicts"
                and e2.flag is None):
            e2.flag = "old_event"
        out.append(e2)
    if demoted:
        method.insert(0, f"Date check: {len(demoted)} item(s) cited as support were "
                         f"re-classified as context because their publication date is not "
                         f"within {lo.isoformat()}..{hi.isoformat()} (claim window "
                         f"{_fmt_window(window)} -{SUPPORT_LOOKBACK_DAYS}/+"
                         f"{SUPPORT_LOOKAHEAD_DAYS} days): {'; '.join(demoted)}.")
    return out, notes, method


# ---------------------------------------------------------- evidence relevance rule

# Function words (en/fr/es/de/pt/it/tr, light) and words every claim/post uses. Removed
# from the claim before matching, so an item is not "related" because both say "the".
_STOP_BY_LANG: dict[str, set[str]] = {
    "en": set("""a an the and or but nor of to in on at by for from with without into onto
        over under about after before during since until than then as is are was were be
        been being am do does did done has have had having it its this that these those
        there their they them he she his her we our you your i me my not no yes so if
        what which who whom whose when where why how all any some more most many much
        each other such only own same too very can could will would shall should may
        might must also just up down out off again further once here both few new""".split()),
    "fr": set("le la les un une des du de et ou mais est sont dans sur pour par avec "
              "sans au aux ce cette ces qui que il elle ils nous vous pas".split()),
    "es": set("el la los las un una unos unas de del y o pero es son en con por para "
              "sin al que se su sus lo como más este esta estos no".split()),
    "de": set("der die das ein eine einen und oder aber ist sind im in mit für von zu "
              "auf den dem des nicht auch sich es wie".split()),
    "pt": set("o a os as um uma de do da dos das e ou mas é são em no na com por para "
              "sem ao que se seu sua não".split()),
    "it": set("il lo la gli le un una di del della e o ma è sono in con per da che "
              "non si".split()),
    "tr": set("ve veya ile bir bu şu için de da mi mı ne olarak gibi daha çok".split()),
}
_STOP = set().union(*_STOP_BY_LANG.values())
_CLAIM_FILLER = set("""post posts posted posting claim claims claimed claiming state states
    stated say says said saying report reports reported reportedly according announce
    announced announcing some one two people users user customer""".split())
_NOTE_SKIP_PREFIXES = ("published", "reviewed:", "claim date:", "rating:", "news search result")
_NOTE_STRIP_PREFIXES = ("snippet:", "reviewed claim:", "claimant:")
_YEAR_RX = re.compile(r"^(?:19|20)\d\d$")
_UNSPACED_SCRIPTS = {"CJK", "HIRAGANA", "KATAKANA", "THAI", "LAO", "KHMER", "MYANMAR",
                     "HANGUL", "TIBETAN"}


def _script(tok: str) -> str:
    for ch in tok:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            return "LATIN" if name.startswith("LATIN") else (name.split(" ")[0] or "OTHER")
    return "DIGIT"


def _raw_tokens(text: str) -> list[str]:
    t = (text or "").replace("’", "'").casefold()
    t = re.sub(r"'s\b", "", t)
    return [w for w in re.findall(r"\w+", t) if w.strip("_")]


def _salient(tokens: list[str], extra_stop: frozenset[str] | set[str] = frozenset()) -> set[str]:
    out = set()
    for w in tokens:
        if w in _STOP or w in extra_stop or _YEAR_RX.match(w):
            continue
        if _script(w) == "DIGIT":
            if len(w) >= 2:
                out.add(w)
        elif len(w) >= 3 or _script(w) != "LATIN":
            out.add(w)
    return out


def _guess_lang(tokens: list[str]) -> str | None:
    hits = {lang: sum(1 for w in tokens if w in words) for lang, words in _STOP_BY_LANG.items()}
    best = max(hits, key=lambda k: hits[k])
    return best if hits[best] else None


def _same_term(a: str, b: str) -> bool:
    """Exact match, or a shared stem (outage/outages, warning/warns, bank/banking)."""
    if a == b:
        return True
    if _script(a) == "DIGIT" or _script(b) == "DIGIT":
        return False
    cp = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        cp += 1
    return cp >= 4 and cp >= min(len(a), len(b)) - 3


def _acronym_hit(acronyms: set[str], words: list[str]) -> bool:
    """True when any acronym (e.g. 'faa') equals the initials of consecutive words."""
    initials = "".join(w[0] for w in words if w)
    return any(len(a) >= 2 and a in initials for a in acronyms)


def _evidence_text(e: Evidence) -> str:
    parts = [e.title or ""]
    for part in (e.note or "").split(";"):
        p = part.strip()
        low = p.casefold()
        if not p or low.startswith(_NOTE_SKIP_PREFIXES):
            continue
        for pref in _NOTE_STRIP_PREFIXES:
            if low.startswith(pref):
                p = p[len(pref):]
                break
        parts.append(p)
    return " ".join(parts)


def _caps(text: str) -> set[str]:
    """Short all-caps tokens (AI, FAA, NSA): kept even when shorter than 3 letters."""
    return {w.casefold() for w in re.findall(r"\b[A-Z]{2,5}\b", text or "")}


def claim_terms(claim_text: str, topic: str = "") -> set[str]:
    """Salient claim terms: claim tokens minus function words, filler and topic words."""
    drop = set(_raw_tokens(topic)) | _CLAIM_FILLER
    return _salient(_raw_tokens(claim_text), drop) | (_caps(claim_text) - drop - _STOP)


def _overlap(a: set[str], b: set[str]) -> bool:
    return any(_same_term(x, y) for x in a for y in b)


def unrelated_flags(claim_text: str, ev: list[Evidence], topic: str = "") -> list[bool]:
    """Per item: True only when it clearly has nothing to do with the claim.

    An item is unrelated when ALL of these hold (conservative; a miss is preferred to a
    wrong flag, since a flag can remove support):
    - it shares no salient claim term (stem match: outage/outages) and no acronym;
    - it does not mention the topic itself (e.g. "banks" for topic "bank of america");
    - it shares no salient term with another retrieved item that does match the claim
      (so "JPMorgan says software bug ..." stays related when a sibling item says
      "Chase blames outages on software bug");
    - it is judgeable: >= 3 salient tokens, mostly in the claim's script, and the same
      guessed language (a translation shares no words yet can be on point). Claims in
      unspaced scripts (CJK, Thai ...) are never judged.
    """
    terms = claim_terms(claim_text, topic)
    if not terms or any(_script(t) in _UNSPACED_SCRIPTS for t in terms):
        return [False] * len(ev)
    topic_terms = _salient(_raw_tokens(topic))
    claim_raw = _raw_tokens(claim_text)
    claim_scripts = {_script(t) for t in terms} - {"DIGIT"}
    lang_c = _guess_lang(claim_raw)
    info = []
    for e in ev:
        txt = _evidence_text(e)
        raw = _raw_tokens(txt)
        sal = _salient(raw, _CLAIM_FILLER) | (_caps(txt) - _STOP)
        hit = (_overlap(terms, sal) or _acronym_hit(_caps(claim_text), raw)
               or _acronym_hit(_caps(txt), claim_raw))
        info.append((raw, sal, hit))
    peers = [sal - topic_terms for _, sal, hit in info if hit]
    out: list[bool] = []
    for raw, sal, hit in info:
        if hit or len(sal) < 3:
            out.append(False)
            continue
        scripts = [_script(t) for t in sal if _script(t) != "DIGIT"]
        if not scripts or sum(1 for x in scripts if x in claim_scripts) < len(scripts) / 2:
            out.append(False)
            continue
        if "LATIN" in claim_scripts and (lang_c is None or lang_c != _guess_lang(raw)):
            out.append(False)
            continue
        own = sal - topic_terms
        out.append(not _overlap(topic_terms, sal)
                   and not any(_overlap(own, p) for p in peers if p is not own))
    return out


def is_unrelated(claim_text: str, e: Evidence, topic: str = "",
                 peers: list[Evidence] | None = None) -> bool:
    """`unrelated_flags` for one item (optionally judged alongside its sibling items)."""
    return unrelated_flags(claim_text, [e, *(peers or [])], topic)[0]


def flag_unrelated(claim_text: str, ev: list[Evidence],
                   topic: str = "") -> tuple[list[Evidence], list[str]]:
    """Copies of `ev` with unrelated items flagged and forced to `context`.

    Returns (copies, method notes). An unrelated item is a search miss: it can never
    count as support (nor as contradiction) whatever a model or rating says.
    """
    out: list[Evidence] = []
    hit: list[str] = []
    for e, bad in zip(ev, unrelated_flags(claim_text, ev, topic), strict=True):
        e2 = e.model_copy()
        if bad:
            e2.flag = "unrelated"
            e2.supports = "context"
            hit.append(e.publisher or e.url)
        out.append(e2)
    notes = ([f"Relevance check: {len(hit)} item(s) share no key term with the claim and "
              f"were treated as unrelated (context only): {'; '.join(hit)}."] if hit else [])
    return out, notes


# ------------------------------------------------------------- key specifics


_MONTHS = set("""january february march april may june july august september october
    november december jan feb mar apr jun jul aug sep sept oct nov dec monday tuesday
    wednesday thursday friday saturday sunday""".split())
# Capitalised words that name a group of people, a quantity or a kind of item rather
# than a specific actor ("Customers of Chase", "Thousands of ...", "Report: ...",
# "Users", "Banks"). They are stripped from the ends of a capitalised span and never
# count as a key specific on their own. Plural/collective forms only: "Bank" (as in
# "Bank of America") is a name part, "Banks" is a category.
_GENERIC_PEOPLE = set("""customers clients users people persons residents citizens consumers
    members employees workers staff officials authorities experts analysts investors
    shareholders holders victims passengers travelers travellers patients voters fans
    viewers readers parents students teachers doctors supporters followers protesters
    opponents critics locals utilisateurs usuarios clientes
    usuários kunden nutzer""".split())
_GENERIC_WORDS = _GENERIC_PEOPLE | set("""thousands millions hundreds dozens billions tens
    several many some most banks companies firms brands businesses agencies governments
    sources reports report breaking update updates alert alerts warning warnings
    exclusive video videos photo photos scam scams scammers hackers today yesterday
    tonight""".split())
# Acronyms that name a technology, role or kind of item, not an actor: "a rogue AI",
# "the CEO", "an ATM", "a PIN". A claim naming one is not made specific by it.
_GENERIC_ACRONYMS = set("""AI AGI CEO CEOS CFO CTO COO IT TV PC ATM ATMS PIN SMS OTP GPS
    DNA ID IDS URL API APP VPN NFT NFTS EV EVS UFO USB PDF FAQ HR PR QR IPO LLM GPU OS
    DM DMS""".split())
_NAME_CONNECTORS = ("of", "de", "del", "du", "von", "van", "la")
_CAP_WORD = r"[A-Z][\w]*(?:['’](?![sS]\b)[\w]+)?"
_ENTITY_RX = re.compile(rf"{_CAP_WORD}(?:\s+(?:{'|'.join(_NAME_CONNECTORS)})\s+{_CAP_WORD}|"
                        rf"\s+{_CAP_WORD})*")
_QUOTE_RX = re.compile(r"""["“‘']([^"“”‘’']{6,120})["”’']""")
_AMOUNT_RX = re.compile(r"(?<![\w.])([$€£¥]\s?)?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
                        r"(\s?(?:%|(?:percent|million|billion|trillion|bn|m|k)\b))?", re.I)
_ABBR_RX = re.compile(r"\b[A-Z](?:\.[A-Z])+\.?")
# Text between two names that makes them one enumeration ("A, B and C", "A or B").
_LIST_GAP_RX = re.compile(r"\s*(?:[,;/&]\s*)?(?:(?:and|or|nor|&|et|ou|y|o|e|und|oder)\s+)?",
                          re.IGNORECASE)
_ACRONYM_RX = re.compile(r"[A-Z][A-Z0-9&]{1,5}")
LIST_MIN = 3  # an enumeration of >= LIST_MIN actors may have one member unnamed


def _unabbr(text: str) -> str:
    """'U.S.' -> 'US' so dotted and undotted abbreviations compare equal."""
    return _ABBR_RX.sub(lambda m: m.group(0).replace(".", ""), text or "")


def _num_core(s: str) -> str:
    return s.replace(",", "").rstrip(".").lstrip("0") or "0"


def _amounts(text: str) -> list[tuple[str, str]]:
    """[(as written, comparable core)] for the figures that make a claim specific:
    >= 3 significant digits (13,260; 365), or any number written with a currency,
    percent or million/billion. Years and small bare numbers (dates, "4 hours") are not
    counted: the date rule covers timing."""
    return [(shown, core) for shown, core, _ in _amount_spans(text)]


def _amount_spans(text: str) -> list[tuple[str, str, int]]:
    out: list[tuple[str, str, int]] = []
    for m in _AMOUNT_RX.finditer(text or ""):
        cur, num, unit = m.group(1), m.group(2), m.group(3)
        core = _num_core(num)
        if _YEAR_RX.match(core) and not (cur or unit):
            continue
        if (cur or unit or len(core.replace(".", "")) >= 3) and core not in {c for _, c, _ in out}:
            out.append((m.group(0).strip(), core, m.start()))
    return out


def _all_numbers(text: str) -> set[str]:
    return {_num_core(m.group(2)) for m in _AMOUNT_RX.finditer(text or "")}


def _name_tokens(words: list[str]) -> list[str]:
    return [w for w in _raw_tokens(" ".join(words)) if w not in _STOP and w not in _MONTHS]


def _is_acronym(word: str) -> bool:
    return bool(_ACRONYM_RX.fullmatch(word)) and any(c.isalpha() for c in word)


def _generic(word: str) -> bool:
    acr = word[:-1] if word.endswith("s") and word[:-1].isupper() else word  # "EVs"
    return word.casefold() in _GENERIC_WORDS or (acr.isupper() and acr in _GENERIC_ACRONYMS)


class _Spec(NamedTuple):
    shown: str
    alts: list[list[str]]
    start: int  # offset in the claim (quotes blanked), for enumeration grouping
    end: int
    kind: str  # quote | name | acronym | figure


def _spec_list(claim_text: str) -> list[_Spec]:
    """The claim's key specifics, each with match alternatives; a specific is found when
    every token of one alternative appears (as a phrase) in an evidence text.

    Capitalised runs are cut into names: a leading or trailing generic word
    ("Customers of Chase" -> "Chase", "Thousands of Bank of America" -> "Bank of
    America") is dropped with its connector, and a span made only of generic words or
    generic acronyms ("Users", "AI", "CEO") is not a specific. A span that starts the
    sentence may begin with an ordinary capitalised word joined by "of", so the part
    after that first "of" is accepted as an alternative when it is a multi-word name.
    German capitalises every noun, so a lone mixed-case word is not a name there."""
    text = _unabbr(claim_text or "")
    out: list[_Spec] = []
    seen: set[str] = set()
    for qm in list(_QUOTE_RX.finditer(text)):
        q = qm.group(1).strip()
        if len(q.split()) >= 2 and q not in seen:
            seen.add(q)
            out.append(_Spec(f"“{q}”", [["\x00" + " ".join(_raw_tokens(q))]],
                             qm.start(), qm.end(), "quote"))
            text = text[:qm.start(1)] + " " * len(qm.group(1)) + text[qm.end(1):]
    german = _guess_lang(_raw_tokens(text)) == "de"
    for m in _ENTITY_RX.finditer(text):
        wl = [(w.group(0), m.start() + w.start()) for w in re.finditer(r"\S+", m.group(0))]
        before = text[:m.start()].rstrip()
        start = not before or before[-1] in ".!?:;\"“("
        while wl and ((wl[0][0].casefold() in _STOP and not _is_acronym(wl[0][0]))
                      or _generic(wl[0][0])):
            wl, start = wl[1:], False  # "The FAA" -> "FAA"; "Customers of Chase" -> "Chase"
        while wl and (wl[-1][0].casefold() in _NAME_CONNECTORS
                      or wl[-1][0].casefold() in _GENERIC_PEOPLE):
            wl = wl[:-1]  # "Chase Customers" -> "Chase" ("USA Today" stays)
        if not wl:
            continue
        words = [w for w, _ in wl]
        span = " ".join(words)
        toks = _name_tokens(words)
        lo, hi = wl[0][1], wl[-1][1] + len(wl[-1][0])
        if len(words) == 1 and _is_acronym(words[0]):  # "WHO", "US": not stop words
            if span in seen:
                continue
            seen.add(span)
            out.append(_Spec(span, [["\x02" + words[0]]], lo, hi, "acronym"))
            continue
        if not toks or len(span) < 2 or span in seen:
            continue
        if german and len(words) == 1:
            continue
        alts = [toks]
        if start and not span.isupper():
            if len(words) == 1:
                continue  # "Officials say ...": a sentence-initial word, not a name
            # the first word may be an ordinary capitalised word ("Partial Bank of
            # America ...", "Thousands of Bank of America ..."): the rest, when it is
            # itself a multi-word name, is accepted too
            for cut in (1, 2):
                rest = _name_tokens(words[cut:])
                if (len(words) - cut >= 2 and len(rest) >= 2 and words[cut] not in
                        _NAME_CONNECTORS and rest not in alts
                        and (cut == 1 or words[1] in _NAME_CONNECTORS)):
                    alts.append(rest)
        seen.add(span)
        out.append(_Spec(span, alts, lo, hi, "name"))
    for shown, core, pos in _amount_spans(text):
        if shown not in seen:
            seen.add(shown)
            out.append(_Spec(shown, [["\x01" + core]], pos, pos + len(shown), "figure"))
    return out


def _specs(claim_text: str) -> list[tuple[str, list[list[str]]]]:
    return [(s.shown, s.alts) for s in _spec_list(claim_text)]


def _enumerations(claim_text: str, specs: list[_Spec]) -> list[list[int]]:
    """Index groups of >= LIST_MIN names/acronyms written as one enumeration
    ("Chase, Bank of America, Wells Fargo, US Bank and Zelle")."""
    text = _unabbr(claim_text or "")
    names = sorted((i for i, s in enumerate(specs) if s.kind in ("name", "acronym")),
                   key=lambda i: specs[i].start)
    groups: list[list[int]] = []
    cur: list[int] = []
    for i in names:
        if cur:
            gap = text[specs[cur[-1]].end:specs[i].start]
            if gap.strip() and _LIST_GAP_RX.fullmatch(gap):
                cur.append(i)
                continue
            if len(cur) >= LIST_MIN:
                groups.append(cur)
        cur = [i]
    if len(cur) >= LIST_MIN:
        groups.append(cur)
    return groups


def claim_specifics(claim_text: str) -> list[str]:
    """The claim's key specifics, in order: quoted titles, named actors (capitalised
    spans such as "Bank of America", "SEC", "Wells Fargo") and figures (13,260 shares,
    $144 million, 80%). A supported verdict needs each in its supporting items.

    Left out on purpose (a miss is preferred to a wrong demotion): a lone capitalised
    word that starts the sentence ("Customers ..."), month and weekday names, and small
    numbers and dates (the code-enforced date rule judges timing)."""
    return [d for d, _ in _specs(claim_text)]


def _acronym_in(acr: str, text: str, slug: str) -> bool:
    """An acronym is found case-sensitively on word boundaries in the title/snippet
    ("AI" in "rogue AI", never Italian "ai" or "said"), as a token of the lower-case URL
    slug, or as the initials of a capitalised name ("Federal Aviation Administration",
    "Department of Homeland Security")."""
    if re.search(rf"(?<![^\W_]){re.escape(acr)}s?(?![^\W_])", text):
        return True
    if acr.casefold() in slug.casefold().split():
        return True
    for m in _ENTITY_RX.finditer(text):
        ws = [w for w in m.group(0).split() if w.casefold() not in _NAME_CONNECTORS
              and w.casefold() not in ("and", "&")]
        if len(ws) >= 2 and "".join(w[0] for w in ws).upper() == acr.upper():
            return True
    return False


def _specific_found(alts: list[list[str]], texts: list[str] | list[tuple[str, str]]) -> bool:
    """`texts`: evidence texts, or (title + snippet, URL slug) pairs."""
    for item in texts:
        body, slug = item if isinstance(item, tuple) else (item, "")
        body = _unabbr(body)
        t = f"{body} {slug}".strip()
        words = _raw_tokens(t)
        raw = [w for w in words if w not in _STOP]
        for toks in alts:
            if toks[0].startswith("\x00"):  # quoted title: the whole phrase, in order
                if toks[0][1:] in " ".join(words):
                    return True
                continue
            if toks[0].startswith("\x01"):  # figure: same number, commas ignored
                if toks[0][1:] in _all_numbers(t):
                    return True
                continue
            if toks[0].startswith("\x02"):  # acronym: case-sensitive, word boundaries
                if _acronym_in(toks[0][1:], body, slug):
                    return True
                continue
            k = len(toks)
            # a multi-word name must appear as a phrase ("Bank of America"), not as words
            # scattered over the snippet ("banking" ... "Trump's America")
            if any(all(_same_term(toks[j], raw[i + j]) for j in range(k))
                   for i in range(len(raw) - k + 1)):
                return True
            initials = "".join(w[0] for w in toks)
            if k >= 2 and any(r in (initials, initials[0] + "of" + initials[1:]) for r in raw):
                return True  # "Bank of America" ~ "BoA" / "BofA"
    return False


_SLUG_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def url_slug_text(url: str | None) -> str:
    """Words of a URL's path ('/news/banks-online-outages-bank-of-america-wells-fargo/'
    -> 'news banks online outages bank of america wells fargo'). Numbers and ID-like
    tokens (hex, mixed digits) are dropped: a slug date is not a figure."""
    try:
        path = urlsplit(url or "").path
    except ValueError:
        return ""
    words = []
    for tok in re.split(r"[/\-_.+%]+", path):
        if not tok or any(c.isdigit() for c in tok):
            continue
        if re.fullmatch(r"[0-9a-f]{8,}", tok) or tok.lower() in ("html", "htm", "php", "aspx"):
            continue
        words += _SLUG_WORD_RE.findall(tok)
    return " ".join(words)


def _supporting_texts(ev: list[Evidence]) -> list[tuple[str, str]]:
    return [(_evidence_text(e), url_slug_text(e.url)) for e in ev
            if e.supports == "supports" and e.flag is None and e.source != "collected_post"]


def _is_topic(alts: list[list[str]], topic: str) -> bool:
    tt = _name_tokens((topic or "").split())
    return bool(tt) and any(all(any(_same_term(t, x) for x in toks) for t in tt)
                            for toks in alts if toks[0][:1] not in ("\x00", "\x01", "\x02"))


def check_specifics(claim_text: str, ev: list[Evidence],
                    topic: str = "") -> tuple[list[str], list[str]]:
    """(missing, partly_named) for the supporting items (title, snippet, URL slug).

    `missing`: required specifics no supporting item names. Every quoted title, figure
    and standalone name is required. In an enumeration of >= LIST_MIN actors ("Chase,
    Bank of America, Wells Fargo, US Bank and Zelle") one member may be unnamed when
    the others are named and it is not the investigated topic; it is then returned in
    `partly_named` (the verdict stands, with a method note) instead of `missing`.
    ([], []) when there is no supporting item."""
    texts = _supporting_texts(ev)
    if not texts:
        return [], []
    specs = _spec_list(claim_text)
    found = [_specific_found(s.alts, texts) for s in specs]
    tolerated: set[int] = set()
    for group in _enumerations(claim_text, specs):
        absent = [i for i in group if not found[i]]
        if len(absent) == 1 and not _is_topic(specs[absent[0]].alts, topic):
            tolerated.add(absent[0])
    missing = [s.shown for i, s in enumerate(specs) if not found[i] and i not in tolerated]
    partly = [specs[i].shown for i in sorted(tolerated)]
    return missing, partly


def missing_specifics(claim_text: str, ev: list[Evidence], topic: str = "") -> list[str]:
    """Key specifics of the claim that no item marked `supports` mentions (in its title,
    snippet or URL path slug: a news URL often names the actors when no excerpt was
    stored). [] when every specific is covered or there is no supporting item. One
    member of a longer enumeration may be unnamed (see check_specifics)."""
    return check_specifics(claim_text, ev, topic)[0]


def _join_or(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


_CONFLICT_NOTE_RX = re.compile(r"Conflict: supporting items ((?:\[\d+\](?:, | and )?)+) "
                               r"disagree with each other")
DISPUTED_CHECK = "Disputed check:"
CONFLICT_CHECK = "Conflict check:"
DISPUTED_MAX_CONFIDENCE = 0.5
_SET_DISPUTED = "Set to disputed:"


def _counted(e: Evidence) -> bool:
    """An item that can carry a relation: not a search miss, and not a collected post
    (posts that report or debunk a claim are never evidence)."""
    return e.flag != "unrelated" and e.source != "collected_post"


def _mixed_rating(e: Evidence) -> bool:
    return (e.source == "google_factcheck" and _counted(e)
            and map_rating(_rating_of(e)) == "disputed")


def valid_conflicts(ev: list[Evidence], conflicts: list[int] | None) -> list[int]:
    """The 1-based item numbers from `conflicts` that are counted supporting items;
    [] unless at least two distinct ones remain."""
    ok = sorted({n for n in (conflicts or []) if 1 <= n <= len(ev)
                 and ev[n - 1].supports == "supports" and _counted(ev[n - 1])})
    return ok if len(ok) >= 2 else []


def _item_refs(ev: list[Evidence], relation: str) -> list[str]:
    """'[4] Fortune' for each counted item with `relation` (1-based list numbers)."""
    out = []
    for n, e in enumerate(ev, start=1):
        if e.supports == relation and _counted(e):
            who = e.publisher or (urlsplit(e.url).hostname or e.url).removeprefix("www.")
            out.append(f"[{n}] {who}")
    return out


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def conflict_why(ev: list[Evidence]) -> str:
    """'item [4] Fortune contradicts the claim while item [1] Facebook supports it'."""
    con, sup = _item_refs(ev, "contradicts"), _item_refs(ev, "supports")
    return (f"item{'s' if len(con) > 1 else ''} {_and(con)} contradict"
            f"{'' if len(con) > 1 else 's'} the claim while item{'s' if len(sup) > 1 else ''} "
            f"{_and(sup)} support{'' if len(sup) > 1 else 's'} it")


def disputed_rationale(rationale: str, why: str) -> str:
    """Rationale for a verdict code changed to disputed: a leading "Set to disputed:
    <why>." and the model's sentences without the ones stating a confidence number (they
    described the dropped verdict). Idempotent."""
    r = (rationale or "").strip()
    if r.startswith(_SET_DISPUTED):
        return r
    kept = [x for x in _SENT_SPLIT.split(r) if x.strip() and not _CONF_NUM.search(x)]
    return " ".join([f"{_SET_DISPUTED} {why.strip().rstrip('.')}.", *kept]).strip()


def conflict_note(items: list[int]) -> str:
    refs = [f"[{n}]" for n in items]
    shown = refs[0] if len(refs) == 1 else ", ".join(refs[:-1]) + " and " + refs[-1]
    return (f"Conflict: supporting items {shown} disagree with each other (as stated by "
            "the model), so the sources conflict.")


def conflicts_from_notes(notes: str) -> list[int]:
    m = _CONFLICT_NOTE_RX.search(notes or "")
    return [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []


def enforce_support(claim_text: str, verdict: str, confidence: float,
                    ev: list[Evidence], topic: str = "",
                    conflicts: list[int] | None = None
                    ) -> tuple[str, float, list[str], list[str]]:
    """Code checks every verdict must pass, whoever produced it (model, rule, old run).

    1. Consistency: supported needs >= 1 item marked supports, contradicted >= 1 marked
       contradicts (unrelated items and collected posts never count).
    2. Disputed means the sources conflict: it needs >= 1 contradicting item, a
       fact-check rated mixed/half true, or >= 2 supporting items the model stated
       disagree with each other (`conflicts`, 1-based item numbers). Support plus
       context alone is not a dispute, and debunking posts never make one.
    3. Specifics: supported needs the claim's key specifics (actor, quoted title, figure)
       in at least one supporting item; topic-adjacent coverage is not support. One
       member of a longer enumeration of actors may be unnamed (method note only).
    4. Conflict: supported with any counted contradicting item, or contradicted with a
       counted supporting item that names the claim's specifics, is disputed with
       confidence <= DISPUTED_MAX_CONFIDENCE (the sources conflict; one side cannot win
       by the model's say-so).
    A failing verdict becomes unverified with confidence <= 0.3.
    Returns (verdict, confidence, method notes, missing specifics)."""
    n_sup = sum(1 for e in ev if e.supports == "supports" and _counted(e))
    n_con = sum(1 for e in ev if e.supports == "contradicts" and _counted(e))
    if verdict == "disputed" and n_con == 0 and not any(_mixed_rating(e) for e in ev):
        if valid_conflicts(ev, conflicts):
            return verdict, confidence, [], []
        n_ctx = sum(1 for e in ev if e.supports == "context" and _counted(e))
        return "unverified", min(confidence, 0.3), [
            f"{DISPUTED_CHECK} 'disputed' needs a contradicting item, a mixed fact-check "
            f"rating or supporting items that disagree with each other; the evidence has "
            f"{n_sup} supporting and {n_ctx} context item(s) and none of these, so the "
            "verdict is set to unverified (debunking or questioning posts are not "
            "evidence)."], []
    consistent = (
        verdict in ("unverified", "disputed")
        or (verdict == "supported" and n_sup > 0)
        or (verdict == "contradicted" and n_con > 0)
    )
    if not consistent:
        return "unverified", min(confidence, 0.3), [
            f"Verdict '{verdict}' was not backed by an item with a matching relation, so "
            "it is set to unverified."], []
    if verdict == "supported":
        missing, partly = check_specifics(claim_text, ev, topic)
        if missing:
            return "unverified", min(confidence, 0.3), [
                f"Specifics check: no supporting item names {_join_or(missing)}, so the "
                "verdict 'supported' is set to unverified (related coverage is not "
                "confirmation of the claim's specific actor or figure)."], missing
        if n_con:
            return "disputed", min(confidence, DISPUTED_MAX_CONFIDENCE), [
                f"{CONFLICT_CHECK} {conflict_why(ev)}, so the sources conflict and the "
                "verdict 'supported' is set to disputed."], []
        if partly:
            return verdict, confidence, [
                f"Specifics check: the supporting items name the other actors the claim "
                f"lists but not {_join_or(partly)}; that part is unconfirmed."], []
    if verdict == "contradicted" and n_sup and not check_specifics(claim_text, ev, topic)[0]:
        # a supporting item that names the claim's specifics is a real conflict; one that
        # only covers related events is not confirmation and leaves 'contradicted' alone
        return "disputed", min(confidence, DISPUTED_MAX_CONFIDENCE), [
            f"{CONFLICT_CHECK} {conflict_why(ev)}, so the sources conflict and the "
            "verdict 'contradicted' is set to disputed."], []
    return verdict, confidence, [], []


_LOWERED = "Lowered to unverified:"
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CONF_NUM = re.compile(r"\bconfiden\w*\b[^.]*?\d|\d(?:\.\d+)?\s*%?[^.]*?\bconfiden\w*",
                       re.IGNORECASE)
_AFFIRM = re.compile(r"\b(?:supported|supports? (?:the|this|it)|confirm(?:s|ed|ing)?|"
                     r"corroborat\w*|(?:is|are) (?:true|accurate|correct))\b", re.IGNORECASE)
_NEGATED = re.compile(r"\b(?:not|no|never|none|nothing|nobody|neither|nor|cannot|unable)\b"
                      r"|n't\b", re.IGNORECASE)


def demoted_rationale(rationale: str, why: str) -> str:
    """Rationale for a verdict code lowered to unverified: a leading "Lowered to
    unverified: <why>." and none of the model's sentences that state a confidence number
    or call the claim supported/confirmed (they described the verdict that was dropped).
    Idempotent: an already-lowered rationale is returned unchanged."""
    r = (rationale or "").strip()
    if r.startswith(_LOWERED):
        return r
    kept = [x for x in _SENT_SPLIT.split(r) if x.strip()
            and not _CONF_NUM.search(x)
            and not (_AFFIRM.search(x) and not _NEGATED.search(x))]
    why = why.strip().rstrip(".")
    return " ".join([f"{_LOWERED} {why}.", *kept]).strip()


# --------------------------------------------------------------------- bottom line


def clip_sentence(text: str, limit: int = BOTTOM_LINE_MAX) -> str:
    """Whitespace-collapsed text, cut at a word boundary to at most `limit` chars."""
    t = re.sub(r"\s+", " ", text or "").strip()
    if len(t) <= limit:
        return t
    cut = t[:limit - 1].rsplit(" ", 1)[0].rstrip(",;:")
    return cut + "…"


def _names(items: list[Evidence]) -> str:
    """'Reuters', 'Reuters and CBS', '2 items from Reuters', '5 sources (A, B and others)'."""
    names: list[str] = []
    for e in items:
        n = e.publisher or (urlsplit(e.url).hostname or e.url).removeprefix("www.")
        if n not in names:
            names.append(n)
    if len(items) >= 3:
        more = " and others" if len(names) > 2 else ""
        return f"{len(items)} sources ({', '.join(names[:2])}{more})"
    if len(items) == 2:
        return " and ".join(names) if len(names) == 2 else f"2 items from {names[0]}"
    return names[0]


def with_partly(line: str, partly: list[str]) -> str:
    """A bottom line that says which listed actors no source names.

    `partly`: actors of an enumeration that no supporting item names (check_specifics);
    the verdict stands but must not read as confirmed for them. The line gets " Not
    confirmed for X." appended; "" (use the rule sentence) when the line itself names
    one of them (it may call that actor confirmed) or the result would be too long."""
    if not partly or not line or "not confirmed for" in line.lower():
        return line
    if any(re.search(rf"(?<!\w){re.escape(x)}(?!\w)", line, re.IGNORECASE) for x in partly):
        return ""
    out = line.rstrip() + ("" if line.rstrip()[-1:] in ".!?" else ".") + \
        f" Not confirmed for {_join_or(partly)}."
    return out if len(out) <= BOTTOM_LINE_MAX else ""


def rule_bottom_line(verdict: str, evidence: list[Evidence], counts: dict[str, int],
                     searched: bool = True, missing: list[str] | None = None,
                     no_conflict: bool = False, partly: list[str] | None = None) -> str:
    """One sentence from the evidence list and stance counts only (never beyond them).

    `missing`: claim specifics no supporting item names (see enforce_support).
    `no_conflict`: a 'disputed' verdict was lowered because no source conflicts.
    `partly`: listed actors no supporting item names (verdict stands; said so)."""
    total = sum(counts.values())
    posts = (f"{counts.get('assert', 0)} of {total} classified posts assert it"
             if total else "")
    rel = [e for e in evidence if e.flag != "unrelated"]
    n_unrel = len(evidence) - len(rel)
    if not evidence:
        lead = ("No news or fact-check evidence was found" if searched
                else "No evidence search ran for this claim")
        return clip_sentence(lead + (f"; {posts}." if posts else "."))
    if not rel:
        what = "the only item retrieved is" if n_unrel == 1 else \
            f"all {n_unrel} items retrieved are"
        return clip_sentence(f"No relevant evidence was found: {what} unrelated to the "
                             "claim (a search miss); worth a manual search.")
    sup = [e for e in rel if e.supports == "supports"]
    con = [e for e in rel if e.supports == "contradicts"]
    old = [e for e in rel if e.flag == "old_event"]
    parts: list[str] = []
    if sup and missing:
        parts.append(f"{_names(sup)} report{'s' if len(sup) == 1 else ''} related events, "
                     f"but none names {_join_or(missing)}, so the claim is {verdict}")
    elif sup and no_conflict and not con:
        parts.append(f"{_names(sup)} report{'s' if len(sup) == 1 else ''} related events "
                     "and no retrieved item contradicts the claim, so the sources do not "
                     "conflict; it is unverified, not disputed")
    elif sup:
        parts.append(f"{_names(sup)} support{'s' if len(sup) == 1 else ''} it")
        if not con and verdict in ("disputed", "unverified"):
            # e.g. an older run's model verdict: never let the sentence read as support
            parts[-1] += f", but this was not judged enough to call it supported ({verdict})"
        elif partly and verdict == "supported":
            parts[-1] += f", but not confirmed for {_join_or(partly)}"
    if con:
        parts.append(f"{_names(con)} contradict{'s' if len(con) == 1 else ''} it")
    if not sup and not con:
        parts.append(f"{len(rel)} retrieved item{'' if len(rel) == 1 else 's'} give"
                     f"{'s' if len(rel) == 1 else ''} context but none confirms or "
                     "contradicts it")
    if old:
        first = min(_as_date(e.published_at) for e in old)  # type: ignore[type-var]
        parts.append(f"{len(old)} item{'' if len(old) == 1 else 's'} describe"
                     f"{'s' if len(old) == 1 else ''} an earlier event "
                     f"(from {first.strftime('%b %Y')})")  # type: ignore[union-attr]
    if n_unrel:
        parts.append(f"{n_unrel} unrelated result{'' if n_unrel == 1 else 's'} ignored")
    s = "; ".join(parts)
    if "." not in s.split(" ", 1)[0]:  # never re-case a hostname ("www.example.com")
        s = s[0].upper() + s[1:]
    return clip_sentence(s + ".")


_CITE_MARK = re.compile(r"\s*\[\d+(?:\s*[,-]\s*\d+)*\]")


def _clean_model_line(text: str) -> str:
    t = _URL_RX.sub("", text or "")
    t = _CITE_MARK.sub("", t)
    return clip_sentence(t)


# ---------------------------------------------------------------------- verdicts


class VerdictResult(NamedTuple):
    verdict: Verdict
    confidence: float
    rationale: str  # analysis only
    evidence: list[Evidence]
    bottom_line: str
    method_notes: list[str]


def _cap_for(n_cited: int) -> float:
    """Confidence ceiling from the amount of valid cited evidence (thin -> low)."""
    if n_cited <= 0:
        return 0.0
    if n_cited == 1:
        return 0.5
    if n_cited == 2:
        return 0.7
    return 0.9


def verdict_llm(claim_text: str, counts: dict[str, int], ev: list[Evidence],
                llm: Any, purpose: str, window: Window | None = None,
                topic: str = "") -> VerdictResult:
    """LLM verdict over retrieved evidence, post-validated. Returns updated evidence copies."""
    ev, method = flag_unrelated(claim_text, ev, topic)
    if window is not None:
        lo, hi = _support_range(window)
        when = (f"The claim circulated in collected posts published {_fmt_window(window)} "
                f"(UTC). An evidence item can support it only if it was published between "
                f"{lo.isoformat()} and {hi.isoformat()}; items published earlier or later "
                "most likely describe a different event and give context only. An older "
                "item may still contradict the claim (recurring hoaxes are debunked "
                "again); then say its date in the rationale.\n\n")
    else:
        when = ""
    prompt = (
        f'Claim: "{claim_text}"\n\n' + when +
        f"How collected social posts present it (context only, NOT evidence; reporting or "
        f"debunking posts are never support): {_stance_summary(counts)}\n\n"
        "Retrieved external evidence (the ONLY sources you may cite):\n"
        + _evidence_block(ev) + "\n\n"
        "Decide a verdict: supported | contradicted | disputed | unverified. Use "
        "'unverified' when the evidence does not directly address this claim. Use "
        "'supported' only if the cited evidence confirms the claim's specific actor, "
        "figure and date; partial confirmation (same event, but the named actor, number "
        "or title is not shown) is 'unverified'. Use 'disputed' only when the retrieved "
        "sources conflict: at least one cited item contradicts the claim, or supporting "
        "items disagree with each other (then list their [n] numbers in "
        "conflicting_items). Support plus context is not a dispute, and posts that debunk "
        "or question the claim never make it disputed. Cite only "
        "URLs from the list above, and for each cited URL say whether it supports, "
        "contradicts or only gives context for the claim. Give a confidence 0-1 (lower "
        "when evidence is thin or indirect) and a short rationale that names the sources "
        "and their publication dates. Also give a bottom_line: ONE plain sentence of at "
        f"most {BOTTOM_LINE_MAX} characters saying what the evidence shows, for an analyst "
        "who reads nothing else."
    )
    out: VerdictOut = llm.parse(
        VerdictOut, prompt,
        system="You assess claims for an OSINT analyst. Never state that something is "
               "simply true or false; only weigh the retrieved evidence.",
        purpose=purpose, max_tokens=4096)
    by_url = {e.url.strip(): e for e in ev}
    relation: dict[str, str] = {}
    dropped: list[str] = []
    for c in out.cited:
        u = (c.url or "").strip()
        if u in by_url:
            relation.setdefault(u, c.relation)
        else:
            dropped.append(u)  # not retrieved by the pipeline -> never cited
    model_rel = dict(relation)
    for u, e in by_url.items():  # unrelated items never support or contradict
        if e.flag == "unrelated" and u in relation:
            relation[u] = "context"
    new_ev = [e.model_copy(update={"supports": relation.get(e.url.strip(), "context")})
              if e.url.strip() in relation else e.model_copy() for e in ev]
    # Code-enforced date rule: never rely on the model alone for this.
    new_ev, date_notes, date_method = apply_date_rule(new_ev, window)
    method += date_method
    for e in new_ev:
        if e.url.strip() in relation:
            relation[e.url.strip()] = e.supports
    rationale = (out.rationale or "").strip()
    if dropped:
        method.append(f"{len(dropped)} cited URL(s) were not in the retrieved evidence and "
                      "were discarded.")
    if date_notes:
        rationale += " " + " ".join(date_notes)
    verdict: Verdict = out.verdict
    n_sup = sum(1 for r in relation.values() if r == "supports")
    n_con = sum(1 for r in relation.values() if r == "contradicts")
    if not relation:
        if verdict != "unverified":
            method.append("No valid citation remained, so the verdict is set to unverified.")
        verdict, conf = "unverified", 0.0
    else:
        conf = max(0.0, min(float(out.confidence or 0.0), 1.0))
        conf = min(conf, _cap_for(n_sup + n_con))
        # Consistency + key-specifics checks (code-enforced, same as for older runs).
        conflicts = valid_conflicts(new_ev, list(getattr(out, "conflicting_items", []) or []))
        verdict, conf, notes, missing = enforce_support(  # type: ignore[assignment]
            claim_text, verdict, conf, new_ev, topic, conflicts)
        if verdict == "disputed" and conflicts and not any(
                e.supports == "contradicts" and _counted(e) for e in new_ev):
            notes = [conflict_note(conflicts), *notes]
        no_conflict = bool(notes) and notes[0].startswith(DISPUTED_CHECK)
        if notes and notes[0].startswith("Verdict '"):
            notes = [f"Model verdict '{out.verdict}' was not backed by a cited item with a "
                     "matching relation, so it is set to unverified."]
        method += notes
        if verdict == "unverified":
            conf = min(conf, 0.3)
    # The model's sentence is kept only when code changed nothing it relied on: same
    # verdict, and every cited relation unchanged (no item demoted by the date or
    # relevance rule). Otherwise the rule sentence, built from the final evidence only.
    model_line = _clean_model_line(getattr(out, "bottom_line", "") or "")
    missing = missing if relation else []
    no_conflict = bool(relation) and no_conflict
    if verdict == "unverified" and out.verdict != "unverified":
        why = ("no cited URL was in the retrieved evidence" if not relation else
               f"no supporting item names {_join_or(missing)}" if missing else
               "no retrieved source contradicts the claim or conflicts with another, so "
               "it is not disputed" if no_conflict else
               f"no cited item has the relation a '{out.verdict}' verdict needs")
        rationale = demoted_rationale(rationale, why)
    elif verdict == "disputed" and out.verdict != "disputed":
        rationale = disputed_rationale(rationale, conflict_why(new_ev))
    untouched = verdict == out.verdict and model_rel == relation and not dropped
    partly = check_specifics(claim_text, new_ev, topic)[1] if verdict == "supported" else []
    bottom = model_line if (model_line and untouched) else ""
    bottom = with_partly(bottom, partly) or rule_bottom_line(
        verdict, new_ev, counts, missing=missing, no_conflict=no_conflict, partly=partly)
    return VerdictResult(verdict, round(conf, 2), rationale, new_ev, bottom, method)


def verdict_rules(counts: dict[str, int], ev: list[Evidence],
                  providers: list[str], note: str = "",
                  window: Window | None = None, claim_text: str = "",
                  topic: str = "") -> VerdictResult:
    """Rule-based verdict from fact-check ratings only. Never true/false."""
    stance_txt = (f"stance distribution among collected posts: {_stance_summary(counts)} "
                  "(debunking/reporting posts are not support)")
    method: list[str] = []
    if not ev:
        if providers:
            method.append(f"Evidence providers queried: {', '.join(providers)}; they "
                          "returned nothing.")
        else:
            method.append("No evidence provider is configured (set "
                          "GOOGLE_FACTCHECK_API_KEY, SERPER_API_KEY or SERPAPI_KEY).")
        if note:
            method.append(note[0].upper() + note[1:] + ".")
        searched = bool(providers) and not note
        lead = ("No external evidence was retrieved for this claim" if searched else
                "No external evidence was retrieved for this claim (no evidence search "
                "ran)")
        return VerdictResult("unverified", 0.0, f"{lead}; {stance_txt}.", [],
                             rule_bottom_line("unverified", [], counts, searched), method)
    ev, method = (flag_unrelated(claim_text, ev, topic) if claim_text
                  else ([e.model_copy() for e in ev], []))
    new_ev: list[Evidence] = []
    rels: list[str | None] = []
    for e in ev:
        e2 = e.model_copy()
        rating = _rating_of(e) if e.source == "google_factcheck" else ""
        rel = None if e.flag == "unrelated" else map_rating(rating)
        rels.append(rel)
        if rel in ("supports", "contradicts"):
            e2.supports = rel  # type: ignore[assignment]
        new_ev.append(e2)
    # Same code-enforced date rule as the LLM path.
    new_ev, date_notes, date_method = apply_date_rule(new_ev, window)
    method += date_method
    tally = {"supports": 0, "contradicts": 0, "disputed": 0}
    rated: list[str] = []
    for e, e2, rel in zip(ev, new_ev, rels, strict=True):
        if rel is None:
            continue
        if rel == "supports" and e2.supports != "supports":
            rated.append(f"{e.publisher or e.url}: '{_rating_of(e)}' -> not counted "
                         f"(published {_date_label(e)})")
            continue
        tally[rel] += 1
        rated.append(f"{e.publisher or e.url}: '{_rating_of(e)}' -> {rel}")
    dates_txt = (" " + " ".join(date_notes)) if date_notes else ""
    n = sum(tally.values())
    if n == 0:
        bottom = rule_bottom_line("unverified", new_ev, counts)
        if rated:
            return VerdictResult(
                "unverified", 0.0,
                "No fact-check rating could be counted: " + "; ".join(rated)
                + f"; the items are listed as context only. {stance_txt}.{dates_txt}",
                new_ev, bottom, method)
        return VerdictResult(
            "unverified", 0.0,
            f"{len(ev)} external item(s) retrieved but none carries a fact-check rating "
            f"that maps to a verdict; they are listed as context only. {stance_txt}."
            + dates_txt, new_ev, bottom, method)
    top = max(tally.values())
    leaders = [k for k, v in tally.items() if v == top]
    verdict: Verdict
    if len(leaders) > 1:
        verdict = "disputed"
    else:
        verdict = {"supports": "supported", "contradicts": "contradicted",
                   "disputed": "disputed"}[leaders[0]]  # type: ignore[assignment]
    conf = HEURISTIC_CONFIDENCE if n >= 2 else THIN_HEURISTIC_CONFIDENCE
    if verdict == "disputed" and len(leaders) > 1:
        conf = THIN_HEURISTIC_CONFIDENCE
    rationale = (f"Rule-based mapping of {n} fact-check rating(s) (majority vote): "
                 + "; ".join(rated) + f". Ratings may refer to a related but not identical "
                 f"claim; check the cited reviews. {stance_txt}." + dates_txt)
    return VerdictResult(verdict, conf, rationale, new_ev,
                         rule_bottom_line(verdict, new_ev, counts), method)


# ------------------------------------------------------------- older run files

_LEGACY_METHOD = re.compile(
    r"\s*\[(?:Method:|Model verdict |No valid citation |\d+ cited URL\(s\) were not |"
    r"\d+ narrative post\(s\) without |Date check:|The claim's time window is unknown|"
    r"evidence lookup skipped )[^\[\]]*\]")
_LEGACY_PROVIDERS = re.compile(
    r" \((providers queried \([^()]*\) returned nothing|no evidence provider is configured "
    r"\([^()]*\))(; [^()]*)?\)")


def split_method_notes(rationale: str) -> tuple[str, str]:
    """Split a rationale written before `method_notes` existed into (analysis, notes).

    Moves the bracketed pipeline notes ("[Method: ...]", "[Model verdict ... set to
    unverified.]", "[N narrative post(s) without ...]", "[Date check: ...]", ...) and the
    provider list of the no-evidence sentence. Citation markers like "[3]" and the
    "[... contradicts the claim but was published ...]" analysis note stay.
    """
    notes: list[str] = []

    def take(m: re.Match[str]) -> str:
        notes.append(m.group(0).strip()[1:-1].strip())
        return ""

    text = _LEGACY_METHOD.sub(take, rationale or "")

    def take_prov(m: re.Match[str]) -> str:
        notes.append((m.group(1) + (m.group(2) or "")).strip())
        return ""

    text = _LEGACY_PROVIDERS.sub(take_prov, text)
    notes = [n if n.endswith(".") else n + "." for n in notes]
    notes = [n[0].upper() + n[1:] for n in notes]
    return re.sub(r"\s+", " ", text).strip(), " ".join(notes)


def backfill_claim(claim: Claim, topic: str = "", window: Window | None = None) -> Claim:
    """A copy of a claim from an older run with the new fields filled by rule.

    Flags (`unrelated`, `old_event`) are set on evidence copies, the rationale is split
    into analysis + `method_notes`, and an empty `bottom_line` gets the rule sentence.
    Relations and the verdict are kept as recorded unless the verdict no longer passes
    the checks every new verdict passes (`enforce_support`): e.g. a "supported" claim
    whose only support is now flagged unrelated, or whose supporting items never name
    the claim's actor or figure, is lowered to unverified (confidence <= 0.3) with a
    method note. An unrelated item never counts as support.
    """
    ev, rel_notes = flag_unrelated(claim.text, claim.evidence, topic)
    if window is not None:
        lo = _support_range(window)[0]
        for e in ev:
            d = _as_date(e.published_at)
            if d is not None and d < lo and e.supports != "contradicts" and e.flag is None:
                e.flag = "old_event"
    changed = [o.url for o, e in zip(claim.evidence, ev, strict=True)
               if o.supports != e.supports]
    rationale, notes = (claim.rationale, claim.method_notes) if claim.method_notes else \
        split_method_notes(claim.rationale)
    extra = rel_notes + ([f"Backfill: {len(changed)} item(s) previously counted were "
                          "flagged unrelated."] if changed else [])
    verdict, conf, checks, missing = enforce_support(
        claim.text, claim.verdict, claim.confidence, ev, topic,
        conflicts_from_notes(claim.method_notes))
    no_conflict = bool(checks) and checks[0].startswith(DISPUTED_CHECK)
    bottom = claim.bottom_line
    if verdict == claim.verdict:
        extra += [c for c in checks if c.startswith("Specifics check")]
    elif verdict == "disputed":
        why = conflict_why(ev)
        extra.append(f"Backfill: verdict {claim.verdict} changed to disputed because {why}.")
        bottom = ""  # the recorded sentence described the old verdict
        rationale = disputed_rationale(rationale, why)
    else:
        why = (f"no supporting item names {_join_or(missing)}" if missing else
               "no retrieved source contradicts the claim or conflicts with another, so "
               "it is not disputed" if no_conflict else
               "its only support was flagged unrelated" if changed else
               "no cited item has a matching relation")
        extra.append(f"Backfill: verdict {claim.verdict} lowered to unverified because "
                     f"{why}.")
        bottom = ""  # the recorded sentence described the old verdict
        rationale = demoted_rationale(rationale, why)
    notes = " ".join([notes, *extra]).strip()
    partly = check_specifics(claim.text, ev, topic)[1] if verdict == "supported" else []
    bottom = with_partly(bottom, partly) or rule_bottom_line(
        verdict, ev, claim.stances, missing=missing, no_conflict=no_conflict, partly=partly)
    return claim.model_copy(update={"text": clean_claim_text(claim.text) or claim.text,
                                    "evidence": ev, "rationale": rationale,
                                    "method_notes": notes, "bottom_line": bottom,
                                    "verdict": verdict, "confidence": round(conf, 2)})


# ------------------------------------------------------------------- orchestrator


def _pool_map(fn, items: list, workers: int) -> list:
    if not items:
        return []
    if workers <= 1 or len(items) == 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def assess_claims(
    posts: list[Post],
    narratives: list[Narrative],
    *,
    topic: str,
    llm: Any,
    cfg: dict[str, str],
    run_dir: Path | None,
    window_start: Any = None,
    window_end: Any = None,
    narrative_order: list[str] | None = None,
    max_narratives: int = MAX_NARRATIVES,
    evidence_budget_s: float | None = None,
) -> list[Claim]:
    """Extract claims per narrative, classify stance, gather evidence and assign verdicts.

    `window_start`/`window_end` (dates or ISO strings) are the run window. When the
    caller does not pass them, the span of the countable posts' timestamps is used.

    `narrative_order`: narrative IDs in priority order (the pipeline passes the
    preliminary ranking, brief.claim_candidates); the first `max_narratives` of them that
    can carry a claim are assessed and narratives not listed are skipped. Without it,
    the largest narratives are taken.
    """
    from socmint.config import analysis_workers
    workers = analysis_workers(cfg)  # concurrent LLM calls (default 12; ANALYSIS_LLM_WORKERS)
    by_id = {p.post_id: p for p in posts}
    window = make_window(window_start, window_end) or window_from_posts(posts)
    # Every claim must link to the posts that carry it. A narrative with no countable,
    # text-bearing member (e.g. media-only posts, or only unverified timestamps) cannot
    # support a claim, so it is skipped rather than emitted with an empty post list.
    eligible: list[Narrative] = []
    for n in narratives:
        if _member_posts(n, by_id):
            eligible.append(n)
        else:
            log.info("claims: narrative %s skipped (no countable text-bearing posts)", n.id)
    limit = max(0, int(max_narratives))
    if narrative_order is not None:
        pos = {nid: i for i, nid in enumerate(narrative_order)}
        chosen = sorted((n for n in eligible if n.id in pos), key=lambda n: pos[n.id])[:limit]
    else:
        chosen = sorted(eligible, key=lambda n: (-len(n.member_post_ids), n.id))[:limit]
    log.info("claims: assessing %d narrative(s): %s", len(chosen),
             ", ".join(n.id for n in chosen))

    # ---- 1. claim extraction ------------------------------------------------------
    def extract(n: Narrative) -> list[tuple[Narrative, str, str]]:
        members = _member_posts(n, by_id)
        if llm is not None and members:
            try:
                texts = _extract_llm(n, members, topic, llm)
                if texts:
                    return [(n, t, "llm") for t in texts]
            except Exception as e:  # noqa: BLE001 - LLMError or anything else: fall back
                log.warning("claim extraction failed for %s: %s", n.id, e)
        return [(n, heuristic_claim_text(n), "heuristic")]

    drafts = [d for ds in _pool_map(extract, chosen, workers) for d in ds]

    # ---- 2. stance ----------------------------------------------------------------
    def stance(d: tuple[Narrative, str, str]) -> dict[str, Any]:
        n, text, _ = d
        members = _member_posts(n, by_id)[:MAX_STANCE_POSTS]
        result: dict[str, tuple[str, str]] = {}
        llm_ok = 0
        if llm is not None:
            for i in range(0, len(members), STANCE_BATCH):
                batch = members[i:i + STANCE_BATCH]
                try:
                    got = _stance_llm_batch(text, batch, llm, f"claims.stance:{n.id}")
                    result.update(got)
                    llm_ok += len(got)
                except Exception as e:  # noqa: BLE001 - incl. LLMError: fall back
                    log.warning("stance batch failed for %s: %s", n.id, e)
        heur = 0
        for p in members:  # posts the LLM skipped (or no LLM) get the keyword rule
            if p.post_id not in result:
                result[p.post_id] = heuristic_stance(p.text)
                heur += 1
        counts, ids = _empty_stances()
        for p in members:
            s = result[p.post_id][0]
            counts[s] += 1
            ids[s].append(p.post_id)
        return {"counts": counts, "ids": ids, "related": [p.post_id for p in members],
                "llm": llm_ok, "heur": heur}

    stances = _pool_map(stance, drafts, workers)

    # ---- 3. evidence ----------------------------------------------------------------
    # Providers run concurrently with each other; each provider's own calls stay
    # strictly sequential (one account, never parallelised): evidence.gather_evidence_many.
    providers = evidence_mod.available_providers(cfg)
    # evidence_budget_s: None = EVIDENCE_BUDGET_S; math.inf when the run has no time limit
    deadline = time.monotonic() + (EVIDENCE_BUDGET_S if evidence_budget_s is None
                                   else evidence_budget_s)
    # One query per normalised claim text per run: identical claims (same LLM claim
    # from two narratives, same representative sentence) reuse the first result, so
    # quota is not spent twice and the first raw response is never overwritten.
    keys: list[str] = []
    unique: dict[str, str] = {}  # key -> query, first occurrence order
    for _, text, _ in drafts:
        q = text[:300]
        key = re.sub(r"\s+", " ", q).strip().casefold()
        keys.append(key)
        unique.setdefault(key, q)
    cache: dict[str, tuple[list[Evidence], list[str]]] = {}
    if providers and unique:
        got = evidence_mod.gather_evidence_many(list(unique.values()), cfg, run_dir,
                                                deadline=deadline)
        cache = dict(zip(unique.keys(), got, strict=True))
    evidences: list[list[Evidence]] = []
    notes: list[str] = []
    for key in keys:
        items, skipped = cache.get(key, ([], []))
        evidences.append([e.model_copy() for e in items])
        notes.append(f"evidence lookup skipped for {', '.join(skipped)}: time budget for "
                     "evidence exhausted" if skipped else "")

    # ---- 4. verdicts --------------------------------------------------------------
    def judge(i: int) -> tuple[VerdictResult, bool]:
        n, text, _ = drafts[i]
        ev, counts = evidences[i], stances[i]["counts"]
        if llm is not None and ev:
            try:
                return verdict_llm(text, counts, ev, llm, f"claims.verdict:{n.id}", window,
                                   topic=topic), True
            except Exception as e:  # noqa: BLE001 - incl. LLMError: fall back
                log.warning("verdict call failed for %s: %s", n.id, e)
        return verdict_rules(counts, ev, providers, notes[i], window, claim_text=text,
                             topic=topic), False

    judged = _pool_map(judge, list(range(len(drafts))), workers)

    # ---- 5. assemble --------------------------------------------------------------
    # `rationale` keeps the analysis; every pipeline note goes to `method_notes`.
    claims: list[Claim] = []
    for i, ((n, text, ext_method), st, (res, v_llm)) in enumerate(
            zip(drafts, stances, judged, strict=True), 1):
        used_llm = ext_method == "llm" or st["llm"] > 0 or v_llm
        stance_m = "llm" if st["llm"] else "heuristic"
        if st["llm"] and st["heur"]:
            stance_m += f" ({st['heur']} by keyword fallback)"
        method = [f"Method: extraction={ext_method}; stance={stance_m}; "
                  f"verdict={'llm' if v_llm else 'rule-based'}; "
                  f"{len(st['related'])} post(s) classified."]
        method += res.method_notes
        if notes[i - 1] and res.evidence:  # partial lookup (empty: verdict_rules noted it)
            method.append(notes[i - 1][0].upper() + notes[i - 1][1:] + ".")
        excluded = _excluded_members(n, by_id)
        if excluded:
            shown = ", ".join(excluded[:10]) + (" ..." if len(excluded) > 10 else "")
            method.append(f"{len(excluded)} narrative post(s) without a verified in-window "
                          f"timestamp were not classified or counted: {shown}.")
        claims.append(Claim(
            id=f"CL{i}",
            text=text,
            narrative_id=n.id,
            related_post_ids=st["related"],
            stances=st["counts"],
            stance_post_ids=st["ids"],
            evidence=res.evidence,
            verdict=res.verdict,
            confidence=res.confidence,
            rationale=res.rationale.strip(),
            method=_llm_name(llm) if (used_llm and llm is not None) else "heuristic",
            bottom_line=res.bottom_line,
            method_notes=" ".join(method),
        ))
    return claims
