"""Unified data model (docs/SPEC.md §5.2, §5.3).

Every source normalizes into `Post`. Every analysis record carries post IDs so each
finding links to its evidence. Fields marked "additive" extend the spec schema without
changing any spec field; they are recorded in docs/DECISIONS.md.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- posts

RelationType = Literal["quote", "reply", "repost", "forward", "link", "mention"]
PostKind = Literal["post", "video", "comment", "ad"]


class Metrics(BaseModel):
    """Platform engagement metrics. None means 'the source does not expose it'."""

    model_config = ConfigDict(extra="forbid")

    views: int | None = None
    likes: int | None = None
    replies: int | None = None
    reposts: int | None = None
    shares: int | None = None
    comments: int | None = None
    forwards: int | None = None
    reactions: int | None = None
    quotes: int | None = None  # additive: X exposes quote counts separately


class Media(BaseModel):
    type: Literal["photo", "video", "animated_gif", "document", "image", "other"]
    url: str | None = None
    thumbnail: str | None = None


class Relation(BaseModel):
    """A DIRECT relationship observed in platform data (never inferred)."""

    type: RelationType
    target_post_id: str | None = None
    target_url: str | None = None
    target_handle: str | None = None
    evidence: Literal["direct"] = "direct"


class AdMeta(BaseModel):
    run_start: str | None = None
    run_end: str | None = None
    advertiser_id: str | None = None
    landing_url: str | None = None
    regions: list[str] = Field(default_factory=list)
    impressions_range: str | None = None


class Post(BaseModel):
    post_id: str  # "<source>:<native_id>"
    source: str
    kind: PostKind = "post"
    author_id: str | None = None
    author_handle: str | None = None
    author_name: str | None = None
    author_created_at: str | None = None
    author_avatar_url: str | None = None  # additive: needed for reused-avatar signal
    text: str = ""
    lang: str | None = None
    published_at: str | None = None  # UTC ISO8601 from the platform
    collected_at: str  # UTC ISO8601, when we fetched it
    timestamp_verified: bool = False
    timestamp_note: str | None = None  # additive: why a timestamp is unverified
    metrics: Metrics = Field(default_factory=Metrics)
    media: list[Media] = Field(default_factory=list)
    url: str | None = None
    relations: list[Relation] = Field(default_factory=list)
    raw_ref: str | None = None
    query: str | None = None  # additive: which query/seed surfaced the post
    ad_meta: AdMeta | None = None
    in_window: bool = True  # additive: published_at within [start, end] (all kinds, ads too)
    active_in_window: bool | None = None  # additive, ads: run [start, end] overlaps window
    context_only: bool = False  # additive: embedded in a response (e.g. a quoted tweet) but
    #                             not matched by the topic search; shown, never analysed
    duplicate_of: str | None = None  # additive: set by dedupe for exact re-collections


# --------------------------------------------------------------------- source status


class SourceState(str, Enum):
    ok = "ok"
    completed_zero = "completed_zero"
    partial = "partial"
    out_of_coverage = "out_of_coverage"
    skipped_no_credentials = "skipped_no_credentials"
    skipped_capability = "skipped_capability"
    failed = "failed"


class SourceStatus(BaseModel):
    source: str
    query: str
    state: SourceState
    count: int = 0
    reason: str = ""
    cursor: str | None = None
    date_filtered: bool = True  # False => source could not bound by date server-side
    requests: int = 0  # additive
    cost_usd: float = 0.0  # additive
    quota_units: int | None = None  # additive (YouTube)
    started_at: str | None = None
    finished_at: str | None = None
    # additive (2026-11-20): plain-language main reason (socmint.status_text), what to do,
    # and when the platform limit resets if it said so. `reason` keeps the technical detail.
    headline: str | None = None
    headline_kind: str | None = None  # account|quota|rate_limit|error|setup|stopped|limit|coverage|none|ok
    action: str | None = None
    resets_at: str | None = None


# ------------------------------------------------------------------- analysis records


class Narrative(BaseModel):
    id: str
    label: str
    claim_summary: str
    member_post_ids: list[str]
    example_post_ids: list[str]
    volume_by_day: dict[str, int] = Field(default_factory=dict)
    volume_by_platform: dict[str, int] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)  # additive
    label_method: Literal["llm", "keyword"] = "keyword"  # additive
    first_seen_at: str | None = None  # additive (earliest verified member)
    unverified_member_count: int = 0  # additive: members excluded from volume
    # additive (owner decision 2026-11-19, analysis readability): how the narrative relates
    # to the topic, judged by the labelling LLM from the member texts. "threat" = a risk to
    # the topic's subject (security incident, fraud, outage, scam, impersonation,
    # disinformation, reputational attack); "related" = on-topic but routine (markets,
    # products, news, opinion); "off_topic" = the topic is only mentioned in passing.
    # None = not assessed (no LLM, or the label call failed). Never used to drop posts.
    relevance: Literal["threat", "related", "off_topic"] | None = None
    relevance_reason: str = ""  # additive: one short clause from the labeller
    # additive (owner decision 2026-11-19): how SEVERE the discussion is for the topic's
    # subject, judged by the same labelling LLM call. Narratives are ranked by it.
    #   critical = active harm to the subject or its customers: leaked/sold card or account
    #              data, fraud or scam campaigns, impersonation, credible attack or breach
    #              claims;
    #   high     = false or unconfirmed claims likely to cause panic or financial loss
    #              (e.g. outage/bank-run rumours, AI-attack lockdown claims), or misleading
    #              content pushed by coordinated accounts;
    #   medium   = criticism, controversy, unconfirmed allegations that could hurt
    #              reputation;
    #   low      = routine news, market chatter, neutral mentions.
    # None = not assessed (keyword label, failed call, older run) or off_topic.
    severity: Literal["critical", "high", "medium", "low"] | None = None
    severity_reason: str = ""  # additive: one short clause from the labeller


class Signal(BaseModel):
    type: Literal[
        "near_duplicate_text",
        "near_simultaneous_posting",
        "similar_creation_dates",
        "reused_image",
        "amplification_loop",
    ]
    score: float
    evidence_post_ids: list[str]
    detail: str = ""  # additive: human-readable explanation of what fired


class Cluster(BaseModel):
    id: str
    member_author_ids: list[str]
    signals: list[Signal]
    supporting_post_ids: list[str]
    benign_explanations: list[str] = Field(default_factory=list)
    member_handles: list[str] = Field(default_factory=list)  # additive
    sources: list[str] = Field(default_factory=list)  # additive
    # additive: one plain-language sentence built by rule from the signal numbers, e.g.
    # "69 Telegram accounts posted near-identical text, a median 15 s apart; 53 were
    # created within 3 days of each other." No LLM.
    summary: str = ""


class Edge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(alias="from")
    to: str
    type: Literal["direct", "inferred"]
    basis: str  # e.g. "forward", "quote", "url", or "similarity=0.91, +3h"
    relation: str | None = None  # additive: quote|reply|repost|forward|link for direct
    similarity: float | None = None  # additive: for inferred
    lag_seconds: float | None = None  # additive


class Hub(BaseModel):
    """additive: a post many direct edges start from (e.g. a video and its comments)."""

    post_id: str
    direct_out: int  # direct edges whose `from` is this post
    relations: dict[str, int] = Field(default_factory=dict)  # relation -> count
    inferred_out: int = 0


class OriginTrace(BaseModel):
    narrative_id: str
    earliest_post_ids: list[str]
    edges: list[Edge]
    platform_first_seen: dict[str, str] = Field(default_factory=dict)  # additive
    platform_sequence: list[str] = Field(default_factory=list)  # additive
    note: str = "Earliest observed in collected data; not necessarily the true origin."
    # additive: how well the earliest post matches the narrative (cosine to the narrative's
    # centroid). "weak" = it only loosely matches, so the analyst should look at
    # first_strong_post_id instead. None = not computed (too few posts / no text).
    earliest_match: Literal["strong", "weak"] | None = None
    earliest_similarity: float | None = None
    # additive: earliest post (by verified platform time) whose match is strong; equals
    # earliest_post_ids[0] when that one is strong; None when no member is strong.
    first_strong_post_id: str | None = None
    hubs: list[Hub] = Field(default_factory=list)  # additive: top posts by direct_out
    # additive (fixer, readable-analysis milestone): per platform, the time of the first
    # member post that closely matches the narrative (same bar as first_strong_post_id).
    # Lets the GUI/brief show the cross-platform path without anchoring it on a loose
    # match. Empty = not computed (match not judged); a platform with no close match is
    # absent.
    platform_first_seen_strong: dict[str, str] = Field(default_factory=dict)


Stance = Literal["assert", "report", "question", "debunk"]
STANCES: tuple[str, ...] = ("assert", "report", "question", "debunk")


class Evidence(BaseModel):
    url: str
    source: str  # google_factcheck | serpapi | serper | official | collected_post
    supports: Literal["supports", "contradicts", "context"]
    note: str = ""
    title: str | None = None  # additive
    publisher: str | None = None  # additive
    retrieved_at: str | None = None  # additive
    # additive: the item's own publication date as the provider gave it (published_raw)
    # and parsed to UTC (published_at: "YYYY-MM-DD" or ISO datetime "...Z"; None when the
    # provider gave nothing parseable — never guessed). Used to keep evidence about an
    # earlier, different event from counting as support (claims.py date rule).
    published_at: str | None = None
    published_raw: str | None = None
    # additive: why this item cannot count as support even if a model said so.
    # "old_event" = published before the support window (claims.py date rule);
    # "unrelated" = shares none of the claim's salient terms (a search miss).
    flag: Literal["old_event", "unrelated"] | None = None


class Claim(BaseModel):
    id: str
    text: str
    narrative_id: str | None = None  # additive
    related_post_ids: list[str]
    stances: dict[str, int] = Field(default_factory=lambda: {s: 0 for s in STANCES})
    stance_post_ids: dict[str, list[str]] = Field(  # additive: evidence per stance
        default_factory=lambda: {s: [] for s in STANCES}
    )
    evidence: list[Evidence] = Field(default_factory=list)
    verdict: Literal["supported", "contradicted", "disputed", "unverified"] = "unverified"
    confidence: float = 0.0
    rationale: str = ""
    method: str = ""  # additive: "llm:<model>" | "heuristic"
    # additive: ONE sentence (<= 220 chars) an analyst reads first: what the cited evidence
    # shows. Written by the verdict LLM call (same call, no extra cost) or by rule.
    bottom_line: str = ""
    # additive: pipeline notes moved out of `rationale` ("[Method: ...]", keyword fallback,
    # posts without verified time, model verdict overridden ...). Shown collapsed.
    method_notes: str = ""


# ---------------------------------------------------------------------------- brief
# additive file brief.json (owner decision 2026-11-19): the run summary page. Built by
# rule from the other analysis files (analyze/brief.py); no LLM, so every number traces.


class NarrativeRank(BaseModel):
    narrative_id: str
    rank: int  # 1 = most important
    score: float  # activity score (size, platforms, surge, coordination, claims); the
    # order is by severity first (analyze/brief.py), so the score explains, not decides
    main: bool  # shown in the main list (else collapsed as smaller / off-topic)
    tags: list[str] = Field(default_factory=list)
    # tags: severity:<critical|high|medium|low> | threat | off_topic | coordinated |
    # disputed_claim | contradicted_claim | supported_claim | surging | multi_platform
    reasons: list[str] = Field(default_factory=list)  # short human phrases behind the score
    cluster_ids: list[str] = Field(default_factory=list)  # clusters whose posts are members
    claim_ids: list[str] = Field(default_factory=list)
    peak_day: str | None = None
    peak_count: int = 0
    # additive (2026-11-19): the narrative's severity, copied here so the ranking is
    # self-explanatory in brief.json; None for older runs / unassessed narratives.
    severity: Literal["critical", "high", "medium", "low"] | None = None


class Finding(BaseModel):
    id: str  # F1..F5
    kind: Literal["narrative", "coordination", "claims", "coverage"]
    headline: str  # one sentence with its numbers
    detail: str = ""
    narrative_ids: list[str] = Field(default_factory=list)
    cluster_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    post_ids: list[str] = Field(default_factory=list)  # evidence behind the sentence


class CoverageNote(BaseModel):
    source: str
    state: str  # worst SourceState across that source's queries
    days_with_posts: int
    days_total: int
    note: str  # e.g. "newest day only; older days not reached"


class Brief(BaseModel):
    narrative_ranks: list[NarrativeRank] = Field(default_factory=list)  # all, by rank
    main_cluster_ids: list[str] = Field(default_factory=list)  # strongest first
    cluster_narratives: dict[str, list[str]] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    coverage: list[CoverageNote] = Field(default_factory=list)
    verdict_counts: dict[str, int] = Field(default_factory=dict)


class RunMeta(BaseModel):
    run_id: str
    topic: str
    start: str
    end: str
    sources: list[str]
    created_at: str
    finished_at: str | None = None
    status: Literal["running", "complete", "failed"] = "running"
    versions: dict[str, str] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    costs: dict[str, float] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    options: dict = Field(default_factory=dict)
    replay: bool = False
    error: str | None = None


def dump(model: BaseModel) -> dict:
    """Canonical JSON-ready dict (aliases on, so Edge.from_ serializes as 'from')."""
    return model.model_dump(mode="json", by_alias=True)
