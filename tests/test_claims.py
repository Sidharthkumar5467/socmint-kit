"""Tests for socmint.analyze.claims and socmint.analyze.evidence. No network.

The provider payloads below are DOCUMENTATION EXAMPLES, not live responses:
- DOC_EXAMPLE_FACTCHECK follows the Claim/ClaimReview/Publisher shape in
  developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims
- DOC_EXAMPLE_SERPAPI_NEWS follows the sample on serpapi.com/google-news-api
URLs and values are placeholders.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from conftest import make_post  # tests/ is on sys.path (no package __init__)

from socmint.analyze import claims as claims_mod
from socmint.analyze import evidence as ev_mod
from socmint.analyze.claims import (
    ExtractedClaims,
    StanceBatch,
    VerdictOut,
    assess_claims,
    heuristic_claim_text,
    heuristic_stance,
    map_rating,
    verdict_rules,
)
from socmint.llm import LLMError
from socmint.schema import Evidence, Narrative

DOC_EXAMPLE_FACTCHECK = {
    "claims": [
        {
            "text": "Example claim text",
            "claimant": "Example claimant",
            "claimDate": "2026-11-01T00:00:00Z",
            "claimReview": [
                {
                    "publisher": {"name": "Example Checker", "site": "checker.example"},
                    "url": "https://checker.example/review-1",
                    "title": "Review one",
                    "reviewDate": "2026-11-02T00:00:00Z",
                    "textualRating": "False",
                    "languageCode": "en",
                },
                {
                    "publisher": {"name": "Other Checker", "site": "other.example"},
                    "url": "https://other.example/review-2",
                    "title": "Review two",
                    "reviewDate": "2026-11-03T00:00:00Z",
                    "textualRating": "Misleading",
                    "languageCode": "en",
                },
            ],
        }
    ],
    "nextPageToken": "",
}

DOC_EXAMPLE_SERPAPI_NEWS = {
    "news_results": [
        {
            "position": 1,
            "title": "Example headline",
            "source": {"name": "Example Publisher", "icon": "https://x.example/i.png"},
            "link": "https://news.example/article",
            "date": "01/02/2026, 10:30 PM, +0700 +07",
            "iso_date": "2026-01-02T15:30:13Z",
        }
    ]
}

CFG_ALL = {"GOOGLE_FACTCHECK_API_KEY": "fc-test-key-123456", "SERPAPI_KEY": "serp-test-key-1234"}


# ----------------------------------------------------------------- fake LLM


class FakeLLM:
    """Returns canned structured outputs per schema; records prompts."""

    name = "llm:fake-model"
    model = "fake-model"

    def __init__(self, *, extract=None, stance=None, verdict=None, fail=()):
        self.extract, self.stance, self.verdict, self.fail = extract, stance, verdict, set(fail)
        self.prompts: list[tuple[str, str]] = []

    def parse(self, schema, prompt, *, system="", purpose="", max_tokens=4096):
        self.prompts.append((purpose, prompt))
        if schema is ExtractedClaims:
            if "extract" in self.fail:
                raise LLMError("boom")
            return ExtractedClaims.model_validate({"claims": [{"text": t} for t in self.extract]})
        if schema is StanceBatch:
            if "stance" in self.fail:
                raise LLMError("boom")
            ids = [line.split("post_id=")[1].split(" |")[0]
                   for line in prompt.splitlines() if line.startswith("- post_id=")]
            items = [it for it in self.stance(ids)]
            return StanceBatch.model_validate({"items": items})
        if schema is VerdictOut:
            if "verdict" in self.fail:
                raise LLMError("boom")
            return VerdictOut.model_validate({"bottom_line": "", **self.verdict})
        raise AssertionError(schema)


def _narr(nid="N1", members=(), summary='Representative post: "Dam collapsed in town X"'):
    return Narrative(id=nid, label="dam collapse", claim_summary=summary,
                     member_post_ids=list(members), example_post_ids=list(members)[:2])


def _posts():
    return [
        make_post("1", "The dam in town X collapsed this morning, hundreds displaced"),
        make_post("2", "FAKE: the dam did not collapse, this video is from 2019"),
        make_post("3", "Is it true the dam collapsed? anyone in town X"),
        make_post("4", "Local media reportedly say the dam collapsed"),
        make_post("5", "Fact check: no evidence the dam failed"),
    ]


# ----------------------------------------------------------- heuristic stance


@pytest.mark.parametrize("text,stance", [
    ("This is a hoax spread by bots", "debunk"),
    ("Fact check: no evidence for this", "debunk"),
    ("هذا الخبر كاذب", "debunk"),
    ("Did this really happen", "question"),
    ("Is that so?", "question"),
    ("هل صحيح هذا الخبر", "question"),
    ("According to state TV, the bridge fell", "report"),
    ("Officials allegedly hid the numbers", "report"),
    ("The bridge fell at 3pm", "assert"),
    ("Bridge fell https://site.example/a?b=1", "assert"),  # '?' in URL is not a question
])
def test_heuristic_stance(text, stance):
    assert heuristic_stance(text)[0] == stance


def test_heuristic_claim_text_strips_wrapper():
    assert heuristic_claim_text(_narr()) == "Dam collapsed in town X"
    assert heuristic_claim_text(_narr(summary="Plain summary")) == "Plain summary"


# ----------------------------------------------------------- rating mapping


@pytest.mark.parametrize("rating,rel", [
    ("False", "contradicts"), ("Pants on Fire", "contradicts"), ("Fake", "contradicts"),
    ("Incorrect", "contradicts"), ("Misleading", "contradicts"), ("Not true", "contradicts"),
    ("Falso", "contradicts"),
    ("True", "supports"), ("Correct", "supports"), ("Accurate", "supports"),
    ("Mostly true", "supports"),
    ("Half true", "disputed"), ("Mixture", "disputed"), ("Partly false", "disputed"),
    ("Unproven", "disputed"),
    ("", None), ("Satire label pending review", "contradicts"), ("Explainer", None),
])
def test_map_rating(rating, rel):
    assert map_rating(rating) == rel


@pytest.mark.parametrize("rating,rel", [
    # Negated support words are debunks, never support (reviewer finding).
    ("Not accurate", "contradicts"), ("Not correct", "contradicts"),
    ("This isn't true.", "contradicts"), ("That's not correct.", "contradicts"),
    ("This isn’t true", "contradicts"), ("This is not true", "contradicts"),
    ("Pas vrai", "contradicts"), ("C'est pas vrai", "contradicts"),
    ("No es cierto", "contradicts"), ("No es verdad", "contradicts"),
    ("Nicht richtig", "contradicts"), ("Não é verdade", "contradicts"),
    ("ليس صحيحا", "contradicts"), ("ليست صحيحة", "contradicts"), ("غير دقيق", "contradicts"),
    ("Неправда", "contradicts"),
    # Could not be established -> same bucket as "unproven", never support.
    ("Not verified", "disputed"), ("Unverified", "disputed"), ("Not proven", "disputed"),
    # Partly right -> disputed.
    ("Not entirely true", "disputed"), ("Not quite accurate", "disputed"),
    # Negated contradiction is ambiguous -> not mappable.
    ("Not false", None),
])
def test_map_rating_negations_never_support(rating, rel):
    assert map_rating(rating) == rel
    assert map_rating(rating) != "supports"


def test_rules_negated_rating_is_not_support():
    ev = [Evidence(url="https://ff.example/r", source="google_factcheck", supports="context",
                   note="rating: This isn't true.")]
    v, _, _, ev2, *_ = verdict_rules({}, ev, ["google_factcheck"])
    assert v == "contradicted" and ev2[0].supports == "contradicts"


def test_rules_majority_from_factcheck_ratings():
    ev = ev_mod.parse_factcheck(DOC_EXAMPLE_FACTCHECK, "2026-11-18T00:00:00Z")
    v, c, r, ev2, *_ = verdict_rules({"assert": 3}, ev, ["google_factcheck"])
    assert v == "contradicted"
    assert c == 0.5
    assert all(e.supports == "contradicts" for e in ev2)
    assert "rating" in r.lower()
    # Input evidence is not mutated (providers always return context).
    assert all(e.supports == "context" for e in ev)


def test_rules_single_rating_is_thin_and_ties_are_disputed():
    one = [Evidence(url="https://a.example", source="google_factcheck", supports="context",
                    note="rating: True")]
    v, c, _, _, *_ = verdict_rules({}, one, ["google_factcheck"])
    assert v == "supported" and c < 0.5
    two = one + [Evidence(url="https://b.example", source="google_factcheck",
                          supports="context", note="rating: False")]
    v, c, _, _, *_ = verdict_rules({}, two, ["google_factcheck"])
    assert v == "disputed" and c < 0.5


def test_rules_news_only_is_unverified():
    ev = ev_mod.parse_serpapi_news(DOC_EXAMPLE_SERPAPI_NEWS, "2026-11-18T00:00:00Z")
    v, c, r, _, *_ = verdict_rules({}, ev, ["serpapi"])
    assert v == "unverified" and c == 0.0 and "context only" in r


# --------------------------------------------------- no-evidence end to end


def test_no_evidence_unverified_with_explanation(tmp_path):
    posts = _posts()
    n = _narr(members=[p.post_id for p in posts])
    out = assess_claims(posts, [n], topic="dam", llm=None, cfg={}, run_dir=tmp_path)
    assert len(out) == 1
    c = out[0]
    assert c.id == "CL1" and c.narrative_id == "N1" and c.method == "heuristic"
    assert c.text == "Dam collapsed in town X"
    assert c.verdict == "unverified" and c.confidence == 0.0
    assert c.rationale.startswith("No external evidence was retrieved for this claim")
    assert "debunking/reporting posts are not support" in c.rationale
    assert c.evidence == []
    assert c.verdict not in ("true", "false")


def test_debunk_and_report_posts_never_count_as_assert(tmp_path):
    posts = _posts()
    n = _narr(members=[p.post_id for p in posts])
    c = assess_claims(posts, [n], topic="dam", llm=None, cfg={}, run_dir=tmp_path)[0]
    assert c.stance_post_ids["assert"] == ["x:1"]
    assert set(c.stance_post_ids["debunk"]) == {"x:2", "x:5"}
    assert c.stance_post_ids["question"] == ["x:3"]
    assert c.stance_post_ids["report"] == ["x:4"]
    assert c.stances == {"assert": 1, "report": 1, "question": 1, "debunk": 2}
    assert sorted(c.related_post_ids) == sorted(p.post_id for p in posts)
    # Collected debunking posts are never external evidence.
    assert all(e.source != "collected_post" for e in c.evidence)


def _supporting_rating():
    # Dated inside the posts' window (make_post default 2026-11-01) so the date rule
    # lets it count as support.
    return [Evidence(url="https://checker.example/true-1", source="google_factcheck",
                     supports="context", note="rating: True", published_at="2026-11-02",
                     published_raw="2026-11-02T00:00:00Z")]


def _patch_evidence(monkeypatch, fn):
    """Replace the evidence lookup: fn(query) -> list[Evidence]. No network."""
    def many(queries, cfg, run_dir, *, deadline=None, parallel=None):
        return [(list(fn(q)), []) for q in queries]
    monkeypatch.setattr(ev_mod, "gather_evidence_many", many)


def test_debunk_posts_not_support_even_with_supporting_rating(tmp_path, monkeypatch):
    """Verdict and evidence relations come only from the external rating; collected
    debunking posts are not evidence, and stance counts never move the verdict."""
    posts = [make_post("1", "This is fake, debunked already"),
             make_post("2", "Hoax, no evidence")]
    n = _narr(members=["x:1", "x:2"])
    _patch_evidence(monkeypatch, lambda q: _supporting_rating())
    monkeypatch.setattr(ev_mod, "available_providers", lambda cfg: ["google_factcheck"])
    c = assess_claims(posts, [n], topic="t", llm=None, cfg=CFG_ALL, run_dir=tmp_path)[0]
    assert c.stances["assert"] == 0 and c.stances["debunk"] == 2
    # The rating (not the posts) decides; the debunking posts are not evidence.
    assert c.verdict == "supported"
    assert [(e.url, e.supports) for e in c.evidence] == [
        ("https://checker.example/true-1", "supports")]
    assert all(e.url not in {p.url for p in posts} for e in c.evidence)
    # Changing the stance distribution does not change the rule verdict.
    base = verdict_rules(c.stances, _supporting_rating(), ["google_factcheck"])
    for counts in ({"assert": 50}, {"debunk": 50}, {"report": 9, "question": 9}, {}):
        v, conf, _, ev2, *_ = verdict_rules(counts, _supporting_rating(), ["google_factcheck"])
        assert (v, conf, [e.supports for e in ev2]) == (base[0], base[1],
                                                        [e.supports for e in base[3]])


def test_llm_verdict_ignores_stance_counts(monkeypatch):
    """Same LLM output and evidence -> same verdict whatever the stance counts."""
    ev = _supporting_rating()
    out = {"verdict": "supported", "confidence": 0.9, "rationale": "r",
           "cited": [{"url": "https://checker.example/true-1", "relation": "supports"}]}
    results = set()
    for counts in ({"debunk": 40}, {"assert": 40}, {}):
        v, c, _, ev2, *_ = claims_mod.verdict_llm("c", counts, ev, FakeLLM(verdict=out), "p")
        results.add((v, c, tuple(e.supports for e in ev2)))
    assert results == {("supported", 0.5, ("supports",))}


def test_unverified_and_out_of_window_posts_not_counted(tmp_path):
    posts = _posts() + [
        make_post("6", "The dam collapsed, I saw it", published_at=None, verified=False),
        make_post("7", "The dam collapsed again", in_window=False),
    ]
    n = _narr(members=[p.post_id for p in posts])
    c = assess_claims(posts, [n], topic="dam", llm=None, cfg={}, run_dir=tmp_path)[0]
    assert sum(c.stances.values()) == 5 and c.stances["assert"] == 1
    assert "x:6" not in c.related_post_ids and "x:7" not in c.related_post_ids
    assert all("x:6" not in v and "x:7" not in v for v in c.stance_post_ids.values())
    assert "2 narrative post(s) without a verified in-window timestamp" in c.method_notes
    assert "x:6" in c.method_notes and "x:7" in c.method_notes
    assert "x:6" not in c.rationale and "[" not in c.rationale


def test_narrative_without_countable_text_posts_is_skipped(tmp_path):
    posts = _posts() + [
        make_post("m1", ""),  # media-only
        make_post("m2", "   "),
        make_post("u1", "text but no timestamp", published_at=None, verified=False),
    ]
    good = _narr(nid="N1", members=["x:1", "x:2"])
    empty = _narr(nid="N2", members=["x:m1", "x:m2", "x:u1", "x:missing"],
                  summary="media only")
    out = assess_claims(posts, [good, empty], topic="t", llm=None, cfg={}, run_dir=tmp_path)
    assert [c.narrative_id for c in out] == ["N1"]
    assert all(c.related_post_ids for c in out)


def test_identical_claim_text_queried_once(tmp_path, monkeypatch):
    calls = []

    def fake_gather(q):
        calls.append(q)
        return _supporting_rating()

    _patch_evidence(monkeypatch, fake_gather)
    monkeypatch.setattr(ev_mod, "available_providers", lambda cfg: ["google_factcheck"])
    posts = _posts()
    same = 'Representative post: "Dam collapsed in town X"'
    narrs = [_narr(nid="N1", members=["x:1", "x:2"], summary=same),
             _narr(nid="N2", members=["x:3", "x:4"], summary="  dam collapsed in TOWN X ")]
    out = assess_claims(posts, narrs, topic="t", llm=None, cfg=CFG_ALL, run_dir=tmp_path)
    assert len(out) == 2 and len(calls) == 1
    assert all(c.verdict == "supported" for c in out)
    # Each claim got its own evidence copies (mutating one never leaks into the other).
    assert out[0].evidence[0] is not out[1].evidence[0]


# ----------------------------------------------------------------- LLM path


def _stance_all(stance):
    return lambda ids: [{"post_id": i, "stance": stance, "rationale": "r"} for i in ids]


def test_llm_cited_url_not_retrieved_is_dropped(tmp_path, monkeypatch):
    retrieved = [Evidence(url="https://checker.example/review-1", source="google_factcheck",
                          supports="context", note="rating: False")]
    _patch_evidence(monkeypatch, lambda q: list(retrieved))
    monkeypatch.setattr(ev_mod, "available_providers", lambda cfg: ["google_factcheck"])
    llm = FakeLLM(
        extract=["The dam collapsed on 1 Nov"],
        stance=_stance_all("assert"),
        verdict={"verdict": "contradicted", "confidence": 0.9, "rationale": "per invented",
                 "cited": [{"url": "https://invented.example/x", "relation": "contradicts"}]},
    )
    posts = _posts()
    c = assess_claims(posts, [_narr(members=[p.post_id for p in posts])], topic="dam",
                      llm=llm, cfg=CFG_ALL, run_dir=tmp_path)[0]
    assert c.verdict == "unverified" and c.confidence == 0.0
    assert all(e.url != "https://invented.example/x" for e in c.evidence)
    assert all(e.supports == "context" for e in c.evidence)
    assert "not in the retrieved evidence" in c.method_notes
    assert "not in the retrieved evidence" not in c.rationale
    assert c.method == "llm:fake-model"


def test_llm_valid_citation_marks_relation_and_caps_confidence(tmp_path, monkeypatch):
    retrieved = [
        Evidence(url="https://checker.example/review-1", source="google_factcheck",
                 supports="context", note="rating: False"),
        Evidence(url="https://news.example/a", source="serpapi", supports="context"),
    ]
    _patch_evidence(monkeypatch, lambda q: list(retrieved))
    monkeypatch.setattr(ev_mod, "available_providers", lambda cfg: ["google_factcheck"])
    llm = FakeLLM(
        extract=["The dam collapsed on 1 Nov"],
        stance=_stance_all("report"),
        verdict={"verdict": "contradicted", "confidence": 0.95, "rationale": "Checker says no",
                 "cited": [{"url": "https://checker.example/review-1",
                            "relation": "contradicts"}]},
    )
    posts = _posts()
    c = assess_claims(posts, [_narr(members=[p.post_id for p in posts])], topic="dam",
                      llm=llm, cfg=CFG_ALL, run_dir=tmp_path)[0]
    assert c.verdict == "contradicted"
    assert c.confidence == 0.5  # one citation -> thin evidence cap
    rel = {e.url: e.supports for e in c.evidence}
    assert rel == {"https://checker.example/review-1": "contradicts",
                   "https://news.example/a": "context"}
    assert c.stances["report"] == 5 and c.stances["assert"] == 0


def test_llm_verdict_inconsistent_with_relations_is_unverified(tmp_path, monkeypatch):
    retrieved = [Evidence(url="https://news.example/a", source="serpapi", supports="context")]
    _patch_evidence(monkeypatch, lambda q: list(retrieved))
    monkeypatch.setattr(ev_mod, "available_providers", lambda cfg: ["serpapi"])
    llm = FakeLLM(extract=["c"], stance=_stance_all("assert"),
                  verdict={"verdict": "supported", "confidence": 0.8, "rationale": "x",
                           "cited": [{"url": "https://news.example/a", "relation": "context"}]})
    c = assess_claims(_posts(), [_narr(members=["x:1"])], topic="t", llm=llm, cfg=CFG_ALL,
                      run_dir=tmp_path)[0]
    assert c.verdict == "unverified"


def test_llm_stance_drops_unknown_ids_and_falls_back(tmp_path):
    # Model returns a hallucinated id plus one real one; others fall back to keywords.
    llm = FakeLLM(extract=["c1", "c2", "c3", "c4"],
                  stance=lambda ids: [{"post_id": "x:999", "stance": "assert", "rationale": "r"},
                                      {"post_id": ids[0], "stance": "debunk", "rationale": "r"}])
    posts = _posts()
    out = assess_claims(posts, [_narr(members=[p.post_id for p in posts])], topic="t",
                        llm=llm, cfg={}, run_dir=tmp_path)
    assert len(out) == 3  # max 3 claims per narrative
    for c in out:
        all_ids = [i for v in c.stance_post_ids.values() for i in v]
        assert "x:999" not in all_ids
        assert sorted(all_ids) == sorted(p.post_id for p in posts)
        assert sum(c.stances.values()) == len(posts)
        assert c.verdict == "unverified"  # no evidence configured
    assert [c.id for c in out] == ["CL1", "CL2", "CL3"]


def test_llm_failures_fall_back_to_heuristic(tmp_path):
    llm = FakeLLM(fail={"extract", "stance", "verdict"})
    posts = _posts()
    c = assess_claims(posts, [_narr(members=[p.post_id for p in posts])], topic="t",
                      llm=llm, cfg={}, run_dir=tmp_path)[0]
    assert c.method == "heuristic"
    assert c.text == "Dam collapsed in town X"
    assert c.stances["debunk"] == 2


def test_narrative_cap_and_order(tmp_path):
    posts = [make_post(str(i), f"post {i}") for i in range(20)]
    narrs = [_narr(nid=f"N{k}", members=[f"x:{j}" for j in range(k + 1)], summary=f"s{k}")
             for k in range(10)]
    out = assess_claims(posts, narrs, topic="t", llm=None, cfg={}, run_dir=tmp_path)
    assert len(out) == claims_mod.MAX_NARRATIVES
    assert out[0].narrative_id == "N9"  # largest first


# -------------------------------------------------------------- providers


@respx.mock
def test_factcheck_provider_parses_and_logs_without_key(tmp_path):
    route = respx.get(ev_mod.FACTCHECK_URL).mock(
        return_value=httpx.Response(200, json=DOC_EXAMPLE_FACTCHECK))
    ev = ev_mod.google_factcheck("dam collapse", CFG_ALL, tmp_path)
    assert route.called
    assert route.calls[0].request.url.params["query"] == "dam collapse"
    assert [e.url for e in ev] == ["https://checker.example/review-1",
                                  "https://other.example/review-2"]
    e = ev[0]
    assert e.source == "google_factcheck" and e.supports == "context"
    assert e.publisher == "Example Checker" and e.title == "Review one"
    assert "rating: False" in e.note and e.retrieved_at
    raw_files = list((tmp_path / "raw" / "evidence").glob("*.json"))
    assert len(raw_files) == 1
    blob = raw_files[0].read_text()
    assert "fc-test-key-123456" not in blob
    log_lines = (tmp_path / "evidence_log.jsonl").read_text().splitlines()
    rec = json.loads(log_lines[0])
    assert rec["provider"] == "google_factcheck" and len(rec["results"]) == 2
    assert "fc-test-key-123456" not in (tmp_path / "evidence_log.jsonl").read_text()


@respx.mock
def test_repeated_query_keeps_every_raw_response(tmp_path):
    respx.get(ev_mod.FACTCHECK_URL).mock(side_effect=[
        httpx.Response(200, json=DOC_EXAMPLE_FACTCHECK),
        httpx.Response(403, text="denied"),
    ])
    assert len(ev_mod.google_factcheck("same q", CFG_ALL, tmp_path)) == 2
    assert ev_mod.google_factcheck("same q", CFG_ALL, tmp_path) == []
    lines = [json.loads(x) for x in (tmp_path / "evidence_log.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and lines[0]["raw_ref"] != lines[1]["raw_ref"]
    first = json.loads((tmp_path / lines[0]["raw_ref"]).read_text())
    assert first["error"] is None and first["response"] == DOC_EXAMPLE_FACTCHECK
    second = json.loads((tmp_path / lines[1]["raw_ref"]).read_text())
    assert second["error"]


@respx.mock
def test_raw_response_is_not_altered_by_generic_redaction(tmp_path):
    payload = json.loads(json.dumps(DOC_EXAMPLE_FACTCHECK))
    payload["nextPageToken"] = "CAoQabcdef123456"
    payload["claims"][0]["claimReview"][0]["title"] = "Key: vaccines and token = chips"
    # A provider echoing the configured key is still scrubbed (literal match only).
    payload["claims"][0]["claimant"] = "echo fc-test-key-123456"
    respx.get(ev_mod.FACTCHECK_URL).mock(return_value=httpx.Response(200, json=payload))
    ev_mod.google_factcheck("q", CFG_ALL, tmp_path)
    blob = next((tmp_path / "raw" / "evidence").glob("*.json")).read_text()
    raw = json.loads(blob)["response"]
    assert raw["nextPageToken"] == "CAoQabcdef123456"
    assert raw["claims"][0]["claimReview"][0]["title"] == "Key: vaccines and token = chips"
    assert "fc-test-key-123456" not in blob


@respx.mock
def test_serpapi_provider_top5(tmp_path):
    many = {"news_results": [
        {"title": f"t{i}", "link": f"https://n.example/{i}", "source": {"name": "P"}}
        for i in range(8)
    ] + [{"highlight": {"title": "h", "link": "https://n.example/h"}}]}
    respx.get(ev_mod.SERPAPI_URL).mock(return_value=httpx.Response(200, json=many))
    ev = ev_mod.serpapi_news("q", CFG_ALL, tmp_path)
    assert len(ev) == 5 and all(e.source == "serpapi" and e.supports == "context" for e in ev)
    doc = ev_mod.parse_serpapi_news(DOC_EXAMPLE_SERPAPI_NEWS, "now")
    assert doc[0].publisher == "Example Publisher" and doc[0].url == "https://news.example/article"


@respx.mock
def test_provider_failure_returns_empty_not_raise(tmp_path):
    respx.get(ev_mod.FACTCHECK_URL).mock(return_value=httpx.Response(403, text="denied"))
    respx.get(ev_mod.SERPAPI_URL).mock(side_effect=httpx.ConnectError("down"))
    assert ev_mod.google_factcheck("q", CFG_ALL, tmp_path) == []
    assert ev_mod.serpapi_news("q", CFG_ALL, tmp_path, deadline=0.0) == []
    lines = [json.loads(x) for x in
             (tmp_path / "evidence_log.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and all(rec["error"] for rec in lines)


@respx.mock
def test_provider_malformed_json_returns_empty(tmp_path):
    respx.get(ev_mod.FACTCHECK_URL).mock(return_value=httpx.Response(200, text="<html>"))
    assert ev_mod.google_factcheck("q", CFG_ALL, tmp_path) == []
    assert ev_mod.parse_factcheck(["not", "a", "dict"], "now") == []


def test_providers_skip_without_keys(tmp_path):
    assert ev_mod.google_factcheck("q", {}, tmp_path) == []
    assert ev_mod.serpapi_news("q", {}, tmp_path) == []
    assert ev_mod.available_providers({}) == []
    assert not (tmp_path / "evidence_log.jsonl").exists()


@respx.mock
def test_assess_end_to_end_with_factcheck_rules(tmp_path):
    respx.get(ev_mod.FACTCHECK_URL).mock(
        return_value=httpx.Response(200, json=DOC_EXAMPLE_FACTCHECK))
    posts = _posts()
    cfg = {"GOOGLE_FACTCHECK_API_KEY": "fc-test-key-123456"}
    c = assess_claims(posts, [_narr(members=[p.post_id for p in posts])], topic="dam",
                      llm=None, cfg=cfg, run_dir=tmp_path)[0]
    assert c.verdict == "contradicted" and c.confidence == 0.5
    assert {e.url for e in c.evidence} == {"https://checker.example/review-1",
                                           "https://other.example/review-2"}
    assert c.method == "heuristic"


# ----------------------------------------------------- serper.dev (live response fixture)
SERPER_LIVE = (Path(__file__).parent / "fixtures" / "evidence" /
               "live_serper_news_2026-11-19.json")


def test_parse_serper_live_response():
    payload = json.loads(SERPER_LIVE.read_text(encoding="utf-8"))
    ev = ev_mod.parse_serper_news(payload, "2026-11-19T12:00:00Z")
    assert len(ev) == 5
    first = payload["news"][0]
    assert ev[0].url == first["link"] and ev[0].title == first["title"]
    assert ev[0].publisher == first["source"] and ev[0].source == "serper"
    assert all(e.supports == "context" for e in ev)  # providers never decide support
    assert f"published: {first['date']}" in ev[0].note


@respx.mock
def test_serper_provider_posts_json_with_header_and_never_logs_key(tmp_path):
    payload = json.loads(SERPER_LIVE.read_text(encoding="utf-8"))
    route = respx.post(ev_mod.SERPER_NEWS_URL).mock(return_value=httpx.Response(200, json=payload))
    cfg = {"SERPER_API_KEY": "serper-test-key-0123456789abcdef"}
    ev = ev_mod.serper_news("chase outage", cfg, tmp_path)
    assert len(ev) == 5 and route.called
    req = route.calls[0].request
    assert req.headers["X-API-KEY"] == cfg["SERPER_API_KEY"]
    assert json.loads(req.content) == {"q": "chase outage", "num": 10}
    blob = "".join(p.read_text() for p in tmp_path.rglob("*") if p.is_file())
    assert "serper-test-key-0123456789abcdef" not in blob
    assert "serper" in ev_mod.available_providers(cfg)


@respx.mock
def test_serper_failure_returns_empty(tmp_path):
    respx.post(ev_mod.SERPER_NEWS_URL).mock(return_value=httpx.Response(403, text="bad key"))
    assert ev_mod.serper_news("q", {"SERPER_API_KEY": "k" * 40}, tmp_path) == []


# ------------------------------------------------------------ evidence dates


@pytest.mark.parametrize("raw,expected", [
    ("7 minutes ago", "2026-11-19T13:49:59Z"),
    ("50 minutes ago", "2026-11-19T13:06:59Z"),
    ("2 hours ago", "2026-11-19T11:56:59Z"),
    ("20 hours ago", "2026-11-18T17:56:59Z"),
    ("1 day ago", "2026-11-18"),
    ("3 days ago", "2026-11-16"),
    ("1 week ago", "2026-11-12"),
    ("3 weeks ago", "2026-10-29"),
    ("1 month ago", "2026-10-20"),
    ("2 months ago", "2026-09-20"),
    ("1 year ago", "2025-11-19"),
    ("Oct 2, 2024", "2024-10-02"),
    ("2 Oct 2024", "2024-10-02"),
    ("2024-10-02", "2024-10-02"),
    ("Sep 16, 2026", "2026-09-16"),
    ("2026-11-02T00:00:00Z", "2026-11-02T00:00:00Z"),
    ("01/02/2026, 10:30 PM, +0700 +07", "2026-01-02T15:30:00Z"),  # SerpApi `date`
    ("yesterday-ish", None), ("", None), (None, None), ("Feb 30, 2024", None),
    ("2026-11-02T00:00:00", None),  # no timezone: never guessed
])
def test_parse_evidence_date(raw, expected):
    assert ev_mod.parse_evidence_date(raw, "2026-11-19T13:56:59Z") == expected


def test_relative_date_without_retrieved_at_is_unknown():
    assert ev_mod.parse_evidence_date("2 days ago", None) is None


def test_parsers_carry_published_dates():
    payload = json.loads(SERPER_LIVE.read_text(encoding="utf-8"))
    ev = ev_mod.parse_serper_news(payload, "2026-11-19T12:00:00Z")
    assert ev[0].published_raw == "2 days ago" and ev[0].published_at == "2026-11-17"
    fc = ev_mod.parse_factcheck(DOC_EXAMPLE_FACTCHECK, "2026-11-18T00:00:00Z")
    assert fc[0].published_raw == "2026-11-02T00:00:00Z"
    assert fc[0].published_at == "2026-11-02T00:00:00Z"
    assert "claim date: 2026-11-01T00:00:00Z" in fc[0].note
    sa = ev_mod.parse_serpapi_news(DOC_EXAMPLE_SERPAPI_NEWS, "2026-11-18T00:00:00Z")
    assert sa[0].published_at == "2026-01-02T15:30:13Z"


# Real serper.dev items (imageUrl removed) for claim CL2 of
# runs/2026-11-19-bank-of-america-8b5c96 (raw/evidence/serper_45323d666c_001.json,
# retrieved 2026-11-19T13:56:59Z). The real verdict was `supported` 0.7 citing the two
# Oct 2024 articles about a different glitch (docs/EVALUATION.md false positive).
CL2_RETRIEVED_AT = "2026-11-19T13:56:59Z"
CL2_SERPER = {"news": [
    {"title": "Bank of America, Chase and others are reporting outages across the country "
              "according to Downdetector. See link below",
     "link": "https://www.facebook.com/detroitfreepress/posts/bank-of-america-chase-and-"
             "others-are-reporting-outages-across-the-country-accord/1618892463616233/",
     "date": "2 days ago", "source": "Facebook"},
    {"title": "Bank of America outage: Glitch caused some customers to see $0 balances in "
              "their accounts",
     "link": "https://abc11.com/post/bank-america-is-down-customers-report-widespread-"
             "outage/15385210/",
     "date": "Oct 2, 2024", "source": "ABC11 News"},
    {"title": "Bank System Outages: Chase, Wells Fargo, Capital One Services Impacted",
     "link": "https://www.newsweek.com/are-chase-wells-fargo-down-latest-update-on-bank-"
             "outages-12722872",
     "date": "2 days ago", "source": "Newsweek"},
    {"title": "Bank of America is down: Users report their accounts showing empty balance "
              "during widespread outage",
     "link": "https://www.independent.co.uk/news/world/americas/bank-of-america-outage-"
             "down-b2622878.html",
     "date": "Oct 2, 2024", "source": "The Independent"},
    {"title": "Boston 25 News. . First Microsoft...then a nationwide ground stop...Now online "
              "banking disruptions",
     "link": "https://www.facebook.com/Boston25News/videos/when-tech-fails-banks-and-travel-"
             "grind-to-a-halt/1396454572389239/",
     "date": "1 day ago", "source": "Facebook"},
]}
CL2_TEXT = ("Bank of America customers woke up to $0 balances after a widespread outage, with "
            "thousands reporting empty accounts in the app and online banking.")
CL2_WINDOW = claims_mod.make_window("2026-11-13", "2026-11-19")


def _cl2_llm_out():
    n = CL2_SERPER["news"]
    return {"verdict": "supported", "confidence": 0.7,
            "rationale": "ABC11 (Oct 2, 2024) reports a glitch with $0 balances ...",
            "cited": [{"url": n[0]["link"], "relation": "context"},
                      {"url": n[1]["link"], "relation": "supports"},
                      {"url": n[2]["link"], "relation": "context"},
                      {"url": n[3]["link"], "relation": "supports"},
                      {"url": n[4]["link"], "relation": "context"}]}


def test_cl2_old_articles_cannot_support_current_claim():
    ev = ev_mod.parse_serper_news(CL2_SERPER, CL2_RETRIEVED_AT)
    assert [e.published_at for e in ev] == ["2026-11-17", "2024-10-02", "2026-11-17",
                                            "2024-10-02", "2026-11-18"]
    llm = FakeLLM(verdict=_cl2_llm_out())
    v, c, r, ev2, bl, mn = claims_mod.verdict_llm(CL2_TEXT, {"assert": 4}, ev, llm, "p", CL2_WINDOW)
    assert v == "unverified" and c <= 0.3
    assert all(e.supports == "context" for e in ev2)
    old = [e for e in ev2 if e.published_at == "2024-10-02"]
    assert len(old) == 2
    assert all("published 2024-10-02, outside the claim's time window" in e.note for e in old)
    assert "Date check" in mn[0] and "2024-10-02" in mn[0] and "Date check" not in r
    assert [e.flag for e in ev2] == [None, "old_event", None, "old_event", None]
    assert "$0" not in bl and len(bl) <= 220
    # The model saw the window and each item's date.
    prompt = llm.prompts[0][1]
    assert "2026-11-13..2026-11-19" in prompt
    assert "published=2024-10-02 (given as 'Oct 2, 2024')" in prompt
    assert "published=2026-11-17 (given as '2 days ago')" in prompt


def test_cl2_end_to_end_with_run_window(tmp_path, monkeypatch):
    _patch_evidence(monkeypatch,
                    lambda q: ev_mod.parse_serper_news(CL2_SERPER, CL2_RETRIEVED_AT))
    monkeypatch.setattr(ev_mod, "available_providers", lambda cfg: ["serper"])
    posts = [make_post("1", "Woke up to $0 in my BofA account", published_at="2026-11-16T10:00:00Z")]
    llm = FakeLLM(extract=[CL2_TEXT], stance=_stance_all("assert"), verdict=_cl2_llm_out())
    c = assess_claims(posts, [_narr(members=["x:1"])], topic="bank of america", llm=llm,
                      cfg={"SERPER_API_KEY": "k" * 32}, run_dir=tmp_path,
                      window_start="2026-11-13", window_end="2026-11-19")[0]
    assert c.verdict == "unverified"
    assert "2024-10-02" in c.method_notes and "Date check" in c.method_notes
    assert sum(e.flag == "old_event" for e in c.evidence) == 2
    # The model said "supported"; code overrode it, so the rule sentence is used.
    assert c.bottom_line == ("5 retrieved items give context but none confirms or "
                             "contradicts it; 2 items describe an earlier event (from "
                             "Oct 2024).")
    assert not any(e.supports == "supports" for e in c.evidence)


def test_in_window_support_still_counts():
    ev = ev_mod.parse_serper_news(CL2_SERPER, CL2_RETRIEVED_AT)
    n = CL2_SERPER["news"]
    out = {"verdict": "supported", "confidence": 0.9, "rationale": "r",
           "cited": [{"url": n[0]["link"], "relation": "supports"},
                     {"url": n[2]["link"], "relation": "supports"}]}
    v, c, _, ev2, *_ = claims_mod.verdict_llm("c", {}, ev, FakeLLM(verdict=out), "p", CL2_WINDOW)
    assert v == "supported" and c == 0.7
    assert [e.supports for e in ev2] == ["supports", "context", "supports", "context",
                                         "context"]


def test_support_window_edges():
    w = claims_mod.make_window("2026-11-13", "2026-11-19")

    def one(d):
        e = Evidence(url="u", source="serper", supports="supports", published_at=d)
        return claims_mod.apply_date_rule([e], w)[0][0].supports

    assert one("2026-10-14") == "supports"   # start - 30 days
    assert one("2026-10-13") == "context"
    assert one("2026-11-26T23:00:00Z") == "supports"  # end + 7 days
    assert one("2026-11-27") == "context"


def _fc(url, rating, review_date):
    return ev_mod.parse_factcheck({"claims": [{"text": "x", "claimReview": [
        {"url": url, "publisher": {"name": url}, "textualRating": rating,
         "reviewDate": review_date}]}]}, CL2_RETRIEVED_AT)


def test_old_debunk_still_contradicts_with_dated_rationale_rules():
    ev = _fc("https://checker.example/old", "False", "2020-03-01T00:00:00Z")
    v, c, r, ev2, *_ = verdict_rules({}, ev, ["google_factcheck"], window=CL2_WINDOW)
    assert v == "contradicted" and ev2[0].supports == "contradicts"
    assert "published 2020-03-01, outside the claim window 2026-11-13..2026-11-19" in r


def test_old_debunk_still_contradicts_with_dated_rationale_llm():
    ev = _fc("https://checker.example/old", "False", "2020-03-01T00:00:00Z")
    out = {"verdict": "contradicted", "confidence": 0.8, "rationale": "Checker says false",
           "cited": [{"url": "https://checker.example/old", "relation": "contradicts"}]}
    v, c, r, ev2, *_ = claims_mod.verdict_llm("c", {}, ev, FakeLLM(verdict=out), "p", CL2_WINDOW)
    assert v == "contradicted" and c == 0.5 and ev2[0].supports == "contradicts"
    assert "published 2020-03-01" in r


def test_old_true_rating_not_counted_rules():
    ev = _fc("https://checker.example/old-true", "True", "2024-10-03T00:00:00Z")
    v, c, r, ev2, *_ = verdict_rules({}, ev, ["google_factcheck"], window=CL2_WINDOW)
    assert v == "unverified" and c == 0.0 and ev2[0].supports == "context"
    assert "not counted" in r and "2024-10-03" in r
    assert "outside the claim's time window" in ev2[0].note


def test_unknown_date_support_alone_is_context():
    ev = [Evidence(url="https://n.example/a", source="serper", supports="context",
                   published_raw="sometime")]
    out = {"verdict": "supported", "confidence": 0.9, "rationale": "r",
           "cited": [{"url": "https://n.example/a", "relation": "supports"}]}
    v, _, r, ev2, _, mn = claims_mod.verdict_llm("c", {}, ev, FakeLLM(verdict=out), "p", CL2_WINDOW)
    assert v == "unverified" and ev2[0].supports == "context"
    assert "date unknown" in " ".join(mn) and "publication date unknown" in ev2[0].note
    # Rule-based path, undated True rating: not counted.
    fc = [Evidence(url="https://c.example/t", source="google_factcheck", supports="context",
                   note="rating: True")]
    v, _, r, ev2, *_ = verdict_rules({}, fc, ["google_factcheck"], window=CL2_WINDOW)
    assert v == "unverified" and ev2[0].supports == "context" and "not counted" in r


def test_unknown_date_support_alongside_dated_in_window_support():
    ev = [Evidence(url="https://n.example/a", source="serper", supports="context"),
          Evidence(url="https://n.example/b", source="serper", supports="context",
                   published_at="2026-11-17")]
    out = {"verdict": "supported", "confidence": 0.9, "rationale": "r",
           "cited": [{"url": "https://n.example/a", "relation": "supports"},
                     {"url": "https://n.example/b", "relation": "supports"}]}
    v, c, _, ev2, *_ = claims_mod.verdict_llm("c", {}, ev, FakeLLM(verdict=out), "p", CL2_WINDOW)
    assert v == "supported" and c == 0.7
    assert [e.supports for e in ev2] == ["supports", "supports"]


def test_window_falls_back_to_posts_when_not_passed():
    posts = [make_post("1", "a", published_at="2026-11-14T10:00:00Z"),
             make_post("2", "b", published_at="2026-11-18T10:00:00Z"),
             make_post("3", "c", published_at=None, verified=False)]
    assert claims_mod.window_from_posts(posts) == claims_mod.make_window("2026-11-14",
                                                                         "2026-11-18")
    assert claims_mod.make_window(None, "2026-11-18") is None


# Review regressions (adversarial review of the evidence-date change).


def test_bare_numbers_are_not_read_as_epoch():
    # "2024" used to become 1970-01-01T00:33:44Z via the epoch fallback: a guess.
    for raw in ("2024", "20241002", "1763500000"):
        assert ev_mod.parse_evidence_date(raw, CL2_RETRIEVED_AT) is None


def test_extreme_dates_never_raise_and_never_drop_the_provider_response():
    for raw in ("999999 years ago", "99999999999999999999"):
        assert ev_mod.parse_evidence_date(raw, CL2_RETRIEVED_AT) is None
    payload = {"news": [{"title": "t", "link": "https://n.example/x", "date": "999999 years ago",
                         "source": "S"},
                        {"title": "t2", "link": "https://n.example/y", "date": "2 days ago",
                         "source": "S"}]}
    ev = ev_mod.parse_serper_news(payload, CL2_RETRIEVED_AT)
    assert [e.published_at for e in ev] == [None, "2026-11-17"]


def test_month_names_must_be_real_months():
    assert ev_mod.parse_evidence_date("Decade 2, 2024", None) is None
    assert ev_mod.parse_evidence_date("2 Marching 2024", None) is None
    assert ev_mod.parse_evidence_date("September 3, 2024", None) == "2024-09-03"
    assert ev_mod.parse_evidence_date("3 Sept 2024", None) == "2024-09-03"
    assert ev_mod.parse_evidence_date("Oct. 2, 2024", None) == "2024-10-02"


def test_contradicts_dated_before_window_but_inside_lookback_is_named():
    # Outside the window (2026-11-13..19) but inside the 30-day support lookback: the
    # rationale must still state the item's date.
    ev = _fc("https://checker.example/recent", "False", "2026-11-01T00:00:00Z")
    v, _, r, ev2, *_ = verdict_rules({}, ev, ["google_factcheck"], window=CL2_WINDOW)
    assert v == "contradicted" and ev2[0].supports == "contradicts"
    assert "published 2026-11-01, outside the claim window 2026-11-13..2026-11-19" in r
    # An in-window debunk gets no such note.
    ev = _fc("https://checker.example/now", "False", "2026-11-15T00:00:00Z")
    _, _, r, _, *_ = verdict_rules({}, ev, ["google_factcheck"], window=CL2_WINDOW)
    assert "outside the claim window" not in r


# ----------------------------------------------------- concurrent evidence providers


def _fake_providers(monkeypatch, log, delay=0.0, barrier=None):
    import threading
    import time as _time

    def make(name):
        def fn(q, cfg, run_dir, *, deadline=None):
            if barrier is not None:
                barrier.wait(timeout=5)  # proves the providers run at the same time
            log.append(("start", name, q, threading.get_ident()))
            _time.sleep(delay)
            log.append(("end", name, q, threading.get_ident()))
            shared = Evidence(url="https://shared.example/x", source=name, supports="context")
            return [Evidence(url=f"https://{name}.example/{q}", source=name,
                             supports="context"), shared]
        return fn

    names = ["google_factcheck", "serpapi", "serper"]
    monkeypatch.setattr(ev_mod, "PROVIDERS", {n: make(n) for n in names})
    return {"GOOGLE_FACTCHECK_API_KEY": "k" * 20, "SERPAPI_KEY": "k" * 20,
            "SERPER_API_KEY": "k" * 20}


def test_providers_concurrent_but_each_sequential(monkeypatch):
    import threading
    log: list = []
    cfg = _fake_providers(monkeypatch, log, delay=0.01, barrier=threading.Barrier(3))
    queries = ["q1", "q2", "q3"]
    # A broken barrier (providers not concurrent) raises inside the provider, which the
    # gatherer turns into an empty result; so non-empty output proves concurrency.
    out = ev_mod.gather_evidence_many(queries[:1], cfg, None, parallel=True)
    assert len(out[0][0]) == 4
    log.clear()
    cfg = _fake_providers(monkeypatch, log, delay=0.01)
    ev_mod.gather_evidence_many(queries, cfg, None, parallel=True)
    for name in ("google_factcheck", "serpapi", "serper"):
        mine = [(k, q) for k, n, q, _ in log if n == name]
        # Strictly sequential per provider, in query order: start/end alternate.
        assert mine == [(k, q) for q in queries for k in ("start", "end")]
        assert len({t for _, n, _, t in log if n == name}) == 1  # one worker thread


def test_parallel_output_identical_to_sequential(monkeypatch):
    log: list = []
    cfg = _fake_providers(monkeypatch, log)
    qs = ["a", "b", "c", "d"]
    par = ev_mod.gather_evidence_many(qs, cfg, None, parallel=True)
    seq = ev_mod.gather_evidence_many(qs, cfg, None, parallel=False)
    dump = [([e.model_dump() for e in ev], sk) for ev, sk in par]
    assert dump == [([e.model_dump() for e in ev], sk) for ev, sk in seq]
    # Merged in PROVIDERS order, de-duplicated by URL.
    assert [e.url for e in par[0][0]] == ["https://google_factcheck.example/a",
                                          "https://shared.example/x",
                                          "https://serpapi.example/a",
                                          "https://serper.example/a"]
    assert ev_mod.gather_evidence("a", cfg, None) == par[0][0]


def test_evidence_budget_respected(monkeypatch):
    import time as _time
    log: list = []
    cfg = _fake_providers(monkeypatch, log)
    out = ev_mod.gather_evidence_many(["a", "b"], cfg, None,
                                      deadline=_time.monotonic() - 1)
    assert log == [] and all(ev == [] and len(sk) == 3 for ev, sk in out)
    # Budget runs out part-way: each provider stops issuing new calls.
    cfg = _fake_providers(monkeypatch, log, delay=0.15)
    out = ev_mod.gather_evidence_many(["a", "b", "c"], cfg, None,
                                      deadline=_time.monotonic() + 0.1)
    assert out[0][1] == [] and out[0][0]
    assert out[1][1] == ["google_factcheck", "serpapi", "serper"]
    assert {q for _, _, q, _ in log} == {"a"}


def test_assess_claims_reports_budget_skip(tmp_path, monkeypatch):
    log: list = []
    cfg = _fake_providers(monkeypatch, log)
    monkeypatch.setattr(claims_mod, "EVIDENCE_BUDGET_S", -1.0)
    posts = _posts()
    c = assess_claims(posts, [_narr(members=[p.post_id for p in posts])], topic="dam",
                      llm=None, cfg=cfg, run_dir=tmp_path)[0]
    assert log == [] and c.verdict == "unverified"
    assert "time budget for evidence exhausted" in c.method_notes
    assert c.bottom_line.startswith("No evidence search ran for this claim")


def test_llm_workers_raised():
    assert claims_mod.LLM_WORKERS == 8


# ------------------------------------- readability: flags, bottom line, method notes

# the pre-re-analysis sample (older format: method text in rationales, CL3 NFL item)
SAMPLE_RUN = Path(__file__).resolve().parent / "fixtures" / "legacy_run"


def _sample_claims():
    from socmint.schema import Claim
    return [Claim.model_validate(c)
            for c in json.loads((SAMPLE_RUN / "claims.json").read_text(encoding="utf-8"))]


def _news(url, title, date="2026-11-17", publisher="Pub", note=None):
    return Evidence(url=url, source="serper", supports="context", title=title,
                    publisher=publisher, published_at=date,
                    note=note or "news search result; published: 2 days ago")


def test_sample_cl3_nfl_article_flagged_unrelated():
    cl3 = next(c for c in _sample_claims() if c.id == "CL3")
    assert "Panthers" in (cl3.evidence[0].title or "")
    b = claims_mod.backfill_claim(cl3, "bank of america", CL2_WINDOW)
    assert [e.flag for e in b.evidence] == ["unrelated"]
    assert b.evidence[0].supports == "context"
    assert b.bottom_line.startswith("No relevant evidence was found")
    assert "Relevance check" in b.method_notes
    # The input object is never mutated (sample data stays as read).
    assert cl3.evidence[0].flag is None


def test_sample_cl2_old_articles_flagged_old_event():
    cl2 = next(c for c in _sample_claims() if c.id == "CL2")
    b = claims_mod.backfill_claim(cl2, "bank of america", CL2_WINDOW)
    flags = {(e.published_at or "")[:10]: e.flag for e in b.evidence}
    assert flags == {"2026-11-17": None, "2024-10-02": "old_event"}
    assert "earlier event (from Oct 2024)" in b.bottom_line


def test_sample_related_items_not_flagged():
    # Conservative: a sibling-confirmed item ("JPMorgan says software bug ...", while
    # another item says "Chase ... software bug") and items that mention the topic stay.
    for c in _sample_claims():
        b = claims_mod.backfill_claim(c, "bank of america", CL2_WINDOW)
        for e in b.evidence:
            if e.title and ("JPMorgan" in e.title or "Bank of America" in e.title):
                assert e.flag != "unrelated", (c.id, e.title)


def test_unrelated_rules():
    claim = "Bank of America sent a warning to customers on the evening of the posted date."
    nfl = _news("https://n.example/nfl", "Panthers get more national notice: The 2026 NFL "
                "schedule is out")
    assert claims_mod.is_unrelated(claim, nfl, "bank of america")
    # Stem overlap (warning / warns) -> related.
    warn = _news("https://n.example/w", "Lender warns clients about phishing texts tonight")
    assert not claims_mod.is_unrelated(claim, warn, "bank of america")
    # Mentions the topic itself -> not judged unrelated.
    topic = _news("https://n.example/t", "Banks report record quarterly earnings this year")
    assert not claims_mod.is_unrelated(claim, topic, "bank of america")
    # Too little text to judge.
    assert not claims_mod.is_unrelated(claim, _news("https://n.example/s", "Panthers win"))
    # Acronym: FAA vs "Federal Aviation Administration".
    assert not claims_mod.is_unrelated(
        "The FAA grounded all departures on Friday.",
        _news("https://n.example/f", "Federal Aviation Administration lifts ground stop after "
              "equipment issue"))


def test_unrelated_not_flagged_across_languages():
    claim = "Bank of America sent a warning to customers on the evening of the posted date."
    es = _news("https://n.example/es", "El banco envió una alerta a sus clientes por la noche")
    ar = _news("https://n.example/ar", "البنك يرسل تحذيرا إلى العملاء في المساء")
    assert not claims_mod.is_unrelated(claim, es, "bank of america")
    assert not claims_mod.is_unrelated(claim, ar, "bank of america")
    # Unspaced-script claims are never judged.
    zh = "美国银行在当天晚上向客户发出了警告"
    assert not claims_mod.is_unrelated(zh, _news("https://n.example/z", "Panthers get more "
                                                 "national notice: The NFL schedule is out"))


def test_unrelated_sibling_overlap_keeps_item():
    claim = "Major US banks reported online banking outages on Monday."
    ev = [_news("https://n.example/a", "Chase blames outages on software bug"),
          _news("https://n.example/b", "JPMorgan says software bug slows its transactions"),
          _news("https://n.example/c", "Panthers get more national notice: the NFL schedule "
                "is out")]
    assert claims_mod.unrelated_flags(claim, ev, "") == [False, False, True]


def test_unrelated_item_never_support_llm_path():
    nfl = _news("https://n.example/nfl", "Panthers get more national notice: The 2026 NFL "
                "schedule is out")
    out = {"verdict": "supported", "confidence": 0.9, "rationale": "r",
           "bottom_line": "The NFL article confirms the warning.",
           "cited": [{"url": nfl.url, "relation": "supports"}]}
    llm = FakeLLM(verdict=out)
    res = claims_mod.verdict_llm("Bank of America sent a warning to customers.", {}, [nfl],
                                 llm, "p", CL2_WINDOW, topic="bank of america")
    assert res.verdict == "unverified" and res.evidence[0].supports == "context"
    assert res.evidence[0].flag == "unrelated"
    assert "FLAG: shares no key term" in llm.prompts[0][1]
    # The model's sentence relied on the demoted item, so the rule sentence replaces it.
    assert "NFL" not in res.bottom_line
    assert res.bottom_line.startswith("No relevant evidence was found")


def test_unrelated_item_never_support_rules_path():
    fc = Evidence(url="https://c.example/r", source="google_factcheck", supports="context",
                  title="Panthers get more national notice: the NFL schedule is out",
                  note="rating: True", published_at="2026-11-15")
    res = verdict_rules({}, [fc], ["google_factcheck"], window=CL2_WINDOW,
                        claim_text="Bank of America sent a warning to customers.",
                        topic="bank of america")
    assert res.verdict == "unverified" and res.evidence[0].supports == "context"
    assert res.evidence[0].flag == "unrelated"
    # Without the claim text there is nothing to compare: the rating counts as before.
    res = verdict_rules({}, [fc], ["google_factcheck"], window=CL2_WINDOW)
    assert res.verdict == "supported"


def test_old_event_flag_rules():
    w = CL2_WINDOW
    old_ctx = Evidence(url="u1", source="serper", supports="context", published_at="2024-10-02")
    old_sup = Evidence(url="u2", source="serper", supports="supports", published_at="2024-10-02")
    old_con = Evidence(url="u3", source="serper", supports="contradicts",
                       published_at="2020-01-01")
    late = Evidence(url="u4", source="serper", supports="supports", published_at="2027-01-01")
    inside = Evidence(url="u5", source="serper", supports="context", published_at="2026-11-01")
    undated = Evidence(url="u6", source="serper", supports="context")
    out, notes, method = claims_mod.apply_date_rule(
        [old_ctx, old_sup, old_con, late, inside, undated], w)
    assert [e.flag for e in out] == ["old_event", "old_event", None, None, None, None]
    assert out[1].supports == "context" and out[2].supports == "contradicts"
    assert notes and "2020-01-01" in notes[0]  # analysis note stays in the rationale
    assert method and method[0].startswith("Date check: 2 item(s)")
    # No window: nothing flagged, a method note says dates were not checked.
    out, notes, method = claims_mod.apply_date_rule([old_sup], None)
    assert out[0].flag is None and not notes and "window is unknown" in method[0]


def test_bottom_line_from_model_kept_and_cleaned():
    ev = [_news("https://n.example/a", "Banks hit by outage", publisher="Reuters")]
    line = ("Reuters [1] reports the outage on 17 Nov (https://n.example/a) " + "x " * 200)
    out = {"verdict": "supported", "confidence": 0.9, "rationale": "r", "bottom_line": line,
           "cited": [{"url": "https://n.example/a", "relation": "supports"}]}
    res = claims_mod.verdict_llm("Banks had an outage.", {}, ev, FakeLLM(verdict=out), "p",
                                 CL2_WINDOW)
    assert res.verdict == "supported"
    assert res.bottom_line.startswith("Reuters reports the outage on 17 Nov")
    assert "http" not in res.bottom_line and "[1]" not in res.bottom_line
    assert len(res.bottom_line) <= claims_mod.BOTTOM_LINE_MAX
    assert "bottom_line" in VerdictOut.model_json_schema()["required"]


def test_bottom_line_rule_when_model_verdict_overridden():
    ev = [_news("https://n.example/a", "Banks hit by outage", publisher="Reuters")]
    out = {"verdict": "supported", "confidence": 0.9, "rationale": "r",
           "bottom_line": "Reuters confirms it.",
           "cited": [{"url": "https://n.example/a", "relation": "context"}]}
    res = claims_mod.verdict_llm("Banks had an outage.", {"assert": 3, "debunk": 1}, ev,
                                 FakeLLM(verdict=out), "p", CL2_WINDOW)
    assert res.verdict == "unverified"
    assert res.bottom_line == "1 retrieved item gives context but none confirms or contradicts it."
    assert any("Model verdict 'supported'" in m for m in res.method_notes)
    assert "Model verdict" not in res.rationale


def test_rule_bottom_line_variants():
    rb = claims_mod.rule_bottom_line
    counts = {"assert": 20, "report": 10, "question": 5, "debunk": 5}
    assert rb("unverified", [], counts) == ("No news or fact-check evidence was found; 20 of "
                                            "40 classified posts assert it.")
    assert rb("unverified", [], {}, searched=False) == "No evidence search ran for this claim."
    sup = [Evidence(url=f"https://www.n{i}.example/a", source="serper", supports="supports",
                    publisher=p) for i, p in enumerate(["Reuters", "CBS", "AP"])]
    con = Evidence(url="https://c.example/r", source="google_factcheck",
                   supports="contradicts", publisher="Checker")
    assert rb("disputed", [*sup, con], counts) == ("3 sources (Reuters, CBS and others) "
                                                   "support it; Checker contradicts it.")
    host = Evidence(url="https://www.tokenpost.com/x", source="serper", supports="supports")
    assert rb("supported", [host], {}) == "tokenpost.com supports it."
    assert len(rb("supported", sup * 40, counts)) <= claims_mod.BOTTOM_LINE_MAX
    # an older run's model verdict 'disputed' with only supporting items (sample CL1):
    # the sentence must not read as plain support
    assert rb("disputed", [host], {}) == ("tokenpost.com supports it, but this was not "
                                          "judged enough to call it supported (disputed).")


def test_assess_claims_rationale_is_analysis_only(tmp_path):
    posts = _posts() + [make_post("6", "The dam collapsed", published_at=None, verified=False)]
    n = _narr(members=[p.post_id for p in posts])
    c = assess_claims(posts, [n], topic="dam", llm=None, cfg={}, run_dir=tmp_path)[0]
    assert "[" not in c.rationale and "Method:" not in c.rationale
    assert "GOOGLE_FACTCHECK_API_KEY" not in c.rationale
    assert c.method_notes.startswith("Method: extraction=heuristic; stance=heuristic; "
                                     "verdict=rule-based; 5 post(s) classified.")
    assert "No evidence provider is configured" in c.method_notes
    assert "1 narrative post(s) without a verified in-window timestamp" in c.method_notes
    assert c.bottom_line == "No evidence search ran for this claim; 1 of 5 classified posts assert it."


def test_split_method_notes_on_sample_rationales():
    for c in _sample_claims():
        text, notes = claims_mod.split_method_notes(c.rationale)
        assert "[Method:" not in text and "narrative post(s) without" not in text
        assert "Model verdict" not in text and "providers queried" not in text
        if "[Method:" in c.rationale:
            assert "Method: extraction=" in notes
    cl1 = next(c for c in _sample_claims() if c.id == "CL1")
    text, _ = claims_mod.split_method_notes(cl1.rationale)
    assert "Men's Journal [4]" in text  # citation markers stay
    cl2 = next(c for c in _sample_claims() if c.id == "CL2")
    _, notes = claims_mod.split_method_notes(cl2.rationale)
    assert "Model verdict 'disputed' was not backed" in notes
    cl4 = next(c for c in _sample_claims() if c.id == "CL4")
    text, notes = claims_mod.split_method_notes(cl4.rationale)
    assert text.startswith("No external evidence was retrieved for this claim; stance")
    assert "Providers queried (google_factcheck, serper) returned nothing." in notes


# ------------------------------------------------- key-specifics + consistency checks
# Built from the re-analysed run's CL6 (2026-11-19): four outlets on the 16 Nov outage,
# none of whose titles names Bank of America or Wells Fargo, yet rated supported 0.8.
CL6_TEXT = ("Bank of America suffered online banking outages on November 16, alongside "
            "JPMorgan Chase and Wells Fargo.")


def _ev(url, title, publisher, supports="context", published="2026-11-17"):
    return Evidence(url=url, source="serper", title=title, publisher=publisher,
                    supports=supports, note="news search result; published: 2 days ago",
                    published_at=published)


def _cl6_evidence():
    return [
        _ev("https://apnews.example/a", "A technical issue at JPMorgan Chase triggers online "
            "banking delays for major US banks", "AP News"),
        _ev("https://aljazeera.example/b", "Thousands report online outages at major US banks",
            "Al Jazeera"),
        _ev("https://tnr.example/c", "Yet Another System Crashes in Trump’s America",
            "The New Republic"),
    ]


def test_claim_specifics_names_figures_quotes():
    sp = claims_mod.claim_specifics
    assert sp(CL6_TEXT) == ["Bank of America", "JPMorgan Chase", "Wells Fargo"]
    assert sp("Bank of America disclosed holding 13,260 shares of the Volatility Shares XRP "
              "ETF in its Q3 SEC filing.") == ["Bank of America", "Volatility Shares XRP ETF",
                                               "Q3 SEC", "13,260"]
    assert sp("Goldman Sachs held about $144 million in XRP funds, up 80% since 2024.") == [
        "Goldman Sachs", "XRP", "$144 million", "80%"]
    assert sp("Bank of America sent an email titled 'Protecting your finances in the AI "
              "World'.")[0] == "“Protecting your finances in the AI World”"
    # sentence-initial plain word, leading article, dotted abbreviation, dates, small numbers
    assert sp("Customers say 12 branches closed on 3 Nov 2026.") == []
    assert sp("The FAA grounded U.S. flights on Monday, November 13 for 4 hours.") == ["FAA", "US"]


def test_missing_specifics_phrase_not_scattered_words():
    ev = [e.model_copy(update={"supports": "supports"}) for e in _cl6_evidence()]
    # "banking" (AP) + "America" (TNR) must not add up to "Bank of America"
    assert claims_mod.missing_specifics(CL6_TEXT, ev) == ["Bank of America", "Wells Fargo"]
    ok = ev + [_ev("https://bbc.example/d", "Customers report outages at Chase, Bank of "
                   "America and Wells Fargo", "BBC", supports="supports")]
    assert claims_mod.missing_specifics(CL6_TEXT, ok) == []
    # acronym and figure matching
    boa = [_ev("u", "BofA says Wells Fargo and JPMorgan Chase hit too", "X", "supports")]
    assert claims_mod.missing_specifics(CL6_TEXT, boa) == []
    fig = [_ev("u", "BofA holds 13,260 shares of Volatility Shares XRP ETF", "X", "supports")]
    assert claims_mod.missing_specifics(
        "Bank of America holds 13260 shares of the Volatility Shares XRP ETF.", fig) == []
    # context or unrelated items never count
    ctx = [e.model_copy(update={"supports": "context"}) for e in ok]
    assert claims_mod.missing_specifics(CL6_TEXT, ctx) == []


def test_enforce_support_rules():
    ev = [e.model_copy(update={"supports": "supports"}) for e in _cl6_evidence()]
    v, c, notes, missing = claims_mod.enforce_support(CL6_TEXT, "supported", 0.8, ev)
    assert (v, c, missing) == ("unverified", 0.3, ["Bank of America", "Wells Fargo"])
    assert notes[0].startswith("Specifics check: no supporting item names Bank of America or "
                               "Wells Fargo")
    # consistency: contradicted without a contradicting item
    v, c, notes, _ = claims_mod.enforce_support(CL6_TEXT, "contradicted", 0.7, ev)
    assert v == "unverified" and c == 0.3 and "matching relation" in notes[0]
    # disputed needs conflicting sources: support alone is not a dispute (see
    # test_disputed_requires_conflicting_sources)
    assert claims_mod.enforce_support(CL6_TEXT, "disputed", 0.5, ev)[0] == "unverified"
    assert claims_mod.enforce_support(CL6_TEXT, "unverified", 0.2, [])[:2] == ("unverified", 0.2)


def test_verdict_llm_demotes_support_missing_named_actor():
    ev = _cl6_evidence()
    out = {"verdict": "supported", "confidence": 0.8,
           "rationale": "AP, Al Jazeera and TNR report the 16 Nov outage.",
           "bottom_line": "Outlets report the outage; Bank of America is not named.",
           "cited": [{"url": e.url, "relation": "supports"} for e in ev]}
    llm = FakeLLM(verdict=out)
    v, c, r, ev2, bl, mn = claims_mod.verdict_llm(CL6_TEXT, {"assert": 3}, ev, llm, "p",
                                                  CL2_WINDOW, topic="bank of america")
    assert v == "unverified" and c <= 0.3
    assert any(m.startswith("Specifics check") for m in mn)
    assert "none names Bank of America or Wells Fargo" in bl and "unverified" in bl
    assert [e.supports for e in ev2] == ["supports"] * 3  # relations kept, verdict lowered
    assert "specific actor, figure and date" in llm.prompts[0][1]


def test_verdict_llm_keeps_support_naming_actor():
    ev = _cl6_evidence() + [_ev("https://bbc.example/d", "Customers report outages at Chase, "
                                "Bank of America and Wells Fargo", "BBC")]
    out = {"verdict": "supported", "confidence": 0.8, "rationale": "BBC names all three.",
           "bottom_line": "BBC reports outages at all three banks on 16 Nov.",
           "cited": [{"url": ev[0].url, "relation": "supports"},
                     {"url": ev[3].url, "relation": "supports"}]}
    v, c, _, _, bl, mn = claims_mod.verdict_llm(CL6_TEXT, {}, ev, FakeLLM(verdict=out), "p",
                                                CL2_WINDOW, topic="bank of america")
    assert v == "supported" and c == 0.7
    assert bl == "BBC reports outages at all three banks on 16 Nov."
    assert not any(m.startswith("Specifics check") for m in mn)


def test_backfill_lowers_supported_when_only_support_is_unrelated():
    """An older run's 'supported' claim whose one supporting item is an NFL article."""
    from socmint.schema import Claim

    nfl = Evidence(url="https://nfl.example/schedule", source="serper",
                   title="NFL Week 12 schedule: kickoff times, TV channels and streaming",
                   publisher="NFL.com", supports="supports",
                   note="news search result; snippet: Sunday games, kickoff times and odds "
                        "for every matchup this weekend",
                   published_at="2026-11-17")
    old = Claim(id="CL9", text="Bank of America customers saw duplicate debit charges "
                "posted overnight.", narrative_id="N1", related_post_ids=["x:1"],
                evidence=[nfl], verdict="supported", confidence=0.7,
                rationale="The cited item supports the claim.")
    b = claims_mod.backfill_claim(old, "bank of america", CL2_WINDOW)
    assert b.evidence[0].flag == "unrelated" and b.evidence[0].supports == "context"
    assert b.verdict == "unverified" and b.confidence <= 0.3
    assert ("Backfill: verdict supported lowered to unverified because its only support "
            "was flagged unrelated.") in b.method_notes
    assert "support" not in b.bottom_line.split(":")[0].lower() or "unrelated" in b.bottom_line
    # a claim that still passes keeps its recorded verdict, confidence and bottom line
    good = old.model_copy(update={"text": "Bank of America outage", "evidence": [
        _ev("https://bbc.example/d", "Bank of America outage hits customers", "BBC",
            "supports")], "bottom_line": "BBC reports it."})
    g = claims_mod.backfill_claim(good, "bank of america", CL2_WINDOW)
    assert (g.verdict, g.confidence, g.bottom_line) == ("supported", 0.7, "BBC reports it.")


# ------------------------------------------------ 2026-11-19 review: slug, claim text
def test_specifics_check_reads_the_evidence_url_slug():
    """CL6/CL11 in the re-analysed sample: no excerpt stored, the headline names only
    Chase, but the article URL slug names Bank of America and Wells Fargo."""
    ev = [_ev("https://www.cbsnews.example/news/banks-online-service-outages-jpmorgan-chase-"
              "bank-of-america-wells-fargo/", "Widespread service outages at major U.S. banks "
              "appear to be over", "CBS News", supports="supports")]
    assert claims_mod.missing_specifics(CL6_TEXT, ev) == []
    v, _c, notes, missing = claims_mod.enforce_support(CL6_TEXT, "supported", 0.7, ev)
    assert v == "supported" and not notes and not missing
    # a slug without the actors does not help; slug numbers are not figures
    bare = [_ev("https://news.example/2026/11/16/us-bank-outage-13260/", "Bank outage",
                "N", supports="supports")]
    assert claims_mod.missing_specifics(CL6_TEXT, bare) == ["Bank of America",
                                                            "JPMorgan Chase", "Wells Fargo"]
    assert "13260" not in claims_mod.url_slug_text(bare[0].url)
    assert claims_mod.url_slug_text(
        "https://apnews.example/article/bank-outage-chase-truist-50fee2693c22f597fa6ca51a97ef2cda"
    ) == "article bank outage chase truist"


@pytest.mark.parametrize("raw, clean", [
    ("Post [2] asserts that Bank of America offers up to three business cards under a "
     "single inquiry.", "Bank of America offers up to three business cards under a single "
     "inquiry."),
    ("Post [3] asserts the Premium Rewards Elite card has a $550 annual fee.",
     "The Premium Rewards Elite card has a $550 annual fee."),
    ("Posts [1] and [4] claim that the bank closed 12 branches.",
     "The bank closed 12 branches."),
    ("According to post 4, the outage lasted six hours.", "The outage lasted six hours."),
    ("[5] says Chase blamed a software bug", "Chase blamed a software bug"),
    ("The bank was fined $5m (post [2]).", "The bank was fined $5m."),
    ("Postal workers at the bank's mail centre went on strike.",
     "Postal workers at the bank's mail centre went on strike."),
])
def test_clean_claim_text_strips_prompt_numbering(raw, clean):
    assert claims_mod.clean_claim_text(raw) == clean


def test_extraction_prompt_forbids_post_numbers():
    class L:
        def __init__(self):
            self.prompt = ""

        def parse(self, schema, prompt, **kw):
            self.prompt = prompt
            return schema(claims=[{"text": "Post [2] asserts that Bank X offers a card."}])

    llm = L()
    n = Narrative(id="N1", label="l", claim_summary="s", member_post_ids=["x:1"],
                  example_post_ids=["x:1"])
    out = claims_mod._extract_llm(n, [make_post("1", "Bank X offers a card")], "bank x", llm)
    assert out == ["Bank X offers a card."]
    assert "never refer to the posts or their numbers" in llm.prompt


def test_demoted_rationale_drops_confidence_and_support_sentences():
    """CL6 (2026-11-20 review): the verdict was lowered but the stored rationale still
    said the claim was supported at 0.8."""
    out = {"verdict": "supported", "confidence": 0.8,
           "rationale": "AP, Al Jazeera and TNR report the 16 Nov outage. The claim is "
                        "supported by three outlets. Given three sources, confidence is "
                        "held at 0.8.",
           "bottom_line": "", "cited": [{"url": e.url, "relation": "supports"}
                                        for e in _cl6_evidence()]}
    v, _c, r, *_ = claims_mod.verdict_llm(CL6_TEXT, {}, _cl6_evidence(), FakeLLM(verdict=out),
                                          "p", CL2_WINDOW, topic="bank of america")
    assert v == "unverified"
    assert r == ("Lowered to unverified: no supporting item names Bank of America or Wells "
                 "Fargo. AP, Al Jazeera and TNR report the 16 Nov outage.")
    assert "0.8" not in r and "supported by" not in r
    assert claims_mod.demoted_rationale(r, "anything") == r  # idempotent
    # negated statements are kept
    assert claims_mod.demoted_rationale("Nothing confirms the figure.", "x") == \
        "Lowered to unverified: x. Nothing confirms the figure."


def test_verdict_llm_kept_verdict_keeps_rationale():
    ev = _cl6_evidence() + [_ev("https://bbc.example/d", "Customers report outages at Chase, "
                                "Bank of America and Wells Fargo", "BBC")]
    out = {"verdict": "supported", "confidence": 0.8, "rationale": "BBC names all three.",
           "bottom_line": "", "cited": [{"url": ev[0].url, "relation": "supports"},
                                        {"url": ev[3].url, "relation": "supports"}]}
    r = claims_mod.verdict_llm(CL6_TEXT, {}, ev, FakeLLM(verdict=out), "p", CL2_WINDOW,
                               topic="bank of america")[2]
    assert r == "BBC names all three."


def test_backfill_demotion_rewrites_rationale():
    from socmint.schema import Claim

    old = Claim(id="CL6", text=CL6_TEXT, narrative_id="N1", related_post_ids=["x:1"],
                evidence=[e.model_copy(update={"supports": "supports"})
                          for e in _cl6_evidence()],
                verdict="supported", confidence=0.8,
                rationale="Three outlets report the outage; confidence is held at 0.8.")
    b = claims_mod.backfill_claim(old, "bank of america", CL2_WINDOW)
    assert b.verdict == "unverified"
    assert b.rationale.startswith("Lowered to unverified: no supporting item names")
    assert "0.8" not in b.rationale


# ------------------------------------- 2026-11-20 tuning: key details + disputed rule
# Sample cases (data/sample_run, topic "bank of america") plus other kinds of topic, so
# the rules are not tuned to one dataset.
CL10_TEXT = ("Customers of Chase, Bank of America, Wells Fargo, U.S. Bank and Zelle reported "
             "being unable to log in or move money on Monday, November 16.")
CL24_TEXT = ("Officials say the recent incidents look unrelated and there is no evidence a "
             "rogue AI was behind them.")


@pytest.mark.parametrize("text, expected", [
    # sample CL10: "Customers of" is a generic head, not part of the name "Chase"
    (CL10_TEXT, ["Chase", "Bank of America", "Wells Fargo", "US Bank", "Zelle"]),
    # sample CL24: "AI" names a kind of technology, not an actor
    (CL24_TEXT, []),
    # election / public figure: generic group heads are dropped, the name stays
    ("Supporters of Donald Trump said Dominion machines flipped votes in Georgia.",
     ["Donald Trump", "Dominion", "Georgia"]),
    ("Thousands of Pfizer vaccine recipients reported heart problems, the CEO admitted.",
     ["Pfizer"]),
    # product recall: generic words and a generic acronym are not details
    ("Users report Tesla recalled 2 million EVs over Autopilot.",
     ["Tesla", "Autopilot", "2 million"]),
    # acronyms that are also stop words stay ("WHO", not "who")
    ("The WHO declared mpox a global emergency.", ["WHO"]),
    # a sentence-initial adjective before a name
    ("Partial Bank of America card numbers leaked on a forum.", ["Partial Bank of America"]),
])
def test_claim_specifics_generic_heads_and_acronyms(text, expected):
    assert claims_mod.claim_specifics(text) == expected


def test_partial_name_matches_without_sentence_initial_word():
    ev = [_ev("u", "Bank of America card data posted online", "X", "supports")]
    assert claims_mod.missing_specifics("Partial Bank of America card numbers leaked on a "
                                        "forum.", ev) == []


def test_acronym_matched_case_sensitively_on_word_boundaries():
    claim = "The WHO declared mpox a global emergency on Monday."
    # "who" as a pronoun, or inside another word, is not the acronym
    bad = [_ev("https://news.example/a", "Doctors who treat mpox say cases rise", "N",
               "supports")]
    assert claims_mod.missing_specifics(claim, bad) == ["WHO"]
    for title, url in [("WHO declares mpox emergency", "https://n.example/x"),
                       ("World Health Organization declares mpox emergency",
                        "https://n.example/x"),
                       ("Mpox emergency declared", "https://n.example/who-mpox-emergency")]:
        ok = [_ev(url, title, "N", "supports")]
        assert claims_mod.missing_specifics(claim, ok) == [], title
    # Italian "ai" (to the) is not "AI"; "FAA" spelled out matches
    assert claims_mod.missing_specifics(
        "The FAA grounded flights.", [_ev("u", "Federal Aviation Administration halts flights",
                                          "N", "supports")]) == []
    assert not claims_mod._acronym_in("AI", "Accesso ai conti bloccato", "")
    assert claims_mod._acronym_in("AI", "Officials see no link to Microsoft or AI", "")


def test_sample_cl10_one_unnamed_list_member_is_tolerated():
    """Sample CL10: CNN/Fortune/CBS/Yahoo name Chase, Bank of America, Wells Fargo and
    U.S. Bank but not Zelle. The old check demoted 'supported' for "Customers of Chase"
    (a glued span) and Zelle; now one member of a 5-name list may be unnamed."""
    ev = [_ev("https://www.cnn.com/2026/11/16/business/bank-outage-chase-us-bank-bank-of-"
              "america", "Thousands of Chase Bank customers report trouble with online "
              "banking services", "CNN", "supports"),
          _ev("https://y.example/x", "Thousands of Chase, Bank of America and Wells Fargo "
              "Customers Say They Can’t Move Their Money", "Yahoo Finance", "supports")]
    missing, partly = claims_mod.check_specifics(CL10_TEXT, ev, "bank of america")
    assert (missing, partly) == ([], ["Zelle"])
    v, c, notes, _ = claims_mod.enforce_support(CL10_TEXT, "supported", 0.8, ev,
                                                "bank of america")
    assert (v, c) == ("supported", 0.8)
    assert notes == ["Specifics check: the supporting items name the other actors the claim "
                     "lists but not Zelle; that part is unconfirmed."]
    # two members unnamed: not tolerated
    assert claims_mod.missing_specifics(CL10_TEXT, [_ev("u", "Chase and US Bank outage",
                                                        "N", "supports")]) == [
        "Bank of America", "Wells Fargo", "Zelle"]


def test_list_tolerance_never_excuses_the_topic():
    """Product recall (other topic): one of four brands may be unnamed, never the
    investigated one."""
    claim = "The recall covers Graco, Chicco, Evenflo and Britax car seats sold since 2024."
    ev = [_ev("u", "Chicco, Evenflo and Britax recall car seats", "Reuters", "supports")]
    assert claims_mod.check_specifics(claim, ev, "britax") == ([], ["Graco"])
    assert claims_mod.check_specifics(claim, ev, "graco") == (["Graco"], [])
    # two-item lists get no tolerance
    pair = "Pfizer and Moderna paused shipments to the EU."
    assert claims_mod.missing_specifics(pair, [_ev("u", "Pfizer pauses EU shipments", "N",
                                                   "supports")]) == ["Moderna"]


def test_sample_cl24_generic_acronym_not_required():
    """Sample CL24: the supporting WaPo / Boston Globe items do not say "AI" (GeekWire
    and Fortune do, but the model cited them as context, which the check never reads).
    "a rogue AI" is a kind of technology, not a named actor, so it is not required."""
    ev = [_ev("https://www.facebook.com/washingtonpost/videos/us-intelligence-officials-say-"
              "recent-service-disruptions-were-unrelated-and-not-/2435770527247749/",
              "U.S. intelligence officials say recent service disruptions were unrelated "
              "and not malicious.", "Washington Post", "supports"),
          _ev("https://www.geekwire.com/2026/dhs-chief-sees-no-microsoft-link-to-faa-ground-"
              "stop-says-possible-ai-role-being-investigated/", "Officials blame aging FAA "
              "telecom gear, see no link to Microsoft or AI", "GeekWire")]
    v, c, notes, missing = claims_mod.enforce_support(CL24_TEXT, "supported", 0.7, ev)
    assert (v, c, notes, missing) == ("supported", 0.7, [], [])
    # a named actor in the same position is still required (vaccine topic)
    claim = "Officials say there is no evidence Moderna hid trial deaths."
    v, *_ , missing = claims_mod.enforce_support(claim, "supported", 0.7, ev[:1])
    assert v == "unverified" and missing == ["Moderna"]


def _with(ev, supports):
    return [e.model_copy(update={"supports": s}) for e, s in zip(ev, supports, strict=True)]


def test_disputed_requires_conflicting_sources():
    """Sample CL2: 3 supporting + 2 context items, nothing contradicts, yet 'disputed'."""
    ev = _with(_cl6_evidence(), ["supports", "supports", "context"])
    v, c, notes, _ = claims_mod.enforce_support(CL6_TEXT, "disputed", 0.45, ev)
    assert (v, c) == ("unverified", 0.3)
    assert notes[0].startswith("Disputed check: 'disputed' needs a contradicting item")
    assert "2 supporting and 1 context item(s)" in notes[0]
    # one contradicting item: disputed stands
    con = _with(_cl6_evidence(), ["supports", "contradicts", "context"])
    assert claims_mod.enforce_support(CL6_TEXT, "disputed", 0.45, con)[:2] == ("disputed",
                                                                                0.45)
    # supporting items the model says disagree (election-count example)
    elec = [_ev("https://a.example/1", "County certifies 51% turnout", "AP", "supports"),
            _ev("https://b.example/2", "County turnout put at 38%", "Local", "supports"),
            _ev("https://c.example/3", "Election explainer", "Explainer")]
    claim = "Turnout in Maricopa County fell below 40% in the 2026 primary."
    assert claims_mod.enforce_support(claim, "disputed", 0.4, elec, "",
                                      [1, 2])[0] == "disputed"
    # a cited number that is not a supporting item, or only one item: not a conflict
    assert claims_mod.enforce_support(claim, "disputed", 0.4, elec, "", [1, 3])[0] == \
        "unverified"
    assert claims_mod.enforce_support(claim, "disputed", 0.4, elec, "", [1])[0] == \
        "unverified"
    # a fact-check rated "half true" is itself a statement that sources conflict
    fc = Evidence(url="https://fc.example/r", source="google_factcheck", supports="context",
                  note="rating: Half true", title="Vaccine claim review")
    assert claims_mod.enforce_support("The vaccine causes X.", "disputed", 0.4,
                                      [fc])[0] == "disputed"
    # a collected (debunking) post is never evidence, even when marked contradicts
    post = Evidence(url="https://x.com/a/status/1", source="collected_post",
                    supports="contradicts", title="This is FAKE, Pfizer never said that")
    assert claims_mod.enforce_support("Pfizer admitted the vaccine causes X.", "disputed",
                                      0.4, [post])[0] == "unverified"


def test_verdict_llm_disputed_without_conflict_lowered_with_explanation():
    ev = _with(_cl6_evidence(), ["context"] * 3)
    out = {"verdict": "disputed", "confidence": 0.45,
           "rationale": "AP reports the outage. The outage is corroborated. Nothing shows "
                        "the $0 balances.",
           "bottom_line": "Outlets report outages but not $0 balances.",
           "cited": [{"url": ev[0].url, "relation": "supports"},
                     {"url": ev[1].url, "relation": "context"}]}
    llm = FakeLLM(verdict=out)
    v, c, r, ev2, bl, mn = claims_mod.verdict_llm(CL6_TEXT, {"debunk": 5}, ev, llm, "p",
                                                  CL2_WINDOW, topic="bank of america")
    assert (v, c) == ("unverified", 0.3)
    assert any(m.startswith("Disputed check") for m in mn)
    assert r.startswith("Lowered to unverified: no retrieved source contradicts the claim")
    assert "corroborated" not in r
    assert "do not conflict" in bl and "not disputed" in bl
    assert "posts that debunk" in llm.prompts[0][1]
    # with conflicting supporting items stated by the model, disputed stands and the
    # conflict is recorded for later backfills
    out2 = dict(out, cited=[{"url": e.url, "relation": "supports"} for e in ev[:2]],
                conflicting_items=[1, 2])
    v, c, _r, _e, _bl, mn = claims_mod.verdict_llm(CL6_TEXT, {}, ev, FakeLLM(verdict=out2),
                                                   "p", CL2_WINDOW, topic="bank of america")
    assert v == "disputed" and c == 0.45
    note = next(m for m in mn if m.startswith("Conflict:"))
    assert claims_mod.conflicts_from_notes(note) == [1, 2]


def test_backfill_lowers_disputed_without_conflict():
    from socmint.schema import Claim

    old = Claim(id="CL2", text="Bank of America customers woke up to $0 balances after a "
                "widespread outage.", narrative_id="N1", related_post_ids=["x:1"],
                evidence=_with(_cl6_evidence(), ["supports", "supports", "context"]),
                verdict="disputed", confidence=0.45, rationale="Outlets report outages.")
    b = claims_mod.backfill_claim(old, "bank of america", CL2_WINDOW)
    assert b.verdict == "unverified" and b.confidence == 0.3
    assert "Backfill: verdict disputed lowered to unverified" in b.method_notes
    assert "not disputed" in b.bottom_line
    kept = old.model_copy(update={"method_notes": claims_mod.conflict_note([1, 2])})
    assert claims_mod.backfill_claim(kept, "bank of america", CL2_WINDOW).verdict == \
        "disputed"


# ------------------------------------- 2026-11-20 review: conflict rule + partly lists
def _cl24_evidence():
    """Live re-run CL24 shape: one supporting and one contradicting counted item."""
    return [
        _ev("https://www.facebook.com/washingtonpost/videos/officials-say-outages-unrelated",
            "US intelligence officials say recent service outages unrelated, not malicious",
            "Facebook"),
        _ev("https://www.bostonglobe.com/2026/11/18/business/network-outages/",
            "Network outages hit several US companies", "The Boston Globe"),
        _ev("https://fortune.com/2026/11/17/administration-ai-outages-banks",
            "White House official links outages to rogue AI agents", "Fortune"),
    ]


def test_cl24_supported_with_contradicting_item_becomes_disputed():
    ev = _cl24_evidence()
    out = {"verdict": "supported", "confidence": 0.6,
           "rationale": "The Washington Post reports officials calling the outages "
                        "unrelated. Confidence 0.6 given one source.",
           "bottom_line": "Officials call the outages unrelated, per the Washington Post.",
           "cited": [{"url": ev[0].url, "relation": "supports"},
                     {"url": ev[1].url, "relation": "context"},
                     {"url": ev[2].url, "relation": "contradicts"}]}
    v, c, r, ev2, bl, mn = claims_mod.verdict_llm(CL24_TEXT, {"assert": 4}, ev,
                                                  FakeLLM(verdict=out), "p", CL2_WINDOW,
                                                  topic="bank of america")
    assert v == "disputed" and c <= 0.5
    note = next(m for m in mn if m.startswith("Conflict check:"))
    assert "[3] Fortune contradicts" in note and "[1] Facebook supports" in note
    # rule-based sentence from the final evidence, not the model's one-sided line
    assert bl == "Facebook supports it; Fortune contradicts it."
    assert r.startswith("Set to disputed:") and "0.6" not in r


def test_vaccine_factcheck_contradicted_with_naming_support_becomes_disputed():
    claim = "Pfizer's RSV vaccine was linked to 40 infant deaths in Brazil."
    ev = [Evidence(url="https://factcheck.example/rsv", source="google_factcheck",
                   title="Claim of 40 infant deaths after Pfizer RSV vaccine in Brazil",
                   publisher="Lupa", supports="contradicts", note="rating: False",
                   published_at="2026-11-15"),
          _ev("https://news.example/brazil-pfizer-rsv-40-infant-deaths",
              "Brazil regulator reviews 40 infant deaths after Pfizer RSV vaccine", "G1",
              supports="supports", published="2026-11-16")]
    v, c, notes, _ = claims_mod.enforce_support(claim, "contradicted", 0.8, ev, "pfizer")
    assert v == "disputed" and c == 0.5 and notes[0].startswith("Conflict check:")
    # a supporting item that only covers related events does not overturn 'contradicted'
    weak = [ev[0], _ev("https://news.example/rsv", "RSV season starts early", "G1",
                       supports="supports", published="2026-11-16")]
    assert claims_mod.enforce_support(claim, "contradicted", 0.8, weak, "pfizer")[0] == \
        "contradicted"
    # the same rule for an older run's claim (backfill)
    from socmint.schema import Claim
    old = Claim(id="CL3", text=claim, narrative_id="N1", related_post_ids=["x:1"],
                evidence=ev, verdict="contradicted", confidence=0.8,
                rationale="Lupa rated it false with 0.8 confidence.",
                bottom_line="Lupa rates the claim false.")
    b = claims_mod.backfill_claim(old, "pfizer", CL2_WINDOW)
    assert b.verdict == "disputed" and b.confidence <= 0.5
    assert "changed to disputed" in b.method_notes
    assert b.bottom_line == "G1 supports it; Lupa contradicts it."
    assert b.rationale.startswith("Set to disputed:")


def test_backfill_cl24_supported_becomes_disputed():
    from socmint.schema import Claim
    ev = [e.model_copy(update={"supports": s}) for e, s in
          zip(_cl24_evidence(), ("supports", "context", "contradicts"), strict=True)]
    old = Claim(id="CL24", text=CL24_TEXT, narrative_id="N1", related_post_ids=["x:1"],
                evidence=ev, verdict="supported", confidence=0.6, rationale="r.",
                bottom_line="Officials call the outages unrelated.")
    b = claims_mod.backfill_claim(old, "bank of america", CL2_WINDOW)
    assert (b.verdict, b.confidence) == ("disputed", 0.5)
    assert "Fortune contradicts" in b.bottom_line


def test_partly_named_list_member_is_said_in_the_bottom_line():
    ev = [_ev("https://www.cnn.com/2026/11/16/business/bank-outage-chase-us-bank-bank-of-"
              "america", "Thousands of Chase Bank customers report trouble", "CNN"),
          _ev("https://y.example/x", "Thousands of Chase, Bank of America and Wells Fargo "
              "Customers Say They Can’t Move Their Money", "Yahoo Finance")]
    cited = [{"url": e.url, "relation": "supports"} for e in ev]
    out = {"verdict": "supported", "confidence": 0.7, "rationale": "CNN and Yahoo.",
           "bottom_line": "CNN and Yahoo Finance report the 16 Nov login failures.",
           "cited": cited}
    v, _, _, _, bl, _ = claims_mod.verdict_llm(CL10_TEXT, {}, ev, FakeLLM(verdict=out), "p",
                                               CL2_WINDOW, topic="bank of america")
    assert v == "supported"
    assert bl == ("CNN and Yahoo Finance report the 16 Nov login failures. Not confirmed "
                  "for Zelle.")
    # a model line naming the unnamed actor may call it confirmed: rule sentence instead
    out["bottom_line"] = "Outlets confirm outages at all five, Zelle included."
    _, _, _, _, bl, _ = claims_mod.verdict_llm(CL10_TEXT, {}, ev, FakeLLM(verdict=out), "p",
                                               CL2_WINDOW, topic="bank of america")
    assert bl == "CNN and Yahoo Finance support it, but not confirmed for Zelle."


def test_partly_named_product_recall_rule_line_and_backfill():
    from socmint.schema import Claim
    claim = "The recall covers Graco, Chicco, Evenflo and Britax car seats sold since 2024."
    ev = [_ev("u", "Chicco, Evenflo and Britax recall car seats", "Reuters", "supports")]
    assert claims_mod.rule_bottom_line("supported", ev, {}, partly=["Graco"]) == \
        "Reuters supports it, but not confirmed for Graco."
    old = Claim(id="CL1", text=claim, narrative_id="N1", related_post_ids=["x:1"],
                evidence=ev, verdict="supported", confidence=0.5, rationale="Reuters.",
                bottom_line="Reuters reports the car seat recall.")
    b = claims_mod.backfill_claim(old, "britax", CL2_WINDOW)
    assert b.verdict == "supported"
    assert b.bottom_line == "Reuters reports the car seat recall. Not confirmed for Graco."
    # idempotent on a second backfill
    assert claims_mod.backfill_claim(b, "britax", CL2_WINDOW).bottom_line == b.bottom_line
