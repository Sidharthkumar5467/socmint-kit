"""Run summary (analyze/brief.py): ranking, clusters, coverage and findings. No network."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import make_post

from socmint.analyze import brief as B
from socmint.runstore import read_json, read_jsonl
from socmint.schema import (
    Claim,
    Cluster,
    Narrative,
    OriginTrace,
    RunMeta,
    Signal,
    SourceState,
    SourceStatus,
)

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_run"
META = RunMeta(run_id="r", topic="bank", start="2026-11-13", end="2026-11-19",
               sources=["x", "telegram", "youtube"], created_at="2026-11-19T00:00:00Z")


def _p(nid, day, source="x", author="a1", **kw):
    return make_post(nid, f"text {nid}", source=source, author=author,
                     published_at=f"2026-11-{day:02d}T12:00:00Z", **kw)


def _narr(nid, posts, **kw):
    vol_day: dict[str, int] = {}
    vol_plat: dict[str, int] = {}
    for p in posts:
        if p.timestamp_verified:
            vol_day[p.published_at[:10]] = vol_day.get(p.published_at[:10], 0) + 1
            vol_plat[p.source] = vol_plat.get(p.source, 0) + 1
    ids = [p.post_id for p in posts]
    return Narrative(id=nid, label=kw.pop("label", f"label {nid}"), claim_summary="s",
                     member_post_ids=ids, example_post_ids=ids[:2], volume_by_day=vol_day,
                     volume_by_platform=vol_plat, **kw)


def _cluster(cid, n_accounts, post_ids, types=("near_duplicate_text",
                                              "near_simultaneous_posting"), score=0.9,
             sources=("telegram",)):
    return Cluster(id=cid, member_author_ids=[f"u{i}" for i in range(n_accounts)],
                   signals=[Signal(type=t, score=score, evidence_post_ids=list(post_ids))
                            for t in types],
                   supporting_post_ids=list(post_ids), sources=list(sources))


def _full_x():
    """X posts on every window day, so X is a fully covered source."""
    return [_p(f"bg{d}", d) for d in range(13, 20)]


# ----------------------------------------------------------------------- helpers
def test_peak_surge_short_day_platform_name():
    assert B.peak({}) == (None, 0)
    assert B.peak({"2026-11-14": 3, "2026-11-13": 3, "2026-11-15": 1}) == ("2026-11-13", 3)
    window = B._window_days(META)
    assert len(window) == 7
    # 12 on one day, 1/day on the 6 others -> 12x
    vol = {d.isoformat(): 1 for d in window}
    vol["2026-11-16"] = 12
    assert B.surge_ratio(vol, window) == 12.0
    # single day only: denominator floored at 1
    assert B.surge_ratio({"2026-11-19": 4}, window) == 4.0
    assert B.surge_ratio({}, window) == 0.0
    assert B.short_day("2026-11-06") == "6 Nov" and B.short_day("junk") == "junk"
    assert B.short_day(None) == ""
    assert B.platform_name("linkedin_ads") == "LinkedIn Ads"
    assert B.platform_name("new_src") == "New Src"


def test_covered_and_full_sources():
    window = B._window_days(META)
    posts = _full_x() + [_p("t1", 19, source="telegram"),
                         _p("t2", 18, source="telegram", verified=False),
                         _p("t3", 17, source="telegram", context_only=True)]
    cov = B.covered_days(posts, window)
    assert len(cov["x"]) == 7 and [d.day for d in cov["telegram"]] == [19]
    assert B.full_sources(cov, window) == {"x"}


# ----------------------------------------------------------------------- ranking
def test_rank_relevance_main_and_empty_volume():
    bg = _full_x()
    big_off = [_p(f"o{i}", 15) for i in range(40)]
    threat = [_p(f"t{i}", 14) for i in range(12)]
    related = [_p(f"r{i}", 14) for i in range(15)]
    small = [_p(f"s{i}", 14) for i in range(4)]
    unverified = [_p(f"u{i}", 14, verified=False) for i in range(20)]
    narrs = [
        _narr("N1", big_off, relevance="off_topic"),
        _narr("N2", threat, relevance="threat"),
        _narr("N3", related, relevance="related"),
        _narr("N4", small),
        _narr("N5", unverified),
    ]
    ranks = B.rank_narratives(narrs, [], [], bg + big_off + threat + related + small
                              + unverified, META)
    by = {r.narrative_id: r for r in ranks}
    assert [r.rank for r in ranks] == list(range(1, 6))
    assert by["N2"].rank < by["N3"].rank  # threat x1.5 beats a slightly larger related one
    assert "threat" in by["N2"].tags and by["N2"].reasons[0] == "threat-relevant"
    assert "off_topic" in by["N1"].tags and not by["N1"].main
    assert by["N1"].rank > by["N3"].rank  # 40 posts, but off-topic is ranked down
    assert not by["N4"].main  # fewer than MAIN_MIN_POSTS
    # all-unverified narrative: empty volume, still ranked by size and can be main
    assert by["N5"].peak_day is None and by["N5"].peak_count == 0
    assert by["N5"].main and by["N5"].score > 0
    assert by["N3"].peak_day == "2026-11-14" and by["N3"].peak_count == 15


def test_main_is_capped(monkeypatch):
    monkeypatch.setattr(B, "MAIN_MAX", 2)
    narrs = [_narr(f"N{k}", [_p(f"{k}-{i}", 14) for i in range(10 + k)]) for k in range(4)]
    ranks = B.rank_narratives(narrs, [], [], [], META)
    assert [r.main for r in ranks] == [True, True, False, False]
    assert ranks[0].narrative_id == "N3"


def test_surge_ignores_sources_that_only_reached_newest_day():
    bg = _full_x()
    # 30 telegram posts on 19 Nov only (telegram did not cover older days) + 1 x/day
    tg = [_p(f"t{i}", 19, source="telegram") for i in range(30)]
    xs = [_p(f"x{d}", d) for d in range(13, 20)]
    real = [_p(f"y{i}", 16) for i in range(12)] + [_p(f"z{d}", d) for d in (13, 14)]
    narrs = [_narr("FAKE", tg + xs), _narr("REAL", real)]
    ranks = {r.narrative_id: r for r in B.rank_narratives(narrs, [], [], bg + tg + xs + real,
                                                          META)}
    assert ranks["FAKE"].peak_day == "2026-11-19"  # peak still reported from volume
    assert "surging" not in ranks["FAKE"].tags
    assert "surging" in ranks["REAL"].tags
    assert any("surge on 16 Nov" in r and "fully covered" in r for r in ranks["REAL"].reasons)


def test_coordination_and_claim_tags():
    posts = [_p(f"{i}", 14) for i in range(12)]
    n = _narr("N1", posts)
    cl_small = _cluster("C2", 2, ["x:0"])
    cl_big = _cluster("C1", 6, ["x:1", "x:2", "x:3"])
    unrelated = _cluster("C3", 9, ["x:zzz"])
    claims = [Claim(id="CL1", text="a", narrative_id="N1", related_post_ids=["x:1"],
                    verdict="disputed"),
              Claim(id="CL2", text="b", narrative_id="N1", related_post_ids=["x:2"],
                    verdict="supported"),
              Claim(id="CL3", text="c", narrative_id="N1", related_post_ids=["x:3"],
                    verdict="contradicted"),
              Claim(id="CL4", text="d", narrative_id="N9", related_post_ids=[])]
    base = B.rank_narratives([n], [], [], posts, META)[0]
    r = B.rank_narratives([n], [cl_small, cl_big, unrelated], claims, posts, META)[0]
    assert r.cluster_ids == ["C1", "C2"]  # larger overlap first, unrelated cluster absent
    assert r.claim_ids == ["CL1", "CL2", "CL3"]
    for t in ("coordinated", "disputed_claim", "contradicted_claim", "supported_claim"):
        assert t in r.tags
    assert r.score > base.score
    assert "coordinated groups C1, C2" in r.reasons


def test_rank_and_map_clusters():
    n1 = _narr("N1", [_p(f"{i}", 14) for i in range(5)])
    n2 = _narr("N2", [_p(f"b{i}", 14) for i in range(5)])
    c_big2 = _cluster("C1", 10, ["x:0", "x:b0", "x:b1"])
    c_big3 = _cluster("C2", 6, ["x:1"], types=("near_duplicate_text",
                                              "near_simultaneous_posting",
                                              "similar_creation_dates"))
    c_small = _cluster("C3", 4, ["x:2"])
    assert B.rank_clusters([c_big2, c_big3, c_small]) == ["C2", "C1"]  # signals beat size
    assert B.rank_clusters_all([c_small, c_big2]) == ["C1", "C3"]
    m = B.map_cluster_narratives([c_big2, c_big3], [n1, n2])
    assert m == {"C1": ["N2", "N1"], "C2": ["N1"]}


# ---------------------------------------------------------------------- coverage
def test_coverage_notes():
    meta = META.model_copy(update={"sources": ["x", "telegram", "youtube", "linkedin",
                                               "linkedin_ads", "never_ran", "gappy"]})
    posts = (_full_x() + [_p("t1", 19, source="telegram"), _p("t2", 18, source="telegram")]
             + [_p("g1", 13, source="gappy"), _p("g2", 19, source="gappy")]
             + [make_post("a1", source="linkedin_ads", kind="ad", verified=False),
                make_post("a2", source="linkedin_ads", kind="ad",
                          published_at="2026-11-15T00:00:00Z")])
    sts = [SourceStatus(source="x", query="q", state=SourceState.ok),
           SourceStatus(source="telegram", query="q", state=SourceState.ok),
           SourceStatus(source="telegram", query="q2", state=SourceState.partial),
           SourceStatus(source="youtube", query="q", state=SourceState.skipped_no_credentials),
           SourceStatus(source="linkedin", query="q", state=SourceState.completed_zero),
           SourceStatus(source="linkedin_ads", query="q", state=SourceState.partial),
           SourceStatus(source="gappy", query="q", state=SourceState.ok)]
    notes = {c.source: c for c in B.coverage_notes(posts, sts, meta)}
    assert notes["x"].note == "all 7 days covered" and notes["x"].days_with_posts == 7
    assert notes["telegram"].state == "partial"  # worst state across queries
    assert notes["telegram"].note == "newest 2 days only; older days not reached"
    assert notes["youtube"].note == "skipped: no credentials"
    assert notes["linkedin"].note == "searched; no posts found"
    assert notes["linkedin_ads"].note == "ads: 1 of 2 with a verified date in the window"
    assert notes["never_ran"].note == "no status recorded"
    assert notes["gappy"].note == ("posts on 2 of 7 days (none on 14 Nov, 15 Nov, 16 Nov, "
                                   "17 Nov, 18 Nov)")
    one = B.coverage_notes([_p("t", 19, source="telegram")],
                           [SourceStatus(source="telegram", query="q",
                                         state=SourceState.partial)], META)
    assert one[0].note == "newest day only; older days not reached"


# ---------------------------------------------------------------------- findings
def _scenario():
    bg = _full_x()
    outage = ([_p(f"o{i}", 16) for i in range(20)] + [_p(f"oy{i}", 17, source="youtube")
                                                     for i in range(10)])
    cards = [_p(f"c{i}", 19, source="telegram", author=f"tg{i}") for i in range(12)]
    narrs = [_narr("N1", outage, label="Outage", first_seen_at="2026-11-16T12:00:00Z"),
             _narr("N2", cards, label="Cards for sale", relevance="threat",
                   relevance_reason="stolen card data offered")]
    clusters = [_cluster("C1", 8, [p.post_id for p in cards[:8]],
                         types=("near_duplicate_text", "near_simultaneous_posting",
                                "similar_creation_dates")),
                _cluster("C2", 5, [p.post_id for p in cards[8:]])]
    clusters[0].summary = "8 Telegram accounts posted near-identical text."
    traces = [OriginTrace(narrative_id="N1", earliest_post_ids=["x:o0"], edges=[])]
    claims = [Claim(id="CL1", text="Banks are down", narrative_id="N1",
                    related_post_ids=["x:o1", "x:o2"], verdict="disputed", confidence=0.5,
                    stances={"assert": 2, "report": 0, "question": 0, "debunk": 0},
                    stance_post_ids={"assert": ["x:o1"], "report": [], "question": [],
                                     "debunk": []},
                    bottom_line="News confirms an outage on 16 Nov."),
              Claim(id="CL2", text="x", narrative_id="N1", related_post_ids=["x:o3"],
                    verdict="supported", confidence=0.8),
              Claim(id="CL3", text="y", narrative_id="N2", related_post_ids=["x:c1"])]
    statuses = [SourceStatus(source="x", query="q", state=SourceState.ok),
                SourceStatus(source="telegram", query="q", state=SourceState.partial),
                SourceStatus(source="youtube", query="q", state=SourceState.ok)]
    posts = bg + outage + cards
    return posts, narrs, clusters, traces, claims, statuses


def test_build_brief_findings():
    posts, narrs, clusters, traces, claims, statuses = _scenario()
    b = B.build_brief(posts, narrs, clusters, traces, claims, statuses, META)
    assert [f.id for f in b.findings] == ["F1", "F2", "F3", "F4", "F5"]
    assert [f.kind for f in b.findings] == ["narrative", "narrative", "coordination",
                                            "claims", "coverage"]
    f1, f2, f3, f4, f5 = b.findings
    assert b.narrative_ranks[0].narrative_id == "N1"
    assert f1.narrative_ids == ["N1"]
    assert f1.headline == ("Top-ranked narrative “Outage”: 30 posts on 2 platforms, "
                           "peaking on 16 Nov (20 posts).")
    assert "Claims: 1 supported, 1 disputed." in f1.detail
    assert f1.post_ids[0] == "x:o0"  # earliest observed post first
    assert "First observed in collected data: 16 Nov." in f1.detail
    # the threat narrative gets its own finding when it is not the lead
    assert f2.narrative_ids == ["N2"]
    assert f2.headline.startswith("Threat-relevant narrative “Cards for sale”: 12 posts")
    assert "12 of them on 19 Nov" in f2.headline
    assert "stolen card data offered" in f2.detail
    assert f3.cluster_ids == ["C1", "C2"]
    assert "C1: 8 Telegram accounts with 3 signal types" in f3.headline
    assert "8 of those posts are in narrative “Cards for sale”" in f3.headline
    assert "8 Telegram accounts posted near-identical text." in f3.detail
    assert "Group C2 (5 more accounts)" in f3.detail
    assert set(f3.post_ids) == set(clusters[0].supporting_post_ids)
    assert f4.headline == ("3 claims checked: 1 supported, 0 contradicted, 1 disputed, "
                           "1 unverified.")
    assert f4.claim_ids[0] == "CL1" and set(f4.claim_ids) == {"CL1", "CL2", "CL3"}
    assert "“Banks are down” is disputed (confidence 0.50)" in f4.detail
    assert "News confirms an outage" in f4.detail
    assert f4.post_ids == ["x:o1", "x:o2"]  # assert-stance posts first, then related
    assert "Telegram 1 of 7 days (newest day only; older days not reached)" in f5.headline
    assert "YouTube 1 of 7 days" in f5.headline
    assert f5.detail.startswith("Full coverage: X. ")
    assert b.verdict_counts == {"supported": 1, "contradicted": 0, "disputed": 1,
                                "unverified": 1}
    assert b.main_cluster_ids == ["C1", "C2"]
    assert b.cluster_narratives["C1"] == ["N2"]
    # every id a finding names exists
    pids = {p.post_id for p in posts}
    for f in b.findings:
        assert set(f.post_ids) <= pids
        assert set(f.narrative_ids) <= {"N1", "N2"}


def test_peak_day_caveat_for_partially_covered_source():
    posts, narrs, clusters, traces, claims, statuses = _scenario()
    tg = [_p(f"lt{i}", 16, source="telegram") for i in range(3)]
    narrs[0] = _narr("N1", [*(posts[p] for p in range(7, 37)), *tg], label="Outage")
    b = B.build_brief(posts + tg, narrs, clusters, traces, claims, statuses, META)
    assert ("Telegram did not cover every day, so 16 Nov counts are not comparable "
            "across days.") in b.findings[0].detail
    # the headline names the coverage effect and the peak on the fully covered sources
    # (many analysts read only the headline); the detail does not repeat it
    head = b.findings[0].headline
    assert head == ("Top-ranked narrative “Outage”: 33 posts on 3 platforms; most on 16 Nov "
                    "(23 posts), but Telegram did not cover every day (peak on fully covered "
                    "sources: 16 Nov, 20 posts)."), head
    assert "On the fully covered sources" not in b.findings[0].detail


def test_peak_headline_only_day_wording():
    posts, narrs, clusters, traces, claims, statuses = _scenario()
    tg = [_p(f"lt{i}", 19, source="telegram") for i in range(30)]
    narrs[0] = _narr("N1", [*(posts[p] for p in range(7, 37)), *tg], label="Outage")
    b = B.build_brief(posts + tg, narrs, clusters, traces, claims, statuses, META)
    head = b.findings[0].headline
    assert "; most on 19 Nov (30 posts), the only day Telegram was collected (peak on fully " \
           "covered sources: 16 Nov, 20 posts)." in head, head
    assert "peaking on" not in head


def test_narrative_finding_cites_first_strong_post_when_earliest_is_weak():
    posts, narrs, clusters, traces, claims, statuses = _scenario()
    strong_id = narrs[0].member_post_ids[3]
    traces[0] = OriginTrace(narrative_id="N1", earliest_post_ids=["x:o0"], edges=[],
                            earliest_match="weak", earliest_similarity=0.4,
                            first_strong_post_id=strong_id)
    b = B.build_brief(posts, narrs, clusters, traces, claims, statuses, META)
    f1 = b.findings[0]
    assert f1.post_ids[0] == strong_id and "x:o0" not in f1.post_ids[:1]
    assert "only loosely matches the narrative; the first close match is from" in f1.detail
    assert f"({strong_id})." in f1.detail


def test_threat_finding_carries_coverage_caveat_and_unverified_note():
    posts, narrs, clusters, traces, claims, statuses = _scenario()
    b = B.build_brief(posts, narrs, clusters, traces, claims, statuses, META)
    f2 = b.findings[1]
    # platform counts are grouped in brackets so they never read as extra posts
    assert f2.headline.startswith("Threat-relevant narrative “Cards for sale”: 12 posts (")
    assert "Why threat-relevant (labeller): stolen card data offered." in f2.detail
    # all 12 posts are on Telegram (partial), none on the fully covered X
    assert "Telegram did not cover every day, so 19 Nov counts" in f2.detail
    assert "None of its posts came from the fully covered sources (X)." in f2.detail


def test_unverified_note_names_count():
    n = Narrative(id="N9", label="l", claim_summary="", member_post_ids=["a"],
                  example_post_ids=[], unverified_member_count=2)
    assert B._unverified_note(n) == ["2 member post(s) without a verified timestamp are "
                                     "not counted by day or platform."]
    n.unverified_member_count = 0
    assert B._unverified_note(n) == []


def test_build_brief_accepts_dicts_and_empty():
    posts, narrs, clusters, traces, claims, statuses = _scenario()
    as_dicts = [[json.loads(x.model_dump_json(by_alias=True)) for x in xs]
                for xs in (posts, narrs, clusters, traces, claims, statuses)]
    b1 = B.build_brief(*as_dicts, META.model_dump())
    b2 = B.build_brief(posts, narrs, clusters, traces, claims, statuses, META)
    assert b1.model_dump() == b2.model_dump()

    empty = B.build_brief([], [], [], [], [], [], META)
    assert empty.narrative_ranks == []
    assert [f.kind for f in empty.findings] == ["coverage"]  # nothing ran: say so
    assert empty.verdict_counts == {"supported": 0, "contradicted": 0, "disputed": 0,
                                    "unverified": 0}
    assert [c.note for c in empty.coverage] == ["no status recorded"] * 3


def test_no_coverage_finding_when_complete_and_cluster_fallback():
    bg = _full_x()
    n = _narr("N1", [_p(f"{i}", 14) for i in range(3)])
    small = _cluster("C9", 2, ["x:0", "x:1"], sources=("x",))
    b = B.build_brief(bg + [_p(f"{i}", 14) for i in range(3)], [n], [small], [], [],
                      [SourceStatus(source="x", query="q", state=SourceState.ok)],
                      META.model_copy(update={"sources": ["x"]}))
    kinds = [f.kind for f in b.findings]
    assert "coverage" not in kinds and "claims" not in kinds
    assert b.main_cluster_ids == []
    coord = next(f for f in b.findings if f.kind == "coordination")
    assert coord.cluster_ids == ["C9"]  # strongest small group still reported
    assert b.findings[0].narrative_ids == ["N1"]  # no main narrative: top-ranked used


def test_sample_run_brief_is_traceable():
    """Real bundled data (read-only): every id in the brief resolves."""
    d = SAMPLE
    posts = read_jsonl(d / "posts.jsonl")
    narrs = read_json(d / "narratives.json")
    clusters = read_json(d / "clusters.json")
    claims = read_json(d / "claims.json")
    b = B.build_brief(posts, narrs, clusters, read_json(d / "origin.json"), claims,
                      read_json(d / "source_status.json"), read_json(d / "run.json"))
    assert len(b.narrative_ranks) == len(narrs)
    assert sum(r.main for r in b.narrative_ranks) == B.MAIN_MAX
    assert 1 <= len(b.findings) <= B.MAX_FINDINGS
    pids = {p["post_id"] for p in posts}
    nids = {n["id"] for n in narrs}
    cids = {c["id"] for c in clusters}
    clids = {c["id"] for c in claims}
    for f in b.findings:
        assert f.headline.endswith(".")
        assert set(f.post_ids) <= pids and set(f.narrative_ids) <= nids
        assert set(f.cluster_ids) <= cids and set(f.claim_ids) <= clids
    assert b.main_cluster_ids[0] == "C1"
    assert sum(b.verdict_counts.values()) == len(claims)
    # Every collected source has a human-readable coverage note (exact wording depends on
    # the run's window type and each source's reach, so assert presence, not literal text).
    cov = {c.source: c.note for c in b.coverage}
    assert cov.get("x") and "covered" in cov["x"]
    assert cov.get("telegram")


def test_main_narratives_get_contiguous_ranks():
    # a small but high-scoring narrative must not take a rank number inside the main list
    big = [_narr(f"N{i}", [_p(f"b{i}{k}", 13) for k in range(12)]) for i in range(1, 4)]
    small = _narr("N9", [_p(f"s{k}", 13) for k in range(4)], relevance="threat")
    small.volume_by_platform = {"x": 2, "youtube": 1, "telegram": 1}
    posts = [_p(f"b{i}{k}", 13) for i in range(1, 4) for k in range(12)]
    ranks = B.rank_narratives([*big, small], [], [], posts, META)
    assert [r.main for r in ranks] == [True, True, True, False]
    assert [r.rank for r in ranks] == [1, 2, 3, 4]
    assert ranks[-1].narrative_id == "N9"


# ----------------------------------------------------------------------- severity
def test_rank_by_severity_before_size_and_surge():
    bg = _full_x()
    low_big = [_p(f"l{i}", 14) for i in range(60)]
    med = [_p(f"m{i}", 14) for i in range(20)]
    high = [_p(f"h{i}", 14) for i in range(11)]
    crit = [_p(f"c{i}", 14) for i in range(12)]
    unrated = [_p(f"u{i}", 14) for i in range(80)]
    off = [_p(f"o{i}", 14) for i in range(90)]
    narrs = [_narr("LOW", low_big, severity="low", relevance="related"),
             _narr("MED", med, severity="medium"),
             _narr("HIGH", high, severity="high", severity_reason="bank-run rumour"),
             _narr("CRIT", crit, severity="critical", severity_reason="cards sold"),
             _narr("NONE", unrated),
             # off_topic: last whatever its size, and a stray severity is ignored
             _narr("OFF", off, relevance="off_topic", severity="critical")]
    ranks = B.rank_narratives(narrs, [], [], bg + low_big + med + high + crit + unrated + off,
                              META)
    assert [r.narrative_id for r in ranks] == ["CRIT", "HIGH", "MED", "LOW", "NONE", "OFF"]
    by = {r.narrative_id: r for r in ranks}
    assert by["CRIT"].main and by["CRIT"].severity == "critical"
    assert by["CRIT"].tags[0] == "severity:critical"
    assert by["CRIT"].reasons[0] == "critical severity"
    assert not by["OFF"].main and by["OFF"].severity is None
    assert not any(t.startswith("severity:") for t in by["OFF"].tags)
    assert by["NONE"].severity is None and by["NONE"].main


def test_severe_main_threshold():
    """5-9 post critical/high narratives are main only with coordination or a
    disputed/contradicted claim (bucket c); fewer than 5 never."""
    posts = ([_p(f"a{i}", 14) for i in range(4)] + [_p(f"b{i}", 14) for i in range(5)]
             + [_p(f"d{i}", 14) for i in range(6)])
    narrs = [_narr("C4", posts[:4], severity="critical"),
             _narr("H5", posts[4:9], severity="high"),
             _narr("H6", posts[9:], severity="high"),
             _narr("M9", [_p(f"m{i}", 14) for i in range(9)], severity="medium")]
    claims = [Claim(id="CL1", text="t", narrative_id="H5", related_post_ids=["x:b1"],
                    verdict="disputed", confidence=0.5)]
    by = {r.narrative_id: r for r in B.rank_narratives(narrs, [], claims, posts, META)}
    assert by["H5"].main and not by["H6"].main
    assert not by["C4"].main and not by["M9"].main
    assert B.MAIN_MIN_POSTS == 10 and B.MAIN_MIN_POSTS_SEVERE == 5
    assert by["H5"].rank == 1  # main ranks first, contiguous
    assert by["C4"].rank == 2  # then the rest in severity order


def test_main_buckets_fill_order_and_display_order():
    """Main = up to 10: (a) critical/high >= 10, (b) each main cluster's narrative at any
    severity, (c) high 5-9 with coordination/dispute, (d) medium >= 10, (e) low >= 10.
    Shown in the global severity order."""
    bg = _full_x()
    groups = {}
    narrs = []

    def add(nid, n, **kw):
        ps = [_p(f"{nid.lower()}{i}", 14) for i in range(n)]
        groups[nid] = ps
        narrs.append(_narr(nid, ps, **kw))

    for i in range(4):
        add(f"H{i}", 12, severity="high")
    add("COORDLOW", 6, severity="low")  # tiny and low, but a 6-account cluster's narrative
    add("HSMALL", 7, severity="high")  # coordinated through a 2-account cluster
    for i in range(6):
        add(f"M{i}", 15 + i, severity="medium")
    for i in range(3):
        add(f"L{i}", 40, severity="low")
    clusters = [_cluster("C1", 6, [p.post_id for p in groups["COORDLOW"]]),
                _cluster("C2", 2, [groups["HSMALL"][0].post_id])]
    posts = bg + [p for ps in groups.values() for p in ps]
    ranks = B.rank_narratives(narrs, clusters, [], posts, META)
    main = [r.narrative_id for r in ranks if r.main]
    assert len(main) == B.MAIN_MAX
    # a: 4 high; b: COORDLOW; c: HSMALL; d: the 4 largest mediums fill the rest
    assert set(main) == {"H0", "H1", "H2", "H3", "COORDLOW", "HSMALL", "M5", "M4", "M3",
                         "M2"}
    assert not any(r.main for r in ranks if r.narrative_id.startswith("L"))
    # display order = global ranking: high (coordinated first), medium, then low
    assert main[0] == "HSMALL" and main[-1] == "COORDLOW"
    assert [r.rank for r in ranks if r.main] == list(range(1, 11))


def test_low_narratives_fill_main_after_medium():
    narrs = [_narr("L1", [_p(f"l{i}", 14) for i in range(30)], severity="low"),
             _narr("M1", [_p(f"m{i}", 14) for i in range(10)], severity="medium"),
             _narr("L2", [_p(f"k{i}", 14) for i in range(9)], severity="low")]
    posts: list = []
    by = {r.narrative_id: r for r in B.rank_narratives(narrs, [], [], posts, META)}
    assert by["L1"].main and by["M1"].main and not by["L2"].main
    assert by["M1"].rank == 1 and by["L1"].rank == 2


def test_within_level_coordinated_and_disputed_first_then_size_then_surge():
    bg = _full_x()
    plain = [_p(f"p{i}", 14) for i in range(30)]
    coord = [_p(f"k{i}", 14) for i in range(12)]
    both = [_p(f"b{i}", 14) for i in range(11)]
    # same size, one surges (12 on 16 Nov, 1/day elsewhere), one flat
    surge = [_p(f"s{i}", 16) for i in range(12)] + [_p(f"s{d}x", d) for d in (13, 14)]
    flat = [_p(f"f{d}{k}", d) for d in range(13, 20) for k in range(2)]
    narrs = [_narr(i, ps, severity="high") for i, ps in
             (("PLAIN", plain), ("COORD", coord), ("BOTH", both), ("SURGE", surge),
              ("FLAT", flat))]
    clusters = [_cluster("C1", 6, ["x:k0", "x:b0"])]
    claims = [Claim(id="CL1", text="t", narrative_id="BOTH", related_post_ids=["x:b1"],
                    verdict="contradicted")]
    ranks = B.rank_narratives(narrs, clusters, claims,
                              bg + plain + coord + both + surge + flat, META)
    order = [r.narrative_id for r in ranks]
    assert order[:3] == ["BOTH", "COORD", "PLAIN"]
    assert len(surge) == len(flat) and order[3:] == ["SURGE", "FLAT"]


def test_unrated_runs_keep_score_order():
    # a narrative without severity keeps the earlier activity-score order (older runs)
    bg = _full_x()
    threat = [_p(f"t{i}", 14) for i in range(12)]
    related = [_p(f"r{i}", 14) for i in range(15)]
    narrs = [_narr("N3", related, relevance="related"), _narr("N2", threat, relevance="threat")]
    ranks = B.rank_narratives(narrs, [], [], bg + threat + related, META)
    assert [r.narrative_id for r in ranks] == ["N2", "N3"]
    assert all(r.severity is None for r in ranks)


def test_first_finding_is_most_severe_with_reason():
    posts, narrs, clusters, traces, claims, statuses = _scenario()
    narrs[0].severity, narrs[0].severity_reason = "high", "outage rumours could cause panic"
    narrs[1].severity, narrs[1].severity_reason = "critical", "card data offered for sale"
    b = B.build_brief(posts, narrs, clusters, traces, claims, statuses, META)
    assert [r.narrative_id for r in b.narrative_ranks] == ["N2", "N1"]
    f1, f2 = b.findings[:2]
    assert f1.narrative_ids == ["N2"]
    # posts per platform, no "on K platforms" emphasis (the counts add up to the total)
    assert f1.headline.startswith("Most severe narrative “Cards for sale” (critical severity): "
                                  "12 posts (Telegram 12)")
    assert "platforms" not in f1.headline
    assert f1.detail.startswith("Rated critical by the labeller: card data offered for sale.")
    # the next critical/high narrative gets the second finding
    assert f2.narrative_ids == ["N1"]
    # levels differ: "Next most severe (high)", not "Also high severity"
    assert f2.headline.startswith("Next most severe (high): narrative “Outage”: 30 posts (")
    assert "Rated high by the labeller: outage rumours could cause panic." in f2.detail
    assert [f.kind for f in b.findings] == ["narrative", "narrative", "coordination",
                                            "claims", "coverage"]


def test_second_finding_says_also_when_levels_equal():
    posts, narrs, clusters, traces, claims, statuses = _scenario()
    narrs[0].severity = "high"
    narrs[1].severity = "high"
    b = B.build_brief(posts, narrs, clusters, traces, claims, statuses, META)
    assert b.findings[1].headline.startswith("Also high severity: narrative ")


def test_first_finding_counts_every_member_by_platform():
    """A severe narrative mixing sources: the headline lists posts per platform (all
    members, so it adds up), and claims nothing about 'K platforms'."""
    bg = _full_x()
    tg = [_p(f"t{i}", 19, source="telegram", author=f"t{i}") for i in range(21)]
    yt = [_p(f"y{i}", 15, source="youtube", author=f"y{i}") for i in range(3)]
    xx = [_p(f"x{i}", 15, author=f"x{i}") for i in range(2)]
    n = _narr("N6", tg + yt + xx, severity="critical", label="Cards")
    b = B.build_brief(bg + tg + yt + xx, [n], [], [], [], [], META)
    head = b.findings[0].headline
    assert "26 posts (Telegram 21, YouTube 3, X 2)" in head
    assert "on 3 platforms" not in head


def test_no_second_severity_finding_when_only_one_severe():
    posts, narrs, clusters, traces, claims, statuses = _scenario()
    narrs[0].severity = "low"
    narrs[1].severity = "critical"
    b = B.build_brief(posts, narrs, clusters, traces, claims, statuses, META)
    assert [f.kind for f in b.findings] == ["narrative", "coordination", "claims", "coverage"]
    assert b.findings[0].narrative_ids == ["N2"]
    assert b.findings[0].detail.startswith("Rated critical by the labeller.")


def test_claims_finding_without_disputed_or_contradicted():
    # a vaccine topic where every claim is supported or unverified: the claims finding
    # still says what was found instead of an empty detail line
    claims = [Claim(id="CL1", text="The vaccine was approved for children under 5",
                    narrative_id="N1", related_post_ids=["x:1"], verdict="supported",
                    confidence=0.8),
              Claim(id="CL2", text="The vaccine changes DNA", narrative_id="N2",
                    related_post_ids=["x:2"], verdict="unverified", confidence=0.3)]
    f = B._finding_claims("F4", claims, {"supported": 1, "unverified": 1})
    assert f.headline == ("2 claims checked: 1 supported, 0 contradicted, 0 disputed, "
                          "1 unverified.")
    assert f.detail == "No retrieved source contradicts a claim, and none is disputed."
    assert set(f.claim_ids) == {"CL1", "CL2"}


# ------------------------------------- 2026-11-20 review: main needs verified posts
def _ads(prefix, n, advertiser="adv-1"):
    """Ads whose run began before the window: members, but never counted (0 verified)."""
    return [_p(f"{prefix}{i}", 1, source="linkedin_ads", author=advertiser, kind="ad")
            for i in range(n)]


def test_low_narrative_without_verified_posts_never_main():
    """Live re-run N8: a low-severity vendor ad narrative (12 members, 0 verified) was
    main. Non-bank case: a gym chain's ads during a marathon."""
    ads = _ads("gym", 12)
    low = _narr("N8", ads, severity="low", label="Gym chain ads near the marathon",
                unverified_member_count=12)
    med = _narr("N9", _ads("tix", 12, "adv-2"), severity="medium",
                label="Resold marathon bib ads", unverified_member_count=12)
    organic = [_p(f"o{i}", 15, author=f"o{i}") for i in range(12)]
    ok = _narr("N1", organic, severity="low", label="Runners discuss the route")
    b = B.build_brief(_full_x() + ads + organic, [low, med, ok], [], [], [], [], META)
    by = {r.narrative_id: r for r in b.narrative_ranks}
    assert not by["N8"].main          # low and 0 verified: never main
    assert by["N9"].main              # medium or above: main even with 0 verified
    assert by["N1"].main              # verified posts: unchanged
    # volume phrases give the verified count; member totals are labelled as members
    assert by["N8"].reasons[1] == "0 verified posts of 12 members"
    assert by["N1"].reasons[1] == "12 posts"


def test_headline_labels_member_total_and_verified_count():
    ads = _ads("ad", 4)
    organic = [_p(f"c{i}", 15, source="telegram", author=f"c{i}") for i in range(6)]
    n = _narr("N2", ads + organic, severity="critical", label="Fake ticket scam",
              unverified_member_count=4)
    b = B.build_brief(_full_x() + ads + organic, [n], [], [], [], [], META)
    head = b.findings[0].headline
    assert "10 member posts (Telegram 6, LinkedIn Ads 4), 6 with a verified in-window date" \
        in head


# ------------------------------------------------- preset windows (partial UTC days)
def _preset_meta(start_at, end_at, sources=("x",)):
    return RunMeta(run_id="p", topic="harbour bridge closure", start=start_at[:10],
                   end=end_at[:10], sources=list(sources), created_at=end_at,
                   options={"window_preset": "1d", "window_start_at": start_at,
                            "window_end_at": end_at})


def _at(nid, ts, source="x"):
    return make_post(nid, f"text {nid}", source=source, author=nid, published_at=ts)


def test_one_day_preset_partial_days_in_coverage_and_findings():
    """Review 2026-11-20: a 1-day preset 09:30 -> 09:30 touches 2 UTC days; the coverage
    line must say 'last 24 hours', not 'all 2 days', and the peak day is marked partial.
    Non-bank topic."""
    meta = _preset_meta("2026-11-19T09:30:00Z", "2026-11-20T09:30:00Z")
    fr = B.day_fractions(meta)
    assert round(fr["2026-11-19"] * 24, 1) == 14.5 and round(fr["2026-11-20"] * 24, 1) == 9.5
    assert B.window_phrase(meta) == "last 24 hours: 2 partial UTC days"
    posts = ([_at(f"a{i}", "2026-11-19T15:00:00Z") for i in range(6)]
             + [_at(f"b{i}", "2026-11-20T03:00:00Z") for i in range(4)])
    st = [SourceStatus(source="x", query="q", state=SourceState.ok, count=10)]
    [note] = B.coverage_notes(posts, st, meta)
    assert note.note == "whole window covered (last 24 hours: 2 partial UTC days)"
    n = _narr("N1", posts, severity="high", relevance="threat")
    brief = B.build_brief(posts, [n], [], [], [], st, meta)
    head = brief.findings[0].headline
    assert "19 Nov (6 posts; partial day, 14 h in the window)" in head
    assert not any(f.kind == "coverage" for f in brief.findings)
    # two partial days: no surge from comparing a 14-hour day with a 10-hour day
    assert "surging" not in brief.narrative_ranks[0].tags


def test_preset_sliver_day_not_required_for_full_coverage():
    meta = _preset_meta("2026-11-19T23:00:00Z", "2026-11-20T23:00:00Z")
    posts = [_at(f"a{i}", "2026-11-20T10:00:00Z") for i in range(3)]
    st = [SourceStatus(source="x", query="q", state=SourceState.ok, count=3)]
    [note] = B.coverage_notes(posts, st, meta)
    assert note.note.startswith("whole window covered (last 24 hours")
    assert "19 Nov sliver" in note.note
    assert B._finding_coverage("F1", [note]) is None


def test_surge_compares_partial_days_per_hour():
    """3-day preset ending at noon: the last UTC day has 12 h in the window. 5 posts in
    those 12 h is 10/day, vs 1-3/day before: a surge per hour, not on raw counts."""
    meta = _preset_meta("2026-11-16T12:00:00Z", "2026-11-19T12:00:00Z")
    window = B._window_days(meta)
    fr = B.day_fractions(meta)
    vol = {"2026-11-16": 1, "2026-11-17": 3, "2026-11-18": 3, "2026-11-19": 5}
    assert B.surge_ratio(vol, window) < B.SURGE_RATIO  # raw counts: 5 vs 2.3
    day, count, ratio = B.surge_peak(vol, window, fr)
    assert day == "2026-11-19" and count == 5 and ratio >= B.SURGE_RATIO
    # a single comparable day never surges
    assert B.surge_ratio({"2026-11-19": 9}, [window[-1]]) == 0.0
