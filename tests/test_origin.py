"""Tests for socmint.analyze.origin (earliest observed posts and spread)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_post  # tests/ is on sys.path

from socmint.analyze import origin
from socmint.analyze.origin import normalize_url, trace_origins
from socmint.schema import Narrative, Relation


@pytest.fixture(autouse=True)
def _tfidf(monkeypatch):
    # Deterministic, offline embeddings.
    monkeypatch.setenv("SOCMINT_EMBEDDINGS", "tfidf")


def narr(*ids: str, nid: str = "n1") -> Narrative:
    return Narrative(id=nid, label="l", claim_summary="c", member_post_ids=list(ids),
                     example_post_ids=list(ids[:1]))


def edges_of(trace, kind=None):
    return [e for e in trace.edges if kind is None or e.type == kind]


def test_normalize_url_equivalences():
    assert normalize_url("https://twitter.com/Foo/status/123?s=20") == \
        normalize_url("http://www.x.com/foo/status/123/")
    assert normalize_url("https://x.com/i/web/status/123") == "x.com/status/123"
    assert normalize_url("https://t.me/s/Chan/55") == normalize_url("t.me/chan/55#frag")
    assert normalize_url("https://youtu.be/AbC123") == \
        normalize_url("https://www.youtube.com/watch?v=AbC123&t=10s")
    assert normalize_url("https://www.youtube.com/watch?v=A") != \
        normalize_url("https://www.youtube.com/watch?v=B")
    assert normalize_url("https://www.youtube.com/watch?v=A&lc=C1") != \
        normalize_url("https://www.youtube.com/watch?v=A")
    assert normalize_url(None) is None


def test_direct_forward_edge_telegram():
    a = make_post("chan:123", "original statement", source="telegram",
                  published_at="2026-11-01T10:00:00Z")
    b = make_post("other:9", "forwarded statement", source="telegram", author="a2",
                  published_at="2026-11-01T12:00:00Z",
                  relations=[Relation(type="forward", target_post_id="telegram:chan:123")])
    [t] = trace_origins([a, b], [narr(a.post_id, b.post_id)])
    [e] = edges_of(t, "direct")
    assert (e.from_, e.to, e.relation) == ("telegram:chan:123", "telegram:other:9", "forward")
    assert e.basis == "forward of telegram:chan:123"
    assert e.lag_seconds == 7200
    assert t.earliest_post_ids == ["telegram:chan:123"]
    assert not edges_of(t, "inferred")  # direct edge suppresses inference on the pair


def test_x_quote_edge_target_outside_narrative():
    a = make_post("1", "some base tweet", published_at="2026-11-01T10:00:00Z")
    b = make_post("2", "quoting it", author="a2", published_at="2026-11-01T10:30:00Z",
                  relations=[Relation(type="quote", target_post_id="x:1")])
    [t] = trace_origins([a, b], [narr("x:2")])  # target not a member: still kept
    [e] = edges_of(t, "direct")
    assert (e.from_, e.to, e.relation, e.type) == ("x:1", "x:2", "quote", "direct")


def test_url_match_youtube_linked_from_x():
    yt = make_post("vid1", "video about the event", source="youtube",
                   url="https://www.youtube.com/watch?v=vid1",
                   published_at="2026-11-01T08:00:00Z")
    x = make_post("10", "watch this", published_at="2026-11-01T09:00:00Z",
                  relations=[Relation(type="link", target_url="https://youtu.be/vid1?si=abc")])
    [t] = trace_origins([yt, x], [narr(yt.post_id, x.post_id)])
    [e] = edges_of(t, "direct")
    assert (e.from_, e.to, e.relation) == ("youtube:vid1", "x:10", "link")
    assert e.basis.startswith("links to ")
    assert t.platform_sequence == ["youtube", "x"]
    assert t.platform_first_seen == {"youtube": "2026-11-01T08:00:00Z",
                                     "x": "2026-11-01T09:00:00Z"}


def test_inferred_edge_cross_platform():
    text = "The dam upstream has collapsed and the city will flood tonight, evacuate now"
    a = make_post("100", text, source="telegram", published_at="2026-11-01T10:00:00Z")
    b = make_post("200", text + "!!", source="x", author="a2",
                  published_at="2026-11-01T13:12:00Z")
    c = make_post("300", "Completely unrelated cooking recipe for pasta with basil", source="x",
                  author="a3", published_at="2026-11-01T14:00:00Z")
    [t] = trace_origins([a, b, c], [narr(a.post_id, b.post_id, c.post_id)])
    inf = edges_of(t, "inferred")
    assert len(inf) == 1
    e = inf[0]
    assert (e.from_, e.to) == ("telegram:100", "x:200")
    assert e.similarity is not None and e.similarity >= 0.75
    assert "+3h12m" in e.basis and "no platform link observed" in e.basis
    assert e.relation is None
    assert not edges_of(t, "direct")


def test_inferred_respects_lag_window():
    text = "identical claim text about the vaccine shipment being fake"
    a = make_post("1", text, source="telegram", published_at="2026-11-01T10:00:00Z")
    b = make_post("2", text, source="x", published_at="2026-11-09T10:00:01Z")
    [t] = trace_origins([a, b], [narr("telegram:1", "x:2")])
    assert not t.edges


def test_unverified_posts_excluded_everywhere():
    a = make_post("1", "same text same text", published_at="2026-10-31T00:00:00Z",
                  verified=False)
    b = make_post("2", "same text same text", published_at="2026-11-01T10:00:00Z",
                  relations=[Relation(type="quote", target_post_id="x:1")])
    out = make_post("3", "same text same text", published_at="2026-11-01T09:00:00Z",
                    in_window=False)
    [t] = trace_origins([a, b, out], [narr("x:1", "x:2", "x:3")])
    assert t.earliest_post_ids == ["x:2"]
    assert not t.edges
    assert "2 member post(s)" in t.note


def test_no_verified_members_yields_empty_trace():
    a = make_post("1", "text", verified=False)
    [t] = trace_origins([a], [narr("x:1")])
    assert t.earliest_post_ids == [] and t.edges == []
    assert "verified" in t.note and "not necessarily the true origin" in t.note


def test_earliest_ties_within_60s_capped_at_3():
    posts = [make_post(str(i), f"t{i}", author=f"a{i}",
                       published_at=f"2026-11-01T10:00:{s:02d}Z")
             for i, s in enumerate([0, 30, 59, 45, 0])]
    late = make_post("9", "late", published_at="2026-11-01T10:05:00Z")
    [t] = trace_origins(posts + [late], [narr(*[p.post_id for p in posts + [late]])])
    assert t.earliest_post_ids == ["x:0", "x:4", "x:1"]
    single = trace_origins([posts[0], late], [narr("x:0", "x:9")])[0]
    assert single.earliest_post_ids == ["x:0"]


def test_inferred_cap(monkeypatch):
    monkeypatch.setattr(origin, "MAX_INFERRED_EDGES", 1)
    text = "breaking the bridge was sabotaged by foreign agents last night"
    ps = [make_post(str(i), text, source=src, author=f"a{i}",
                    published_at=f"2026-11-01T1{i}:00:00Z")
          for i, src in enumerate(["telegram", "x", "youtube"])]
    [t] = trace_origins(ps, [narr(*[p.post_id for p in ps])])
    assert len(edges_of(t, "inferred")) == 1
    assert "1 weaker inferred edge(s) dropped" in t.note


def test_no_origin_wording():
    text = "The dam upstream has collapsed and the city will flood tonight"
    a = make_post("1", text, source="telegram", published_at="2026-11-01T10:00:00Z")
    b = make_post("2", text, source="x", published_at="2026-11-01T11:00:00Z")
    c = make_post("3", "see", source="x", published_at="2026-11-01T12:00:00Z",
                  relations=[Relation(type="link", target_url="https://example.test/x/2")])
    traces = trace_origins([a, b, c], [narr("telegram:1", "x:2", "x:3"), narr("x:9", nid="n2")])
    for t in traces:
        assert "earliest observed in collected data" in t.note.lower()
        assert "origin" not in t.note.lower().replace("not necessarily the true origin", "")
        for e in t.edges:
            assert "origin" not in e.basis.lower()
    assert {e.type for e in traces[0].edges} == {"direct", "inferred"}


def test_comparable_content_gate():
    assert not origin.has_comparable_content("")
    assert not origin.has_comparable_content(None)
    assert not origin.has_comparable_content("https://news.example/a")
    assert not origin.has_comparable_content("  https://a.example/x  @someone www.b.example ")
    assert not origin.has_comparable_content("wow look")
    assert origin.has_comparable_content("The dam upstream has collapsed https://x.example")
    assert origin.has_comparable_content("大坝已经倒塌了城市今晚")  # no-space script, kept
    assert origin.comparable_text("see https://a.example/x @bob now") == "see now"


def test_media_only_and_url_only_posts_get_no_inferred_edges():
    # Reviewer repro: empty / URL-only texts all embed to the same placeholder and used
    # to score cosine 1.0, fabricating "similar content" spread edges.
    a = make_post("1", "https://news.example/a", source="x",
                  published_at="2026-11-01T10:00:00Z")
    b = make_post("9", "", source="telegram", author="a2",
                  published_at="2026-11-01T12:00:00Z")
    c = make_post("vid", "https://other.example/zzz", source="youtube", author="a3",
                  published_at="2026-11-01T13:00:00Z")
    d = make_post("10", "   ", source="telegram", author="a4",
                  published_at="2026-11-01T14:00:00Z")
    [t] = trace_origins([a, b, c, d], [narr(a.post_id, b.post_id, c.post_id, d.post_id)])
    assert not edges_of(t, "inferred")
    assert t.earliest_post_ids == ["x:1"]  # still ordered; just not compared
    assert "4 member post(s) had too little text" in t.note


def test_urls_do_not_drive_similarity_but_text_still_does():
    text = "The dam upstream has collapsed and the city will flood tonight, evacuate now"
    a = make_post("100", text + " https://a.example/1", source="telegram",
                  published_at="2026-11-01T10:00:00Z")
    b = make_post("200", text + " https://b.example/2", source="x", author="a2",
                  published_at="2026-11-01T11:00:00Z")
    media = make_post("300", "", source="telegram", author="a3",
                      published_at="2026-11-01T12:00:00Z")
    [t] = trace_origins([a, b, media], [narr(a.post_id, b.post_id, media.post_id)])
    inf = edges_of(t, "inferred")
    assert [(e.from_, e.to) for e in inf] == [("telegram:100", "x:200")]
    assert "1 member post(s) had too little text" in t.note


def test_exclusion_note_states_each_reason():
    ok = make_post("1", "kept post with a verified time", published_at="2026-11-01T10:00:00Z")
    dup = make_post("2", "dup", published_at="2026-11-01T10:05:00Z", duplicate_of="x:1")
    unv = make_post("3", "unverified", verified=False)
    out = make_post("4", "outside", published_at="2026-12-01T10:00:00Z", in_window=False)
    bad = make_post("5", "verified flag but garbage time", published_at="not-a-date")
    [t] = trace_origins([ok, dup, unv, out, bad],
                        [narr("x:1", "x:2", "x:3", "x:4", "x:5", "x:missing")])
    assert t.earliest_post_ids == ["x:1"]
    note = t.note
    assert "5 member post(s) excluded" in note
    assert "1 duplicate of another collected post" in note
    assert "1 no verified platform timestamp" in note
    assert "1 outside the date window" in note
    assert "1 unparseable platform timestamp" in note
    assert "1 not in the collected posts" in note


def test_all_duplicates_note_does_not_blame_timestamps():
    dup = make_post("2", "dup", published_at="2026-11-01T10:05:00Z", duplicate_of="x:1")
    [t] = trace_origins([dup], [narr("x:2")])
    assert t.earliest_post_ids == []
    assert "1 duplicate of another collected post" in t.note
    assert "no verified platform timestamp" not in t.note


# ------------------------------------------------ earliest-post match and hubs


def _fake_embed(monkeypatch, vec_by_text: dict[str, list[float]], space: str = "dense"):
    """Deterministic vectors per comparable text (no model, no TF-IDF)."""
    calls = []

    def fake(texts):
        calls.append(list(texts))
        arr = np.array([vec_by_text[t] for t in texts], dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True), space

    monkeypatch.setattr(origin, "_embed", fake)
    return calls


ON = [1.0, 0.0]
OFF = [0.0, 1.0]
CLOSE = [0.95, 0.05]


def _members(texts_vecs, start_hour=10, **kw):
    posts, vecs = [], {}
    for i, (text, vec) in enumerate(texts_vecs):
        posts.append(make_post(str(i), text, author=f"a{i}",
                               published_at=f"2026-11-01T{start_hour + i:02d}:00:00Z", **kw))
        vecs[origin.comparable_text(text)] = vec
    return posts, vecs


def test_weak_earliest_points_to_first_strong(monkeypatch):
    """Like sample-run N1: the earliest is an off-narrative reply, later posts match."""
    posts, vecs = _members([
        ("wishing all these banks would just go down already", OFF),
        ("the bank app is down again cannot log in at all", ON),
        ("bank outage today, card payments failing everywhere", ON),
        ("online banking down for hours, customers locked out", CLOSE),
        ("major bank outage confirmed by many users this morning", ON),
    ])
    _fake_embed(monkeypatch, vecs)
    [t] = trace_origins(posts, [narr(*[p.post_id for p in posts])])
    assert t.earliest_post_ids == ["x:0"]
    assert t.earliest_match == "weak"
    assert t.earliest_similarity is not None and t.earliest_similarity < 0.2
    assert t.first_strong_post_id == "x:1"
    assert "loosely matches" in t.note and "x:1" in t.note
    assert "origin" not in t.note.lower().replace("not necessarily the true origin", "")


def test_strong_earliest_is_its_own_first_strong(monkeypatch):
    posts, vecs = _members([(f"bank outage report number {i} from users", ON) for i in range(5)])
    _fake_embed(monkeypatch, vecs)
    [t] = trace_origins(posts, [narr(*[p.post_id for p in posts])])
    assert t.earliest_match == "strong"
    assert t.earliest_similarity == pytest.approx(1.0, abs=1e-4)
    assert t.first_strong_post_id == t.earliest_post_ids[0] == "x:0"
    assert "loosely matches" not in t.note


def test_match_not_computed_for_small_narratives(monkeypatch):
    posts, vecs = _members([(f"bank outage report number {i} from users", ON)
                            for i in range(origin.MIN_MATCH_MEMBERS - 1)])
    _fake_embed(monkeypatch, vecs)
    [t] = trace_origins(posts, [narr(*[p.post_id for p in posts])])
    assert (t.earliest_match, t.earliest_similarity, t.first_strong_post_id) == (None, None, None)


def test_earliest_without_text_not_judged_but_first_strong_found(monkeypatch):
    media = make_post("m", "https://img.example/1.jpg", published_at="2026-11-01T09:00:00Z")
    posts, vecs = _members([(f"bank outage report number {i} from users", ON) for i in range(5)])
    _fake_embed(monkeypatch, vecs)
    [t] = trace_origins([media] + posts, [narr(media.post_id, *[p.post_id for p in posts])])
    assert t.earliest_post_ids == ["x:m"]
    assert t.earliest_match is None and t.earliest_similarity is None
    assert t.first_strong_post_id == "x:0"


def test_unverified_member_never_first_strong(monkeypatch):
    unv = make_post("u", "bank outage report from an unverified time", verified=False,
                    published_at="2026-11-01T01:00:00Z")
    posts, vecs = _members([("wishing all these banks would just go down already", OFF)]
                           + [(f"bank outage report number {i} from users", ON) for i in range(5)])
    vecs[origin.comparable_text(unv.text)] = ON
    _fake_embed(monkeypatch, vecs)
    [t] = trace_origins([unv] + posts, [narr(unv.post_id, *[p.post_id for p in posts])])
    assert t.earliest_post_ids == ["x:0"] and t.earliest_match == "weak"
    assert t.first_strong_post_id == "x:1"


def test_relative_bar_in_loose_narrative(monkeypatch):
    """A loose narrative (all members ~0.6 to each other): the bar drops to the
    narrative's own 10th percentile, so a typical earliest post is not called weak."""
    rng = np.random.default_rng(0)
    base = np.array([1.0, 0, 0, 0, 0, 0])
    texts_vecs = []
    for i in range(8):
        v = base + rng.normal(0, 0.9, 6)
        texts_vecs.append((f"loosely related bank post number {i} here", v.tolist()))
    posts, vecs = _members(texts_vecs)
    _fake_embed(monkeypatch, vecs)
    ordered = origin._comparable_sorted(posts, {p.post_id: origin.parse_ts(p.published_at) for p in posts})
    arr, space = origin._embed([origin.comparable_text(p.text) for p in ordered])
    scores, bar = origin._match_scores(ordered, arr, space)
    assert bar < origin.MATCH_STRONG_ABS["dense"]
    assert bar == pytest.approx(np.percentile(list(scores.values()), origin.MATCH_TYPICAL_QUANTILE))
    assert sum(s >= bar for s in scores.values()) >= 7  # all but the bottom tenth


def test_match_scores_leave_one_out():
    """A post never vouches for itself: its score uses the centroid of the others."""
    ps = [make_post(str(i), f"text number {i} words", published_at=f"2026-11-01T1{i}:00:00Z")
          for i in range(5)]
    vecs = np.array([ON, OFF, OFF, OFF, OFF], dtype=np.float32)
    scores, _bar = origin._match_scores(ps, vecs, "dense")
    assert scores["x:0"] == pytest.approx(0.0)  # including itself would give 1/sqrt(5)
    assert scores["x:1"] == pytest.approx(3 / np.sqrt(10))  # others: ON + 3 x OFF


def test_embeddings_computed_once_per_narrative(monkeypatch):
    posts, vecs = _members([(f"bank outage report number {i} from users", ON) for i in range(5)])
    calls = _fake_embed(monkeypatch, vecs)
    trace_origins(posts, [narr(*[p.post_id for p in posts])])
    assert len(calls) == 1  # shared by inferred edges and match scores


def test_hubs_count_direct_edges_from_each_post():
    video = make_post("vid", "a long video about the bank outage today", source="youtube",
                      published_at="2026-11-01T08:00:00Z")
    other = make_post("o", "another post on the outage that people quote", author="o1",
                      published_at="2026-11-01T08:30:00Z")
    replies = [make_post(f"c{i}", f"comment {i} on it", source="youtube", author=f"c{i}",
                         published_at=f"2026-11-01T09:{i:02d}:00Z",
                         relations=[Relation(type="reply", target_post_id="youtube:vid")])
               for i in range(3)]
    linker = make_post("l", "see this", author="l1", published_at="2026-11-01T10:00:00Z",
                       relations=[Relation(type="link", target_post_id="youtube:vid")])
    quoter = make_post("q", "quoting", author="q1", published_at="2026-11-01T10:30:00Z",
                       relations=[Relation(type="quote", target_post_id="x:o")])
    allp = [video, other, *replies, linker, quoter]
    [t] = trace_origins(allp, [narr(*[p.post_id for p in allp])])
    assert [h.post_id for h in t.hubs] == ["youtube:vid", "x:o"]
    vid = t.hubs[0]
    assert vid.direct_out == 4
    assert vid.relations == {"reply": 3, "link": 1}
    assert list(vid.relations) == ["reply", "link"]  # most frequent first
    assert t.hubs[1].direct_out == 1 and t.hubs[1].relations == {"quote": 1}
    # Hubs summarise edges; they never add or remove any.
    assert sum(h.direct_out for h in t.hubs) == len(edges_of(t, "direct"))


def test_hubs_capped_and_count_inferred_out():
    ps = []
    for k in range(origin.MAX_HUBS + 2):
        src = make_post(f"s{k}", f"source post {k}", author=f"s{k}",
                        published_at=f"2026-11-01T0{k}:00:00Z")
        kids = [make_post(f"s{k}r{j}", f"reply {j}", author=f"r{k}{j}",
                          published_at=f"2026-11-01T0{k}:3{j}:00Z",
                          relations=[Relation(type="reply", target_post_id=src.post_id)])
                for j in range(k + 1)]
        ps += [src, *kids]
    hubs = origin._hubs(trace_origins(ps, [narr(*[p.post_id for p in ps])])[0].edges)
    assert len(hubs) == origin.MAX_HUBS
    assert [h.direct_out for h in hubs] == sorted((h.direct_out for h in hubs), reverse=True)
    from socmint.schema import Edge
    edges = [Edge(**{"from": "x:a"}, to="x:b", type="direct", basis="reply of x:a", relation="reply"),
             Edge(**{"from": "x:a"}, to="x:c", type="inferred", basis="similar", similarity=0.9),
             Edge(**{"from": "x:z"}, to="x:d", type="inferred", basis="similar", similarity=0.9)]
    [hub] = origin._hubs(edges)  # inferred-only posts are not hubs
    assert (hub.post_id, hub.direct_out, hub.inferred_out) == ("x:a", 1, 1)


def test_platform_first_seen_strong_skips_loose_anchor(monkeypatch):
    """Like N2 in the re-analysed run: a loose YouTube post comes first, so the raw
    platform path starts on YouTube; the close-match path must start on X."""
    loose = make_post("y0", "choosing a bank account for your savings this year",
                      source="youtube", published_at="2026-11-01T08:00:00Z")
    posts, vecs = _members([(f"bank reported stock trades in the turkish market {i}", ON)
                            for i in range(4)])
    yt = make_post("y1", "bank stock trades in the turkish market explained",
                   source="youtube", published_at="2026-11-01T20:00:00Z")
    vecs[origin.comparable_text(loose.text)] = OFF
    vecs[origin.comparable_text(yt.text)] = CLOSE
    _fake_embed(monkeypatch, vecs)
    allp = [loose, *posts, yt]
    [t] = trace_origins(allp, [narr(*[p.post_id for p in allp])])
    assert t.earliest_match == "weak"
    assert t.platform_sequence[0] == "youtube"  # raw path still recorded as observed
    assert t.platform_first_seen_strong == {"x": "2026-11-01T10:00:00Z",
                                            "youtube": "2026-11-01T20:00:00Z"}
    assert list(t.platform_first_seen_strong) == ["x", "youtube"]  # time order


def test_platform_first_seen_strong_empty_when_not_judged(monkeypatch):
    posts, vecs = _members([(f"bank outage report number {i} from users", ON)
                            for i in range(origin.MIN_MATCH_MEMBERS - 1)])
    _fake_embed(monkeypatch, vecs)
    [t] = trace_origins(posts, [narr(*[p.post_id for p in posts])])
    assert t.platform_first_seen_strong == {}


def test_platform_first_seen_strong_matches_raw_when_all_close(monkeypatch):
    posts, vecs = _members([(f"bank outage report number {i} from users", ON) for i in range(5)])
    _fake_embed(monkeypatch, vecs)
    [t] = trace_origins(posts, [narr(*[p.post_id for p in posts])])
    assert t.platform_first_seen_strong == t.platform_first_seen


# --------------------------- relative "loose match" bar (10th percentile AND abs bar)


def _at(deg: float) -> list[float]:
    r = np.deg2rad(deg)
    return [float(np.cos(r)), float(np.sin(r))]


def _angled(topic_texts: list[str], degs: list[float]):
    """Members at the given angles from a shared axis (the earliest first). Angles
    alternate sides so the leave-one-out centroid stays near the axis."""
    return _members([(t, _at(d if i % 2 == 0 else -d))
                     for i, (t, d) in enumerate(zip(topic_texts, degs, strict=True))])


RECALL = [  # product recall, not a bank: earliest is on-narrative, just outer-quarter
    "Recall alert: several stroller brands pulled after reports of folding hinges",
    "Brand X recalls its foldable strollers over a hinge that can pinch fingers",
    "stroller recall announced, hinge defect, stop using it now",
    "consumer agency confirms stroller recall due to hinge injuries",
    "my kid's stroller is on the recall list, hinge problem",
    "full list of recalled stroller models, hinge can collapse",
    "retailers pull recalled strollers from shelves after hinge reports",
    "how to get a refund for the recalled stroller with the bad hinge",
    "stroller hinge recall: what parents need to know",
    "the hinge defect recall covers strollers sold since 2024",
]


def test_outer_quarter_earliest_is_strong_not_loose(monkeypatch):
    """Like sample-run N4 (an outage round-up at 0.717, 25th percentile 0.728): an
    on-narrative earliest post that sits in the narrative's outer quarter but not in its
    bottom tenth is strong. Shown on a product-recall narrative."""
    degs = [42, 52, 36, 34, 30, 26, 22, 18, 14, 10]  # earliest 2nd-furthest of 10
    posts, vecs = _angled(RECALL, degs)
    _fake_embed(monkeypatch, vecs)
    [t] = trace_origins(posts, [narr(*[p.post_id for p in posts])])
    sim = t.earliest_similarity
    assert sim is not None and sim < origin.MATCH_STRONG_ABS["dense"]
    others = sorted(origin._match_scores(
        origin._comparable_sorted(posts, {p.post_id: origin.parse_ts(p.published_at)
                                          for p in posts}),
        np.array([vecs[origin.comparable_text(p.text)] for p in posts]), "dense")[0].values())
    assert sim < float(np.percentile(others, 25))  # the old bar would call it weak
    assert t.earliest_match == "strong"
    assert t.first_strong_post_id == "x:0"
    assert "loosely matches" not in t.note


def test_bottom_decile_earliest_below_abs_bar_is_weak(monkeypatch):
    """Election narrative: the earliest post is the narrative's clear outlier (bottom
    tenth) and well under the absolute bar, so it is a loose match."""
    texts = ["polling stations open late tomorrow, bring your ID",
             "claims that ballots were shredded in district 4 are spreading",
             "video alleges shredded ballots in district 4, officials deny it",
             "district 4 shredded ballot rumour reaches 1M views",
             "election board: no ballots were shredded in district 4",
             "fact check: district 4 ballot shredding video is from 2019",
             "they shredded our ballots in district 4!!",
             "district 4 shredding claim shared by candidate's campaign",
             "reporters visit district 4 warehouse after shredding claim",
             "shredded ballots story: what we know so far"]
    degs = [70, 25, 20, 18, 15, 12, 10, 8, 5, 3]
    posts, vecs = _angled(texts, degs)
    _fake_embed(monkeypatch, vecs)
    [t] = trace_origins(posts, [narr(*[p.post_id for p in posts])])
    assert t.earliest_match == "weak"
    assert t.first_strong_post_id == "x:1"
    assert "loosely matches" in t.note and "x:1" in t.note


def test_bottom_decile_but_above_abs_bar_is_strong(monkeypatch):
    """Near-copy vaccine narrative (members ~0.99 to the centre): the earliest is its
    furthest member yet still above the absolute similar-content bar, so it is not
    called loose just because the others are near-identical copies."""
    texts = [f"health agency pauses vaccine lot 7 after reports, update {i}" for i in range(8)]
    degs = [30, 3, 2, 2, 1, 1, 0.5, 0.5]
    posts, vecs = _angled(texts, degs)
    _fake_embed(monkeypatch, vecs)
    [t] = trace_origins(posts, [narr(*[p.post_id for p in posts])])
    assert t.earliest_similarity is not None
    assert t.earliest_similarity >= origin.MATCH_STRONG_ABS["dense"]
    assert t.earliest_match == "strong"


def test_relative_bar_uses_tenth_percentile_of_own_members():
    """The bar is min(absolute bar, the narrative's 10th percentile), whatever the topic."""
    ps = [make_post(str(i), f"public figure resignation rumour post {i}",
                    published_at=f"2026-11-01T1{i}:00:00Z") for i in range(6)]
    vecs = np.array([_at(d) for d in (60, -40, 35, -30, 25, -20)], dtype=np.float32)
    scores, bar = origin._match_scores(ps, vecs, "dense")
    p10 = float(np.percentile(list(scores.values()), 10))
    assert bar == pytest.approx(min(origin.MATCH_STRONG_ABS["dense"], p10))
    assert origin.MATCH_TYPICAL_QUANTILE == 10
