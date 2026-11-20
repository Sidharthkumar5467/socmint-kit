"""Tests for socmint.analyze.coordination (synthetic posts; no network)."""

from __future__ import annotations

from conftest import make_post

from socmint.analyze.coordination import (
    GENERIC_CAVEAT,
    detect_coordination,
    is_placeholder_image,
    normalize_text,
    phash_distance,
    shingles,
)
from socmint.schema import Media, Relation

TEXT = ("Breaking: the new port deal hands control of the harbour to a foreign company "
        "and nobody voted for it, share before they delete this")
OTHER = ("Weather update for the weekend: light rain on Saturday morning and sunny "
         "spells across the coast through Sunday afternoon")


def _types(cluster) -> set[str]:
    return {s.type for s in cluster.signals}


# ------------------------------------------------------------------- helpers


def test_normalize_text_strips_urls_mentions_punct():
    t = normalize_text("Hello @bob!!  See https://ex.com/x?y=1 NOW, ok?")
    assert t == "hello see now ok"


def test_shingles_short_text_ignored():
    assert shingles("too short to compare") is None
    assert shingles(TEXT)


def test_shingles_unspaced_script_fallback():
    cjk = "港口协议将港口控制权交给外国公司没有人投票支持请在他们删除之前分享这个消息"
    assert shingles(cjk)


def test_phash_distance():
    assert phash_distance("ff00", "ff01") == 1
    assert phash_distance("ff00", "zz") is None


def test_placeholder_detection():
    assert is_placeholder_image("https://abs.twimg.com/sticky/default_profile_images/x.png")
    assert not is_placeholder_image("https://pbs.twimg.com/profile_images/123/a.jpg")


# ---------------------------------------------------------- required scenarios


def test_a_near_dup_and_simultaneous_flagged_with_evidence():
    posts = [
        make_post("1", TEXT, author="a1", published_at="2026-11-01T10:00:00Z"),
        make_post("2", TEXT + "!!", author="a2", published_at="2026-11-01T10:03:00Z"),
        make_post("3", "RT-free copy: " + TEXT, author="a3", published_at="2026-11-01T10:06:00Z"),
        make_post("4", OTHER, author="a4", published_at="2026-11-01T10:01:00Z"),
    ]
    clusters = detect_coordination(posts)
    assert len(clusters) == 1
    c = clusters[0]
    assert c.id == "C1"
    assert c.member_author_ids == ["x:a1", "x:a2", "x:a3"]
    assert _types(c) == {"near_duplicate_text", "near_simultaneous_posting"}
    for s in c.signals:
        assert s.evidence_post_ids and 0 < s.score <= 1 and s.detail
    assert set(c.supporting_post_ids) == {"x:1", "x:2", "x:3"}
    assert GENERIC_CAVEAT in c.benign_explanations
    assert c.sources == ["x"] and "@a1" in c.member_handles


def test_b_near_duplicate_only_not_flagged():
    # Same text but days apart and no other shared signal -> one signal type only.
    posts = [
        make_post("1", TEXT, author="a1", published_at="2026-11-01T10:00:00Z"),
        make_post("2", TEXT, author="a2", published_at="2026-11-03T10:00:00Z"),
        make_post("3", TEXT, author="a3", published_at="2026-11-05T10:00:00Z"),
    ]
    assert detect_coordination(posts) == []


def test_c_amplification_loop_plus_creation_dates_flagged():
    posts = [
        make_post("1", "original one", author="a1", author_created_at="2026-10-01T00:00:00Z"),
        make_post("2", "original two", author="a2", author_created_at="2026-10-02T12:00:00Z"),
        make_post("3", "", author="a1", relations=[Relation(type="repost", target_post_id="x:2")]),
        make_post("4", "", author="a2", relations=[Relation(type="repost", target_post_id="x:1")]),
    ]
    clusters = detect_coordination(posts)
    assert len(clusters) == 1
    c = clusters[0]
    assert _types(c) == {"amplification_loop", "similar_creation_dates"}
    amp = next(s for s in c.signals if s.type == "amplification_loop")
    assert set(amp.evidence_post_ids) == {"x:3", "x:4"}
    assert amp.score == 1.0


def test_creation_dates_alone_never_flag():
    posts = [
        make_post("1", "hello there", author="a1", author_created_at="2026-10-01T00:00:00Z"),
        make_post("2", "other words", author="a2", author_created_at="2026-10-01T00:00:00Z"),
    ]
    assert detect_coordination(posts) == []


def test_d_reposts_not_counted_as_near_duplicate():
    # a2 and a3 repost a1's text verbatim at the same time: that is amplification
    # (one-way, below the repeat threshold), not copy-paste coordination.
    posts = [
        make_post("1", TEXT, author="a1", published_at="2026-11-01T10:00:00Z"),
        make_post("2", TEXT, author="a2", published_at="2026-11-01T10:01:00Z",
                  relations=[Relation(type="repost", target_post_id="x:1")]),
        make_post("3", TEXT, author="a3", published_at="2026-11-01T10:02:00Z",
                  relations=[Relation(type="forward", target_post_id="x:1")]),
    ]
    assert detect_coordination(posts) == []


def test_e_unverified_timestamps_not_used_for_simultaneity():
    posts = [
        make_post("1", TEXT, author="a1", published_at="2026-11-01T10:00:00Z", verified=False),
        make_post("2", TEXT, author="a2", published_at="2026-11-01T10:01:00Z", verified=False),
    ]
    assert detect_coordination(posts) == []
    # Same posts with verified timestamps DO fire both signals (control).
    for p in posts:
        p.timestamp_verified = True
    clusters = detect_coordination(posts)
    assert len(clusters) == 1
    assert "near_simultaneous_posting" in _types(clusters[0])


def test_f_default_avatar_ignored():
    default = "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png"
    posts = [
        make_post("1", TEXT, author="a1", published_at="2026-11-01T10:00:00Z",
                  author_avatar_url=default),
        make_post("2", TEXT, author="a2", published_at="2026-11-04T10:00:00Z",
                  author_avatar_url=default),
    ]
    # Near-dup text + (ignored) default avatar = one signal type -> not flagged.
    assert detect_coordination(posts) == []

    # A real shared avatar URL is a second signal -> flagged.
    real = "https://pbs.twimg.com/profile_images/123/same.jpg"
    for p in posts:
        p.author_avatar_url = real
    clusters = detect_coordination(posts)
    assert len(clusters) == 1
    assert _types(clusters[0]) == {"near_duplicate_text", "reused_image"}


# ------------------------------------------------------------------ extra cases


def test_shared_link_simultaneous_plus_phash_and_syndication_note():
    url = "https://news.example.org/story/42"
    posts = [
        make_post("1", "look", author="a1", published_at="2026-11-01T10:00:00Z",
                  relations=[Relation(type="link", target_url=url)],
                  media=[Media(type="photo", url="https://img.test/a.jpg")]),
        make_post("2", "see", author="a2", published_at="2026-11-01T10:02:00Z",
                  relations=[Relation(type="link", target_url=url + "/")],
                  media=[Media(type="photo", url="https://img.test/b.jpg")]),
    ]
    # Without hashes: only simultaneity -> not flagged.
    assert detect_coordination(posts) == []
    hashes = {"https://img.test/a.jpg": "ffffffff00000000",
              "https://img.test/b.jpg": "ffffffff00000001"}
    clusters = detect_coordination(posts, image_hashes=hashes)
    assert len(clusters) == 1
    c = clusters[0]
    assert _types(c) == {"near_simultaneous_posting", "reused_image"}
    assert any("news.example.org" in b for b in c.benign_explanations)


def test_repeated_one_way_amplification_and_same_name_note():
    posts = [make_post("o", "official statement", author="brand", author_name="Acme News",
                       author_avatar_url="https://cdn.test/acme.png")]
    posts += [
        make_post(f"r{i}", "", source="x", author="fan", author_name="Acme News",
                  author_avatar_url="https://cdn.test/acme.png",
                  relations=[Relation(type="repost", target_handle="@brand")])
        for i in range(3)
    ]
    clusters = detect_coordination(posts)
    assert len(clusters) == 1
    c = clusters[0]
    assert _types(c) == {"amplification_loop", "reused_image"}
    notes = " ".join(c.benign_explanations)
    assert "display name 'acme news'" in notes
    assert "repeatedly reposted" in notes


def test_single_author_never_cluster_and_ids_sorted():
    # Same author posting the same text twice is not coordination.
    solo = [make_post("1", TEXT, author="a1"), make_post("2", TEXT, author="a1")]
    assert detect_coordination(solo) == []

    big = [make_post(f"b{i}", TEXT, author=f"b{i}", published_at="2026-11-01T10:00:00Z")
           for i in range(3)]
    small = [make_post(f"s{i}", OTHER, author=f"s{i}", published_at="2026-11-02T10:00:00Z")
             for i in range(2)]
    clusters = detect_coordination(big + small)
    assert [c.id for c in clusters] == ["C1", "C2"]
    assert len(clusters[0].member_author_ids) == 3


def test_posts_without_author_or_duplicates_skipped():
    p1 = make_post("1", TEXT, author="a1")
    p2 = make_post("2", TEXT, author="a2", duplicate_of="x:1")
    p3 = make_post("3", TEXT, author="a3")
    p3.author_id = None
    p3.author_handle = None
    assert detect_coordination([p1, p2, p3]) == []


# ------------------------------------------------- review regressions (two signals agree)

UNRELATED = ("Recipe of the week: slow roasted tomatoes with garlic, thyme and a splash "
             "of olive oil served over fresh pasta for dinner")


def test_second_signal_on_one_edge_does_not_flag_text_only_chain():
    # a0..a4 post the same text on 5 different days (text only). a4 and an unrelated
    # account F share a real avatar. No pair has two signal types -> nothing flagged.
    avatar = "https://pbs.twimg.com/profile_images/9/shared.jpg"
    posts = [make_post(f"t{i}", TEXT, author=f"a{i}",
                       published_at=f"2026-11-0{i + 1}T10:00:00Z",
                       author_avatar_url=avatar if i == 4 else None)
             for i in range(5)]
    posts.append(make_post("f", UNRELATED, author="F", published_at="2026-11-07T10:00:00Z",
                           author_avatar_url=avatar))
    assert detect_coordination(posts) == []


def test_signals_on_disjoint_pairs_do_not_bridge_accounts():
    # A-B same text 4 days apart; B-E share an avatar. A and E share nothing.
    avatar = "https://pbs.twimg.com/profile_images/9/b.jpg"
    posts = [
        make_post("a", TEXT, author="A", published_at="2026-11-01T10:00:00Z"),
        make_post("b", TEXT, author="B", published_at="2026-11-05T10:00:00Z",
                  author_avatar_url=avatar),
        make_post("e", UNRELATED, author="E", published_at="2026-11-06T10:00:00Z",
                  author_avatar_url=avatar),
    ]
    assert detect_coordination(posts) == []


def test_only_doubly_linked_members_flagged_in_larger_component():
    # a0..a4 share text (days apart); only a0 and a1 also share an avatar.
    # Flag exactly {a0, a1}; a2..a4 are linked by text alone and stay out.
    avatar = "https://pbs.twimg.com/profile_images/9/pair.jpg"
    posts = [make_post(f"t{i}", TEXT, author=f"a{i}",
                       published_at=f"2026-11-0{i + 1}T10:00:00Z",
                       author_avatar_url=avatar if i < 2 else None)
             for i in range(5)]
    clusters = detect_coordination(posts)
    assert len(clusters) == 1
    assert clusters[0].member_author_ids == ["x:a0", "x:a1"]
    assert _types(clusters[0]) == {"near_duplicate_text", "reused_image"}


def test_out_of_window_posts_are_not_evidence():
    posts = [
        make_post("1", TEXT, author="a1", published_at="2026-09-01T10:00:00Z", in_window=False),
        make_post("2", TEXT, author="a2", published_at="2026-09-01T10:05:00Z", in_window=False),
    ]
    assert detect_coordination(posts) == []
    for p in posts:  # control: the same posts in the window are flagged
        p.in_window = True
    assert len(detect_coordination(posts)) == 1


def test_in_window_repost_of_out_of_window_original_resolves_target():
    # fan reposts brand's (out-of-window) post 3 times in the window; shared avatar.
    av = "https://cdn.test/acme.png"
    posts = [make_post("o", "official statement", author="brand", in_window=False,
                       published_at="2026-09-01T10:00:00Z", author_avatar_url=av)]
    posts += [make_post(f"r{i}", "", author="fan", author_avatar_url=av,
                        relations=[Relation(type="repost", target_post_id="x:o")])
              for i in range(3)]
    # brand's only post is out of window, so its avatar is not evidence: one signal type.
    assert detect_coordination(posts) == []
    posts.append(make_post("o2", "another statement", author="brand", author_avatar_url=av))
    clusters = detect_coordination(posts)
    assert len(clusters) == 1
    amp = next(s for s in clusters[0].signals if s.type == "amplification_loop")
    assert set(amp.evidence_post_ids) == {"x:r0", "x:r1", "x:r2"}  # never the old original


def test_one_way_amplification_score_is_mean_of_edge_scores():
    # f1 reposts brand 3x (0.5), f2 reposts brand 7x (0.9); all share an avatar.
    av = "https://cdn.test/shared.png"
    posts = [make_post("o", "official statement", author="brand", author_avatar_url=av)]
    for fan, n in (("f1", 3), ("f2", 7)):
        posts += [make_post(f"{fan}{i}", "", author=fan, author_avatar_url=av,
                            relations=[Relation(type="repost", target_handle="@brand")])
                  for i in range(n)]
    clusters = detect_coordination(posts)
    assert len(clusters) == 1
    amp = next(s for s in clusters[0].signals if s.type == "amplification_loop")
    assert amp.score == 0.7


# ------------------------------------------------------------- cluster summaries


def test_summary_text_and_timing_single_platform():
    posts = [
        make_post("1", TEXT, author="a1", published_at="2026-11-01T10:00:00Z"),
        make_post("2", TEXT, author="a2", published_at="2026-11-01T10:00:20Z"),
        make_post("3", TEXT, author="a3", published_at="2026-11-01T10:00:40Z"),
    ]
    c = detect_coordination(posts)[0]
    assert c.summary == "3 X accounts posted near-identical text, a median 20 s apart."


def test_summary_same_second_and_creation_dates():
    posts = [
        make_post(str(i), TEXT, source="telegram", author=f"a{i}",
                  published_at="2026-11-01T10:00:00Z",
                  author_created_at=f"2026-01-0{i}T00:00:00Z")
        for i in range(1, 5)
    ]
    posts[3].author_created_at = "2020-01-01T00:00:00Z"  # outside the 3-day window
    c = detect_coordination(posts)[0]
    assert c.summary == ("4 Telegram accounts posted near-identical text, within the same "
                         "second; 3 were created within 3 days of each other.")


def test_summary_all_created_together_and_multi_platform():
    posts = [
        make_post("1", TEXT, source="x", author="a1", published_at="2026-11-01T10:00:00Z",
                  author_created_at="2026-01-01T00:00:00Z"),
        make_post("2", TEXT, source="telegram", author="a2",
                  published_at="2026-11-01T10:02:00Z",
                  author_created_at="2026-01-02T00:00:00Z"),
    ]
    c = detect_coordination(posts)[0]
    assert c.summary == ("2 accounts across Telegram and X posted near-identical text, a "
                         "median 2.0 min apart; both were created within 3 days of each "
                         "other.")


def test_summary_link_timing_and_images():
    url = "https://news.example.org/story/42"
    posts = [
        make_post("1", "look", author="a1", published_at="2026-11-01T10:00:00Z",
                  relations=[Relation(type="link", target_url=url)],
                  media=[Media(type="photo", url="https://img.test/a.jpg")]),
        make_post("2", "see", author="a2", published_at="2026-11-01T10:02:00Z",
                  relations=[Relation(type="link", target_url=url)],
                  media=[Media(type="photo", url="https://img.test/a.jpg")]),
    ]
    c = detect_coordination(posts)[0]
    assert c.summary == ("2 X accounts shared the same link within 10 min of each other "
                         "(median gap 2.0 min); both used the same images.")


def test_summary_amplification_and_avatar():
    posts = [make_post("o", "official statement", author="brand",
                       author_avatar_url="https://cdn.test/acme.png")]
    posts += [make_post(f"r{i}", "", author="fan", author_avatar_url="https://cdn.test/acme.png",
                        relations=[Relation(type="repost", target_handle="@brand")])
              for i in range(3)]
    c = detect_coordination(posts)[0]
    assert c.summary == ("2 X accounts used the same profile picture; 1 repeatedly reposted "
                         "or forwarded @brand.")


def test_summary_amplification_loop_and_partial_text():
    # a1 <-> a2 reshare each other (loop) and also post the same text as each other;
    # a3 shares the text only with... nobody: a3 is not strongly linked, so not a member.
    posts = [
        make_post("1", TEXT, author="a1", published_at="2026-11-01T10:00:00Z"),
        make_post("2", TEXT, author="a2", published_at="2026-11-03T10:00:00Z"),
        make_post("3", "", author="a1", published_at="2026-11-01T11:00:00Z",
                  relations=[Relation(type="repost", target_post_id="x:2")]),
        make_post("4", "", author="a2", published_at="2026-11-03T11:00:00Z",
                  relations=[Relation(type="repost", target_post_id="x:1")]),
    ]
    c = detect_coordination(posts)[0]
    assert c.summary == ("2 X accounts posted near-identical text; both reposted or "
                         "forwarded each other in a loop.")


def test_cluster_summary_partial_lead_and_fallback():
    from socmint.analyze.coordination import cluster_summary
    stats = {"near_duplicate_text": {"accounts": 4, "posts": 4, "mean": 1.0},
             "near_simultaneous_posting": {"accounts": 3, "pairs": 3, "text_pairs": 2,
                                           "median_lag_s": 90.0}}
    assert cluster_summary(5, ["youtube", "linkedin_ads"], stats) == (
        "4 of 5 accounts across LinkedIn Ads and YouTube posted near-identical text; 3 "
        "posted the same content within 10 min of each other (median gap 1.5 min).")
    assert cluster_summary(2, ["x"], {}) == "2 X accounts acted alike on two or more signals."
