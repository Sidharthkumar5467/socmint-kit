"""socmint.collect.window: time slices, shared cap, phrase matching."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from socmint.collect.window import (
    SliceShare,
    is_search_expression,
    match_mode,
    search_topic,
    time_slices,
    topic_matches,
)
from socmint.interfaces import Query
from socmint.util.timeutil import day_end, day_start


def Q(topic="bank of america", match=None, start=date(2026, 11, 13), end=date(2026, 11, 19)):
    opts = {"match": match} if match else {}
    return Query(topic=topic, start=day_start(start), end=day_end(end), limit=300, options=opts)


# ---------------------------------------------------------------- slices
def test_one_slice_per_day_newest_first():
    q = Q()
    sl = time_slices(q.start, q.end)
    assert len(sl) == 7
    assert sl[0] == (day_start(date(2026, 11, 19)), day_end(date(2026, 11, 19)))
    assert sl[-1] == (day_start(date(2026, 11, 13)), day_end(date(2026, 11, 13)))
    # contiguous, no gaps or overlaps
    for (s1, _e1), (_s2, e2) in zip(sl, sl[1:], strict=False):
        assert (s1 - e2).total_seconds() == 1


def test_long_window_groups_days_and_covers_everything():
    q = Q(start=date(2026, 10, 21), end=date(2026, 11, 19))  # 30 days
    sl = time_slices(q.start, q.end, max_slices=10)
    assert len(sl) == 10
    assert sl[0][1] == q.end and sl[-1][0] == q.start
    days = sum((e.date() - s.date()).days + 1 for s, e in sl)
    assert days == 30


def test_uneven_grouping_and_single_day_and_empty():
    q = Q(start=date(2026, 11, 9), end=date(2026, 11, 19))  # 11 days -> 4 slices: 3,3,3,2
    sl = time_slices(q.start, q.end, max_slices=4)
    assert [(e.date() - s.date()).days + 1 for s, e in sl] == [3, 3, 3, 2]
    one = Q(start=date(2026, 11, 19), end=date(2026, 11, 19))
    assert time_slices(one.start, one.end) == [(one.start, one.end)]
    assert time_slices(one.end, one.start) == []


def test_partial_day_window_is_clipped():
    s = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)
    e = datetime(2026, 11, 19, 6, 0, tzinfo=UTC)
    assert time_slices(s, e) == [(datetime(2026, 11, 19, tzinfo=UTC), e),
                                 (s, datetime(2026, 11, 18, 23, 59, 59, tzinfo=UTC))]


# ---------------------------------------------------------------- share
def test_share_splits_and_rolls_over_unused():
    sh = SliceShare(300, 7)
    assert sh.allowance(0) == 43
    sh.record(0, 43)
    assert sh.allowance(1) == 43
    sh.record(1, 5)  # quiet day: 38 unused
    assert sh.allowance(2) == 51  # (300-48)/5 rounded up
    sh.record(2, 51)
    for i in range(3, 7):
        sh.record(i, sh.allowance(i))
    assert sh.used == 300


def test_share_never_exceeds_total_and_summary():
    sh = SliceShare(10, 3)
    caps = []
    for i in range(3):
        caps.append(sh.allowance(i))
        sh.record(i, caps[-1])
    assert sum(caps) == 10 and sh.used == 10
    sl = time_slices(*[day_start(date(2026, 11, 17)), day_end(date(2026, 11, 19))])
    sh2 = SliceShare(9, 3)
    sh2.allowance(0)
    sh2.record(0, 3)
    sh2.allowance(1)
    sh2.record(1, 1)
    s = sh2.summary(sl)
    assert s.startswith("per-day: 2026-11-19 3, 2026-11-18 1 (fewer than its share of 3)")
    assert "2026-11-17 not reached" in s


# ---------------------------------------------------------------- phrase
@pytest.mark.parametrize("topic,expected", [
    ("bank of america", '"bank of america"'),
    ("  bank   of america ", '"bank of america"'),
    ("tesla", "tesla"),
    ("#BankOfAmerica", "#BankOfAmerica"),
    ('"bank of america" fraud', '"bank of america" fraud'),
    ("vaccine OR vaccination", "vaccine OR vaccination"),
    ("bank of america -stadium", "bank of america -stadium"),
    ("from:BofA_Help outage", "from:BofA_Help outage"),
    ("(bank of america)", "(bank of america)"),
])
def test_search_topic_phrase_mode(topic, expected):
    assert search_topic(Q(topic)) == expected


def test_words_mode_and_unknown_mode():
    assert search_topic(Q(match="words")) == "bank of america"
    assert match_mode(Q(match="WORDS")) == "words"
    assert match_mode(Q(match="nonsense")) == "phrase"
    assert match_mode(Q()) == "phrase"


def test_hyphenated_words_are_not_operators():
    assert not is_search_expression("state-sponsored hackers")
    assert search_topic(Q("state-sponsored hackers")) == '"state-sponsored hackers"'
    assert is_search_expression("hackers -stadium")


def test_topic_matches_client_side():
    q = Q()
    assert topic_matches(q, "Outage at Bank of America today")
    assert topic_matches(q, "BANK-OF-AMERICA\nlogin down")
    assert not topic_matches(q, "bank stadium in america")
    assert not topic_matches(q, None)
    assert topic_matches(Q(match="words"), "bank stadium in america")
    assert topic_matches(Q("tesla"), "nothing here")  # single word: platform's job
    assert topic_matches(Q("bank of america OR bofa"), "bofa")  # expression: platform's job


def test_fill_slices_gives_unreached_share_to_reached_days():
    from socmint.collect.window import fill_slices
    # Telegram live case: only the newest day reached, 900 candidates, cap 300
    assert fill_slices([900, 0, 0, 0, 0, 0, 0], 300) == [300, 0, 0, 0, 0, 0, 0]
    # quiet day's leftover goes to the others
    assert fill_slices([100, 5, 100], 90) == [43, 5, 42]
    assert sum(fill_slices([10, 10, 10], 100)) == 30
    assert fill_slices([], 10) == [] and fill_slices([3, 3], 0) == [0, 0]
    k = fill_slices([7, 1, 50, 2, 9, 0, 30], 40)
    assert sum(k) == 40 and all(a <= b for a, b in zip(k, [7, 1, 50, 2, 9, 0, 30], strict=True))
