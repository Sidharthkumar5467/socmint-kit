"""Telegram collector: web-preview parser, Telethon mapping, flood handling, modes, auth.

Fixtures:
  tests/fixtures/telegram/live_preview_durov_search.html  live t.me/s/durov?q=telegram
  tests/fixtures/telegram/live_preview_telegram.html      live t.me/s/telegram
  tests/fixtures/telegram/live_preview_telegram_search_zero.html  live zero-result search
(fetched 2026-11-18 with a plain browser user-agent; no login, no cookies). The only
edit to the captures: expiring CDN `token=` query values replaced with `REDACTED` so
secret scanners don't flag them; the parser does not depend on them.
No test touches the network: HTTP is mocked with respx, Telethon with fake clients.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import respx
from telethon.errors import FloodWaitError, SessionPasswordNeededError

from socmint.collect import telegram as tg
from socmint.collect import telegram_auth as auth
from socmint.interfaces import CollectContext, Query
from socmint.schema import SourceState

FIX = Path(__file__).parent / "fixtures" / "telegram"
DUROV = (FIX / "live_preview_durov_search.html").read_text(encoding="utf-8")
TG_CHANNEL = (FIX / "live_preview_telegram.html").read_text(encoding="utf-8")
ZERO = (FIX / "live_preview_telegram_search_zero.html").read_text(encoding="utf-8")
COLLECTED = "2026-11-18T00:00:00Z"


def utc(*a: int) -> datetime:
    return datetime(*a, tzinfo=UTC)


def make_ctx(tmp_path: Path, cfg: dict[str, str] | None = None, budget: float = 60.0
             ) -> CollectContext:
    return CollectContext(run_dir=tmp_path, raw_dir=tmp_path / "raw", config=cfg or {},
                          deadline=time.monotonic() + budget)


def q(start: datetime, end: datetime, topic: str = "telegram", **opts) -> Query:
    return Query(topic=topic, start=start, end=end, limit=500, options=opts)


def by_id(posts, pid):
    return next(p for p in posts if p.post_id == pid)


# =================================================================== small helpers


@pytest.mark.parametrize("text,expected", [
    ("987", (987, False)), ("1,234", (1234, False)), ("12.6K", (12600, True)),
    ("7.24M", (7240000, True)), ("1.2B", (1200000000, True)), ("", (None, False)),
    (None, (None, False)), ("n/a", (None, False)),
])
def test_parse_count(text, expected):
    assert tg.parse_count(text) == expected


@pytest.mark.parametrize("raw,expected", [
    ("@Durov", "durov"), ("durov", "durov"), ("https://t.me/durov", "durov"),
    ("t.me/s/durov", "durov"), ("https://t.me/durov/123", "durov"), ("  ", None),
    ("https://t.me/joinchat/abc", None), ("bad name!", None), (None, None),
])
def test_normalize_channel(raw, expected):
    assert tg.normalize_channel(raw) == expected


def test_seed_channels_merges_config_and_options():
    cfg = {"TELEGRAM_SEED_CHANNELS": "@durov, telegram,durov"}
    seeds = tg.seed_channels(cfg, q(utc(2026, 1, 1), utc(2026, 1, 2),
                                    telegram_channels=["https://t.me/BBCNews", "x!"]))
    assert seeds == ["durov", "telegram", "bbcnews"]


# =================================================================== web preview


def test_parse_live_search_page_basic_fields():
    page = tg.parse_preview(DUROV, channel="durov", collected_at=COLLECTED, topic="telegram",
                            raw_ref="raw/telegram/p1.html")
    assert len(page.posts) == 20
    assert page.oldest_id == 532
    assert page.channel_found and page.channel_title == "Pavel Durov"
    assert page.approximate_counts  # "7.24M" style views
    assert page.unverified == 0
    p = by_id(page.posts, "telegram:durov:533")
    assert p.published_at == "2026-07-21T14:36:55Z"  # from <time datetime>, platform
    assert p.timestamp_verified and p.collected_at == COLLECTED
    assert p.author_handle == "@durov" and p.author_name == "Pavel Durov"
    assert p.author_avatar_url and p.author_avatar_url.startswith("https://")
    assert p.url == "https://t.me/durov/533"
    assert p.metrics.views == 7240000
    assert p.metrics.reactions and p.metrics.reactions > 0
    assert p.raw_ref == "raw/telegram/p1.html" and p.query == "telegram"


def test_parse_live_video_duration_time_is_not_the_timestamp():
    # post 532 carries a video whose duration is also a <time> element (no datetime);
    # the parser must take the message date, not the duration.
    p = by_id(tg.parse_preview(DUROV, channel="durov", collected_at=COLLECTED).posts,
              "telegram:durov:532")
    assert p.published_at == "2026-07-19T21:45:01Z"
    assert [m.type for m in p.media] == ["video"]
    assert p.media[0].url and p.media[0].thumbnail


def test_parse_live_forward_relation():
    p = by_id(tg.parse_preview(DUROV, channel="durov", collected_at=COLLECTED).posts,
              "telegram:durov:552")
    fwd = [r for r in p.relations if r.type == "forward"]
    assert len(fwd) == 1
    assert fwd[0].target_post_id == "telegram:designers:266"
    assert fwd[0].target_handle == "@designers"
    assert fwd[0].target_url == "https://t.me/designers/266"
    assert fwd[0].evidence == "direct"


def test_parse_live_reply_relation_and_text_excludes_quoted_parent():
    p = by_id(tg.parse_preview(DUROV, channel="durov", collected_at=COLLECTED).posts,
              "telegram:durov:558")
    rep = [r for r in p.relations if r.type == "reply"]
    assert rep and rep[0].target_post_id == "telegram:durov:557"
    assert p.text.startswith("CORRECTION")
    assert "Unfair that CEOs" not in p.text  # the quoted parent's text
    assert [m.type for m in p.media] == ["photo"]


def test_parse_live_links_and_mentions():
    posts = tg.parse_preview(DUROV, channel="durov", collected_at=COLLECTED).posts
    p = by_id(posts, "telegram:durov:542")
    links = [r for r in p.relations if r.type == "link"]
    assert any(r.target_url == "https://stats.ioinformatics.org/results/2026" for r in links)
    assert any(r.target_post_id == "telegram:durov:540" for r in links)  # t.me post link
    p = by_id(posts, "telegram:durov:543")
    assert any(r.type == "mention" and r.target_handle == "@monk" for r in p.relations)
    # relative hashtag links ("?q=%23tag") never become relations
    assert all((r.target_url or "").startswith("http") or r.target_post_id
               for x in posts for r in x.relations if r.type in ("link", "mention"))


def test_parse_live_channel_page_preserves_newlines():
    page = tg.parse_preview(TG_CHANNEL, channel="telegram", collected_at=COLLECTED)
    assert len(page.posts) == 20 and page.oldest_id == 441
    p = by_id(page.posts, "telegram:telegram:445")
    # <br> becomes "\n"; inline <b>/<a> fragments are not split onto new lines
    assert p.text.startswith("For all the features from this update, like statistics")
    assert "full blog:\nhttps://telegram.org/blog/" in p.text


def test_parse_live_zero_result_search_page():
    page = tg.parse_preview(ZERO, channel="telegram", collected_at=COLLECTED)
    assert page.posts == [] and page.channel_found and page.oldest_id is None


SYNTHETIC = """
<div class="tgme_channel_info_header_title"><span>Test</span></div>
<div class="tgme_widget_message" data-post="testchan/7">
  <div class="tgme_widget_message_forwarded_from">Forwarded from <span>Hidden</span></div>
  <div class="tgme_widget_message_text js-message_text">hello<br/>world</div>
  <span class="tgme_widget_message_views">15</span>
</div>
"""


def test_parse_synthetic_missing_datetime_and_hidden_forward():
    # Hand-written edge case (not a live response): no <time datetime>, hidden forward.
    page = tg.parse_preview(SYNTHETIC, channel="testchan", collected_at=COLLECTED)
    (p,) = page.posts
    assert p.published_at is None and not p.timestamp_verified and p.timestamp_note
    assert page.unverified == 1
    assert p.text == "hello\nworld"
    assert p.metrics.views == 15 and not page.approximate_counts
    assert [r.type for r in p.relations] == ["forward"]
    assert p.relations[0].target_post_id is None


# ----------------------------------------------------------------- web collector


@respx.mock
def test_web_mode_bounds_to_window_and_finishes(tmp_path):
    respx.get("https://t.me/s/durov").mock(return_value=httpx.Response(200, text=DUROV))
    col = tg.TelegramCollector(sleep=lambda s: None)
    ctx = make_ctx(tmp_path, {"TELEGRAM_SEED_CHANNELS": "durov"})
    res = col.collect(q(utc(2026, 8, 1), utc(2026, 8, 31, 23, 59, 59)), ctx)
    (st,) = res.status
    assert st.state == SourceState.ok, st.reason
    ids = {p.post_id.rsplit(":", 1)[1] for p in res.posts}
    assert ids == {"538", "539", "540", "541", "542", "543", "544", "545", "546"}
    assert st.count == 9 and st.requests == 1
    assert st.date_filtered is False  # no server-side date filter in web preview
    assert "web-preview mode" in st.reason and "outside the window" in st.reason
    assert "rounded" in st.reason
    assert all(p.raw_ref and (tmp_path / p.raw_ref).is_file() for p in res.posts)
    req = respx.calls[0].request
    assert req.url.params["q"] == "telegram"


@respx.mock
def test_web_mode_paginates_with_before(tmp_path):
    route = respx.get("https://t.me/s/durov")
    route.side_effect = [httpx.Response(200, text=DUROV), httpx.Response(200, text=ZERO)]
    col = tg.TelegramCollector(sleep=lambda s: None)
    ctx = make_ctx(tmp_path, {"TELEGRAM_SEED_CHANNELS": "durov"})
    res = col.collect(q(utc(2026, 7, 1), utc(2026, 7, 31, 23, 59, 59)), ctx)
    (st,) = res.status
    assert sorted(p.post_id for p in res.posts) == [
        "telegram:durov:532", "telegram:durov:533", "telegram:durov:535"]
    assert st.state == SourceState.ok and st.requests == 2
    assert respx.calls[1].request.url.params["before"] == "532"


@respx.mock
def test_web_mode_zero_results_is_completed_zero(tmp_path):
    respx.get("https://t.me/s/telegram").mock(return_value=httpx.Response(200, text=ZERO))
    col = tg.TelegramCollector(sleep=lambda s: None)
    ctx = make_ctx(tmp_path, {"TELEGRAM_SEED_CHANNELS": "telegram"})
    res = col.collect(q(utc(2026, 7, 1), utc(2026, 7, 31)), ctx)
    assert res.posts == [] and res.status[0].state == SourceState.completed_zero


@respx.mock
def test_web_mode_missing_channel_is_out_of_coverage(tmp_path):
    respx.get("https://t.me/s/nochan").mock(return_value=httpx.Response(
        302, headers={"location": "https://t.me/nochan"}))
    respx.get("https://t.me/nochan").mock(return_value=httpx.Response(200, text="<html/>"))
    col = tg.TelegramCollector(sleep=lambda s: None)
    ctx = make_ctx(tmp_path, {"TELEGRAM_SEED_CHANNELS": "nochan"})
    res = col.collect(q(utc(2026, 7, 1), utc(2026, 7, 31)), ctx)
    st = res.status[0]
    assert st.state == SourceState.out_of_coverage and "no public web preview" in st.reason


@respx.mock
def test_web_mode_http_error_is_failed_not_empty(tmp_path):
    respx.get("https://t.me/s/durov").mock(return_value=httpx.Response(403, text="no"))
    col = tg.TelegramCollector(sleep=lambda s: None)
    ctx = make_ctx(tmp_path, {"TELEGRAM_SEED_CHANNELS": "durov"})
    res = col.collect(q(utc(2026, 7, 1), utc(2026, 7, 31)), ctx)
    assert res.status[0].state == SourceState.failed and "403" in res.status[0].reason


def test_web_mode_out_of_time_is_partial(tmp_path):
    col = tg.TelegramCollector(sleep=lambda s: None)
    ctx = make_ctx(tmp_path, {"TELEGRAM_SEED_CHANNELS": "durov,telegram"}, budget=-1)
    res = col.collect(q(utc(2026, 7, 1), utc(2026, 7, 31)), ctx)
    assert [s.state for s in res.status] == [SourceState.partial, SourceState.partial]
    # Cursor prefixed with the first date of the slice where collection stopped (newest).
    assert res.status[0].cursor == "2026-07-28:durov:start"


# ==================================================================== mode selection


def _write_session(cfg: dict[str, str]) -> None:
    p = tg.session_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_mode_none_is_skipped_no_credentials(tmp_path):
    res = tg.TelegramCollector().collect(q(utc(2026, 7, 1), utc(2026, 7, 31)),
                                         make_ctx(tmp_path, {}))
    st = res.status[0]
    assert st.state == SourceState.skipped_no_credentials
    assert "TELEGRAM_SEED_CHANNELS" in st.reason and "Premium" in st.reason


def test_select_mode_variants():
    keys = {"TELEGRAM_API_ID": "1", "TELEGRAM_API_HASH": "abcdef123456"}
    assert tg.select_mode({}, ["durov"])[0] == "web"
    mode, why = tg.select_mode(keys, ["durov"])
    assert mode == "web" and "log in" in why  # keys but no session file
    assert tg.select_mode(keys, [])[0] == "none"
    _write_session(keys)
    assert tg.select_mode(keys, [])[0] == "api"
    assert tg.session_path({"TELEGRAM_SESSION": "../evil"}).name == "evil.session"


def test_capabilities():
    caps = tg.TelegramCollector().capabilities()
    assert caps.needs_credentials is False and caps.credential_keys == ()


# ================================================================ Telethon mapping


class MessageEntityUrl(SimpleNamespace):
    pass


class MessageEntityTextUrl(SimpleNamespace):
    pass


def channel(cid=111, username="SrcChan", title="Source Channel", left=True,
            date=utc(2020, 5, 1)):
    return SimpleNamespace(id=cid, username=username, title=title, broadcast=True,
                           megagroup=False, left=left, date=date)


def msg(mid, date, text="", **kw):
    base = dict(id=mid, date=date, message=text, views=None, forwards=None, replies=None,
                reactions=None, media=None, fwd_from=None, reply_to=None, entities=None,
                from_id=None, post=True, peer_id=SimpleNamespace(channel_id=111))
    base.update(kw)
    return SimpleNamespace(**base)


def test_message_to_post_full_mapping():
    chat = channel()
    origin = channel(cid=222, username="OrigChan", title="Origin")
    text = "🚀 see example.com/a and more"  # emoji = 2 UTF-16 units before the URL
    m = msg(
        10, utc(2026, 7, 3, 12, 0), text,
        views=1500, forwards=12,
        replies=SimpleNamespace(replies=4),
        reactions=SimpleNamespace(results=[SimpleNamespace(count=3),
                                           SimpleNamespace(count=5)]),
        media=SimpleNamespace(document=SimpleNamespace(mime_type="video/mp4")),
        fwd_from=SimpleNamespace(from_id=SimpleNamespace(channel_id=222), channel_post=77,
                                 from_name=None),
        reply_to=SimpleNamespace(reply_to_msg_id=9, reply_to_peer_id=None),
        entities=[MessageEntityUrl(offset=7, length=13),
                  MessageEntityTextUrl(offset=0, length=2, url="https://t.me/other/5")],
    )
    p = tg.message_to_post(m, chat=chat, collected_at=COLLECTED, topic="t",
                           lookup=tg.entity_lookup(origin))
    assert p.post_id == "telegram:srcchan:10"
    assert p.url == "https://t.me/srcchan/10"
    assert p.author_id == "111" and p.author_handle == "@SrcChan"
    assert p.author_name == "Source Channel"
    assert p.author_created_at == "2020-05-01T00:00:00Z"  # left=True -> creation date
    assert p.published_at == "2026-07-03T12:00:00Z" and p.timestamp_verified
    assert p.text == text
    assert (p.metrics.views, p.metrics.forwards, p.metrics.replies, p.metrics.reactions) == (
        1500, 12, 4, 8)
    assert [x.type for x in p.media] == ["video"] and p.media[0].url is None
    rels = {(r.type, r.target_post_id, r.target_url) for r in p.relations}
    assert ("forward", "telegram:origchan:77", "https://t.me/origchan/77") in rels
    assert ("reply", "telegram:srcchan:9", "https://t.me/srcchan/9") in rels
    assert ("link", None, "https://example.com/a") in rels
    assert ("link", "telegram:other:5", "https://t.me/other/5") in rels
    assert all(r.evidence == "direct" for r in p.relations)


def test_message_to_post_unresolved_forward_and_member_join_date():
    chat = channel(left=False)  # member: Channel.date is the JOIN date, not creation
    m = msg(5, utc(2026, 7, 3), "x", media=SimpleNamespace(photo=object()),
            fwd_from=SimpleNamespace(from_id=SimpleNamespace(channel_id=999),
                                     channel_post=3, from_name=None))
    p = tg.message_to_post(m, chat=chat, collected_at=COLLECTED)
    assert p.author_created_at is None
    (f,) = p.relations
    assert f.type == "forward" and f.target_post_id == "telegram:999:3"
    assert f.target_url is None and f.target_handle is None
    assert [x.type for x in p.media] == ["photo"]


def test_message_to_post_group_sender_and_naive_date():
    group = SimpleNamespace(id=50, username=None, title="Grp", megagroup=True, left=True,
                            date=None)
    user = SimpleNamespace(id=7, username="alice", first_name="Alice", last_name=None)
    m = msg(1, datetime(2026, 7, 3), "hi", post=False,
            from_id=SimpleNamespace(user_id=7))
    p = tg.message_to_post(m, chat=group, collected_at=COLLECTED,
                           lookup=tg.entity_lookup(user))
    assert p.post_id == "telegram:50:1" and p.url == "https://t.me/c/50/1"  # members-only link
    assert p.author_id == "7" and p.author_handle == "@alice" and p.author_name == "Alice"
    assert p.published_at is None and not p.timestamp_verified  # naive -> unverifiable


# =================================================================== API collector


class FakeClient:
    """Minimal async stand-in for TelegramClient (sequential, no network)."""

    def __init__(self, entities: dict, messages: dict, *, authorized=True, flood=None,
                 call_handler=None):
        self.entities, self.messages = entities, messages
        self.authorized = authorized
        self.flood = dict(flood or {})  # channel -> (after_n_messages, seconds, times)
        self.call_handler = call_handler
        self.iter_calls: list[dict] = []
        self.disconnected = False

    async def connect(self):
        return None

    async def is_user_authorized(self):
        return self.authorized

    async def disconnect(self):
        self.disconnected = True

    async def get_entity(self, name):
        if name not in self.entities:
            raise ValueError(f'No user has "{name}" as username')
        return self.entities[name]

    async def __call__(self, req):
        return await self.call_handler(req)

    def iter_messages(self, entity, search=None, offset_date=None, offset_id=0, limit=None):
        self.iter_calls.append(dict(entity=entity.username, search=search,
                                    offset_date=offset_date, offset_id=offset_id,
                                    limit=limit))
        msgs = self.messages[entity.username.lower()]
        flood = self.flood.get(entity.username.lower())

        async def gen():
            n = 0
            for m in msgs:  # newest first, like Telegram
                if offset_date and m.date >= offset_date:
                    continue
                if offset_id and m.id >= offset_id:
                    continue
                if limit is not None and n >= limit:
                    return  # Telethon stops after `limit` messages
                if flood and flood[2] > 0 and n == flood[0]:
                    self.flood[entity.username.lower()] = (flood[0], flood[1], flood[2] - 1)
                    raise FloodWaitError(request=None, capture=flood[1])
                n += 1
                yield m
        return gen()


API_CFG = {"TELEGRAM_API_ID": "12345", "TELEGRAM_API_HASH": "0123456789abcdef"}


def _api_msgs():
    """Six channel messages, newest first: ids 6..2 in July, id 0 on 20 June."""
    return [msg(i, utc(2026, 7, d), f"post {i}") for i, d in
            ((6, 30), (5, 20), (4, 10), (3, 2), (2, 1))] + [msg(0, utc(2026, 6, 20), "old")]


def test_api_mode_date_bounded_channel_search(tmp_path):
    _write_session(API_CFG)
    chat = channel(username="news")
    fake = FakeClient({"news": chat}, {"news": _api_msgs()})
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_SEED_CHANNELS": "news"})
    res = col.collect(q(utc(2026, 7, 2), utc(2026, 7, 25, 23, 59, 59), topic="t"), ctx)
    (st,) = res.status
    assert st.state == SourceState.ok and st.date_filtered is True
    assert "API mode" in st.reason and "reached start date" in st.reason
    assert [p.post_id for p in res.posts] == ["telegram:news:5", "telegram:news:4",
                                              "telegram:news:3"]
    assert fake.iter_calls[0]["search"] == "t"
    assert fake.iter_calls[0]["offset_date"] == utc(2026, 7, 26, 0, 0, 0)
    assert fake.disconnected
    assert all(p.raw_ref and (tmp_path / p.raw_ref).is_file() for p in res.posts)


def test_api_mode_long_flood_wait_is_partial_and_stops_account(tmp_path):
    _write_session(API_CFG)
    a, b = channel(username="news"), channel(cid=2, username="other")
    fake = FakeClient({"news": a, "other": b},
                      {"news": _api_msgs(), "other": _api_msgs()},
                      flood={"news": (2, 120, 1)})
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_SEED_CHANNELS": "news,other"})
    res = col.collect(q(utc(2026, 6, 1), utc(2026, 7, 31, 23, 59, 59), topic="t"), ctx)
    s1, s2 = res.status
    # Updated for per-day slices (2026-11-19): the window Jun 1..Jul 31 is 10 slices of
    # 6-7 days; the fake floods when a single search has yielded 2 messages, which first
    # happens in the Jul 1-6 slice (ids 3, 2 then the flood before id 0).
    assert s1.state == SourceState.partial and "flood wait 120s" in s1.reason
    assert s1.cursor == "2026-07-01:news:offset_id=2" and s1.count == 5
    assert "2026-06-01..2026-06-06 not reached" in s1.reason
    assert s2.state == SourceState.partial and "account in flood wait 120s" in s2.reason
    assert {c["entity"] for c in fake.iter_calls} == {"news"}  # never touched "other"


def test_api_mode_short_flood_wait_sleeps_once_and_resumes(tmp_path):
    _write_session(API_CFG)
    fake = FakeClient({"news": channel(username="news")}, {"news": _api_msgs()},
                      flood={"news": (2, 1, 1)})
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_SEED_CHANNELS": "news"})
    res = col.collect(q(utc(2026, 6, 1), utc(2026, 7, 31, 23, 59, 59), topic="t"), ctx)
    (st,) = res.status
    assert st.state == SourceState.ok, st.reason
    assert len(res.posts) == 6
    # Flood in the Jul 1-6 slice after ids 3, 2: the retry resumes after the last message
    # seen, within the same slice (same offset_date).
    flood_i = next(i for i, c in enumerate(fake.iter_calls) if c["offset_id"] == 2)
    assert fake.iter_calls[flood_i]["offset_date"] == fake.iter_calls[flood_i - 1]["offset_date"]
    assert "history exhausted" in st.reason


def test_api_mode_unresolvable_channel_is_failed(tmp_path):
    _write_session(API_CFG)
    fake = FakeClient({}, {})
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_SEED_CHANNELS": "ghost"})
    res = col.collect(q(utc(2026, 6, 1), utc(2026, 7, 31)), ctx)
    assert res.status[0].state == SourceState.failed
    assert "cannot resolve" in res.status[0].reason


@respx.mock
def test_api_unauthorized_session_falls_back_to_web(tmp_path):
    _write_session(API_CFG)
    respx.get("https://t.me/s/durov").mock(return_value=httpx.Response(200, text=DUROV))
    fake = FakeClient({}, {}, authorized=False)
    col = tg.TelegramCollector(client_factory=lambda cfg: fake, sleep=lambda s: None)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_SEED_CHANNELS": "durov"})
    res = col.collect(q(utc(2026, 8, 1), utc(2026, 8, 31, 23, 59, 59)), ctx)
    st = res.status[0]
    assert st.state == SourceState.ok and "API mode unavailable" in st.reason
    assert "not authorized" in st.reason and len(res.posts) == 9


def test_api_no_seeds_no_premium_is_skipped_capability(tmp_path):
    _write_session(API_CFG)
    fake = FakeClient({}, {})
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    res = col.collect(q(utc(2026, 6, 1), utc(2026, 7, 31)), make_ctx(tmp_path, API_CFG))
    st = res.status[0]
    assert st.state == SourceState.skipped_capability and "Premium" in st.reason


def _global_handler(fl_remains=5, pages=None):
    pages = list(pages or [])

    async def handler(req):
        name = type(req).__name__
        if name == "CheckSearchPostsFloodRequest":
            return SimpleNamespace(total_daily=10, remains=fl_remains, stars_amount=10,
                                   query_is_free=False, wait_till=None)
        if name == "SearchPostsRequest":
            assert req.query == "t" and req.allow_paid_stars is None  # never pay
            return pages.pop(0)
        raise AssertionError(name)
    return handler


def test_api_global_search_premium(tmp_path):
    _write_session(API_CFG)
    a = channel(cid=111, username="alpha")
    page = SimpleNamespace(
        messages=[msg(3, utc(2026, 7, 5), "in window"),
                  msg(2, utc(2026, 5, 1), "too old")],
        chats=[a], users=[], next_rate=None)
    fake = FakeClient({}, {}, call_handler=_global_handler(pages=[page]))
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_HAS_PREMIUM": "true"})
    res = col.collect(q(utc(2026, 7, 1), utc(2026, 7, 31, 23, 59, 59), topic="t"), ctx)
    (st,) = res.status
    assert st.state == SourceState.ok, st.reason
    assert [p.post_id for p in res.posts] == ["telegram:alpha:3"]
    assert st.date_filtered is False and "1 result(s) outside the window" in st.reason


def test_api_global_search_quota_exhausted_never_pays(tmp_path):
    _write_session(API_CFG)
    fake = FakeClient({}, {}, call_handler=_global_handler(fl_remains=0))
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_HAS_PREMIUM": "true"})
    res = col.collect(q(utc(2026, 7, 1), utc(2026, 7, 31), topic="t"), ctx)
    st = res.status[0]
    assert st.state == SourceState.skipped_capability and "never used" in st.reason


# =========================================================================== auth


class FakeAuthClient:
    def __init__(self, *, authorized=False, needs_password=False, premium=True):
        self.authorized = authorized
        self.needs_password = needs_password
        self.premium = premium
        self.connected = False
        self.sign_in_calls: list[dict] = []
        self.logged_out = False

    def is_connected(self):
        return self.connected

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def is_user_authorized(self):
        return self.authorized

    async def send_code_request(self, phone):
        return SimpleNamespace(phone_code_hash="HASH")

    async def sign_in(self, phone=None, code=None, *, password=None, phone_code_hash=None):
        self.sign_in_calls.append(dict(phone=phone, code=code, password=password,
                                       phone_code_hash=phone_code_hash))
        if password is None and self.needs_password:
            raise SessionPasswordNeededError(request=None)
        self.authorized = True

    async def get_me(self):
        return SimpleNamespace(id=1, username="analyst", first_name="A", premium=self.premium)

    async def log_out(self):
        self.logged_out = True
        self.authorized = False


def test_auth_status_not_configured_and_no_session():
    assert auth.status({})["configured"] is False
    st = auth.status(API_CFG)
    assert st["configured"] and not st["authorized"] and "no session file" in st["reason"]
    assert st["session_path"].endswith("sessions/socmint.session")


def test_auth_login_flow_with_2fa(monkeypatch):
    fake = FakeAuthClient(needs_password=True)
    monkeypatch.setattr(auth, "_make_client", lambda cfg: fake)
    assert auth.send_code(API_CFG, "+44 7000 000000") == {"ok": True, "error": None}
    r = auth.sign_in(API_CFG, "12345")
    assert r["ok"] is False and r["needs_password"] is True
    r = auth.sign_in(API_CFG, "", password="pw")
    assert r["ok"] is True and r["premium"] is True and r["user"] == "@analyst"
    assert fake.sign_in_calls[0]["phone_code_hash"] == "HASH"
    assert fake.sign_in_calls[0]["phone"] == "+447000000000"
    assert not fake.connected  # released so the collector can open the session
    # a second sign_in without a pending login is an explicit error, not a crash
    assert auth.sign_in(API_CFG, "1")["error"].startswith("no login in progress")


def test_auth_status_authorized_and_logout(monkeypatch):
    _write_session(API_CFG)
    fake = FakeAuthClient(authorized=True, premium=False)
    monkeypatch.setattr(auth, "_make_client", lambda cfg: fake)
    st = auth.status(API_CFG)
    assert st["authorized"] and st["user"] == "@analyst" and st["premium"] is False
    out = auth.logout(API_CFG)
    assert out["ok"] and fake.logged_out and not tg.session_path(API_CFG).exists()


def test_auth_errors_are_reported_not_raised(monkeypatch):
    class Boom(FakeAuthClient):
        async def send_code_request(self, phone):
            raise RuntimeError("api_hash=0123456789abcdef bad")

    monkeypatch.setattr(auth, "_make_client", lambda cfg: Boom())
    r = auth.send_code(API_CFG, "+1555")
    assert r["ok"] is False and "0123456789abcdef" not in r["error"]
    assert auth.send_code({}, "+1")["ok"] is False


# ============================================================ review regressions


def test_parse_live_paid_reaction_not_counted():
    # Live durov/564: paid reaction "2.08M" (Stars spent) + 28.6K + 9.83K + 7.98K.
    p = by_id(tg.parse_preview(DUROV, channel="durov", collected_at=COLLECTED).posts,
              "telegram:durov:564")
    assert p.metrics.views == 899000
    assert p.metrics.reactions == 28600 + 9830 + 7980
    assert p.metrics.reactions < p.metrics.views


class ReactionPaid(SimpleNamespace):
    pass


class ReactionEmoji(SimpleNamespace):
    pass


def test_message_to_post_skips_paid_reaction_counts():
    m = msg(1, utc(2026, 7, 3), "x", reactions=SimpleNamespace(results=[
        SimpleNamespace(reaction=ReactionPaid(), count=2_000_000),
        SimpleNamespace(reaction=ReactionEmoji(emoticon="👍"), count=7)]))
    p = tg.message_to_post(m, chat=channel(), collected_at=COLLECTED)
    assert p.metrics.reactions == 7


def _collectible_channel(cid=333):
    # Username=None, public name only in `usernames` (e.g. a Fragment collectible).
    ch = channel(cid=cid, username=None, title="Collectible")
    ch.usernames = [SimpleNamespace(username="OldName", active=False),
                    SimpleNamespace(username="Fancy", active=True)]
    return ch


def test_public_handle_falls_back_to_active_usernames():
    assert tg.public_handle(_collectible_channel()) == "Fancy"
    assert tg.public_handle(channel(username="Main")) == "Main"
    bare = channel(username=None)
    assert tg.public_handle(bare) is None


def test_message_to_post_collectible_username_gets_url_and_key():
    chat = _collectible_channel()
    src = _collectible_channel(cid=444)
    m = msg(9, utc(2026, 7, 3), "x",
            fwd_from=SimpleNamespace(from_id=SimpleNamespace(channel_id=444),
                                     channel_post=2, from_name=None))
    p = tg.message_to_post(m, chat=chat, collected_at=COLLECTED,
                           lookup=tg.entity_lookup(src))
    assert p.post_id == "telegram:fancy:9" and p.url == "https://t.me/fancy/9"
    assert p.author_handle == "@Fancy"
    (f,) = p.relations
    assert f.target_post_id == "telegram:fancy:2" and f.target_url == "https://t.me/fancy/2"


def test_message_to_post_seed_handle_fallback():
    chat = channel(username=None)  # no username, no usernames list
    p = tg.message_to_post(msg(4, utc(2026, 7, 3)), chat=chat, collected_at=COLLECTED,
                           handle="seedchan")
    assert p.post_id == "telegram:seedchan:4" and p.url == "https://t.me/seedchan/4"
    assert p.author_handle == "@seedchan"


def test_api_channel_with_collectible_username_uses_seed(tmp_path):
    _write_session(API_CFG)
    chat = channel(username=None)

    class Fake(FakeClient):
        def iter_messages(self, entity, **kw):
            async def gen():
                for m in _api_msgs():
                    if m.date < kw["offset_date"]:
                        yield m
            return gen()

    fake = Fake({"news": chat}, {})
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_SEED_CHANNELS": "news"})
    res = col.collect(q(utc(2026, 7, 2), utc(2026, 7, 25, 23, 59, 59), topic="t"), ctx)
    assert [p.url for p in res.posts] == [f"https://t.me/news/{i}" for i in (5, 4, 3)]


def test_api_global_search_limit_with_more_results_is_partial(tmp_path, monkeypatch):
    _write_session(API_CFG)
    monkeypatch.setattr(tg, "GLOBAL_MAX_PAGES", 1)
    a = channel(cid=111, username="alpha")
    # Updated 2026-11-19 (cap decided after the walk): the page cap (1) stops the walk at
    # Jul 20 with more results available -> partial, older slices named "not reached".
    # The cap (2) is split over the slices that have candidates, newest first.
    # One message's chat (999) is absent from res.chats; next_rate says more exist.
    page1 = SimpleNamespace(
        messages=[msg(10, utc(2026, 7, 31), peer_id=SimpleNamespace(channel_id=999)),
                  msg(9, utc(2026, 7, 31)), msg(8, utc(2026, 7, 30)),
                  msg(7, utc(2026, 7, 27)), msg(6, utc(2026, 7, 20))],
        chats=[a], users=[], next_rate=42)
    fake = FakeClient({}, {}, call_handler=_global_handler(pages=[page1]))
    monkeypatch.setattr("telethon.utils.get_input_peer", lambda e: ("peer", e.id))
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_HAS_PREMIUM": "true"})
    query = q(utc(2026, 7, 1), utc(2026, 7, 31, 23, 59, 59), topic="t")
    query.limit = 2
    res = col.collect(query, ctx)
    (st,) = res.status
    assert st.state == SourceState.partial, st.reason
    assert "page cap (1) reached before the window start" in st.reason
    # Resumes after the last result of the page, from the oldest slice reached (Jul 19-21).
    assert st.cursor == "2026-07-19:global:offset_rate=42:offset_id=6"
    assert "1 result(s) skipped" in st.reason  # unresolved peer counted, not silent
    assert [p.post_id for p in res.posts] == ["telegram:alpha:9", "telegram:alpha:7"]
    assert "2026-07-28..2026-07-31 1, 2026-07-25..2026-07-27 1" in st.reason
    assert "4 in-window candidate(s)" in st.reason
    assert "2026-07-16..2026-07-18 not reached" in st.reason
    assert "2026-07-01..2026-07-03 not reached" in st.reason


@respx.mock
def test_web_mode_cap_split_after_walk(tmp_path):
    respx.get("https://t.me/s/durov").mock(return_value=httpx.Response(200, text=DUROV))
    col = tg.TelegramCollector(sleep=lambda s: None)
    ctx = make_ctx(tmp_path, {"TELEGRAM_SEED_CHANNELS": "durov"})
    query = q(utc(2026, 8, 1), utc(2026, 8, 31, 23, 59, 59))
    query.limit = 3
    res = col.collect(query, ctx)
    (st,) = res.status
    # Updated 2026-11-19 (cap decided after the walk): the page runs past Aug 1, so every
    # slice was covered -> ok. 9 candidates in 7 slices; the cap of 3 goes one each to
    # the newest slices with candidates (546 newest of Aug 28-31; 544; 543).
    assert st.state == SourceState.ok, st.reason
    kept = sorted(int(p.post_id.rsplit(":", 1)[1]) for p in res.posts)
    assert kept == [543, 544, 546]
    assert st.cursor is None and "9 in-window candidate(s)" in st.reason
    assert "2026-08-01..2026-08-03 0" in st.reason and "not reached" not in st.reason


@respx.mock
def test_web_mode_stops_early_when_every_day_has_its_share(tmp_path):
    respx.get("https://t.me/s/durov").mock(return_value=httpx.Response(200, text=DUROV))
    col = tg.TelegramCollector(sleep=lambda s: None)
    ctx = make_ctx(tmp_path, {"TELEGRAM_SEED_CHANNELS": "durov"})
    query = q(utc(2026, 8, 30), utc(2026, 8, 31, 23, 59, 59))
    query.limit = 2  # 546 (Aug 31) and 545 (Aug 30) fill both days' share of 1
    res = col.collect(query, ctx)
    (st,) = res.status
    assert st.state == SourceState.ok and "stopped early" in st.reason
    assert sorted(p.post_id for p in res.posts) == ["telegram:durov:545", "telegram:durov:546"]
    assert "per-day: 2026-08-31 1, 2026-08-30 1" in st.reason


@respx.mock
def test_malformed_api_id_falls_back_to_web(tmp_path):
    cfg = {"TELEGRAM_API_ID": "abc ", "TELEGRAM_API_HASH": "0123456789abcdef",
           "TELEGRAM_SEED_CHANNELS": "durov"}
    _write_session(cfg)
    respx.get("https://t.me/s/durov").mock(return_value=httpx.Response(200, text=DUROV))
    col = tg.TelegramCollector(sleep=lambda s: None)  # real factory: int("abc ") fails
    res = col.collect(q(utc(2026, 8, 1), utc(2026, 8, 31, 23, 59, 59)), make_ctx(tmp_path, cfg))
    st = res.status[0]
    assert st.state == SourceState.ok and "API mode unavailable" in st.reason
    assert "client setup failed" in st.reason and len(res.posts) == 9


# ================================================ per-day slices + phrase matching
# Added 2026-11-19: the cap is spread across day slices (socmint.collect.window) and,
# in phrase mode, posts without the exact phrase are not kept. All fakes, no network.

NOV = dict(start=utc(2026, 11, 17), end=utc(2026, 11, 19, 23, 59, 59))


def _day_msgs(spec):
    """spec: [(day, count, text)] newest first -> newest-first messages, ids descending."""
    out, mid = [], 1000
    for day, count, text in spec:
        for h in range(count):
            out.append(msg(mid, utc(2026, 11, day, 20 - h), text))
            mid -= 1
    return out


def _api_run(tmp_path, messages, *, limit, topic="t", seeds="news", start=None, end=None,
             ctx_hook=None, **opts):
    _write_session(API_CFG)
    entities = {s: channel(cid=i + 1, username=s) for i, s in enumerate(seeds.split(","))}
    fake = FakeClient(entities, messages)
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_SEED_CHANNELS": seeds})
    if ctx_hook:
        ctx_hook(ctx, fake)
    query = Query(topic=topic, start=start or NOV["start"], end=end or NOV["end"],
                  limit=limit, options=opts)
    return col.collect(query, ctx), fake


def test_api_channel_one_search_per_day_with_shared_cap_and_rollover(tmp_path):
    # Nov 19: 5 posts, Nov 18: quiet, Nov 17: 5 posts. Cap 6 -> 2 for Nov 19, the quiet
    # Nov 18 rolls its share over, Nov 17 gets the remaining 4.
    msgs = _day_msgs([(19, 5, "t a"), (17, 5, "t b")])
    res, fake = _api_run(tmp_path, {"news": msgs}, limit=6)
    (st,) = res.status
    assert st.state == SourceState.ok, st.reason
    days = [p.published_at[:10] for p in res.posts]
    assert days.count("2026-11-19") == 2 and days.count("2026-11-17") == 4
    assert len(res.posts) == 6
    # Per-slice requests carry the slice bounds: offset_date = slice end + 1 s.
    assert [c["offset_date"] for c in fake.iter_calls] == [
        utc(2026, 11, 20), utc(2026, 11, 19), utc(2026, 11, 18)]
    assert [c["limit"] for c in fake.iter_calls] == [2, 2, 4]  # words: ask for allowance
    assert ("per-day: 2026-11-19 2, 2026-11-18 0 (fewer than its share of 2), "
            "2026-11-17 4") in st.reason


def test_api_channel_quiet_slices_skipped_without_request(tmp_path):
    # Only Nov 19 and Nov 13 have posts: the Nov 19 search runs on into a Nov 13
    # message, so Nov 18..14 are known empty and cost no request.
    msgs = _day_msgs([(19, 1, "t"), (13, 1, "t")])
    res, fake = _api_run(tmp_path, {"news": msgs}, limit=70,
                         start=utc(2026, 11, 13), end=utc(2026, 11, 19, 23, 59, 59))
    (st,) = res.status
    assert st.state == SourceState.ok and len(res.posts) == 2
    assert [c["offset_date"].day for c in fake.iter_calls] == [20, 14]
    assert "5 slice(s) known empty" in st.reason
    assert "2026-11-15 0" in st.reason and "not reached" not in st.reason


def test_api_channel_time_budget_mid_way_is_partial_naming_unreached_days(tmp_path):
    msgs = _day_msgs([(19, 5, "t"), (18, 5, "t"), (17, 5, "t")])

    def hook(ctx, fake):
        ctx.out_of_time = lambda: len(fake.iter_calls) >= 1  # runs out during slice 1

    res, fake = _api_run(tmp_path, {"news": msgs}, limit=9, ctx_hook=hook)
    (st,) = res.status
    assert st.state == SourceState.partial and "time budget exhausted" in st.reason
    assert len(fake.iter_calls) == 1 and len(res.posts) == 1
    assert "2026-11-18 not reached" in st.reason and "2026-11-17 not reached" in st.reason
    assert st.cursor == "2026-11-19:news:offset_id=1000"


def test_api_channel_phrase_mode_sends_plain_topic_and_drops_non_phrase(tmp_path):
    msgs = [msg(5, utc(2026, 11, 19, 10), "Bank-of-America cuts rates"),
            msg(4, utc(2026, 11, 19, 9), "a bank in north america"),
            msg(3, utc(2026, 11, 18, 9), "BANK OF AMERICA statement"),
            msg(2, utc(2026, 11, 17, 9), "america's bank, of course")]
    res, fake = _api_run(tmp_path, {"news": msgs}, limit=30, topic="bank  of america")
    (st,) = res.status
    assert fake.iter_calls[0]["search"] == "bank of america"  # no quotes added
    assert fake.iter_calls[0]["limit"] is None  # phrase mode pages until the slice ends
    assert sorted(p.post_id for p in res.posts) == ["telegram:news:3", "telegram:news:5"]
    assert ("2 result(s) without the exact phrase were not kept (topic match: phrase)"
            in st.reason)
    assert st.state == SourceState.ok


def test_api_channel_words_mode_keeps_everything(tmp_path):
    msgs = [msg(5, utc(2026, 11, 19, 10), "Bank-of-America cuts rates"),
            msg(4, utc(2026, 11, 19, 9), "a bank in north america")]
    res, fake = _api_run(tmp_path, {"news": msgs}, limit=30, topic="bank of america",
                         match="words")
    (st,) = res.status
    assert fake.iter_calls[0]["search"] == "bank of america"
    assert len(res.posts) == 2 and "exact phrase" not in st.reason


def test_api_channels_share_cap_and_quiet_channel_rolls_over(tmp_path):
    busy = _day_msgs([(19, 9, "t"), (18, 9, "t"), (17, 9, "t")])
    quiet = _day_msgs([(19, 1, "t")])
    # Busy first: ceil(5/2)=3 for busy, the remaining 2 for quiet (which has 1).
    res, _ = _api_run(tmp_path, {"busy": busy, "quiet": quiet}, limit=5, seeds="busy,quiet")
    assert len(res.posts) == 4 and [s.count for s in res.status] == [3, 1]
    # Quiet first: it uses 1 of its 3, busy gets the remaining 4; total never above cap.
    res, _ = _api_run(tmp_path, {"busy": busy, "quiet": quiet}, limit=5, seeds="quiet,busy")
    assert len(res.posts) == 5 and [s.count for s in res.status] == [1, 4]


def _global_pages_handler(pages, seen_queries):
    pages = list(pages)

    async def handler(req):
        name = type(req).__name__
        seen_queries.append((name, req.query))
        if name == "CheckSearchPostsFloodRequest":
            return SimpleNamespace(total_daily=10, remains=5, query_is_free=False)
        assert req.allow_paid_stars is None  # never pay
        return pages.pop(0)
    return handler


def _global_run(tmp_path, monkeypatch, pages, *, limit, topic="t", **opts):
    _write_session(API_CFG)
    seen: list = []
    fake = FakeClient({}, {}, call_handler=_global_pages_handler(pages, seen))
    monkeypatch.setattr("telethon.utils.get_input_peer", lambda e: ("peer", e.id))
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_HAS_PREMIUM": "true"})
    query = Query(topic=topic, start=NOV["start"], end=NOV["end"], limit=limit,
                  options=opts)
    return col.collect(query, ctx), seen


def test_api_global_balances_days_client_side_and_stops_past_start(tmp_path, monkeypatch):
    a = channel(cid=111, username="alpha")
    page = SimpleNamespace(
        messages=_day_msgs([(19, 4, "t"), (17, 1, "t"), (16, 1, "t")]),
        chats=[a], users=[], next_rate=7)
    # One page only: the walk passes the window start (Nov 16), so no second request.
    res, seen = _global_run(tmp_path, monkeypatch, [page], limit=6)
    (st,) = res.status
    assert st.state == SourceState.ok, st.reason
    assert [n for n, _ in seen].count("SearchPostsRequest") == 1
    days = [p.published_at[:10] for p in res.posts]
    # Updated 2026-11-19 (cap decided after the walk): Nov 18/17 cannot use their share
    # of 2, so Nov 19 keeps all 4 of its candidates (5 of the cap of 6 are used).
    assert days == ["2026-11-19"] * 4 + ["2026-11-17"]
    assert "global search has no date filter; reached back to 2026-11-16" in st.reason
    assert ("per-day: 2026-11-19 4, 2026-11-18 0 (fewer than its share of 2), "
            "2026-11-17 1 (fewer than its share of 2)") in st.reason


def test_api_global_page_cap_names_unreached_days(tmp_path, monkeypatch):
    monkeypatch.setattr(tg, "GLOBAL_MAX_PAGES", 2)
    a = channel(cid=111, username="alpha")
    pages = [SimpleNamespace(messages=[msg(10 - i, utc(2026, 11, 19, 20 - i), "t")],
                             chats=[a], users=[], next_rate=5) for i in range(3)]
    res, _ = _global_run(tmp_path, monkeypatch, pages, limit=30)
    (st,) = res.status
    assert st.state == SourceState.partial and "page cap (2) reached" in st.reason
    assert "reached back to 2026-11-19 19:00 UTC" in st.reason
    assert "2026-11-18 not reached" in st.reason and "2026-11-17 not reached" in st.reason
    assert st.cursor == "2026-11-19:global:offset_rate=5:offset_id=9"
    assert len(res.posts) == 2


def test_api_global_phrase_mode_filters_and_sends_plain_topic(tmp_path, monkeypatch):
    a = channel(cid=111, username="alpha")
    page = SimpleNamespace(
        messages=[msg(3, utc(2026, 11, 19), "Bank of America"),
                  msg(2, utc(2026, 11, 18), "bank ... america")],
        chats=[a], users=[], next_rate=None)
    res, seen = _global_run(tmp_path, monkeypatch, [page], limit=30, topic="bank of america")
    (st,) = res.status
    assert {q_ for _, q_ in seen} == {"bank of america"}  # quota check + search: no quotes
    assert [p.post_id for p in res.posts] == ["telegram:alpha:3"]
    assert "1 result(s) without the exact phrase were not kept" in st.reason
    assert st.state == SourceState.ok


@respx.mock
def test_web_mode_phrase_filter_and_plain_query(tmp_path):
    respx.get("https://t.me/s/durov").mock(return_value=httpx.Response(200, text=DUROV))
    col = tg.TelegramCollector(sleep=lambda s: None)
    ctx = make_ctx(tmp_path, {"TELEGRAM_SEED_CHANNELS": "durov"})
    res = col.collect(q(utc(2026, 8, 1), utc(2026, 8, 31, 23, 59, 59),
                        topic="International  Olympiad"), ctx)
    (st,) = res.status
    assert respx.calls[0].request.url.params["q"] == "International Olympiad"
    assert sorted(p.post_id for p in res.posts) == [
        "telegram:durov:539", "telegram:durov:540", "telegram:durov:542"]
    assert "6 result(s) without the exact phrase were not kept" in st.reason
    assert st.state == SourceState.ok and "per-slice:" in st.reason


@respx.mock
def test_web_mode_words_mode_keeps_all(tmp_path):
    respx.get("https://t.me/s/durov").mock(return_value=httpx.Response(200, text=DUROV))
    col = tg.TelegramCollector(sleep=lambda s: None)
    ctx = make_ctx(tmp_path, {"TELEGRAM_SEED_CHANNELS": "durov"})
    res = col.collect(q(utc(2026, 8, 1), utc(2026, 8, 31, 23, 59, 59),
                        topic="International Olympiad", match="words"), ctx)
    assert len(res.posts) == 9 and "exact phrase" not in res.status[0].reason


# ================================================ review regressions (2026-11-19)


def test_api_channel_phrase_mode_checks_time_budget_on_dropped_results(tmp_path):
    # Phrase mode pages a slice with limit=None; a long run of non-phrase hits must still
    # honour the time budget (it used to be checked only after a kept post).
    msgs = [msg(1000 - i, utc(2026, 11, 19, 20, 0, 59 - i), "bank in america")
            for i in range(50)]
    msgs.append(msg(900, utc(2026, 11, 19, 1), "bank of america"))

    def hook(ctx, fake):
        ctx.out_of_time = lambda: len(fake.iter_calls) >= 1

    res, fake = _api_run(tmp_path, {"news": msgs}, limit=9, topic="bank of america",
                         ctx_hook=hook)
    (st,) = res.status
    assert st.state == SourceState.partial and "time budget exhausted" in st.reason
    assert res.posts == []
    assert st.cursor == "2026-11-19:news:offset_id=1000"  # stopped at the first result
    assert "2026-11-18 not reached" in st.reason


def test_unreached_channel_and_resolve_flood_reasons_carry_per_day_summary(tmp_path):
    _write_session(API_CFG)
    a, b = channel(username="news"), channel(cid=2, username="other")

    class FloodOnResolve(FakeClient):
        async def get_entity(self, name):
            raise FloodWaitError(request=None, capture=500)

    fake = FloodOnResolve({"news": a, "other": b}, {"news": [], "other": []})
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_SEED_CHANNELS": "news,other"})
    res = col.collect(Query(topic="t", limit=9, **NOV), ctx)
    s1, s2 = res.status
    for s in (s1, s2):
        assert s.state == SourceState.partial
        assert ("per-day: 2026-11-19 not reached, 2026-11-18 not reached, "
                "2026-11-17 not reached") in s.reason, s.reason


@respx.mock
def test_web_unreached_channel_reason_carries_per_day_summary(tmp_path):
    respx.get("https://t.me/s/durov").mock(return_value=httpx.Response(200, text=DUROV))
    col = tg.TelegramCollector(sleep=lambda s: None)
    ctx = make_ctx(tmp_path, {"TELEGRAM_SEED_CHANNELS": "durov,other"})
    query = q(utc(2026, 8, 1), utc(2026, 8, 31, 23, 59, 59))
    query.limit = 1  # durov takes ceil(1/2)=1 -> @other is never reached
    res = col.collect(query, ctx)
    st = res.status[-1]
    assert "@other: not reached" in st.reason and st.state == SourceState.partial
    assert "2026-08-01..2026-08-03 not reached" in st.reason


# ============================== cap split after the walk (no date filter), 2026-11-19
# Live run 2026-11-19 ("bank of america", 7 days, cap 300): the global walk's 10 pages
# only reached 2026-11-19 01:02 and the old walk kept 300/7 = 43 posts. Candidates are
# now bucketed per day and the cap is split after the walk (window.fill_slices).

WEEK = dict(start=utc(2026, 11, 13), end=utc(2026, 11, 19, 23, 59, 59))


def _page(messages, next_rate=5):
    return SimpleNamespace(messages=messages, chats=[channel(cid=111, username="alpha")],
                           users=[], next_rate=next_rate)


def _global_week(tmp_path, monkeypatch, pages, *, limit, topic="t", **opts):
    _write_session(API_CFG)
    seen: list = []
    fake = FakeClient({}, {}, call_handler=_global_pages_handler(pages, seen))
    monkeypatch.setattr("telethon.utils.get_input_peer", lambda e: ("peer", e.id))
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_HAS_PREMIUM": "true"})
    return col.collect(Query(topic=topic, limit=limit, options=opts, **WEEK), ctx), seen


def _ids(posts):
    return [int(p.post_id.rsplit(":", 1)[1]) for p in posts]


def test_global_live_scenario_all_candidates_on_newest_day_fill_the_cap(tmp_path,
                                                                         monkeypatch):
    monkeypatch.setattr(tg, "GLOBAL_MAX_PAGES", 9)
    # 900 phrase-matching posts, all on Nov 19, newest first (one every 90 s).
    base = utc(2026, 11, 19, 23, 59, 0)
    msgs = [msg(5000 - k, base - timedelta(seconds=90 * k), "Bank of America news")
            for k in range(900)]
    pages = [_page(msgs[i:i + 100]) for i in range(0, 900, 100)]
    res, seen = _global_week(tmp_path, monkeypatch, pages, limit=300,
                             topic="bank of america")
    (st,) = res.status
    assert [n for n, _ in seen].count("SearchPostsRequest") == 9  # no early stop
    assert len(res.posts) == 300 and st.count == 300
    assert _ids(res.posts) == list(range(5000, 4700, -1))  # the newest 300
    assert st.state == SourceState.partial, st.reason
    assert "global search has no date filter; reached back to 2026-11-19 01:30" in st.reason
    assert "page cap (9) reached before the window start" in st.reason
    assert "900 in-window candidate(s)" in st.reason
    assert ("per-day: 2026-11-19 300, 2026-11-18 not reached, 2026-11-17 not reached"
            in st.reason)
    assert "2026-11-13 not reached" in st.reason
    assert st.cursor.startswith("2026-11-19:global:")


def test_global_three_days_with_a_quiet_day_share_the_cap(tmp_path, monkeypatch):
    # Nov 19: 10, Nov 18: none, Nov 17: 3, then a Nov 16 post (window start passed).
    msgs = _day_msgs([(19, 10, "t"), (17, 3, "t"), (16, 1, "t")])
    res, seen = _global_run(tmp_path, monkeypatch, [_page(msgs)], limit=9)
    (st,) = res.status
    assert st.state == SourceState.ok, st.reason
    assert [n for n, _ in seen].count("SearchPostsRequest") == 1
    days = [p.published_at[:10] for p in res.posts]
    assert days.count("2026-11-19") == 6 and days.count("2026-11-17") == 3
    # Newest 6 of Nov 19 (20:00 down to 15:00).
    assert [p.published_at[11:13] for p in res.posts][:6] == ["20", "19", "18", "17",
                                                              "16", "15"]
    assert ("per-day: 2026-11-19 6, 2026-11-18 0 (fewer than its share of 3), "
            "2026-11-17 3") in st.reason
    assert "not reached" not in st.reason


def test_global_undated_posts_never_push_the_total_over_the_cap(tmp_path, monkeypatch):
    dated = _day_msgs([(19, 3, "t"), (18, 3, "t")])
    undated = [msg(1, None, "t"), msg(2, None, "t")]
    msgs = dated[:2] + undated + dated[2:] + [msg(50, utc(2026, 11, 16), "t")]
    # Cap 4: dated candidates (6) fill it; the 2 undated posts are not kept.
    res, _ = _global_run(tmp_path, monkeypatch, [_page(msgs)], limit=4)
    (st,) = res.status
    assert len(res.posts) == 4 and all(p.timestamp_verified for p in res.posts)
    assert "2 without a platform timestamp not kept (cap used by dated posts)" in st.reason
    # Cap 8: 6 dated + the 2 undated fit; total still within the cap.
    res, _ = _global_run(tmp_path, monkeypatch, [_page(msgs)], limit=8)
    (st,) = res.status
    assert len(res.posts) == 8
    assert sum(not p.timestamp_verified for p in res.posts) == 2
    assert "2 without a platform timestamp kept as unverified" in st.reason
    # Cap 7: one undated slot left.
    res, _ = _global_run(tmp_path, monkeypatch, [_page(msgs)], limit=7)
    assert len(res.posts) == 7 and sum(not p.timestamp_verified for p in res.posts) == 1


def test_global_out_of_order_and_duplicate_results(tmp_path, monkeypatch):
    # Results arrive out of order and page 2 repeats page 1's last message.
    p1 = [msg(10, utc(2026, 11, 18, 9), "t"), msg(11, utc(2026, 11, 19, 8), "t"),
          msg(12, utc(2026, 11, 19, 21), "t"), msg(13, utc(2026, 11, 17, 5), "t")]
    p2 = [msg(13, utc(2026, 11, 17, 5), "t"), msg(14, utc(2026, 11, 17, 7), "t"),
          msg(9, utc(2026, 11, 16, 23), "t")]
    # Cap 4 (equal share 2): page 1 leaves Nov 18 and Nov 17 short, so page 2 is read.
    res, seen = _global_run(tmp_path, monkeypatch, [_page(p1), _page(p2)], limit=4)
    (st,) = res.status
    assert st.state == SourceState.ok, st.reason
    assert [n for n, _ in seen].count("SearchPostsRequest") == 2
    # Nov 19 keeps both; Nov 17 keeps its newest (14, not 13); walk order is preserved.
    assert _ids(res.posts) == [10, 11, 12, 14]
    assert "1 duplicate result(s) ignored" in st.reason
    # Equal shares of 4 over 3 days are 2/1/1 (the remainder goes to the newest day).
    assert "per-day: 2026-11-19 2, 2026-11-18 1, 2026-11-17 1" in st.reason
    # Cap 3 (share 1): page 1 already gives every day one candidate -> stop early; the
    # newer of Nov 19's two out-of-order results (12) is kept.
    res, seen = _global_run(tmp_path, monkeypatch, [_page(p1), _page(p2)], limit=3)
    (st,) = res.status
    assert [n for n, _ in seen].count("SearchPostsRequest") == 1
    assert _ids(res.posts) == [10, 12, 13] and st.state == SourceState.ok
    assert "stopped early" in st.reason


def test_global_phrase_drops_counted_once_and_words_mode_keeps_all(tmp_path, monkeypatch):
    def pages():
        p1 = [msg(5, utc(2026, 11, 19, 10), "Bank of America"),
              msg(4, utc(2026, 11, 19, 9), "bank in america")]
        p2 = [msg(4, utc(2026, 11, 19, 9), "bank in america"),  # repeated: not re-counted
              msg(3, utc(2026, 11, 18, 9), "bank, america"),
              msg(1, utc(2026, 11, 15), "Bank of America")]
        return [_page(p1), _page(p2)]

    res, _ = _global_run(tmp_path, monkeypatch, pages(), limit=30, topic="bank of america")
    (st,) = res.status
    assert _ids(res.posts) == [5]
    assert "2 result(s) without the exact phrase were not kept" in st.reason
    res, _ = _global_run(tmp_path, monkeypatch, pages(), limit=30, topic="bank of america",
                         match="words")
    (st,) = res.status
    assert _ids(res.posts) == [5, 4, 3] and "exact phrase" not in st.reason
    assert st.state == SourceState.ok


def test_global_limit_zero_is_not_reported_as_no_posts(tmp_path, monkeypatch):
    # Global search runs last (after targeted channels); with no cap left it is reported
    # as "not reached (post limit reached)", a partial, never completed_zero, and never
    # calls SearchPostsRequest.
    res, seen = _global_run(tmp_path, monkeypatch, [], limit=0)
    (st,) = res.status
    assert st.state == SourceState.partial and "global search: not reached" in st.reason
    assert [n for n, _ in seen].count("SearchPostsRequest") == 0


def test_api_channel_tops_up_busy_day_with_cap_left_by_quiet_days(tmp_path):
    # Only Nov 19 has posts. Main pass: Nov 19 takes its share (3), Nov 18 runs out of
    # history, Nov 17 is known empty. 6 of the cap of 9 are left -> Nov 19 is searched
    # again from where it stopped (offset_id) for the rest.
    msgs = _day_msgs([(19, 12, "t")])
    res, fake = _api_run(tmp_path, {"news": msgs}, limit=9)
    (st,) = res.status
    assert st.state == SourceState.ok, st.reason
    assert len(res.posts) == 9 and _ids(res.posts) == list(range(1000, 991, -1))
    last = fake.iter_calls[-1]
    assert last["offset_date"] == utc(2026, 11, 20) and last["offset_id"] == 998
    assert last["limit"] == 6
    assert "6 more post(s) taken from busier days" in st.reason
    assert "per-day: 2026-11-19 9, 2026-11-18 0" in st.reason


def test_api_channel_top_up_split_between_busy_days_and_time_budget(tmp_path):
    # Nov 19 and Nov 18 busy, Nov 17 quiet: cap 9 -> 3 + 3 + 0, then the leftover 3 is
    # split ceil(3/2)=2 for Nov 19 and 1 for Nov 18.
    msgs = _day_msgs([(19, 6, "t"), (18, 6, "t")])
    res, fake = _api_run(tmp_path, {"news": msgs}, limit=9)
    (st,) = res.status
    days = [p.published_at[:10] for p in res.posts]
    assert days.count("2026-11-19") == 5 and days.count("2026-11-18") == 4
    assert st.state == SourceState.ok and "per-day: 2026-11-19 5, 2026-11-18 4" in st.reason

    # Time runs out before the top-up: every slice was searched, so still ok, with a note.
    def hook(ctx, fake):
        ctx.out_of_time = lambda: len(fake.iter_calls) >= 3

    res, fake = _api_run(tmp_path, {"news": msgs}, limit=9, ctx_hook=hook)
    (st,) = res.status
    assert len(res.posts) == 6 and st.state == SourceState.ok, st.reason
    assert "top-up of busier days stopped (time budget exhausted)" in st.reason


def test_api_channel_words_mode_no_top_up_when_cap_used(tmp_path):
    msgs = _day_msgs([(19, 5, "t a"), (18, 5, "t b"), (17, 5, "t c")])
    res, fake = _api_run(tmp_path, {"news": msgs}, limit=6)
    assert len(res.posts) == 6 and len(fake.iter_calls) == 3
    assert "more post(s) taken" not in res.status[0].reason


# ======================================= adversarial review regressions (2026-11-19)


def test_api_channel_top_up_repeats_until_cap_used_or_days_exhausted(tmp_path):
    # Nov 19 busy (100), Nov 18 has exactly its share (10), Nov 17 quiet. Cap 30: main
    # pass 10 + 10 + 0. One top-up round split the leftover 10 as 5 (Nov 19) + 5 (Nov 18,
    # which had nothing more), so 5 were wasted while Nov 19 still had 90 posts.
    msgs = ([msg(2000 - k, utc(2026, 11, 19, 20) - timedelta(minutes=k), "t")
             for k in range(100)] + _day_msgs([(18, 10, "t")]))
    res, fake = _api_run(tmp_path, {"news": msgs}, limit=30)
    (st,) = res.status
    assert st.state == SourceState.ok, st.reason
    days = [p.published_at[:10] for p in res.posts]
    assert len(res.posts) == 30, st.reason
    assert days.count("2026-11-19") == 20 and days.count("2026-11-18") == 10
    assert len({p.post_id for p in res.posts}) == 30  # no duplicates from resumed searches
    assert "per-day: 2026-11-19 20, 2026-11-18 10" in st.reason


# ================================================== joined groups / channels (searchGlobal)


def _joined_handler(seen: list, per_scope: dict):
    """messages.searchGlobal fake: per_scope["groups"/"channels"] -> list of pages (each a
    SimpleNamespace(messages, chats, users, next_rate)), served in order for every slice."""
    async def handler(req):
        name = type(req).__name__
        assert name == "SearchGlobalRequest", name
        scope = "groups" if req.groups_only else "channels" if req.broadcasts_only else None
        assert scope is not None and not req.users_only  # one-to-one chats never searched
        seen.append((scope, req.min_date, req.max_date, req.q))
        pages = per_scope.get(scope) or []
        return pages.pop(0) if pages else SimpleNamespace(messages=[], chats=[], users=[],
                                                          next_rate=None)
    return handler


def test_joined_chats_search_groups_and_channels(tmp_path):
    _write_session(API_CFG)
    grp = SimpleNamespace(id=50, username=None, title="Private group", megagroup=True,
                          broadcast=False, left=False, date=None)
    chan = channel(cid=111, username="alpha")
    alice = SimpleNamespace(id=7, username="alice", first_name="Alice", last_name=None)
    g_page = SimpleNamespace(
        messages=[msg(1, utc(2026, 7, 2, 10), "bank of america is down", post=False,
                      from_id=SimpleNamespace(user_id=7), peer_id=SimpleNamespace(channel_id=50)),
                  msg(2, utc(2026, 7, 2, 9), "america bank of", post=False,     # no phrase
                      from_id=SimpleNamespace(user_id=7), peer_id=SimpleNamespace(channel_id=50)),
                  msg(3, utc(2026, 7, 2, 8), "bank of america dm", post=False,  # a DM: never kept
                      peer_id=SimpleNamespace(user_id=7))],
        chats=[grp], users=[alice], next_rate=None)
    c_page = SimpleNamespace(messages=[msg(9, utc(2026, 7, 2, 7), "Bank of America news")],
                             chats=[chan], users=[], next_rate=None)
    seen: list = []
    fake = FakeClient({}, {}, call_handler=_joined_handler(seen, {"groups": [g_page],
                                                                  "channels": [c_page]}))
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_SEARCH_JOINED_CHATS": "true"})
    res = col.collect(q(utc(2026, 7, 1), utc(2026, 7, 2, 23, 59, 59), topic="bank of america"),
                      ctx)
    by_id = {p.post_id: p for p in res.posts}
    assert set(by_id) == {"telegram:50:1", "telegram:alpha:9"}
    assert by_id["telegram:50:1"].url == "https://t.me/c/50/1"
    assert by_id["telegram:50:1"].author_handle == "@alice"
    assert by_id["telegram:alpha:9"].url == "https://t.me/alpha/9"
    # date-bounded: one search per day slice and scope, newest slice first
    assert [(sc, a.date().isoformat()) for sc, a, _b, _q in seen] == [
        ("groups", "2026-07-02"), ("groups", "2026-07-01"),
        ("channels", "2026-07-02"), ("channels", "2026-07-01")]
    sts = {s.reason.split(" (")[0]: s for s in res.status}
    g = sts["API mode; joined groups"]
    assert g.state == SourceState.ok and g.date_filtered
    assert "1 result(s) without the exact phrase were not kept" in g.reason
    assert "1 result(s) from one-to-one chats not kept" in g.reason


def test_joined_chats_off_unless_enabled(tmp_path):
    _write_session(API_CFG)
    seen: list = []
    fake = FakeClient({}, {}, call_handler=_joined_handler(seen, {}))
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    res = col.collect(q(utc(2026, 7, 1), utc(2026, 7, 2)), make_ctx(tmp_path, API_CFG))
    assert seen == [] and res.status[0].state == SourceState.skipped_capability


def test_joined_chats_flood_is_partial(tmp_path):
    _write_session(API_CFG)

    async def handler(req):
        raise FloodWaitError(request=None, capture=600)
    fake = FakeClient({}, {}, call_handler=handler)
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_SEARCH_JOINED_CHATS": "true"})
    res = col.collect(q(utc(2026, 7, 1), utc(2026, 7, 2)), ctx)
    g, c = res.status
    assert g.state == SourceState.partial and "flood wait 600s" in g.reason
    assert c.state == SourceState.partial and "not reached (account in flood wait 600s)" in c.reason


def test_default_config_enables_joined_chats():
    from socmint.config import DEFAULTS
    assert tg.is_true(DEFAULTS["TELEGRAM_SEARCH_JOINED_CHATS"])


# ================================================================= channel discovery


def _discover_handler(seen: list, found, recs=None):
    async def handler(req):
        name = type(req).__name__
        seen.append(name)
        if name == "SearchRequest":
            assert req.q == "t" and req.limit == tg.DISCOVER_SEARCH_LIMIT
            return found
        if name == "GetChannelRecommendationsRequest":
            return recs or SimpleNamespace(chats=[])
        if name == "SearchGlobalRequest":  # joined-chats search: nothing there
            return SimpleNamespace(messages=[], chats=[], users=[], next_rate=None)
        raise AssertionError(name)
    return handler


def test_channel_discovery_searches_found_public_channels_by_date(tmp_path, monkeypatch):
    _write_session(API_CFG)
    monkeypatch.setattr("telethon.utils.get_input_channel", lambda c: c)
    news = channel(cid=1, username="news")                    # public channel: kept
    grp = SimpleNamespace(id=2, username="talkgrp", title="Talk", megagroup=True,
                          broadcast=False, left=True, date=None)   # public supergroup: kept
    priv = channel(cid=3, username=None)                      # no username: cannot search
    bot = SimpleNamespace(id=4, username="somebot", first_name="Bot", bot=True)  # not a chat
    joined = channel(cid=5, username="mine", left=False)      # joined: left to joined search
    similar = channel(cid=6, username="similar")              # from recommendations
    found = SimpleNamespace(
        my_results=[SimpleNamespace(channel_id=5)],
        results=[SimpleNamespace(channel_id=1), SimpleNamespace(channel_id=2),
                 SimpleNamespace(channel_id=3), SimpleNamespace(user_id=4)],
        chats=[news, grp, priv, joined], users=[bot])
    seen: list = []
    fake = FakeClient({}, {"news": _api_msgs(), "talkgrp": [], "similar": _api_msgs()},
                      call_handler=_discover_handler(seen, found,
                                                     SimpleNamespace(chats=[similar, news])))
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_DISCOVER_CHANNELS": "true",
                              "TELEGRAM_SEARCH_JOINED_CHATS": "true"})
    res = col.collect(q(utc(2026, 7, 2), utc(2026, 7, 25, 23, 59, 59), topic="t"), ctx)
    disc = next(s for s in res.status if "channel discovery:" in s.reason)
    assert "1 channel(s) you have joined left to the joined-chats search" in disc.reason
    assert disc.reason.startswith('API mode; channel discovery: 3 public channel(s) found for '
                                  '"t" (2 by name, 1 similar), each '
                                  'searched by date below: @news, @talkgrp, @similar')
    # each discovered channel searched date-bounded, without a username lookup
    assert [c["entity"] for c in fake.iter_calls if c["entity"]] and \
        {c["entity"] for c in fake.iter_calls} == {"news", "talkgrp", "similar"}
    rows = [s.reason for s in res.status if s.reason.startswith("API mode; @")]
    assert len(rows) == 3 and all("(found by channel discovery)" in r for r in rows)
    ids = {p.post_id for p in res.posts}
    assert "telegram:news:5" in ids and "telegram:similar:5" in ids
    assert not any(p.post_id.startswith("telegram:mine") for p in res.posts)


def test_channel_discovery_skips_seeds_and_reports_nothing_found(tmp_path):
    _write_session(API_CFG)
    found = SimpleNamespace(my_results=[], results=[], chats=[], users=[])
    seen: list = []
    fake = FakeClient({"news": channel(username="news")}, {"news": []},
                      call_handler=_discover_handler(seen, found))
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_DISCOVER_CHANNELS": "true",
                              "TELEGRAM_SEED_CHANNELS": "news"})
    res = col.collect(q(utc(2026, 7, 2), utc(2026, 7, 3), topic="t"), ctx)
    disc = res.status[0]
    assert disc.state == SourceState.completed_zero and "listed no public channel" in disc.reason
    assert seen == ["SearchRequest"]  # no recommendations without a match


def test_channel_discovery_off_unless_enabled(tmp_path):
    _write_session(API_CFG)
    seen: list = []
    fake = FakeClient({}, {}, call_handler=_discover_handler(seen, None))
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    col.collect(q(utc(2026, 7, 2), utc(2026, 7, 3), topic="t"), make_ctx(tmp_path, API_CFG))
    assert seen == []


def test_channel_discovery_flood_is_partial(tmp_path):
    _write_session(API_CFG)

    async def handler(req):
        raise FloodWaitError(request=None, capture=900)
    fake = FakeClient({}, {}, call_handler=handler)
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_DISCOVER_CHANNELS": "true"})
    res = col.collect(q(utc(2026, 7, 2), utc(2026, 7, 3), topic="t"), ctx)
    assert res.status[0].state == SourceState.partial
    assert "flood wait 900s" in res.status[0].reason


def test_default_config_enables_channel_discovery():
    from socmint.config import DEFAULTS
    assert tg.is_true(DEFAULTS["TELEGRAM_DISCOVER_CHANNELS"])
    assert DEFAULTS["TELEGRAM_DISCOVER_MAX"] == "15"


def test_channel_discovery_searches_key_word_and_ranks_by_audience(tmp_path):
    # Live 2026-11-20: "saudi aramco" and "aramco" return different ~10-channel lists.
    _write_session(API_CFG)
    small = channel(cid=1, username="small")
    small.participants_count = 50
    big = channel(cid=2, username="big")
    big.participants_count = 9000
    asked: list = []

    async def handler(req):
        name = type(req).__name__
        if name == "SearchRequest":
            asked.append(req.q)
            chats = [small] if req.q == "saudi aramco" else [big, small]
            return SimpleNamespace(my_results=[], results=[], chats=chats, users=[])
        if name == "GetChannelRecommendationsRequest":
            return SimpleNamespace(chats=[])
        raise AssertionError(name)
    fake = FakeClient({}, {"small": [], "big": []}, call_handler=handler)
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_DISCOVER_CHANNELS": "true",
                              "TELEGRAM_DISCOVER_MAX": "1"})
    res = col.collect(q(utc(2026, 7, 2), utc(2026, 7, 3), topic="saudi aramco"), ctx)
    assert asked == ["saudi aramco", "aramco"]
    disc = res.status[0]
    assert 'found for "saudi aramco" / "aramco"' in disc.reason
    assert disc.reason.endswith("@big; stopped at TELEGRAM_DISCOVER_MAX=1")  # biggest first


def test_global_search_runs_last_and_does_not_seed_discovery(tmp_path):
    # Owner decision 2026-11-20: the targeted channels are searched FIRST and the noisy
    # global search runs LAST with the leftover cap. Global-search result channels are no
    # longer used as discovery seeds (discovery uses the directory + the known list); global
    # still collects its own in-window posts afterwards, deduped.
    _write_session(API_CFG)
    big = channel(cid=7, username="bignews")
    glob = SimpleNamespace(
        messages=[msg(50, utc(2026, 7, 2, 12), "t update", peer_id=SimpleNamespace(channel_id=7)),
                  msg(49, utc(2026, 7, 2, 11), "t again", peer_id=SimpleNamespace(channel_id=7))],
        chats=[big], users=[], next_rate=None)
    order: list = []

    async def handler(req):
        name = type(req).__name__
        order.append(name)
        if name == "CheckSearchPostsFloodRequest":
            return SimpleNamespace(total_daily=10, remains=5, query_is_free=True)
        if name == "SearchPostsRequest":
            return glob
        if name == "SearchRequest":              # directory discovery: nothing found by name
            return SimpleNamespace(my_results=[], results=[], chats=[], users=[])
        if name == "GetChannelRecommendationsRequest":
            return SimpleNamespace(chats=[])
        raise AssertionError(name)
    fake = FakeClient({}, {}, call_handler=handler)
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_HAS_PREMIUM": "true",
                              "TELEGRAM_DISCOVER_CHANNELS": "true"})
    res = col.collect(q(utc(2026, 7, 1), utc(2026, 7, 3, 23, 59, 59), topic="t"), ctx)
    # global search's own in-window posts are still collected (it just runs last)
    assert {p.post_id for p in res.posts} == {"telegram:bignews:50", "telegram:bignews:49"}
    # discovery found nothing by name and never used global-search results as a seed
    disc = next(s for s in res.status if "channel discovery:" in s.reason)
    assert "from global-search results" not in disc.reason
    # order: the discovery directory search happened BEFORE the global post search
    assert order.index("SearchRequest") < order.index("SearchPostsRequest")


# ============================================================ underground / leak discovery


def test_underground_discovery_vocabulary_and_curated_list(tmp_path, monkeypatch):
    # Non-premium: no global search, so underground = leak-vocabulary directory matches +
    # the bundled known-channel list; each is searched for the topic by date.
    _write_session(API_CFG)
    monkeypatch.setattr(tg.ug, "VOCAB", ("logs",))
    monkeypatch.setattr(tg.ug, "curated_channels", lambda: (
        {"username": "knownleak", "status": "online", "label": "", "categories": ["infostealer"]},
        {"username": "deadchan", "status": "online", "label": "", "categories": ["market"]}))
    monkeypatch.setattr(tg.ug, "curated_usernames", lambda: {"knownleak", "deadchan", "freshlogs"})
    fresh = channel(cid=10, username="freshlogs")        # found by vocabulary (also curated)
    plain = SimpleNamespace(id=11, username="chatty", title="Chat", megagroup=True,
                            broadcast=False, left=True, date=None)  # found by vocab, not curated
    knownleak = channel(cid=12, username="knownleak")    # curated, resolved
    asked: list = []

    async def handler(req):
        name = type(req).__name__
        asked.append(name)
        if name == "SearchRequest":
            assert req.q == "logs"
            return SimpleNamespace(my_results=[], results=[], chats=[fresh, plain], users=[])
        raise AssertionError(name)
    msgs = {"freshlogs": _api_msgs(), "chatty": [], "knownleak": _api_msgs()}
    fake = FakeClient({"knownleak": knownleak}, msgs, call_handler=handler)
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_UNDERGROUND_DISCOVERY": "true",
                              "TELEGRAM_UNDERGROUND_MAX": "10",
                              "TELEGRAM_DISCOVER_CHANNELS": "false",
                              "TELEGRAM_SEARCH_JOINED_CHATS": "false"})
    res = col.collect(q(utc(2026, 7, 2), utc(2026, 7, 25, 23, 59, 59), topic="t"), ctx)
    disc = next(s for s in res.status if "underground discovery:" in s.reason)
    assert disc.state == SourceState.ok
    # known list searched first, then leak-named vocabulary matches; @chatty (not a leak
    # name) is dropped as noise, @deadchan fails to resolve.
    assert "2 leak/underground channel(s) to search" in disc.reason
    assert "(1 from the known list, 1 by leak vocabulary)" in disc.reason
    assert disc.reason.index("@knownleak") < disc.reason.index("@freshlogs")
    assert "@chatty" not in disc.reason
    assert "1 known channel(s) could not be resolved" in disc.reason  # deadchan
    rows = [s.reason for s in res.status if "API mode; @" in s.reason]
    assert any('leak vocabulary "logs"' in r for r in rows)
    assert any("known infostealer channel" in r for r in rows)
    ids = {p.post_id for p in res.posts}
    assert "telegram:freshlogs:5" in ids and "telegram:knownleak:5" in ids


def test_underground_discovery_off_unless_enabled(tmp_path):
    _write_session(API_CFG)
    seen: list = []

    async def handler(req):
        seen.append(type(req).__name__)
        return SimpleNamespace(my_results=[], results=[], chats=[], users=[])
    fake = FakeClient({}, {}, call_handler=handler)
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    col.collect(q(utc(2026, 7, 2), utc(2026, 7, 3), topic="t"),
                make_ctx(tmp_path, {**API_CFG, "TELEGRAM_UNDERGROUND_DISCOVERY": "false",
                                    "TELEGRAM_DISCOVER_CHANNELS": "false"}))
    assert seen == []


def test_underground_discovery_nothing_found(tmp_path, monkeypatch):
    _write_session(API_CFG)
    monkeypatch.setattr(tg.ug, "VOCAB", ("logs",))
    monkeypatch.setattr(tg.ug, "curated_channels", lambda: ())
    monkeypatch.setattr(tg.ug, "curated_usernames", lambda: set())

    async def handler(req):
        return SimpleNamespace(my_results=[], results=[], chats=[], users=[])
    fake = FakeClient({}, {}, call_handler=handler)
    col = tg.TelegramCollector(client_factory=lambda cfg: fake)
    ctx = make_ctx(tmp_path, {**API_CFG, "TELEGRAM_UNDERGROUND_DISCOVERY": "true",
                              "TELEGRAM_DISCOVER_CHANNELS": "false"})
    res = col.collect(q(utc(2026, 7, 2), utc(2026, 7, 3), topic="t"), ctx)
    disc = next(s for s in res.status if "underground discovery:" in s.reason)
    assert disc.state == SourceState.completed_zero
    assert "no public leak/underground channel found" in disc.reason


def test_default_config_enables_underground_discovery():
    from socmint.config import DEFAULTS
    assert tg.is_true(DEFAULTS["TELEGRAM_UNDERGROUND_DISCOVERY"])
    assert DEFAULTS["TELEGRAM_UNDERGROUND_MAX"] == "20"


def test_bundled_underground_channel_list_loads():
    from socmint.collect import underground as u
    chans = u.curated_channels()
    assert len(chans) > 50  # ~396 public live channels bundled from deepdarkCTI
    assert all(c["username"] for c in chans)
    # round-robin across categories: the first channel is the highest-signal category
    # (infostealer), and the first few are not all one category
    first_cats = [(_c.get("categories") or ["?"])[0] for _c in chans[:4]]
    assert first_cats[0] == "infostealer" and len(set(first_cats)) > 1
