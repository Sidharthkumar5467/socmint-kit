"""Payment card masking (socmint/util/pan.py) and where the pipeline applies it.

Every card number here is MADE UP: a network prefix, zeros, and a Luhn check digit
computed by `_card()`. None comes from collected data.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from conftest import make_post

from socmint import pipeline
from socmint.interfaces import CollectContext, CollectResult
from socmint.runstore import read_json, read_jsonl
from socmint.schema import SourceState, SourceStatus
from socmint.util.pan import (
    is_pan,
    luhn_ok,
    mask_file,
    mask_json,
    mask_json_text,
    mask_pans,
)


def _card(prefix: str, length: int) -> str:
    body = prefix + "0" * (length - len(prefix) - 1)
    return next(body + str(d) for d in range(10) if luhn_ok(body + str(d)))


VISA = _card("411111", 16)  # 4111110000000005
MC = _card("527517", 16)
AMEX = _card("371234", 15)
MASKED_VISA = VISA[:6] + "******" + VISA[-4:]
MASKED_MC = MC[:6] + "******" + MC[-4:]


def test_made_up_numbers_are_luhn_valid_cards():
    assert all(luhn_ok(c) and is_pan(c) for c in (VISA, MC, AMEX))
    assert not luhn_ok(VISA[:-1] + str((int(VISA[-1]) + 1) % 10))


@pytest.mark.parametrize("raw, masked", [
    # card-dump bot format: PAN|MM|YYYY|CVV -> BIN + last 4 kept, CVV masked
    (f"💳 NEW CARD Approved! ✅\n▪️ Card: {VISA}|08|2030|956\n▪️ BIN: 411111",
     f"💳 NEW CARD Approved! ✅\n▪️ Card: {MASKED_VISA}|08|2030|***\n▪️ BIN: 411111"),
    (f"💳 Card ➛ {MC}|06|29|664 🔥 Gateway ➛ Shopify",
     f"💳 Card ➛ {MASKED_MC}|06|29|*** 🔥 Gateway ➛ Shopify"),
    (f"-» {MC}|11|26", f"-» {MASKED_MC}|11|26"),  # no CVV given: nothing else changes
    # grouped with spaces or dashes: separators kept
    (f"card {VISA[:4]} {VISA[4:8]} {VISA[8:12]} {VISA[12:]} exp 08/30",
     f"card {VISA[:4]} {VISA[4:6]}** **** {VISA[12:]} exp 08/30"),
    (f"{MC[:4]}-{MC[4:8]}-{MC[8:12]}-{MC[12:]}", f"{MC[:4]}-{MC[4:6]}**-****-{MC[12:]}"),
    # a run "PAN MM YYYY": the 16-digit PAN is found inside it
    (f"{VISA[:4]} {VISA[4:8]} {VISA[8:12]} {VISA[12:]} 08 2030",
     f"{VISA[:4]} {VISA[4:6]}** **** {VISA[12:]} 08 2030"),
    # Amex (15) and a labelled CVV
    (f"amex {AMEX} exp 05/27 cvv: 1234", f"amex {AMEX[:6]}*****{AMEX[-4:]} exp 05/27 cvv: ****"),
])
def test_mask_pans_formats(raw, masked):
    out = mask_pans(raw)
    assert out == masked
    assert mask_pans(out) == out  # idempotent


@pytest.mark.parametrize("text", [
    "https://x.com/someone/status/2122479614538690954",  # 19-digit X status id
    "post id 2122479614538690954 and 7529150605387423744",  # X / LinkedIn ids
    f"https://shop.example/item/{VISA}",  # inside a URL
    f"tel +{VISA}",  # phone-number style
    f"order#{VISA}",  # glued to '#'
    f"ref_{VISA}",  # glued to '_'
    f"{VISA}xxxx|11|26",  # glued to letters (already partly hidden by the poster)
    VISA[:-1] + str((int(VISA[-1]) + 1) % 10),  # Luhn fails
    "1234567890123456",  # no card network starts with 1
    "Call 0044 7911 123456 or 555-0100",  # phone numbers
    "Balance 12,345,678.90 on 2026-11-16 at 12:00",
])
def test_mask_pans_leaves_non_cards_alone(text):
    assert mask_pans(text) == text


def test_mask_pans_inside_json_escaped_text():
    raw = json.dumps({"text": f"Card:\n{VISA}|01|29|123"})
    out = mask_pans(raw)
    assert VISA not in out and MASKED_VISA in out and "|01|29|***" in out
    assert json.loads(out)["text"] == f"Card:\n{MASKED_VISA}|01|29|***"


def test_mask_file_and_save_raw(tmp_path):
    f = tmp_path / "a.html"
    f.write_text(f"<p>{VISA}|08|2030|956</p>", encoding="utf-8")
    assert mask_file(f) is True and mask_file(f) is False
    assert VISA not in f.read_text(encoding="utf-8")
    ctx = CollectContext(run_dir=tmp_path, raw_dir=tmp_path / "raw", config={}, deadline=0)
    for name, payload in (("p.json", {"messages": [{"text": f"cc {MC}|1|27|111"}]}),
                          ("p.html", f"<b>{MC}</b>"), ("p.bin", f"{MC}".encode())):
        rel = ctx.save_raw("telegram", name, payload)
        body = (tmp_path / rel).read_text(encoding="utf-8")
        assert MC not in body and MASKED_MC in body
    ctx.save_raw("telegram", "img.bin", b"\xff\xd8\xff" + MC.encode())  # binary: untouched
    assert (tmp_path / "raw/telegram/img.bin").read_bytes().endswith(MC.encode())


def _dump_run(monkeypatch, tmp_path):
    def fake_collect_one(name, query, ctx, prog):
        ctx.save_raw(name, "page_1.json", {"text": f"Card: {VISA}|08|2030|956"})
        posts = [make_post(f"t{i}", f"bank of america card dump {i}: Card: {VISA}|08|2030|956",
                           source=name, author=f"a{i}", published_at="2026-11-07T10:00:00Z")
                 for i in range(3)]
        return CollectResult(posts=posts, status=[SourceStatus(
            source=name, query=query.topic, state=SourceState.ok, count=3)])

    monkeypatch.setattr(pipeline, "_collect_one", fake_collect_one)
    return pipeline.run("bank of america", date(2026, 11, 6), date(2026, 11, 7),
                        ["telegram"], out_root=tmp_path, cfg={})


def _all_text(run_dir) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore")
                     for p in run_dir.rglob("*") if p.is_file())


def test_pipeline_masks_posts_before_writing_everything(monkeypatch, tmp_path):
    run_dir = _dump_run(monkeypatch, tmp_path)
    posts = read_jsonl(run_dir / "posts.jsonl")
    assert all(MASKED_VISA in p["text"] and "|08|2030|***" in p["text"] for p in posts)
    # posts.jsonl, posts.csv, report.html, raw/: no full card number anywhere
    assert (run_dir / "posts.csv").is_file() and (run_dir / "report.html").is_file()
    assert VISA not in _all_text(run_dir)


def test_reanalyze_masks_an_older_run_in_place(monkeypatch, tmp_path):
    run_dir = _dump_run(monkeypatch, tmp_path)
    # simulate a run saved before masking existed: full numbers on disk
    rows = read_jsonl(run_dir / "posts.jsonl")
    for r in rows:
        r["text"] = r["text"].replace(MASKED_VISA, VISA).replace("|***", "|956")
    (run_dir / "posts.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (run_dir / "raw" / "telegram" / "old.html").write_text(f"<p>{VISA}</p>", encoding="utf-8")
    (run_dir / "llm_log.jsonl").write_text(json.dumps({"prompt": f"[1] {VISA}"}) + "\n",
                                           encoding="utf-8")
    assert VISA in _all_text(run_dir)
    events: list[dict] = []
    pipeline.reanalyze(run_dir, cfg={}, progress=events.append)
    assert VISA not in _all_text(run_dir)
    assert MASKED_VISA in (run_dir / "posts.csv").read_text(encoding="utf-8-sig")
    assert any("masked payment card numbers in" in (e.get("message") or "") for e in events)
    assert read_json(run_dir / "run.json")["status"] == "complete"


# ------------------------------------- 2026-11-20 review: JSON is walked, never re-texted
# A Telegram-style 19-digit id with prefix 6 that passes the card rules (made up here).
TG_ID = _card("604881", 19)


def test_tg_id_would_pass_the_text_rules():
    assert is_pan(TG_ID)  # the reason serialised-JSON masking corrupted raw files


def test_numeric_and_string_ids_untouched_message_masked():
    doc = {"_": "Message", "id": int(TG_ID), "peer_id": {"channel_id": int(TG_ID)},
           "media": {"document": {"id": int(TG_ID), "access_hash": int(TG_ID),
                                  "file_reference": TG_ID, "document_id": TG_ID,
                                  "file_name": f"mtproto:{TG_ID}"}},
           "post_id": TG_ID, "ref": TG_ID,
           "message": f"Card: {VISA}|08|2030|956 id {TG_ID[:16]}",
           TG_ID: "a key is never masked"}
    text = json.dumps(doc, ensure_ascii=False)
    out = json.loads(mask_json_text(text))
    assert out["id"] == int(TG_ID) and out["peer_id"]["channel_id"] == int(TG_ID)
    d = out["media"]["document"]
    assert d["id"] == d["access_hash"] == int(TG_ID)
    assert d["file_reference"] == d["document_id"] == TG_ID == out["post_id"] == out["ref"]
    assert d["file_name"] == f"mtproto:{TG_ID}"
    assert TG_ID in out
    assert out["message"].startswith(f"Card: {MASKED_VISA}|08|2030|***")
    assert mask_json(out) == out  # idempotent


def test_all_digit_string_in_text_field_is_still_masked():
    assert mask_json({"text": VISA}) == {"text": MASKED_VISA}
    assert mask_json({"value": VISA}) == {"value": VISA}  # an identifier-looking string


def test_masked_run_raw_files_all_parse(monkeypatch, tmp_path):
    def fake_collect_one(name, query, ctx, prog):
        ctx.save_raw(name, "api_global_search.json", [
            {"id": int(TG_ID), "access_hash": int(TG_ID), "document_id": int(TG_ID),
             "media": {"file_name": f"mtproto:{TG_ID}"},
             "message": f"Card: {VISA}|08|2030|956"}])
        ctx.save_raw(name, "page.json", json.dumps({"id": int(TG_ID), "text": f"{MC}"}))
        ctx.save_raw(name, "lines.jsonl", f'{{"id": {TG_ID}, "text": "{MC}"}}\n')
        posts = [make_post(f"t{i}", f"bank of america card dump {i}: Card: {VISA}|08|2030|956",
                           source=name, author=f"a{i}", published_at="2026-11-07T10:00:00Z")
                 for i in range(3)]
        return CollectResult(posts=posts, status=[SourceStatus(
            source=name, query=query.topic, state=SourceState.ok, count=3)])

    monkeypatch.setattr(pipeline, "_collect_one", fake_collect_one)
    run_dir = pipeline.run("bank of america", date(2026, 11, 6), date(2026, 11, 7),
                           ["telegram"], out_root=tmp_path, cfg={})
    # simulate an older run: raw file written unmasked, then reanalyze masks it in place
    old = run_dir / "raw" / "telegram" / "old.json"
    old.write_text(json.dumps({"id": int(TG_ID), "message": VISA + "|01|29|123"}, indent=1),
                   encoding="utf-8")
    pipeline.reanalyze(run_dir, cfg={})
    raws = [p for p in (run_dir / "raw").rglob("*") if p.is_file()]
    assert len(raws) >= 4
    for p in raws:
        body = p.read_text(encoding="utf-8")
        if p.suffix == ".json":
            json.loads(body)
        elif p.suffix == ".jsonl":
            [json.loads(ln) for ln in body.splitlines() if ln.strip()]
        assert VISA not in body and MC not in body
    tg = json.loads((run_dir / "raw/telegram/api_global_search.json").read_text("utf-8"))
    assert tg[0]["id"] == tg[0]["access_hash"] == tg[0]["document_id"] == int(TG_ID)
    assert tg[0]["media"]["file_name"] == f"mtproto:{TG_ID}"
    assert json.loads(old.read_text("utf-8"))["id"] == int(TG_ID)
    for name in ("posts.jsonl", "llm_log.jsonl", "evidence_log.jsonl"):
        f = run_dir / name
        if f.is_file():
            [json.loads(ln) for ln in f.read_text("utf-8").splitlines() if ln.strip()]


# ------------------------------------------------------------------ credential masking
@pytest.mark.parametrize("text,expect", [
    ("victim@acme.com:Hunter2!", "victim@acme.com:***"),
    ("mail.aramco.com|hr_admin|Winter2026", "mail.aramco.com|hr_admin|***"),
    ("https://portal.acme.com:443 : jdoe : S3cr3t", "https://portal.acme.com:443 : jdoe : ***"),
    ("PASSWORD: correct-horse", "PASSWORD: ***"),
    ("pwd = letmein", "pwd = ***"),
    ("meeting at 10:30 today", "meeting at 10:30 today"),            # time, not a secret
    ("ratio is 3:2", "ratio is 3:2"),
    ("see http://example.com now", "see http://example.com now"),    # bare url, no secret
    ("foo@bar.com:***", "foo@bar.com:***"),                          # idempotent
])
def test_mask_credentials(text, expect):
    from socmint.util.pan import mask_credentials
    assert mask_credentials(text) == expect


def test_mask_credentials_multiline_stealer_block():
    from socmint.util.pan import mask_free_text
    block = ("URL: https://sso.aramco.com\nUSERNAME: j.doe@aramco.com\nPASSWORD: Qwerty!2026\n"
             "card 5275170000003226")
    out = mask_free_text(block)
    assert "https://sso.aramco.com" in out and "j.doe@aramco.com" in out  # identifiers kept
    assert "Qwerty!2026" not in out and "PASSWORD: ***" in out            # secret masked
    assert "527517******3226" in out                                      # card masked too


def test_mask_post_masks_credentials(tmp_path):
    from socmint.schema import Post
    from socmint.util.pan import mask_post
    p = Post(post_id="telegram:x:1", source="telegram", kind="post",
             text="fresh combo\nuser@corp.com:Spring2026\ncontact @seller", url=None,
             collected_at="2026-11-20T00:00:00Z")
    mask_post(p)
    assert "user@corp.com:***" in p.text and "Spring2026" not in p.text
    assert "@seller" in p.text  # handles are not secrets
