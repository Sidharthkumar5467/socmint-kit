"""CSV exports: columns, unicode preservation, formula-injection guard."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from conftest import make_post  # tests/ is on sys.path (no package __init__)

from socmint.export import (
    CLAIM_COLUMNS,
    CLUSTER_COLUMNS,
    NARRATIVE_COLUMNS,
    POST_COLUMNS,
    SOURCE_STATUS_COLUMNS,
    safe_cell,
    write_exports,
)
from socmint.runstore import write_json, write_jsonl
from socmint.schema import (
    AdMeta,
    Claim,
    Cluster,
    Evidence,
    Media,
    Metrics,
    Narrative,
    Relation,
    Signal,
    SourceState,
    SourceStatus,
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], bytes]:
    raw = path.read_bytes()
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    return header, [dict(zip(header, r, strict=True)) for r in body], raw


def _make_run(tmp_path: Path) -> Path:
    run = tmp_path / "run1"
    run.mkdir()
    posts = [
        make_post("1", "Привет мир — مرحبا بالعالم 你好 🌍", lang="ru",
                  metrics=Metrics(views=10, likes=2, quotes=1),
                  media=[Media(type="photo", url="https://img.test/a.jpg"),
                         Media(type="video", url="https://v.test/b.mp4", thumbnail="https://v.test/b.jpg")],
                  relations=[Relation(type="quote", target_post_id="x:0"),
                             Relation(type="link", target_url="https://news.test/a")],
                  raw_ref="raw/x/page1.json", query="topic"),
        make_post("2", "=HYPERLINK(\"http://evil.test\",\"click\")", author="b"),
        make_post("3", "+cmd", author="c", published_at=None, verified=False,
                  timestamp_note="snippet date only"),
        make_post("4", "-2+3", author="@d", author_handle="@d"),
        make_post("5", "ad text", source="linkedin_ads", kind="ad",
                  ad_meta=AdMeta(run_start="2026-11-01", run_end="2026-11-05",
                                 advertiser_id="adv1", landing_url="https://land.test")),
    ]
    write_jsonl(run / "posts.jsonl", posts)
    write_json(run / "narratives.json", [Narrative(
        id="n1", label="Label", claim_summary="Summary", member_post_ids=["x:1", "x:2"],
        example_post_ids=["x:1"], volume_by_day={"2026-11-01": 2}, volume_by_platform={"x": 2},
        keywords=["a", "b"], unverified_member_count=1)])
    write_json(run / "clusters.json", [Cluster(
        id="c1", member_author_ids=["a1", "b"], member_handles=["@a1", "@b"], sources=["x"],
        signals=[Signal(type="near_duplicate_text", score=0.9, evidence_post_ids=["x:1", "x:2"],
                        detail="same text"),
                 Signal(type="near_simultaneous_posting", score=0.8,
                        evidence_post_ids=["x:1", "x:2"])],
        supporting_post_ids=["x:1", "x:2"], benign_explanations=["news wire"])])
    write_json(run / "claims.json", [Claim(
        id="cl1", text="Claim text", related_post_ids=["x:1"],
        stances={"assert": 1, "report": 0, "question": 0, "debunk": 1},
        stance_post_ids={"assert": ["x:1"], "report": [], "question": [], "debunk": ["x:2"]},
        evidence=[Evidence(url="https://fc.test/1", source="google_factcheck",
                           supports="contradicts", publisher="FC", note="rated false")],
        verdict="contradicted", confidence=0.7, rationale="because", method="heuristic")])
    write_json(run / "source_status.json", [
        SourceStatus(source="x", query="topic", state=SourceState.ok, count=4),
        SourceStatus(source="telegram", query="topic",
                     state=SourceState.skipped_no_credentials,
                     reason="missing credential(s): TELEGRAM_API_ID"),
    ])
    return run


def test_posts_csv_columns_and_rows(tmp_path):
    run = _make_run(tmp_path)
    write_exports(run)
    header, rows, raw = _read_csv(run / "posts.csv")
    assert raw.startswith(b"\xef\xbb\xbf"), "posts.csv must start with a UTF-8 BOM for Excel"
    assert header == POST_COLUMNS
    assert len(rows) == 5
    r1 = rows[0]
    assert r1["post_id"] == "x:1"
    assert r1["views"] == "10" and r1["quotes"] == "1" and r1["shares"] == ""
    assert r1["media_urls"] == "https://img.test/a.jpg | https://v.test/b.mp4"
    assert r1["relations"] == "quote:x:0 | link:https://news.test/a"
    assert r1["timestamp_verified"] == "true"
    assert r1["raw_ref"] == "raw/x/page1.json"
    assert r1["published_at"] and r1["collected_at"] and r1["published_at"] != r1["collected_at"]
    r3 = rows[2]
    assert r3["published_at"] == "" and r3["timestamp_verified"] == "false"
    assert r3["timestamp_note"] == "snippet date only"
    r5 = rows[4]
    assert r5["kind"] == "ad" and r5["ad_advertiser_id"] == "adv1"
    assert r5["ad_landing_url"] == "https://land.test" and r5["ad_run_start"] == "2026-11-01"


def test_unicode_text_preserved(tmp_path):
    run = _make_run(tmp_path)
    write_exports(run)
    _, rows, _ = _read_csv(run / "posts.csv")
    assert rows[0]["text"] == "Привет мир — مرحبا بالعالم 你好 🌍"
    assert rows[0]["lang"] == "ru"


def test_csv_injection_guard(tmp_path):
    run = _make_run(tmp_path)
    write_exports(run)
    _, rows, _ = _read_csv(run / "posts.csv")
    assert rows[1]["text"].startswith("'=HYPERLINK")
    assert rows[2]["text"] == "'+cmd"
    assert rows[3]["text"] == "'-2+3"
    assert rows[3]["author_handle"] == "'@d"
    # Benign text is untouched
    assert rows[0]["text"].startswith("Привет")


def test_safe_cell_unit():
    assert safe_cell(None) == ""
    assert safe_cell(True) == "true"
    assert safe_cell(-1.5) == "-1.5"  # numbers stay numeric
    assert safe_cell("@x") == "'@x"
    assert safe_cell("\t=1") == "'\t=1"
    assert safe_cell("a=b") == "a=b"
    assert safe_cell({"b": 1, "a": "é"}) == json.dumps({"a": "é", "b": 1}, ensure_ascii=False)


def test_analysis_csvs(tmp_path):
    run = _make_run(tmp_path)
    write_exports(run)
    h, rows, _ = _read_csv(run / "narratives.csv")
    assert h == NARRATIVE_COLUMNS
    assert rows[0]["member_post_ids"] == "x:1 | x:2" and rows[0]["size"] == "2"
    assert rows[0]["volume_by_day"] == "2026-11-01:2"
    assert rows[0]["unverified_member_count"] == "1"

    h, rows, _ = _read_csv(run / "clusters.csv")
    assert h == CLUSTER_COLUMNS
    assert rows[0]["signal_types"] == "near_duplicate_text | near_simultaneous_posting"
    assert rows[0]["benign_explanations"] == "news wire"
    assert "posts=x:1,x:2" in rows[0]["signals"]

    h, rows, _ = _read_csv(run / "claims.csv")
    assert h == CLAIM_COLUMNS
    c = rows[0]
    assert c["verdict"] == "contradicted" and c["stance_debunk"] == "1"
    assert c["assert_post_ids"] == "x:1" and c["debunk_post_ids"] == "x:2"
    assert c["evidence_urls"] == "https://fc.test/1"

    h, rows, _ = _read_csv(run / "source_status.csv")
    assert h == SOURCE_STATUS_COLUMNS
    assert [r["state"] for r in rows] == ["ok", "skipped_no_credentials"]
    assert rows[1]["reason"].startswith("missing credential")


def test_empty_run_still_writes_headers(tmp_path):
    run = tmp_path / "empty"
    run.mkdir()
    (run / "posts.jsonl").write_text("", encoding="utf-8")
    write_exports(run)
    for name in ("posts.csv", "narratives.csv", "clusters.csv", "claims.csv", "source_status.csv"):
        header, rows, _ = _read_csv(run / name)
        assert header and rows == []


def test_rank_relevance_summary_and_bottom_line_columns(tmp_path):
    run = _make_run(tmp_path)
    narrs = [Narrative(id=f"n{i}", label=f"L{i}", claim_summary="s", member_post_ids=["x:1"],
                       example_post_ids=["x:1"]) for i in (1, 2, 3)]
    narrs[1].relevance = "threat"
    narrs[1].relevance_reason = "card data for sale"
    narrs[1].severity = "critical"
    narrs[1].severity_reason = "card numbers offered for sale"
    write_json(run / "narratives.json", narrs)
    write_json(run / "brief.json", {"narrative_ranks": [
        {"narrative_id": "n2", "rank": 1, "score": 4.5, "main": True,
         "tags": ["threat", "surging"], "reasons": ["threat-relevant", "12 posts"]},
        {"narrative_id": "n1", "rank": 2, "score": 1.0, "main": False, "tags": [],
         "reasons": ["1 post"]}]})
    cl = Cluster(id="c1", member_author_ids=["a"], signals=[], supporting_post_ids=["x:1"],
                 summary="3 accounts posted the same text.")
    write_json(run / "clusters.json", [cl])
    claim = Claim(id="cl1", text="t", related_post_ids=["x:1"], verdict="disputed",
                  bottom_line="Sources disagree.", method_notes="[Method: heuristic]")
    write_json(run / "claims.json", [claim])
    write_exports(run)

    _, rows, _ = _read_csv(run / "narratives.csv")
    assert [r["id"] for r in rows] == ["n2", "n1", "n3"]  # ranked first, unranked last
    assert rows[0]["rank"] == "1" and rows[0]["main"] == "true"
    assert rows[0]["rank_score"] == "4.5" and rows[0]["rank_tags"] == "threat | surging"
    assert rows[0]["relevance"] == "threat"
    assert rows[0]["relevance_reason"] == "card data for sale"
    assert rows[0]["severity"] == "critical"
    assert rows[0]["severity_reason"] == "card numbers offered for sale"
    assert rows[1]["main"] == "false" and rows[1]["relevance"] == ""
    assert rows[1]["severity"] == "" and rows[1]["severity_reason"] == ""
    assert rows[2]["rank"] == "" and rows[2]["main"] == ""
    _, rows, _ = _read_csv(run / "clusters.csv")
    assert rows[0]["summary"] == "3 accounts posted the same text."
    _, rows, _ = _read_csv(run / "claims.csv")
    assert rows[0]["verdict"] == "disputed" and rows[0]["bottom_line"] == "Sources disagree."
    assert rows[0]["method_notes"] == "[Method: heuristic]"


def test_narratives_csv_without_brief_keeps_file_order(tmp_path):
    run = _make_run(tmp_path)
    write_exports(run)
    _, rows, _ = _read_csv(run / "narratives.csv")
    assert rows[0]["id"] == "n1" and rows[0]["rank"] == ""
