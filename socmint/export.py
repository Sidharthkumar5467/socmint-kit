"""Flat CSV exports (docs/SPEC.md §8).

`write_exports(run_dir)` reads the JSON files already written by the pipeline
(posts.jsonl, narratives.json, clusters.json, claims.json, source_status.json and, when
present, brief.json for the narrative rank columns) and writes one CSV per record type
next to them. JSON files remain the canonical, lossless export;
the CSVs are flat views for spreadsheets.

Evidence rules encoded here:
- Text is written exactly as collected, in its original language (UTF-8 with BOM so
  Excel opens non-Latin scripts correctly). Nothing is translated or dropped.
- `published_at` and `collected_at` stay separate columns, and `timestamp_verified` /
  `timestamp_note` travel with every row so an unverified timestamp is never mistaken for
  a platform timestamp.
- Every analysis row keeps its post IDs, so each finding remains traceable in the CSV.

CSV injection: any cell beginning with = + - @ (or tab / carriage return, which some
spreadsheet apps also treat as formula prefixes) is prefixed with a single quote. This
alters the cell text slightly for those cells; the JSON exports stay unmodified.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from socmint.runstore import FILES, read_json, read_jsonl

POST_COLUMNS = [
    "post_id", "source", "kind", "author_id", "author_handle", "author_name",
    "author_created_at", "published_at", "timestamp_verified", "timestamp_note", "in_window",
    "collected_at", "lang", "text", "url", "views", "likes", "replies", "reposts", "quotes",
    "shares", "comments", "forwards", "reactions", "media_urls", "relations", "ad_run_start",
    "ad_run_end", "ad_advertiser_id", "ad_landing_url", "query", "raw_ref",
]
METRIC_KEYS = ["views", "likes", "replies", "reposts", "quotes", "shares", "comments",
               "forwards", "reactions"]

NARRATIVE_COLUMNS = [
    "id", "rank", "main", "rank_score", "rank_tags", "rank_reasons", "severity",
    "severity_reason", "relevance", "relevance_reason", "label", "label_method",
    "claim_summary", "size", "first_seen_at", "unverified_member_count", "keywords", "volume_by_day", "volume_by_platform",
    "example_post_ids", "member_post_ids",
]
CLUSTER_COLUMNS = [
    "id", "member_count", "summary", "member_author_ids", "member_handles", "sources", "signal_types",
    "signals", "benign_explanations", "supporting_post_ids",
]
CLAIM_COLUMNS = [
    "id", "text", "narrative_id", "verdict", "confidence", "bottom_line", "method",
    "rationale", "method_notes",
    "stance_assert", "stance_report", "stance_question", "stance_debunk",
    "assert_post_ids", "report_post_ids", "question_post_ids", "debunk_post_ids",
    "evidence_urls", "evidence", "related_post_ids",
]
SOURCE_STATUS_COLUMNS = [
    "source", "query", "state", "count", "reason", "cursor", "date_filtered", "requests",
    "cost_usd", "quota_units", "started_at", "finished_at",
]

SEP = " | "
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def safe_cell(value: Any) -> str:
    """Stringify a value for CSV and neutralise spreadsheet formula injection."""
    if value is None:
        return ""
    if isinstance(value, bool):
        s = "true" if value else "false"
    elif isinstance(value, (dict, list)):
        s = json.dumps(value, ensure_ascii=False, sort_keys=True)
    elif isinstance(value, (int, float)):
        return str(value)  # numbers are never formulas; keep e.g. -0.5 numeric
    else:
        s = str(value)
    if s.startswith(_FORMULA_PREFIXES):
        s = "'" + s
    return s


def _join(items: Iterable[Any]) -> str:
    return SEP.join(str(x) for x in items if x not in (None, ""))


def _kv(d: dict | None) -> str:
    return _join(f"{k}:{v}" for k, v in (d or {}).items())


def post_row(p: dict) -> dict[str, Any]:
    metrics = p.get("metrics") or {}
    ad = p.get("ad_meta") or {}
    media_urls = [m.get("url") or m.get("thumbnail") for m in p.get("media") or []]
    relations = []
    for r in p.get("relations") or []:
        target = r.get("target_post_id") or r.get("target_url") or r.get("target_handle") or ""
        relations.append(f"{r.get('type')}:{target}")
    row: dict[str, Any] = {
        "post_id": p.get("post_id"),
        "source": p.get("source"),
        "kind": p.get("kind"),
        "author_id": p.get("author_id"),
        "author_handle": p.get("author_handle"),
        "author_name": p.get("author_name"),
        "author_created_at": p.get("author_created_at"),
        "published_at": p.get("published_at"),
        "timestamp_verified": bool(p.get("timestamp_verified")),
        "timestamp_note": p.get("timestamp_note"),
        "in_window": p.get("in_window", True),
        "collected_at": p.get("collected_at"),
        "lang": p.get("lang"),
        "text": p.get("text") or "",
        "url": p.get("url"),
        "media_urls": _join(media_urls),
        "relations": _join(relations),
        "ad_run_start": ad.get("run_start"),
        "ad_run_end": ad.get("run_end"),
        "ad_advertiser_id": ad.get("advertiser_id"),
        "ad_landing_url": ad.get("landing_url"),
        "query": p.get("query"),
        "raw_ref": p.get("raw_ref"),
    }
    for k in METRIC_KEYS:
        row[k] = metrics.get(k)
    return row


def narrative_row(n: dict, rank: dict | None = None) -> dict[str, Any]:
    """`rank`: this narrative's entry from brief.json narrative_ranks (None if absent)."""
    members = n.get("member_post_ids") or []
    rank = rank or {}
    return {
        "id": n.get("id"),
        "rank": rank.get("rank"),
        "main": rank.get("main") if rank else None,
        "rank_score": rank.get("score"),
        "rank_tags": _join(rank.get("tags") or []),
        "rank_reasons": _join(rank.get("reasons") or []),
        "severity": n.get("severity"),
        "severity_reason": n.get("severity_reason") or "",
        "relevance": n.get("relevance"),
        "relevance_reason": n.get("relevance_reason") or "",
        "label": n.get("label"),
        "label_method": n.get("label_method"),
        "claim_summary": n.get("claim_summary"),
        "size": len(members),
        "first_seen_at": n.get("first_seen_at"),
        "unverified_member_count": n.get("unverified_member_count", 0),
        "keywords": _join(n.get("keywords") or []),
        "volume_by_day": _kv(n.get("volume_by_day")),
        "volume_by_platform": _kv(n.get("volume_by_platform")),
        "example_post_ids": _join(n.get("example_post_ids") or []),
        "member_post_ids": _join(members),
    }


def cluster_row(c: dict) -> dict[str, Any]:
    signals = c.get("signals") or []
    sig_text = []
    for s in signals:
        ev = ",".join(s.get("evidence_post_ids") or [])
        sig_text.append(f"{s.get('type')}(score={s.get('score')}; posts={ev}; {s.get('detail') or ''})")
    return {
        "id": c.get("id"),
        "member_count": len(c.get("member_author_ids") or []),
        "summary": c.get("summary") or "",
        "member_author_ids": _join(c.get("member_author_ids") or []),
        "member_handles": _join(c.get("member_handles") or []),
        "sources": _join(c.get("sources") or []),
        "signal_types": _join(sorted({s.get("type") for s in signals if s.get("type")})),
        "signals": _join(sig_text),
        "benign_explanations": _join(c.get("benign_explanations") or []),
        "supporting_post_ids": _join(c.get("supporting_post_ids") or []),
    }


def claim_row(c: dict) -> dict[str, Any]:
    stances = c.get("stances") or {}
    spids = c.get("stance_post_ids") or {}
    evidence = c.get("evidence") or []
    ev_text = [
        f"{e.get('supports')}: {e.get('url')} ({e.get('publisher') or e.get('source') or ''})"
        + (f" - {e.get('note')}" if e.get("note") else "")
        for e in evidence
    ]
    row: dict[str, Any] = {
        "id": c.get("id"),
        "text": c.get("text"),
        "narrative_id": c.get("narrative_id"),
        "verdict": c.get("verdict"),
        "confidence": c.get("confidence"),
        "method": c.get("method"),
        "rationale": c.get("rationale"),
        "bottom_line": c.get("bottom_line") or "",
        "method_notes": c.get("method_notes") or "",
        "evidence_urls": _join(e.get("url") for e in evidence),
        "evidence": _join(ev_text),
        "related_post_ids": _join(c.get("related_post_ids") or []),
    }
    for s in ("assert", "report", "question", "debunk"):
        row[f"stance_{s}"] = stances.get(s, 0)
        row[f"{s}_post_ids"] = _join(spids.get(s) or [])
    return row


def source_status_row(s: dict) -> dict[str, Any]:
    return {k: s.get(k) for k in SOURCE_STATUS_COLUMNS}


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> int:
    """Write rows as UTF-8-with-BOM CSV (Excel friendly). Returns the row count."""
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(columns)
        for row in rows:
            w.writerow([safe_cell(row.get(c)) for c in columns])
            n += 1
    tmp.replace(path)
    return n


def _list(value: Any) -> list[dict]:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def write_exports(run_dir: Path) -> None:
    """Write posts.csv, narratives.csv, clusters.csv, claims.csv, source_status.csv."""
    run_dir = Path(run_dir)
    posts = read_jsonl(run_dir / FILES["posts"])
    write_csv(run_dir / FILES["posts_csv"], POST_COLUMNS, (post_row(p) for p in posts))
    brief = read_json(run_dir / FILES["brief"], {}) or {}
    ranks = {r.get("narrative_id"): r for r in _list(
        brief.get("narrative_ranks") if isinstance(brief, dict) else [])}
    narratives = _list(read_json(run_dir / FILES["narratives"], []))
    # ranked order when brief.json exists (rank 1 first), else the file order
    order = {nid: r.get("rank") or 0 for nid, r in ranks.items()}
    narratives.sort(key=lambda n: order.get(n.get("id"), float("inf")))
    write_csv(run_dir / "narratives.csv", NARRATIVE_COLUMNS,
              (narrative_row(n, ranks.get(n.get("id"))) for n in narratives))
    write_csv(run_dir / "clusters.csv", CLUSTER_COLUMNS,
              (cluster_row(c) for c in _list(read_json(run_dir / FILES["clusters"], []))))
    write_csv(run_dir / "claims.csv", CLAIM_COLUMNS,
              (claim_row(c) for c in _list(read_json(run_dir / FILES["claims"], []))))
    write_csv(run_dir / "source_status.csv", SOURCE_STATUS_COLUMNS,
              (source_status_row(s)
               for s in _list(read_json(run_dir / FILES["source_status"], []))))
