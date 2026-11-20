"""Fill the 2026-11-19 analysis fields for runs saved before they existed, IN MEMORY.

Runs made before brief.json / Claim.bottom_line / Evidence.flag (including the bundled
data/sample_run) are shown with:
  - claims passed through analyze.claims.backfill_claim (evidence flags, method notes
    split out of the rationale, a rule-built bottom line; the verdict is NOT recomputed);
  - a brief built by analyze.brief.build_brief from the saved files (rule-based, no LLM).
Nothing is written back to disk. `upgrade_bundle` records what it derived under the
bundle key "derived_on_load" so the GUI and report can say so. Narrative relevance,
severity and the origin match fields need the LLM / embeddings and stay empty until
`socmint reanalyze <run_dir>` is run. A narrative without severity (severity None) is
ranked after every rated level, in the earlier activity-score order (analyze/brief.py),
so older runs, and older brief.json files whose ranks lack `severity`, still render.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

log = logging.getLogger(__name__)


def _window(run: dict[str, Any]) -> tuple[date, date] | None:
    try:
        return (date.fromisoformat(str(run.get("start") or "")[:10]),
                date.fromisoformat(str(run.get("end") or "")[:10]))
    except ValueError:
        return None


def needs_claim_backfill(claims: list[dict[str, Any]]) -> bool:
    """True for a run saved before bottom_line/method_notes existed (all claims lack both)."""
    return bool(claims) and all(
        not (c.get("bottom_line") or c.get("method_notes")) for c in claims)


def _with_status_headlines(data: dict[str, Any]) -> dict[str, Any]:
    """Runs saved before plain-language headlines existed (socmint.status_text) get them
    computed from the stored technical reason. Stored headlines are never replaced."""
    sts = data.get("source_status") or []
    if not sts or all(isinstance(x, dict) and x.get("headline") for x in sts):
        return data
    try:
        from socmint.status_text import headline

        filled = []
        for x in sts:
            if isinstance(x, dict) and not x.get("headline"):
                h = headline(x)
                x = {**x, "headline": h.text, "headline_kind": h.kind, "action": h.action,
                     "resets_at": h.resets_at}
            filled.append(x)
        return {**data, "source_status": filled}
    except Exception as e:  # noqa: BLE001 - never block viewing a run
        log.warning("status headlines skipped: %s: %s", type(e).__name__, e)
        return data


def upgrade_bundle(data: dict[str, Any]) -> dict[str, Any]:
    """Return the load_run bundle with derived fields filled for older, complete runs."""
    run = data.get("run") or {}
    data = _with_status_headlines(data)
    if run.get("status") != "complete":
        return data  # a running or failed run: show exactly what is on disk
    out = dict(data)
    derived: list[str] = []
    claims = out.get("claims") or []
    if needs_claim_backfill(claims):
        try:
            from socmint.analyze.claims import backfill_claim
            from socmint.schema import Claim, dump

            topic, win = str(run.get("topic") or ""), _window(run)
            out["claims"] = [dump(backfill_claim(Claim.model_validate(c), topic, win))
                             for c in claims]
            derived.append("claims")
        except Exception as e:  # noqa: BLE001 - never block viewing a run
            log.warning("claim backfill skipped: %s: %s", type(e).__name__, e)
    if not out.get("brief") and out.get("narratives"):
        try:
            from socmint.analyze.brief import build_brief
            from socmint.schema import dump

            out["brief"] = dump(build_brief(
                out.get("posts") or [], out["narratives"], out.get("clusters") or [],
                out.get("origin") or [], out.get("claims") or [],
                out.get("source_status") or [], run))
            derived.append("brief")
        except Exception as e:  # noqa: BLE001
            log.warning("brief derivation skipped: %s: %s", type(e).__name__, e)
    if derived:
        out["derived_on_load"] = derived
    return out
