"""socmint.legacy: older runs get the new analysis fields in memory, never on disk."""

from __future__ import annotations

import hashlib
from pathlib import Path

from socmint.legacy import needs_claim_backfill, upgrade_bundle
from socmint.runstore import load_run

# the sample run as it was before the 2026-11-19 re-analysis (older-run format)
SAMPLE = Path(__file__).resolve().parent / "fixtures" / "legacy_run"


def _digest(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in d.iterdir() if p.is_file()):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()


def test_upgrade_sample_run_in_memory_only():
    before = _digest(SAMPLE)
    data = upgrade_bundle(load_run(SAMPLE))
    assert _digest(SAMPLE) == before  # nothing written back
    assert data["derived_on_load"] == ["claims", "brief"]
    claims = {c["id"]: c for c in data["claims"]}
    # CL3's only evidence (an NFL schedule article) is a search miss
    assert [e["flag"] for e in claims["CL3"]["evidence"]] == ["unrelated"]
    assert all(c["bottom_line"] for c in claims.values())
    assert all("[Method:" not in c["rationale"] for c in claims.values())
    # verdicts are kept on load unless they fail the code checks every new verdict
    # passes: CL9 was "supported" by items that never name the banks. CL12 keeps
    # "supported": its CBS and Forbes article URL slugs name Bank of America and Wells
    # Fargo (the specifics check reads the URL slug since the 2026-11-19 review).
    orig = {c["id"]: c["verdict"] for c in load_run(SAMPLE)["claims"]}
    got = {k: v["verdict"] for k, v in claims.items()}
    # CL1, CL22 and CL23 were "disputed" with supporting and context items only; the
    # disputed rule (2026-11-20) needs a contradicting item, a mixed fact-check rating or
    # a stated conflict between supporting items, so they load as "unverified".
    assert {k for k in orig if orig[k] != got[k]} == {"CL1", "CL9", "CL22", "CL23"}
    for cid in ("CL1", "CL22", "CL23"):
        assert orig[cid] == "disputed" and got[cid] == "unverified"
        assert claims[cid]["confidence"] <= 0.3
    assert got["CL12"] == "supported"
    for cid in ("CL9",):
        assert got[cid] == "unverified" and claims[cid]["confidence"] <= 0.3
        assert "Backfill: verdict supported lowered to unverified because no supporting " \
               "item names" in claims[cid]["method_notes"]
        assert "Wells Fargo" in claims[cid]["bottom_line"]
    brief = data["brief"]
    assert brief["findings"] and brief["narrative_ranks"]
    nids = {n["id"] for n in data["narratives"]}
    pids = {p["post_id"] for p in data["posts"]}
    for f in brief["findings"]:
        assert set(f["narrative_ids"]) <= nids and set(f["post_ids"]) <= pids
    # the sample run predates severity: no level, no severity tag, earlier order kept
    assert all(r.get("severity") is None for r in brief["narrative_ranks"])
    assert not any(t.startswith("severity:") for r in brief["narrative_ranks"]
                   for t in r["tags"])
    assert brief["findings"][0]["headline"].startswith("Top-ranked narrative ")


def test_upgrade_leaves_new_and_running_runs_alone():
    new = {"run": {"status": "complete"}, "claims": [{"id": "CL1", "bottom_line": "x"}],
           "narratives": [], "brief": {"findings": []}}
    assert upgrade_bundle(new) == new
    running = {"run": {"status": "running"}, "claims": [{"id": "CL1"}], "narratives": [{}]}
    assert upgrade_bundle(running) is running


def test_needs_claim_backfill():
    assert not needs_claim_backfill([])
    assert needs_claim_backfill([{"rationale": "r"}, {"bottom_line": ""}])
    assert not needs_claim_backfill([{"rationale": "r"}, {"method_notes": "Method: x"}])


def test_upgrade_survives_bad_data():
    bad = {"run": {"status": "complete", "start": "x"}, "claims": [{"nonsense": 1}],
           "narratives": [{"also": "bad"}], "brief": {}}
    out = upgrade_bundle(bad)  # logs and returns what it could; never raises
    assert out["claims"] == bad["claims"] and out["brief"] == {}
    assert "derived_on_load" not in out
