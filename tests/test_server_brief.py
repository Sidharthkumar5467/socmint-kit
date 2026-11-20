"""The GUI API serves brief.json (run summary) with the run and as a download."""

from __future__ import annotations

import io
import zipfile

from test_server import RUN_ID, build_run, client_for

from socmint.config import runs_dir
from socmint.runstore import write_json
from socmint.server.app import create_app


def test_run_api_includes_brief_and_serves_file():
    d = build_run(runs_dir())
    c = client_for(create_app(initial_run=d))
    assert c.get(f"/api/runs/{RUN_ID}").json()["brief"] == {}  # older run: no brief yet
    assert c.get(f"/api/runs/{RUN_ID}/files/brief.json").status_code == 404

    brief = {"narrative_ranks": [], "main_cluster_ids": [], "cluster_narratives": {},
             "findings": [{"id": "F1", "kind": "claims", "headline": "1 claim checked."}],
             "coverage": [], "verdict_counts": {"unverified": 1}}
    write_json(d / "brief.json", brief)
    assert c.get(f"/api/runs/{RUN_ID}").json()["brief"] == brief
    r = c.get(f"/api/runs/{RUN_ID}/files/brief.json")
    assert r.status_code == 200 and r.json() == brief
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/runs/{RUN_ID}/export.zip").content))
    assert f"{RUN_ID}/brief.json" in z.namelist()


# ------------------------------------------------------------- reanalyze from the GUI
def test_reanalyze_endpoint_runs_job_and_reports_progress(tmp_path):
    import threading

    from fastapi.testclient import TestClient
    from test_server import build_run

    from socmint.config import runs_dir
    from socmint.runstore import write_json
    from socmint.server.app import create_app
    from socmint.server.jobs import JobManager

    d = build_run(runs_dir(), "2026-11-10-reanalyse-me-abc123")
    write_json(d / "progress.json", {"stage": "done", "done": True, "sources": {}})
    gate, seen = threading.Event(), []

    def fake_reanalyze(run_dir, *, cfg):
        seen.append(run_dir)
        gate.wait(5)
        write_json(run_dir / "progress.json", {"stage": "done", "done": True})
        return run_dir

    jobs = JobManager(reanalyzer=fake_reanalyze)
    c = TestClient(create_app(jobs=jobs), base_url="http://127.0.0.1")
    r = c.post(f"/api/runs/{d.name}/reanalyze")
    assert r.status_code == 200 and r.json() == {"run_id": d.name}
    # while the job runs, the old "done" progress must not be reported as finished
    assert c.get(f"/api/runs/{d.name}/progress").json()["done"] is False
    # one job at a time
    assert c.post(f"/api/runs/{d.name}/reanalyze").status_code == 409
    gate.set()
    jobs.wait(5)
    assert seen == [d]
    assert c.get(f"/api/runs/{d.name}/progress").json()["done"] is True


def test_reanalyze_endpoint_refusals(tmp_path):
    from fastapi.testclient import TestClient
    from test_server import build_run

    from socmint.config import runs_dir
    from socmint.server.app import create_app
    from socmint.server.jobs import JobManager

    def boom(*a, **k):  # must never be called
        raise AssertionError("reanalyze must not start")

    c = TestClient(create_app(jobs=JobManager(reanalyzer=boom)), base_url="http://127.0.0.1")
    assert c.post("/api/runs/no-such-run/reanalyze").status_code == 404
    runs = c.get("/api/runs").json()
    sample = next((r["run_id"] for r in runs if r.get("path_kind") == "sample"
                   or r.get("kind") == "sample"), None)
    assert sample  # data/sample_run is always listed
    assert c.post(f"/api/runs/{sample}/reanalyze").status_code == 403
    d = build_run(runs_dir(), "2026-11-10-still-running-abc123", progress=True)
    import json
    meta = json.loads((d / "run.json").read_text())
    meta["status"] = "running"
    (d / "run.json").write_text(json.dumps(meta))
    assert c.post(f"/api/runs/{d.name}/reanalyze").status_code == 409
    ro = TestClient(create_app(read_only=True), base_url="http://127.0.0.1")
    assert ro.post(f"/api/runs/{d.name}/reanalyze").status_code in (403, 405)
