"""Deleting runs: runstore guards, DELETE /api/runs/{id}, and `socmint delete`."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from test_server import RUN_ID, build_run, client_for

from socmint import cli
from socmint.config import runs_dir
from socmint.runstore import RunDeleteError, delete_run, read_json, run_in_progress, write_json
from socmint.server.app import create_app
from socmint.server.jobs import Job, JobManager


# ------------------------------------------------------------------ runstore guards
def test_delete_run_removes_directory():
    d = build_run(runs_dir())
    assert delete_run(d, runs_dir()) == d.resolve()
    assert not d.exists()


def test_delete_run_refuses_outside_runs_root(tmp_path):
    other = build_run(tmp_path / "elsewhere")
    with pytest.raises(RunDeleteError):
        delete_run(other, runs_dir())
    assert other.exists()


def test_delete_run_refuses_nested_and_non_run_dirs():
    d = build_run(runs_dir())
    with pytest.raises(RunDeleteError):
        delete_run(d / "raw", runs_dir())  # not a direct child of runs/
    plain = runs_dir() / "not-a-run"
    plain.mkdir()
    with pytest.raises(RunDeleteError):
        delete_run(plain, runs_dir())  # no run.json
    with pytest.raises(RunDeleteError):
        delete_run(runs_dir(), runs_dir())  # the root itself
    assert d.exists() and plain.exists()


def test_delete_run_refuses_symlink(tmp_path):
    real = build_run(tmp_path / "real")
    link = runs_dir() / "linked-run"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(RunDeleteError):
        delete_run(link, runs_dir())
    assert real.exists()


def test_delete_run_refuses_sample_run(monkeypatch, tmp_path):
    sample = build_run(runs_dir(), "sample-run")
    monkeypatch.setattr("socmint.config.sample_run_dir", lambda: sample)
    with pytest.raises(RunDeleteError, match="sample"):
        delete_run(sample, runs_dir())
    assert sample.exists()


def test_run_in_progress():
    d = build_run(runs_dir())
    assert not run_in_progress(d)  # status complete
    meta = read_json(d / "run.json")
    write_json(d / "run.json", {**meta, "status": "running"})
    assert run_in_progress(d)  # running + just written
    old = time.time() - 3600
    import os

    os.utime(d / "run.json", (old, old))
    assert not run_in_progress(d)  # running but stale -> interrupted, deletable


# ------------------------------------------------------------------ server
def test_api_delete_run():
    d = build_run(runs_dir())
    c = client_for(create_app())
    assert c.get(f"/api/runs/{RUN_ID}").json()["path_kind"] == "run"
    r = c.delete(f"/api/runs/{RUN_ID}")
    assert r.status_code == 200 and r.json() == {"deleted": RUN_ID}
    assert not d.exists()
    assert all(x["run_id"] != RUN_ID for x in c.get("/api/runs").json())
    assert c.delete(f"/api/runs/{RUN_ID}").status_code == 404


def test_api_delete_blocked_in_read_only():
    d = build_run(runs_dir())
    c = client_for(create_app(initial_run=d, read_only=True))
    assert c.delete(f"/api/runs/{RUN_ID}").status_code == 403
    assert d.exists()


def test_api_delete_refuses_sample(monkeypatch, tmp_path):
    sample = build_run(tmp_path / "data", "2026-11-01-sample-000000")
    monkeypatch.setattr("socmint.server.app.sample_run_dir", lambda: sample)
    monkeypatch.setattr("socmint.config.sample_run_dir", lambda: sample)
    c = client_for(create_app())
    r = c.delete("/api/runs/2026-11-01-sample-000000")
    assert r.status_code == 403 and "sample" in r.json()["detail"]
    assert sample.exists()


def test_api_delete_refuses_run_outside_runs_dir(tmp_path):
    other = build_run(tmp_path / "elsewhere")
    c = client_for(create_app(initial_run=other))
    assert c.delete(f"/api/runs/{RUN_ID}").status_code == 403
    assert other.exists()


def test_api_delete_refuses_active_job():
    d = build_run(runs_dir())
    jobs = JobManager()
    jobs.current = Job(run_id=RUN_ID, run_dir=d)  # done=False -> active
    c = client_for(create_app(jobs=jobs))
    r = c.delete(f"/api/runs/{RUN_ID}")
    assert r.status_code == 409
    assert d.exists()


def test_api_delete_refuses_cross_origin():
    d = build_run(runs_dir())
    c = client_for(create_app())
    r = c.delete(f"/api/runs/{RUN_ID}", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert d.exists()


# ------------------------------------------------------------------ CLI
def test_cli_delete_by_id_with_yes(capsys):
    d = build_run(runs_dir())
    assert cli.main(["delete", RUN_ID, "--yes"]) == 0
    assert not d.exists()
    assert "deleted" in capsys.readouterr().out


def test_cli_delete_without_yes_and_no_tty_refuses(monkeypatch):
    d = build_run(runs_dir())
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert cli.main(["delete", RUN_ID]) == 1
    assert d.exists()


def test_cli_delete_confirm_no_keeps_run(monkeypatch):
    d = build_run(runs_dir())
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    assert cli.main(["delete", RUN_ID]) == 1
    assert d.exists()


def test_cli_delete_unknown_run():
    assert cli.main(["delete", "no-such-run", "--yes"]) == 1


def test_cli_delete_refuses_path_outside_runs(tmp_path: Path):
    other = build_run(tmp_path / "elsewhere")
    assert cli.main(["delete", str(other), "--yes"]) == 1
    assert other.exists()


# ------------------------------------------------------------------ review regressions
def _mark_running(d: Path, pid: int | None, age: float) -> None:
    """status=running, progress.json naming `pid`, both files last written `age` s ago."""
    import os

    meta = read_json(d / "run.json")
    write_json(d / "run.json", {**meta, "status": "running"})
    write_json(d / "progress.json", {"done": False, "pid": pid})
    t = time.time() - age
    for f in ("run.json", "progress.json"):
        os.utime(d / f, (t, t))


def _spawn_sleeper():
    import subprocess
    import sys

    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def test_cli_refuses_live_writer_quiet_for_minutes():
    # finding 1: a live writer can go >3 min without writing (flood wait, model download)
    d = build_run(runs_dir())
    p = _spawn_sleeper()
    try:
        _mark_running(d, p.pid, age=300)
        assert run_in_progress(d)
        assert cli.main(["delete", RUN_ID, "--yes"]) == 1
        assert d.exists()
    finally:
        p.kill()
        p.wait()
    # writer gone, but written 5 min ago: still inside the 30 min stale window
    assert run_in_progress(d)
    _mark_running(d, p.pid, age=3600)  # dead pid + stale -> interrupted, deletable
    assert not run_in_progress(d)
    assert cli.main(["delete", RUN_ID, "--yes"]) == 0


def test_api_refuses_run_live_in_other_process_started_before_server():
    # finding 2: a `socmint run` that last wrote before the GUI started is still live
    d = build_run(runs_dir())
    p = _spawn_sleeper()
    try:
        _mark_running(d, p.pid, age=5)
        c = client_for(create_app())
        listed = {x["run_id"]: x for x in c.get("/api/runs").json()}
        assert listed[RUN_ID]["status"] == "running"  # not shown as failed/deletable
        assert c.delete(f"/api/runs/{RUN_ID}").status_code == 409
        assert d.exists()
    finally:
        p.kill()
        p.wait()


def test_api_deletes_interrupted_run_with_dead_writer():
    d = build_run(runs_dir())
    p = _spawn_sleeper()
    p.kill()
    p.wait()
    _mark_running(d, p.pid, age=3600)
    c = client_for(create_app())
    assert c.delete(f"/api/runs/{RUN_ID}").status_code == 200
    assert not d.exists()


def test_external_run_is_not_deletable_kind(tmp_path):
    # finding 3: a --run dir outside runs/ is listed as 'external' (no Delete button)
    other = build_run(tmp_path / "ext")
    c = client_for(create_app(initial_run=other))
    listed = {x["run_id"]: x for x in c.get("/api/runs").json()}
    assert listed[RUN_ID]["path_kind"] == "external"
    assert c.get(f"/api/runs/{RUN_ID}").json()["path_kind"] == "external"
    r = c.delete(f"/api/runs/{RUN_ID}")
    assert r.status_code == 403 and str(tmp_path) not in r.json()["detail"]
    assert other.exists()


def test_api_delete_already_gone_is_404_not_500(monkeypatch):
    # finding 4: the loser of a delete race gets a clean 404
    build_run(runs_dir())
    c = client_for(create_app())

    def gone(_d, _root):
        raise FileNotFoundError("posts.csv")

    monkeypatch.setattr("socmint.runstore.delete_run", gone)
    r = c.delete(f"/api/runs/{RUN_ID}")
    assert r.status_code == 404 and "already deleted" in r.json()["detail"]


def test_api_delete_partial_failure_is_explained(monkeypatch):
    build_run(runs_dir())
    c = client_for(create_app())

    def denied(_d, _root):
        raise PermissionError("raw/x.json")

    monkeypatch.setattr("socmint.runstore.delete_run", denied)
    r = c.delete(f"/api/runs/{RUN_ID}")
    assert r.status_code == 500 and "partly deleted" in r.json()["detail"]


def test_meta_drops_deleted_initial_run():
    # finding 5: after deleting the initial run, the GUI root must not route to it
    d = build_run(runs_dir())
    c = client_for(create_app(initial_run=d))
    assert c.get("/api/meta").json()["initial_run_id"] == RUN_ID
    assert c.delete(f"/api/runs/{RUN_ID}").status_code == 200
    assert c.get("/api/meta").json()["initial_run_id"] is None


def test_pid_alive_never_signals_and_handles_bad_pids():
    import os

    from socmint.runstore import pid_alive

    assert pid_alive(os.getpid())
    assert not pid_alive(0) and not pid_alive(-1)
