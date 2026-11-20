"""Server API contract tests (socmint/server/app.py). No network: TestClient only."""

from __future__ import annotations

import io
import json
import sys
import threading
import types
import zipfile
from pathlib import Path

import pytest
from conftest import make_post  # tests/ is on sys.path (no package __init__)
from fastapi.testclient import TestClient

from socmint import pipeline
from socmint.config import load_config, runs_dir
from socmint.runstore import write_json, write_jsonl
from socmint.schema import RunMeta, SourceState, SourceStatus
from socmint.server.app import create_app
from socmint.server.jobs import JobManager

RUN_ID = "2026-11-10-test-topic-abc123"
SECRET = "sk-test-SECRETVALUE-1234"


def build_run(root: Path, run_id: str = RUN_ID, *, progress: bool = False) -> Path:
    d = root / run_id
    (d / "raw" / "x").mkdir(parents=True)
    meta = RunMeta(run_id=run_id, topic="test topic", start="2026-11-01", end="2026-11-10",
                   sources=["x", "youtube"], created_at="2026-11-10T00:00:00Z",
                   finished_at="2026-11-10T00:01:00Z", status="complete",
                   counts={"posts": 1})
    write_json(d / "run.json", meta)
    write_jsonl(d / "posts.jsonl", [make_post("1", "hello world")])
    write_json(d / "source_status.json", [
        SourceStatus(source="x", query="test topic", state=SourceState.ok, count=1),
        SourceStatus(source="youtube", query="test topic",
                     state=SourceState.skipped_no_credentials, reason="missing YOUTUBE_API_KEY")])
    for name in ("narratives.json", "clusters.json", "origin.json", "claims.json"):
        write_json(d / name, [])
    (d / "posts.csv").write_text("post_id,text\nx:1,hello world\n", encoding="utf-8")
    (d / "report.html").write_text("<!doctype html><title>r</title>", encoding="utf-8")
    (d / "raw" / "x" / "page1.json").write_text('{"tweets": []}', encoding="utf-8")
    (d / "raw" / "x" / "page.html").write_text("<script>alert(1)</script>", encoding="utf-8")
    (d / "secret.txt").write_text("outside raw", encoding="utf-8")  # must never be served
    if progress:
        write_json(d / "progress.json", {"stage": "collect", "done": False, "sources": {}})
    return d


def client_for(app) -> TestClient:
    # Host must be 127.0.0.1/localhost: the app rejects other Host headers (DNS rebinding).
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def run_dir() -> Path:
    return build_run(runs_dir())


@pytest.fixture
def client(run_dir: Path) -> TestClient:
    return client_for(create_app(initial_run=run_dir))


@pytest.fixture
def ro_client(run_dir: Path) -> TestClient:
    return client_for(create_app(initial_run=run_dir, read_only=True))


# ------------------------------------------------------------------------- meta
def test_meta_contract(client: TestClient) -> None:
    r = client.get("/api/meta")
    assert r.status_code == 200
    m = r.json()
    assert m["read_only"] is False
    assert m["initial_run_id"] == RUN_ID
    assert m["max_window_days"] == 30
    names = {s["name"]: s for s in m["sources"]}
    assert {"x", "telegram", "youtube", "linkedin"} <= set(names)
    assert "linkedin_ads" not in names  # removed from the active source set
    for s in m["sources"]:
        assert {"name", "kind", "needs_credentials", "credential_keys", "ready",
                "reason"} <= set(s)
    yt = names["youtube"]
    if yt["kind"] is not None:  # collector importable
        assert yt["ready"] is False and "YOUTUBE_API_KEY" in yt["reason"]


def test_meta_read_only_flag(ro_client: TestClient) -> None:
    assert ro_client.get("/api/meta").json()["read_only"] is True


# --------------------------------------------------------------------- settings
def test_settings_roundtrip_masks_secrets(client: TestClient) -> None:
    r = client.post("/api/settings", json={"TWITTERAPI_IO_KEY": SECRET, "MAX_WINDOW_DAYS": "20"})
    assert r.status_code == 200
    assert SECRET not in r.text  # never echoed in clear
    body = r.json()
    assert body["config"]["TWITTERAPI_IO_KEY"].endswith("1234")
    assert body["config"]["MAX_WINDOW_DAYS"] == "20"
    keys = {k["key"]: k for k in body["keys"]}
    assert keys["TWITTERAPI_IO_KEY"] == {"key": "TWITTERAPI_IO_KEY", "secret": True, "set": True}
    assert keys["MAX_WINDOW_DAYS"]["secret"] is False
    assert load_config()["TWITTERAPI_IO_KEY"] == SECRET

    # Posting the masked placeholder back must not overwrite the stored secret.
    client.post("/api/settings", json={"TWITTERAPI_IO_KEY": body["config"]["TWITTERAPI_IO_KEY"]})
    assert load_config()["TWITTERAPI_IO_KEY"] == SECRET
    # Empty string clears.
    r = client.post("/api/settings", json={"TWITTERAPI_IO_KEY": ""})
    assert r.json()["config"]["TWITTERAPI_IO_KEY"] == ""
    assert client.get("/api/settings").json()["config"]["MAX_WINDOW_DAYS"] == "20"


def test_settings_unknown_key_rejected(client: TestClient) -> None:
    assert client.post("/api/settings", json={"NOT_A_KEY": "x"}).status_code == 400


def test_read_only_blocks_writes(ro_client: TestClient) -> None:
    assert ro_client.get("/api/settings").status_code == 403
    assert ro_client.post("/api/settings", json={"YOUTUBE_API_KEY": "x" * 10}).status_code == 403
    assert ro_client.get("/api/telegram/status").status_code == 403
    assert ro_client.post("/api/telegram/logout").status_code == 403
    r = ro_client.post("/api/runs", json={"topic": "t", "start": "2026-11-01",
                                          "end": "2026-11-02", "sources": ["youtube"]})
    assert r.status_code == 403
    # reading results still works
    assert ro_client.get(f"/api/runs/{RUN_ID}").status_code == 200


def test_cross_origin_write_refused(client: TestClient) -> None:
    r = client.post("/api/settings", json={"MAX_WINDOW_DAYS": "5"},
                    headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    r = client.post("/api/settings", json={"MAX_WINDOW_DAYS": "5"},
                    headers={"Origin": "null"})
    assert r.status_code == 403
    # Another app on a different localhost port is a foreign origin.
    r = client.post("/api/settings", json={"MAX_WINDOW_DAYS": "5"},
                    headers={"Origin": "http://localhost:3000"})
    assert r.status_code == 403
    r = client.post("/api/settings", json={"MAX_WINDOW_DAYS": "5"},
                    headers={"Origin": "http://127.0.0.1:3000"})
    assert r.status_code == 403
    assert load_config()["MAX_WINDOW_DAYS"] == "30"
    # Same scheme + host + port as the server (TestClient Host is "127.0.0.1").
    r = client.post("/api/settings", json={"MAX_WINDOW_DAYS": "5"},
                    headers={"Origin": "http://127.0.0.1"})
    assert r.status_code == 200


def test_same_origin_with_port_accepted(run_dir: Path) -> None:
    c = TestClient(create_app(initial_run=run_dir), base_url="http://127.0.0.1:8765")
    r = c.post("/api/settings", json={"MAX_WINDOW_DAYS": "7"},
               headers={"Origin": "http://127.0.0.1:8765"})
    assert r.status_code == 200
    r = c.post("/api/settings", json={"MAX_WINDOW_DAYS": "7"},
               headers={"Origin": "http://localhost:8765"})  # host differs from Host header
    assert r.status_code == 403


@pytest.mark.parametrize("value", ["31", "365", "0", "-5", "thirty", "7.5"])
def test_settings_max_window_validated(client: TestClient, value: str) -> None:
    r = client.post("/api/settings", json={"MAX_WINDOW_DAYS": value})
    assert r.status_code == 400 and "MAX_WINDOW_DAYS" in r.json()["detail"]
    assert load_config()["MAX_WINDOW_DAYS"] == "30"  # nothing saved
    assert client.get("/api/meta").status_code == 200


def test_settings_other_numeric_keys_validated(client: TestClient) -> None:
    assert client.post("/api/settings", json={"RUN_TIME_BUDGET_SECONDS": "abc"}).status_code == 400
    assert client.post("/api/settings", json={"TELEGRAM_HAS_PREMIUM": "maybe"}).status_code == 400
    assert client.post("/api/settings", json={"RUN_TIME_BUDGET_SECONDS": "900"}).status_code == 200
    # empty string still clears back to the default
    assert client.post("/api/settings", json={"MAX_WINDOW_DAYS": ""}).status_code == 200


def test_invalid_stored_window_falls_back_to_cap(client: TestClient, monkeypatch) -> None:
    """A hand-edited config.json/.env with a bad or oversized value never 500s or lifts 30."""
    import socmint.server.app as app_mod

    for bad, expect in (("thirty", 30), ("365", 30), ("10", 10)):
        monkeypatch.setattr(app_mod, "load_config", lambda b=bad: {**load_config(),
                                                                   "MAX_WINDOW_DAYS": b})
        assert client.get("/api/meta").json()["max_window_days"] == expect
    monkeypatch.setattr(app_mod, "load_config", lambda: {**load_config(),
                                                         "MAX_WINDOW_DAYS": "365"})
    r = client.post("/api/runs", json={"topic": "t", "start": "2025-01-01",
                                       "end": "2025-12-01", "sources": ["youtube"]})
    assert r.status_code == 400 and "maximum is 30" in r.json()["detail"]


def test_foreign_host_header_refused(run_dir: Path) -> None:
    c = TestClient(create_app(initial_run=run_dir), base_url="http://evil.example")
    assert c.get("/api/meta").status_code == 400


# --------------------------------------------------------------------- telegram
def test_telegram_unavailable_module(client: TestClient, monkeypatch) -> None:
    import socmint.collect as collect

    monkeypatch.setitem(sys.modules, "socmint.collect.telegram_auth", None)  # import fails
    monkeypatch.delattr(collect, "telegram_auth", raising=False)
    r = client.get("/api/telegram/status")
    assert r.status_code == 200
    assert r.json()["configured"] is False and r.json()["reason"]


def test_telegram_wrappers_call_module(client: TestClient, monkeypatch) -> None:
    calls: dict = {}
    fake = types.ModuleType("socmint.collect.telegram_auth")

    def status(cfg):
        calls["status"] = "TELEGRAM_API_ID" in cfg
        return {"configured": True, "authorized": False}

    async def send_code(cfg, phone):
        calls["send_code"] = phone
        return {"sent": True}

    def sign_in(cfg, code, password=None):
        calls["sign_in"] = (code, password)
        raise ValueError(f"bad code; token={SECRET}")

    def logout(cfg):
        return True

    fake.status, fake.send_code, fake.sign_in, fake.logout = status, send_code, sign_in, logout
    _install_fake_auth(monkeypatch, fake)

    assert client.get("/api/telegram/status").json() == {"configured": True,
                                                         "authorized": False}
    assert client.post("/api/telegram/send_code", json={"phone": "+100"}).json() == {"sent": True}
    r = client.post("/api/telegram/sign_in", json={"code": "12345", "password": "pw"})
    assert r.status_code == 400 and SECRET not in r.text  # error redacted
    assert client.post("/api/telegram/logout").json() == {"ok": True}
    assert calls == {"status": True, "send_code": "+100", "sign_in": ("12345", "pw")}


def _install_fake_auth(monkeypatch, fake) -> None:
    import socmint.collect as collect

    monkeypatch.setitem(sys.modules, "socmint.collect.telegram_auth", fake)
    monkeypatch.setattr(collect, "telegram_auth", fake, raising=False)


# ------------------------------------------------------------------------- runs
def test_list_runs(client: TestClient) -> None:
    rows = client.get("/api/runs").json()
    mine = [r for r in rows if r["run_id"] == RUN_ID]
    assert len(mine) == 1
    r = mine[0]
    assert r["path_kind"] == "run" and r["topic"] == "test topic" and r["status"] == "complete"
    assert set(r) == {"run_id", "topic", "start", "end", "status", "error", "created_at",
                      "counts", "sources", "path_kind"}
    assert "path" not in r


def test_get_run_and_unknown(client: TestClient) -> None:
    body = client.get(f"/api/runs/{RUN_ID}").json()
    assert body["run_id"] == RUN_ID
    assert body["run"]["topic"] == "test topic"
    assert body["posts"][0]["post_id"] == "x:1"
    assert {s["state"] for s in body["source_status"]} == {"ok", "skipped_no_credentials"}
    assert client.get("/api/runs/nope").status_code == 404


def test_progress(client: TestClient, run_dir: Path) -> None:
    assert client.get(f"/api/runs/{RUN_ID}/progress").json() == {"done": True}
    # Written after this server started (e.g. by a `socmint run` process): may be live.
    write_json(run_dir / "progress.json", {"stage": "verify", "done": False})
    body = client.get(f"/api/runs/{RUN_ID}/progress").json()
    assert body["stage"] == "verify" and body["done"] is False


def _set_running(run_dir: Path) -> None:
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    meta["status"] = "running"
    write_json(run_dir / "run.json", meta)
    write_json(run_dir / "progress.json", {"stage": "collect", "done": False, "sources": {}})


def test_run_interrupted_before_server_start_reported(run_dir: Path) -> None:
    """Server stopped mid-run, then restarted: the run must not spin as in progress."""
    _set_running(run_dir)
    old = 1_000_000_000  # files last written long before this server process started
    import os

    for name in ("run.json", "progress.json"):
        os.utime(run_dir / name, (old, old))
    c = client_for(create_app(initial_run=run_dir))
    prog = c.get(f"/api/runs/{RUN_ID}/progress").json()
    assert prog["done"] is True and prog["interrupted"] is True
    assert "interrupted" in prog["error"]
    row = next(r for r in c.get("/api/runs").json() if r["run_id"] == RUN_ID)
    assert row["status"] == "failed" and "interrupted" in row["error"]
    assert c.get(f"/api/runs/{RUN_ID}").json()["run"]["status"] == "failed"


def test_dead_job_error_exposed_in_progress(run_dir: Path) -> None:
    """pipeline.run raised after announcing run_id: the job error reaches the GUI."""
    rid = "2026-11-12-crash-after-start-aaaaaa"

    def runner(topic, start, end, sources, *, limit, progress, out_root, cfg, options=None, stop=None):
        d = out_root / rid
        d.mkdir(parents=True)
        write_json(d / "run.json", RunMeta(run_id=rid, topic=topic, start="2026-11-01",
                                           end="2026-11-02", sources=sources,
                                           created_at="2026-11-12T00:00:00Z"))
        write_json(d / "progress.json", {"stage": "collect", "done": False})
        progress({"run_id": rid})
        raise OSError("disk full")

    jobs = JobManager(runner=runner)
    c = client_for(create_app(initial_run=run_dir, jobs=jobs))
    r = c.post("/api/runs", json={"topic": "crash", "start": "2026-11-01",
                                  "end": "2026-11-02", "sources": ["youtube"]})
    assert r.status_code == 200 and r.json()["run_id"] == rid
    jobs.wait(5)
    prog = c.get(f"/api/runs/{rid}/progress").json()
    assert prog["done"] is True and "disk full" in prog["error"]
    row = next(x for x in c.get("/api/runs").json() if x["run_id"] == rid)
    assert row["status"] == "failed" and "disk full" in row["error"]


def test_files_whitelist(client: TestClient) -> None:
    r = client.get(f"/api/runs/{RUN_ID}/files/posts.csv")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    assert "x:1" in r.text
    r = client.get(f"/api/runs/{RUN_ID}/files/report.html", params={"inline": "true"})
    assert r.status_code == 200 and "inline" in r.headers["content-disposition"]
    assert client.get(f"/api/runs/{RUN_ID}/files/secret.txt").status_code == 404
    assert client.get(f"/api/runs/{RUN_ID}/files/progress.json").status_code == 404
    assert client.get(f"/api/runs/{RUN_ID}/files/..%2Fsecret.txt").status_code == 404


UNICODE_RUN_ID = "2026-11-01-انتخابات-مصر-49154f"


def test_non_ascii_run_id_downloads(tmp_path: Path) -> None:
    """Any-language topics give Unicode run ids; downloads must not 500 on the header."""
    d = build_run(runs_dir(), UNICODE_RUN_ID)
    c = client_for(create_app(initial_run=d))
    r = c.get(f"/api/runs/{UNICODE_RUN_ID}/export.zip")
    assert r.status_code == 200
    cd = r.headers["content-disposition"]
    assert cd.startswith("attachment;") and "filename*=utf-8''" in cd
    assert cd.encode("latin-1")  # header is encodable
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert f"{UNICODE_RUN_ID}/posts.csv" in names
    r = c.get(f"/api/runs/{UNICODE_RUN_ID}/files/posts.csv")
    assert r.status_code == 200 and "filename*=utf-8''" in r.headers["content-disposition"]


def test_export_zip(client: TestClient) -> None:
    r = client.get(f"/api/runs/{RUN_ID}/export.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert f"{RUN_ID}/posts.csv" in names and f"{RUN_ID}/run.json" in names
    assert not any(n.endswith("secret.txt") for n in names)


def test_raw_served_and_sandboxed(client: TestClient) -> None:
    r = client.get(f"/api/runs/{RUN_ID}/raw/x/page1.json")
    assert r.status_code == 200 and r.json() == {"tweets": []}
    assert "sandbox" in r.headers["content-security-policy"]
    r = client.get(f"/api/runs/{RUN_ID}/raw/x/page.html")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")


@pytest.mark.parametrize("path", ["..%2Fsecret.txt", "x%2F..%2F..%2Fsecret.txt",
                                  "%2E%2E/secret.txt", "..%5Csecret.txt", "", "x",
                                  "%2Fetc%2Fpasswd"])
def test_raw_path_traversal_rejected(client: TestClient, path: str) -> None:
    r = client.get(f"/api/runs/{RUN_ID}/raw/{path}")
    assert r.status_code == 404
    assert "outside raw" not in r.text


def test_raw_symlink_escape_rejected(client: TestClient, run_dir: Path) -> None:
    link = run_dir / "raw" / "link.txt"
    try:
        link.symlink_to(run_dir / "secret.txt")
    except OSError:
        pytest.skip("symlinks unsupported")
    assert client.get(f"/api/runs/{RUN_ID}/raw/link.txt").status_code == 404


def test_index_serves_html(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and "html" in r.headers["content-type"]


@pytest.mark.parametrize("path,ctype", [
    ("/static/fonts/inter-latin.woff2", "font/woff2"),
    ("/static/fonts/jost-latin.woff2", "font/woff2"),
    ("/static/brand/spidersilk-logo.svg", "image/svg+xml"),
    ("/static/brand/favicon.svg", "image/svg+xml"),
])
def test_static_brand_and_font_assets(client: TestClient, path: str, ctype: str) -> None:
    r = client.get(path)
    assert r.status_code == 200 and r.headers["content-type"].startswith(ctype)


def test_static_traversal_refused(client: TestClient) -> None:
    for path in ("/static/../server/app.py", "/static/fonts/../../server/app.py",
                 "/static/%2e%2e/server/app.py"):
        r = client.get(path)
        assert r.status_code == 404 or "def create_app" not in r.text, path


# ------------------------------------------------------------------- POST /runs
@pytest.mark.parametrize("body,needle", [
    ({"topic": "t", "start": "2026-10-01", "end": "2026-11-15"}, "maximum is 30"),
    ({"topic": "t", "start": "2026-11-10", "end": "2026-11-01"}, "before start"),
    ({"topic": "  ", "start": "2026-11-01", "end": "2026-11-02"}, "topic"),
    ({"topic": "t", "start": "not-a-date", "end": "2026-11-02"}, "invalid date"),
    ({"topic": "t", "start": "2026-11-01", "end": "2026-11-02", "sources": ["myspace"]},
     "unknown source"),
    ({"topic": "t", "start": "2026-11-01", "end": "2026-11-02", "limit": 0}, "limit"),
])
def test_post_run_invalid(client: TestClient, body: dict, needle: str) -> None:
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    assert needle in r.json()["detail"]


def test_post_run_all_sources_skipped(run_dir: Path, monkeypatch) -> None:
    """Real pipeline, no credentials: youtube is skipped and the run still completes."""
    monkeypatch.setattr(pipeline, "_analyze", lambda posts, meta, rd, cfg, prog: None)

    def light_export(rd, meta, prog):
        (rd / "report.html").write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(pipeline, "_export_and_report", light_export)
    jobs = JobManager()
    c = client_for(create_app(initial_run=run_dir, jobs=jobs))
    r = c.post("/api/runs", json={"topic": "skip test", "start": "2026-11-01",
                                  "end": "2026-11-05", "sources": ["youtube"]})
    assert r.status_code == 200, r.text
    rid = r.json()["run_id"]
    jobs.wait(30)
    assert c.get(f"/api/runs/{rid}/progress").json()["done"] is True
    body = c.get(f"/api/runs/{rid}").json()
    assert body["run"]["status"] == "complete"
    assert [s["state"] for s in body["source_status"]] == ["skipped_no_credentials"]
    assert "YOUTUBE_API_KEY" in body["source_status"][0]["reason"]
    assert rid in {x["run_id"] for x in c.get("/api/runs").json()}


def test_post_run_conflict_while_running(run_dir: Path) -> None:
    release = threading.Event()
    seen: dict = {}

    def fake_run(topic, start, end, sources, *, limit, progress, out_root, cfg, options=None, stop=None):
        d = out_root / "fake-run-1"
        d.mkdir(parents=True)
        write_json(d / "run.json", {"run_id": "fake-run-1", "topic": topic, "status": "running",
                                    "created_at": "2026-11-18T00:00:00Z"})
        seen.update(sources=sources, limit=limit)
        progress({"stage": "collect", "run_id": "fake-run-1"})
        release.wait(10)
        return d

    jobs = JobManager(runner=fake_run)
    c = client_for(create_app(initial_run=run_dir, jobs=jobs))
    body = {"topic": "t", "start": "2026-11-01", "end": "2026-11-02", "limit": 7}
    r = c.post("/api/runs", json=body)
    assert r.json() == {"run_id": "fake-run-1"}
    assert seen == {"sources": list(pipeline.DEFAULT_SOURCES), "limit": 7}
    assert c.get("/api/runs/fake-run-1/progress").json()["done"] is False
    assert c.post("/api/runs", json=body).status_code == 409
    release.set()
    jobs.wait(10)
    assert jobs.busy() is False


def test_job_start_error_when_runner_crashes(run_dir: Path) -> None:
    def boom(*a, **k):
        raise RuntimeError(f"exploded token={SECRET}")

    c = client_for(create_app(initial_run=run_dir, jobs=JobManager(runner=boom)))
    r = c.post("/api/runs", json={"topic": "t", "start": "2026-11-01", "end": "2026-11-02"})
    assert r.status_code == 500
    assert SECRET not in r.text


def test_sample_run_kind(monkeypatch, tmp_path: Path) -> None:
    sample = build_run(tmp_path / "data", "sample-run-id")
    import socmint.server.app as app_mod

    monkeypatch.setattr(app_mod, "sample_run_dir", lambda: sample)
    c = client_for(create_app(initial_run=sample, read_only=True))
    rows = {r["run_id"]: r for r in c.get("/api/runs").json()}
    assert rows["sample-run-id"]["path_kind"] == "sample"
    assert c.get("/api/meta").json()["initial_run_id"] == "sample-run-id"
    assert json.loads(c.get("/api/runs/sample-run-id/files/run.json").content)["run_id"] == \
        "sample-run-id"


# ------------------------------------------------------------- topic match option
class _OptionsJobs(JobManager):
    """Records what the server passes to a run (a JobManager that forwards options)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def start(self, *, topic, start, end, sources, limit, out_root, cfg, options=None,
              timeout=60.0):
        from socmint.server.jobs import Job

        self.calls.append({"topic": topic, "limit": limit, "options": options,
                           "start": start, "end": end})
        return Job(run_id="2026-11-19-match-aaaaaa", done=True)


def _post_run(c: TestClient, **extra):
    return c.post("/api/runs", json={"topic": "bank of america", "start": "2026-11-13",
                                     "end": "2026-11-19", "sources": ["youtube"], **extra})


def test_run_match_defaults_to_phrase_and_passes_words(run_dir: Path) -> None:
    jobs = _OptionsJobs()
    c = client_for(create_app(initial_run=run_dir, jobs=jobs))
    assert _post_run(c).status_code == 200
    assert _post_run(c, match="words").status_code == 200
    assert _post_run(c, match=" Phrase ").status_code == 200
    assert [x["options"] for x in jobs.calls] == [
        {"match": "phrase", "window_preset": "custom"},
        {"match": "words", "window_preset": "custom"},
        {"match": "phrase", "window_preset": "custom"}]


def test_run_match_rejects_unknown_value(run_dir: Path) -> None:
    jobs = _OptionsJobs()
    c = client_for(create_app(initial_run=run_dir, jobs=jobs))
    r = _post_run(c, match="fuzzy")
    assert r.status_code == 400 and "match must be one of" in r.json()["detail"]
    assert jobs.calls == []


def test_real_job_manager_forwards_match_to_the_pipeline(tmp_path: Path) -> None:
    """The real JobManager passes options through to pipeline.run (was a 501 gap)."""
    from datetime import date

    from socmint.server.jobs import JobManager

    seen: dict = {}

    def runner(topic, start, end, sources, *, limit, options, progress, out_root, cfg, stop=None):
        seen.update(options=options)
        progress({"run_id": "2026-11-19-match-bbbbbb"})
        return out_root / "2026-11-19-match-bbbbbb"

    jm = JobManager(runner=runner)
    jm.start(topic="t", start=date(2026, 11, 18), end=date(2026, 11, 19), sources=["x"],
             limit=10, out_root=tmp_path, cfg={}, options={"match": "words"})
    jm.current.started.wait(5)
    for _ in range(100):
        if jm.current.done:
            break
        import time
        time.sleep(0.01)
    assert seen["options"] == {"match": "words"}
