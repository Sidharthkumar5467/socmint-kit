"""CLI + doctor tests. No server is started and no network is touched."""

from __future__ import annotations

import socket
import sys
import types
from datetime import date
from pathlib import Path

import pytest

from socmint import cli, doctor
from socmint.runstore import write_json
from socmint.schema import RunMeta, SourceState, SourceStatus

SECRET = "AIzaSyTESTSECRETVALUE0000000000"


def _run_dir(root: Path) -> Path:
    d = root / "r1"
    d.mkdir(parents=True)
    write_json(d / "run.json", RunMeta(run_id="r1", topic="t", start="2026-11-01",
                                       end="2026-11-02", sources=["youtube"],
                                       created_at="2026-11-18T00:00:00Z", status="complete"))
    write_json(d / "source_status.json", [SourceStatus(
        source="youtube", query="t", state=SourceState.skipped_no_credentials,
        reason="missing credential(s): YOUTUBE_API_KEY")])
    return d


# ---------------------------------------------------------------------- parser
def test_parser_run() -> None:
    a = cli.build_parser().parse_args(["run", "--topic", "floods", "--start", "2026-11-01",
                                       "--end", "2026-11-10", "--sources", "x,youtube",
                                       "--limit", "50"])
    assert a.command == "run" and a.topic == "floods"
    assert a.start == date(2026, 11, 1) and a.end == date(2026, 11, 10)
    assert a.sources == "x,youtube" and a.limit == 50 and a.out is None


def test_parser_run_match_option() -> None:
    base = ["run", "--topic", "bank of america", "--start", "2026-11-13", "--end", "2026-11-19"]
    assert cli.build_parser().parse_args(base).match == "phrase"  # default
    assert cli.build_parser().parse_args(base + ["--match", "words"]).match == "words"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(base + ["--match", "any"])


def test_cmd_run_passes_words_match(monkeypatch, tmp_path: Path) -> None:
    d = _run_dir(tmp_path)
    seen = {}

    def fake_run(topic, start, end, sources, *, limit, options, progress, out_root):
        seen["options"] = options
        return d

    from socmint import pipeline

    monkeypatch.setattr(pipeline, "run", fake_run)
    assert cli.main(["run", "--topic", "a b", "--start", "2026-11-01", "--end", "2026-11-02",
                     "--match", "words"]) == 0
    assert seen["options"] == {"match": "words", "window_preset": "custom"}


def test_parser_rejects_bad_date_and_host() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--topic", "t", "--start", "yesterday",
                                       "--end", "2026-11-10"])
    with pytest.raises(SystemExit):  # there is no --host: the GUI is loopback-only
        cli.build_parser().parse_args(["gui", "--host", "0.0.0.0"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["gui", "--port", "70000"])


def test_parser_gui_replay_doctor_report() -> None:
    p = cli.build_parser()
    g = p.parse_args(["gui", "--port", "9000", "--no-browser"])
    assert (g.port, g.no_browser, g.run) == (9000, True, None)
    r = p.parse_args(["replay", "data/sample_run"])
    assert r.run_dir == Path("data/sample_run") and r.port is None
    assert p.parse_args(["doctor", "--prefetch"]).prefetch is True
    assert p.parse_args(["report", "runs/x"]).run_dir == Path("runs/x")
    assert p.parse_args([]).command is None  # -> gui


# --------------------------------------------------------------------- helpers
def test_bind_host_only_loopback_or_docker() -> None:
    assert cli.bind_host({}) == "127.0.0.1"
    assert cli.bind_host({"SOCMINT_BIND": "0.0.0.0"}) == "0.0.0.0"
    assert cli.bind_host({"SOCMINT_BIND": "192.168.1.5"}) == "127.0.0.1"


def test_find_free_port_is_bindable() -> None:
    port = cli.find_free_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_status_table_explains_empty_sources() -> None:
    out = cli.status_table([{"source": "youtube", "state": "skipped_no_credentials",
                             "count": 0, "reason": "missing YOUTUBE_API_KEY"}])
    assert "skipped_no_credentials" in out and "missing YOUTUBE_API_KEY" in out
    assert "no source status" in cli.status_table([])


# -------------------------------------------------------------------- commands
def test_gui_defaults_and_replay_read_only(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(cli, "serve", lambda initial, **kw: calls.append((initial, kw)) or 0)
    monkeypatch.setattr(cli, "default_initial_run", lambda: None)
    assert cli.main([]) == 0
    assert calls[-1] == (None, {"read_only": False, "port": None, "open_browser": True})

    d = _run_dir(tmp_path)
    assert cli.main(["replay", str(d), "--no-browser", "--port", "9999"]) == 0
    assert calls[-1] == (d.resolve(), {"read_only": True, "port": 9999, "open_browser": False})
    assert cli.main(["gui", "--run", str(d)]) == 0
    assert calls[-1][0] == d.resolve() and calls[-1][1]["read_only"] is False

    # Not a run directory -> clear error, exit 1, server not started.
    n = len(calls)
    assert cli.main(["replay", str(tmp_path / "missing")]) == 1
    assert cli.main(["gui", "--run", str(tmp_path / "missing")]) == 1
    assert len(calls) == n


def test_cmd_run_prints_status_and_exits_zero(monkeypatch, tmp_path: Path, capsys) -> None:
    d = _run_dir(tmp_path)
    seen = {}

    def fake_run(topic, start, end, sources, *, limit, options, progress, out_root):
        seen.update(topic=topic, sources=sources, limit=limit, out_root=out_root,
                    options=options)
        progress({"message": "youtube: skipped_no_credentials (0 posts)"})
        return d

    from socmint import pipeline

    monkeypatch.setattr(pipeline, "run", fake_run)
    rc = cli.main(["run", "--topic", "t", "--start", "2026-11-01", "--end", "2026-11-02",
                   "--sources", "youtube, x", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert seen == {"topic": "t", "sources": ["youtube", "x"], "limit": 500, "out_root": tmp_path,
                    "options": {"match": "phrase", "window_preset": "custom"}}
    assert str(d) in out and "skipped_no_credentials" in out and "YOUTUBE_API_KEY" in out


def test_cmd_run_invalid_window_exit_2(capsys) -> None:
    rc = cli.main(["run", "--topic", "t", "--start", "2026-01-01", "--end", "2026-03-01"])
    assert rc == 2
    assert "maximum is 30" in capsys.readouterr().err


def test_cmd_report_regenerates_offline(monkeypatch, tmp_path: Path) -> None:
    d = _run_dir(tmp_path)
    order = []
    fake_export = types.ModuleType("socmint.export")
    fake_export.write_exports = lambda rd: order.append(("export", rd))
    fake_report = types.ModuleType("socmint.report")
    fake_report.build_report = lambda rd: order.append(("report", rd)) or rd / "report.html"
    import socmint

    monkeypatch.setitem(sys.modules, "socmint.export", fake_export)
    monkeypatch.setitem(sys.modules, "socmint.report", fake_report)
    monkeypatch.setattr(socmint, "export", fake_export, raising=False)
    monkeypatch.setattr(socmint, "report", fake_report, raising=False)
    assert cli.main(["report", str(d)]) == 0
    assert order == [("export", d.resolve()), ("report", d.resolve())]
    assert cli.main(["report", str(tmp_path / "nope")]) == 1


# ---------------------------------------------------------------------- doctor
def test_source_readiness_reports_missing_keys(monkeypatch) -> None:
    rows = {r["name"]: r for r in doctor.source_readiness({"YOUTUBE_API_KEY": ""})}
    assert {"x", "telegram", "youtube", "linkedin"} <= set(rows)
    yt = rows["youtube"]
    if yt["kind"] is not None:
        assert yt["ready"] is False and "YOUTUBE_API_KEY" in yt["reason"]
        assert yt["keys_set"] == {"YOUTUBE_API_KEY": False}
    rows = {r["name"]: r for r in doctor.source_readiness({"YOUTUBE_API_KEY": SECRET})}
    if rows["youtube"]["kind"] is not None:
        assert rows["youtube"]["ready"] is True


def test_source_readiness_reports_broken_collector(monkeypatch) -> None:
    import socmint.collect as collect

    monkeypatch.setattr(collect, "available", lambda: {**collect.BUILTIN, "broken": "no.mod:X"})
    rows = {r["name"]: r for r in doctor.source_readiness({})}
    assert rows["broken"]["ready"] is False
    assert "collector unavailable" in rows["broken"]["reason"]


def test_doctor_never_prints_secret_values(monkeypatch, capsys) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", SECRET)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-test-value-123")
    monkeypatch.setattr(doctor, "embedding_model_cached", lambda cfg: (False, "not cached"))
    assert doctor.main() == 0
    out = capsys.readouterr().out
    assert SECRET not in out and "fake-anthropic-test-value-123" not in out
    assert "YOUTUBE_API_KEY=set" in out or "youtube" in out
    assert "ANTHROPIC_API_KEY: set" in out
    assert "GOOGLE_FACTCHECK_API_KEY: missing" in out
    assert "Telegram session: not configured" in out
    assert "runs dir" in out and "Python" in out


def test_doctor_prefetch_calls_embed(monkeypatch, capsys) -> None:
    import socmint.analyze.embed as embed

    monkeypatch.setattr(embed, "prefetch_model", lambda cfg: (True, "model cached and loadable"))
    monkeypatch.setattr(doctor, "embedding_model_cached", lambda cfg: (True, "cached"))
    assert doctor.main(prefetch=True) == 0
    assert "prefetch: model cached and loadable" in capsys.readouterr().out


def test_telegram_auth_call_flexible_signature(monkeypatch) -> None:
    fake = types.ModuleType("socmint.collect.telegram_auth")

    async def status(config):
        return {"authorized": config.get("TELEGRAM_API_ID") == "1"}

    fake.status = status
    import socmint.collect as collect

    monkeypatch.setitem(sys.modules, "socmint.collect.telegram_auth", fake)
    monkeypatch.setattr(collect, "telegram_auth", fake, raising=False)
    cfg = {"TELEGRAM_API_ID": "1", "TELEGRAM_API_HASH": "h" * 10}
    assert doctor.telegram_auth_call("status", cfg) == {"authorized": True}
    assert doctor.telegram_session_status(cfg) == (True, "authorized")
    with pytest.raises(doctor.TelegramAuthUnavailable):
        doctor.telegram_auth_call("missing_fn", cfg)


# ------------------------------------------------------------------- reanalyze
def test_parser_reanalyze() -> None:
    a = cli.build_parser().parse_args(["reanalyze", "runs/r1"])
    assert a.command == "reanalyze" and a.run_dir == Path("runs/r1")


def test_cmd_reanalyze_calls_pipeline(monkeypatch, tmp_path: Path, capsys) -> None:
    from socmint import pipeline

    d = _run_dir(tmp_path)
    calls = []

    def fake_reanalyze(run_dir, *, progress=None, cfg=None):
        calls.append(run_dir)
        progress({"message": "analysis: brief"})
        return run_dir

    monkeypatch.setattr(pipeline, "reanalyze", fake_reanalyze)
    assert cli.main(["reanalyze", str(d)]) == 0
    assert calls == [d.resolve()]
    out = capsys.readouterr().out
    assert "analysis: brief" in out and "status: complete" in out

    assert cli.main(["reanalyze", str(tmp_path / "missing")]) == 1
    assert len(calls) == 1


def test_cmd_reanalyze_reports_step_errors_and_failure(monkeypatch, tmp_path: Path,
                                                       capsys) -> None:
    from socmint import pipeline
    from socmint.runstore import read_json

    d = _run_dir(tmp_path)

    def fake_reanalyze(run_dir, *, progress=None, cfg=None):
        meta = read_json(run_dir / "run.json")
        meta["options"] = {"analysis_errors": {"claims": "RuntimeError: x"}}
        meta["status"] = "failed"
        write_json(run_dir / "run.json", meta)
        return run_dir

    monkeypatch.setattr(pipeline, "reanalyze", fake_reanalyze)
    assert cli.main(["reanalyze", str(d)]) == 1
    assert "analysis step claims failed: RuntimeError: x" in capsys.readouterr().out


def test_cmd_reanalyze_refuses_running_run(monkeypatch, tmp_path: Path, capsys) -> None:
    from socmint import pipeline, runstore

    d = _run_dir(tmp_path)
    monkeypatch.setattr(runstore, "run_in_progress", lambda run_dir: True)
    monkeypatch.setattr(pipeline, "reanalyze", lambda *a, **k: pytest.fail("ran"))
    assert cli.main(["reanalyze", str(d)]) == 1
    assert "still being written" in capsys.readouterr().err
