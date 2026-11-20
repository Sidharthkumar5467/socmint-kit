"""Bootstrap logic (socmint/bootstrap.py) and its stdlib-only twin in run.py."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import types
from pathlib import Path

import pytest

from socmint import bootstrap as bs
from socmint.config import project_root


def _load_runpy():
    spec = importlib.util.spec_from_file_location("socmint_run_py", project_root() / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def proj(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    return root


def _fake_venv(venv_dir: Path) -> None:
    py = bs.venv_python(venv_dir)
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")


def test_version_ok() -> None:
    assert bs.version_ok((3, 11, 0)) and bs.version_ok((3, 13)) and not bs.version_ok((3, 10, 9))


def test_deps_hash_changes_with_either_file(proj: Path) -> None:
    h1 = bs.deps_hash(proj)
    assert h1 == bs.deps_hash(proj)
    (proj / "requirements.txt").write_text("fastapi\nhttpx\n", encoding="utf-8")
    h2 = bs.deps_hash(proj)
    (proj / "pyproject.toml").write_text("[project]\nname='y'\n", encoding="utf-8")
    assert len({h1, h2, bs.deps_hash(proj)}) == 3


def test_needs_install_skip_logic(proj: Path) -> None:
    venv_dir = proj / ".venv"
    assert bs.needs_install(venv_dir, proj)  # no venv
    _fake_venv(venv_dir)
    assert bs.needs_install(venv_dir, proj)  # venv but never installed
    bs.mark_installed(venv_dir, proj)
    assert not bs.needs_install(venv_dir, proj)  # second launch: skip
    (proj / "requirements.txt").write_text("fastapi\nnumpy\n", encoding="utf-8")
    assert bs.needs_install(venv_dir, proj)  # deps changed: reinstall


def test_install_marks_only_on_success(proj: Path) -> None:
    venv_dir = proj / ".venv"
    _fake_venv(venv_dir)
    py = bs.venv_python(venv_dir)
    cmds: list = []

    def fail_second(cmd, check=False):
        cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0 if len(cmds) == 1 else 1)

    assert bs.install(py, proj, venv_dir, runner=fail_second) is False
    assert bs.needs_install(venv_dir, proj)
    assert cmds[0][-2:] == ["-r", str(proj / "requirements.txt")]
    assert cmds[1][-3:] == ["-e", str(proj), "--no-deps"]
    ok = bs.install(py, proj, venv_dir,
                    runner=lambda cmd, check=False: subprocess.CompletedProcess(cmd, 0))
    assert ok and not bs.needs_install(venv_dir, proj)


def test_find_newer_python_checks_real_version() -> None:
    paths = {"python3.12": "/x/python3.12", "python3.11": "/x/python3.11"}
    versions = {"/x/python3.12": (3, 10), "/x/python3.11": (3, 11)}  # a mislabelled binary
    assert bs.find_newer_python(paths.get, versions.get) == "/x/python3.11"
    assert bs.find_newer_python(lambda n: None, versions.get) is None


def test_ensure_config_creates_once(tmp_path: Path) -> None:
    home = tmp_path / "h"
    p = bs.ensure_config(home)
    assert json.loads(p.read_text()) == {}
    if os.name != "nt":
        assert (p.stat().st_mode & 0o777) == 0o600
    p.write_text('{"YOUTUBE_API_KEY": "k"}', encoding="utf-8")
    bs.ensure_config(home)
    assert json.loads(p.read_text()) == {"YOUTUBE_API_KEY": "k"}  # never overwritten


def test_gui_command_has_no_host() -> None:
    cmd = bs.gui_command(Path("/v/bin/python"), 8000, True, None)
    assert cmd[1:] == ["-m", "socmint.cli", "gui", "--port", "8000", "--no-browser"]
    assert "--host" not in cmd


# ------------------------------------------------------------- run.py (stdlib twin)
def test_runpy_helpers_match_bootstrap(proj: Path) -> None:
    rp = _load_runpy()
    assert rp.deps_hash(str(proj)) == bs.deps_hash(proj)
    assert rp.HASH_FILE == bs.HASH_FILE and rp.MIN_VERSION == bs.MIN_VERSION
    assert rp.CANDIDATE_PYTHONS == bs.CANDIDATE_PYTHONS
    venv_dir = proj / ".venv"
    _fake_venv(venv_dir)
    assert rp.needs_install(str(venv_dir), str(proj))
    bs.mark_installed(venv_dir, proj)  # written by one copy, honoured by the other
    assert not rp.needs_install(str(venv_dir), str(proj))
    rp.mark_installed(str(venv_dir), str(proj))
    assert not bs.needs_install(venv_dir, proj)


def test_runpy_rejects_host_flag() -> None:
    rp = _load_runpy()
    a = rp.parse_args(["--port", "8123", "--no-browser"])
    assert (a.port, a.no_browser, a.run, a.reinstall) == (8123, True, None, False)
    with pytest.raises(SystemExit):
        rp.parse_args(["--host", "0.0.0.0"])


def test_runpy_old_python_reexecs_or_exits(monkeypatch) -> None:
    rp = _load_runpy()
    fake_sys = types.SimpleNamespace(version_info=(3, 9, 7), argv=["run.py", "--no-browser"],
                                     exit=lambda code: (_ for _ in ()).throw(SystemExit(code)))
    monkeypatch.setattr(rp, "sys", fake_sys)
    monkeypatch.setenv(rp.REEXEC_FLAG, "")

    # No newer interpreter on PATH -> clear message and exit 1.
    monkeypatch.setattr(rp, "find_newer_python", lambda: None)
    with pytest.raises(SystemExit) as ei:
        rp.ensure_python()
    assert ei.value.code == 1

    # A newer interpreter exists -> re-exec run.py with it, passing the args through.
    execs = []
    monkeypatch.setattr(rp, "find_newer_python", lambda: "/usr/bin/python3.12")
    monkeypatch.setattr(rp.os, "execv", lambda exe, argv: execs.append((exe, argv)))
    with pytest.raises(SystemExit):  # our fake execv returns, so it falls through to exit
        rp.ensure_python()
    assert execs and execs[0][0] == "/usr/bin/python3.12"
    assert execs[0][1][1].endswith("run.py") and execs[0][1][2:] == ["--no-browser"]

    # Loop guard: after one re-exec it does not try again.
    execs.clear()
    with pytest.raises(SystemExit):
        rp.ensure_python()
    assert execs == []


def test_runpy_ensure_config(monkeypatch, tmp_path: Path) -> None:
    rp = _load_runpy()
    monkeypatch.setenv("SOCMINT_HOME", str(tmp_path / "home2"))
    p = rp.ensure_config()
    assert json.loads(Path(p).read_text()) == {}
