"""Bootstrap helpers: Python check, venv, dependency install, local config (SPEC §3.1).

STDLIB ONLY. `run.py` at the repo root carries a self-contained copy of these helpers
(it must run before any dependency is installed, possibly under an old interpreter);
`tests/test_bootstrap.py` checks the two copies agree. Keep them in sync.

Install skipping: the dependency set is identified by sha256(requirements.txt +
pyproject.toml). After a successful install the digest is written to
`.venv/.requirements.sha256`; a later launch with the same digest skips pip entirely, so
the second launch is instant.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import venv
from collections.abc import Callable, Sequence
from pathlib import Path

MIN_VERSION: tuple[int, int] = (3, 11)
# Interpreter names tried, newest first, when run.py is launched by an older python3.
CANDIDATE_PYTHONS: tuple[str, ...] = ("python3.13", "python3.12", "python3.11")
HASH_FILE = ".requirements.sha256"
DEP_FILES: tuple[str, ...] = ("requirements.txt", "pyproject.toml")


def version_ok(version_info: Sequence[int] | None = None) -> bool:
    vi = tuple(version_info if version_info is not None else sys.version_info)[:2]
    return vi >= MIN_VERSION


def python_version_of(exe: str) -> tuple[int, int] | None:
    """Ask an interpreter for its (major, minor); None if it cannot be run."""
    try:
        out = subprocess.run(
            [exe, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout.strip()
        major, minor = out.split(".")
        return int(major), int(minor)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def find_newer_python(
    which: Callable[[str], str | None] = shutil.which,
    probe: Callable[[str], tuple[int, int] | None] = python_version_of,
) -> str | None:
    """First python3.13/3.12/3.11 on PATH whose real version is >= MIN_VERSION."""
    for name in CANDIDATE_PYTHONS:
        path = which(name)
        if path and (v := probe(path)) and v >= MIN_VERSION:
            return path
    return None


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def deps_hash(root: Path) -> str:
    """sha256 over requirements.txt + pyproject.toml (a missing file hashes as empty)."""
    h = hashlib.sha256()
    for name in DEP_FILES:
        p = root / name
        h.update(name.encode())
        h.update(b"\0")
        h.update(p.read_bytes() if p.is_file() else b"")
        h.update(b"\0")
    return h.hexdigest()


def needs_install(venv_dir: Path, root: Path) -> bool:
    """True unless the venv exists and its recorded digest matches the current files."""
    if not venv_python(venv_dir).exists():
        return True
    marker = venv_dir / HASH_FILE
    try:
        recorded = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return True
    return recorded != deps_hash(root)


def mark_installed(venv_dir: Path, root: Path) -> None:
    """Record the digest. Call ONLY after pip succeeded."""
    (venv_dir / HASH_FILE).write_text(deps_hash(root) + "\n", encoding="utf-8")


def ensure_venv(venv_dir: Path) -> Path:
    """Create the venv with pip if missing; return its interpreter."""
    py = venv_python(venv_dir)
    if not py.exists():
        venv.EnvBuilder(with_pip=True, clear=False, upgrade_deps=False).create(str(venv_dir))
    return py


def install_commands(py: Path, root: Path) -> list[list[str]]:
    # --timeout/--retries: the dependency set includes large wheels (e.g. torch, pulled by
    # sentence-transformers); pip's short default read timeout drops these mid-download on a
    # slow or unstable connection (ReadTimeoutError from files.pythonhosted.org).
    return [
        [str(py), "-m", "pip", "install", "--timeout", "300", "--retries", "5",
         "-r", str(root / "requirements.txt")],
        [str(py), "-m", "pip", "install", "-e", str(root), "--no-deps"],
    ]


def install(py: Path, root: Path, venv_dir: Path,
            runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> bool:
    """pip install requirements + the package; mark the digest only on full success."""
    for cmd in install_commands(py, root):
        if runner(cmd, check=False).returncode != 0:
            return False
    mark_installed(venv_dir, root)
    return True


def socmint_home(env: dict[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("SOCMINT_HOME") or Path.home() / ".socmint")


def ensure_config(home: Path | None = None) -> Path:
    """Create ~/.socmint/config.json as `{}` (mode 0600) if absent; never overwrite."""
    home = home or socmint_home()
    home.mkdir(parents=True, exist_ok=True)
    p = home / "config.json"
    if not p.exists():
        p.write_text(json.dumps({}) + "\n", encoding="utf-8")
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass
    return p


def gui_command(py: Path, port: int | None, no_browser: bool, run_dir: str | None) -> list[str]:
    """argv that starts the GUI. There is deliberately no --host: always 127.0.0.1."""
    cmd = [str(py), "-m", "socmint.cli", "gui"]
    if port:
        cmd += ["--port", str(int(port))]
    if no_browser:
        cmd.append("--no-browser")
    if run_dir:
        cmd += ["--run", run_dir]
    return cmd
