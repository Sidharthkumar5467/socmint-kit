#!/usr/bin/env python3
"""One-command launcher: `python run.py` (docs/SPEC.md section 3.1).

STDLIB ONLY and deliberately self-contained: it runs before any dependency is installed
and may be started by an old python3, so it must parse under Python 3.6+. The helpers
below mirror socmint/bootstrap.py (tests check they agree).

Steps: Python >= 3.11 (re-exec with python3.13/3.12/3.11 from PATH if needed) ->
create/reuse .venv -> pip install only when sha256(requirements.txt + pyproject.toml)
changed -> ensure ~/.socmint/config.json -> exec `.venv python -m socmint.cli gui`
(server bound to 127.0.0.1 on a free port; browser opened at its URL).

Options: --port N, --no-browser, --run DIR, --reinstall. There is no --host option:
the GUI holds credentials and only ever listens on 127.0.0.1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

MIN_VERSION = (3, 11)
CANDIDATE_PYTHONS = ("python3.13", "python3.12", "python3.11")
HASH_FILE = ".requirements.sha256"
DEP_FILES = ("requirements.txt", "pyproject.toml")
ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")
REEXEC_FLAG = "SOCMINT_RUNPY_REEXEC"


def say(msg):
    print("[socmint] " + msg, flush=True)


def python_version_of(exe):
    try:
        out = subprocess.run(
            [exe, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True,
            timeout=20,
        ).stdout.strip()
        major, minor = out.split(".")
        return int(major), int(minor)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def find_newer_python():
    for name in CANDIDATE_PYTHONS:
        path = shutil.which(name)
        if path:
            v = python_version_of(path)
            if v and v >= MIN_VERSION:
                return path
    return None


def venv_python(venv_dir):
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def deps_hash(root):
    h = hashlib.sha256()
    for name in DEP_FILES:
        p = os.path.join(root, name)
        h.update(name.encode())
        h.update(b"\0")
        if os.path.isfile(p):
            with open(p, "rb") as f:
                h.update(f.read())
        h.update(b"\0")
    return h.hexdigest()


def needs_install(venv_dir, root):
    if not os.path.exists(venv_python(venv_dir)):
        return True
    try:
        with open(os.path.join(venv_dir, HASH_FILE), encoding="utf-8") as f:
            recorded = f.read().strip()
    except OSError:
        return True
    return recorded != deps_hash(root)


def mark_installed(venv_dir, root):
    with open(os.path.join(venv_dir, HASH_FILE), "w", encoding="utf-8") as f:
        f.write(deps_hash(root) + "\n")


def ensure_python():
    """Exit 1 with a clear message, or re-exec under a suitable interpreter."""
    if sys.version_info[:2] >= MIN_VERSION:
        return
    have = "{}.{}".format(*sys.version_info[:2])
    need = "{}.{}".format(*MIN_VERSION)
    if not os.environ.get(REEXEC_FLAG):
        newer = find_newer_python()
        if newer:
            say(f"Python {have} is too old; re-launching with {newer}")
            os.environ[REEXEC_FLAG] = "1"
            os.execv(newer, [newer, os.path.abspath(__file__)] + sys.argv[1:])
    say(f"Python {need} or newer is required (this is Python {have}).")
    say("Install Python 3.11+ (https://www.python.org/downloads/) and run: "
        "python3.11 run.py")
    sys.exit(1)


def ensure_venv():
    py = venv_python(VENV)
    if os.path.exists(py):
        v = python_version_of(py)
        if v is None or v < MIN_VERSION:
            found = "?" if v is None else "{}.{}".format(*v)
            need = "{}.{}".format(*MIN_VERSION)
            say(f"Existing .venv uses Python {found} (need >= {need}). Delete the .venv "
                "folder and run again.")
            sys.exit(1)
        return py
    import venv  # stdlib

    say("creating virtual environment in .venv/ ...")
    venv.EnvBuilder(with_pip=True).create(VENV)
    return py


def install(py, force):
    if not force and not needs_install(VENV, ROOT):
        say("dependencies up to date (skipping install)")
        return
    say("installing dependencies into .venv/ (first launch can take several minutes) ...")
    cmds = [
        # --timeout/--retries: big wheels (e.g. torch via sentence-transformers) drop on
        # pip's short default timeout on a slow link; give the download room and resume.
        [py, "-m", "pip", "install", "--timeout", "300", "--retries", "5",
         "-r", os.path.join(ROOT, "requirements.txt")],
        [py, "-m", "pip", "install", "-e", ROOT, "--no-deps"],
    ]
    for cmd in cmds:
        if subprocess.call(cmd) != 0:
            say("install failed: " + " ".join(cmd[1:]))
            say("Fix the error above and run `python run.py` again "
                "(the install is retried until it succeeds).")
            sys.exit(1)
    mark_installed(VENV, ROOT)
    say("dependencies installed")


def ensure_config():
    home = os.environ.get("SOCMINT_HOME") or os.path.join(os.path.expanduser("~"), ".socmint")
    os.makedirs(home, exist_ok=True)
    p = os.path.join(home, "config.json")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps({}) + "\n")
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass
        say("created local config " + p)
    return p


def parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="run.py",
        description="Set up (once) and open the SOCMINT GUI on http://127.0.0.1:<port>/",
    )
    ap.add_argument("--port", type=int, default=None, help="port (default: a free one)")
    ap.add_argument("--no-browser", action="store_true", help="only print the URL")
    ap.add_argument("--run", default=None, help="open this run directory instead of the sample")
    ap.add_argument("--reinstall", action="store_true", help="force the dependency install")
    return ap.parse_args(argv)


def main(argv=None):
    ensure_python()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    py = ensure_venv()
    install(py, args.reinstall)
    ensure_config()
    cmd = [py, "-m", "socmint.cli", "gui"]
    if args.port:
        cmd += ["--port", str(args.port)]
    if args.no_browser:
        cmd.append("--no-browser")
    if args.run:
        cmd += ["--run", os.path.abspath(args.run)]
    os.chdir(ROOT)
    if os.name == "nt":  # os.execv does not replace the process cleanly on Windows
        sys.exit(subprocess.call(cmd))
    os.execv(py, cmd)


if __name__ == "__main__":
    main()
