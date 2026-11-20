"""Read/write the on-disk run layout (docs/SPEC.md §8). The GUI and report read ONLY this.

runs/<id>/
  posts.jsonl  posts.csv  narratives.json  clusters.json  origin.json  claims.json
  source_status.json  brief.json  report.html  raw/  run.json  progress.json  llm_log.jsonl

brief.json (run summary, analyze/brief.py) is absent in runs made before it existed;
load_run then returns {} under "brief".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from socmint.schema import dump

FILES = {
    "posts": "posts.jsonl",
    "posts_csv": "posts.csv",
    "narratives": "narratives.json",
    "clusters": "clusters.json",
    "origin": "origin.json",
    "claims": "claims.json",
    "source_status": "source_status.json",
    "brief": "brief.json",
    "report": "report.html",
    "run": "run.json",
    "progress": "progress.json",
}


def write_json(path: Path, data: Any) -> None:
    if isinstance(data, BaseModel):
        data = dump(data)
    elif isinstance(data, list):
        data = [dump(x) if isinstance(x, BaseModel) else x for x in data]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dump(r), ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    # Split on "\n" ONLY. str.splitlines() also breaks on U+2028/U+2029/\x85 etc., which
    # json.dumps(ensure_ascii=False) leaves raw inside post text (original text is kept),
    # so it would cut a record in half (seen live: a post containing U+2028).
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def load_run(run_dir: Path) -> dict[str, Any]:
    """Everything a viewer needs, as plain JSON-ready dicts."""
    return {
        "run": read_json(run_dir / FILES["run"], {}),
        "posts": read_jsonl(run_dir / FILES["posts"]),
        "narratives": read_json(run_dir / FILES["narratives"], []),
        "clusters": read_json(run_dir / FILES["clusters"], []),
        "origin": read_json(run_dir / FILES["origin"], []),
        "claims": read_json(run_dir / FILES["claims"], []),
        "source_status": read_json(run_dir / FILES["source_status"], []),
        "brief": read_json(run_dir / FILES["brief"], {}) or {},
    }


class RunDeleteError(Exception):
    """A run directory that must not (or cannot) be deleted."""


def delete_run(run_dir: Path, runs_root: Path) -> Path:
    """Permanently delete one run directory, with guards against deleting anything else.

    Only a directory that (1) sits DIRECTLY inside `runs_root`, (2) contains run.json and
    (3) is not a symlink can be deleted. The bundled sample run and arbitrary paths are
    refused. Returns the deleted path.
    """
    import shutil

    from socmint.config import sample_run_dir

    d = Path(run_dir)
    if d.is_symlink():
        raise RunDeleteError(f"{d} is a symlink; refusing to delete")
    d = d.resolve()
    root = Path(runs_root).resolve()
    if d == sample_run_dir().resolve():
        raise RunDeleteError("the bundled sample run (data/sample_run) cannot be deleted")
    if d.parent != root:
        raise RunDeleteError(f"{d} is not a run inside {root}; only runs there can be deleted")
    if not (d / FILES["run"]).is_file():
        raise RunDeleteError(f"{d} is not a run directory (no run.json)")
    shutil.rmtree(d)
    return d


# A "running" run whose files have not changed for this long, and whose writer process is
# gone, is treated as dead (interrupted). Shared by the server and the CLI.
STALE_RUN_SECONDS = 1800
# A live writer pid only counts while the run was written within this window, so a pid
# recycled by the OS days later cannot block deleting an old interrupted run forever.
PID_TRUST_SECONDS = 6 * 3600


def pid_alive(pid: int) -> bool:
    """True if a process with this pid exists on this machine. Never signals it."""
    import os
    import sys

    if pid <= 0:
        return False
    if sys.platform == "win32":  # os.kill on Windows would TERMINATE the process
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # signal 0: existence check only
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except OSError:
        return False
    return True


def last_write(run_dir: Path) -> float:
    """Latest mtime of run.json / progress.json (0.0 if neither exists)."""
    d = Path(run_dir)
    mtimes = [p.stat().st_mtime for p in (d / FILES["progress"], d / FILES["run"])
              if p.is_file()]
    return max(mtimes, default=0.0)


def writer_alive(run_dir: Path, *, ignore_pid: int | None = None) -> bool:
    """True if progress.json names a live writer process (other than `ignore_pid`)."""
    import time

    prog = read_json(Path(run_dir) / FILES["progress"], {}) or {}
    if not isinstance(prog, dict) or prog.get("done"):
        return False
    try:
        pid = int(prog.get("pid") or 0)
    except (TypeError, ValueError):
        return False
    if not pid or pid == ignore_pid:
        return False
    if time.time() - last_write(run_dir) > PID_TRUST_SECONDS:
        return False
    return pid_alive(pid)


def run_in_progress(run_dir: Path, *, ignore_pid: int | None = None,
                    stale_seconds: float = STALE_RUN_SECONDS) -> bool:
    """True if a run may still be being written, so it must not be deleted.

    status == "running" AND (its writer process is alive OR it was written recently).
    A live writer can go minutes without writing (flood waits, model download, LLM
    calls), so recency alone is not enough; the pid in progress.json is checked too.
    """
    import time

    meta = read_json(Path(run_dir) / FILES["run"], {}) or {}
    if meta.get("status") != "running":
        return False
    if writer_alive(run_dir, ignore_pid=ignore_pid):
        return True
    last = last_write(run_dir)
    return bool(last) and time.time() - last < stale_seconds


def list_runs(*roots: Path) -> list[dict[str, Any]]:
    out = []
    for root in roots:
        if not root.is_dir():
            continue
        candidates = [root] if (root / "run.json").is_file() else sorted(root.iterdir())
        for d in candidates:
            meta = read_json(d / "run.json") if d.is_dir() else None
            if meta:
                out.append({**meta, "path": str(d)})
    out.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return out
