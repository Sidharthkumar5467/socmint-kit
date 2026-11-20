"""Command line (docs/SPEC.md §3.3). The GUI is the primary surface; the CLI exists for
automation and for building the sample run, and never has a feature the GUI lacks.

  socmint [gui] [--port N] [--no-browser] [--run DIR]
  socmint run --topic T (--window 1d|3d|7d | --start YYYY-MM-DD --end YYYY-MM-DD [--limit N])
                 [--sources x,telegram] [--match phrase|words]
                 [--out DIR]
  socmint report <run_dir>
  socmint reanalyze <run_dir>
  socmint replay <run_dir> [--port N] [--no-browser]
  socmint doctor [--prefetch]
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

DEFAULT_HOST = "127.0.0.1"
# SOCMINT_BIND exists only so the server can listen inside a Docker container (where
# compose publishes it on the host's 127.0.0.1). Any other value is ignored.
ALLOWED_BINDS = ("127.0.0.1", "0.0.0.0")
# Mirrors pipeline.WINDOW_PRESETS (kept literal so building the parser imports nothing
# heavy; tests/test_cli.py checks they match).
WINDOW_PRESET_NAMES = ("1d", "3d", "7d")


def bind_host(env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    v = (env.get("SOCMINT_BIND") or DEFAULT_HOST).strip()
    if v not in ALLOWED_BINDS:
        print(f"[socmint] ignoring SOCMINT_BIND={v!r}; binding {DEFAULT_HOST}", file=sys.stderr)
        return DEFAULT_HOST
    return v


def find_free_port(host: str = DEFAULT_HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _date(s: str) -> Any:
    from datetime import date

    try:
        return date.fromisoformat(s[:10])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid date {s!r} (use YYYY-MM-DD)") from e


def _port(s: str) -> int:
    p = int(s)
    if not 0 < p < 65536:
        raise argparse.ArgumentTypeError("port must be 1-65535")
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="socmint", description="SOCMINT narrative & "
                                 "coordination module (run with no command to open the GUI)")
    sub = ap.add_subparsers(dest="command")

    g = sub.add_parser("gui", help="open the local browser GUI (default)")
    g.add_argument("--port", type=_port, default=None)
    g.add_argument("--no-browser", action="store_true")
    g.add_argument("--run", type=Path, default=None,
                   help="run directory to open first (default: data/sample_run)")

    r = sub.add_parser("run", help="collect + analyse a topic over a date window")
    r.add_argument("--topic", required=True)
    win = r.add_mutually_exclusive_group(required=True)
    win.add_argument("--window", choices=tuple(WINDOW_PRESET_NAMES),
                     help="last 1, 3 or 7 days ending now (UTC), with that preset's "
                          "per-source post cap (1d 150, 3d 300, 7d 500); "
                          "instead of --start/--end")
    win.add_argument("--start", type=_date, help="first UTC day (needs --end)")
    r.add_argument("--end", type=_date, help="last UTC day, inclusive (with --start)")
    r.add_argument("--sources", default=None,
                   help="comma-separated (default: x,telegram,youtube,linkedin_ads)")
    r.add_argument("--limit", type=int, default=None,
                   help="soft per-source post cap for a --start/--end run (default: 70 per "
                        "day, at least 500, at most 2000); a --window preset uses its own cap")
    r.add_argument("--match", choices=("phrase", "words"), default="phrase",
                   help="multi-word topic: exact phrase (default) or all words anywhere; "
                        "topics with search operators (quotes, OR, -word, from:) are sent "
                        "as written")
    r.add_argument("--out", type=Path, default=None, help="parent dir for the run (default runs/)")

    rep = sub.add_parser("report", help="re-generate exports + report.html from disk (offline)")
    rep.add_argument("run_dir", type=Path)

    ra = sub.add_parser("reanalyze", help="re-run only the analysis (narratives, "
                        "coordination, origin, claims, brief) on a saved run's posts, then "
                        "exports + report; no collection (uses the configured LLM/evidence "
                        "keys)")
    ra.add_argument("run_dir", type=Path)

    rp = sub.add_parser("replay", help="open a saved run read-only (no keys, no network)")
    rp.add_argument("run_dir", type=Path)
    rp.add_argument("--port", type=_port, default=None)
    rp.add_argument("--no-browser", action="store_true")

    dl = sub.add_parser("delete", help="permanently delete a run (directory under runs/)")
    dl.add_argument("run", help="run id (e.g. 2026-11-18-bank-a1b2c3) or path to its directory")
    dl.add_argument("--yes", action="store_true", help="do not ask for confirmation")

    d = sub.add_parser("doctor", help="check credentials and what will run or be skipped")
    d.add_argument("--prefetch", action="store_true", help="download the embedding model")
    return ap


# ----------------------------------------------------------------------------- gui
def default_initial_run() -> Path | None:
    from socmint.config import sample_run_dir

    d = sample_run_dir()
    return d if (d / "run.json").is_file() else None


def serve(initial_run: Path | None, *, read_only: bool, port: int | None,
          open_browser: bool) -> int:
    import uvicorn

    from socmint.config import ensure_config_file
    from socmint.server.app import create_app

    if not read_only:
        ensure_config_file()
    host = bind_host()
    port = port or find_free_port(DEFAULT_HOST if host == DEFAULT_HOST else host)
    url = f"http://127.0.0.1:{port}/"
    app = create_app(initial_run=initial_run, read_only=read_only)
    mode = "read-only replay" if read_only else "GUI"
    print(f"[socmint] {mode} at {url}  (Ctrl+C to stop)", flush=True)
    if host != DEFAULT_HOST:
        print(f"[socmint] listening on {host} (SOCMINT_BIND); publish only to 127.0.0.1",
              flush=True)
    if open_browser:
        threading.Timer(1.0, _open, args=(url,)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


def _open(url: str) -> None:
    try:
        if not webbrowser.open(url):
            print(f"[socmint] could not open a browser; visit {url}", flush=True)
    except Exception:  # noqa: BLE001
        print(f"[socmint] could not open a browser; visit {url}", flush=True)


def _require_run_dir(d: Path) -> Path | None:
    if not (d / "run.json").is_file():
        print(f"[socmint] {d} is not a run directory (no run.json)", file=sys.stderr)
        return None
    return d.resolve()


# ----------------------------------------------------------------------------- run
def print_progress(ev: dict[str, Any]) -> None:
    msg = ev.get("message")
    if msg:
        print(f"[socmint] {msg}", flush=True)


def status_table(rows: list[dict[str, Any]]) -> str:
    """Per source × query status, so an empty source is always explained."""
    if not rows:
        return "(no source status recorded)"
    head = ("source", "state", "posts", "reason")
    data = [(str(r.get("source", "")), str(r.get("state", "")), str(r.get("count", "")),
             str(r.get("reason") or "")) for r in rows]
    w = [max(len(head[i]), *(len(d[i]) for d in data)) for i in range(3)]
    lines = ["  ".join(h.ljust(w[i]) if i < 3 else h for i, h in enumerate(head))]
    for d in data:
        lines.append("  ".join(c.ljust(w[i]) if i < 3 else c for i, c in enumerate(d)))
    return "\n".join(lines)


def run_window(args: argparse.Namespace) -> tuple[Any, Any, int, dict[str, Any]]:
    """(start, end, limit, options) for `socmint run`: a --window preset (exact bounds
    ending now, the preset's cap) or a --start/--end custom range (whole UTC days).
    Raises pipeline.RunInputError for a bad combination."""
    from socmint import pipeline

    if args.window:
        if args.end is not None:
            raise pipeline.RunInputError("--end cannot be combined with --window")
        if args.limit is not None:
            raise pipeline.RunInputError("--limit cannot be combined with --window (each "
                                         "preset has its own per-source cap)")
        start, end = pipeline.preset_window(args.window)
        return start, end, pipeline.PRESET_LIMITS[args.window], {"window_preset": args.window}
    if args.end is None:
        raise pipeline.RunInputError("--start needs --end (or use --window 1d|3d|7d)")
    limit = (pipeline.custom_limit(pipeline.window_days(args.start, args.end))
             if args.limit is None else args.limit)
    return args.start, args.end, limit, {"window_preset": pipeline.CUSTOM_PRESET}


def cmd_run(args: argparse.Namespace) -> int:
    from socmint import pipeline
    from socmint.runstore import read_json

    sources = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
    try:
        start, end, limit, win_opts = run_window(args)
        run_dir = pipeline.run(args.topic, start, end, sources, limit=limit,
                               options={"match": args.match, **win_opts},
                               progress=print_progress, out_root=args.out)
    except pipeline.RunInputError as e:
        print(f"[socmint] invalid input: {e}", file=sys.stderr)
        return 2
    meta = read_json(run_dir / "run.json", {}) or {}
    print(f"\nrun dir: {run_dir}\nstatus: {meta.get('status')}"
          f"{' — ' + meta['error'] if meta.get('error') else ''}")
    print(status_table(read_json(run_dir / "source_status.json", []) or []))
    return 0  # skipped/failed sources are explained above; they do not fail the command


def cmd_report(args: argparse.Namespace) -> int:
    d = _require_run_dir(args.run_dir)
    if d is None:
        return 1
    from socmint import export, report

    export.write_exports(d)
    out = report.build_report(d)
    print(f"[socmint] exports + report written: {out or d / 'report.html'}")
    return 0


def cmd_reanalyze(args: argparse.Namespace) -> int:
    d = _require_run_dir(args.run_dir)
    if d is None:
        return 1
    from socmint import pipeline
    from socmint.runstore import read_json, run_in_progress

    if run_in_progress(d):
        print("[socmint] that run is still being written; wait for it to finish",
              file=sys.stderr)
        return 1
    try:
        pipeline.reanalyze(d, progress=print_progress)
    except pipeline.RunInputError as e:
        print(f"[socmint] {e}", file=sys.stderr)
        return 1
    meta = read_json(d / "run.json", {}) or {}
    errors = (meta.get("options") or {}).get("analysis_errors") or {}
    print(f"\nrun dir: {d}\nstatus: {meta.get('status')}"
          f"{' — ' + meta['error'] if meta.get('error') else ''}")
    for step, err in errors.items():
        print(f"  analysis step {step} failed: {err}")
    return 0 if meta.get("status") != "failed" else 1


def cmd_delete(args: argparse.Namespace) -> int:
    from socmint.config import runs_dir
    from socmint.runstore import RunDeleteError, delete_run, read_json, run_in_progress

    root = runs_dir()
    cand = Path(args.run)
    d = cand if (cand / "run.json").is_file() else root / args.run
    if not (d / "run.json").is_file():
        print(f"[socmint] no run '{args.run}' (looked for {d})", file=sys.stderr)
        return 1
    if run_in_progress(d):
        print("[socmint] that run is still being written; wait for it to finish",
              file=sys.stderr)
        return 1
    meta = read_json(d / "run.json", {}) or {}
    label = f"{meta.get('topic', '?')!r} ({d.name})"
    if not args.yes:
        if not sys.stdin.isatty():
            print("[socmint] refusing to delete without --yes (no terminal to confirm)",
                  file=sys.stderr)
            return 1
        ans = input(f"Delete run {label} permanently? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("[socmint] not deleted")
            return 1
    try:
        gone = delete_run(d, root)
    except RunDeleteError as e:
        print(f"[socmint] {e}", file=sys.stderr)
        return 1
    print(f"[socmint] deleted {gone}")
    return 0


# ---------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cmd = args.command or "gui"
    if cmd == "gui":
        run_arg = getattr(args, "run", None)
        initial = _require_run_dir(run_arg) if run_arg else default_initial_run()
        if run_arg and initial is None:
            return 1
        return serve(initial, read_only=False, port=getattr(args, "port", None),
                     open_browser=not getattr(args, "no_browser", False))
    if cmd == "run":
        return cmd_run(args)
    if cmd == "report":
        return cmd_report(args)
    if cmd == "reanalyze":
        return cmd_reanalyze(args)
    if cmd == "replay":
        d = _require_run_dir(args.run_dir)
        if d is None:
            return 1
        return serve(d, read_only=True, port=args.port, open_browser=not args.no_browser)
    if cmd == "delete":
        return cmd_delete(args)
    if cmd == "doctor":
        from socmint.doctor import main as doctor_main

        return doctor_main(prefetch=args.prefetch)
    return 2


if __name__ == "__main__":
    sys.exit(main())
