"""Local GUI backend (docs/SPEC.md §3.2): JSON API + the bundled frontend in socmint/web/.

Security posture (the server can read and write credentials):
- the CLI binds it to 127.0.0.1 only (SOCMINT_BIND=0.0.0.0 exists solely for Docker,
  where the compose file publishes the port on the host's 127.0.0.1);
- no CORS; a Host-header allowlist blocks DNS-rebinding, and state-changing requests
  with a foreign Origin are refused (CSRF from other web pages);
- settings are returned masked, never in clear; every error message is redacted;
- raw provenance payloads are served path-traversal-safe and sandboxed (CSP) so a stored
  HTML payload can never run script against this API.
- read_only (replay) mode blocks every write and every network-touching endpoint.
"""

from __future__ import annotations

import io
import os
import re
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from socmint import __version__
from socmint.collect.window import MATCH_MODES, MATCH_PHRASE
from socmint.config import (
    KNOWN_KEYS,
    SECRET_KEYS,
    has,
    load_config,
    masked,
    redact,
    run_time_budget,
    runs_dir,
    sample_run_dir,
    save_config,
)
from socmint.legacy import upgrade_bundle
from socmint.runstore import (
    STALE_RUN_SECONDS,
    last_write,
    list_runs,
    load_run,
    read_json,
    writer_alive,
)
from socmint.server.jobs import COLLECT_STAGES, JobBusy, JobManager, JobStartError
from socmint.util.log import get_logger

log = get_logger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# Files a run exposes for download. Anything else under a run dir is NOT served here.
DOWNLOADABLE: dict[str, str] = {
    "posts.csv": "text/csv; charset=utf-8",
    "posts.jsonl": "application/x-ndjson",
    "narratives.json": "application/json",
    "clusters.json": "application/json",
    "origin.json": "application/json",
    "claims.json": "application/json",
    "source_status.json": "application/json",
    "brief.json": "application/json",  # run summary (absent in older runs)
    "run.json": "application/json",
    "report.html": "text/html; charset=utf-8",
    # extra flat tables written by socmint.export (served only if present)
    "narratives.csv": "text/csv; charset=utf-8",
    "clusters.csv": "text/csv; charset=utf-8",
    "claims.csv": "text/csv; charset=utf-8",
    "source_status.csv": "text/csv; charset=utf-8",
}
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
MASK_PREFIX = "••••"  # masked() output; never save it back as a value
# Raw payload types served as-is; everything else (html, svg, js...) is sent as text.
RAW_SAFE_TYPES = {
    ".json": "application/json", ".jsonl": "application/x-ndjson", ".txt": "text/plain",
    ".csv": "text/csv", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".mp4": "video/mp4",
}
# Numeric settings accepted from the GUI: key -> (min, max). MAX_WINDOW_DAYS is capped at
# the agreed 30-day maximum (CLAUDE.md §2); a GUI write must never lift it.
NUMERIC_SETTINGS: dict[str, tuple[int, int]] = {
    "MAX_WINDOW_DAYS": (1, 30),
    "RUN_TIME_BUDGET_SECONDS": (0, 86400),  # 0 = no limit
    "YOUTUBE_QUOTA_LIMIT": (1, 1_000_000),
    "TELEGRAM_DISCOVER_MAX": (1, 50),
    "TELEGRAM_UNDERGROUND_MAX": (1, 60),
    "TELEGRAM_API_ID": (1, 2**63 - 1),
    "X_MAX_SCROLLS": (1, 500),
    "LINKEDIN_MAX_SCROLLS": (1, 200),
}
X_PROVIDERS = ("web_session", "twitterapi_io", "official")
LINKEDIN_PROVIDERS = ("browser", "voyager")
VOYAGER_QID_RE = re.compile(r"[A-Za-z0-9_-]+\.[0-9A-Za-z]+")
WINDOW_CAP_DAYS = 30
INTERRUPTED = ("interrupted: no live job owns this run (the server was stopped or crashed "
               "during the run); results may be incomplete")
SANDBOX_HEADERS = {"Content-Security-Policy": "sandbox; default-src 'none'",
                   "X-Content-Type-Options": "nosniff"}


def max_window_days(cfg: dict[str, str]) -> int:
    """MAX_WINDOW_DAYS clamped to 1..30; an invalid value falls back to 30 (never a 500)."""
    try:
        days = int(str(cfg.get("MAX_WINDOW_DAYS") or WINDOW_CAP_DAYS).strip())
    except ValueError:
        return WINDOW_CAP_DAYS
    return max(1, min(days, WINDOW_CAP_DAYS))


def attachment_header(filename: str) -> str:
    """Content-Disposition safe for any-language run ids.

    Header values must be latin-1, but run ids keep Unicode topic words. Send an ASCII
    fallback plus the RFC 5987 `filename*` form (what Starlette's FileResponse does).
    """
    ascii_name = filename.encode("ascii", "ignore").decode() or "export"
    ascii_name = ascii_name.replace('"', "").replace("\\", "")
    return f"attachment; filename=\"{ascii_name}\"; filename*=utf-8''{quote(filename)}"


class RunRequest(BaseModel):
    topic: str
    # Either a window preset ("1d" | "3d" | "7d": the last N days ending now, with the
    # preset's per-source cap; pipeline.WINDOW_PRESETS) or a custom start + end date.
    window: str | None = None
    start: str | None = None
    end: str | None = None
    sources: list[str] = Field(default_factory=list)
    limit: int | None = None
    # Topic match mode (socmint.collect.window): "phrase" (default) or "words".
    match: str | None = None


class PhoneBody(BaseModel):
    phone: str


class SignInBody(BaseModel):
    code: str
    password: str | None = None


def _run_index(initial_run: Path | None, jobs: JobManager) -> dict[str, dict[str, Any]]:
    """run_id -> {path, kind, meta}. Rebuilt per request so new runs appear at once."""
    index: dict[str, dict[str, Any]] = {}
    sample = sample_run_dir().resolve()
    runs_root = runs_dir().resolve()
    roots = [runs_dir(), sample_run_dir()]
    if initial_run is not None:
        roots.append(initial_run)
    for meta in list_runs(*roots):
        path = Path(meta.pop("path")).resolve()
        rid = str(meta.get("run_id") or path.name)
        # "external": a --run directory outside runs/ (viewable, never deletable here)
        kind = ("sample" if path == sample
                else "run" if path.parent == runs_root else "external")
        index.setdefault(rid, {"path": path, "kind": kind, "meta": meta})
    job = jobs.current
    # A cancelled run's directory is deleted; never resurrect it as a phantom "running" row.
    if job and job.run_id and job.run_dir and not job.cancelled and job.run_id not in index:
        index[job.run_id] = {"path": job.run_dir, "kind": "run",
                             "meta": {"run_id": job.run_id, "status": "running"}}
    return index


def create_app(initial_run: Path | None = None, read_only: bool = False,
               jobs: JobManager | None = None) -> FastAPI:
    app = FastAPI(title="SOCMINT", version=__version__, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    jobs = jobs or JobManager()
    app.state.jobs = jobs
    initial = Path(initial_run).resolve() if initial_run else None
    initial_meta = read_json(initial / "run.json", None) if initial else None
    initial_run_id = (initial_meta or {}).get("run_id") or (initial.name if initial_meta else None)
    app_started = time.time()
    delete_lock = threading.Lock()

    # ------------------------------------------------------------- middleware
    @app.middleware("http")
    async def same_origin_writes(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Refuse state-changing requests sent by another web page (CSRF)."""
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin is not None:
            # Same origin means same scheme, host AND port as this server (the Host header,
            # already allowlisted above). Another app on another localhost port is foreign.
            o = urlsplit(origin)
            host = request.headers.get("host", "").lower()
            if not (o.scheme in ("http", "https") and o.netloc.lower() == host
                    and (o.hostname or "") in ALLOWED_HOSTS):  # "null" fails here too
                return JSONResponse({"detail": "cross-origin request refused"}, 403)
        return await call_next(request)

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled server error: %s: %s", type(exc).__name__, exc)
        return JSONResponse({"detail": redact(f"{type(exc).__name__}: {exc}")}, 500)

    def require_writable() -> None:
        if read_only:
            raise HTTPException(403, "read-only replay: settings, logins and new runs are "
                                     "disabled (start the GUI with `socmint gui` instead)")

    def resolve_run(run_id: str) -> Path:
        entry = _run_index(initial, jobs).get(run_id)
        if entry is None:
            raise HTTPException(404, f"unknown run '{run_id}'")
        return Path(entry["path"])

    def dead_reason(run_id: str, run_dir: Path) -> str | None:
        """Why an unfinished run is not actually running, or None if it may be live.

        A failure must never look like a run still in progress (CLAUDE.md §3). A run is
        dead when this process's job for it has ended, or when no job here owns it and
        its files were last written before this server started or long ago (a run in a
        separate `socmint run` process keeps writing, so it stays "running").
        """
        job = jobs.current
        if job and job.run_id == run_id:
            if job.active:
                return None
            return job.error or "run thread ended before the run was finished"
        # A separate `socmint run` process that is still alive owns it, even if it has
        # not written for a while (flood wait, model download) or started before us.
        if writer_alive(run_dir, ignore_pid=os.getpid()):
            return None
        last = last_write(run_dir)
        if last < app_started or time.time() - last > STALE_RUN_SECONDS:
            return INTERRUPTED
        return None

    # ------------------------------------------------------------------ meta
    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        from socmint.doctor import source_readiness
        from socmint.pipeline import (
            CUSTOM_LIMIT,
            CUSTOM_LIMIT_MAX,
            CUSTOM_PER_DAY,
            DEFAULT_PRESET,
            PRESET_LIMITS,
            WINDOW_PRESETS,
        )

        cfg = load_config()
        sources = [{k: r.get(k) for k in ("name", "kind", "needs_credentials",
                                          "credential_keys", "ready", "reason", "description",
                                          "provider", "outcome")}
                   for r in source_readiness(cfg)]
        return {"version": __version__, "read_only": read_only,
                "initial_run_id": (initial_run_id if initial_run_id
                                   and initial_run_id in _run_index(initial, jobs) else None),
                "sources": sources,
                "max_window_days": max_window_days(cfg),
                # New Run window presets (owner decision 2026-11-20)
                "window_presets": [{"id": k, "days": d, "limit": PRESET_LIMITS[k]}
                                   for k, d in WINDOW_PRESETS.items()],
                "default_window": DEFAULT_PRESET, "custom_limit": CUSTOM_LIMIT,
                # custom range default limit = clamp(per_day x days, custom_limit, max)
                "custom_limit_per_day": CUSTOM_PER_DAY, "custom_limit_max": CUSTOM_LIMIT_MAX,
                "time_limit_s": run_time_budget(cfg)}

    # -------------------------------------------------------------- settings
    def settings_payload() -> dict[str, Any]:
        cfg = load_config()
        return {"config": masked(cfg),
                "keys": [{"key": k, "secret": k in SECRET_KEYS, "set": has(cfg, k)}
                         for k in KNOWN_KEYS]}

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        require_writable()
        return settings_payload()

    @app.post("/api/settings")
    def post_settings(body: dict[str, Any] = Body(...)) -> dict[str, Any]:  # noqa: B008
        require_writable()
        unknown = sorted(k for k in body if k not in KNOWN_KEYS)
        if unknown:
            raise HTTPException(400, f"unknown setting(s): {', '.join(unknown)}")
        updates: dict[str, str] = {}
        for k, v in body.items():
            v = "" if v is None else str(v).strip()
            if v.startswith(MASK_PREFIX):
                continue  # the masked placeholder came back unchanged: keep the stored value
            if k in NUMERIC_SETTINGS and v:  # "" clears the key back to its default
                lo, hi = NUMERIC_SETTINGS[k]
                if not v.isdigit() or not lo <= int(v) <= hi:
                    raise HTTPException(400, f"{k} must be a whole number from {lo} to {hi}")
                if k == "RUN_TIME_BUDGET_SECONDS" and 0 < int(v) < 60:
                    raise HTTPException(400, "RUN_TIME_BUDGET_SECONDS must be 0 (no time "
                                             "limit) or at least 60 seconds")
            if k == "TELEGRAM_HAS_PREMIUM" and v and v.lower() not in ("true", "false"):
                raise HTTPException(400, "TELEGRAM_HAS_PREMIUM must be true or false")
            if k == "TELEGRAM_SEARCH_JOINED_CHATS" and v and v.lower() not in ("true", "false"):
                raise HTTPException(400, "TELEGRAM_SEARCH_JOINED_CHATS must be true or false")
            if k == "TELEGRAM_DISCOVER_CHANNELS" and v and v.lower() not in ("true", "false"):
                raise HTTPException(400, "TELEGRAM_DISCOVER_CHANNELS must be true or false")
            if k == "TELEGRAM_UNDERGROUND_DISCOVERY" and v and v.lower() not in ("true", "false"):
                raise HTTPException(400, "TELEGRAM_UNDERGROUND_DISCOVERY must be true or false")
            if k == "X_BROWSER_HEADLESS" and v and v.lower() not in ("true", "false"):
                raise HTTPException(400, "X_BROWSER_HEADLESS must be true or false")
            if k == "LINKEDIN_BROWSER_HEADLESS" and v and v.lower() not in ("true", "false"):
                raise HTTPException(400, "LINKEDIN_BROWSER_HEADLESS must be true or false")
            if k == "LINKEDIN_PROVIDER" and v and v.lower() not in LINKEDIN_PROVIDERS:
                raise HTTPException(400, "LINKEDIN_PROVIDER must be one of "
                                         f"{', '.join(LINKEDIN_PROVIDERS)}")
            if k == "LINKEDIN_VOYAGER_QUERY_ID" and v and not VOYAGER_QID_RE.fullmatch(v):
                raise HTTPException(400, "LINKEDIN_VOYAGER_QUERY_ID must look like "
                                         "voyagerSearchDashClusters.<hash> (copy it from "
                                         "DevTools > Network); empty restores the default")
            if k == "LINKEDIN_USER_AGENT" and any(ord(c) < 32 or ord(c) == 127 for c in v):
                raise HTTPException(400, "LINKEDIN_USER_AGENT must be a single line")
            if k == "X_PROVIDER" and v and v.lower() not in X_PROVIDERS:
                raise HTTPException(400, f"X_PROVIDER must be one of {', '.join(X_PROVIDERS)}")
            updates[k] = v
        save_config(updates)
        return settings_payload()

    # -------------------------------------------------------------- telegram
    def tg(fn: str, **kw: Any) -> Any:
        from socmint.doctor import TelegramAuthUnavailable, telegram_auth_call

        require_writable()
        cfg = load_config()
        try:
            out = telegram_auth_call(fn, cfg, **kw)
        except TelegramAuthUnavailable as e:
            return {"configured": False, "reason": str(e)}
        except Exception as e:  # noqa: BLE001 - e.g. wrong code, flood wait
            return JSONResponse({"detail": redact(f"{type(e).__name__}: {e}", cfg)}, 400)
        if isinstance(out, dict):
            return out
        return {"ok": bool(out)} if isinstance(out, bool) or out is None else {"result": out}

    @app.get("/api/telegram/status")
    def telegram_status() -> Any:
        return tg("status")

    @app.post("/api/telegram/send_code")
    def telegram_send_code(body: PhoneBody) -> Any:
        return tg("send_code", phone=body.phone)

    @app.post("/api/telegram/sign_in")
    def telegram_sign_in(body: SignInBody) -> Any:
        return tg("sign_in", code=body.code, password=body.password)

    @app.post("/api/telegram/logout")
    def telegram_logout() -> Any:
        return tg("logout")

    # ------------------------------------------------------------------ X (web_session)
    def xw(fn: str) -> Any:
        """Call socmint.collect.x_web.<fn>(cfg). Imported lazily: the GUI keeps working
        (and says why) when the module or Playwright is not available."""
        require_writable()
        cfg = load_config()
        try:
            from socmint.collect import x_web  # type: ignore[attr-defined]

            func = getattr(x_web, fn)
        except (ImportError, AttributeError) as e:
            reason = f"X web session unavailable ({type(e).__name__}: {e})"
            return {"configured": False, "authorized": False, "ok": False,
                    "reason": redact(reason, cfg)}
        try:
            out = func(cfg)
        except Exception as e:  # noqa: BLE001 - never leak a cookie in an error
            return JSONResponse({"detail": redact(f"{type(e).__name__}: {e}", cfg)}, 400)
        if not isinstance(out, dict):
            return {"ok": bool(out)}
        for k in ("reason", "error"):
            if isinstance(out.get(k), str):
                out[k] = redact(out[k], cfg)
        return out

    def x_login_error() -> str | None:
        """Why the last X login failed (x_web.login_state()['error']), redacted.

        session_status() only carries the bare login_state string; without this the GUI
        could never say e.g. "no browser: run socmint doctor --prefetch"."""
        mod = sys.modules.get("socmint.collect.x_web")
        fn = getattr(mod, "login_state", None) if mod is not None else None
        if fn is None:
            return None
        try:
            err = (fn() or {}).get("error")
        except Exception:  # noqa: BLE001 - status polling must never fail on this
            return None
        return redact(str(err), load_config()) if err else None

    def x_login_waiting() -> bool:
        # Only consult x_web if already imported: if it was never imported, no login is
        # running in this process (and importing it just to check would be wasteful).
        mod = sys.modules.get("socmint.collect.x_web")
        fn = getattr(mod, "login_state", None) if mod is not None else None
        try:
            return bool(fn and (fn() or {}).get("state") == "waiting")
        except Exception:  # noqa: BLE001
            return False

    @app.get("/api/x/status")
    def x_status() -> Any:
        out = xw("session_status")
        if isinstance(out, dict) and "login_error" not in out:
            out["login_error"] = x_login_error()
        return out

    @app.post("/api/x/login")
    def x_login() -> Any:
        # Non-blocking: x_web opens a visible browser window in a background thread; the
        # GUI then polls /api/x/status while login_state == "waiting".
        # Never drive the owner's X account from two browsers at once (CLAUDE.md §5): a
        # running collection may be using the saved session file this would rewrite.
        require_writable()
        if jobs.busy():
            raise HTTPException(409, "a run is in progress; log in to X after it finishes")
        return xw("start_login")

    @app.post("/api/x/logout")
    def x_logout() -> Any:
        # x_web.logout returns "message" describing what actually happened (deleted or
        # nothing to delete; pasted cookies remain). The server adds nothing of its own.
        return xw("logout")

    # ------------------------------------------------------------------ runs
    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        out = []
        for rid, e in _run_index(initial, jobs).items():
            m = e["meta"]
            status, error = m.get("status"), m.get("error")
            if status == "running":
                dead = dead_reason(rid, Path(e["path"]))
                if dead:
                    status, error = "failed", dead
            out.append({"run_id": rid, "topic": m.get("topic"), "start": m.get("start"),
                        "end": m.get("end"), "status": status, "error": error,
                        "created_at": m.get("created_at"), "counts": m.get("counts") or {},
                        "sources": m.get("sources") or [], "path_kind": e["kind"]})
        out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return out

    @app.post("/api/runs")
    def start_run(req: RunRequest) -> dict[str, Any]:
        from socmint.collect import DEFAULT_SOURCES, available
        from socmint.pipeline import (
            CUSTOM_PRESET,
            PRESET_LIMITS,
            WINDOW_PRESETS,
            RunInputError,
            custom_limit,
            parse_window,
            preset_window,
            validate_inputs,
            window_days,
        )

        require_writable()
        cfg = load_config()
        win = (req.window or "").strip().lower()
        if win:
            if win not in WINDOW_PRESETS:
                raise HTTPException(400, "window must be one of: "
                                         f"{', '.join(WINDOW_PRESETS)}")
            if req.start or req.end:
                raise HTTPException(400, "send either a window preset or start + end "
                                         "dates, not both")
            if req.limit is not None:
                raise HTTPException(400, "limit cannot be combined with a window preset "
                                         "(each preset has its own per-source cap)")
            start, end = preset_window(win)
        else:
            if not req.start or not req.end:
                raise HTTPException(400, "send a window preset (1d, 3d, 7d) or both a "
                                         "start and an end date")
            try:
                start, end = parse_window(req.start, req.end)
            except ValueError as e:
                raise HTTPException(400, f"invalid date: {e}") from e
        try:
            validate_inputs(req.topic, start, end, max_window_days(cfg))
        except RunInputError as e:
            raise HTTPException(400, str(e)) from e
        sources = [s.strip() for s in req.sources if s.strip()] or list(DEFAULT_SOURCES)
        unknown = [s for s in sources if s not in available()]
        if unknown:
            raise HTTPException(400, f"unknown source(s): {', '.join(unknown)}")
        if "x" in sources and x_login_waiting():
            raise HTTPException(409, "an X login is in progress; finish it (or close the "
                                     "login window) before starting a run")
        if win:
            limit = PRESET_LIMITS[win]
        else:
            limit = req.limit if req.limit is not None else custom_limit(window_days(start, end))
        if not 1 <= limit <= 10000:
            raise HTTPException(400, "limit must be between 1 and 10000")
        match = (req.match or MATCH_PHRASE).strip().lower()
        if match not in MATCH_MODES:
            raise HTTPException(400, f"match must be one of: {', '.join(MATCH_MODES)}")
        try:
            job = jobs.start(topic=req.topic, start=start, end=end, sources=sources,
                             limit=limit, out_root=runs_dir(), cfg=cfg,
                             options={"match": match,
                                      "window_preset": win or CUSTOM_PRESET})
        except JobBusy as e:
            raise HTTPException(409, str(e)) from e
        except JobStartError as e:
            raise HTTPException(500, redact(str(e), cfg)) from e
        return {"run_id": job.run_id}

    @app.post("/api/runs/{run_id}/stop")
    def stop_run_route(run_id: str) -> dict[str, Any]:
        """Cancel the active collection run: collectors stop at once and the whole run is
        abandoned — no analysis, no report, and the partial run is deleted."""
        require_writable()
        if not jobs.stop(run_id):
            raise HTTPException(409, "this run is not collecting in this server (collection "
                                     "has finished, it is being re-analysed, or it is not "
                                     "active)")
        return {"run_id": run_id, "stopping": True, "cancel": True}

    @app.post("/api/runs/{run_id}/reanalyze")
    def reanalyze_route(run_id: str) -> dict[str, Any]:
        """Re-run only the analysis of a saved run (no collection), like
        `socmint reanalyze`. Uses the configured LLM / evidence keys. The bundled sample
        is refused: it would overwrite the shipped analysis."""
        from socmint.runstore import run_in_progress

        require_writable()
        entry = _run_index(initial, jobs).get(run_id)
        if entry is None:
            raise HTTPException(404, f"unknown run '{run_id}'")
        if entry["kind"] == "sample":
            raise HTTPException(403, "the bundled sample run is not re-analysed in place; "
                                     "copy data/sample_run into the runs folder first")
        d = Path(entry["path"])
        if not (d / "run.json").is_file() or not (d / "posts.jsonl").is_file():
            raise HTTPException(400, "this run has no saved posts to re-analyse")
        if run_in_progress(d, ignore_pid=os.getpid()) or (
                (read_json(d / "run.json", {}) or {}).get("status") == "running"):
            raise HTTPException(409, "this run is still being written; wait for it to end")
        try:
            jobs.start_reanalyze(d, load_config())
        except JobBusy as e:
            raise HTTPException(409, str(e)) from e
        return {"run_id": run_id}

    @app.delete("/api/runs/{run_id}")
    def delete_run_route(run_id: str) -> dict[str, Any]:
        """Permanently delete a run's directory (results, raw payloads, logs)."""
        from socmint.runstore import RunDeleteError, delete_run, run_in_progress

        require_writable()
        with delete_lock:  # two tabs deleting the same run: second one gets a clean 404
            entry = _run_index(initial, jobs).get(run_id)
            if entry is None:
                raise HTTPException(404, f"unknown run '{run_id}'")
            if entry["kind"] == "sample":
                raise HTTPException(403, "the bundled sample run cannot be deleted")
            if entry["kind"] != "run":
                raise HTTPException(403, "only runs inside the runs/ folder can be deleted "
                                         "from the GUI")
            d = Path(entry["path"])
            job = jobs.current
            if job and job.run_id == run_id:
                if job.active:
                    raise HTTPException(409, "this run is still in progress; wait for it "
                                             "to finish")
            elif run_in_progress(d, ignore_pid=os.getpid()):
                # Not ours: a separate process may still be writing it. Unlike the status
                # display, a delete never trusts "written before this server started".
                raise HTTPException(409, "this run is still being written (by another "
                                         "socmint process?); wait for it to finish")
            try:
                delete_run(d, runs_dir())
            except RunDeleteError as e:
                raise HTTPException(403, str(e)) from e
            except FileNotFoundError as e:
                raise HTTPException(404, f"run '{run_id}' was already deleted") from e
            except OSError as e:
                log.error("delete of run %s failed: %s", run_id, e)
                raise HTTPException(500, f"could not fully delete run '{run_id}' "
                                         f"({type(e).__name__}); it may be partly deleted. "
                                         f"Remove {d.name} from the runs folder by hand."
                                    ) from e
        log.info("deleted run %s", run_id)
        return {"deleted": run_id}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        d = resolve_run(run_id)
        entry = _run_index(initial, jobs).get(run_id) or {}
        body = {**upgrade_bundle(load_run(d)), "run_id": run_id,
                "path_kind": entry.get("kind", "run")}
        run = body.get("run")
        if isinstance(run, dict) and run.get("status") == "running":
            dead = dead_reason(run_id, d)
            if dead:
                body["run"] = {**run, "status": "failed", "error": run.get("error") or dead}
        return body

    @app.get("/api/runs/{run_id}/progress")
    def get_progress(run_id: str) -> Any:
        # A cancelled run's directory is deleted, so report the cancellation from the job
        # before resolve_run (which would 404 on the missing directory).
        job0 = jobs.current
        if job0 and job0.run_id == run_id and job0.cancelled:
            return {"stage": "cancelled", "done": True, "cancelled": True,
                    "sources": {}, "log": []}
        d = resolve_run(run_id)
        prog = read_json(d / "progress.json", None)
        job = jobs.current
        if prog is not None and prog.get("done") and job and job.run_id == run_id \
                and job.active:
            # a re-analysis just started on a finished run: its old progress.json still
            # says done until the job writes its first event
            return {**prog, "stage": "reanalyze", "done": False, "error": None}
        if prog is not None and not prog.get("done"):
            dead = dead_reason(run_id, d)
            if dead:  # never let a dead run spin as "in progress" forever
                return {**prog, "done": True, "interrupted": True,
                        "error": prog.get("error") or dead}
            # "Stop scan" is offered while this server's run job is collecting.
            if job and job.run_id == run_id and job.active and job.kind == "run":
                collecting = prog.get("stage") in COLLECT_STAGES
                return {**prog, "stopping": job.stop.is_set() and collecting,
                        "can_stop": not job.stop.is_set() and collecting}
        if prog is not None:
            return prog
        job = jobs.current
        if job and job.run_id == run_id:
            if job.active:
                return {"stage": "starting", "done": False, "sources": {}, "log": []}
            if job.error:
                return {"done": True, "error": job.error}
        return {"done": True}

    @app.get("/api/runs/{run_id}/files/{name}")
    def get_file(run_id: str, name: str, inline: bool = False) -> FileResponse:
        if name not in DOWNLOADABLE:
            raise HTTPException(404, f"'{name}' is not a downloadable run file")
        p = resolve_run(run_id) / name
        if not p.is_file():
            raise HTTPException(404, f"{name} has not been written for this run")
        disposition = "inline" if inline and name == "report.html" else "attachment"
        return FileResponse(p, media_type=DOWNLOADABLE[name], filename=f"{run_id}-{name}",
                            content_disposition_type=disposition)

    @app.get("/api/runs/{run_id}/export.zip")
    def export_zip(run_id: str) -> Response:
        d = resolve_run(run_id)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in DOWNLOADABLE:
                if (d / name).is_file():
                    zf.write(d / name, arcname=f"{run_id}/{name}")
        return Response(buf.getvalue(), media_type="application/zip", headers={
            "Content-Disposition": attachment_header(f"{run_id}.zip")})

    @app.get("/api/runs/{run_id}/raw/{path:path}")
    def get_raw(run_id: str, path: str) -> FileResponse:
        raw_root = (resolve_run(run_id) / "raw").resolve()
        target = (raw_root / path).resolve()
        # Path-traversal guard: the resolved file must stay inside run_dir/raw.
        if not target.is_relative_to(raw_root) or target == raw_root:
            raise HTTPException(404, "not found")
        if not target.is_file():
            raise HTTPException(404, "not found")
        media = RAW_SAFE_TYPES.get(target.suffix.lower(), "text/plain; charset=utf-8")
        return FileResponse(target, media_type=media, headers=SANDBOX_HEADERS)

    # ---------------------------------------------------------------- frontend
    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> Response:
        page = WEB_DIR / "index.html"
        if page.is_file():
            return FileResponse(page, media_type="text/html; charset=utf-8",
                                headers={"Cache-Control": "no-cache"})
        return HTMLResponse(
            "<!doctype html><meta charset=utf-8><title>SOCMINT</title>"
            "<p>The frontend bundle (socmint/web/index.html) is missing. "
            "The API is running: <a href='/api/meta'>/api/meta</a>, "
            "<a href='/api/runs'>/api/runs</a>.</p>")

    return app

