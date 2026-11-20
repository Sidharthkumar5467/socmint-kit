"""`socmint doctor`: what will run, what will be skipped, and why (CLAUDE.md §4).

Never prints a credential value — only whether each key is set. The same readiness
logic backs the GUI (`GET /api/meta`), so the CLI and the GUI always agree about which
sources will run.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from socmint.config import config_path, has, load_config, redact, runs_dir

# Analysis / evidence keys: not sources, but they change what a run can do.
ANALYSIS_KEYS: tuple[tuple[str, str], ...] = (
    ("ANTHROPIC_API_KEY", "LLM narrative labels, claim extraction and stance"),
    ("GOOGLE_FACTCHECK_API_KEY", "Google Fact Check evidence lookups"),
    ("SERPAPI_KEY", "news/evidence search lookups (serpapi.com)"),
    ("SERPER_API_KEY", "news/evidence search lookups (serper.dev)"),
)


# ------------------------------------------------------------------ source readiness
def source_readiness(cfg: dict[str, str]) -> list[dict[str, Any]]:
    """One row per registered source: will it run with the current config, and why not.

    A source whose module cannot be imported is reported (not hidden) with the error,
    so an empty result can never be mistaken for "no posts".
    """
    from socmint.collect import DEFAULT_SOURCES, available, get_collector

    names = list(DEFAULT_SOURCES) + [n for n in sorted(available()) if n not in DEFAULT_SOURCES]
    rows: list[dict[str, Any]] = []
    for name in names:
        row: dict[str, Any] = {"name": name, "kind": None, "needs_credentials": False,
                               "credential_keys": [], "keys_set": {}, "ready": False,
                               "reason": "", "description": ""}
        try:
            caps = get_collector(name).capabilities()
        except Exception as e:  # noqa: BLE001 - report, never crash the doctor/GUI
            row["reason"] = redact(f"collector unavailable ({type(e).__name__}: {e})", cfg)
            row["outcome"] = "fail"
            rows.append(row)
            continue
        keys = list(caps.credential_keys)
        row.update(kind=caps.kind, needs_credentials=caps.needs_credentials,
                   credential_keys=keys, description=caps.description,
                   keys_set={k: has(cfg, k) for k in keys})
        missing = [k for k in keys if not has(cfg, k)]
        if name == "x" and x_provider(cfg) == WEB_SESSION:
            row.update(_x_web_session_readiness(cfg))
        elif name == "linkedin":
            row.update(linkedin_readiness(cfg))
        elif caps.needs_credentials and missing:
            row["reason"] = f"will skip: missing {', '.join(missing)} (skipped_no_credentials)"
        elif name == "x" and x_provider(cfg) == "official" and _x_bearer_problem(cfg):
            row["reason"] = f"will fail: {_x_bearer_problem(cfg)}"
            row["outcome"] = "fail"
        else:
            row["ready"] = True
            row["reason"] = "ready" if keys else "ready (no credentials needed)"
        row.setdefault("outcome", "run" if row["ready"] else "skip")
        rows.append(row)
    return rows


def _x_bearer_problem(cfg: dict[str, str]) -> str | None:
    from socmint.collect.x import official_bearer_problem

    return official_bearer_problem(cfg.get("X_BEARER_TOKEN"))


# ------------------------------------------------------------------- X web session
WEB_SESSION = "web_session"
X_LOGIN_HINT = "log in to X from Settings or paste X_AUTH_TOKEN + X_CT0"
# Standard Google Chrome locations (Playwright channel="chrome" looks in the same places).
CHROME_PATHS: tuple[str, ...] = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/opt/google/chrome/chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "%LOCALAPPDATA%\\Google\\Chrome\\Application\\chrome.exe",
)


def x_provider(cfg: dict[str, str]) -> str:
    # Same fallback as config.DEFAULTS: web_session is the default provider.
    return (cfg.get("X_PROVIDER") or WEB_SESSION).strip().lower()


def find_chrome() -> str | None:
    """Path of an installed Google Chrome, or None. Filesystem only, never launches it."""
    for raw in CHROME_PATHS:
        path = Path(os.path.expandvars(os.path.expanduser(raw)))
        if "%" not in str(path) and path.is_file():
            return str(path)
    for exe in ("google-chrome", "google-chrome-stable"):
        found = shutil.which(exe)
        if found:
            return found
    return None


def playwright_browsers_dir() -> Path:
    env = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if env and env != "0":
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ms-playwright"
    if sys.platform.startswith("win"):
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "ms-playwright"
    return Path.home() / ".cache" / "ms-playwright"


def playwright_chromium_installed() -> bool:
    d = playwright_browsers_dir()
    try:
        return any(p.is_dir() and p.name.startswith("chromium") for p in d.iterdir())
    except OSError:
        return False


def x_browser_status() -> tuple[str | None, str]:
    """(browser, detail): "chrome" | "chromium" | None. No browser is launched."""
    chrome = find_chrome()
    if chrome:
        return "chrome", f"Google Chrome found ({chrome})"
    try:
        import playwright  # noqa: F401
    except Exception:  # noqa: BLE001
        return None, "playwright is not installed (pip install -e .)"
    if playwright_chromium_installed():
        return "chromium", f"Playwright Chromium installed ({playwright_browsers_dir()})"
    return None, ("no browser: install Google Chrome, or run `socmint doctor --prefetch` "
                  "(installs Playwright Chromium)")


def _x_web() -> Any:
    from socmint.collect import x_web  # type: ignore[attr-defined]

    return x_web


def x_session_status(cfg: dict[str, str]) -> dict[str, Any]:
    """x_web.session_status(cfg), or a not-authorized stand-in if x_web is unavailable.

    session_status makes no network call (contract), so this is cheap and offline.
    Never returns a cookie value: only flags, the screen name and the session source.
    """
    try:
        st = _x_web().session_status(cfg)
    except (ImportError, AttributeError) as e:
        return {"configured": False, "authorized": False, "user": None, "source": None,
                "browser": None, "login_state": "idle",
                "reason": f"X web session module unavailable ({type(e).__name__})"}
    except Exception as e:  # noqa: BLE001 - report, never crash the doctor/GUI
        return {"configured": False, "authorized": False, "user": None, "source": None,
                "browser": None, "login_state": "idle",
                "reason": redact(f"X session check failed: {type(e).__name__}: {e}", cfg)}
    if not isinstance(st, dict):
        st = {"authorized": bool(st)}
    keep = ("configured", "authorized", "user", "source", "session_path", "browser",
            "login_state", "reason")
    out = {k: st.get(k) for k in keep}
    out["authorized"] = bool(out["authorized"])
    if out["reason"]:
        out["reason"] = redact(str(out["reason"]), cfg)
    return out


def _x_web_session_readiness(cfg: dict[str, str]) -> dict[str, Any]:
    """Row fields for X with X_PROVIDER=web_session (the owner's own logged-in account)."""
    st = x_session_status(cfg)
    browser, browser_detail = x_browser_status()
    browser = st.get("browser") or browser
    who = f" as @{str(st['user']).lstrip('@')}" if st.get("user") else ""
    src = f" via {st['source']}" if st.get("source") else ""
    fields: dict[str, Any] = {
        "needs_credentials": False, "provider": WEB_SESSION,
        "keys_set": {"x_session": st["authorized"]},
        "x_session": {**st, "browser": browser, "browser_detail": browser_detail},
    }
    if not st["authorized"]:
        fields.update(ready=False, outcome="skip",
                      reason=f"will skip: {X_LOGIN_HINT} (skipped_no_credentials)")
    elif not browser:
        # The collector returns `failed` (no browser) here, not a skip: say so.
        fields.update(ready=False, outcome="fail",
                      reason=f"will fail: X session authorized{who}, but {browser_detail}")
    else:
        fields.update(ready=True, outcome="run",
                      reason=f"ready: X session{who}{src}, browser {browser}")
    return fields


def ensure_x_browser(cfg: dict[str, str]) -> tuple[bool | None, str]:
    """--prefetch: x_web.ensure_browser() (installs Chromium only when Chrome is absent)."""
    try:
        fn = _x_web().ensure_browser
    except (ImportError, AttributeError) as e:
        return None, f"X web session module unavailable ({type(e).__name__})"
    try:
        out = fn()
    except Exception as e:  # noqa: BLE001
        return False, redact(f"{type(e).__name__}: {e}", cfg)
    if not isinstance(out, dict):
        return bool(out), "ensure_browser finished"
    msg = str(out.get("message") or "")
    if out.get("browser"):
        msg = f"{out['browser']}: {msg}" if msg else str(out["browser"])
    return bool(out.get("ok")), redact(msg, cfg)


# ------------------------------------------------------------------ LinkedIn posts
def linkedin_readiness(cfg: dict[str, str]) -> dict[str, Any]:
    """Row fields for "linkedin" (posts via the owner's own logged-in session).

    Never prints a cookie value: only whether LINKEDIN_COOKIE is set, how many cookies it
    holds and whether li_at (the login cookie) is among them.
    """
    from socmint.collect.linkedin_posts import has_li_at, parse_cookie_header

    raw = str(cfg.get("LINKEDIN_COOKIE") or "").strip()
    if not raw:
        return {"ready": False, "outcome": "skip",
                "reason": "will skip: missing LINKEDIN_COOKIE (skipped_no_credentials)"}
    if not has_li_at(raw):
        warn = ("LINKEDIN_COOKIE has no li_at cookie (the login cookie): paste the full "
                "Cookie header of a logged-in linkedin.com tab, or the li_at value alone")
        return {"ready": False, "outcome": "skip", "warning": warn,
                "reason": f"will skip: {warn} (skipped_no_credentials)"}
    from socmint.collect.linkedin_voyager import NO_JSESSIONID, jsessionid, provider, query_id

    n = len(parse_cookie_header(raw))
    what = f"{n} cookie(s) incl. li_at" if n > 1 else "li_at only"
    prov = provider(cfg)
    if prov == "voyager":
        if not jsessionid(raw):
            return {"ready": False, "outcome": "skip", "provider": prov,
                    "warning": NO_JSESSIONID,
                    "reason": f"will skip: provider voyager: {NO_JSESSIONID} "
                              "(skipped_no_credentials)"}
        return {"ready": True, "outcome": "run", "provider": prov,
                "reason": f"ready: your own LinkedIn account ({what}), provider voyager "
                          f"(direct Voyager search API calls, queryId {query_id(cfg)})"}
    browser, detail = x_browser_status()
    if not browser:
        return {"ready": False, "outcome": "fail", "provider": prov,
                "reason": f"will fail: LinkedIn session set ({what}), provider browser, "
                          f"but {detail}"}
    return {"ready": True, "outcome": "run", "provider": prov,
            "reason": f"ready: your own LinkedIn account ({what}), provider browser, "
                      f"browser {browser}"}


# --------------------------------------------------------------- telegram auth adapter
class TelegramAuthUnavailable(RuntimeError):
    pass


def telegram_auth_call(fn_name: str, cfg: dict[str, str], **kwargs: Any) -> Any:
    """Call socmint.collect.telegram_auth.<fn_name>, tolerating its exact signature.

    Passes `cfg` when the function accepts a cfg/config parameter, runs coroutines to
    completion (callers must not already be inside an event loop — the server calls this
    from sync endpoints, which run in a worker thread). Raises TelegramAuthUnavailable
    when the module or function does not exist.
    """
    try:
        from socmint.collect import telegram_auth  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        raise TelegramAuthUnavailable(f"telegram_auth not available ({type(e).__name__})") from e
    fn = getattr(telegram_auth, fn_name, None)
    if fn is None:
        raise TelegramAuthUnavailable(f"telegram_auth.{fn_name} not implemented")
    params = inspect.signature(fn).parameters
    call_kw = {k: v for k, v in kwargs.items() if k in params or _has_var_kw(params)}
    for alias in ("cfg", "config"):
        if alias in params:
            call_kw[alias] = cfg
            break
    out = fn(**call_kw)
    if inspect.isawaitable(out):
        out = asyncio.run(_await(out))
    return out


def _has_var_kw(params: Any) -> bool:
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


async def _await(aw: Any) -> Any:
    return await aw


def telegram_session_status(cfg: dict[str, str]) -> tuple[bool | None, str]:
    """(authorized?, detail). None = could not be determined."""
    if not has(cfg, "TELEGRAM_API_ID", "TELEGRAM_API_HASH"):
        return None, "not configured (TELEGRAM_API_ID / TELEGRAM_API_HASH missing)"
    try:
        st = telegram_auth_call("status", cfg)
    except TelegramAuthUnavailable as e:
        return None, str(e)
    except Exception as e:  # noqa: BLE001
        return None, redact(f"status check failed: {type(e).__name__}: {e}", cfg)
    if isinstance(st, dict):
        auth = st.get("authorized")
        detail = st.get("reason") or st.get("detail") or ("authorized" if auth else
                                                           "not authorized: log in from Settings")
        return (bool(auth) if auth is not None else None), redact(str(detail), cfg)
    return bool(st), "authorized" if st else "not authorized: log in from Settings"


# ------------------------------------------------------------------- embedding model
def embedding_model_cached(cfg: dict[str, str]) -> tuple[bool | None, str]:
    """Is the sentence-transformers model in the local HF cache? (no network)."""
    try:
        from socmint.analyze.embed import model_name
    except Exception as e:  # noqa: BLE001
        return None, f"embedding module unavailable ({type(e).__name__})"
    name = model_name(cfg)
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(name, "config.json")
    except Exception as e:  # noqa: BLE001
        return None, f"{name}: cache check failed ({type(e).__name__})"
    if isinstance(hit, str) and hit:
        return True, f"{name} cached"
    return False, (f"{name} not cached: first run downloads it (or run "
                   "`socmint doctor --prefetch`); offline runs fall back to TF-IDF")


# ---------------------------------------------------------------------------- checks
def runs_dir_writable() -> tuple[bool, str]:
    try:
        d = runs_dir()
        with tempfile.NamedTemporaryFile(dir=d, prefix=".doctor-", delete=True):
            pass
        return True, str(d)
    except Exception as e:  # noqa: BLE001
        return False, f"not writable: {type(e).__name__}: {e}"


def in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def collect_report(cfg: dict[str, str] | None = None, *, check_telegram: bool = True,
                   check_model: bool = True) -> dict[str, Any]:
    cfg = cfg if cfg is not None else load_config()
    vi = sys.version_info
    writable, runs_detail = runs_dir_writable()
    report: dict[str, Any] = {
        "python": {"version": f"{vi.major}.{vi.minor}.{vi.micro}", "ok": vi[:2] >= (3, 11),
                   "executable": sys.executable},
        "venv": {"active": in_venv(), "prefix": sys.prefix},
        "config_file": {"path": str(config_path()), "exists": config_path().is_file()},
        "runs_dir": {"writable": writable, "detail": runs_detail},
        "sources": source_readiness(cfg),
        "analysis_keys": [{"key": k, "set": has(cfg, k), "enables": what}
                          for k, what in ANALYSIS_KEYS],
    }
    x_row = next((r for r in report["sources"] if r["name"] == "x"), None)
    if x_row is not None:
        report["x"] = {"provider": x_provider(cfg), "session": x_row.get("x_session")}
    if check_telegram:
        ok, detail = telegram_session_status(cfg)
        report["telegram_session"] = {"authorized": ok, "detail": detail}
    if check_model:
        ok, detail = embedding_model_cached(cfg)
        report["embedding_model"] = {"cached": ok, "detail": detail}
    return report


OUTCOME_LABEL = {"run": "will run", "skip": "will skip", "fail": "will fail"}


def row_outcome(row: dict[str, Any]) -> str:
    """run | skip | fail for a readiness row (older rows without `outcome`: from `ready`)."""
    out = row.get("outcome")
    if out in OUTCOME_LABEL:
        return str(out)
    return "run" if row.get("ready") else "skip"


def _mark(v: bool | None) -> str:
    return "OK  " if v else ("--  " if v is None else "NO  ")


def format_report(rep: dict[str, Any]) -> str:
    lines: list[str] = ["SOCMINT doctor", "=============="]
    py = rep["python"]
    lines.append(f"{_mark(py['ok'])}Python {py['version']} ({py['executable']})")
    lines.append(f"{_mark(rep['venv']['active'])}virtualenv: "
                 f"{'active' if rep['venv']['active'] else 'not active'} ({rep['venv']['prefix']})")
    cf = rep["config_file"]
    lines.append(f"{_mark(cf['exists'])}config file: {cf['path']}"
                 f"{'' if cf['exists'] else ' (missing; created on first GUI launch)'}")
    lines.append(f"{_mark(rep['runs_dir']['writable'])}runs dir: {rep['runs_dir']['detail']}")
    lines += ["", "Sources", "-------"]
    rows = [(r["name"], OUTCOME_LABEL.get(row_outcome(r), "will skip"),
             ", ".join(f"{k}={'set' if s else 'missing'}" for k, s in r["keys_set"].items())
             or "(none needed)", r["reason"]) for r in rep["sources"]]
    head = ("source", "status", "credentials", "reason")
    w = [max(len(head[i]), *(len(row[i]) for row in rows)) for i in range(3)] if rows else [0] * 3
    for row in [head, *rows]:
        lines.append("  ".join(c.ljust(w[i]) for i, c in enumerate(row[:3])) + "  " + row[3])
    lines += ["", "Analysis / evidence keys", "------------------------"]
    for k in rep["analysis_keys"]:
        lines.append(f"{_mark(k['set'])}{k['key']}: {'set' if k['set'] else 'missing'} "
                     f"-> {k['enables']}{'' if k['set'] else ' (disabled)'}")
    if "x" in rep:
        lines.extend(_format_x(rep["x"]))
    if "telegram_session" in rep:
        t = rep["telegram_session"]
        lines.append(f"{_mark(t['authorized'])}Telegram session: {t['detail']}")
    if "embedding_model" in rep:
        m = rep["embedding_model"]
        lines.append(f"{_mark(m['cached'])}Embedding model: {m['detail']}")
    return "\n".join(lines)


def _format_x(x: dict[str, Any]) -> list[str]:
    """X provider lines. Prints flags, source and screen name only, never cookie values."""
    lines = [f"OK  X provider: {x['provider']}"]
    st = x.get("session")
    if x["provider"] != WEB_SESSION or not isinstance(st, dict):
        return lines
    if st.get("authorized"):
        who = f" as @{str(st['user']).lstrip('@')}" if st.get("user") else ""
        lines.append(f"OK  X session: authorized{who} (source: {st.get('source') or 'unknown'})")
    else:
        lines.append(f"NO  X session: not authorized: {X_LOGIN_HINT}"
                     + (f" ({st['reason']})" if st.get("reason") else ""))
    lines.append(f"{_mark(bool(st.get('browser')))}X browser: "
                 f"{st.get('browser') or 'none'}: {st.get('browser_detail') or ''}".rstrip(": "))
    return lines


def main(prefetch: bool = False) -> int:
    cfg = load_config()
    if prefetch:
        try:
            from socmint.analyze.embed import prefetch_model

            os.environ.setdefault("SOCMINT_MODEL_DOWNLOAD_TIMEOUT", "900")
            ok, msg = prefetch_model(cfg)
            print(f"{_mark(ok)}prefetch: {msg}")
        except Exception as e:  # noqa: BLE001
            print(f"NO  prefetch failed: {redact(f'{type(e).__name__}: {e}', cfg)}")
        if x_provider(cfg) == WEB_SESSION:
            ok, msg = ensure_x_browser(cfg)
            print(f"{_mark(ok)}X browser: {msg}")
        else:
            print(f"--  X browser: not needed (X_PROVIDER={x_provider(cfg)})")
    print(format_report(collect_report(cfg)))
    return 0
