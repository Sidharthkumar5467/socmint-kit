"""X collection through the owner's OWN logged-in x.com session (X_PROVIDER=web_session).

Owner decision 2026-11-18 (overrides CLAUDE.md §2 "no login-only endpoints" for X only):
X is read through the owner's own account in a local browser driven by Playwright.

What this module does and does NOT do:
  - The owner logs in THEMSELVES in a visible browser window (`start_login`); CAPTCHA and
    2FA are theirs to answer. The login form is never automated.
  - No stealth / fingerprint-spoofing plugins, no user-agent spoofing (the real
    browser's default UA is used), no account rotation, no parallel requests on the
    account: one browser, one page, one scroll at a time with 2–4 s pauses.
  - X's own web app builds every request (queryId, features, x-client-transaction-id).
    We only OBSERVE the JSON responses of GraphQL `SearchTimeline` via
    `page.on("response")` and parse them.

Evidence rules (CLAUDE.md §3), mirrored from socmint.collect.x:
  - `published_at` comes only from the tweet's `legacy.created_at`;
    `timestamp_verified=True` only when it parses.
  - Relations are DIRECT only (reply id, quoted / retweeted tweet, entity URLs,
    entity mentions).
  - Quoted / retweeted tweets embedded in a response were NOT matched by the search:
    they are emitted with `context_only=True` and never analysed.
  - Every SearchTimeline response is saved raw (raw/x/search_page_<n>.json) and
    referenced from `Post.raw_ref`.
  - Every stop reason is written to the SourceStatus; `completed_zero` only when every
    time slice's first response genuinely contained no tweet entries.
  - Search returns newest first, so the window is split into whole-day time slices
    (socmint.collect.window) searched one after another in the same browser session,
    newest first, sharing the per-source limit; the status reason lists posts per day.

GraphQL response shape: UNVERIFIED. The parser follows the publicly documented /
widely observed shape; fixtures are hand-built (tests/fixtures/x/doc_example_*.json)
until a live capture replaces them.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from socmint.collect.window import (
    DEFAULT_MAX_SLICES,
    SliceShare,
    search_topic,
    slice_query,
    time_slices,
)
from socmint.collect.x import (
    SOURCE,
    _context_posts,
    _handle,
    _int,
    _is_self_status_link,
    _lang,
    _platform_time,
    _status_url,
    _tio_media,
)
from socmint.config import home_dir, redact
from socmint.interfaces import CollectContext, CollectResult, Query, skipped
from socmint.schema import Metrics, Post, Relation, SourceState, SourceStatus
from socmint.util.log import get_logger
from socmint.util.timeutil import iso, now_iso, parse_ts

log = get_logger(__name__)

LOGIN_URL = "https://x.com/i/flow/login"
SEARCH_URL = "https://x.com/search?q={q}&src=typed_query&f=live"
COOKIE_URLS = ["https://x.com"]

REASON_PREFIX = "web_session (owner's logged-in account)"
NO_SESSION = "no X session: log in from Settings (X) or set X_AUTH_TOKEN + X_CT0"
NO_BROWSER = ("no browser: run socmint doctor --prefetch (installs Chromium) or install "
              "Google Chrome")
SESSION_EXPIRED = "X session expired or invalid: log in again from Settings"
ACCOUNT_LOCKED = ("X shows an account access / challenge page (/account/access): open "
                  "x.com in a normal browser, resolve what X asks for, then log in again; "
                  "if it recurs, set X_BROWSER_HEADLESS=false (headless Chrome announces "
                  "itself as HeadlessChrome)")
LOGIN_BUSY = ("X login in progress: finish or close the login window, then run the "
              "collection again (one browser on the account at a time)")
COLLECT_BUSY = ("another X collection is already running on this account (one browser "
                "on the account at a time)")
RATE_LIMITED = ("X rate limit (429) — the account's search quota is exhausted; try again "
                "later")

# Timing (module constants so tests can shrink them).
FIRST_RESPONSE_TIMEOUT_S = 30.0  # after opening the search page
RESPONSE_TIMEOUT_S = 15.0  # after each scroll
POLL_MS = 250
PAUSE_MS = (2000, 4000)  # human-ish pacing between scrolls
LOGIN_POLL_MS = 2000
MAX_TIMEOUTS = 2  # consecutive scrolls without a SearchTimeline response
NO_NEW_LIMIT = 2  # consecutive responses without a new tweet => end of results
VIEWPORT = {"width": 1280, "height": 900}

_TWEET_ITEM_TYPES = {"TimelineTweet"}


# ------------------------------------------------------------------ session files


def session_file() -> Path:
    """Playwright storage_state of the owner's login (chmod 0600, outside the repo)."""
    return home_dir() / "sessions" / "x_storage_state.json"


def _meta_file() -> Path:
    return home_dir() / "sessions" / "x_session_meta.json"


def __getattr__(name: str) -> Any:
    # SESSION_FILE is resolved on every access so $SOCMINT_HOME changes are honoured.
    if name == "SESSION_FILE":
        return session_file()
    raise AttributeError(name)


def _write_private(path: Path, text: str) -> None:
    """Write a file that is 0600 from the moment it exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _auth_cookie(state: dict | None, now: float | None = None) -> dict | None:
    """The non-expired auth_token cookie for x.com in a storage_state dict, if any."""
    now = time.time() if now is None else now
    for c in (state or {}).get("cookies") or []:
        if not isinstance(c, dict) or c.get("name") != "auth_token" or not c.get("value"):
            continue
        if not str(c.get("domain") or "").lstrip(".").endswith("x.com"):
            continue
        exp = c.get("expires")
        if exp in (None, -1) or (isinstance(exp, (int, float)) and (exp < 0 or exp > now)):
            return c
    return None


def _pasted(cfg: dict) -> bool:
    return bool(str(cfg.get("X_AUTH_TOKEN") or "").strip()
                and str(cfg.get("X_CT0") or "").strip())


def build_cookie_dicts(cfg: dict) -> list[dict]:
    """Playwright cookie dicts for pasted X_AUTH_TOKEN + X_CT0 (never logged)."""
    # UNVERIFIED: sameSite values mirror what x.com sets in a normal browser
    # (auth_token SameSite=None, ct0 SameSite=Lax); not confirmed against a live login.
    return [
        {"name": "auth_token", "value": str(cfg.get("X_AUTH_TOKEN") or "").strip(),
         "domain": ".x.com", "path": "/", "secure": True, "httpOnly": True,
         "sameSite": "None"},
        {"name": "ct0", "value": str(cfg.get("X_CT0") or "").strip(),
         "domain": ".x.com", "path": "/", "secure": True, "httpOnly": False,
         "sameSite": "Lax"},
    ]


def _session_secrets(cfg: dict) -> list[str]:
    """Cookie values that must never appear in a reason or a log line."""
    out = [str(cfg.get(k) or "") for k in ("X_AUTH_TOKEN", "X_CT0")]
    for c in (_read_json(session_file()) or {}).get("cookies") or []:
        if isinstance(c, dict) and c.get("name") in ("auth_token", "ct0", "twid", "kdt",
                                                     "att", "guest_id"):
            out.append(str(c.get("value") or ""))
    return [s for s in out if len(s) >= 6]


def _redact(text: str, cfg: dict, extra: list[str] | None = None) -> str:
    s = str(text)
    for v in sorted(set(_session_secrets(cfg) + (extra or [])), key=len, reverse=True):
        s = s.replace(v, "[REDACTED]")
    return redact(s, cfg)


# --------------------------------------------------------------- browser detection


def _chrome_installed() -> bool:
    candidates = [
        Path("/Applications/Google Chrome.app"),
        Path.home() / "Applications" / "Google Chrome.app",
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
    ]
    if any(p.exists() for p in candidates if str(p) not in ("", ".")):
        return True
    return any(shutil.which(n) for n in ("google-chrome", "google-chrome-stable"))


def _playwright_cache() -> Path | None:
    env = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if env and env != "0":
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ms-playwright"
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) / "ms-playwright" if base else None
    return Path.home() / ".cache" / "ms-playwright"


def _chromium_installed() -> bool:
    cache = _playwright_cache()
    if not cache or not cache.is_dir():
        return False
    return any(p.is_dir() and any(p.iterdir()) for p in cache.glob("chromium-*"))


def _detect_browser() -> str | None:
    """Cheap filesystem check (no process start): 'chrome', 'chromium' or None."""
    if _chrome_installed():
        return "chrome"
    if _chromium_installed():
        return "chromium"
    return None


def ensure_browser(install: bool = True) -> dict:
    """Make sure a browser Playwright can drive exists; install Chromium only if needed."""
    found = _detect_browser()
    if found == "chrome":
        return {"ok": True, "browser": "chrome",
                "message": "Google Chrome found; X collection will use it (channel=chrome)"}
    if found == "chromium":
        return {"ok": True, "browser": "chromium",
                "message": "Playwright Chromium found; X collection will use it"}
    if importlib.util.find_spec("playwright") is None:
        return {"ok": False, "browser": None,
                "message": "playwright is not installed: run `pip install -e .`"}
    if not install:
        return {"ok": False, "browser": None, "message": NO_BROWSER}
    log.info("no Chrome or Chromium found; installing Playwright Chromium (one-time, ~150 MB)")
    try:
        proc = subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"],
                              capture_output=True, text=True, timeout=900, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "browser": None,
                "message": f"`python -m playwright install chromium` failed: {e}"}
    if proc.returncode == 0 and _chromium_installed():
        return {"ok": True, "browser": "chromium",
                "message": "installed Playwright Chromium for X collection"}
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
    return {"ok": False, "browser": None,
            "message": "`python -m playwright install chromium` failed: " + " | ".join(tail)}


# ------------------------------------------------------------------- browser layer


class NoBrowserError(RuntimeError):
    """Neither Google Chrome nor Playwright Chromium could be launched."""


@dataclass
class BrowserHandle:
    browser_name: str
    context: Any
    page: Any
    close: Callable[[], None]


def _open_browser(*, headless: bool, storage_state: str | None = None,
                  cookies: list[dict] | None = None) -> BrowserHandle:
    """Launch Chrome (channel=chrome) or fall back to Playwright Chromium.

    Tests replace this function with a fake; nothing else in the module starts a browser.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise NoBrowserError("playwright is not installed") from e
    pw = sync_playwright().start()
    browser, name, errors = None, None, []
    for channel, label in (("chrome", "chrome"), (None, "chromium")):
        try:
            kwargs: dict[str, Any] = {"headless": headless}
            if channel:
                kwargs["channel"] = channel
            browser = pw.chromium.launch(**kwargs)
            name = label
            break
        except Exception as e:  # noqa: BLE001 — try the next browser
            errors.append(f"{label}: {str(e).splitlines()[0][:200] if str(e) else type(e).__name__}")
    if browser is None:
        pw.stop()
        raise NoBrowserError("; ".join(errors))
    try:
        # Real browser defaults otherwise: no user-agent override, no stealth plugins.
        context = browser.new_context(storage_state=storage_state, viewport=VIEWPORT,
                                      locale="en-US", timezone_id="UTC")
        if cookies:
            context.add_cookies(cookies)
        page = context.new_page()
    except Exception:
        browser.close()
        pw.stop()
        raise

    def close() -> None:
        for fn in (context.close, browser.close, pw.stop):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass

    return BrowserHandle(browser_name=name or "chromium", context=context, page=page,
                         close=close)


# ------------------------------------------------------------------ session status


def session_status(cfg: dict) -> dict:
    """Cheap status for GUI polling: files and config only, never a network call."""
    path = session_file()
    state = _read_json(path) if path.exists() else None
    meta = _read_json(_meta_file()) or {}
    file_ok = _auth_cookie(state) is not None
    pasted = _pasted(cfg)
    login = login_state()
    if file_ok:
        source, user = "browser_login", meta.get("user") or None
        reason = (f"logged in via browser{' as ' + user if user else ''}; session saved at "
                  f"{path}")
    elif pasted:
        source, user = "pasted_cookies", None
        reason = "using pasted X_AUTH_TOKEN + X_CT0 from config"
        if state is not None:
            reason += " (the saved browser session has no valid auth_token: expired)"
    else:
        source, user = None, None
        reason = NO_SESSION
        if state is not None:
            reason = "saved X session has expired or has no auth_token: log in again from Settings"
    return {
        "configured": state is not None or pasted,
        "authorized": file_ok or pasted,
        "user": user,
        "source": source,
        "session_path": str(path),
        "browser": _detect_browser(),
        "login_state": login["state"],
        "reason": reason,
    }


# -------------------------------------------------------------------------- login

_login_lock = threading.Lock()
# "collecting" is set while collect_web drives a browser on the account; login and
# collection never run at the same time (CLAUDE.md §5: no parallel use of one account).
_login: dict[str, Any] = {"state": "idle", "error": None, "thread": None,
                          "collecting": False}


def login_state() -> dict:
    with _login_lock:
        return {"state": _login["state"], "error": _login["error"]}


def _set_login(state: str, error: str | None = None) -> None:
    with _login_lock:
        _login["state"], _login["error"] = state, error


def _twid_user_id(cookies: list[dict]) -> str | None:
    for c in cookies or []:
        if isinstance(c, dict) and c.get("name") == "twid":
            v = unquote(str(c.get("value") or "")).strip('"')
            if v.startswith("u="):
                return v[2:] or None
    return None


_SCREEN_NAME_JS = """() => {
  const b = document.querySelector('[data-testid="SideNav_AccountSwitcher_Button"]');
  const t = b ? b.innerText : '';
  const m = t.match(/@([A-Za-z0-9_]{1,15})/);
  return m ? m[1] : null;
}"""


def _discover_screen_name(page: Any) -> str | None:
    # UNVERIFIED: relies on x.com's SideNav account switcher test id; best effort only.
    for _ in range(5):
        try:
            name = page.evaluate(_SCREEN_NAME_JS)
        except Exception:  # noqa: BLE001
            return None
        if name:
            return f"@{name}"
        try:
            page.wait_for_timeout(1000)
        except Exception:  # noqa: BLE001
            return None
    return None


def _page_closed(page: Any) -> bool:
    try:
        return bool(page.is_closed())
    except Exception:  # noqa: BLE001
        return True


def _login_worker(cfg: dict, timeout_s: int) -> None:
    """Headful login: the owner types credentials themselves; we only wait for cookies."""
    handle: BrowserHandle | None = None
    try:
        handle = _open_browser(headless=False)
        page = handle.page
        page.goto(LOGIN_URL, wait_until="domcontentloaded")
        deadline = time.monotonic() + timeout_s
        cookies: list[dict] = []
        while True:
            if _page_closed(page):
                _set_login("failed", "login window was closed before the login completed")
                return
            cookies = handle.context.cookies(COOKIE_URLS)
            if _auth_cookie({"cookies": cookies}):
                break
            if time.monotonic() >= deadline:
                _set_login("failed", f"login not completed within {timeout_s} s")
                return
            try:
                page.wait_for_timeout(LOGIN_POLL_MS)
            except Exception:  # noqa: BLE001 — window closed while waiting
                if _page_closed(page):
                    _set_login("failed", "login window was closed before the login completed")
                    return
                raise
        try:  # let ct0 / twid settle after the redirect to /home
            page.wait_for_timeout(LOGIN_POLL_MS)
        except Exception:  # noqa: BLE001
            pass
        user = _discover_screen_name(page)
        state = handle.context.storage_state()
        _write_private(session_file(), json.dumps(state))
        cookies = state.get("cookies") or cookies
        _write_private(_meta_file(), json.dumps({
            "user": user, "user_id": _twid_user_id(cookies), "saved_at": now_iso(),
            "browser": handle.browser_name}))
        _set_login("done")
        log.info("X login saved (%s)", user or "screen name not discovered")
    except NoBrowserError:
        _set_login("failed", NO_BROWSER)
    except Exception as e:  # noqa: BLE001
        _set_login("failed", _redact(f"{type(e).__name__}: {e}", cfg))
    finally:
        if handle is not None:
            handle.close()


def start_login(cfg: dict, timeout_s: int = 300) -> dict:
    """Open a visible browser at the X login page in a background thread (non-blocking)."""
    if importlib.util.find_spec("playwright") is None:
        return {"ok": False, "error": "playwright is not installed: run `pip install -e .`"}
    with _login_lock:
        t = _login.get("thread")
        if _login["state"] == "waiting" or (t is not None and t.is_alive()):
            return {"ok": False, "error": "an X login is already in progress"}
        if _login.get("collecting"):
            return {"ok": False, "error": "an X collection is running on this account; "
                                          "log in after it finishes"}
        _login["state"], _login["error"] = "waiting", None
        t = threading.Thread(target=_login_worker, args=(dict(cfg), int(timeout_s)),
                             name="socmint-x-login", daemon=True)
        _login["thread"] = t
    t.start()
    return {"ok": True, "error": None}


def logout(cfg: dict) -> dict:
    """Delete the saved browser session. Pasted cookies in config are NOT touched."""
    removed = False
    for p in (session_file(), _meta_file()):
        try:
            p.unlink()
            removed = True
        except FileNotFoundError:
            pass
    with _login_lock:
        if _login["state"] in ("done", "failed"):
            _login["state"], _login["error"] = "idle", None
    msg = "saved X browser session deleted" if removed else "no saved X browser session"
    if _pasted(cfg):
        msg += ("; pasted X_AUTH_TOKEN + X_CT0 remain in config and will still be used — "
                "clear them in Settings to fully log out")
    return {"ok": True, "message": msg}


# ------------------------------------------------------------------------ parsing


def _unwrap(result: Any) -> dict | None:
    """Tweet object from tweet_results.result; None for tombstones / unavailable."""
    if not isinstance(result, dict):
        return None
    tn = result.get("__typename")
    if tn == "TweetWithVisibilityResults":
        return _unwrap(result.get("tweet"))
    if tn in ("TweetTombstone", "TweetUnavailable"):
        return None
    if tn not in (None, "Tweet"):
        return None
    return result if result.get("rest_id") and isinstance(result.get("legacy"), dict) else None


def _user(tweet: dict) -> dict:
    """Author fields from BOTH observed user shapes (legacy.* and result.core.*)."""
    u = (((tweet.get("core") or {}).get("user_results") or {}).get("result")) or {}
    if not isinstance(u, dict) or u.get("__typename") == "UserUnavailable":
        return {}
    leg = u.get("legacy") or {}
    core = u.get("core") or {}  # UNVERIFIED: newer shape moved these fields under core
    return {
        "id": u.get("rest_id"),
        "screen_name": core.get("screen_name") or leg.get("screen_name"),
        "name": core.get("name") or leg.get("name"),
        "created_at": core.get("created_at") or leg.get("created_at"),
        "avatar": (u.get("avatar") or {}).get("image_url") or leg.get("profile_image_url_https"),
    }


def _note_text(tweet: dict) -> tuple[str | None, dict]:
    res = (((tweet.get("note_tweet") or {}).get("note_tweet_results") or {})
           .get("result")) or {}
    return (res.get("text") or None), (res.get("entity_set") or {})


def _relations(tweet: dict, leg: dict, note_entities: dict) -> list[Relation]:
    """DIRECT relations only, each from a named platform field."""
    rels: list[Relation] = []
    linked: set[str] = set()
    reply_to = str(leg.get("in_reply_to_status_id_str") or "").strip()
    if reply_to:
        rels.append(Relation(type="reply", target_post_id=f"x:{reply_to}",
                             target_handle=_handle(leg.get("in_reply_to_screen_name"))))
    quoted = _unwrap((tweet.get("quoted_status_result") or {}).get("result"))
    qid = str(leg.get("quoted_status_id_str") or (quoted or {}).get("rest_id") or "").strip()
    if qid:
        linked.add(qid)
        quser = _user(quoted).get("screen_name") if quoted else None
        # UNVERIFIED: quoted_status_permalink.expanded as URL fallback.
        perma = (leg.get("quoted_status_permalink") or {}).get("expanded")
        rels.append(Relation(type="quote", target_post_id=f"x:{qid}",
                             target_url=_status_url(quser, qid, perma),
                             target_handle=_handle(quser)))
    rt = _unwrap((leg.get("retweeted_status_result") or {}).get("result"))
    if rt:
        rid = str(rt["rest_id"])
        linked.add(rid)
        ruser = _user(rt).get("screen_name")
        rels.append(Relation(type="repost", target_post_id=f"x:{rid}",
                             target_url=_status_url(ruser, rid), target_handle=_handle(ruser)))
    ent = leg.get("entities") or {}
    seen_urls: set[str] = set()
    for u in list(ent.get("urls") or []) + list(note_entities.get("urls") or []):
        url = (u or {}).get("expanded_url") or (u or {}).get("url")
        if url and url not in seen_urls and not _is_self_status_link(url, linked):
            seen_urls.add(url)
            rels.append(Relation(type="link", target_url=url))
    seen_handles: set[str] = set()
    for m in list(ent.get("user_mentions") or []) + list(note_entities.get("user_mentions") or []):
        name = (m or {}).get("screen_name")
        if name and name.lower() not in seen_handles:
            seen_handles.add(name.lower())
            rels.append(Relation(type="mention", target_handle=_handle(name)))
    return rels


def parse_tweet(tweet: dict, *, collected_at: str, raw_ref: str | None,
                query: str | None) -> Post | None:
    """Map one GraphQL Tweet (already unwrapped) to a Post."""
    tid = str(tweet.get("rest_id") or "").strip()
    leg = tweet.get("legacy") or {}
    if not tid:
        return None
    user = _user(tweet)
    screen = user.get("screen_name")
    published, verified, note = _platform_time(leg.get("created_at"), "legacy.created_at")
    author_created = parse_ts(user.get("created_at")) if user.get("created_at") else None
    long_text, note_entities = _note_text(tweet)
    return Post(
        post_id=f"x:{tid}",
        source=SOURCE,
        kind="post",
        author_id=str(user["id"]) if user.get("id") else None,
        author_handle=_handle(screen),
        author_name=user.get("name") or None,
        author_created_at=iso(author_created),
        author_avatar_url=user.get("avatar") or None,
        text=long_text or leg.get("full_text") or "",
        lang=_lang(leg.get("lang")),
        published_at=published,
        collected_at=collected_at,
        timestamp_verified=verified,
        timestamp_note=note,
        metrics=Metrics(
            views=_int((tweet.get("views") or {}).get("count")),
            likes=_int(leg.get("favorite_count")),
            replies=_int(leg.get("reply_count")),
            reposts=_int(leg.get("retweet_count")),
            quotes=_int(leg.get("quote_count")),
        ),
        # Same classic media shape as TwitterAPI.io's extendedEntities (snake_case here).
        media=_tio_media({"extendedEntities": leg.get("extended_entities"),
                          "entities": leg.get("entities")}),
        url=_status_url(screen, tid),
        relations=_relations(tweet, leg, note_entities),
        raw_ref=raw_ref,
        query=query,
    )


def _instructions(payload: dict) -> list[dict]:
    tl = ((((payload or {}).get("data") or {}).get("search_by_raw_query") or {})
          .get("search_timeline") or {}).get("timeline") or {}
    return [i for i in tl.get("instructions") or [] if isinstance(i, dict)]


def _entries(payload: dict) -> list[dict]:
    out: list[dict] = []
    for ins in _instructions(payload):
        kind = ins.get("type")
        if kind == "TimelineAddEntries":
            out.extend(e for e in ins.get("entries") or [] if isinstance(e, dict))
        elif kind == "TimelineReplaceEntry" and isinstance(ins.get("entry"), dict):
            out.append(ins["entry"])
    return out


def _is_promoted(entry_id: Any, item_content: dict) -> bool:
    # UNVERIFIED: X injects ads into search timelines as entryId "promoted-tweet-..." with
    # itemContent.promotedMetadata. An ad did not match the query: never a matched post.
    return (str(entry_id or "").startswith("promoted")
            or bool(item_content.get("promotedMetadata")))


def _item_contents(entry: dict) -> list[tuple[bool, dict]]:
    """(is_promoted, itemContent) for every tweet item of an item or module entry."""
    content = entry.get("content") or {}
    kind = content.get("entryType") or content.get("__typename")
    contents: list[tuple[Any, dict]] = []
    if kind == "TimelineTimelineItem":
        contents.append((entry.get("entryId"), content.get("itemContent") or {}))
    elif kind == "TimelineTimelineModule":
        for it in content.get("items") or []:
            it = it or {}
            item_id = it.get("entryId")
            if str(entry.get("entryId") or "").startswith("promoted"):
                item_id = entry.get("entryId")
            contents.append((item_id, (it.get("item") or {}).get("itemContent") or {}))
    out = []
    for eid, ic in contents:
        itype = ic.get("itemType") or ic.get("__typename")
        if itype in _TWEET_ITEM_TYPES or (itype is None and "tweet_results" in ic):
            out.append((_is_promoted(eid, ic), ic))
    return out


def _tweet_results(entry: dict) -> list[Any]:
    """Raw tweet_results.result objects of an item or module entry (promoted skipped)."""
    return [(ic.get("tweet_results") or {}).get("result")
            for promoted, ic in _item_contents(entry) if not promoted]


def count_tweet_entries(payload: dict) -> int:
    """Organic tweet items in a response, including tombstones (they are real search
    hits). Promoted items are not counted: a page of only ads is not a page of results."""
    return sum(len(_tweet_results(e)) for e in _entries(payload))


def count_promoted(payload: dict) -> int:
    """Promoted (ad) tweet items in a response; skipped by the parser."""
    return sum(1 for e in _entries(payload) for promoted, _ in _item_contents(e) if promoted)


def _cursor_of(payload: dict, kind: str) -> str | None:
    for e in _entries(payload):
        c = e.get("content") or {}
        if c.get("cursorType") == kind and c.get("value"):
            return str(c["value"])
    return None


def bottom_cursor(payload: dict) -> str | None:
    return _cursor_of(payload, "Bottom")


def top_cursor(payload: dict) -> str | None:
    return _cursor_of(payload, "Top")


_UNKNOWN = object()


def _request_variables(resp: Any) -> dict | None:
    """The JSON `variables` of the SearchTimeline request behind a response, or None.

    UNVERIFIED: x.com's web app sends SearchTimeline as GET with a JSON `variables` query
    parameter (cursor and rawQuery inside it); not yet confirmed against a live capture.
    """
    url = ""
    try:
        req = getattr(resp, "request", None)
        url = (getattr(req, "url", None) if req is not None else None) or resp.url or ""
    except Exception:  # noqa: BLE001
        return None
    raw = parse_qs(urlparse(url).query).get("variables")
    if not raw:
        return None
    try:
        variables = json.loads(raw[0])
    except ValueError:
        return None
    return variables if isinstance(variables, dict) else None


def request_cursor(resp: Any) -> Any:
    """`variables.cursor` of the SearchTimeline request behind a response.

    None = a request without a cursor (the first page, or a refetch of it); _UNKNOWN when
    the variables cannot be read from the URL.
    """
    variables = _request_variables(resp)
    if variables is None:
        return _UNKNOWN
    cur = variables.get("cursor")
    return str(cur) if cur else None


def request_raw_query(resp: Any) -> str | None:
    """`variables.rawQuery` of the request (None when absent or unreadable).

    Used to tell a time slice's pages from late responses of the previous slice's search.
    """
    rq = (_request_variables(resp) or {}).get("rawQuery")
    return " ".join(str(rq).split()) if rq else None


def is_terminated(payload: dict) -> bool:
    # UNVERIFIED: X timelines may end with TimelineTerminateTimeline direction Bottom.
    return any(i.get("type") == "TimelineTerminateTimeline"
               and i.get("direction") in ("Bottom", "TopAndBottom")
               for i in _instructions(payload))


def parse_search_timeline(payload: dict, *, collected_at: str, raw_ref: str | None,
                          query_str: str) -> tuple[list[Post], list[Post], str | None]:
    """(matched posts, context_only embedded posts, bottom cursor) from one response."""
    main: list[Post] = []
    embedded: list[Post] = []
    seen: set[str] = set()
    for entry in _entries(payload):
        for raw in _tweet_results(entry):
            tweet = _unwrap(raw)
            if tweet is None:
                continue
            p = parse_tweet(tweet, collected_at=collected_at, raw_ref=raw_ref, query=query_str)
            if p is None or p.post_id in seen:
                continue
            seen.add(p.post_id)
            main.append(p)
            leg = tweet.get("legacy") or {}
            for res, label in (((tweet.get("quoted_status_result") or {}).get("result"),
                                "embedded:quoted"),
                               ((leg.get("retweeted_status_result") or {}).get("result"),
                                "embedded:retweeted")):
                inner = _unwrap(res)
                # Only with its own platform timestamp; otherwise the relation still
                # carries target_post_id / target_url.
                if inner and (inner.get("legacy") or {}).get("created_at"):
                    ep = parse_tweet(inner, collected_at=collected_at, raw_ref=raw_ref,
                                     query=label)
                    if ep is not None:
                        embedded.append(ep.model_copy(update={"context_only": True}))
    main_ids = {p.post_id for p in main}
    uniq: dict[str, Post] = {}
    for p in embedded:
        if p.post_id not in main_ids:
            uniq.setdefault(p.post_id, p)
    return main, list(uniq.values()), bottom_cursor(payload)


# --------------------------------------------------------------------- collection


def build_search(query: Query) -> str:
    """X web search string bounded with Unix-second operators since_time:/until_time:.

    Seen live 2026-11-19 (owner's session, "bank of america"): the date operators
    since:/until: are NOT UTC days. `until:2026-11-15` returned posts up to 18:21Z on
    11-15 and `since:2026-11-18 until:2026-11-19` returned posts from 12:22Z on 11-19,
    so day slices built with them overlapped. `since_time:/until_time:` cut exactly:
    a 2026-11-16 slice's newest post was 23:59:50Z. `query.end` is the inclusive last
    second, so until_time is end + 1. UNVERIFIED: the since_time lower bound was not
    paged to its end live; verify.gate_timestamps still bounds every post to the window.

    The topic goes through window.search_topic: a plain multi-word topic is sent as an
    exact phrase ("bank of america") unless the run's match option is 'words'.
    """
    since = int(query.start.timestamp())
    until = int(query.end.timestamp()) + 1
    return f"({search_topic(query)}) since_time:{since} until_time:{until}"


def _truthy(v: Any, default: bool) -> bool:
    s = str(v if v is not None else "").strip().lower()
    if not s:
        return default
    return s in ("1", "true", "yes", "on")


def _int_cfg(v: Any, default: int) -> int:
    try:
        return max(1, int(str(v).strip()))
    except (TypeError, ValueError):
        return default


def _body_gone(ex: BaseException) -> bool:
    """True for Playwright's error when a response body was released by a navigation
    (seen live 2026-11-20: "Network.getResponseBody: No resource with given identifier
    found ... a response that was navigated away from")."""
    msg = str(ex)
    return ("No resource with given identifier" in msg or "navigated away" in msg
            or "Response body is unavailable" in msg)


def _is_search_timeline(url: str) -> bool:
    path = urlparse(url or "").path
    return "/graphql/" in path and "/SearchTimeline" in path


def _page_problem(page: Any) -> str | None:
    """'locked' | 'login' | None from the page URL."""
    try:
        path = urlparse(page.url or "").path
    except Exception:  # noqa: BLE001
        return None
    if "/account/access" in path:
        return "locked"
    if path.startswith("/login") or "/i/flow/login" in path or path.endswith("/login"):
        return "login"
    return None


def _reset_time(headers: Any) -> str | None:
    try:
        v = (headers or {}).get("x-rate-limit-reset")
    except Exception:  # noqa: BLE001
        return None
    dt = parse_ts(v) if v else None
    return iso(dt) if dt else None


class _Stop(Exception):
    """End of a time slice (halt=False: the run goes on with the next slice) or of the
    whole run (halt=True: account / session / budget problems). `error` marks a slice
    that was cut by an error rather than by a cap or a soft limit."""

    def __init__(self, state: SourceState, reason: str, *, halt: bool = False,
                 error: bool = False) -> None:
        super().__init__(reason)
        self.state, self.reason, self.halt, self.error = state, reason, halt, error


@dataclass
class _Slice:
    """Per-slice collection state."""
    label: str
    first_date: str
    search: str
    cap: int
    cursor: str | None = None  # Bottom cursor the NEXT accepted page was requested with
    resume: str | None = None  # cursor to resume this slice from
    got: int = 0
    parsed: int = 0  # posts parsed from this slice's pages, incl. ones already kept
    first_entries: int | None = None
    no_new: int = 0
    outcome: str = "running"  # done | cut | halted
    error: bool = False
    note: str = ""


def _acquire_collection() -> str | None:
    """Claim the account for one collection; reason string when it is busy."""
    with _login_lock:
        t = _login.get("thread")
        if _login["state"] == "waiting" or (t is not None and t.is_alive()):
            return LOGIN_BUSY
        if _login.get("collecting"):
            return COLLECT_BUSY
        _login["collecting"] = True
    return None


def _release_collection() -> None:
    with _login_lock:
        _login["collecting"] = False


def collect_web(query: Query, ctx: CollectContext) -> CollectResult:
    cfg = ctx.config
    search = build_search(query)
    st = session_status(cfg)
    if not st["authorized"]:
        res = skipped(SOURCE, search, SourceState.skipped_no_credentials, NO_SESSION)
        res.status[0].date_filtered = True
        return res
    busy = _acquire_collection()
    if busy:
        # One browser on the owner's account at a time (CLAUDE.md §5): never in parallel
        # with the headful login window or with another collection.
        return CollectResult(posts=[], status=[SourceStatus(
            source=SOURCE, query=search, state=SourceState.failed, date_filtered=True,
            started_at=now_iso(), finished_at=now_iso(), cost_usd=0.0,
            reason=f"{REASON_PREFIX}: {busy}; 0 SearchTimeline response(s)")])
    try:
        return _collect_web(query, ctx, search, st)
    finally:
        _release_collection()


def _collect_web(query: Query, ctx: CollectContext, search: str, st: dict) -> CollectResult:
    """One browser session for the whole collection; one X search per time slice
    (newest first), the per-source limit shared across slices (socmint.collect.window)."""
    cfg = ctx.config
    status = SourceStatus(source=SOURCE, query=search, state=SourceState.ok,
                          date_filtered=True, started_at=now_iso(), cost_usd=0.0)
    use_file = st["source"] == "browser_login"
    headless = _truthy(cfg.get("X_BROWSER_HEADLESS"), True)
    max_scrolls = _int_cfg(cfg.get("X_MAX_SCROLLS"), 40)  # per slice
    slices = time_slices(query.start, query.end, DEFAULT_MAX_SLICES)
    share = SliceShare(query.limit, len(slices))

    main: dict[str, Post] = {}
    embedded: list[Post] = []
    inbox: list[Any] = []
    runs: list[_Slice] = []
    handle: BrowserHandle | None = None
    halt: _Stop | None = None
    halt_date: str | None = None  # first date of the slice where the run stopped
    capless = 0
    tally = {"pages": 0, "ignored": 0, "promoted": 0, "unconfirmed": 0, "gone": 0}

    def exhausted(what: str, confirmed: bool) -> _Stop:
        if confirmed:
            return _Stop(SourceState.ok, f"search exhausted: {what}")
        # The response could not be tied to the scroll's Bottom cursor: it might be a
        # Top poll / refetch, so "no more results" is not proven.
        return _Stop(SourceState.partial,
                     f"{what}, but the response could not be matched to the requested "
                     "Bottom cursor (request variables unreadable); results may be "
                     "incomplete")

    def account_check(resp: Any, code: int) -> None:
        # 429 and auth errors are about the account, whatever request they answer.
        if code == 429:
            reset = _reset_time(getattr(resp, "headers", None))
            raise _Stop(SourceState.partial, RATE_LIMITED
                        + (f" (X reports the limit resets at {reset})" if reset else ""),
                        halt=True)
        if code in (401, 403):
            raise _Stop(SourceState.failed, f"{SESSION_EXPIRED} (HTTP {code})", halt=True)

    def ignore(resp: Any, code: int) -> None:
        tally["ignored"] += 1
        try:
            body = resp.json() if code < 400 else None
            if isinstance(body, dict):
                ctx.save_raw(SOURCE, f"search_ignored_{tally['ignored']}.json", body)
        except Exception:  # noqa: BLE001 — audit copy only
            pass

    def handle_response(resp: Any, sl: _Slice) -> bool:
        """Process one captured SearchTimeline response; True when it was the page we
        scrolled for (same slice search, request cursor == current Bottom cursor)."""
        status.requests += 1
        code = int(getattr(resp, "status", 200) or 200)
        account_check(resp, code)
        rc = request_cursor(resp)
        confirmed = rc is not _UNKNOWN
        rq = request_raw_query(resp)
        if confirmed and (rc != sl.cursor or (rq is not None and rq != sl.search)):
            # A Top-cursor poll for "new posts", a refetch of an earlier page, a late page
            # of the previous slice's search, or a page we did not ask for: never counted
            # as a page and never ends the slice.
            ignore(resp, code)
            return False
        body: Any = None
        if code < 400:
            try:
                body = resp.json()
            except Exception as ex:  # noqa: BLE001 — classified below
                if not _body_gone(ex):
                    raise
                # Seen live 2026-11-20: the browser frees a response body once the page
                # navigates (the next slice's search), so a late response can no longer be
                # read. It is not counted as a page; if it was the page we scrolled for,
                # the wait for the next page times out and the slice says so.
                tally["gone"] += 1
                log.info("X web: SearchTimeline body no longer available (page navigated)")
                return False
        if not confirmed:
            tally["unconfirmed"] += 1
        tally["pages"] += 1
        n = tally["pages"]
        if code >= 400:
            raise _Stop(SourceState.partial, f"HTTP {code} from SearchTimeline on page {n}",
                        error=True)
        if not isinstance(body, dict):
            raise _Stop(SourceState.partial, f"non-object JSON from SearchTimeline on page {n}",
                        error=True)
        raw_ref = ctx.save_raw(SOURCE, f"search_page_{n}.json", body)
        if not _instructions(body) and body.get("errors"):
            msg = "; ".join(str((e or {}).get("message") or e)
                            for e in body["errors"])[:300]
            raise _Stop(SourceState.partial,
                        f"SearchTimeline returned errors on page {n}: {msg}", error=True)
        got, emb, nxt = parse_search_timeline(body, collected_at=now_iso(),
                                              raw_ref=raw_ref, query_str=sl.search)
        entries = count_tweet_entries(body)
        tally["promoted"] += count_promoted(body)
        if sl.first_entries is None:
            sl.first_entries = entries
        sl.parsed += len(got)
        new: dict[str, Post] = {}
        for p in got:
            if p.post_id not in main:
                new.setdefault(p.post_id, p)
        keep = list(new.values())[: max(0, sl.cap - sl.got)]
        dropped = len(new) - len(keep)
        for p in keep:
            main[p.post_id] = p
        sl.got += len(keep)
        embedded.extend(emb)
        ctx.progress(SOURCE, {"state": "collecting", "count": len(main), "pages": n,
                              "slice": sl.label})
        if sl.got >= sl.cap:
            msg = f"reached its share of {sl.cap} post(s)"
            if dropped:
                # Posts are cut from THIS page: keep the cursor that fetched it (as
                # socmint.collect.x does), so a resume does not skip them.
                sl.resume = sl.cursor
                where = ("the slice's saved cursor" if sl.cursor else
                         "the slice's first page (no cursor: the cut was on page 1)")
                msg += (f"; {dropped} post(s) on the last page were not kept; resume from "
                        f"{where}, which re-fetches that page, so up to {len(got)} post(s) "
                        "repeat")
            elif nxt:
                sl.resume = nxt
                msg += "; more results were available"
            raise _Stop(SourceState.ok, msg)
        if entries == 0:
            raise exhausted(f"page {n} had no tweet entries", confirmed)
        if is_terminated(body):
            raise exhausted(f"X ended the timeline on page {n}", confirmed)
        if not nxt or nxt == sl.cursor:
            raise exhausted(f"Bottom cursor {'missing' if not nxt else 'unchanged'} on "
                            f"page {n}", confirmed)
        sl.no_new = 0 if new else sl.no_new + 1
        sl.cursor = nxt
        if sl.no_new >= NO_NEW_LIMIT:
            raise exhausted(f"no new tweets in {NO_NEW_LIMIT} consecutive responses",
                            confirmed)
        return True

    def run_slice(page: Any, sl: _Slice) -> None:
        """Raises _Stop when the slice ends (halt=True when the whole run must stop)."""

        def wait_for(timeout_s: float) -> str:
            end = time.monotonic() + timeout_s
            while True:
                if inbox:
                    return "got"
                problem = _page_problem(page)
                if problem:
                    return problem
                if ctx.out_of_time():
                    return "deadline"
                if time.monotonic() >= end:
                    return "timeout"
                page.wait_for_timeout(POLL_MS)

        def await_page(timeout_s: float) -> str:
            """Wait for the response to the page we asked for; other SearchTimeline
            responses arriving meanwhile are drained and ignored."""
            end = time.monotonic() + timeout_s
            while True:
                outcome = wait_for(max(0.0, end - time.monotonic()))
                if outcome != "got":
                    return outcome
                accepted = False
                while inbox:
                    accepted = handle_response(inbox.pop(0), sl) or accepted
                if accepted:
                    return "page"

        def check_problem(outcome: str) -> None:
            if outcome == "locked":
                raise _Stop(SourceState.failed, ACCOUNT_LOCKED, halt=True)
            if outcome == "login":
                raise _Stop(SourceState.failed, SESSION_EXPIRED, halt=True)
            if outcome == "deadline":
                raise _Stop(SourceState.partial,
                            f"time budget exhausted after {tally['pages']} search page(s)",
                            halt=True)

        # Responses still arriving from the previous slice's search are not this slice's.
        while inbox:
            r = inbox.pop(0)
            status.requests += 1
            code = int(getattr(r, "status", 200) or 200)
            account_check(r, code)
            ignore(r, code)
        page.goto(SEARCH_URL.format(q=quote(sl.search, safe="")), wait_until="domcontentloaded")
        outcome = await_page(FIRST_RESPONSE_TIMEOUT_S)
        check_problem(outcome)
        if outcome == "timeout":
            raise _Stop(SourceState.partial,
                        f"no SearchTimeline first page within {FIRST_RESPONSE_TIMEOUT_S:.0f} s "
                        "of opening the search page (x.com layout or login state may have "
                        "changed)", error=True)
        scrolls = timeouts = 0
        while True:
            if scrolls >= max_scrolls:
                raise _Stop(SourceState.partial,
                            f"stopped at X_MAX_SCROLLS={max_scrolls}; more results may exist")
            if ctx.out_of_time():
                check_problem("deadline")
            page.wait_for_timeout(random.randint(*PAUSE_MS))
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            scrolls += 1
            outcome = await_page(RESPONSE_TIMEOUT_S)
            check_problem(outcome)
            if outcome == "timeout":
                timeouts += 1
                if timeouts >= MAX_TIMEOUTS:
                    raise _Stop(SourceState.partial,
                                f"no SearchTimeline response for the next page after "
                                f"{MAX_TIMEOUTS} scrolls ({RESPONSE_TIMEOUT_S:.0f} s each); "
                                "results may be incomplete")
            else:
                timeouts = 0

    try:
        try:
            handle = _open_browser(
                headless=headless,
                storage_state=str(session_file()) if use_file else None,
                cookies=None if use_file else build_cookie_dicts(cfg))
        except NoBrowserError as e:
            log.warning("X web: no browser: %s", _redact(str(e), cfg))
            raise _Stop(SourceState.failed, NO_BROWSER, halt=True) from e
        page = handle.page
        page.on("response", lambda r: inbox.append(r) if _is_search_timeline(r.url) else None)
        for i, (s, e) in enumerate(slices):
            label = (f"{s:%Y-%m-%d}" if s.date() == e.date()
                     else f"{s:%Y-%m-%d}..{e:%Y-%m-%d}")
            if ctx.out_of_time():
                halt_date = f"{s:%Y-%m-%d}"
                raise _Stop(SourceState.partial,
                            f"time budget exhausted before slice {label} "
                            f"(after {tally['pages']} search page(s))", halt=True)
            cap = share.allowance(i)
            if cap <= 0:  # the per-source limit is used up: no request for this slice
                capless += 1
                continue
            sl = _Slice(label=label, first_date=f"{s:%Y-%m-%d}", cap=cap,
                        search=build_search(slice_query(query, s, e, cap)))
            runs.append(sl)
            try:
                run_slice(page, sl)
            except _Stop as stop:
                sl.note, sl.error = stop.reason, stop.error
                if stop.halt:
                    sl.outcome, halt_date = "halted", sl.first_date
                    raise
                sl.outcome = "done" if stop.state == SourceState.ok else "cut"
                if (sl.outcome == "done" and sl.got == 0 and sl.parsed == 0
                        and sl.first_entries):
                    # Entries existed (e.g. tombstones) but none were usable: not "no posts".
                    sl.outcome = "cut"
                    sl.note += ("; the search returned entries but none could be parsed "
                                "into posts (tombstones / unavailable / unknown shape)")
            except Exception as ex:  # noqa: BLE001 — page/browser failure must never look like zero
                sl.outcome, halt_date = "halted", sl.first_date
                sl.note = _redact(f"browser error after {tally['pages']} search page(s): "
                                  f"{type(ex).__name__}: {ex}", cfg)
                log.warning("X web collection error: %s", sl.note)
                raise _Stop(SourceState.partial if main else SourceState.failed, sl.note,
                            halt=True) from ex
            finally:
                share.record(i, sl.got)
    except _Stop as stop:
        halt = stop
    except Exception as ex:  # noqa: BLE001 — e.g. page.on failing before any slice
        halt = _Stop(SourceState.partial if main else SourceState.failed,
                     _redact(f"browser error: {type(ex).__name__}: {ex}", cfg), halt=True)
        log.warning("X web collection error: %s", halt.reason)
    finally:
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass

    posts = list(main.values())[: query.limit]
    cut = [r for r in runs if r.outcome == "cut"]
    if halt is not None:
        state = halt.state
        if state == SourceState.failed and posts:
            state = SourceState.partial  # failed only when nothing was collected
        head = halt.reason
    elif cut:
        if posts or not all(r.error for r in runs):
            state = SourceState.partial
        else:
            state = SourceState.failed  # every slice was cut by an error, nothing collected
        head = ("every slice failed" if state == SourceState.failed
                else "some slices were cut short")
    elif slices and capless == len(slices):
        # Nothing was requested (per-source limit 0): never report "0 matching posts".
        state, head = SourceState.partial, "no slice searched"
        halt_date = f"{slices[0][0]:%Y-%m-%d}"
    elif posts:
        state = SourceState.ok
        head = (f"collected {len(posts)} post(s) across {len(slices)} time slice(s), newest "
                f"first, sharing the per-source limit of {query.limit}")
    else:
        state = SourceState.completed_zero
        head = f"search completed with 0 matching posts for query {search!r}"
    if capless:
        head += (f"; {capless} older slice(s) not searched: the per-source limit of "
                 f"{query.limit} was used up by newer slices")
    if state == SourceState.partial:
        stop = next((r for r in reversed(runs) if r.outcome in ("halted", "cut")), None)
        if (halt is not None or not runs) and (stop is None or stop.outcome != "halted"):
            status.cursor = f"{halt_date}:" if halt_date else None
        elif stop is not None:
            status.cursor = f"{stop.first_date}:{stop.resume or stop.cursor or ''}"
    notes = " | ".join(f"{r.label}: {r.note}" for r in runs if r.note)
    extras = [f"{status.requests} SearchTimeline response(s), {tally['pages']} used as "
              "search pages"]
    if tally["ignored"]:
        extras.append(f"{tally['ignored']} ignored (not the requested Bottom page: Top "
                      "poll / refetch / previous slice)")
    if tally["gone"]:
        extras.append(f"{tally['gone']} discarded: the browser had already released the "
                      "response body (the page had moved on)")
    if tally["unconfirmed"]:
        extras.append(f"{tally['unconfirmed']} page(s) not matched to a cursor "
                      "(request variables unreadable)")
    if tally["promoted"]:
        extras.append(f"{tally['promoted']} promoted (ad) tweet(s) skipped")
    if headless:
        extras.append("headless browser (X_BROWSER_HEADLESS=true; its user agent "
                      "identifies as HeadlessChrome)")
    parts = [head]
    if runs or halt_date or capless:
        parts.append(share.summary(slices))
    if notes:
        parts.append("slices: " + notes)
    status.state = state
    status.count = len(posts)
    status.reason = (f"{REASON_PREFIX}: {_redact('; '.join(parts), cfg)}; " + "; ".join(extras))
    status.finished_at = now_iso()
    return CollectResult(posts=posts + _context_posts(posts, embedded), status=[status])


__all__ = [
    "SESSION_FILE",  # noqa: F822 — provided by module __getattr__
    "BrowserHandle",
    "NoBrowserError",
    "build_cookie_dicts",
    "collect_web",
    "ensure_browser",
    "login_state",
    "logout",
    "parse_search_timeline",
    "session_file",
    "session_status",
    "start_login",
]
