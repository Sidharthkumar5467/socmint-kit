"""Local configuration: credentials and run settings.

Resolution order (later wins, empty values never override):
  1. built-in defaults
  2. process environment and a `.env` file in the working directory
  3. the GUI-managed file `~/.socmint/config.json` (override dir with $SOCMINT_HOME)

Secrets live only on the local machine, never in the repo. `redact()` masks them in
any string before it is logged or shown.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Every key the module understands. Kept in sync with .env.example.
SECRET_KEYS: tuple[str, ...] = (
    "TWITTERAPI_IO_KEY",
    "X_BEARER_TOKEN",
    "X_AUTH_TOKEN",
    "X_CT0",
    "TELEGRAM_API_HASH",
    "YOUTUBE_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_FACTCHECK_API_KEY",
    "SERPAPI_KEY",
    "SERPER_API_KEY",
    "LINKEDIN_COOKIE",
)
DEFAULTS: dict[str, str] = {
    "TWITTERAPI_IO_KEY": "",
    # web_session (owner decision 2026-11-18): the owner's own logged-in x.com session
    # driven in a local browser. Alternatives: twitterapi_io | official.
    "X_PROVIDER": "web_session",
    "X_BEARER_TOKEN": "",
    "X_AUTH_TOKEN": "",  # optional: paste instead of the browser login (cookie auth_token)
    "X_CT0": "",  # optional: paste instead of the browser login (cookie ct0)
    "X_BROWSER_HEADLESS": "true",  # owner decision 2026-11-19: no window during runs (the login window stays visible); HeadlessChrome UA may raise the challenge risk, set false if X challenges the account
    "X_MAX_SCROLLS": "40",  # per-run cap on search pages (one SearchTimeline call each)
    "TELEGRAM_API_ID": "",
    "TELEGRAM_API_HASH": "",
    "TELEGRAM_SESSION": "socmint.session",
    "TELEGRAM_HAS_PREMIUM": "false",
    "TELEGRAM_SEED_CHANNELS": "",
    # Also search the groups and channels the logged-in account has joined (owner request
    # 2026-11-20). One-to-one chats are never searched. "false" turns it off.
    "TELEGRAM_SEARCH_JOINED_CHATS": "true",
    # Find public channels for the topic in Telegram's directory (and similar channels) and
    # search each by date (owner request 2026-11-20). "false" turns it off.
    "TELEGRAM_DISCOVER_CHANNELS": "true",
    "TELEGRAM_DISCOVER_MAX": "15",
    # Search known and newly-found data-leak / infostealer / breach channels for the topic
    # (owner request 2026-11-20): a leak about the subject, not general chatter. "false"
    # turns it off. UNDERGROUND_MAX caps how many such channels one run searches.
    "TELEGRAM_UNDERGROUND_DISCOVERY": "true",
    "TELEGRAM_UNDERGROUND_MAX": "20",
    # Coordination's reused-image signal: download avatars/thumbnails and perceptual-hash
    # them so the SAME image under a DIFFERENT URL is matched too (owner request
    # 2026-11-20). "false" falls back to identical-URL matching only (no image downloads).
    "COORD_IMAGE_HASHING": "true",
    # Concurrent LLM calls in the analysis stages (narrative labels, claim extract/stance/
    # judge). Higher = faster analysis if the Anthropic rate-limit tier allows; the SDK
    # retries transient 429s. Clamped to 1-32 (owner request 2026-11-20: 8 -> 12).
    "ANALYSIS_LLM_WORKERS": "12",
    "YOUTUBE_API_KEY": "",
    "YOUTUBE_QUOTA_LIMIT": "6000",
    "LINKEDIN_COOKIE": "",  # owner's linkedin.com Cookie header (or li_at alone): LinkedIn posts; also sent by LinkedIn Ads when set
    "LINKEDIN_MAX_SCROLLS": "25",  # per-run cap on scrolls of the LinkedIn content search (3 results per page load seen live)
    "LINKEDIN_BROWSER_HEADLESS": "true",  # no browser window during LinkedIn collection
    # LinkedIn posts provider: browser (default: linkedin.com's search page in a local
    # browser) | voyager (owner decision 2026-11-19: LinkedIn's Voyager search API called
    # directly with the same session cookie + its CSRF token). LINKEDIN_MAX_SCROLLS caps
    # Voyager pages too.
    "LINKEDIN_PROVIDER": "browser",
    # Persisted GraphQL query id of the Voyager content search; LinkedIn rotates it with
    # deploys (copy the current one from DevTools > Network when HTTP 400/404 appears).
    "LINKEDIN_VOYAGER_QUERY_ID": "voyagerSearchDashClusters.5ba32757c00b31aea747c8bebb92855c",
    # User agent sent by the voyager provider: a desktop Chrome string; set it to your own
    # browser's (the one LINKEDIN_COOKIE comes from).
    "LINKEDIN_USER_AGENT": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"),
    "ANTHROPIC_API_KEY": "",
    "ANALYSIS_MODEL": "claude-haiku-5-5",
    "GOOGLE_FACTCHECK_API_KEY": "",
    "SERPAPI_KEY": "",  # serpapi.com (64-char key)
    "SERPER_API_KEY": "",  # serper.dev (40-char key); a different service from SerpApi
    "MAX_WINDOW_DAYS": "30",
    # Optional wall-clock limit for a run, in seconds. 0 = no limit (owner decision
    # 2026-11-20: a time limit costs results; the analyst stops a long run from the GUI).
    "RUN_TIME_BUDGET_SECONDS": "0",
    "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
KNOWN_KEYS: tuple[str, ...] = tuple(DEFAULTS)


def home_dir() -> Path:
    """~/.socmint (or $SOCMINT_HOME). Holds config.json and the Telegram session."""
    return Path(os.environ.get("SOCMINT_HOME") or Path.home() / ".socmint")


def config_path() -> Path:
    return home_dir() / "config.json"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def runs_dir() -> Path:
    """Where new runs are written. $SOCMINT_RUNS_DIR overrides <project>/runs."""
    d = Path(os.environ.get("SOCMINT_RUNS_DIR") or project_root() / "runs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def sample_run_dir() -> Path:
    return project_root() / "data" / "sample_run"


def ensure_config_file() -> Path:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("{}\n", encoding="utf-8")
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass
    return p


def _read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        out[k.strip()] = v
    return out


def load_file_config() -> dict[str, str]:
    p = config_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        return {}
    return {k: str(v) for k, v in data.items() if v is not None}


def load_config(dotenv_dir: Path | None = None) -> dict[str, str]:
    cfg = dict(DEFAULTS)
    layers = [
        _read_dotenv((dotenv_dir or Path.cwd()) / ".env"),
        {k: v for k, v in os.environ.items() if k in DEFAULTS},
        load_file_config(),
    ]
    for layer in layers:
        for k, v in layer.items():
            if v not in (None, ""):
                cfg[k] = v
    return cfg


def save_config(updates: dict[str, str]) -> dict[str, str]:
    """Merge updates into config.json. An empty string removes the key."""
    p = ensure_config_file()
    current = load_file_config()
    for k, v in updates.items():
        if k not in KNOWN_KEYS:
            continue
        if v in (None, ""):
            current.pop(k, None)
        else:
            current[k] = str(v)
    p.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return current


WINDOW_CAP_DAYS = 30


def max_window_days(cfg: dict[str, str]) -> int:
    """MAX_WINDOW_DAYS clamped to 1..30; an invalid value falls back to 30."""
    try:
        days = int(str(cfg.get("MAX_WINDOW_DAYS") or WINDOW_CAP_DAYS).strip())
    except ValueError:
        return WINDOW_CAP_DAYS
    return max(1, min(days, WINDOW_CAP_DAYS))


def analysis_workers(cfg: dict[str, str]) -> int:
    """ANALYSIS_LLM_WORKERS as an int clamped to 1-32 (default 12)."""
    try:
        n = int(str(cfg.get("ANALYSIS_LLM_WORKERS") or 12).strip())
    except (TypeError, ValueError):
        n = 12
    return max(1, min(32, n))


def run_time_budget(cfg: dict[str, str]) -> float | None:
    """RUN_TIME_BUDGET_SECONDS as seconds, or None for no limit (0, empty, negative or
    invalid). A limit, when set, is at least 60 s."""
    try:
        secs = float(str(cfg.get("RUN_TIME_BUDGET_SECONDS") or 0).strip())
    except ValueError:
        return None
    if not secs > 0 or secs != secs or secs == float("inf"):
        return None
    return max(60.0, secs)


def has(cfg: dict[str, str], *keys: str) -> bool:
    return all(str(cfg.get(k, "")).strip() for k in keys)


def masked(cfg: dict[str, str]) -> dict[str, str]:
    """Config safe to send to the GUI: secrets reduced to a set/not-set hint."""
    out = {}
    for k in KNOWN_KEYS:
        v = cfg.get(k, "")
        if k in SECRET_KEYS:
            out[k] = ("••••" + v[-4:]) if len(v) > 8 else ("••••" if v else "")
        else:
            out[k] = v
    return out


_PATTERNS = [
    re.compile(r"(?i)(x-api-key|api[_-]?key|key|token|access_token|authorization|cookie|"
               r"li_at|bearer)([\"'\s:=]+)([A-Za-z0-9_\-\.%/+=]{6,})"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
]


def redact(text: str, cfg: dict[str, str] | None = None) -> str:
    """Mask secret values and secret-looking tokens in arbitrary text."""
    if not text:
        return text
    s = str(text)
    cfg = cfg if cfg is not None else load_config()
    for k in SECRET_KEYS:
        v = cfg.get(k, "")
        if v and len(v) >= 6:
            s = s.replace(v, "[REDACTED]")
    s = _PATTERNS[0].sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", s)
    for p in _PATTERNS[1:]:
        s = p.sub("[REDACTED]", s)
    return s
