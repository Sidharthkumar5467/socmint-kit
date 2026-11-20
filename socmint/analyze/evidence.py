"""External evidence providers for claim assessment (docs/SPEC.md §6, misinformation).

Evidence rules encoded here:
- Providers only return what an external service actually returned. Nothing is
  synthesised: every `Evidence` carries the URL the provider gave us and `retrieved_at`.
- Providers never decide support. Every item is returned as `supports="context"`; the
  verdict step (claims.py) is the only place that marks an item supports/contradicts,
  and it can only do so for items in this retrieved list.
- Collected social posts (including ones that debunk a claim) are NOT external evidence.
  They only appear in a claim's stance counts.
- Providers never raise. Missing key, HTTP error or a malformed response is logged and
  returns []. Every query and raw response is written under `run_dir/raw/evidence/` and
  summarised in `run_dir/evidence_log.jsonl` so a reviewer can see exactly what was
  retrieved. The API key is never written to disk or logs.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from socmint.config import SECRET_KEYS, redact
from socmint.schema import Evidence
from socmint.util.http import HttpFailure, make_client, request_json
from socmint.util.log import get_logger
from socmint.util.pan import dumps_masked
from socmint.util.timeutil import iso, now_iso, parse_ts

log = get_logger(__name__)

FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
SERPAPI_URL = "https://serpapi.com/search.json"
SERPAPI_TOP_N = 5
# serper.dev (a different service from SerpApi): POST JSON, key in the X-API-KEY header.
# Response shape verified live 2026-11-19 (tests/fixtures/evidence/live_serper_news_*.json):
# {"searchParameters": {...}, "news": [{title, link, date, source, imageUrl[, snippet]}],
#  "credits": 1}. `date` is relative ("2 days ago"), `source` is a plain string.
SERPER_NEWS_URL = "https://google.serper.dev/news"
SERPER_TOP_N = 5

_LOG_LOCK = threading.Lock()

# Run the configured providers concurrently with each other (one worker per provider).
# Each provider's own calls stay strictly sequential (CLAUDE.md §5: never parallelise
# requests against one account). False = the old fully sequential order (tests compare).
PARALLEL_PROVIDERS = True


# -------------------------------------------------------------------- evidence dates

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_REL_RX = re.compile(
    r"^\s*(an?|\d+)\s*(minutes?|mins?|m|hours?|hrs?|h|days?|d|weeks?|w|months?|years?|y)"
    r"\s+ago\s*$", re.IGNORECASE)
# Approximate length of each coarse unit. "1 month ago" is rendered by the provider
# for a range of ages, so day+ units are stored with date precision only.
_REL_UNIT = {"minute": timedelta(minutes=1), "hour": timedelta(hours=1),
             "day": timedelta(days=1), "week": timedelta(days=7),
             "month": timedelta(days=30), "year": timedelta(days=365)}
_MDY_RX = re.compile(r"^\s*([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\s*$")  # Oct 2, 2024
_DMY_RX = re.compile(r"^\s*(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\s*$")  # 2 Oct 2024
_YMD_RX = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")  # 2024-10-02
# SerpApi google_news `date`: "01/02/2026, 10:30 PM, +0700 +07" (documented example).
_SERPAPI_DATE_RX = re.compile(
    r"^\s*(\d{2}/\d{2}/\d{4}, \d{1,2}:\d{2} [AP]M, [+-]\d{4})(?:\s+\S+)?\s*$")


def _unit(u: str) -> str | None:
    u = u.lower()
    if u.startswith("mo"):
        return "month"
    if u.startswith("mi") or u == "m":
        return "minute"
    if u.startswith("h"):
        return "hour"
    if u.startswith("d"):
        return "day"
    if u.startswith("w"):
        return "week"
    if u.startswith("y"):
        return "year"
    return None


_MONTH_NAMES = {**_MONTHS, "sept": 9, **{n: i for i, n in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}}


def _month(name: str) -> int | None:
    # Exact month names/abbreviations only: "Decade 2, 2024" is not December.
    return _MONTH_NAMES.get(name.lower())


def parse_evidence_date(raw: Any, retrieved_at: str | None) -> str | None:
    """Parse a provider's publication date to UTC. Returns None when unparseable.

    - relative ("7 minutes ago", "2 hours ago", "3 days ago", "1 week ago",
      "2 months ago", "1 year ago") is resolved against `retrieved_at`; minutes/hours
      give an ISO datetime, day and coarser units give a date ("YYYY-MM-DD") only;
    - absolute "Oct 2, 2024", "2 Oct 2024", "2024-10-02" give a date;
    - ISO 8601 / RFC 2822 datetimes with a timezone give an ISO datetime.
    Nothing is guessed: a relative date without `retrieved_at`, a datetime without a
    timezone or any other format returns None.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.isdigit():
        # A bare number ("2024", "20241002") has no documented meaning for these
        # providers; reading it as unix epoch would be a guess (1970 dates).
        return None
    m = _REL_RX.match(s)
    if m:
        base = parse_ts(retrieved_at) if retrieved_at else None
        unit = _unit(m.group(2))
        if base is None or unit is None:
            return None
        n = 1 if m.group(1).lower() in ("a", "an") else int(m.group(1))
        try:
            dt = base - n * _REL_UNIT[unit]
        except (OverflowError, ValueError):  # "999999 years ago": never raise
            return None
        return iso(dt) if unit in ("minute", "hour") else dt.date().isoformat()
    for rx, order in ((_MDY_RX, "mdy"), (_DMY_RX, "dmy"), (_YMD_RX, "ymd")):
        m = rx.match(s)
        if not m:
            continue
        try:
            if order == "mdy":
                mo, d, y = _month(m.group(1)), int(m.group(2)), int(m.group(3))
            elif order == "dmy":
                d, mo, y = int(m.group(1)), _month(m.group(2)), int(m.group(3))
            else:
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return date(y, mo, d).isoformat() if mo else None
        except (TypeError, ValueError):
            return None
    m = _SERPAPI_DATE_RX.match(s)
    if m:
        try:
            return iso(datetime.strptime(m.group(1), "%m/%d/%Y, %I:%M %p, %z").astimezone(UTC))
        except ValueError:
            return None
    try:
        return iso(parse_ts(s))
    except (TypeError, ValueError, OverflowError, OSError):
        return None


# ------------------------------------------------------------------------ logging


def _slug(provider: str, query: str) -> str:
    h = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    return f"{provider}_{h}"


def _strip_secrets(text: str, cfg: dict[str, str] | None) -> str:
    """Replace only the literal configured secret values.

    The raw evidence file must stay the provider's actual response, so the generic
    pattern-based config.redact() (which rewrites e.g. "nextPageToken" values or a
    title containing "Key: ...") is NOT applied to it. The API key is never in
    `params` (callers remove it); this is defence in depth against an echoed key.
    """
    for k in SECRET_KEYS:
        v = (cfg or {}).get(k, "")
        if v and len(v) >= 6:
            text = text.replace(v, "[REDACTED]")
    return text


def _record(
    run_dir: Path | None,
    provider: str,
    query: str,
    params: dict[str, Any],
    raw: Any,
    evidence: list[Evidence],
    error: str | None,
    cfg: dict[str, str] | None = None,
) -> str | None:
    """Persist the query + raw response, append a summary line to evidence_log.jsonl.

    `params` must already have the API key removed by the caller.
    Each call gets its own raw file (`<provider>_<hash>_<NNN>.json`), so a repeated
    query never overwrites an earlier response that a log line points to.
    Returns the raw file path relative to run_dir (or None without run_dir).
    """
    if run_dir is None:
        return None
    try:
        raw_dir = Path(run_dir) / "raw" / "evidence"
        raw_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": provider,
            "query": query,
            "params": params,
            "retrieved_at": now_iso(),
            "error": error,
            "response": raw,
        }
        body = _strip_secrets(dumps_masked(payload, default=str, indent=1), cfg)
        with _LOG_LOCK:  # pick a free sequence number atomically within this process
            n = 1
            while (raw_dir / f"{_slug(provider, query)}_{n:03d}.json").exists():
                n += 1
            name = f"{_slug(provider, query)}_{n:03d}.json"
            (raw_dir / name).write_text(body, encoding="utf-8")
        rel = f"raw/evidence/{name}"
        line = {
            "at": now_iso(),
            "provider": provider,
            "query": query,
            "raw_ref": rel,
            "error": error,
            "results": [e.model_dump(mode="json") for e in evidence],
        }
        with _LOG_LOCK, (Path(run_dir) / "evidence_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(dumps_masked(line) + "\n")
        return rel
    except OSError as e:  # logging must never break assessment
        log.warning("could not write evidence log: %s", e)
        return None


# ------------------------------------------------------------------------- parsers


def parse_factcheck(payload: Any, retrieved_at: str) -> list[Evidence]:
    """Map a claims:search response to Evidence (one per claimReview).

    Field names (claims[].text/claimant/claimReview[].publisher.name/url/title/
    reviewDate/textualRating) are from the REST reference
    developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims.
    # UNVERIFIED: confirmed against documentation only, not a live response.
    """
    out: list[Evidence] = []
    if not isinstance(payload, dict):
        return out
    seen: set[str] = set()
    for claim in payload.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        claim_text = (claim.get("text") or "").strip()
        claimant = (claim.get("claimant") or "").strip()
        for rv in claim.get("claimReview") or []:
            if not isinstance(rv, dict):
                continue
            url = (rv.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            pub = rv.get("publisher") or {}
            publisher = (pub.get("name") or pub.get("site") or None) if isinstance(pub, dict) else None
            rating = (rv.get("textualRating") or "").strip()
            bits = [f"rating: {rating}" if rating else "rating: (none given)"]
            if claim_text:
                bits.append(f'reviewed claim: "{claim_text[:300]}"')
            if claimant:
                bits.append(f"claimant: {claimant}")
            if rv.get("reviewDate"):
                bits.append(f"reviewed: {rv['reviewDate']}")
            if claim.get("claimDate"):
                bits.append(f"claim date: {claim['claimDate']}")
            # The review's own publication date; the claim's date only as a fallback.
            date_raw = rv.get("reviewDate") or claim.get("claimDate") or None
            out.append(
                Evidence(
                    url=url,
                    source="google_factcheck",
                    supports="context",  # the verdict step decides, never the provider
                    note="; ".join(bits),
                    title=(rv.get("title") or None),
                    publisher=publisher,
                    retrieved_at=retrieved_at,
                    published_at=parse_evidence_date(date_raw, retrieved_at),
                    published_raw=str(date_raw) if date_raw else None,
                )
            )
    return out


def parse_serpapi_news(payload: Any, retrieved_at: str, top_n: int = SERPAPI_TOP_N) -> list[Evidence]:
    """Map a SerpApi engine=google_news response to Evidence (top N news results).

    # UNVERIFIED: field names (news_results[].link/title/source.name/date, and story
    # clusters carrying `highlight`/`stories`) are from serpapi.com/google-news-api docs,
    # not a live response.
    """
    out: list[Evidence] = []
    if not isinstance(payload, dict):
        return out
    seen: set[str] = set()

    def _add(item: dict) -> None:
        url = (item.get("link") or "").strip()
        if not url or url in seen or len(out) >= top_n:
            return
        seen.add(url)
        src = item.get("source")
        publisher = src.get("name") if isinstance(src, dict) else (src if isinstance(src, str) else None)
        date = item.get("iso_date") or item.get("date")
        out.append(
            Evidence(
                url=url,
                source="serpapi",
                supports="context",
                note=f"news search result; published: {date}" if date else "news search result",
                title=item.get("title") or None,
                publisher=publisher,
                retrieved_at=retrieved_at,
                published_at=parse_evidence_date(date, retrieved_at) if date else None,
                published_raw=str(date) if date else None,
            )
        )

    for item in payload.get("news_results") or []:
        if not isinstance(item, dict):
            continue
        if item.get("link"):
            _add(item)
        else:  # story cluster: take the highlighted / first story
            for sub in [item.get("highlight")] + list(item.get("stories") or []):
                if isinstance(sub, dict) and sub.get("link"):
                    _add(sub)
                    break
        if len(out) >= top_n:
            break
    return out


def parse_serper_news(payload: Any, retrieved_at: str, top_n: int = SERPER_TOP_N) -> list[Evidence]:
    """Map a serper.dev /news response to Evidence (top N results, as context)."""
    out: list[Evidence] = []
    if not isinstance(payload, dict):
        return out
    seen: set[str] = set()
    for item in payload.get("news") or []:
        if not isinstance(item, dict):
            continue
        url = (item.get("link") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        date = item.get("date")
        note = f"news search result; published: {date}" if date else "news search result"
        if item.get("snippet"):
            note += f"; snippet: {str(item['snippet'])[:300]}"
        src = item.get("source")
        out.append(Evidence(
            url=url, source="serper", supports="context", note=note,
            title=item.get("title") or None,
            publisher=src if isinstance(src, str) and src else None,
            retrieved_at=retrieved_at,
            published_at=parse_evidence_date(date, retrieved_at) if date else None,
            published_raw=str(date) if date else None))
        if len(out) >= top_n:
            break
    return out


# ----------------------------------------------------------------------- providers


def _fetch(url: str, params: dict[str, Any], client: httpx.Client | None,
           deadline: float | None) -> Any:
    own = client is None
    c = client or make_client(timeout=20.0)
    try:
        return request_json(c, "GET", url, params=params, max_retries=2, deadline=deadline)
    finally:
        if own:
            c.close()


def google_factcheck(
    query: str,
    cfg: dict[str, str],
    run_dir: Path | None = None,
    *,
    language: str | None = None,
    client: httpx.Client | None = None,
    deadline: float | None = None,
) -> list[Evidence]:
    """Google Fact Check Tools claims:search. Never raises."""
    key = (cfg.get("GOOGLE_FACTCHECK_API_KEY") or "").strip()
    if not key or not query.strip():
        return []
    public = {"query": query[:500], "pageSize": 10}
    if language:
        public["languageCode"] = language
    raw: Any = None
    error: str | None = None
    ev: list[Evidence] = []
    try:
        raw = _fetch(FACTCHECK_URL, {**public, "key": key}, client, deadline)
        ev = parse_factcheck(raw, now_iso())
    except (HttpFailure, httpx.HTTPError, ValueError, TypeError) as e:
        error = redact(f"{type(e).__name__}: {e}", cfg)
        log.warning("fact-check lookup failed: %s", error)
    except Exception as e:  # noqa: BLE001 - a provider must never abort assessment
        error = redact(f"{type(e).__name__}: {e}", cfg)
        log.warning("fact-check lookup failed unexpectedly: %s", error)
    _record(run_dir, "google_factcheck", query, public, raw, ev, error, cfg)
    return ev


def serpapi_news(
    query: str,
    cfg: dict[str, str],
    run_dir: Path | None = None,
    *,
    client: httpx.Client | None = None,
    deadline: float | None = None,
) -> list[Evidence]:
    """SerpApi Google News search, top 5 results as context. Never raises."""
    key = (cfg.get("SERPAPI_KEY") or "").strip()
    if not key or not query.strip():
        return []
    public = {"engine": "google_news", "q": query[:500]}
    raw: Any = None
    error: str | None = None
    ev: list[Evidence] = []
    try:
        raw = _fetch(SERPAPI_URL, {**public, "api_key": key}, client, deadline)
        ev = parse_serpapi_news(raw, now_iso())
    except (HttpFailure, httpx.HTTPError, ValueError, TypeError) as e:
        error = redact(f"{type(e).__name__}: {e}", cfg)
        log.warning("news search failed: %s", error)
    except Exception as e:  # noqa: BLE001
        error = redact(f"{type(e).__name__}: {e}", cfg)
        log.warning("news search failed unexpectedly: %s", error)
    _record(run_dir, "serpapi", query, public, raw, ev, error, cfg)
    return ev


def serper_news(
    query: str,
    cfg: dict[str, str],
    run_dir: Path | None = None,
    *,
    client: httpx.Client | None = None,
    deadline: float | None = None,
) -> list[Evidence]:
    """serper.dev Google News search, top 5 results as context. Never raises."""
    key = (cfg.get("SERPER_API_KEY") or "").strip()
    if not key or not query.strip():
        return []
    public = {"q": query[:500], "num": 10}
    raw: Any = None
    error: str | None = None
    ev: list[Evidence] = []
    own = client is None
    c = client or make_client(timeout=20.0)
    try:
        raw = request_json(c, "POST", SERPER_NEWS_URL, json_body=public,
                           headers={"X-API-KEY": key}, max_retries=2, deadline=deadline)
        ev = parse_serper_news(raw, now_iso())
    except (HttpFailure, httpx.HTTPError, ValueError, TypeError) as e:
        error = redact(f"{type(e).__name__}: {e}", cfg)
        log.warning("serper news search failed: %s", error)
    except Exception as e:  # noqa: BLE001
        error = redact(f"{type(e).__name__}: {e}", cfg)
        log.warning("serper news search failed unexpectedly: %s", error)
    finally:
        if own:
            c.close()
    _record(run_dir, "serper", query, public, raw, ev, error, cfg)
    return ev


Provider = Callable[..., list[Evidence]]
PROVIDERS: dict[str, Provider] = {"google_factcheck": google_factcheck, "serpapi": serpapi_news,
                                  "serper": serper_news}


def gather_evidence(
    query: str,
    cfg: dict[str, str],
    run_dir: Path | None = None,
    *,
    deadline: float | None = None,
) -> list[Evidence]:
    """Run every configured provider for one claim; de-duplicate by URL."""
    return gather_evidence_many([query], cfg, run_dir, deadline=deadline)[0][0]


def gather_evidence_many(
    queries: list[str],
    cfg: dict[str, str],
    run_dir: Path | None = None,
    *,
    deadline: float | None = None,
    parallel: bool | None = None,
) -> list[tuple[list[Evidence], list[str]]]:
    """Run every configured provider for every query.

    Providers run concurrently with each other (one worker thread per provider, when
    `parallel`), and each provider handles the queries strictly one after another in
    the given order, so no account ever receives parallel requests. Results are merged
    per query in the fixed PROVIDERS order and de-duplicated by URL, so the output is
    identical to the sequential mode. A provider stops issuing new calls once
    `deadline` (time.monotonic()) has passed.

    Returns, per query, (evidence, names of providers skipped because of the budget).
    """
    names = [n for n in PROVIDERS if n in available_providers(cfg)]
    use_parallel = PARALLEL_PROVIDERS if parallel is None else parallel

    def run_provider(name: str) -> list[list[Evidence] | None]:
        fn = PROVIDERS[name]
        res: list[list[Evidence] | None] = []
        for q in queries:
            if deadline is not None and time.monotonic() >= deadline:
                res.append(None)  # budget exhausted: not queried
                continue
            try:
                res.append(list(fn(q, cfg, run_dir, deadline=deadline)))
            except Exception as e:  # noqa: BLE001 - defence in depth
                log.warning("evidence provider %s raised: %s", name, redact(str(e), cfg))
                res.append([])
        return res

    if use_parallel and len(names) > 1:
        with ThreadPoolExecutor(max_workers=len(names)) as ex:
            per_provider = list(ex.map(run_provider, names))
    else:
        per_provider = [run_provider(n) for n in names]

    out: list[tuple[list[Evidence], list[str]]] = []
    for qi in range(len(queries)):
        merged: list[Evidence] = []
        seen: set[str] = set()
        skipped: list[str] = []
        for name, res in zip(names, per_provider, strict=True):
            items = res[qi]
            if items is None:
                skipped.append(name)
                continue
            for e in items:
                if e.url not in seen:
                    seen.add(e.url)
                    merged.append(e)
        out.append((merged, skipped))
    return out


def available_providers(cfg: dict[str, str]) -> list[str]:
    keys = {"google_factcheck": "GOOGLE_FACTCHECK_API_KEY", "serpapi": "SERPAPI_KEY",
            "serper": "SERPER_API_KEY"}
    return [n for n, k in keys.items() if (cfg.get(k) or "").strip()]
