"""HTTP with polite retry/backoff. Sequential by design: never parallelise one account.

`request_json` / `request_text` retry on 429 and 5xx, honouring Retry-After, and stop
early when the caller's deadline would be exceeded. Errors surface as `HttpFailure`
with a redacted message so collectors can turn them into a SourceStatus reason.
"""

from __future__ import annotations

import random
import ssl
import threading
import time
from typing import Any

import httpx

from socmint.config import redact
from socmint.util.log import get_logger

log = get_logger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class HttpFailure(Exception):
    def __init__(self, message: str, status: int | None = None, rate_limited: bool = False):
        super().__init__(redact(message))
        self.status = status
        self.rate_limited = rate_limited


def _ssl_context() -> ssl.SSLContext | bool:
    """Verify TLS against the OS trust store (truststore), falling back to certifi.

    Corporate networks (e.g. Cloudflare Gateway, Zscaler) re-sign TLS with a CA that the
    OS trusts but certifi does not; without this, such a laptop sees every request fail
    with CERTIFICATE_VERIFY_FAILED (observed live against t.me, see docs/DECISIONS.md).
    Verification is never disabled.
    """
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:  # noqa: BLE001 — truststore missing/unsupported: default verification
        return True


def make_client(timeout: float = 30.0, headers: dict[str, str] | None = None) -> httpx.Client:
    h = {"user-agent": DEFAULT_UA}
    if headers:
        h.update(headers)
    return httpx.Client(timeout=timeout, headers=h, follow_redirects=True, verify=_ssl_context())


def _retry_after(resp: httpx.Response) -> float | None:
    ra = resp.headers.get("retry-after")
    if not ra:
        return None
    try:
        return float(ra)
    except ValueError:
        return None


def request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: Any = None,
    headers: dict[str, str] | None = None,
    max_retries: int = 4,
    deadline: float | None = None,
    base_delay: float = 2.0,
    stop: threading.Event | None = None,
) -> httpx.Response:
    """`stop` (the run's analyst-stop event) ends a back-off wait at once: with no run
    time limit a long Retry-After is honoured, but the analyst can still stop the run."""
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.request(method, url, params=params, json=json_body, headers=headers)
        except httpx.HTTPError as e:  # network-level
            if attempt > max_retries:
                raise HttpFailure(f"network error: {type(e).__name__}: {e}") from e
            wait = base_delay * 2 ** (attempt - 1)
        else:
            if resp.status_code < 400:
                return resp
            retryable = resp.status_code == 429 or resp.status_code >= 500
            if not retryable or attempt > max_retries:
                body = resp.text[:300]
                raise HttpFailure(
                    f"HTTP {resp.status_code} from {httpx.URL(url).host}: {body}",
                    status=resp.status_code,
                    rate_limited=resp.status_code == 429,
                )
            wait = _retry_after(resp) or base_delay * 2 ** (attempt - 1)
            log.warning("HTTP %s from %s; backing off %.1fs", resp.status_code,
                        httpx.URL(url).host, wait)
        wait += random.uniform(0, 0.5)
        if deadline is not None and time.monotonic() + wait > deadline:
            raise HttpFailure("time budget exhausted while backing off", rate_limited=True)
        if stop is not None:
            if stop.wait(wait):
                raise HttpFailure("stopped by the analyst while backing off", rate_limited=True)
        else:
            time.sleep(wait)


def request_json(client: httpx.Client, method: str, url: str, **kw: Any) -> Any:
    resp = request(client, method, url, **kw)
    try:
        return resp.json()
    except ValueError as e:
        raise HttpFailure(f"non-JSON response from {httpx.URL(url).host}: {resp.text[:200]}") from e
