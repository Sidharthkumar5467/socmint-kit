"""Perceptual hashes for coordination's reused-image signal (owner request 2026-11-20).

`analyze.coordination` flags a "reused_image" link two ways:
  * identical image URL used by different authors — matched with no download, always on;
  * the SAME image re-uploaded under a DIFFERENT URL (or re-encoded) — needs a perceptual
    hash of each image, which is what this module provides.

Before this, nothing computed those hashes, so the second path never fired. Here we download
the candidate profile pictures and media thumbnails that are already in the collected posts,
compute a perceptual hash (imagehash.phash) for each, and return {url: hex}. coordination
then compares the hashes (Hamming distance) to link accounts whose images look the same.

Strictly bounded and best-effort:
  * placeholders (default avatars, silhouettes) are skipped — many unrelated accounts share
    them, so they are never evidence;
  * avatars are hashed first (the classic sock-puppet signal), then media thumbnails;
  * at most `cap` distinct URLs, each with a short per-request timeout, under one wall-clock
    budget; any download/decode failure drops that URL and never fails the run;
  * only the hash is kept — no image is stored, and the GET carries a generic User-Agent and
    none of the user's data.

Network, so it is skipped in replay/offline; the pipeline passes the result to
coordination.detect_coordination only for a live run or an explicit re-analysis.
"""

from __future__ import annotations

import io
import time
import urllib.request
from collections.abc import Callable

from socmint.analyze.coordination import _image_urls
from socmint.schema import Post
from socmint.util.log import get_logger

log = get_logger(__name__)

# Generic client name only — never the user's name, project or email (privacy rule).
USER_AGENT = "socmint-image-hash/1.0"
DEFAULT_CAP = 250          # distinct image URLs hashed per run
DEFAULT_TIMEOUT_S = 5.0    # per image
DEFAULT_BUDGET_S = 90.0    # all downloads in one run
MAX_BYTES = 8 * 1024 * 1024  # do not read more than 8 MiB of any one image


def candidate_urls(posts: list[Post], cap: int = DEFAULT_CAP) -> list[str]:
    """Distinct non-placeholder image URLs worth hashing, avatars first then media, so a
    small cap still covers the profile pictures that carry the sock-puppet signal."""
    avatars: list[str] = []
    media: list[str] = []
    seen: set[str] = set()
    for p in posts:
        for url, kind in _image_urls(p):
            if url in seen:
                continue
            seen.add(url)
            (avatars if kind == "avatar" else media).append(url)
    return (avatars + media)[:cap]


def _fetch(url: str, timeout: float) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 - http(s) only below
    if not url.lower().startswith(("http://", "https://")):
        return None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read(MAX_BYTES + 1)
    except Exception as e:  # noqa: BLE001 - a bad URL never fails the run
        log.debug("image fetch failed for %s: %s", url, type(e).__name__)
        return None


def compute_image_hashes(
    posts: list[Post],
    *,
    cap: int = DEFAULT_CAP,
    timeout: float = DEFAULT_TIMEOUT_S,
    budget_s: float = DEFAULT_BUDGET_S,
    fetch: Callable[[str, float], bytes | None] | None = None,
) -> dict[str, str]:
    """{image URL -> hex perceptual hash} for the candidate images, within the caps above.

    `fetch(url, timeout) -> bytes|None` is injectable for tests; the default downloads over
    HTTP(S). Returns {} (and logs once) when Pillow/imagehash are unavailable."""
    urls = candidate_urls(posts, cap)
    if not urls:
        return {}
    try:
        import imagehash
        from PIL import Image
    except Exception as e:  # noqa: BLE001
        log.warning("perceptual image hashing unavailable (%s); reused-image uses identical "
                    "URLs only", type(e).__name__)
        return {}

    get = fetch or _fetch
    out: dict[str, str] = {}
    t0 = time.monotonic()
    fetched = decoded = 0
    for url in urls:
        if time.monotonic() - t0 > budget_s:
            log.info("image hashing stopped at the %.0fs budget: %d/%d URLs hashed",
                     budget_s, len(out), len(urls))
            break
        data = get(url, timeout)
        if not data or len(data) > MAX_BYTES:
            continue
        fetched += 1
        try:
            with Image.open(io.BytesIO(data)) as im:
                out[url] = str(imagehash.phash(im.convert("RGB")))
            decoded += 1
        except Exception as e:  # noqa: BLE001 - not an image / truncated
            log.debug("image decode failed for %s: %s", url, type(e).__name__)
    log.info("image hashing: %d URL(s), %d fetched, %d hashed", len(urls), fetched, decoded)
    return out
