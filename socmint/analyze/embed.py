"""Text embeddings for narrative clustering (docs/SPEC.md §6, Narratives).

Primary: a multilingual sentence-transformers model (cfg EMBEDDING_MODEL, default
paraphrase-multilingual-MiniLM-L12-v2) so posts in different languages that say the
same thing land near each other.

Loading order (never blocks a run on the network):
1. load from the local HF cache only (`local_files_only=True`, no network);
2. if not cached and env SOCMINT_ALLOW_MODEL_DOWNLOAD != "0", download it, guarded by a
   timeout (SOCMINT_MODEL_DOWNLOAD_TIMEOUT seconds, default 90);
3. on any failure (package missing, offline, timeout), fall back to TF-IDF over
   character 3-5 grams (`char_wb`, robust across scripts and languages) reduced with
   TruncatedSVD (<=128 dims).

Setting SOCMINT_EMBEDDINGS=tfidf forces the fallback (used by tests and replay).
Every vector returned is L2-normalised, so euclidean distance is monotone in cosine
distance. The method used is returned alongside the vectors and recorded by callers.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any

import numpy as np

from socmint.util.log import get_logger

log = get_logger(__name__)

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SVD_DIMS = 128
TFIDF_METHOD = "tfidf-char_wb-3-5"

# Module-level cache: model name -> loaded model, or None if loading already failed
# in this process (so we do not retry a slow/failed download for every call).
_MODELS: dict[str, Any] = {}
_LOCK = threading.Lock()


def model_name(cfg: dict[str, str] | None = None) -> str:
    if cfg is None:
        try:
            from socmint.config import load_config

            cfg = load_config()
        except Exception:  # noqa: BLE001 - config problems must not break embeddings
            cfg = {}
    return (cfg.get("EMBEDDING_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _try_load(name: str, *, local_only: bool) -> Any:
    from sentence_transformers import SentenceTransformer  # heavy import, lazy

    return SentenceTransformer(name, device="cpu", local_files_only=local_only)


def _load_with_timeout(name: str, timeout: float) -> Any:
    """Download + load in a worker thread; give up (fall back) after `timeout`s.

    The worker may keep downloading in the background after we give up; the result is
    then simply unused in this process.
    """
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["model"] = _try_load(name, local_only=False)
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=work, name="embed-model-download", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"model download exceeded {timeout:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["model"]


def _load_model(name: str, *, allow_download: bool | None = None) -> Any:
    """Return a SentenceTransformer or None. Cached per process."""
    with _LOCK:
        if name in _MODELS:
            return _MODELS[name]
        model = None
        try:
            model = _try_load(name, local_only=True)
            log.info("embedding model loaded from local cache: %s", name)
        except ImportError:
            log.warning("sentence-transformers not installed; using TF-IDF embeddings")
        except Exception as e:  # noqa: BLE001 - not cached / corrupt cache
            if allow_download is None:
                allow_download = os.environ.get("SOCMINT_ALLOW_MODEL_DOWNLOAD", "1") != "0"
            if allow_download:
                timeout = float(os.environ.get("SOCMINT_MODEL_DOWNLOAD_TIMEOUT", "90") or 90)
                try:
                    log.info("embedding model not cached (%s); downloading %s", type(e).__name__, name)
                    model = _load_with_timeout(name, timeout)
                except Exception as e2:  # noqa: BLE001
                    log.warning("embedding model download failed (%s); using TF-IDF", e2)
            else:
                log.info("embedding model not cached and download disabled; using TF-IDF")
        _MODELS[name] = model
        return model


def prefetch_model(cfg: dict[str, str] | None = None) -> tuple[bool, str]:
    """Download/cache the embedding model ahead of a run (`socmint doctor --prefetch`).

    Returns (ok, message). Does not raise.
    """
    name = model_name(cfg)
    with _LOCK:
        _MODELS.pop(name, None)  # retry even if an earlier attempt failed
    model = _load_model(name, allow_download=True)
    if model is None:
        return False, f"could not load {name}; narratives will use TF-IDF char n-gram fallback"
    return True, f"{name} cached and loadable"


def _clean(text: str) -> str:
    return _URL_RE.sub(" ", text or "").strip()


def tfidf_embed(texts: list[str]) -> np.ndarray:
    """Multilingual-robust lexical fallback: char_wb 3-5 grams + SVD, L2-normalised."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [_clean(t).lower() or "_" for t in texts]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
    x = vec.fit_transform(docs)
    # Reduce only when it is a real reduction: with <= SVD_DIMS documents the full
    # TF-IDF matrix is small, and a rank-deficient SVD would distort distances
    # (observed in tests: two unrelated 5-post groups collapsed to distance 0.01).
    if x.shape[0] > SVD_DIMS and x.shape[1] > SVD_DIMS:
        arr = TruncatedSVD(n_components=SVD_DIMS, random_state=0).fit_transform(x)
    else:
        arr = x.toarray()
    return _normalize(np.asarray(arr, dtype=np.float32))


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_texts(texts: list[str], cfg: dict[str, str] | None = None) -> tuple[np.ndarray, str]:
    """Embed texts. Returns (L2-normalised float32 matrix [n, d], method description)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), "none"
    if os.environ.get("SOCMINT_EMBEDDINGS", "").lower() != "tfidf":
        name = model_name(cfg)
        model = _load_model(name)
        if model is not None:
            try:
                arr = model.encode(
                    [_clean(t) or "_" for t in texts],
                    batch_size=64,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                return _normalize(np.asarray(arr, dtype=np.float32)), f"sentence-transformers:{name}"
            except Exception as e:  # noqa: BLE001
                log.warning("embedding with %s failed (%s); using TF-IDF", name, e)
    return tfidf_embed(texts), TFIDF_METHOD
