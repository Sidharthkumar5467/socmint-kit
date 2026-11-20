"""Shared test fixtures. Tests never touch the network or the user's real config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from socmint.schema import Post

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point config + runs at a temp dir and strip credential env vars."""
    monkeypatch.setenv("SOCMINT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SOCMINT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.chdir(tmp_path)
    from socmint.config import DEFAULTS

    for k in DEFAULTS:
        if k in os.environ:
            monkeypatch.delenv(k, raising=False)
    yield


def make_post(
    native_id: str,
    text: str = "",
    *,
    source: str = "x",
    author: str = "a1",
    published_at: str | None = "2026-11-01T10:00:00Z",
    verified: bool = True,
    **kw,
) -> Post:
    return Post(
        post_id=f"{source}:{native_id}",
        source=source,
        author_id=author,
        author_handle=kw.pop("author_handle", f"@{author}"),
        text=text,
        published_at=published_at,
        collected_at=kw.pop("collected_at", "2026-11-18T00:00:00Z"),
        timestamp_verified=verified,
        url=kw.pop("url", f"https://example.test/{source}/{native_id}"),
        **kw,
    )


@pytest.fixture
def post_factory():
    return make_post
