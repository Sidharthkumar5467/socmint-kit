"""Collector interface (docs/SPEC.md §5.1).

A collector NEVER raises for network or credential reasons: it returns a
`CollectResult` whose `status` explains what happened (ok, partial, skipped, failed...).
The pipeline still wraps every collector in try/except and converts any escape into a
`failed` status, so one broken source can never abort a run.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from socmint.schema import Post, SourceState, SourceStatus
from socmint.util.pan import dumps_masked, mask_text_for


@dataclass(frozen=True)
class Capabilities:
    supports_date_filter: bool
    needs_credentials: bool
    kind: Literal["social", "messaging", "video", "ads", "plugin"]
    credential_keys: tuple[str, ...] = ()  # config keys that must be non-empty to run
    description: str = ""


@dataclass
class Query:
    topic: str
    start: datetime  # tz-aware UTC, inclusive (00:00:00 of start date)
    end: datetime  # tz-aware UTC, inclusive (23:59:59 of end date)
    limit: int = 500  # soft per-source cap on posts
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectContext:
    """Runtime services handed to a collector by the pipeline."""

    run_dir: Path
    raw_dir: Path  # raw payloads go here; reference them via Post.raw_ref
    config: dict[str, str]  # resolved credentials/settings (see socmint.config)
    # time.monotonic() after which the collector must stop -> partial. math.inf when the
    # run has no time limit (the default, RUN_TIME_BUDGET_SECONDS=0).
    deadline: float
    progress: Callable[[str, dict[str, Any]], None] = lambda source, info: None
    # Set when the analyst presses "Stop scan" in the GUI: every collector then stops as
    # if its time were up (out_of_time() is True, time_left() is 0). The pipeline then
    # discards the whole run (no analysis or report; the partial run directory is deleted).
    stop: threading.Event = field(default_factory=threading.Event)

    def time_left(self) -> float:
        if self.stop.is_set():
            return 0.0
        return self.deadline - time.monotonic()

    def out_of_time(self) -> bool:
        return self.stop.is_set() or time.monotonic() >= self.deadline

    def save_raw(self, source: str, name: str, payload: Any) -> str:
        """Persist a raw payload; returns a path relative to run_dir for Post.raw_ref.
        Payment card numbers in text payloads are masked before writing (util.pan)."""
        d = self.raw_dir / source
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        if isinstance(payload, (bytes, bytearray)):
            try:
                p.write_text(mask_text_for(name, bytes(payload).decode("utf-8")),
                             encoding="utf-8")
            except UnicodeDecodeError:  # binary (e.g. media): nothing to mask
                p.write_bytes(payload)
        elif isinstance(payload, str):
            p.write_text(mask_text_for(name, payload), encoding="utf-8")
        else:  # JSON: only string values are masked, never numeric ids (util.pan)
            p.write_text(dumps_masked(payload, default=str), encoding="utf-8")
        return p.relative_to(self.run_dir).as_posix()


@dataclass
class CollectResult:
    posts: list[Post]
    status: list[SourceStatus]  # one per source × query (a collector may run several)


@runtime_checkable
class Collector(Protocol):
    name: str

    def capabilities(self) -> Capabilities: ...

    def collect(self, query: Query, ctx: CollectContext) -> CollectResult: ...


def skipped(source: str, query: str, state: SourceState, reason: str) -> CollectResult:
    """A result with no posts and an explanatory status."""
    return CollectResult(
        posts=[], status=[SourceStatus(source=source, query=query, state=state, reason=reason)]
    )
