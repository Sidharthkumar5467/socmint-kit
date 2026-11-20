"""Background run jobs for the GUI: one active pipeline run per server process.

`pipeline.run` creates the run directory and computes the run_id itself (from the topic
and its own creation timestamp), so the id cannot be precomputed reliably. Instead the
job passes a progress callback and waits for the first event carrying `run_id` (the
pipeline emits it right after creating the directory), then returns the id to the HTTP
handler immediately while the run continues in its thread.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from socmint.config import redact
from socmint.util.log import get_logger

log = get_logger(__name__)


# progress.json stages during which "Stop scan" applies (pipeline.run emits
# stage="collect" first, then "verify" once every collector has returned).
COLLECT_STAGES = (None, "starting", "collect")


def _read_progress(run_dir: Path) -> dict[str, Any]:
    import json

    try:
        data = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


class JobBusy(RuntimeError):
    """A run is already in progress in this process."""


class JobStartError(RuntimeError):
    """The pipeline failed before it announced a run_id."""


@dataclass
class Job:
    run_id: str | None = None
    run_dir: Path | None = None
    thread: threading.Thread | None = None
    error: str | None = None
    done: bool = False
    cancelled: bool = False  # the analyst pressed Stop; the run was abandoned, nothing saved
    started: threading.Event = field(default_factory=threading.Event)
    # Set by "Stop scan" (JobManager.stop): collectors stop at once and the whole run is
    # abandoned (no analysis, no report, partial run deleted). Re-analysis jobs ignore it.
    stop: threading.Event = field(default_factory=threading.Event)
    kind: str = "run"  # "run" | "reanalyze"

    @property
    def active(self) -> bool:
        return not self.done


class JobManager:
    def __init__(self, runner: Callable[..., Path] | None = None,
                 reanalyzer: Callable[..., Path] | None = None):
        # `runner` / `reanalyzer` are injectable for tests; they default to
        # socmint.pipeline.run / .reanalyze (lazy import keeps server start-up light).
        self._runner = runner
        self._reanalyzer = reanalyzer
        self._lock = threading.Lock()
        self.current: Job | None = None

    def busy(self) -> bool:
        with self._lock:
            return self.current is not None and self.current.active

    def start(self, *, topic: str, start: date, end: date, sources: list[str], limit: int,
              out_root: Path, cfg: dict[str, str], timeout: float = 60.0,
              options: dict[str, Any] | None = None) -> Job:
        """Start a run in a daemon thread; block only until its run_id is known."""
        with self._lock:
            if self.current is not None and self.current.active:
                raise JobBusy(f"run {self.current.run_id} is still in progress")
            job = Job()
            self.current = job

        runner = self._runner
        if runner is None:
            from socmint.pipeline import run as runner

        def on_progress(ev: dict[str, Any]) -> None:
            if job.run_id is None and ev.get("run_id"):
                job.run_id = str(ev["run_id"])
                job.run_dir = out_root / job.run_id
                job.started.set()

        def work() -> None:
            try:
                run_dir = runner(topic, start, end, sources, limit=limit,
                                 options=dict(options or {}), progress=on_progress,
                                 out_root=out_root, cfg=cfg, stop=job.stop)
                if run_dir is None and job.stop.is_set():
                    # Hard cancel: pipeline.run abandoned the run and deleted its directory.
                    job.cancelled = True
                elif job.run_id is None and run_dir is not None:
                    job.run_dir = Path(run_dir)
                    job.run_id = Path(run_dir).name
            except Exception as e:  # noqa: BLE001 - surface to the GUI, never crash the server
                job.error = redact(f"{type(e).__name__}: {e}", cfg)
                log.error("run job failed: %s", redact(traceback.format_exc(), cfg))
            finally:
                job.done = True
                job.started.set()

        job.thread = threading.Thread(target=work, name="socmint-run", daemon=True)
        job.thread.start()
        job.started.wait(timeout)
        if job.run_id is None:
            raise JobStartError(job.error or "run did not start within the time limit")
        return job

    def start_reanalyze(self, run_dir: Path, cfg: dict[str, str]) -> Job:
        """Re-run only the analysis of a saved run in a daemon thread (same one-job rule).

        The run_id is the directory name, so this returns at once; progress is read from
        the run's progress.json like a normal run.
        """
        with self._lock:
            if self.current is not None and self.current.active:
                raise JobBusy(f"run {self.current.run_id} is still in progress")
            job = Job(run_id=Path(run_dir).name, run_dir=Path(run_dir), kind="reanalyze")
            self.current = job
        reanalyzer = self._reanalyzer
        if reanalyzer is None:
            from socmint.pipeline import reanalyze as reanalyzer

        def work() -> None:
            try:
                reanalyzer(Path(run_dir), cfg=cfg)
            except Exception as e:  # noqa: BLE001 - surface to the GUI, never crash the server
                job.error = redact(f"{type(e).__name__}: {e}", cfg)
                log.error("reanalyze job failed: %s", redact(traceback.format_exc(), cfg))
            finally:
                job.done = True
                job.started.set()

        job.thread = threading.Thread(target=work, name="socmint-reanalyze", daemon=True)
        job.thread.start()
        return job

    def stop(self, run_id: str) -> bool:
        """Ask the active collection run `run_id` to cancel: collectors stop at once and
        the whole run is abandoned (no analysis, no report, partial run deleted). True if
        the request was registered (or already was); False if no such run is collecting:
        not active, a re-analysis, or past the collect stage (progress.json says so)."""
        with self._lock:
            job = self.current
            if job is None or not job.active or job.kind != "run" or job.run_id != run_id:
                return False
            if job.run_dir is not None:
                prog = _read_progress(job.run_dir)
                if prog.get("stage") not in COLLECT_STAGES:
                    return False
            job.stop.set()
            return True

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current job finishes (tests, CLI)."""
        job = self.current
        if job and job.thread:
            job.thread.join(timeout)
