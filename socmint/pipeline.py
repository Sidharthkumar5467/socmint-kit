"""Run orchestrator (docs/SPEC.md §4).

collect → normalize → verify timestamps → dedupe → analyze → export → report.
Every stage writes under runs/<id>/ so any stage can be re-run from disk.
A run never aborts because one source or one analysis step fails: failures are recorded
in source_status.json / run.json and the remaining stages still run.

Stage contracts (implemented in their modules):
  analyze.narratives.build_narratives(posts, *, topic, llm) -> list[Narrative]
  analyze.coordination.detect_coordination(posts) -> list[Cluster]
  analyze.origin.trace_origins(posts, narratives) -> list[OriginTrace]
  analyze.claims.assess_claims(posts, narratives, *, topic, llm, cfg, run_dir,
                                window_start=None, window_end=None) -> list[Claim]
  analyze.brief.build_brief(posts, narratives, clusters, traces, claims, statuses,
                            meta) -> Brief        (rule-based run summary, brief.json)
  export.write_exports(run_dir) -> None          (reads files already in run_dir)
  report.build_report(run_dir) -> Path

reanalyze(run_dir) re-runs only the analysis stages (narratives → brief) on a run's
stored posts.jsonl + source_status.json + run.json, then export + report. No collection.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from socmint import __version__, status_text
from socmint.collect import DEFAULT_SOURCES, get_collector
from socmint.config import has, load_config, max_window_days, redact, run_time_budget, runs_dir
from socmint.interfaces import CollectContext, CollectResult, Query
from socmint.runstore import (
    FILES,
    read_json,
    read_jsonl,
    run_in_progress,
    write_json,
    write_jsonl,
)
from socmint.schema import Brief, Post, RunMeta, SourceState, SourceStatus
from socmint.util.log import get_logger
from socmint.util.pan import mask_file, mask_post, mask_tree
from socmint.util.timeutil import day_end, day_start, iso, now_iso, parse_ts
from socmint.verify import analysable, in_scope, run_verification

log = get_logger(__name__)

ProgressFn = Callable[[dict[str, Any]], None]

# Share of the overall run budget given to collection (sources run concurrently).
COLLECT_SHARE = 0.65


class RunInputError(ValueError):
    pass


def slugify(s: str, n: int = 40) -> str:
    s = re.sub(r"[^\w\-]+", "-", s.strip().lower(), flags=re.UNICODE).strip("-")
    return (s or "topic")[:n]


def make_run_id(topic: str, created: str) -> str:
    h = hashlib.sha1(f"{topic}|{created}".encode()).hexdigest()[:6]
    return f"{created[:10]}-{slugify(topic)}-{h}"


# Window presets (owner decision 2026-11-20, docs/DECISIONS.md): the New Run page offers
# "last N days" windows ending now, each with its own per-source post cap, to limit API
# and LLM usage. A preset window has exact UTC datetime bounds (end = now, start = end -
# N days), so "1 day" is the last 24 hours; every collector takes datetime bounds
# (window.time_slices clips the first and last UTC day). A custom range keeps whole UTC
# days and CUSTOM_LIMIT unless the analyst sets a limit.
WINDOW_PRESETS: dict[str, int] = {"1d": 1, "3d": 3, "7d": 7}
PRESET_LIMITS: dict[str, int] = {"1d": 150, "3d": 300, "7d": 500}
DEFAULT_PRESET = "3d"
CUSTOM_PRESET = "custom"
# A custom range's default per-source post limit grows with its length (owner decision
# 2026-11-20): CUSTOM_PER_DAY posts per day, at least CUSTOM_LIMIT and at most
# CUSTOM_LIMIT_MAX (a 30-day range gets 2,000). The analyst can still type any limit.
CUSTOM_LIMIT = 500
CUSTOM_PER_DAY = 70
CUSTOM_LIMIT_MAX = 2000
# LinkedIn Ads lookback (owner decision 2026-11-20): the Ad Library is searched over a
# longer span than the post window, up to its own ~12-month depth. 0 = the post window.
# Ads found only in the longer span are analysed (narratives, coordination) but never
# enter timelines, counts or first-observed analysis (verify.in_scope / analysable).
# ADS_LOOKBACK_ALL = "all time": the Ad Library is searched with no date filter, so every
# ad it lists for the keyword is collected, as the owner's LinkedIn Ads tool does (owner
# request 2026-11-20; live: "sahibinden" listed 54 ads with no filter vs 13 for 1-30 Oct).
# An option; the default is the window's own dates (owner request 2026-11-20, later).
ADS_LOOKBACK_ALL = 3650
ADS_LOOKBACK_CHOICES: tuple[int, ...] = (0, 90, 180, 365, ADS_LOOKBACK_ALL)
ADS_LOOKBACK_MAX = 365
ADS_EPOCH = datetime(2000, 1, 1, tzinfo=UTC)  # "any time" start for the ads span


def custom_limit(days: int) -> int:
    """Default per-source post limit for a custom range of `days` days."""
    return max(CUSTOM_LIMIT, min(CUSTOM_LIMIT_MAX, CUSTOM_PER_DAY * max(1, int(days))))


def ads_window_start(start_at: datetime, end_at: datetime, lookback_days: int | None) -> datetime:
    """Start of the LinkedIn Ads search span: the post window start, or `lookback_days`
    before the window end when that is earlier (clamped to ADS_LOOKBACK_MAX)."""
    lb = int(lookback_days or 0)
    if lb <= 0:
        return start_at
    lb = min(lb, ADS_LOOKBACK_MAX)
    return min(start_at, end_at - timedelta(days=lb))


# Reason fragments collectors write when their time runs out; after an analyst stop they
# are reworded, because the stop (not a time limit) ended collection.
_TIME_UP = ("time budget exhausted",)
STOPPED_BY_ANALYST = "stopped by the analyst"


class _Cancelled(Exception):
    """Raised internally when the analyst pressed Stop: the run is abandoned and its
    partial directory deleted (no analysis, export or report)."""


def mark_analyst_stop(statuses: list[SourceStatus]) -> int:
    """After an analyst stop: reword time-limit stop reasons as an analyst stop, and make
    a status that failed only because of the stop (e.g. an HTTP back-off cut short)
    partial, never failed. Returns how many statuses changed."""
    n = 0
    for st in statuses:
        r = st.reason or ""
        new = r
        for frag in _TIME_UP:
            new = new.replace(frag, STOPPED_BY_ANALYST)
        changed = new != r
        if changed:
            st.reason = new
        if st.state == SourceState.failed and STOPPED_BY_ANALYST in new:
            st.state = SourceState.partial
            changed = True
        n += changed
    return n


def preset_window(preset: str, now: datetime | None = None) -> tuple[datetime, datetime]:
    """(start, end) for a window preset: end = now (UTC, whole seconds), start = end - N
    days. Both bounds are inclusive datetimes. Raises RunInputError for an unknown preset."""
    if preset not in WINDOW_PRESETS:
        raise RunInputError(f"unknown window preset {preset!r} (use one of: "
                            f"{', '.join(WINDOW_PRESETS)})")
    end = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    return end - timedelta(days=WINDOW_PRESETS[preset]), end


def _bound(v: date, *, is_end: bool) -> datetime:
    """A window bound as an aware UTC datetime: a date is the start/end of its UTC day."""
    if isinstance(v, datetime):
        return v.replace(tzinfo=UTC) if v.tzinfo is None else v.astimezone(UTC)
    return day_end(v) if is_end else day_start(v)


def _day(v: date) -> date:
    return _bound(v, is_end=False).date() if isinstance(v, datetime) else v


def window_days(start: date, end: date) -> int:
    """Length of a window in days: whole UTC days for dates (inclusive), rounded up for
    exact datetime bounds (a 7-day preset is 7 days even though it touches 8 UTC days)."""
    if isinstance(start, datetime) or isinstance(end, datetime):
        span = _bound(end, is_end=True) - _bound(start, is_end=False)
        return max(1, -(-int(span.total_seconds()) // 86400))
    return (end - start).days + 1


def validate_inputs(topic: str, start: date, end: date, max_days: int) -> None:
    """start/end are dates (whole UTC days) or datetimes (exact bounds, window presets)."""
    if not topic.strip():
        raise RunInputError("topic is required")
    if _bound(end, is_end=True) < _bound(start, is_end=False):
        raise RunInputError("end date is before start date")
    today = datetime.now(UTC).date()
    if _day(start) > today:
        raise RunInputError(f"start date {_day(start).isoformat()} is in the future "
                            f"(today is {today.isoformat()}, UTC)")
    days = window_days(start, end)
    if days > max_days:
        raise RunInputError(f"window is {days} days; maximum is {max_days}")


class Progress:
    """Collects progress events and mirrors them to progress.json for the GUI."""

    def __init__(self, run_dir: Path, sink: ProgressFn | None):
        self.path = run_dir / FILES["progress"]
        self.sink = sink
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {"stage": "starting", "sources": {}, "log": [],
                                      "done": False, "cost_usd": 0.0, "pid": os.getpid()}

    def emit(self, **ev: Any) -> None:
        with self.lock:
            ev.setdefault("at", now_iso())
            if "source" in ev:
                cur = self.state["sources"].setdefault(ev["source"], {})
                cur.update({k: v for k, v in ev.items() if k != "source"})
            if "stage" in ev and "source" not in ev:
                self.state["stage"] = ev["stage"]
            if "cost_usd" in ev and "source" in ev:
                self.state["cost_usd"] = round(sum(
                    float(s.get("cost_usd") or 0) for s in self.state["sources"].values()), 4)
            if ev.get("message"):
                self.state["log"].append(f"{ev['at']} {ev['message']}")
                self.state["log"] = self.state["log"][-200:]
            for k in ("done", "error", "run_id"):
                if k in ev:
                    self.state[k] = ev[k]
            write_json(self.path, self.state)
        if self.sink:
            try:
                self.sink(ev)
            except Exception:  # noqa: BLE001
                pass


def _collect_one(name: str, query: Query, ctx: CollectContext, prog: Progress) -> CollectResult:
    prog.emit(source=name, state="collecting", count=0, message=f"{name}: collecting")
    try:
        collector = get_collector(name)
        caps = collector.capabilities()
        if caps.needs_credentials and not has(ctx.config, *caps.credential_keys):
            missing = [k for k in caps.credential_keys if not str(ctx.config.get(k, "")).strip()]
            res = CollectResult(posts=[], status=[SourceStatus(
                source=name, query=query.topic, state=SourceState.skipped_no_credentials,
                reason=f"missing credential(s): {', '.join(missing)}")])
        else:
            res = collector.collect(query, ctx)
            if ctx.stop.is_set():  # reworded before the live progress shows the reason
                mark_analyst_stop(res.status)
    except Exception as e:  # noqa: BLE001 — a collector must never take the run down
        log.error("collector %s crashed: %s", name, redact(traceback.format_exc()))
        res = CollectResult(posts=[], status=[SourceStatus(
            source=name, query=query.topic, state=SourceState.failed,
            reason=redact(f"{type(e).__name__}: {e}"))])
    for st in res.status:
        st.finished_at = st.finished_at or now_iso()
    status_text.apply(res.status)
    head = status_text.source_headline(res.status)
    states = sorted({s.state.value for s in res.status})
    cost = sum(s.cost_usd for s in res.status)
    prog.emit(source=name, state=",".join(states), count=len(res.posts), cost_usd=cost,
              reason="; ".join(s.reason for s in res.status if s.reason)[:500],
              headline=head.text if head else None,
              headline_kind=head.kind if head else None,
              action=head.action if head else None,
              resets_at=head.resets_at if head else None,
              message=f"{name}: {head.text if head else ','.join(states)}")
    return res


def run(
    topic: str,
    start: date,
    end: date,
    sources: list[str] | None = None,
    *,
    limit: int = 500,
    options: dict[str, Any] | None = None,
    progress: ProgressFn | None = None,
    out_root: Path | None = None,
    cfg: dict[str, str] | None = None,
    stop: threading.Event | None = None,
) -> Path | None:
    """Execute a full run and return its directory, or None if the analyst cancelled it.

    Never raises for source failures.

    No time limit unless RUN_TIME_BUDGET_SECONDS is set (config.run_time_budget). `stop`
    (set by the GUI's "Stop scan") is a HARD cancel (owner decision 2026-11-20): every
    collector stops at once, and the run is then abandoned entirely — no verification, no
    analysis, no export and no report are produced, and the partial run directory is
    deleted. The function returns None for a cancelled run (a completed run returns its
    directory). options["ads_lookback_days"] widens the LinkedIn Ads search
    (ads_window_start).

    start/end are dates (whole UTC days, a custom range) or datetimes (exact bounds, a
    window preset). run.json start/end always hold the UTC dates the window touches (the
    day-based views and the coverage line use them); the exact bounds are recorded in
    options window_start_at / window_end_at and are what collection and timestamp
    verification use."""
    cfg = cfg or load_config()
    max_days = max_window_days(cfg)
    validate_inputs(topic, start, end, max_days)
    start_at, end_at = _bound(start, is_end=False), _bound(end, is_end=True)
    options = dict(options or {})
    # An end date after today (UTC) is clamped to now: the future part has no posts, and
    # platforms reject searches that start in the future (seen live 2026-11-20: YouTube
    # HTTP 400 for a 28-30 Nov slice on 20 Nov, which failed the whole source).
    now = datetime.now(UTC).replace(microsecond=0)
    if _day(end) > now.date():
        options["window_end_requested"] = _day(end).isoformat()
        end_at = now
        end = now
    start, end = _day(start), _day(end)
    sources = list(dict.fromkeys(sources or DEFAULT_SOURCES))  # analyst's choice, de-duplicated
    try:
        lookback = int(options.get("ads_lookback_days") or 0)
    except (TypeError, ValueError):
        lookback = 0
    if "linkedin_ads" not in sources:
        lookback = 0  # nothing would use it; never shown in the run header
    ads_all_time = lookback >= ADS_LOOKBACK_ALL
    if ads_all_time:
        ads_start_at = ADS_EPOCH  # no date filter at all (collect.linkedin_ads)
    else:
        ads_start_at = ads_window_start(start_at, end_at, lookback)
        # The Ad Library's date filter reaches about one year back
        # (collect.linkedin_ads.choose_date_params); record the span actually searched.
        ads_start_at = min(start_at, max(ads_start_at, now - timedelta(days=ADS_LOOKBACK_MAX)))
    options.pop("ads_lookback_days", None)
    if ads_all_time:
        options["ads_lookback_days"] = ADS_LOOKBACK_ALL
        options["ads_all_time"] = True
    elif ads_start_at < start_at:
        options["ads_lookback_days"] = min(lookback, ADS_LOOKBACK_MAX)
        options["ads_window_start_at"] = iso(ads_start_at)
    stop = stop or threading.Event()

    created = now_iso()
    run_id = make_run_id(topic, created)
    run_dir = (out_root or runs_dir()) / run_id
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    meta = RunMeta(run_id=run_id, topic=topic.strip(), start=start.isoformat(),
                   end=end.isoformat(), sources=sources, created_at=created,
                   options={"limit": limit, **options,
                            "window_start_at": iso(start_at), "window_end_at": iso(end_at)},
                   versions={"socmint": __version__})
    write_json(run_dir / FILES["run"], meta)
    prog = Progress(run_dir, progress)
    prog.emit(stage="collect", run_id=run_id, message=f"run {run_id} started")
    not_selected = [s for s in DEFAULT_SOURCES if s not in sources]
    prog.emit(message=f"sources selected: {', '.join(sources)}"
              + (f" (not selected: {', '.join(not_selected)})" if not_selected else ""))
    if options.get("window_end_requested"):
        prog.emit(message=f"end date {options['window_end_requested']} is in the future; "
                          f"the window ends now ({iso(end_at)})")
    for s in sources:  # every selected source is visible on the progress page from the start
        prog.emit(source=s, state="pending", count=0)

    t_run = time.monotonic()
    budget = run_time_budget(cfg)  # None = no time limit (the default)
    meta.options["time_limit_s"] = budget
    # Collectors see which sources share this run (LinkedIn Ads uses it to avoid driving the
    # same LinkedIn account in parallel with the LinkedIn posts source).
    q_opts: dict[str, Any] = {**options, "sources": tuple(sources)}
    if ads_start_at < start_at:
        q_opts["ads_start_at"] = iso(ads_start_at)
    if ads_all_time:
        q_opts["ads_all_time"] = True
    query = Query(topic=topic.strip(), start=start_at, end=end_at, limit=limit, options=q_opts)
    ctx = CollectContext(run_dir=run_dir, raw_dir=run_dir / "raw", config=cfg,
                         deadline=math.inf if budget is None else t_run + budget * COLLECT_SHARE,
                         progress=lambda s, info: prog.emit(source=s, **info), stop=stop)

    cancelled = False
    try:
        # ---------------------------------------------------------------- collect
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=max(1, len(sources))) as pool:
            futures = {s: pool.submit(_collect_one, s, query, ctx, prog) for s in sources}
            results = {s: f.result() for s, f in futures.items()}
        meta.timings["collect"] = round(time.monotonic() - t0, 2)

        # Hard cancel (owner decision 2026-11-20): if the analyst pressed Stop while
        # collecting, abandon the whole run here. No verification, analysis, export or
        # report is produced, and the partial run directory is removed in `finally`.
        if stop.is_set():
            cancelled = True
            raise _Cancelled

        posts: list[Post] = []
        statuses: list[SourceStatus] = []
        for s in sources:
            posts.extend(results[s].posts)
            statuses.extend(results[s].status)
        meta.costs = {s: round(sum(st.cost_usd for st in results[s].status), 4) for s in sources}

        # ----------------------------------------------------- verify + dedupe
        prog.emit(stage="verify", message="verifying timestamps, deduplicating")
        t0 = time.monotonic()
        posts = run_verification(posts, query.start, query.end, ads_start=ads_start_at)
        # Card numbers are masked once, here, before posts.jsonl is written, so every
        # file built from it (CSV, report, analysis, LLM prompts) carries the masked text.
        for p in posts:
            mask_post(p)
        meta.timings["verify"] = round(time.monotonic() - t0, 2)
        add_coverage(statuses, posts, start, end)
        write_jsonl(run_dir / FILES["posts"], posts)
        write_json(run_dir / FILES["source_status"], statuses)

        # --------------------------------------------------------------- analyze
        _analyze(posts, meta, run_dir, cfg, prog)
        _post_counts(posts, meta)
        meta.status = "complete"
    except _Cancelled:
        pass  # handled below; nothing is written for a cancelled run
    except Exception as e:  # noqa: BLE001
        meta.status = "failed"
        meta.error = redact(f"{type(e).__name__}: {e}")
        log.error("run failed: %s", redact(traceback.format_exc()))
    finally:
        if cancelled:
            # Remove the partial run so nothing (posts, status, export or report) is
            # left behind; the GUI job reports the cancellation to the browser.
            shutil.rmtree(run_dir, ignore_errors=True)
            log.info("run %s cancelled by the analyst; partial run removed", run_id)
        else:
            meta.finished_at = now_iso()
            meta.timings["total"] = round(time.monotonic() - t_run, 2)
            write_json(run_dir / FILES["run"], meta)
            _export_and_report(run_dir, meta, prog)
            write_json(run_dir / FILES["run"], meta)
            prog.emit(stage="done", done=True, error=meta.error,
                      message=f"run {meta.status} in {meta.timings['total']}s")
    return None if cancelled else run_dir


# Missing days listed by name in the coverage line before "+K more".
COVERAGE_LIST_MAX = 6


def coverage_line(posts: list[Post], source: str, start: date, end: date) -> str | None:
    """'coverage: posts on N of M days (no posts: d1, d2)' for one source, or None if the
    source returned no posts (its status reason already explains why).

    A day counts when at least one post of the source has a verified platform timestamp
    on that UTC day inside the window and is not context-only. This shows the analyst
    whether volume-over-time and "first observed" rest on the whole window or on a part.
    """
    own = [p for p in posts if p.source == source and not p.context_only]
    if not own:
        return None
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    seen: set[date] = set()
    for p in own:
        if not (p.timestamp_verified and p.in_window):
            continue
        ts = parse_ts(p.published_at)
        if ts is not None:
            seen.add(ts.astimezone(UTC).date())
    missing = [d for d in days if d not in seen]
    line = f"coverage: posts on {len(days) - len(missing)} of {len(days)} days"
    if missing:
        shown = ", ".join(d.isoformat() for d in missing[:COVERAGE_LIST_MAX])
        more = len(missing) - COVERAGE_LIST_MAX
        line += f" (no posts: {shown}{f' +{more} more' if more > 0 else ''})"
    return line


def add_coverage(statuses: list[SourceStatus], posts: list[Post], start: date,
                 end: date) -> None:
    """Append the coverage line to each source's first status reason. States unchanged."""
    done: set[str] = set()
    for st in statuses:
        if st.source in done:
            continue
        done.add(st.source)
        line = coverage_line(posts, st.source, start, end)
        if line:
            st.reason = f"{st.reason}; {line}" if st.reason else line


def _step(name: str, meta: RunMeta, prog: Progress, fn: Callable[[], Any], default: Any) -> Any:
    prog.emit(stage=name, message=f"analysis: {name}")
    t0 = time.monotonic()
    try:
        out = fn()
    except Exception as e:  # noqa: BLE001
        log.error("%s failed: %s", name, redact(traceback.format_exc()))
        meta.options.setdefault("analysis_errors", {})[name] = redact(f"{type(e).__name__}: {e}")
        out = default
    meta.timings[name] = round(time.monotonic() - t0, 2)
    return out


def _post_counts(posts: list[Post], meta: RunMeta) -> None:
    meta.counts.update({
        "posts": len(posts),
        "posts_analysable": len(analysable(posts)),
        "posts_unverified_timestamp": sum(1 for p in posts if not p.timestamp_verified),
        "posts_out_of_window": sum(1 for p in posts if not p.in_window),
    })


def _load_statuses(run_dir: Path) -> list[SourceStatus]:
    rows = read_json(run_dir / FILES["source_status"], []) or []
    return [SourceStatus.model_validate(r) for r in rows if isinstance(r, dict)]


ANALYSIS_FILES = ("narratives", "clusters", "origin", "claims", "brief")
STAGING_DIR = ".reanalyze_tmp"


def _analyze(posts: list[Post], meta: RunMeta, run_dir: Path, cfg: dict[str, str],
             prog: Progress, out_dir: Path | None = None) -> None:
    """Analysis stages shared by run() and reanalyze(). Source statuses are read from
    run_dir/source_status.json, which both callers have written (or kept) beforehand.

    The analysis files (ANALYSIS_FILES) are written to `out_dir` (default run_dir);
    reanalyze() points it at a staging directory so readers never see a mix of old and
    new files. Raw evidence and the LLM log still go to run_dir."""
    out_dir = run_dir if out_dir is None else out_dir
    from socmint.analyze import brief as brief_mod
    from socmint.analyze import claims as claims_mod
    from socmint.analyze import coordination, narratives, origin
    from socmint.llm import get_llm

    llm = get_llm(cfg, run_dir)
    meta.options["llm"] = llm.model if llm else None

    # Scope: drop context-only, duplicate and verified-out-of-window posts (ads active in
    # the window stay). Timelines/origin are further restricted to analysable() inside
    # each stage. Origin also sees context posts so direct edges can resolve endpoints;
    # analysable() keeps them out of its timeline and edges.
    scoped = in_scope(posts)
    meta.counts["posts_in_scope"] = len(scoped)

    narr_stats: dict[str, int] = {}
    for k in (*narratives.STATS_KEYS, "narratives_claims_assessed"):
        meta.counts.pop(k, None)  # a re-analysis never keeps the previous run's totals
    narr_trace: dict[str, dict] = {}
    for k in narratives.TRACE_KEYS:
        meta.options.pop(k, None)
    # exact window start for preset runs (ads begun earlier that day stay off timelines)
    window_start = meta.options.get("window_start_at") or meta.start
    # No run time limit (the default) -> no wall-clock cap on LLM labelling or evidence
    # lookups either; with a limit set, the modules' own caps apply.
    unlimited = run_time_budget(cfg) is None
    from socmint.config import analysis_workers
    workers = analysis_workers(cfg)  # concurrent LLM calls (default 12; ANALYSIS_LLM_WORKERS)
    narr = _step("narratives", meta, prog, lambda: narratives.build_narratives(
        scoped, topic=meta.topic, llm=llm, window_start=window_start, stats=narr_stats,
        trace=narr_trace, label_budget_s=math.inf if unlimited else None, workers=workers), [])
    meta.counts.update(narr_stats)  # narratives.STATS_KEYS: pruning and label counts
    # narratives.TRACE_KEYS: which posts left which narrative (evidence for the GUI/report)
    meta.options.update({k: v for k, v in narr_trace.items() if v})
    write_json(out_dir / FILES["narratives"], narr)
    # Perceptual hashes of avatars/thumbnails so coordination can catch the same image
    # re-uploaded under a different URL (identical URLs are matched without this). Network,
    # best-effort, bounded; any failure falls back to identical-URL matching only.
    image_hashes: dict[str, str] = {}
    if str(cfg.get("COORD_IMAGE_HASHING", "true")).strip().lower() in ("1", "true", "yes", "on"):
        try:
            from socmint.analyze import image_hash
            image_hashes = image_hash.compute_image_hashes(scoped)
        except Exception as e:  # noqa: BLE001 - never let image hashing take the run down
            log.warning("image hashing skipped: %s", redact(f"{type(e).__name__}: {e}"))
    clusters = _step("coordination", meta, prog,
                     lambda: coordination.detect_coordination(scoped, image_hashes), [])
    write_json(out_dir / FILES["clusters"], clusters)
    traces = _step("origin", meta, prog, lambda: origin.trace_origins(posts, narr), [])
    write_json(out_dir / FILES["origin"], traces)
    # Claims only for the most important narratives (owner decision 2026-11-20): a
    # preliminary ranking without claim inputs (brief.claim_candidates) picks them.
    try:
        order, n_claims = brief_mod.claim_candidates(posts, narr, clusters, meta,
                                                     claims_mod.MAX_NARRATIVES)
    except Exception:  # noqa: BLE001 - fall back to the size order inside assess_claims
        log.error("claim candidates failed: %s", redact(traceback.format_exc()))
        order, n_claims = None, claims_mod.MAX_NARRATIVES
    claims = _step("claims", meta, prog, lambda: claims_mod.assess_claims(
        scoped, narr, topic=meta.topic, llm=llm, cfg=cfg, run_dir=run_dir,
        window_start=meta.start, window_end=meta.end, narrative_order=order,
        max_narratives=n_claims, evidence_budget_s=math.inf if unlimited else None), [])
    meta.counts["narratives_claims_assessed"] = len({c.narrative_id for c in claims
                                                     if c.narrative_id})
    write_json(out_dir / FILES["claims"], claims)
    statuses = _load_statuses(run_dir)
    brief = _step("brief", meta, prog, lambda: brief_mod.build_brief(
        posts, narr, clusters, traces, claims, statuses, meta), Brief())
    write_json(out_dir / FILES["brief"], brief)

    if llm:
        meta.costs.update(llm.usage())
    meta.counts.update({"narratives": len(narr), "clusters": len(clusters),
                        "origin_traces": len(traces), "claims": len(claims)})


def _export_and_report(run_dir: Path, meta: RunMeta, prog: Progress) -> None:
    from socmint import export, report

    for name in (FILES["narratives"], FILES["clusters"], FILES["origin"], FILES["claims"],
                 FILES["source_status"]):
        if not (run_dir / name).exists():
            write_json(run_dir / name, [])
    if not (run_dir / FILES["posts"]).exists():
        (run_dir / FILES["posts"]).write_text("", encoding="utf-8")
    _step("export", meta, prog, lambda: export.write_exports(run_dir), None)
    _step("report", meta, prog, lambda: report.build_report(run_dir), None)


def _swap_in(stage: Path, run_dir: Path) -> None:
    """Move every staged analysis file over the run's copy (each an atomic rename)."""
    for name in ANALYSIS_FILES:
        src = stage / FILES[name]
        if src.is_file():
            os.replace(src, run_dir / FILES[name])


def mask_stored_card_data(run_dir: Path) -> int:
    """Mask payment card numbers in a saved run's evidence files, in place: posts.jsonl
    (the canonical post file; CSV and report are rebuilt from it), llm_log.jsonl,
    evidence_log.jsonl and every file under raw/. .json/.jsonl files are walked as JSON
    (string values only; numeric and identifier fields untouched, util.pan.mask_json),
    so they still parse afterwards. Runs saved before masking existed are
    brought in line by reanalyze(). Returns the number of files changed."""
    n = mask_tree(run_dir / "raw")
    for name in (FILES["posts"], "llm_log.jsonl", "evidence_log.jsonl"):
        f = run_dir / name
        if f.is_file() and mask_file(f):
            n += 1
    if n:
        log.info("reanalyze: masked payment card numbers in %d file(s)", n)
    return n


def reanalyze(run_dir: Path, *, cfg: dict[str, str] | None = None,
              progress: ProgressFn | None = None) -> Path:
    """Re-run ONLY the analysis stages on a saved run, then exports + report.

    Reads posts.jsonl, source_status.json and run.json; never collects. Uses the
    configured LLM / evidence keys like a normal run. Updates run.json timings, costs and
    counts; collection timings and per-source costs are kept.

    All or nothing: the new analysis files (narratives, clusters, origin, claims, brief)
    are written to run_dir/.reanalyze_tmp and moved over the old ones only when every
    step succeeded, so narrative IDs in claims/brief always match narratives.json. If a
    step fails, the previous analysis is kept and the failure is recorded in run.json
    (options.analysis_errors, error). While it works, run.json says status "running" with
    this process' pid in progress.json, so the GUI, run_in_progress(), delete and
    upgrade-on-load treat the run as being written. Never raises for an analysis failure.
    """
    run_dir = Path(run_dir)
    meta_raw = read_json(run_dir / FILES["run"], None)
    if not isinstance(meta_raw, dict):
        raise RunInputError(f"{run_dir} is not a run directory (no run.json)")
    if run_in_progress(run_dir):
        raise RunInputError(f"{run_dir} is still being written (collection or another "
                            "re-analysis); try again when it has finished")
    cfg = load_config() if cfg is None else cfg
    meta = RunMeta.model_validate(meta_raw)
    masked = mask_stored_card_data(run_dir)
    posts = [Post.model_validate(r) for r in read_jsonl(run_dir / FILES["posts"])]
    if not (run_dir / FILES["source_status"]).is_file():
        write_json(run_dir / FILES["source_status"], [])

    # keep the collection progress (per-source states and log) visible in progress.json
    old = read_json(run_dir / FILES["progress"], {}) or {}
    prog = Progress(run_dir, progress)
    if isinstance(old, dict):
        for k in ("sources", "log", "cost_usd"):
            if k in old:
                prog.state[k] = old[k]
    prog.emit(stage="reanalyze", run_id=meta.run_id,
              message=f"re-analysing {len(posts)} stored posts (no collection)")
    if masked:
        prog.emit(message=f"masked payment card numbers in {masked} stored file(s)")

    before = meta.model_copy(deep=True)
    prev_status, prev_error = meta.status, meta.error
    meta.options.pop("analysis_errors", None)
    for k in ("llm_calls", "llm_input_tokens", "llm_output_tokens"):
        meta.costs.pop(k, None)  # replaced by this analysis' usage (none if no LLM)
    # mark the run as being written (progress.json already carries this pid)
    write_json(run_dir / FILES["run"], meta.model_copy(update={"status": "running"}))
    stage = run_dir / STAGING_DIR
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    t0 = time.monotonic()
    try:
        _analyze(posts, meta, run_dir, cfg, prog, out_dir=stage)
        errors = meta.options.get("analysis_errors") or {}
        if errors:
            # keep the previous, consistent analysis; record what failed
            kept = before.model_copy(deep=True)
            kept.options["analysis_errors"] = errors
            kept.timings.update({k: v for k, v in meta.timings.items() if k in errors})
            meta = kept
            meta.status = prev_status
            meta.error = redact("reanalyze: step(s) failed (" + ", ".join(sorted(errors))
                                + "); the previous analysis was kept")
        else:
            _swap_in(stage, run_dir)
            _post_counts(posts, meta)
            meta.status = prev_status
            if (prev_error or "").startswith("reanalyze"):  # an earlier re-analysis failed
                meta.status = "complete" if prev_status == "failed" else prev_status
                meta.error = None
    except Exception as e:  # noqa: BLE001
        meta = before.model_copy(deep=True)
        meta.status = prev_status
        meta.error = redact(f"reanalyze: {type(e).__name__}: {e}; the previous analysis "
                            "was kept")
        log.error("reanalyze failed: %s", redact(traceback.format_exc()))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        meta.timings["reanalyze_total"] = round(time.monotonic() - t0, 2)
        meta.options["reanalyzed_at"] = now_iso()
        # the analysis files are consistent again: final status before the report embeds it
        write_json(run_dir / FILES["run"], meta)
        _export_and_report(run_dir, meta, prog)
        write_json(run_dir / FILES["run"], meta)
        prog.emit(stage="done", done=True, error=meta.error,
                  message=f"re-analysis finished in {meta.timings['reanalyze_total']}s")
    return run_dir


def parse_window(start: str, end: str) -> tuple[date, date]:
    return (datetime.fromisoformat(start[:10]).date(), datetime.fromisoformat(end[:10]).date())
