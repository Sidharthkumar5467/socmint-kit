"""Only the sources the analyst ticked are collected, and all of them are visible at once."""

from __future__ import annotations

from datetime import date

from socmint import pipeline
from socmint.interfaces import CollectResult
from socmint.runstore import read_json
from socmint.schema import SourceState, SourceStatus


def test_run_collects_only_selected_sources(monkeypatch, tmp_path):
    called: list[str] = []

    def fake_collect_one(name, query, ctx, prog):
        # by the time any collector starts, every selected source is already listed
        seen = read_json(ctx.run_dir / "progress.json", {})["sources"]
        assert set(seen) == {"telegram", "youtube"}
        called.append(name)
        return CollectResult(posts=[], status=[SourceStatus(
            source=name, query=query.topic, state=SourceState.completed_zero)])

    monkeypatch.setattr(pipeline, "_collect_one", fake_collect_one)
    events: list[dict] = []
    run_dir = pipeline.run("test topic", date(2026, 11, 1), date(2026, 11, 7),
                           ["telegram", "youtube", "telegram"], out_root=tmp_path,
                           progress=events.append, cfg={})
    assert sorted(called) == ["telegram", "youtube"]  # de-duplicated, x/linkedin not run
    assert read_json(run_dir / "run.json")["sources"] == ["telegram", "youtube"]
    msgs = [e.get("message") or "" for e in events]
    assert any("sources selected: telegram, youtube" in m
               and "not selected: x, linkedin" in m for m in msgs)
    assert any(e.get("source") == "youtube" and e.get("state") == "pending" for e in events)


# ------------------------------------------------------------------ coverage line
def _cov_post(nid, day, source="x", **kw):
    from conftest import make_post

    return make_post(nid, "bank of america", source=source,
                     published_at=f"2026-11-{day:02d}T12:00:00Z", **kw)


def test_coverage_line_counts_days_and_lists_missing():
    posts = [_cov_post("1", 19), _cov_post("2", 19), _cov_post("3", 18),
             _cov_post("4", 13, verified=False),          # unverified: does not count
             _cov_post("5", 14, context_only=True),       # context-only: does not count
             _cov_post("6", 15, in_window=False),         # out of window: does not count
             _cov_post("7", 16, source="youtube")]        # other source
    line = pipeline.coverage_line(posts, "x", date(2026, 11, 13), date(2026, 11, 19))
    assert line == ("coverage: posts on 2 of 7 days (no posts: 2026-11-13, 2026-11-14, "
                    "2026-11-15, 2026-11-16, 2026-11-17)")
    assert pipeline.coverage_line(posts, "telegram", date(2026, 11, 13),
                                  date(2026, 11, 19)) is None


def test_coverage_line_full_and_truncated():
    full = [_cov_post(str(d), d) for d in range(13, 20)]
    assert pipeline.coverage_line(full, "x", date(2026, 11, 13), date(2026, 11, 19)) == \
        "coverage: posts on 7 of 7 days"
    line = pipeline.coverage_line([_cov_post("1", 30)], "x", date(2026, 11, 1),
                                  date(2026, 11, 30))
    assert line == ("coverage: posts on 1 of 30 days (no posts: 2026-11-01, 2026-11-02, "
                    "2026-11-03, 2026-11-04, 2026-11-05, 2026-11-06 +23 more)")


def test_add_coverage_first_status_only_state_unchanged():
    sts = [SourceStatus(source="x", query="t", state=SourceState.partial, reason="time budget"),
           SourceStatus(source="x", query="t2", state=SourceState.ok),
           SourceStatus(source="youtube", query="t", state=SourceState.completed_zero,
                        reason="no results")]
    pipeline.add_coverage(sts, [_cov_post("1", 19)], date(2026, 11, 19), date(2026, 11, 19))
    assert sts[0].reason == "time budget; coverage: posts on 1 of 1 days"
    assert sts[0].state == SourceState.partial
    assert sts[1].reason == ""
    assert sts[2].reason == "no results"  # 0 posts: nothing extra


def test_run_writes_coverage_into_source_status(monkeypatch, tmp_path):
    def fake_collect_one(name, query, ctx, prog):
        return CollectResult(posts=[_cov_post("1", 7, source=name)], status=[SourceStatus(
            source=name, query=query.topic, state=SourceState.ok, count=1,
            reason="per-day: 2026-11-07 1")])

    monkeypatch.setattr(pipeline, "_collect_one", fake_collect_one)
    run_dir = pipeline.run("test topic", date(2026, 11, 6), date(2026, 11, 7), ["x"],
                           out_root=tmp_path, options={"match": "words"}, cfg={})
    st = read_json(run_dir / "source_status.json")
    assert st[0]["reason"] == ("per-day: 2026-11-07 1; coverage: posts on 1 of 2 days "
                               "(no posts: 2026-11-06)")
    assert st[0]["state"] == "ok"
    assert read_json(run_dir / "run.json")["options"]["match"] == "words"


# ------------------------------------------------------------- brief + reanalyze
def _two_source_run(monkeypatch, tmp_path):
    def fake_collect_one(name, query, ctx, prog):
        posts = [_cov_post(f"{name}{i}", 6 + i % 2, source=name, author=f"{name}{i}")
                 for i in range(3)]
        return CollectResult(posts=posts, status=[SourceStatus(
            source=name, query=query.topic, state=SourceState.ok, count=3)])

    monkeypatch.setattr(pipeline, "_collect_one", fake_collect_one)
    return pipeline.run("bank of america", date(2026, 11, 6), date(2026, 11, 7),
                        ["x", "youtube"], out_root=tmp_path, cfg={})


def test_run_writes_brief(monkeypatch, tmp_path):
    from socmint.runstore import load_run

    run_dir = _two_source_run(monkeypatch, tmp_path)
    brief = read_json(run_dir / "brief.json")
    assert {"narrative_ranks", "findings", "coverage", "verdict_counts"} <= set(brief)
    assert [c["source"] for c in brief["coverage"]] == ["x", "youtube"]
    assert brief["coverage"][0]["note"] == "all 2 days covered"
    meta = read_json(run_dir / "run.json")
    assert "brief" in meta["timings"] and meta["status"] == "complete"
    assert load_run(run_dir)["brief"] == brief


def test_load_run_without_brief_returns_empty(tmp_path):
    from socmint.runstore import load_run

    assert load_run(tmp_path)["brief"] == {}


def test_reanalyze_reruns_analysis_without_collecting(monkeypatch, tmp_path):
    from socmint.analyze import claims as claims_mod
    from socmint.schema import Claim

    run_dir = _two_source_run(monkeypatch, tmp_path)
    collected_meta = read_json(run_dir / "run.json")
    (run_dir / "brief.json").unlink()  # an older run without a brief

    def boom(*a, **k):
        raise AssertionError("reanalyze must not collect")

    monkeypatch.setattr(pipeline, "_collect_one", boom)
    seen = {}

    def fake_claims(scoped, narr, **kw):
        seen["n_posts"] = len(scoped)
        seen["cfg"] = kw["cfg"]
        return [Claim(id="CL1", text="t", related_post_ids=[scoped[0].post_id],
                      verdict="disputed", confidence=0.5)]

    monkeypatch.setattr(claims_mod, "assess_claims", fake_claims)
    events: list[dict] = []
    out = pipeline.reanalyze(run_dir, cfg={"SOME_KEY": "v"}, progress=events.append)
    assert out == run_dir
    assert seen == {"n_posts": 6, "cfg": {"SOME_KEY": "v"}}  # configured keys passed on
    assert read_json(run_dir / "claims.json")[0]["verdict"] == "disputed"
    brief = read_json(run_dir / "brief.json")
    assert brief["verdict_counts"]["disputed"] == 1
    meta = read_json(run_dir / "run.json")
    assert meta["status"] == "complete" and meta["error"] is None
    assert meta["timings"]["collect"] == collected_meta["timings"]["collect"]  # kept
    assert "reanalyze_total" in meta["timings"] and "brief" in meta["timings"]
    assert meta["options"]["reanalyzed_at"]
    assert meta["counts"]["posts"] == 6 and meta["counts"]["claims"] == 1
    assert (run_dir / "claims.csv").read_text(encoding="utf-8-sig").count("CL1") == 1
    prog = read_json(run_dir / "progress.json")
    assert prog["done"] is True and set(prog["sources"]) == {"x", "youtube"}  # kept
    assert any("re-analysing 6 stored posts" in (e.get("message") or "") for e in events)


def test_reanalyze_records_step_failure(monkeypatch, tmp_path):
    from socmint.analyze import coordination

    run_dir = _two_source_run(monkeypatch, tmp_path)

    def broken(posts, image_hashes=None):
        raise RuntimeError("bad")

    before = {n: (run_dir / n).read_bytes() for n in
              ("narratives.json", "clusters.json", "origin.json", "claims.json", "brief.json")}
    monkeypatch.setattr(coordination, "detect_coordination", broken)
    pipeline.reanalyze(run_dir, cfg={})
    meta = read_json(run_dir / "run.json")
    assert meta["options"]["analysis_errors"] == {"coordination": "RuntimeError: bad"}
    assert meta["status"] == "complete"
    assert meta["error"] == ("reanalyze: step(s) failed (coordination); the previous "
                             "analysis was kept")
    # all or nothing: no file of the failed re-analysis replaced the previous analysis
    assert {n: (run_dir / n).read_bytes() for n in before} == before
    assert not (run_dir / ".reanalyze_tmp").exists()
    # a later clean re-analysis clears the recorded failure
    monkeypatch.undo()
    pipeline.reanalyze(run_dir, cfg={})
    meta = read_json(run_dir / "run.json")
    assert meta["error"] is None and "analysis_errors" not in meta["options"]


def test_reanalyze_marks_run_running_and_swaps_atomically(monkeypatch, tmp_path):
    from socmint import runstore
    from socmint.analyze import claims as claims_mod

    run_dir = _two_source_run(monkeypatch, tmp_path)
    old_narr = (run_dir / "narratives.json").read_bytes()
    seen = {}

    def spy_claims(scoped, narr, **kw):
        # mid-analysis: the run reads as being written, and readers still see the old
        # narratives (the new ones are staged)
        seen["status"] = read_json(run_dir / "run.json")["status"]
        seen["in_progress"] = runstore.run_in_progress(run_dir)
        seen["narr_same"] = (run_dir / "narratives.json").read_bytes() == old_narr
        seen["staged"] = (run_dir / ".reanalyze_tmp" / "narratives.json").is_file()
        return []

    monkeypatch.setattr(claims_mod, "assess_claims", spy_claims)
    pipeline.reanalyze(run_dir, cfg={})
    assert seen == {"status": "running", "in_progress": True, "narr_same": True,
                    "staged": True}
    assert read_json(run_dir / "run.json")["status"] == "complete"
    assert not runstore.run_in_progress(run_dir)
    assert "\"status\":\"running\"" not in (run_dir / "report.html").read_text(encoding="utf-8")


def test_reanalyze_refuses_run_in_progress(monkeypatch, tmp_path):
    import pytest

    from socmint import runstore

    run_dir = _two_source_run(monkeypatch, tmp_path)
    meta = read_json(run_dir / "run.json")
    meta["status"] = "running"
    runstore.write_json(run_dir / "run.json", meta)  # progress.json pid = this process
    with pytest.raises(pipeline.RunInputError, match="still being written"):
        pipeline.reanalyze(run_dir, cfg={})


def test_reanalyze_rejects_non_run_dir(tmp_path):
    import pytest

    with pytest.raises(pipeline.RunInputError):
        pipeline.reanalyze(tmp_path, cfg={})
