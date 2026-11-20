"""Tests for analyze.embed and analyze.narratives.

All tests force the TF-IDF embedding path (SOCMINT_EMBEDDINGS=tfidf and a patched model
loader) so nothing is ever downloaded. Posts are synthetic, written for these tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_post  # tests/ is on sys.path (no package __init__)

from socmint.analyze import embed, narratives
from socmint.analyze.narratives import NarrativeLabel, build_narratives
from socmint.llm import LLMError

_REAL_LOAD_MODEL = embed._load_model  # captured before the autouse fixture patches it


@pytest.fixture(autouse=True)
def _force_tfidf(monkeypatch):
    monkeypatch.setenv("SOCMINT_EMBEDDINGS", "tfidf")

    def no_model(*a, **k):
        raise AssertionError("tests must not load a sentence-transformers model")

    monkeypatch.setattr(embed, "_load_model", no_model)


# Three clearly distinct topics, each in a different language/script.
DAM = [  # English
    "The new dam on the river collapsed overnight and flooded three villages downstream",
    "Breaking: dam collapse floods villages downstream, residents evacuated from the river valley",
    "Officials hide the truth about the dam collapse that flooded the river villages",
    "Dam collapsed last night, villages downstream flooded, river still rising",
    "Footage shows the river dam collapse and flooded villages, thousands evacuated",
    "Why did the dam collapse? Villages downstream flooded and nobody warned residents",
]
VACCINE = [  # Spanish
    "La nueva vacuna contiene microchips para rastrear a la población, dicen los médicos",
    "Médicos denuncian que la vacuna contiene microchips de rastreo para la población",
    "¡Cuidado! La vacuna trae microchips para rastrear a toda la población",
    "Confirmado por médicos: microchips en la vacuna para rastrear a la población",
    "La vacuna con microchips de rastreo ya se aplica a la población sin aviso",
    "Microchips en la vacuna: los médicos advierten sobre el rastreo de la población",
]
ELECTION = [  # Russian
    "Выборы были сфальсифицированы, бюллетени вбрасывали на участках всю ночь",
    "На участках вбрасывали бюллетени, выборы сфальсифицированы, есть видео",
    "Видео вброса бюллетеней на участках доказывает, что выборы сфальсифицированы",
    "Сфальсифицированные выборы: наблюдатели сняли вброс бюллетеней на участках",
    "Вброс бюллетеней на участках всю ночь, выборы сфальсифицированы полностью",
    "Наблюдатели сообщают о вбросе бюллетеней, выборы на участках сфальсифицированы",
]


def _corpus():
    posts = []
    sources = ["x", "telegram", "youtube"]
    for t, (name, texts) in enumerate((("dam", DAM), ("vax", VACCINE), ("elec", ELECTION))):
        for i, text in enumerate(texts):
            posts.append(
                make_post(
                    f"{name}{i}",
                    text,
                    source=sources[i % 3],
                    author=f"{name}_author{i}",
                    published_at=f"2026-11-0{1 + (i % 3)}T1{t}:00:00Z",
                )
            )
    return posts


def _by_prefix(narrs, prefix):
    return [n for n in narrs if all(pid.split(":")[1].startswith(prefix) for pid in n.member_post_ids)]


# ------------------------------------------------------------------------------ embed


def test_embed_texts_tfidf_normalised():
    vecs, method = embed.embed_texts(["hello world post", "another post here", "третий пост"])
    assert method.startswith("tfidf")
    assert vecs.shape[0] == 3
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_embed_texts_empty():
    vecs, method = embed.embed_texts([])
    assert vecs.shape[0] == 0 and method == "none"


def test_embed_falls_back_when_model_unavailable(monkeypatch):
    monkeypatch.delenv("SOCMINT_EMBEDDINGS", raising=False)
    monkeypatch.setattr(embed, "_load_model", lambda name, **k: None)
    vecs, method = embed.embed_texts(["some text about a topic", "other words entirely"])
    assert method.startswith("tfidf") and vecs.shape[0] == 2


def test_load_model_falls_back_without_download(monkeypatch):
    """Not cached + downloads disabled -> None (TF-IDF), and the failure is cached."""
    monkeypatch.setenv("SOCMINT_ALLOW_MODEL_DOWNLOAD", "0")
    calls = []

    def fake_try_load(name, *, local_only):
        calls.append(local_only)
        raise OSError("not in cache")

    monkeypatch.setattr(embed, "_try_load", fake_try_load)
    monkeypatch.setattr(embed, "_MODELS", {})
    assert _REAL_LOAD_MODEL("some/model") is None
    assert _REAL_LOAD_MODEL("some/model") is None
    assert calls == [True]  # local-only attempt once, never a download


# ------------------------------------------------------------------------- narratives


def test_empty_input_returns_empty():
    assert build_narratives([], topic="anything", llm=None) == []


def test_trivial_text_only_returns_empty():
    posts = [make_post(str(i), "ok") for i in range(10)]
    assert build_narratives(posts, topic="x", llm=None) == []


def test_three_topics_three_narratives():
    narrs = build_narratives(_corpus(), topic="news", llm=None)
    assert len(narrs) == 3
    assert [n.id for n in narrs] == ["N1", "N2", "N3"]
    for prefix in ("dam", "vax", "elec"):
        match = _by_prefix(narrs, prefix)
        assert len(match) == 1, f"{prefix} not cleanly clustered: {[n.member_post_ids for n in narrs]}"
        n = match[0]
        assert len(n.member_post_ids) == 6
        assert 1 <= len(n.example_post_ids) <= 5
        assert set(n.example_post_ids) <= set(n.member_post_ids)
        # examples prefer distinct sources: first three cover all three platforms
        srcs = [pid.split(":")[0] for pid in n.example_post_ids[:3]]
        assert len(set(srcs)) == 3
        assert n.volume_by_platform == {"telegram": 2, "x": 2, "youtube": 2}
        assert sum(n.volume_by_day.values()) == 6
        assert n.unverified_member_count == 0
        assert n.label_method == "keyword"
        assert n.claim_summary.startswith('Representative post: "')


def test_keyword_label_excludes_topic_and_stopwords():
    narrs = build_narratives(_corpus(), topic="dam collapse", llm=None)
    dam = _by_prefix(narrs, "dam")[0]
    assert "dam" not in dam.keywords and "collapse" not in dam.keywords
    assert "the" not in dam.keywords
    assert any(k in dam.keywords for k in ("villages", "flooded", "river", "downstream"))
    assert dam.label == " / ".join(dam.keywords[:4])
    vax = _by_prefix(narrs, "vax")[0]
    assert "la" not in vax.keywords and "microchips" in vax.keywords


def test_volume_excludes_unverified_and_out_of_window():
    posts = _corpus()
    # Two dam posts lose their verified timestamp, one is outside the window.
    posts[0].timestamp_verified = False
    posts[0].timestamp_note = "no parseable platform timestamp"
    posts[1].published_at = None
    posts[1].timestamp_verified = False
    posts[2].in_window = False
    narrs = build_narratives(posts, topic="news", llm=None)
    dam = _by_prefix(narrs, "dam")[0]
    # Unverified posts are still members (their content is evidence of the narrative)...
    assert {"x:dam0", "telegram:dam1", "youtube:dam2"} <= set(dam.member_post_ids)
    # ...but never counted in volume or first-seen.
    assert dam.unverified_member_count == 3
    assert sum(dam.volume_by_day.values()) == 3
    assert sum(dam.volume_by_platform.values()) == 3
    verified_times = sorted(p.published_at for p in posts[3:6])
    assert dam.first_seen_at == verified_times[0]


def test_volume_by_day_keys_and_first_seen():
    narrs = build_narratives(_corpus(), topic="news", llm=None)
    for n in narrs:
        assert all(len(k) == 10 and k[4] == "-" for k in n.volume_by_day)
        assert n.first_seen_at and n.first_seen_at.endswith("Z")
        assert n.first_seen_at.startswith("2026-11-01")


def test_duplicates_marked_by_dedupe_are_skipped():
    posts = _corpus()
    copy = make_post("dup", DAM[0], duplicate_of="x:dam0")
    narrs = build_narratives(posts + [copy], topic="news", llm=None)
    assert all("x:dup" not in n.member_post_ids for n in narrs)


def test_noise_post_not_forced_into_narrative():
    posts = _corpus() + [
        make_post("odd", "Quarterly results for the bakery cooperative: sourdough sales up 4%")
    ]
    narrs = build_narratives(posts, topic="news", llm=None)
    assert all("x:odd" not in n.member_post_ids for n in narrs)


def test_tiny_dataset_uses_agglomerative():
    posts = [
        make_post("a", DAM[0], author="u1"),
        make_post("b", DAM[3], author="u2", source="telegram"),
        make_post("c", DAM[1], author="u3", source="youtube"),
        make_post("d", VACCINE[0], author="u4"),
        make_post("e", VACCINE[2], author="u5"),
    ]
    narrs = build_narratives(posts, topic="news", llm=None)
    members = [set(n.member_post_ids) for n in narrs]
    assert {"x:a", "telegram:b", "youtube:c"} in members
    assert {"x:d", "x:e"} in members


def test_ad_posts_participate():
    posts = _corpus()
    posts.append(make_post("ad1", VACCINE[1] + " Más información aquí", source="linkedin_ads", kind="ad"))
    narrs = build_narratives(posts, topic="news", llm=None)
    vax = [n for n in narrs if "linkedin_ads:ad1" in n.member_post_ids]
    assert len(vax) == 1 and any(pid.startswith("x:vax") or ":vax" in pid for pid in vax[0].member_post_ids)


# -------------------------------------------------------------------------------- llm


class FakeLLM:
    def __init__(self, fail_on: set[str] | None = None):
        self.calls: list[dict] = []
        self.fail_on = fail_on or set()

    def parse(self, schema, prompt, *, system="", purpose="", max_tokens=4096):
        self.calls.append({"schema": schema, "prompt": prompt, "system": system, "purpose": purpose})
        if purpose.split(":")[-1] in self.fail_on:
            raise LLMError("simulated failure")
        return schema(
            label="one two three four five six seven eight nine ten",
            claim_summary="Posts claim that something happened.",
        )


def test_llm_labels_one_call_per_narrative():
    llm = FakeLLM()
    narrs = build_narratives(_corpus(), topic="news", llm=llm)
    assert len(llm.calls) == len(narrs) == 3
    calls = {c["purpose"]: c for c in llm.calls}  # calls run concurrently: match by id
    for n in narrs:
        call = calls[f"narrative_label:{n.id}"]
        assert call["schema"] is NarrativeLabel
        assert "must not" in call["system"].lower() or "do not add" in call["system"].lower()
        assert call["purpose"] == f"narrative_label:{n.id}"
        # The model only sees member texts (each example text appears in the prompt).
        assert call["prompt"].count("\n[") <= narratives.MAX_LLM_TEXTS
        assert n.label_method == "llm"
        # A 10-word, 49-char label is kept whole (no word cap; regression, see below).
        assert n.label == "one two three four five six seven eight nine ten"
        assert n.relevance is None and n.relevance_reason == ""  # fake gave no relevance
        assert n.claim_summary == "Posts claim that something happened."


def test_llm_prompt_contains_only_member_texts():
    llm = FakeLLM()
    narrs = build_narratives(_corpus(), topic="news", llm=llm)
    by_id = {p.post_id: p.text for p in _corpus()}
    calls = {c["purpose"]: c for c in llm.calls}
    for n in narrs:
        call = calls[f"narrative_label:{n.id}"]
        others = [t for pid, t in by_id.items() if pid not in n.member_post_ids]
        assert not any(t[:40] in call["prompt"] for t in others)
        assert all(by_id[pid][:40] in call["prompt"] for pid in n.member_post_ids)


def test_llm_failure_falls_back_to_keywords():
    llm = FakeLLM(fail_on={"N2"})
    narrs = build_narratives(_corpus(), topic="news", llm=llm)
    methods = {n.id: n.label_method for n in narrs}
    assert methods == {"N1": "llm", "N2": "keyword", "N3": "llm"}
    n2 = next(n for n in narrs if n.id == "N2")
    assert n2.claim_summary.startswith('Representative post: "')


def test_first_sentence_helper():
    assert narratives._first_sentence("First claim here. Second one! https://x.y/z") == "First claim here."
    long = "word " * 100
    assert len(narratives._first_sentence(long)) <= 240


# ------------------------------------------------------- regression: review findings


def test_single_topic_corpus_gives_one_narrative():
    """HDBSCAN (allow_single_cluster=False) fragmented a 12-post single-topic set into
    2x4 and dropped 4 to noise; the agglomerative cross-check must keep it whole."""
    texts = DAM + [d.replace("villages", "towns") + " again" for d in DAM]
    posts = [make_post(f"s{i}", t, author=f"u{i}") for i, t in enumerate(texts)]
    narrs = build_narratives(posts, topic="news", llm=None)
    assert len(narrs) >= 1
    assert len(narrs[0].member_post_ids) >= 10  # the dominant narrative is not split


def test_identical_copies_by_many_accounts_form_one_narrative():
    """20 accounts posting the same text: HDBSCAN alone labels every row noise."""
    posts = [make_post(f"c{i}", DAM[0], author=f"acct{i}") for i in range(20)]
    narrs = build_narratives(posts, topic="news", llm=None)
    assert len(narrs) == 1
    assert len(narrs[0].member_post_ids) == 20


def test_copy_campaign_plus_unrelated_posts():
    posts = [make_post(f"bot{i}", DAM[0], author=f"bot{i}") for i in range(8)]
    posts += [make_post("v", VACCINE[0], author="v1"), make_post("e", ELECTION[0], author="e1")]
    narrs = build_narratives(posts, topic="news", llm=None)
    assert len(narrs) == 1
    assert set(narrs[0].member_post_ids) == {f"x:bot{i}" for i in range(8)}


def test_n_at_tiny_threshold_six_plus_two():
    """n == TINY_DATASET (HDBSCAN path): 6 agreeing posts must form a narrative."""
    posts = [make_post(f"d{i}", t, author=f"d{i}") for i, t in enumerate(DAM)]
    posts += [make_post(f"v{i}", t, author=f"v{i}") for i, t in enumerate(VACCINE[:2])]
    assert len(posts) == narratives.TINY_DATASET
    narrs = build_narratives(posts, topic="news", llm=None)
    dam = _by_prefix(narrs, "d")
    assert any(len(n.member_post_ids) == 6 for n in dam)


def test_exact_copy_pair_rescued_from_noise():
    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = narratives._rescue_exact_copies(vecs, np.array([-1, -1, -1]))
    assert labels[0] == labels[1] >= 0 and labels[2] == -1


class RaisingLLM(FakeLLM):
    def parse(self, schema, prompt, *, system="", purpose="", max_tokens=4096):
        self.calls.append({"purpose": purpose})
        if purpose.endswith(":N2"):
            raise ValueError("structured output failed validation")  # not an LLMError
        return schema(label="Label", claim_summary="Posts claim that something happened.")


def test_non_llmerror_exception_falls_back_for_that_narrative_only():
    narrs = build_narratives(_corpus(), topic="news", llm=RaisingLLM())
    assert len(narrs) == 3
    assert {n.id: n.label_method for n in narrs} == {"N1": "llm", "N2": "keyword", "N3": "llm"}


def _ad(native_id, text, run_start):
    from socmint.schema import AdMeta

    return make_post(native_id, text, source="linkedin_ads", kind="ad", author=f"adv_{native_id}",
                     published_at=run_start, ad_meta=AdMeta(run_start=run_start))


def test_ad_started_before_window_not_on_timeline():
    posts = _corpus()
    posts.append(_ad("old", VACCINE[1] + " Más información", "2026-09-20T00:00:00Z"))
    posts.append(_ad("new", VACCINE[3] + " Más información", "2026-11-02T08:00:00Z"))
    narrs = build_narratives(posts, topic="news", llm=None, window_start="2026-11-01")
    vax = next(n for n in narrs if "linkedin_ads:old" in n.member_post_ids)
    assert "linkedin_ads:new" in vax.member_post_ids
    assert "2026-09-20" not in vax.volume_by_day
    assert all(k >= "2026-11-01" for k in vax.volume_by_day)
    assert vax.first_seen_at >= "2026-11-01"
    assert vax.volume_by_platform.get("linkedin_ads") == 1  # only the in-window start
    assert vax.unverified_member_count == 1
    assert sum(vax.volume_by_day.values()) == len(vax.member_post_ids) - 1


def test_ads_not_on_timeline_without_window():
    posts = _corpus()
    posts.append(_ad("old", VACCINE[1] + " Más información", "2026-09-20T00:00:00Z"))
    narrs = build_narratives(posts, topic="news", llm=None)
    vax = next(n for n in narrs if "linkedin_ads:old" in n.member_post_ids)
    assert "2026-09-20" not in vax.volume_by_day
    assert "linkedin_ads" not in vax.volume_by_platform
    assert vax.unverified_member_count == 1


# ------------------------------------------------- concurrent labelling and time budget


class SleepyLLM(FakeLLM):
    """Fake LLM with a fixed per-call latency (no network)."""

    def __init__(self, delay: float, **kw):
        super().__init__(**kw)
        self.delay = delay

    def parse(self, schema, prompt, *, system="", purpose="", max_tokens=4096):
        import time

        time.sleep(self.delay)
        return super().parse(schema, prompt, system=system, purpose=purpose, max_tokens=max_tokens)


def _fields(narrs):
    return [n.model_dump() for n in narrs]


def test_concurrent_labels_identical_to_sequential(monkeypatch):
    posts = _corpus()
    monkeypatch.setattr(narratives, "LABEL_WORKERS", 1)
    seq = build_narratives(posts, topic="news", llm=SleepyLLM(0.01, fail_on={"N2"}))
    monkeypatch.setattr(narratives, "LABEL_WORKERS", 8)
    par = build_narratives(posts, topic="news", llm=SleepyLLM(0.01, fail_on={"N2"}))
    assert _fields(seq) == _fields(par)
    assert [n.id for n in par] == ["N1", "N2", "N3"]
    assert {n.id: n.label_method for n in par} == {"N1": "llm", "N2": "keyword", "N3": "llm"}
    # Clustering and ids do not depend on the LLM at all.
    plain = build_narratives(posts, topic="news", llm=None)
    assert [(n.id, n.member_post_ids, n.example_post_ids) for n in plain] == [
        (n.id, n.member_post_ids, n.example_post_ids) for n in par
    ]


def test_label_benchmark_wall_time_scales_with_workers(monkeypatch):
    """Offline benchmark: 16 labels at 0.1 s each. Sequential ~1.6 s; 8 workers ~0.2 s."""
    import time

    jobs = [narratives._LabelJob(f"N{i}", [f"post text {i}"]) for i in range(1, 17)]
    timings = {}
    for workers in (1, 8):
        monkeypatch.setattr(narratives, "LABEL_WORKERS", workers)
        llm = SleepyLLM(0.1)
        t0 = time.perf_counter()
        res = narratives._label_all(llm, "news", jobs)
        timings[workers] = time.perf_counter() - t0
        assert [why for _, why in res] == ["llm"] * 16
        assert sorted(c["purpose"] for c in llm.calls) == sorted(f"narrative_label:{j.nid}" for j in jobs)
    assert timings[1] >= 1.5
    assert timings[8] < timings[1] / 4  # ideal 8x; allow scheduling slack


def test_label_time_budget_keyword_labels_smallest(monkeypatch):
    """Deterministic fake clock: each LLM call 'takes' 100 s; budget 150 s; 1 worker.
    N1 starts at t=0, N2 at t=100, N3 at t=200 > budget -> keyword label, no call."""
    clock = {"t": 0.0}

    class ClockLLM(FakeLLM):
        def parse(self, schema, prompt, *, system="", purpose="", max_tokens=4096):
            clock["t"] += 100.0
            return super().parse(schema, prompt, system=system, purpose=purpose, max_tokens=max_tokens)

    monkeypatch.setattr(narratives, "_clock", lambda: clock["t"])
    monkeypatch.setattr(narratives, "LABEL_WORKERS", 1)
    monkeypatch.setattr(narratives, "LABEL_TIME_BUDGET_S", 150.0)
    llm = ClockLLM()
    narrs = build_narratives(_corpus(), topic="news", llm=llm)
    assert [c["purpose"] for c in llm.calls] == ["narrative_label:N1", "narrative_label:N2"]
    assert {n.id: n.label_method for n in narrs} == {"N1": "llm", "N2": "llm", "N3": "keyword"}
    n3 = narrs[2]
    assert n3.claim_summary.startswith('Representative post: "')
    # Biggest first: the budget-skipped narrative is the smallest.
    assert len(n3.member_post_ids) <= min(len(n.member_post_ids) for n in narrs[:2])


def test_label_budget_zero_means_no_llm_calls(monkeypatch):
    monkeypatch.setattr(narratives, "LABEL_TIME_BUDGET_S", 0.0)
    llm = FakeLLM()
    narrs = build_narratives(_corpus(), topic="news", llm=llm)
    assert llm.calls == []
    assert all(n.label_method == "keyword" for n in narrs)
    assert _fields(narrs) == _fields(build_narratives(_corpus(), topic="news", llm=None))


def test_llm_counters_thread_safe():
    """LLM.parse counters under concurrent calls (fake client, no network)."""
    import threading
    from types import SimpleNamespace

    from socmint.llm import LLM

    llm = LLM.__new__(LLM)
    llm._anthropic = SimpleNamespace(NOT_GIVEN=None, APIStatusError=RuntimeError,
                                     APIConnectionError=RuntimeError)
    resp = SimpleNamespace(usage=SimpleNamespace(input_tokens=3, output_tokens=2),
                           stop_reason="end_turn",
                           parsed_output=NarrativeLabel(label="a", claim_summary="b"))
    llm.client = SimpleNamespace(messages=SimpleNamespace(parse=lambda **k: resp))
    llm.model, llm.log_path, llm._lock = "fake", None, threading.Lock()
    llm.calls = llm.input_tokens = llm.output_tokens = 0

    def work():
        for _ in range(500):
            llm.parse(NarrativeLabel, "p")

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert llm.usage() == {"llm_calls": 4000, "llm_input_tokens": 12000, "llm_output_tokens": 8000}


def test_label_budget_skips_only_a_suffix_of_jobs(monkeypatch):
    """Regression (review): with ex.map the budget was checked inside each worker, after
    it had dequeued its job. A worker that dequeued N1 but read the clock late could be
    budget-skipped while another worker had already started N2, so a BIGGER narrative got
    the keyword label. The clock is monotonic here: the delayed reader simply reads later.
    The skipped set must always be a suffix of the (biggest-first) job order."""
    import threading
    import time

    state = {"t": 0.0, "worker_reads": 0}
    lock = threading.Lock()

    def clock():
        if threading.current_thread().name.startswith("narr-label"):
            with lock:
                state["worker_reads"] += 1
                first = state["worker_reads"] == 1
            if first:
                time.sleep(0.3)  # this worker is descheduled between dequeue and clock read
        return state["t"]

    class SlowLLM(FakeLLM):
        def parse(self, schema, prompt, *, system="", purpose="", max_tokens=4096):
            state["t"] = 1000.0  # every call 'takes' longer than the budget
            return super().parse(schema, prompt, system=system, purpose=purpose, max_tokens=max_tokens)

    monkeypatch.setattr(narratives, "_clock", clock)
    monkeypatch.setattr(narratives, "LABEL_WORKERS", 2)
    monkeypatch.setattr(narratives, "LABEL_TIME_BUDGET_S", 150.0)
    jobs = [narratives._LabelJob(f"N{i}", [f"post text {i}"]) for i in range(1, 5)]
    res = narratives._label_all(SlowLLM(), "news", jobs)
    whys = [why for _, why in res]
    assert "budget" in whys
    first_skip = whys.index("budget")
    assert all(w == "budget" for w in whys[first_skip:]), whys
    assert whys[0] == "llm"  # the biggest narrative is never the one skipped


# ------------------------------------------------ label completeness and relevance


# Real labels from the bundled sample run's llm_log.jsonl: the model wrote them whole,
# the old 8-word cap cut them to "... holdings in", "... card numbers and", "... to rogue".
REAL_LONG_LABELS = [
    "Bank of America increasing Bitcoin ETF holdings in Q3",
    "Posts sharing Bank of America card numbers and approval listings",
    "Posts linking a week of outages to rogue AI",
]


class LabelLLM(FakeLLM):
    """Returns fixed label fields (only those given) for every call."""

    def __init__(self, **fields):
        super().__init__()
        self.fields = {"label": "Label", "claim_summary": "Posts claim that x.", **fields}

    def parse(self, schema, prompt, *, system="", purpose="", max_tokens=4096):
        self.calls.append({"schema": schema, "prompt": prompt, "system": system, "purpose": purpose})
        return schema(**self.fields)


@pytest.mark.parametrize("label", REAL_LONG_LABELS)
def test_llm_label_not_cut_mid_phrase(label):
    """Regression: the sample run showed labels ending in "in", "and", "rogue"."""
    narrs = build_narratives(_corpus(), topic="bank of america", llm=LabelLLM(label=label))
    assert {n.label for n in narrs} == {label}


@pytest.mark.parametrize("text, expected", [
    ("Short label", "Short label"),
    ('  "Quoted   label"  ', "Quoted label"),
    ("Disputed statements and tribute to 'Hass'", "Disputed statements and tribute to 'Hass'"),
    ("Outage claims, ", "Outage claims"),
    ("U.S.", "U.S."),
])
def test_clean_label_keeps_short_labels_whole(text, expected):
    assert narratives._clean_label(text) == expected


def test_clean_label_cuts_long_label_at_word_without_dangling_word():
    long = ("Posts linking a week of outages to rogue AI and blaming the operators of "
            "the banks for it")
    out = narratives._clean_label(long)
    assert len(out) <= narratives.LABEL_MAX_CHARS
    assert long.startswith(out)
    assert out.split()[-1].lower() not in narratives._DANGLING
    assert out == "Posts linking a week of outages to rogue AI and blaming the operators"
    # Word boundary: a cut never splits a word.
    assert long[len(out)] == " "


def test_clean_label_single_overlong_word_hard_cut():
    assert narratives._clean_label("x" * 200) == "x" * narratives.LABEL_MAX_CHARS


def test_label_prompt_has_relevance_definitions_and_topic():
    llm = LabelLLM()
    build_narratives(_corpus(), topic="bank of america", llm=llm)
    prompt = llm.calls[0]["prompt"]
    assert "bank of america" in prompt
    for word in ("threat", "related", "off_topic", "security incident", "markets",
                 "mentioned in passing"):
        assert word in prompt
    assert len(llm.calls) == 3  # same single call per narrative, no extra calls


@pytest.mark.parametrize("given, expected", [
    ("threat", "threat"), ("related", "related"), ("off_topic", "off_topic"),
    ("Off-topic", "off_topic"), (" THREAT ", "threat"),
    ("critical", None), ("", None), ("threat, related", None),
])
def test_relevance_validated(given, expected):
    llm = LabelLLM(relevance=given, relevance_reason="Reports a bank outage.")
    narrs = build_narratives(_corpus(), topic="bank", llm=llm)
    for n in narrs:
        assert n.relevance == expected
        assert n.relevance_reason == ("Reports a bank outage." if expected else "")


def test_relevance_missing_is_none():
    narrs = build_narratives(_corpus(), topic="bank", llm=LabelLLM())  # no relevance field
    assert all(n.relevance is None and n.relevance_reason == "" for n in narrs)


def test_relevance_reason_capped():
    llm = LabelLLM(relevance="related", relevance_reason="word " * 80)
    narrs = build_narratives(_corpus(), topic="bank", llm=llm)
    r = narrs[0].relevance_reason
    assert len(r) <= narratives.RELEVANCE_REASON_MAX_CHARS and r.endswith("…")


def test_keyword_labels_have_no_relevance():
    narrs = build_narratives(_corpus(), topic="news", llm=None)
    assert all(n.relevance is None and n.relevance_reason == "" for n in narrs)
    failed = build_narratives(_corpus(), topic="news", llm=FakeLLM(fail_on={"N1", "N2", "N3"}))
    assert all(n.relevance is None and n.label_method == "keyword" for n in failed)


# ------------------------------------------------------------------------- severity
def test_label_prompt_has_severity_definitions_in_same_call():
    llm = LabelLLM()
    build_narratives(_corpus(), topic="bank of america", llm=llm)
    prompt = llm.calls[0]["prompt"]
    for word in ("critical", "high", "medium", "low", "impersonation", "bank-run",
                 "routine news", "relative to the search topic"):
        assert word in prompt
    assert len(llm.calls) == 3  # severity comes from the existing label call only


@pytest.mark.parametrize("given, expected", [
    ("critical", "critical"), ("high", "high"), ("medium", "medium"), ("low", "low"),
    (" HIGH ", "high"), ("severe", None), ("", None), ("high, medium", None),
])
def test_severity_validated(given, expected):
    llm = LabelLLM(relevance="threat", severity=given,
                   severity_reason="Card numbers offered for sale.")
    for n in build_narratives(_corpus(), topic="bank", llm=llm):
        assert n.severity == expected
        assert n.severity_reason == ("Card numbers offered for sale." if expected else "")


def test_severity_none_for_off_topic_missing_and_keyword():
    off = build_narratives(_corpus(), topic="bank",
                           llm=LabelLLM(relevance="off_topic", severity="low",
                                        severity_reason="passing mention"))
    assert all(n.severity is None and n.severity_reason == "" for n in off)
    missing = build_narratives(_corpus(), topic="bank", llm=LabelLLM(relevance="related"))
    assert all(n.severity is None for n in missing)
    kw = build_narratives(_corpus(), topic="news", llm=None)
    assert all(n.severity is None and n.severity_reason == "" for n in kw)
    failed = build_narratives(_corpus(), topic="news", llm=FakeLLM(fail_on={"N1", "N2", "N3"}))
    assert all(n.severity is None for n in failed)


def test_bad_severity_keeps_label_and_relevance():
    llm = LabelLLM(label="Card sales", relevance="threat", relevance_reason="r",
                   severity="catastrophic", severity_reason="x" * 300)
    n = build_narratives(_corpus(), topic="bank", llm=llm)[0]
    assert n.label == "Card sales" and n.label_method == "llm" and n.relevance == "threat"
    assert n.severity is None and n.severity_reason == ""


def test_severity_reason_capped():
    llm = LabelLLM(relevance="threat", severity="high", severity_reason="word " * 80)
    r = build_narratives(_corpus(), topic="bank", llm=llm)[0].severity_reason
    assert len(r) <= narratives.RELEVANCE_REASON_MAX_CHARS and r.endswith("…")


# ------------------------------------------------- labeller context (2026-11-19 review)
SUPPORT = [  # one official support account replying to customers (sample-run N41 shape)
    "Hello, we noticed you tagged us. If we can help with an account concern, please use the DM link",
    "We noticed you tagged us. Please use the DM link below to share your name and zip code",
    "Hi, we saw your post. If you have account questions, please click the DM link below",
    "Hello, we saw that you mentioned us. If we can help, please click the DM link below",
    "We would like to learn more about how we can help. Please use the DM link below",
    "Hi, we noticed your post. Please use the DM link to connect with us privately",
]


def test_label_prompt_names_platform_account_kind_and_distinct_accounts():
    posts = [make_post(f"s{i}", t, author="BofA_Help", author_handle="BofA_Help")
             for i, t in enumerate(SUPPORT)]
    llm = LabelLLM()
    narrs = build_narratives(posts, topic="bank of america", llm=llm)
    assert len(narrs) == 1
    prompt = llm.calls[0]["prompt"]
    assert "(X, @BofA_Help, post)" in prompt
    k = len(narrs[0].member_post_ids)
    assert k >= 2 and f"These {k} posts from 1 distinct account " in prompt


def test_label_prompt_names_the_advertiser_for_ads():
    from socmint.schema import AdMeta

    text = "Bank of America Employees: get a free laser teeth whitening at our dental practice"
    posts = [make_post(f"ad{i}", f"{text} offer {i}", source="linkedin_ads", author=None,
                       author_handle=None, author_name="212 Dental Care - NYC", kind="ad",
                       ad_meta=AdMeta(run_start="2026-11-14"))
             for i in range(4)]
    llm = LabelLLM()
    build_narratives(posts, topic="bank of america", llm=llm,
                     window_start="2026-11-13")
    prompt = llm.calls[0]["prompt"]
    assert "(LinkedIn Ads, ad, advertiser: 212 Dental Care - NYC)" in prompt


def test_severity_rubric_rates_what_content_does_not_truth():
    llm = LabelLLM()
    build_narratives(_corpus(), topic="bank of america", llm=llm)
    prompt = llm.calls[0]["prompt"]
    low = prompt.lower()
    assert "false or unconfirmed" not in low
    assert "not by whether it is true" in low
    assert "plain news reporting of an incident" in low
    assert "official accounts" in low and "not impersonation" in low
    assert "one ordinary advertiser" in low
    assert "do not actually discuss the subject, give severity low" in low
    assert "never use the words 'true', 'false', 'unconfirmed'" in low


@pytest.mark.parametrize("reason, expected", [
    ("Unconfirmed outage claims across banks could cause panic and financial worry.",
     "Outage claims across banks could cause panic and financial worry."),
    ("Confirmed-appearing outage news from reputable outlets, since resolved",
     "Outage news from reputable outlets, since resolved"),
    ("Spreads false claims of a bank lockdown", "Spreads claims of a bank lockdown"),
    ("Card numbers offered for sale", "Card numbers offered for sale"),
    # live re-run 2026-11-19: "false-flag" lost its first word; compounds are kept
    ("Alarm-spreading false-flag and panic claims about banks",
     "Alarm-spreading false-flag and panic claims about banks"),
    ("A true-crime channel retells the robbery", "A true-crime channel retells the robbery"),
])
def test_severity_reason_drops_truth_words(reason, expected):
    llm = LabelLLM(relevance="threat", severity="high", severity_reason=reason)
    n = build_narratives(_corpus(), topic="bank", llm=llm)[0]
    assert n.severity_reason == expected
    for w in ("unconfirmed", "false", "true", "confirmed"):
        assert w not in n.severity_reason.lower().split()


@pytest.mark.parametrize("sev_reason, rel_reason, expected", [
    ("Alarmist claims", "Bank of America is not named in these posts", "medium"),
    ("Posts do not mention the bank at all", "", "medium"),
    ("Calls to withdraw money could cause a bank run", "Outage at Bank of America", "high"),
    ("No fraud claimed but rumours spread", "Rumours about the bank", "high"),
])
def test_severity_capped_when_subject_not_named(sev_reason, rel_reason, expected):
    llm = LabelLLM(relevance="threat", relevance_reason=rel_reason, severity="high",
                   severity_reason=sev_reason)
    assert build_narratives(_corpus(), topic="bank", llm=llm)[0].severity == expected


def test_reason_kept_whole_up_to_hard_cap_then_cut_at_clause():
    # the sample-run N41 reason was cut at 140 chars mid-clause ("... phone raises…")
    full = ("Many accounts post the same DM-solicitation wording, which suggests coordinated "
            "lure content; request for name, zip and phone raises the risk of phishing of "
            "customers who tag the bank")
    assert len(full) > narratives.REASON_MAX_CHARS
    out = narratives._clean_reason(full)
    assert out == ("Many accounts post the same DM-solicitation wording, which suggests "
                   "coordinated lure content")
    short = "Request for name, zip and phone raises phishing concerns for bank customers"
    assert narratives._clean_reason(short) == short
    mid = "x" * 100 + " and then a long tail of words without any clause boundary at all here ok"
    cut = narratives._clean_reason(mid)
    assert cut.endswith("…") and len(cut) <= narratives.REASON_MAX_CHARS
    assert cut[:-1].split()[-1] not in narratives._DANGLING


# 2026-11-20 review: phrasings from the live re-run that the cap missed, and negatives
@pytest.mark.parametrize("sev_reason, rel_reason, expected", [
    ("Card-dump bots; no Bank of America-specific attack is named", "", "medium"),
    ("Generic fraud tutorial", "Does not focus on Bank of America specifically", "medium"),
    ("Bank of America only appears in one unrelated map video", "", "medium"),
    ("Hostile posts, but not a direct attack on the bank", "", "medium"),
    ("Threats against staff", "Names Bank of America directly", "high"),
    ("Directly targets Bank of America customers with phishing links", "", "high"),
    ("Not only names the bank but threatens its branches", "", "high"),
    ("No withdrawals possible, bank named as frozen", "", "high"),
])
def test_severity_cap_live_phrasings(sev_reason, rel_reason, expected):
    llm = LabelLLM(relevance="threat", relevance_reason=rel_reason, severity="high",
                   severity_reason=sev_reason)
    assert build_narratives(_corpus(), topic="bank", llm=llm)[0].severity == expected


def test_relevance_reason_drops_truth_words_like_severity_reason():
    llm = LabelLLM(relevance="related", relevance_reason="Spreads false claims of a bank "
                   "lockdown", severity="low", severity_reason="Unconfirmed outage claims")
    n = build_narratives(_corpus(), topic="bank", llm=llm)[0]
    assert n.relevance_reason == "Spreads claims of a bank lockdown"
    assert n.severity_reason == "Outage claims"
    # compounds naming a kind of content are kept, as in severity_reason
    llm = LabelLLM(relevance="related", relevance_reason="A false-flag theory about the bank")
    assert build_narratives(_corpus(), topic="bank", llm=llm)[0].relevance_reason == \
        "A false-flag theory about the bank"


# ------------------------------------- severity rubric for any subject (2026-11-20 tuning)
def test_severity_rubric_covers_any_kind_of_subject():
    llm = LabelLLM()
    build_narratives(_corpus(), topic="measles vaccine", llm=llm)
    low = llm.calls[0]["prompt"].lower()
    for words in ("company, person, product, event, policy", "voters", "patients",
                  "the public", "recall", "stay away from a vote"):
        assert words in low
    # explicit guidance for the over-ratings seen in the sample (N55, N202, N218, N129)
    assert "satire, jokes and mockery of a claim are low" in low
    assert "personal appeals (donations" in low and "solicit money fraudulently" in low
    assert "complaints about service" in low and "medium at most" in low
    assert len(llm.calls) == 3  # still one call per narrative


@pytest.mark.parametrize("label, summary, given, expected", [
    # sample run N55 / N202 (bank) were "high"
    ("Commenters mock claims of a cyber attack on Bank of America",
     "Posts claim the attack story is overblown.", "high", "medium"),
    ("Users joke about withdrawing money and praising Bank of America",
     "Posts claim that x.", "high", "medium"),
    # other kinds of subject
    ("Satirical posts about microchips in the measles vaccine", "Posts claim that x.",
     "high", "medium"),
    ("Parody accounts ridicule the candidate's debate gaffe", "Posts claim that x.",
     "high", "medium"),
    # not mockery, negated, or critical: unchanged
    ("Rumours that the recalled stroller collapses", "Posts claim that x.", "high", "high"),
    ("Voters say the ballot fraud is no joke", "Posts claim that x.", "high", "high"),
    ("Ridiculous fees anger airline customers", "Posts claim that x.", "high", "high"),
    ("Sellers mock victims while selling leaked card data", "Posts claim that x.",
     "critical", "critical"),
    ("Users joke about the outage", "Posts claim that x.", "medium", "medium"),
    # "spoof"/"parody" name fraud and impersonation, not mockery (review 2026-11-20)
    ("Spoofed calls impersonate the bank", "Posts claim that x.", "high", "high"),
    ("Scammers send spoofed emails", "Posts claim that x.", "high", "high"),
    ("Parody accounts impersonate the senator", "Posts claim that x.", "high", "high"),
    # mockery around a harm (leak, breach, scam, phishing, threats) is about the harm
    ("Users mock victims after data leak", "Posts claim that x.", "high", "high"),
    ("Commenters joke about the phishing texts", "Posts claim that x.", "high", "high"),
])
def test_mockery_label_caps_high_at_medium(label, summary, given, expected):
    llm = LabelLLM(label=label, claim_summary=summary, relevance="threat", severity=given,
                   severity_reason="Spreads alarm about the subject.")
    for n in build_narratives(_corpus(), topic="anything", llm=llm):
        assert n.severity == expected


@pytest.mark.parametrize("raw, n_shown, expected", [
    ([2, 5], 12, (1, 4)), ([5, 2, 2], 12, (1, 4)), ([0, 13, 3], 12, (2,)),
    ([], 12, ()), (None, 12, ()),
])
def test_clean_outliers(raw, n_shown, expected):
    assert narratives._clean_outliers(raw, n_shown) == expected


def test_outliers_field_is_lenient_and_optional():
    assert NarrativeLabel(label="L", claim_summary="S").outliers == []
    assert NarrativeLabel(label="L", claim_summary="S", outliers=["[3]", "x", 4]).outliers == [3, 4]
    assert NarrativeLabel(label="L", claim_summary="S", outliers="3").outliers == []


def test_label_prompt_asks_for_outliers_in_the_same_call():
    llm = LabelLLM()
    build_narratives(_corpus(), topic="x", llm=llm)
    low = llm.calls[0]["prompt"].lower()
    assert "outliers" in low and "do not fit your label" in low
    assert len(llm.calls) == 3


# ---------------------------------------------------- outlier pruning (mixed narratives)
def _unit(*xs):
    v = np.array(xs, dtype=float)
    return v / np.linalg.norm(v)


def test_prune_outliers_synthetic_vectors():
    # rows 0-3 near axis A (fit), rows 4-6 near axis B (outlier theme), row 7 a near-tie
    vecs = np.array([_unit(1, .1, 0), _unit(1, 0, .1), _unit(1, .05, .05), _unit(.9, .1, .1),
                     _unit(.1, 1, 0), _unit(.1, 1, .1), _unit(0, 1, .1), _unit(.72, .69, 0)])
    idx = list(range(8))
    shown = [0, 1, 4]  # model saw two fitting posts and one outlier (position 2)
    kept, removed = narratives._prune_outliers(idx, shown, (2,), vecs, 0.02)
    assert sorted(removed) == [4, 5, 6] and kept == [0, 1, 2, 3, 7]  # near-tie 7 kept by margin
    assert removed[0] == 4  # the flagged member first, then by distance
    # never prune without outliers, or with fewer than 2 fitting examples
    assert narratives._prune_outliers(idx, shown, (), vecs, 0.02) == (idx, [])
    assert narratives._prune_outliers(idx, [0, 4, 5], (1, 2), vecs, 0.02) == (idx, [])
    # a fitting example is never removed, even if it lies closer to the outliers
    kept, removed = narratives._prune_outliers(idx, [0, 7, 4], (2,), vecs, 0.02)
    assert 7 in kept and 0 in kept and 4 in removed


def _mixed_case(monkeypatch, fit_texts, out_texts, max_shown):
    """One forced cluster of fitting posts plus off-theme posts (earliest first), with
    synthetic vectors; the labeller is shown only `max_shown` posts."""
    posts, rows = [], []
    for i, t in enumerate(out_texts):
        posts.append(make_post(f"out{i}", t, source="youtube", author="reviewer",
                               published_at=f"2026-11-0{1 + i}T09:00:00Z"))
        rows.append(_unit(.1 + .01 * i, 1, .05))
    for i, t in enumerate(fit_texts):
        posts.append(make_post(f"fit{i}", t, source="telegram", author=f"ch{i % 2}",
                               published_at=f"2026-11-1{i % 5}T10:00:00Z"))
        rows.append(_unit(1, .05 + .01 * i, .1))
    vecs = np.array(rows)
    monkeypatch.setattr(narratives, "embed_texts", lambda texts: (vecs, "sentence-transformers:test"))
    monkeypatch.setattr(narratives, "_cluster", lambda v, m: np.zeros(len(v), dtype=int))
    monkeypatch.setattr(narratives, "MAX_LLM_TEXTS", max_shown)
    return posts


class OutlierLLM(LabelLLM):
    """Flags as outliers the shown posts whose text contains `marker`."""

    def __init__(self, marker, **fields):
        super().__init__(**fields)
        self.marker = marker

    def parse(self, schema, prompt, *, system="", purpose="", max_tokens=4096):
        self.calls.append({"schema": schema, "prompt": prompt, "system": system, "purpose": purpose})
        nums = [int(m.group(1)) for m in __import__("re").finditer(r"\[(\d+)\] \([^)]*\) ([^\n]*)", prompt)
                if self.marker in m.group(2)]
        return schema(**self.fields, outliers=nums)


CARD_DUMPS = [f"NEW CARD Approved 4111******{1000 + i}|0{1 + i % 9}|2030 VISA DEBIT BANK OF AMERICA"
              for i in range(8)]
CARD_REVIEWS = ["review: $40,000 limit on one Bank of America card, how to ask for more",
                "review: 3 Bank of America business cards with 1 inquiry",
                "review: Bank of America Premium Rewards Elite first look, $550 metal card"]


def test_mixed_card_dump_narrative_pruned_and_first_seen_recomputed(monkeypatch):
    """Sample-run N6 shape: card dumps plus card-review videos; the reviews are earliest."""
    posts = _mixed_case(monkeypatch, CARD_DUMPS, CARD_REVIEWS, max_shown=5)
    llm = OutlierLLM("review:", label="Telegram channels posting card data",
                     relevance="threat", severity="critical", severity_reason="Card data.")
    stats: dict = {}
    trace: dict = {}
    n, split = build_narratives(posts, topic="bank of america", llm=llm, stats=stats,
                                trace=trace)
    assert trace["pruned_split_into"] == {"N1": "N2"}
    assert set(trace["pruned_post_ids"]["N1"]) == {f"youtube:out{i}" for i in range(3)}
    # the 3 reviews are >= the minimum narrative size (3) and coherent: their own narrative,
    # labelled by one extra call within the label budget
    assert len(llm.calls) == 2 and llm.calls[1]["purpose"] == "narrative_label:N2"
    assert split.id == "N2" and split.label_method == "llm"
    assert {p for p in split.member_post_ids} == {f"youtube:out{i}" for i in range(3)}
    assert stats == {"posts_pruned": 3, "posts_pruned_unassigned": 0, "narratives_split": 1, "narratives_mixed": 0,
                     "narratives_label_llm": 2, "narratives_label_keyword": 0,
                     "narratives_label_small": 0}
    shown_reviews = llm.calls[0]["prompt"].count("review:")
    assert 1 <= shown_reviews < len(CARD_REVIEWS)  # some reviews were never shown...
    review_ids = {p.post_id for p in posts if p.post_id.startswith("youtube:out")}
    assert not review_ids & set(n.member_post_ids)  # ...yet all were pruned by similarity
    assert not review_ids & set(n.example_post_ids)
    assert len(n.member_post_ids) == len(CARD_DUMPS)
    assert n.volume_by_platform == {"telegram": len(CARD_DUMPS)}
    assert sum(n.volume_by_day.values()) == len(CARD_DUMPS)
    assert n.first_seen_at == "2026-11-10T10:00:00Z"  # was 2026-11-01 (a review)
    assert "review" not in " ".join(n.keywords)
    assert n.severity == "critical" and len(posts) == 11  # input posts untouched


def test_mixed_election_narrative_pruned_non_bank(monkeypatch):
    fit = [f"Observers filmed ballot stuffing at polling station {i}, the election was rigged"
           for i in range(6)]
    off = ["Polling stations open from 8am to 8pm, bring your ID card",
           "Reminder: polling station opening hours and what ID to bring"]
    posts = _mixed_case(monkeypatch, fit, off, max_shown=12)
    llm = OutlierLLM("bring", label="Claims of ballot stuffing", relevance="threat",
                     severity="high", severity_reason="Fraud allegations.")
    stats: dict = {}
    [n] = build_narratives(posts, topic="regional election", llm=llm, stats=stats)
    assert all(pid.startswith("telegram:fit") for pid in n.member_post_ids)
    assert len(n.member_post_ids) == 6 and n.unverified_member_count == 0
    # 2 removed posts are below the minimum narrative size (3): unassigned, no split
    assert stats == {"posts_pruned": 2, "posts_pruned_unassigned": 2, "narratives_split": 0, "narratives_mixed": 0,
                     "narratives_label_llm": 1, "narratives_label_keyword": 0,
                     "narratives_label_small": 0}
    assert len(llm.calls) == 1


def _spread_case(monkeypatch, n_fit, n_near, near_vec, out_text, fit_text, near_text):
    """One forced cluster: n_fit posts on the label's theme, n_near posts lying nearer the
    outlier than the fitting examples (near-ties in embedding space) and ONE off-theme
    post from a source no other member uses (so the labeller is shown it)."""
    posts = [make_post("out0", out_text, source="youtube", author="vlog",
                       published_at="2026-11-01T09:00:00Z")]
    rows = [_unit(.2, 1, 0)]
    for i in range(n_fit):
        posts.append(make_post(f"fit{i}", f"{fit_text} {i}", source="telegram",
                               author=f"ch{i}", published_at="2026-11-10T10:00:00Z"))
        rows.append(_unit(1, .1 + .001 * i, .02))
    for i in range(n_near):
        posts.append(make_post(f"near{i}", f"{near_text} {i}", source="x", author=f"u{i}",
                               published_at="2026-11-11T10:00:00Z"))
        rows.append(_unit(*near_vec) + np.array([0, 0, .001 * i]))
    vecs = np.array(rows)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    monkeypatch.setattr(narratives, "embed_texts", lambda texts: (vecs, "sentence-transformers:test"))
    monkeypatch.setattr(narratives, "_cluster", lambda v, m: np.zeros(len(v), dtype=int))
    return posts


def test_one_flagged_of_twelve_never_spreads_to_a_large_nearby_group(monkeypatch):
    """Live re-run 2026-11-20 (N3: 2 of 12 flagged, 72 of 119 removed). Product-recall
    case: 1 of 12 shown flagged, 30 members lie nearer the outlier than the fitting
    examples. Removal is capped at ceil(1/12 * 101) = 9, furthest first, the flagged one
    included; the 9 are coherent and >= the minimum size, so they become a narrative."""
    posts = _spread_case(monkeypatch, 70, 30, (.6, .8, 0),
                         "review: unboxing the new stroller model, great wheels",
                         "Recall warning: stroller frame can collapse, stop using it",
                         "Recall refund: how to claim money back for the stroller")
    llm = OutlierLLM("review:", label="Stroller recall warnings", relevance="threat",
                     severity="high", severity_reason="Safety warnings about a product.")
    stats: dict = {}
    narrs = build_narratives(posts, topic="stroller recall", llm=llm, stats=stats)
    assert llm.calls[0]["prompt"].count("review:") == 1
    assert "(12 shown)" in llm.calls[0]["prompt"]
    main = narrs[0]
    assert len(main.member_post_ids) == 101 - 9  # not 101 - 31 as before the cap
    assert "youtube:out0" not in main.member_post_ids
    assert stats["posts_pruned"] == 9 and stats["narratives_split"] == 1
    assert stats["posts_pruned_unassigned"] == 0
    [split] = narrs[1:]
    assert split.id == "N2" and "youtube:out0" in split.member_post_ids
    assert len(split.member_post_ids) == 9
    # the nearest near-ties stay; only those furthest toward the outlier left
    assert all(p.startswith(("x:near", "youtube:")) for p in split.member_post_ids)
    # every member is in at most one narrative and nothing was lost
    ids = [p for n in narrs for p in n.member_post_ids]
    assert len(ids) == len(set(ids)) == 101


def test_split_narrative_gets_keyword_label_when_budget_is_spent(monkeypatch):
    posts = _spread_case(monkeypatch, 70, 30, (.6, .8, 0),
                         "review: campaign rally livestream highlights",
                         "Ballots were destroyed at the county office, election rigged",
                         "Ballot drop box moved, voters told to vote elsewhere")
    state = {"t": 0.0}
    monkeypatch.setattr(narratives, "_clock", lambda: state["t"])

    class SlowLLM(OutlierLLM):
        def parse(self, *a, **k):
            out = super().parse(*a, **k)
            state["t"] = 1000.0  # the whole label budget is spent by the first call
            return out

    llm = SlowLLM("review:", label="Claims of destroyed ballots", relevance="threat",
                  severity="high", severity_reason="Alarm about the vote.")
    narrs = build_narratives(posts, topic="county election", llm=llm)
    assert len(llm.calls) == 1  # no call for the split narrative
    assert [n.label_method for n in narrs] == ["llm", "keyword"]
    assert narrs[1].severity is None and narrs[1].relevance is None


def test_coherent_group_uses_the_clusterers_own_thresholds():
    a = [_unit(1, .02 * i, 0) for i in range(4)]
    b = [_unit(0, .02 * i, 1) for i in range(4)]
    vecs = np.array(a + b + [_unit(0, 1, 0)])
    # one theme: kept whole
    assert narratives._coherent_group([0, 1, 2, 3], vecs, "dense") == [0, 1, 2, 3]
    # a lone off member with no similar co-member is dropped, the rest stay
    assert narratives._coherent_group([0, 1, 2, 8], vecs, "dense") == [0, 1, 2]
    # two unrelated themes in one removed set are not one narrative
    assert narratives._coherent_group([0, 1, 4, 5], vecs, "dense") == []
    assert narratives._coherent_group([0], vecs, "dense") == []


def test_min_cluster_size_matches_clusterer():
    assert narratives._min_cluster_size(5) == 2
    assert narratives._min_cluster_size(100) == 3
    assert narratives._min_cluster_size(1935) == 15


def test_no_pruning_without_outliers_or_enough_fitting_examples(monkeypatch):
    posts = _mixed_case(monkeypatch, CARD_DUMPS[:2], CARD_REVIEWS, max_shown=12)
    # model flags everything except one post: fewer than 2 fitting examples -> no pruning
    llm = OutlierLLM("", label="Product recall warnings")
    [n] = build_narratives(posts, topic="stroller recall", llm=llm)
    assert len(n.member_post_ids) == len(posts)
    [n] = build_narratives(posts, topic="stroller recall", llm=LabelLLM())
    assert len(n.member_post_ids) == len(posts)


def test_coherent_small_removed_group_in_large_run_becomes_its_own_narrative(monkeypatch):
    """Review 2026-11-20 (live re-run N3): the split threshold was HDBSCAN's
    min_cluster_size (15 for 1,935 candidates), so a coherent 10-post story removed from a
    narrative went to no narrative. Non-bank shape: a 1,000-post heatwave narrative with
    10 posts about one hospital's generator failure. The minimum split size is now the
    smallest narrative the run labels (3), and the pruning is recorded per narrative."""
    posts, rows = [], []
    for i in range(1000):
        posts.append(make_post(f"heat{i}", f"Heatwave warning for the city, stay indoors {i}",
                               source="telegram", author=f"ch{i % 40}",
                               published_at="2026-11-10T10:00:00Z"))
        rows.append(_unit(1, .05 + .00001 * i, .1))
    for i in range(10):
        posts.append(make_post(f"gen{i}", f"generator: St Mary hospital generator failed {i}",
                               source=("x", "youtube", "linkedin")[i % 3], author=f"u{i}",
                               published_at="2026-11-11T10:00:00Z"))
        rows.append(_unit(.1 + .001 * i, 1, .05))
    vecs = np.array(rows)
    monkeypatch.setattr(narratives, "embed_texts", lambda texts: (vecs, "sentence-transformers:test"))
    monkeypatch.setattr(narratives, "_cluster", lambda v, m: np.zeros(len(v), dtype=int))
    assert narratives._min_cluster_size(1010) == 15 and narratives._split_min_size(1010) == 3
    llm = OutlierLLM("generator:", label="Heatwave warnings", relevance="related",
                     severity="medium", severity_reason="Public safety advice.")
    stats: dict = {}
    trace: dict = {}
    narrs = build_narratives(posts, topic="city heatwave", llm=llm, stats=stats, trace=trace)
    gen_ids = {p.post_id for p in posts if ":gen" in p.post_id}
    assert len(narrs) == 2
    assert not gen_ids & set(narrs[0].member_post_ids)
    assert set(narrs[1].member_post_ids) == gen_ids and narrs[1].id == "N2"
    assert stats["posts_pruned"] == 10 and stats["posts_pruned_unassigned"] == 0
    assert set(trace["pruned_post_ids"]["N1"]) == gen_ids
    assert trace["pruned_split_into"] == {"N1": "N2"} and trace["narratives_mixed"] == {}


def test_pruned_posts_recorded_even_when_left_unassigned(monkeypatch):
    fit = [f"Observers filmed ballot stuffing at polling station {i}, the election was rigged"
           for i in range(6)]
    off = ["Polling stations open from 8am to 8pm, bring your ID card",
           "Reminder: polling station opening hours and what ID to bring"]
    posts = _mixed_case(monkeypatch, fit, off, max_shown=12)
    trace: dict = {}
    build_narratives(posts, topic="regional election", llm=OutlierLLM("bring", label="Ballot "
                     "stuffing claims", severity="high"), trace=trace)
    assert sorted(trace["pruned_post_ids"]["N1"]) == ["youtube:out0", "youtube:out1"]
    assert trace["pruned_split_into"] == {}


def test_half_or_more_flagged_keeps_the_narrative_whole_as_mixed(monkeypatch):
    """Review 2026-11-20 (live re-run N53: 3 of 5 removed, label kept). Non-bank: a
    vaccine-rumour cluster where the model flags 4 of the 8 posts it was shown."""
    fit = [f"Rumour: the measles vaccine batch {i} was contaminated" for i in range(4)]
    off = [f"clinic: opening hours for the walk-in clinic, week {i}" for i in range(4)]
    posts = _mixed_case(monkeypatch, fit, off, max_shown=12)
    stats: dict = {}
    trace: dict = {}
    [n] = build_narratives(posts, topic="measles vaccine", llm=OutlierLLM(
        "clinic:", label="Contaminated vaccine batch rumours", severity="high"),
        stats=stats, trace=trace)
    assert len(n.member_post_ids) == 8
    assert stats["posts_pruned"] == 0 and stats["narratives_mixed"] == 1
    assert trace["narratives_mixed"]["N1"]["reason"] == "share"
    assert trace["narratives_mixed"]["N1"]["flagged"] == 4
    assert trace["narratives_mixed"]["N1"]["shown"] == 8
    assert sorted(trace["narratives_mixed"]["N1"]["flagged_post_ids"]) == [
        f"youtube:out{i}" for i in range(4)] and trace["pruned_post_ids"] == {}


def test_pruning_never_leaves_fewer_members_than_a_labelled_narrative(monkeypatch):
    fit = ["Stadium roof leak floods the east stand", "East stand flooded by roof leak"]
    off = ["tickets: resale prices for the cup final"]
    posts = _mixed_case(monkeypatch, fit, off, max_shown=12)
    trace: dict = {}
    [n] = build_narratives(posts, topic="city stadium", llm=OutlierLLM(
        "tickets:", label="Stadium roof leak"), trace=trace)
    assert len(n.member_post_ids) == 3  # 1 of 3 flagged, but 2 left would be < 3
    assert trace["narratives_mixed"] == {"N1": {"flagged": 1, "shown": 3, "reason": "size",
                                         "flagged_post_ids": ["youtube:out0"]}}


@pytest.mark.parametrize("label, given, expected", [
    # posts pushing back on a claim do not spread it (non-bank subjects)
    ("Fact-checkers debunk claim the minister resigned", "high", "medium"),
    ("Posts refute viral claim that the vaccine causes infertility", "high", "medium"),
    ("Users fact-check the stadium collapse rumour", "high", "medium"),
    # "dismiss X as Y" asserts Y: live re-run N86 (bank) spread the attack claim; kept
    ("Commenters dismiss Bank of America software bug as cyberattack", "high", "high"),
    ("Fans dismiss the stadium inspection report as a cover-up", "high", "high"),
    # the subject (not the posters) acts, negated, or a harm named: kept
    ("CEO dismissed safety warnings before the crash", "high", "high"),
    ("Claims of rigged ballots were not debunked", "high", "high"),
    ("Users debunk warnings about the phishing texts", "high", "high"),
    ("Fact-checkers debunk claim of a data leak", "critical", "critical"),
])
def test_dismissal_label_caps_high_at_medium(label, given, expected):
    llm = LabelLLM(label=label, relevance="threat", severity=given,
                   severity_reason="Spreads alarm about the subject.")
    for n in build_narratives(_corpus(), topic="anything", llm=llm):
        assert n.severity == expected


# ------------------------------------------------- single-advertiser ad narratives
def _ad_posts(advertisers):
    from socmint.schema import AdMeta
    texts = [f"Smile Dental Studio: same-day crowns and whitening near Acme Arena, book {i}"
             for i in range(len(advertisers))]
    return [make_post(f"ad{i}", t, source="linkedin_ads", author=adv, kind="ad",
                      ad_meta=AdMeta(advertiser_id=adv), published_at="2026-11-05T00:00:00Z")
            for i, (t, adv) in enumerate(zip(texts, advertisers, strict=True))]


@pytest.mark.parametrize("advertisers, label, given, expected", [
    # one ordinary advertiser (a dental clinic naming the topic): capped at medium
    (["adv-smile"] * 4, "Dental clinic ads mentioning Acme Arena", "high", "medium"),
    (["adv-smile"] * 4, "Dental clinic ads mentioning Acme Arena", "critical", "medium"),
    (["adv-smile"] * 4, "Dental clinic ads mentioning Acme Arena", "low", "low"),
    # the ads themselves are described as impersonation / scam / phishing: kept
    (["adv-smile"] * 4, "Ads impersonate Acme Arena to sell fake tickets", "high", "high"),
    (["adv-smile"] * 4, "Phishing ads posing as Acme Arena support", "critical", "critical"),
    # several advertisers: not capped by this rule
    (["adv-a", "adv-b", "adv-a", "adv-c"], "Clinic ads mentioning Acme Arena", "high", "high"),
])
def test_single_advertiser_ads_capped_at_medium(advertisers, label, given, expected):
    llm = LabelLLM(label=label, relevance="related", severity=given,
                   severity_reason="Advertises services near the venue.")
    [n] = build_narratives(_ad_posts(advertisers), topic="acme arena", llm=llm,
                           window_start="2026-11-01")
    assert n.severity == expected


def test_single_advertiser_helper_needs_all_ads():
    posts = _ad_posts(["adv-smile"] * 3)
    assert narratives._single_advertiser_ads(posts)
    assert not narratives._single_advertiser_ads(posts + [make_post("p1", "organic post")])
    assert not narratives._single_advertiser_ads([])


@pytest.mark.parametrize("text, fraud", [
    # a product that fights the harm names a defence, not the harm (live re-run N8)
    ("Vendor ad promoting behavioral fraud intelligence for banks", False),
    ("Anti-phishing training ads for hospital staff", False),
    ("Scam awareness webinar for retirees", False),
    ("Fraud detection software ad", False),
    # the ads themselves do the harm
    ("Ads impersonate the airline to sell fake refunds", True),
    ("Phishing ads posing as the ministry's tax portal", True),
    ("Crypto giveaway scam ads using the senator's face", True),
    # a hedge is not a description (live re-run N75 reasons)
    ("Repeated lure ad invoking the bank's employees; possible impersonation", False),
    ("Ad may be a scam", False),
    ("Ad could potentially be phishing", False),
])
def test_fraud_words_exclude_defensive_products(text, fraud):
    assert narratives._describes_fraud(text) is fraud


def test_single_advertiser_cap_ignores_hedged_impersonation_reason():
    """Live re-run N75 shape (non-bank here): one clinic's ads, reason 'possible
    impersonation', label/summary describe an ordinary offer -> medium."""
    llm = LabelLLM(label="Dental clinic ads offering free whitening to Acme Arena staff",
                   claim_summary="Posts claim the clinic offers free whitening.",
                   relevance="related", relevance_reason="Uses the venue's name, a possible "
                   "impersonation", severity="high",
                   severity_reason="Lure ad invoking the venue; possible impersonation")
    [n] = build_narratives(_ad_posts(["adv-smile"] * 4), topic="acme arena", llm=llm,
                           window_start="2026-11-01")
    assert n.severity == "medium"


def test_split_dominant_breaks_up_a_mega_cluster():
    """A single cluster swallowing most posts (homogeneous-topic under-segmentation, live
    2026-11-20 "bank of america": 1,278 of 1,314 assigned in one cluster) is re-split with
    agglomerative when it holds > 2 distinct stories, and left whole when it is one story."""
    rng = np.random.default_rng(0)

    def unit(base, k):
        v = np.tile(np.array(base, dtype=float), (k, 1)) + rng.normal(0, 0.01, (k, len(base)))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    # cluster 0 (100 posts) is really two orthogonal stories A and B; cluster 1 is a small
    # separate story C (10 posts). 100 / 110 assigned > DOMINANT_CLUSTER_SHARE and >= MIN.
    vecs = np.vstack([unit([1, 0, 0, 0], 50), unit([0, 1, 0, 0], 50), unit([0, 0, 1, 0], 10)])
    labels = np.array([0] * 100 + [1] * 10)
    out = narratives._split_dominant(vecs, labels, "dense")
    # the mega-cluster is gone: its 100 posts now carry >= 2 distinct labels
    mega = set(out[:100].tolist())
    assert len(mega) >= 2
    # the small separate cluster is untouched
    assert set(out[100:].tolist()) == {1}

    # one genuine story (all vectors aligned) is NOT split
    one = np.vstack([unit([1, 0, 0, 0], 100), unit([0, 0, 1, 0], 10)])
    lab2 = np.array([0] * 100 + [1] * 10)
    out2 = narratives._split_dominant(one, lab2, "dense")
    assert set(out2[:100].tolist()) == {0}  # left whole
