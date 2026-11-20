"""Results views v2 (socmint/web/app.js): the pure helpers behind Summary, ranked
narratives, coordination, origin & spread and claims. The helpers sit between
`// BEGIN viewHelpers` and `// END viewHelpers` and use no DOM, so they run under node.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB_DIR = Path(__file__).resolve().parent.parent / "socmint" / "web"
JS = (WEB_DIR / "app.js").read_text(encoding="utf-8")

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _helpers() -> str:
    m = re.search(r"// BEGIN viewHelpers\n(.*?)// END viewHelpers", JS, re.S)
    assert m, "viewHelpers block missing in app.js"
    return m.group(1)


def run_js(body: str):
    res = subprocess.run(["node", "-e", _helpers() + "\nconsole.log(JSON.stringify((() => {" + body + "})()));"],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout.strip())


def test_helpers_block_has_no_dom_access():
    src = _helpers()
    assert "document." not in src and "window." not in src
    assert not re.search(r"(?<![\w.])[hs]\(", src), "no DOM builders inside viewHelpers"


@needs_node
def test_dates_and_numbers():
    out = run_js("""return [
      dayList('2026-11-13', '2026-11-15'), dayList('2026-11-15', '2026-11-13'), dayList('', 'x'),
      shortDay('2026-11-16'), shortDay('2024-10-02', '2026'), shortDay('2026-01-05T10:00:00Z', '2026'), shortDay(null),
      shortTs('2026-11-16T23:30:59Z'), shortTs('bad'),
      windowLabel('2026-11-13', '2026-11-19'), windowLabel('2026-10-28', '2026-11-03'),
      windowLabel('2025-12-28', '2026-01-03'),
      compactNum(950), compactNum(3456), compactNum(209123), compactNum(999999), compactNum(1250000), compactNum(null),
      gapLabel(900), gapLabel(79200), gapLabel(5 * 86400), gapLabel(-1),
      dayOf('2026-11-13T23:59:59Z'), dayOf('2026-11-13T23:30:00-02:00'), dayOf(null),
      joinNames(['A']), joinNames(['A', 'B']), joinNames(['A', 'B', 'C']),
      firstLine('\\n  Title line \\nbody', 140), truncate('abcdef', 4)];""")
    assert out == [
        ["2026-11-13", "2026-11-14", "2026-11-15"], [], [],
        "16 Nov", "2 Oct 2024", "5 Jan", "–",
        "16 Nov · 23:30", "–",
        "13–19 Nov 2026", "28 Oct – 3 Nov 2026", "28 Dec 2025 – 3 Jan 2026",
        "950", "3.5k", "209k", "1M", "1.3M", "–",
        "+15 min", "+22 h", "+5 d", "",
        "2026-11-13", "2026-11-14", None,
        "A", "A and B", "A, B and C",
        "Title line", "abc…",
    ]


@needs_node
def test_rationale_split_and_bottom_line_fallback():
    out = run_js("""
      const r = "Men's Journal [4] lists U.S. Bank customers. AP [3] disagrees. " +
        "[Method: extraction=llm; verdict=llm.] [3 narrative post(s) without a verified time.]";
      return [splitRationale(r), firstSentence(r), claimBottomLine({rationale: r}),
        claimBottomLine({rationale: r, bottom_line: 'Own line.'}),
        claimMethodNotes({rationale: r}), claimMethodNotes({rationale: r, method_notes: 'given'}),
        firstSentence('No end punctuation here'), firstSentence('x'.repeat(300)).length,
        claimBottomLine({})];""")
    split, first, bl, own, notes, given, noend, capped, empty = out
    assert split["body"] == "Men's Journal [4] lists U.S. Bank customers. AP [3] disagrees."
    assert split["notes"] == ["Method: extraction=llm; verdict=llm.", "3 narrative post(s) without a verified time."]
    assert first == "Men's Journal [4] lists U.S. Bank customers."  # "U.S." does not end it
    assert bl == first and own == "Own line."
    assert notes.startswith("Method: extraction=llm") and given == "given"
    assert noend == "No end punctuation here" and capped == 220 and empty == ""


@needs_node
def test_evidence_order_and_unrelated_handling():
    out = run_js("""
      const ev = [
        {url: 'c1', supports: 'context'}, {url: 'old', supports: 'context', flag: 'old_event'},
        {url: 'x1', supports: 'contradicts'}, {url: 'u1', supports: 'supports', flag: 'unrelated'},
        {url: 's1', supports: 'supports'}, {url: 'c2', supports: 'context'}];
      const a = arrangeEvidence(ev, 3);
      const b = arrangeEvidence([{url: 'u', supports: 'context', flag: 'unrelated'}], 3);
      return [a.shown.map((e) => e.url), a.rest.map((e) => e.url), a.allUnrelated,
        b.shown.length, b.rest.length, b.allUnrelated, arrangeEvidence(null).allUnrelated];""")
    # supports first, then contradicts, then context (old events after plain context);
    # an unrelated item never shows up front, even when it claims to support.
    assert out == [["s1", "x1", "c1"], ["c2", "old", "u1"], False, 0, 1, True, False]


@needs_node
def test_signal_key_parses_rule_built_details():
    out = run_js("""return [
      signalKey({type: 'near_duplicate_text', evidence_post_ids: [1, 2],
        detail: '69 posts by 69 accounts with near-identical text (shingle Jaccard >= 0.8, mean 1.00)'}),
      signalKey({type: 'near_simultaneous_posting', evidence_post_ids: [1],
        detail: '2346 same-content post pairs (2346 near-duplicate text, 0 same link) by 69 accounts posted within 10 min of each other (median lag 15 s; platform timestamps)'}),
      signalKey({type: 'similar_creation_dates', evidence_post_ids: [1],
        detail: '53 of 69 accounts with a platform-reported creation date were created within 3 days of each other (max gap 2.1 days)'}),
      signalKey({type: 'reused_image', evidence_post_ids: [1, 2, 3], detail: 'same image hash'}),
      signalKey({type: 'near_duplicate_text', evidence_post_ids: [], detail: 'unexpected wording'})];""")
    assert out[0] == {"label": "Same text", "key": "69 posts", "sub": "by 69 accounts, near-identical (similarity 1.00)"}
    assert out[1] == {"label": "Posted together", "key": "15 s", "sub": "median gap · 2,346 pairs within 10 min"}
    assert out[2] == {"label": "Created together", "key": "53 / 69", "sub": "accounts created within 2.1 days"}
    assert out[3] == {"label": "Same image", "key": "3 posts", "sub": "same image hash"}
    assert out[4]["key"] == "–" and out[4]["sub"] == "unexpected wording"  # falls back, never invents


@needs_node
def test_rank_narratives_with_and_without_brief():
    out = run_js("""
      const mk = (id, n, rel) => ({id, relevance: rel, member_post_ids: Array.from({length: n}, (_, i) => id + i)});
      const ns = [mk('N1', 50), mk('N2', 5), mk('N3', 40, 'off_topic'), mk('N4', 12, 'threat')]
        .concat(Array.from({length: 12}, (_, i) => mk('M' + i, 20 - i)));
      const fb = rankNarratives(ns, {});
      const br = rankNarratives(ns, {narrative_ranks: [
        {narrative_id: 'N2', rank: 1, main: true, tags: ['threat']}, {narrative_id: 'GONE', rank: 2, main: true},
        {narrative_id: 'N1', rank: 3, main: false, tags: ['off_topic']}]});
      return [fb.map((r) => [r.narrative_id, r.rank, r.main, r.group]).slice(0, 5),
        fb.filter((r) => r.main).length, fb.find((r) => r.narrative_id === 'N2').main,
        br.slice(0, 3).map((r) => [r.narrative_id, r.rank, r.main, r.group, r.fromBrief]),
        br.length, br.filter((r) => r.main).length];""")
    top5, n_main, small_main, brief3, total, brief_main = out
    # by size; off-topic N3 is never main; at most 10 main, each with >= 10 posts
    assert top5 == [["N1", 1, True, ""], ["N3", 2, False, "off_topic"], ["M0", 3, True, ""],
                    ["M1", 4, True, ""], ["M2", 5, True, ""]]
    assert n_main == 10 and small_main is False
    # brief order wins, unknown ids are dropped, relevance comes from the narrative first
    assert brief3 == [["N2", 1, True, "threat", True], ["N1", 2, False, "off_topic", True],
                      ["N3", 3, False, "off_topic", False]]
    assert total == 16 and brief_main == 1  # with a brief, only the brief marks main


@needs_node
def test_coverage_marks_days_not_reached_for_sources_that_stopped_early():
    out = run_js("""
      const days = dayList('2026-11-13', '2026-11-19');
      const p = (source, day, kind) => ({source, kind: kind || 'post', timestamp_verified: true, published_at: day + 'T10:00:00Z'});
      const posts = days.map((d) => p('x', d)).concat([p('telegram', '2026-11-19'), p('linkedin', '2026-11-17'),
        p('linkedin', '2026-11-19'), p('linkedin_ads', '2026-11-18', 'ad'),
        {source: 'telegram', kind: 'post', timestamp_verified: false, published_at: '2026-11-13T00:00:00Z'}]);
      const status = [{source: 'x', state: 'ok'}, {source: 'telegram', state: 'ok'}, {source: 'telegram', state: 'partial'},
        {source: 'linkedin', state: 'partial'}, {source: 'linkedin_ads', state: 'partial'},
        {source: 'youtube', state: 'skipped_no_credentials'}, {source: 'gone', state: 'failed'}];
      return coverageOf(status, posts, days);""")
    notes = {n["source"]: (n["state"], n["days_with_posts"], n["note"]) for n in out["notes"]}
    assert notes["x"] == ("ok", 7, "all 7 days covered")
    assert notes["telegram"] == ("partial", 1, "newest day only; older days not reached")
    assert notes["linkedin"] == ("partial", 2, "from 17 Nov only; older days not reached")
    assert notes["youtube"][2] == "not collected (skipped)"
    assert notes["gone"][2] == "failed; no posts collected"
    # ads carry run dates, not a newest-first crawl, so they never mark days "not reached"
    assert out["from"] == {"telegram": "2026-11-19", "linkedin": "2026-11-17"}


@needs_node
def test_hubs_relations_clusters_and_top_posts():
    out = run_js("""
      const edges = [];
      for (let i = 0; i < 5; i++) edges.push({from: 'yt:v', to: 'yt:c' + i, type: 'direct', relation: 'reply'});
      edges.push({from: 'yt:v', to: 'x:9', type: 'inferred'}, {from: 'x:1', to: 'x:2', type: 'direct', relation: 'quote'});
      const hubs = hubsOf({edges});
      const given = hubsOf({edges, hubs: [{post_id: 'h', direct_out: 1, relations: {}}]});
      const narrOf = new Map([['p1', ['N1']], ['p2', ['N1', 'N2']], ['p3', ['N2']], ['p4', ['N3']]]);
      const cl = [{id: 'C1', supporting_post_ids: ['p1', 'p2', 'p3', 'p4']}, {id: 'C2', supporting_post_ids: []}];
      const big = (id, n, s) => ({id, member_author_ids: Array(n).fill('a'), signals: Array(s).fill({})});
      return [hubs, given.map((x) => x.post_id),
        relationSummary({reply: 100}, 'youtube'), relationSummary({reply: 1, link: 3}, 'x'), relationSummary({}),
        clusterNarratives(cl, narrOf, {}), clusterNarratives(cl, narrOf, {cluster_narratives: {C1: ['N9']}}).C1,
        mainClusters([big('A', 3, 2), big('B', 9, 2), big('C', 5, 3)], {}),
        mainClusters([big('A', 3, 2), big('B', 2, 3)], {}),
        mainClusters([big('A', 3, 2)], {main_cluster_ids: ['A', 'ZZ']}),
        topPosts([{post_id: 'a', metrics: {likes: 50}}, {post_id: 'b', metrics: {views: 10}},
          {post_id: 'c', metrics: {views: 900}}, null], 2).map((p) => p.post_id)];""")
    hubs, given, yt, x, none, cn, cn_brief, main_a, main_b, main_c, top = out
    assert hubs == [{"post_id": "yt:v", "direct_out": 5, "relations": {"reply": 5}, "inferred_out": 1}]
    assert given == ["h"]
    assert (yt, x, none) == ("100 comments", "3 links, 1 reply", "")
    assert cn == {"C1": ["N1", "N2"], "C2": []} and cn_brief == ["N9"]
    assert main_a == ["B", "C"] and main_b == ["A", "B"] and main_c == ["A"]
    assert top == ["c", "b"]


@needs_node
def test_peak_and_group():
    out = run_js("""return [peakOf({'2026-11-14': 3, '2026-11-13': 3, '2026-11-15': 1}), peakOf({}),
      narrGroup({relevance: 'related'}, ['threat']), narrGroup({}, ['threat']), narrGroup({}, ['off_topic']), narrGroup(null)];""")
    assert out == [{"day": "2026-11-13", "count": 3}, {"day": None, "count": 0}, "related", "threat", "off_topic", ""]


def test_routes_and_sections_wired():
    assert "['summary', 'Summary']" in JS
    assert "tab === 'narrative' && sub" in JS  # #/run/<id>/narrative/<nid> detail page
    assert ": 'summary');" in JS  # a run opens on Summary
    for fn in ("tabSummary", "pageNarrative", "tabNarratives", "tabCoordination", "tabOrigin", "tabClaims"):
        assert f"function {fn}(" in JS, fn
    # "first observed in collected data" wording is kept (CLAUDE.md section 3)
    assert "First observed in collected data" in JS and "not necessarily" in JS


def test_type_scale_tokens_and_dark_tokens():
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")
    light = re.search(r":root \{(.*?)\n\}", css, re.S).group(1)
    dark = re.search(r':root\[data-theme="dark"\] \{(.*?)\n\}', css, re.S).group(1)
    system = re.search(r':root:not\(\[data-theme="light"\]\) \{(.*?)\n  \}', css, re.S).group(1)
    for tok in ("--fs-xs", "--fs-base", "--fs-xl", "--sp-2", "--sp-4"):
        assert f"{tok}:" in light, tok
    for tok in ("--on-verdict", "--bar", "--bar-lo", "--hatch"):
        assert f"{tok}:" in light and f"{tok}:" in dark and f"{tok}:" in system, tok
    assert "font-variant-numeric: tabular-nums" in css


def test_reanalyse_button_and_derived_note_wired():
    # GUI parity with `socmint reanalyze` (CLAUDE.md section 4): button -> endpoint ->
    # progress page; never offered in the standalone report, replay or for the sample
    assert "function canReanalyze(" in JS and "function reanalyzeButton(" in JS
    assert "/reanalyze`, { method: 'POST'" in JS
    body = re.search(r"function canReanalyze\(r\) \{(.*?)\n  \}", JS, re.S).group(1)
    assert "!STATIC" in body and "read_only" in body and "path_kind === 'run'" in body
    # older runs upgraded on load say so on the Summary page
    assert "derived_on_load" in JS
    # a claims finding never shows its narrative's platform chips as if they were claims'
    assert "f.kind !== 'claims'" in JS and "noCoord" in JS


@needs_node
def test_gap_sources_only_narrative_platforms():
    out = run_js("""
      const cf = {telegram: '2026-11-19', linkedin: '2026-11-19'};
      return [gapSources('2026-11-14', cf, {youtube: 5}),
              gapSources('2026-11-14', cf, {youtube: 5, telegram: 2, linkedin: 0}),
              gapSources('2026-11-19', cf, {telegram: 2}),
              gapSources('2026-11-14', {}, {telegram: 2})];
    """)
    # N271-like: YouTube only (fully covered) -> no hatching on any day
    assert out == [[], ["telegram"], [], []]


@needs_node
def test_loose_first_platforms_and_strong_seq():
    out = run_js("""
      const tr = {platform_first_seen: {youtube: '2026-11-13T08:00:00Z', x: '2026-11-13T10:00:00Z',
                                        telegram: '2026-11-19T01:00:00Z'},
                  platform_first_seen_strong: {x: '2026-11-13T10:00:00Z', youtube: '2026-11-15T09:00:00Z'}};
      return [looseFirstPlatforms(tr), strongSeq(tr), looseFirstPlatforms({platform_first_seen: {x: 'a'}}),
              strongSeq({})];
    """)
    assert out == [["telegram", "youtube"], ["x", "youtube"], [], []]


@needs_node
def test_hubs_with_membership_flags_parent_posts():
    out = run_js("""
      return hubsWithMembership([{post_id: 'youtube:v1', direct_out: 4}, {post_id: 'x:1', direct_out: 3}],
                                ['x:1', 'x:2']).map((h) => [h.post_id, h.member, h.direct_out]);
    """)
    assert out == [["youtube:v1", False, 4], ["x:1", True, 3]]


@needs_node
def test_compact_lines_collapses_blank_lines():
    out = run_js("""
      return [compactLines('US BANKS GO DOWN\\n\\nChase, BofA out\\n\\n\\nmore', 2),
              compactLines('one line', 3), compactLines('a\\n\\nb', 3), compactLines('', 3),
              compactLines('abcdefghij', 3, 6)];
    """)
    assert out == ["US BANKS GO DOWN\nChase, BofA out…", "one line", "a\nb", "", "abcde…"]


def test_css_alignment_rules_present():
    css = (Path(__file__).resolve().parents[1] / "socmint" / "web" / "app.css").read_text(encoding="utf-8")
    assert ".view .card.table-wrap { padding: 0; }" in css
    assert css.index(".view .card.table-wrap { padding: 0; }") > css.index(".view .card { padding: var(--sp-5); }")
    assert "grid-template-columns: 104px minmax(0, 1fr) max-content" in css
    assert ".tbl td { vertical-align: top; }" in css
    assert ".section-title h1 { font-size: var(--fs-2xl);" in css
    assert "summary:focus-visible" in css
    assert ".run-header .eyebrow .rid { text-transform: none;" in css
    assert ".ev-unrelated { opacity" not in css


@needs_node
def test_severity_helpers_and_fallback_rank():
    out = run_js("""
      const mk = (id, n, sev, rel) => ({id, severity: sev, relevance: rel,
        member_post_ids: Array.from({length: n}, (_, i) => id + i)});
      const ns = [mk('BIG', 40, 'low'), mk('CRIT', 6, 'critical'), mk('MED', 20, 'medium'),
                  mk('OFF', 50, 'high', 'off_topic'), mk('OLD', 30), mk('H4', 4, 'high')];
      const fb = rankNarratives(ns, {});
      const br = rankNarratives(ns, {narrative_ranks: [{narrative_id: 'OLD', rank: 1, main: true, severity: 'medium'}]});
      return [fb.map((r) => [r.narrative_id, r.main, r.severity]),
        narrSeverity({severity: 'bogus'}), narrSeverity({}, {severity: 'high'}),
        narrSeverity({severity: 'critical', relevance: 'off_topic'}), severityIndex(null),
        br[0].severity];""")
    ranked, bogus, from_rank, off, idx_none, brief_sev = out
    # severity first (off-topic never rated), then size; without a brief there is no
    # coordination/claim context, so main needs >= 10 posts at any severity
    assert ranked == [["CRIT", False, "critical"], ["H4", False, "high"], ["MED", True, "medium"],
                      ["BIG", True, "low"], ["OFF", False, None], ["OLD", True, None]]
    assert (bogus, from_rank, off, idx_none, brief_sev) == (None, "high", None, 4, "medium")


def test_severity_ui_wired_and_tokens_in_every_theme():
    for needle in ("function sevBadge(", "function sevReasonLine(", "sevReasonLine(n, r.severity)",
                   "sevReasonLine(n, narrSeverity(n, r))", "SEVERITY_LABEL[r.severity]} · ",
                   "Sorted by: ${hasSev ? 'Severity'", "out.push(sevBadge(sev"):
        assert needle in JS, needle
    # the old relevance pills are gone; severity pills replace them
    assert "'Threat-relevant']" not in JS and "'Other on-topic'" not in JS
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")
    light = re.search(r":root \{(.*?)\n\}", css, re.S).group(1)
    dark = re.search(r':root\[data-theme="dark"\] \{(.*?)\n\}', css, re.S).group(1)
    system = re.search(r':root:not\(\[data-theme="light"\]\) \{(.*?)\n  \}', css, re.S).group(1)
    for lvl in ("critical", "high", "medium", "low"):
        for part in ("bg", "fg"):
            tok = f"--sev-{lvl}-{part}:"
            assert tok in light and tok in dark and tok in system, tok


def _lum(hexc: str) -> float:
    c = [int(hexc.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def test_severity_colours_aa_and_distinct_lightness():
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")
    blocks = {"light": re.search(r":root \{(.*?)\n\}", css, re.S).group(1),
              "dark": re.search(r':root\[data-theme="dark"\] \{(.*?)\n\}', css, re.S).group(1)}
    for name, block in blocks.items():
        lums = []
        for lvl in ("critical", "high", "medium", "low"):
            bg = re.search(rf"--sev-{lvl}-bg: (#[0-9a-f]{{6}})", block).group(1)
            fg = re.search(rf"--sev-{lvl}-fg: (#[0-9a-f]{{6}})", block).group(1)
            a, b = sorted((_lum(bg), _lum(fg)), reverse=True)
            assert (a + 0.05) / (b + 0.05) >= 4.5, (name, lvl)
            lums.append(_lum(bg))
        # critical / high / medium fills differ in lightness, not hue alone
        assert len({round(x, 2) for x in lums[:3]}) == 3, (name, lums)


def test_severity_pill_dots_visible_on_surface():
    # The medium/low badge fills are pale (light) or dark (dark) by design; as 10px pill
    # dots they vanished (found in the 2026-11-19 screenshots). Dots have their own
    # tokens, in every theme block, at >= 3:1 (non-text contrast) against --surface.
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")
    blocks = {"light": re.search(r":root \{(.*?)\n\}", css, re.S).group(1),
              "dark": re.search(r':root\[data-theme="dark"\] \{(.*?)\n\}', css, re.S).group(1),
              "system": re.search(r':root:not\(\[data-theme="light"\]\) \{(.*?)\n  \}',
                                  css, re.S).group(1)}
    for name, block in blocks.items():
        surface = re.search(r"--surface: (#[0-9a-f]{6})", block).group(1)
        for lvl in ("critical", "high", "medium", "low"):
            dot = re.search(rf"--sev-{lvl}-dot: (#[0-9a-f]{{6}})", block).group(1)
            a, b = sorted((_lum(dot), _lum(surface)), reverse=True)
            assert (a + 0.05) / (b + 0.05) >= 3.0, (name, lvl)
            assert f".sev-dot.sev-{lvl} {{ background: var(--sev-{lvl}-dot)" in css


def test_severity_dots_visible_on_selected_pill():
    """A selected pill is --text with --bg text. Each dot gets a 2px ring in the pill's
    text colour, so: ring vs pill >= 3:1, and dot vs ring >= 3:1, in light and dark."""
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")
    assert ".pill.on { background: var(--text); color: var(--bg);" in css
    assert ".pill.on .sev-dot { box-shadow: 0 0 0 2px currentColor; }" in css
    blocks = {"light": re.search(r":root \{(.*?)\n\}", css, re.S).group(1),
              "dark": re.search(r':root\[data-theme="dark"\] \{(.*?)\n\}', css, re.S).group(1),
              "system": re.search(r':root:not\(\[data-theme="light"\]\) \{(.*?)\n  \}',
                                  css, re.S).group(1)}

    def ratio(a, b):
        x, y = sorted((_lum(a), _lum(b)), reverse=True)
        return (x + 0.05) / (y + 0.05)

    for name, block in blocks.items():
        pill = re.search(r"--text: (#[0-9a-f]{6})", block).group(1)
        ring = re.search(r"--bg: (#[0-9a-f]{6})", block).group(1)
        assert ratio(ring, pill) >= 3.0, name
        for lvl in ("critical", "high", "medium", "low"):
            dot = re.search(rf"--sev-{lvl}-dot: (#[0-9a-f]{{6}})", block).group(1)
            assert ratio(dot, ring) >= 3.0, (name, lvl, round(ratio(dot, ring), 2))


@needs_node
def test_window_preset_helpers():
    """New Run presets (owner decision 2026-11-20): lines under the choice and the run header."""
    now = "Date.parse('2026-11-20T09:30:15.400Z')"
    out = run_js(f"""return [
      WINDOW_PRESETS.map((x) => [x.id, x.days, x.limit]), DEFAULT_WINDOW,
      presetName(1), presetName(7), presetSpan(1, {now}),
      presetLines(1, 150, {now}), presetLines(3, 300, {now}), presetLines(null, 500, {now}),
      windowPresetLabel({{ options: {{ window_preset: '3d', limit: 300 }} }}),
      windowPresetLabel({{ options: {{ window_preset: 'custom', limit: 1200 }} }}),
      windowPresetLabel({{ options: {{ limit: 500, match: 'phrase' }} }}), windowPresetLabel(null),
      windowBounds({{ start: '2026-11-19', end: '2026-11-20', options: {{ window_preset: '1d',
        window_start_at: '2026-11-19T09:30:15Z', window_end_at: '2026-11-20T09:30:15Z' }} }}),
      windowBounds({{ start: '2026-11-13', end: '2026-11-19', options: {{}} }})];""")
    assert out[0] == [["1d", 1, 150], ["3d", 3, 300], ["7d", 7, 500]] and out[1] == "3d"
    assert out[2:4] == ["1 day", "7 days"]
    assert out[4] == {"start": "2026-11-19T09:30:15Z", "end": "2026-11-20T09:30:15Z"}
    one, three, custom = out[5], out[6], out[7]
    assert one[0] == "Last 24 hours: 19 Nov · 09:30 → 20 Nov · 09:30 UTC, up to 150 posts per source."
    assert "X and YouTube: the whole 24 hours." in one[1]
    assert three[0].startswith("Last 3 days: 17 Nov · 09:30 → 20 Nov · 09:30 UTC, up to 300")
    assert "X and YouTube: every day, searched day by day." in three[1]
    assert "Telegram and LinkedIn posts: most recent first" in three[1]
    assert custom[0] == "Whole UTC days from the start date to the end date, up to 500 posts per source."
    assert out[8] == "3-day window · up to 300 posts per source"
    assert out[9] == "Custom range · up to 1,200 posts per source"
    assert out[10] == "" and out[11] == ""  # runs saved before presets: no label
    assert out[12] == "19 Nov · 09:30 → 20 Nov · 09:30"
    assert out[13] == "13–19 Nov 2026"


@needs_node
def test_prune_trace_from_run_options():
    """Review 2026-11-20: the narrative page shows which posts left it (run.json options)."""
    out = run_js("""const o = { pruned_post_ids: { N3: ['x:1', 'yt:2', 'li:3'], N9: ['x:9'] },
      pruned_split_into: { N3: 'N40' }, narratives_mixed: { N12: { flagged: 6, shown: 12, reason: 'share' } } };
      return [pruneTrace(o, 'N3', ['x:1', 'yt:2']), pruneTrace(o, 'N40', ['x:1']),
        pruneTrace(o, 'N9', null), pruneTrace(o, 'N12', null), pruneTrace(o, 'N1', null),
        pruneTrace({}, 'N3', null)];""")
    assert out[0] == {"removed": ["x:1", "yt:2", "li:3"], "moved": ["x:1", "yt:2"],
                      "left": ["li:3"], "into": "N40", "from": None, "mixed": None}
    assert out[1]["from"] == "N3" and out[1]["removed"] == []
    assert out[2]["left"] == ["x:9"] and out[2]["into"] is None
    assert out[3]["mixed"] == {"flagged": 6, "shown": 12, "reason": "share"}
    assert out[4] is None and out[5] is None
