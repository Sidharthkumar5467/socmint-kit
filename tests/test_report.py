"""Standalone report: self-contained, embeds data, escapes </script> in post text."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import make_post  # tests/ is on sys.path (no package __init__)

from socmint.report import WEB_DIR, build_report, embed_json
from socmint.runstore import write_json, write_jsonl
from socmint.schema import RunMeta

# NB: no U+2028 in post text: runstore.read_jsonl uses str.splitlines(), which splits
# on it (reported as a foundation request). embed_json's U+2028 escaping is tested directly.
EVIL = 'bad </script><script>alert(1)</script> <!-- x --> end'


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "r"
    run.mkdir()
    write_json(run / "run.json", RunMeta(run_id="r1", topic="Test <b>topic</b>", start="2026-11-01",
                                         end="2026-11-07", sources=["x"],
                                         created_at="2026-11-08T00:00:00Z", status="complete"))
    write_jsonl(run / "posts.jsonl", [make_post("1", EVIL), make_post("2", "مرحبا 你好")])
    for name in ("narratives", "clusters", "origin", "claims", "source_status"):
        write_json(run / f"{name}.json", [])
    return run


def _embedded(doc: str) -> dict:
    m = re.search(r'<script type="application/json" id="socmint-data">(.*?)</script>', doc, re.S)
    assert m, "embedded data element missing"
    return json.loads(m.group(1))


def test_report_is_self_contained(tmp_path):
    out = build_report(_run(tmp_path))
    assert out.name == "report.html" and out.is_file()
    doc = out.read_text(encoding="utf-8")
    assert not re.search(r"<script[^>]*\bsrc\s*=", doc, re.I), "no external scripts"
    assert not re.search(r"<link[^>]*\bhref\s*=", doc, re.I), "no external stylesheets"
    assert not re.search(r"@import", doc), "no CSS imports"
    # app.css and app.js are inlined
    assert (WEB_DIR / "app.css").read_text(encoding="utf-8").strip()[:40] in doc
    assert "socmint-data" in doc and "function safeUrl" in doc


def test_report_embeds_data_and_escapes(tmp_path):
    doc = build_report(_run(tmp_path)).read_text(encoding="utf-8")
    # Exactly two script elements (data, app); the evil text never closes one.
    assert doc.lower().count("</script>") == 2
    assert "<!-- x" not in doc
    data = _embedded(doc)
    texts = [p["text"] for p in data["posts"]]
    assert EVIL in texts and "مرحبا 你好" in texts  # round-trips exactly, unicode kept
    assert data["run"]["topic"] == "Test <b>topic</b>"
    # Title is HTML-escaped
    assert "<title>SOCMINT report: Test &lt;b&gt;topic&lt;/b&gt;" in doc


def test_embed_json_roundtrip():
    payload = {"t": EVIL + "\u2028\u2029", "n": [1, None, "</SCRIPT>"]}
    s = embed_json(payload)
    assert "</" not in s and "<!--" not in s and "\u2028" not in s
    assert json.loads(s) == payload


def test_report_handles_empty_run_dir(tmp_path):
    run = tmp_path / "empty"
    run.mkdir()
    data = _embedded(build_report(run).read_text(encoding="utf-8"))
    assert data["posts"] == [] and data["run"] == {}


def test_web_assets_present_and_safe():
    index = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert '/static/app.css' in index and '/static/app.js' in index
    assert not re.search(r'(src|href)="https?://', index), "no CDN references"
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in js.replace("innerHTML is never used", "")
    assert "</script" not in js.lower()
    assert not re.search(r"https?://(?!www\.w3\.org/2000/svg)", js), "no external URLs in app.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_js_syntax():
    res = subprocess.run(["node", "--check", str(WEB_DIR / "app.js")], capture_output=True,
                         text=True, timeout=60)
    assert res.returncode == 0, res.stderr


def _js_helper(name: str) -> str:
    """Return the source of a marked helper function in app.js (between BEGIN/END markers)."""
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    m = re.search(rf"// BEGIN {name}\n(.*?)// END {name}", js, re.S)
    assert m, f"marker for {name} missing in app.js"
    return m.group(1)


def _run_node(src: str) -> str:
    res = subprocess.run(["node", "-e", src], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return res.stdout.strip()


def test_report_button_opens_inline():
    # The server only sends Content-Disposition: inline for report.html when ?inline=true,
    # otherwise the "open" button downloads the file into a blank tab.
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    m = re.search(r"href: `/api/runs/\$\{enc\(ctx\.id\)\}/files/report\.html([^`]*)`, target: '_blank'", js)
    assert m and m.group(1) == "?inline=true"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_platform_seq_falls_back_to_first_seen():
    src = _js_helper("platformSeq") + """
    const out = [
      platformSeq({platform_sequence: [], platform_first_seen: {
        youtube: '2026-03-02T00:00:00Z', x: '2026-03-01T00:00:00Z', telegram: '2026-03-03T00:00:00Z'}}),
      platformSeq({platform_sequence: ['telegram', 'x'], platform_first_seen: {x: '2026-03-01T00:00:00Z'}}),
      platformSeq({}),
      platformSeq({platform_first_seen: {b: 'bad', a: '2026-03-01T00:00:00Z'}}),
    ];
    console.log(JSON.stringify(out));
    """
    assert json.loads(_run_node(src)) == [
        ["x", "youtube", "telegram"], ["telegram", "x"], [], ["a", "b"]]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_running_bundles_are_not_final():
    src = _js_helper("isFinalBundle") + """
    console.log(JSON.stringify([
      isFinalBundle({run: {status: 'running'}}), isFinalBundle({run: {}}), isFinalBundle(null),
      isFinalBundle({run: {status: 'complete'}}), isFinalBundle({run: {status: 'failed'}})]));
    """
    assert json.loads(_run_node(src)) == [False, False, False, True, True]


def test_get_run_does_not_cache_partial_bundles():
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    body = re.search(r"async function getRun\(id\) \{(.*?)\n  \}\n", js, re.S).group(1)
    assert "isFinalBundle(bundle)" in body and "isFinalBundle(cached)" in body
    runs_page = re.search(r"async function pageRuns\(main, seq\) \{(.*?)\n  \}\n", js, re.S).group(1)
    assert "S.runs = {}" in runs_page


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_match_label_shown_only_when_recorded():
    src = _js_helper("matchLabel") + """
    console.log(JSON.stringify([
      matchLabel({options: {match: 'phrase'}}), matchLabel({options: {match: 'words'}}),
      matchLabel({options: {limit: 500}}), matchLabel({}), matchLabel(null)]));
    """
    assert json.loads(_run_node(src)) == ["exact phrase", "all words", "", "", ""]


def test_report_embeds_run_match_option(tmp_path):
    run = _run(tmp_path)
    write_json(run / "run.json", RunMeta(run_id="r1", topic="bank of america",
                                         start="2026-11-13", end="2026-11-19", sources=["x"],
                                         created_at="2026-11-19T00:00:00Z", status="complete",
                                         options={"limit": 300, "match": "words"}))
    data = _embedded(build_report(run).read_text(encoding="utf-8"))
    assert data["run"]["options"]["match"] == "words"
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    header = re.search(r"function resultsHeader\(ctx\) \{(.*?)\n  \}\n", js, re.S).group(1)
    assert "matchLabel(r)" in header  # the shared header (GUI + report) shows it


# ---------------------------------------------------------------- branding & theme
def test_brand_logo_template_matches_asset():
    logo = (WEB_DIR / "brand" / "spidersilk-logo.svg").read_text(encoding="utf-8").strip()
    assert logo.startswith("<svg") and "currentColor" in logo and "#E40E53" in logo
    assert not re.search(r"<script|\bon\w+\s*=|href|url\(", logo, re.I), "logo must be inert"
    index = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<template id="brand-logo">(.*?)</template>', index, re.S)
    assert m and m.group(1).strip() == logo, "index.html logo template out of sync with the asset"
    assert '/static/brand/favicon.svg' in index


def test_report_carries_logo_and_embedded_fonts(tmp_path):
    doc = build_report(_run(tmp_path)).read_text(encoding="utf-8")
    assert '<template id="brand-logo"><svg' in doc
    # Fonts are embedded as data: URIs (same type as the GUI, offline); no font file URLs.
    assert 'url("fonts/' not in doc, "report must not reference font files"
    assert doc.count('url("data:font/woff2;base64,') == 3
    for fam in ('"Jost"', '"Inter"', '"JetBrains Mono"'):
        assert f"font-family: {fam}" in doc
    assert "font-src data:" in doc
    assert doc.lower().count("</script>") == 2


def test_embed_fonts_missing_file_degrades(tmp_path):
    from socmint.report import _embed_fonts

    out = _embed_fonts('src: url("fonts/nope.woff2") format("woff2");')
    assert out == 'src: url("data:font/woff2;base64,") format("woff2");'


def test_bundled_fonts_present_with_licenses():
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")
    for name in re.findall(r'url\("fonts/([^"]+)"\)', css):
        assert (WEB_DIR / "fonts" / name).is_file(), name
    for lic in ("OFL-jost.txt", "OFL-inter.txt", "OFL-jetbrainsmono.txt"):
        assert (WEB_DIR / "fonts" / lic).is_file()
    assert not re.search(r"url\((?![\"']?fonts/)", css), "only local font URLs in app.css"


def test_theme_tokens_defined_for_light_and_dark():
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")
    light = re.search(r":root \{(.*?)\n\}", css, re.S).group(1)
    dark = re.search(r':root\[data-theme="dark"\] \{(.*?)\n\}', css, re.S).group(1)
    system = re.search(r':root:not\(\[data-theme="light"\]\) \{(.*?)\n  \}', css, re.S).group(1)
    names = set(re.findall(r"(--[\w-]+):", light))
    for required in ("--ok", "--warn", "--skip", "--bad", "--direct", "--inferred", "--p-linkedin",
                     "--p-linkedin_ads", "--p-x", "--p-telegram", "--p-youtube"):
        assert required in names
        assert f"{required}:" in dark and f"{required}:" in system, required
    color = lambda block, k: re.search(rf"{k}:\s*([^;]+);", block).group(1).strip()  # noqa: E731
    for block in (light, dark):
        assert color(block, "--p-linkedin") != color(block, "--p-linkedin_ads")
        assert color(block, "--direct") != color(block, "--inferred")


def test_theme_storage_is_guarded():
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    for fn in ("getTheme", "setTheme"):
        body = re.search(rf"function {fn}\(.*?\n  \}}\n", js, re.S).group(0)
        assert "try" in body and "catch" in body, fn
