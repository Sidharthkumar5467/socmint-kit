"""Standalone HTML report (docs/SPEC.md §8).

`build_report(run_dir)` writes run_dir/report.html: ONE self-contained file that opens
offline from file://. It inlines the same app.css and app.js the local GUI uses and
embeds the run data (runstore.load_run) as JSON in
<script type="application/json" id="socmint-data">. The app detects that element and
renders the results views only (no settings, no new run, no server API).

Because the GUI and the report share one renderer and one data source (the run files),
the report cannot drift from what the analyst saw in the GUI.
"""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path
from typing import Any

from socmint.legacy import upgrade_bundle
from socmint.runstore import FILES, load_run

WEB_DIR = Path(__file__).resolve().parent / "web"

# The report allows only inline script/style (its own) and images from http(s)/data.
# No network fetches, no external scripts or stylesheets.
CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src https: http: data:; font-src data:; base-uri 'none'; form-action 'none'"
)


def embed_json(data: Any) -> str:
    """Serialise data for safe embedding inside a <script> element.

    "</" -> "<\\/" prevents any "</script>" in post text from closing the element
    ("\\/" is a valid JSON escape for "/", so the JSON still parses to the same value).
    "<!--" is also neutralised because it can switch the HTML parser into an escaped
    state inside script data. U+2028/2029 are escaped for older JS parsers.
    """
    s = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    s = s.replace("</", "<\\/").replace("<!--", "\\u003c!--")
    return s.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _inline_safe(code: str, tag: str) -> str:
    """Asset text inlined into <script>/<style> must not contain its own closing tag."""
    return code.replace(f"</{tag}", f"<\\/{tag}")


# The GUI's bundled web fonts (Jost, Inter, JetBrains Mono, ~106 KB of woff2) are
# embedded in the report as base64 data: URIs so the report renders in the same type as
# the GUI on any machine, offline. CSP allows fonts only from data: (font-src data:).
_FONT_URL = re.compile(r'url\("fonts/([A-Za-z0-9._-]+\.woff2)"\)')


def _embed_fonts(css: str) -> str:
    """Replace each url("fonts/x.woff2") with a data: URI of that bundled file.

    A font file that cannot be read drops to the system font stack declared in app.css
    (its @font-face then has no usable src, which browsers ignore)."""

    def sub(m: re.Match[str]) -> str:
        path = WEB_DIR / "fonts" / m.group(1)
        try:
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            return 'url("data:font/woff2;base64,")'
        return f'url("data:font/woff2;base64,{b64}")'

    return _FONT_URL.sub(sub, css)


LOGO_SVG = WEB_DIR / "brand" / "spidersilk-logo.svg"


def _logo_template() -> str:
    """The SpiderSilk wordmark as an inert <template> the app clones (no network)."""
    try:
        svg = LOGO_SVG.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not svg.startswith("<svg") or re.search(r"<script|\bon\w+\s*=|href", svg, re.I):
        return ""
    return f'<template id="brand-logo">{svg}</template>\n'


def render_report(data: dict[str, Any]) -> str:
    css = _embed_fonts((WEB_DIR / "app.css").read_text(encoding="utf-8"))
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    run = data.get("run") or {}
    topic = str(run.get("topic") or "run")
    window = f"{str(run.get('start') or '')[:10]} to {str(run.get('end') or '')[:10]}"
    title = html.escape(f"SOCMINT report: {topic} ({window})")
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="referrer" content="no-referrer">\n'
        f'<meta http-equiv="Content-Security-Policy" content="{CSP}">\n'
        f"<title>{title}</title>\n"
        f"<style>\n{_inline_safe(css, 'style')}\n</style>\n"
        "</head>\n<body>\n"
        "<noscript>This report needs JavaScript to render. The raw data is embedded in this "
        "file and also available as JSON/CSV next to it in the run folder.</noscript>\n"
        '<div id="app"></div>\n'
        f"{_logo_template()}"
        f'<script type="application/json" id="socmint-data">{embed_json(data)}</script>\n'
        f"<script>\n{_inline_safe(js, 'script')}\n</script>\n"
        "</body>\n</html>\n"
    )


def build_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / FILES["report"]
    tmp = out.with_suffix(".html.tmp")
    tmp.write_text(render_report(upgrade_bundle(load_run(run_dir))), encoding="utf-8")
    tmp.replace(out)
    return out
