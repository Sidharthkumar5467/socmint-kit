/* SOCMINT single-page frontend.
 *
 * Vanilla ES2020, no build step, no external dependencies, works offline.
 * Two modes:
 *   - server mode: served by the local backend; talks to the /api/* routes.
 *   - static mode: inlined into a standalone report.html together with an embedded
 *     <script type="application/json" id="socmint-data"> block; renders results only.
 *
 * Safety rules applied throughout:
 *   - All user/platform content is inserted with textContent (via h()/text nodes).
 *     innerHTML is never used.
 *   - Only http(s) URLs are ever placed into href/src of external links and images.
 *
 * Evidence rules applied in the views:
 *   - Every finding links to its evidence posts (post ids open a detail panel).
 *   - Charts/timelines only use posts with a verified platform timestamp inside the
 *     window; unverified posts are shown but labelled and excluded from charts.
 *   - "Earliest observed in collected data" wording; never a claim of true origin.
 *   - Direct and inferred spread edges are drawn and tabulated distinctly.
 *   - Verdicts are shown as neutral assessments, never TRUE/FALSE.
 */
'use strict';
(function () {
  // ------------------------------------------------------------------ mode & state
  const DATA_EL = document.getElementById('socmint-data');
  const STATIC = !!DATA_EL;
  let EMBEDDED = null;
  if (STATIC) {
    try { EMBEDDED = JSON.parse(DATA_EL.textContent || '{}'); } catch (e) { EMBEDDED = null; }
  }

  const S = {
    meta: null,          // /api/meta response (server mode)
    runs: {},            // run id -> run bundle
    ctx: null,           // current results context (for the post modal)
    timer: null,         // progress poll timer
    filters: {},         // run id -> posts filter state
    renderSeq: 0,        // guards against out-of-order async renders
  };

  const TABS = [
    ['summary', 'Summary'], ['sources', 'Sources'], ['posts', 'Posts'], ['narratives', 'Narratives'],
    ['coordination', 'Coordination'], ['origin', 'Origin & spread'], ['claims', 'Claims'],
    ['exports', 'Exports'],
  ];

  // ------------------------------------------------------------------ theme
  // 'system' (no data-theme: follows prefers-color-scheme), 'light' or 'dark'. Stored per
  // browser as a convenience only; storage may be unavailable (private mode, file://).
  const THEME_KEY = 'socmint.theme';
  const THEMES = ['system', 'light', 'dark'];
  function getTheme() {
    try { const v = localStorage.getItem(THEME_KEY); return THEMES.includes(v) ? v : 'system'; } catch (_e) { return 'system'; }
  }
  function applyTheme(t) {
    if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
    else document.documentElement.removeAttribute('data-theme');
  }
  function setTheme(t) {
    applyTheme(t);
    try { localStorage.setItem(THEME_KEY, t); } catch (_e) { /* storage unavailable */ }
  }
  applyTheme(getTheme());

  // Feather-style stroke icons (24x24 viewBox): [element, attributes] pairs.
  const ICONS = {
    runs: [['rect', { x: 3, y: 3, width: 7, height: 7, rx: 1.5 }], ['rect', { x: 14, y: 3, width: 7, height: 7, rx: 1.5 }], ['rect', { x: 14, y: 14, width: 7, height: 7, rx: 1.5 }], ['rect', { x: 3, y: 14, width: 7, height: 7, rx: 1.5 }]],
    new: [['circle', { cx: 12, cy: 12, r: 10 }], ['line', { x1: 12, y1: 8, x2: 12, y2: 16 }], ['line', { x1: 8, y1: 12, x2: 16, y2: 12 }]],
    settings: [['line', { x1: 4, y1: 21, x2: 4, y2: 14 }], ['line', { x1: 4, y1: 10, x2: 4, y2: 3 }], ['line', { x1: 12, y1: 21, x2: 12, y2: 12 }], ['line', { x1: 12, y1: 8, x2: 12, y2: 3 }], ['line', { x1: 20, y1: 21, x2: 20, y2: 16 }], ['line', { x1: 20, y1: 12, x2: 20, y2: 3 }], ['line', { x1: 1, y1: 14, x2: 7, y2: 14 }], ['line', { x1: 9, y1: 8, x2: 15, y2: 8 }], ['line', { x1: 17, y1: 16, x2: 23, y2: 16 }]],
    summary: [['path', { d: 'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z' }], ['polyline', { points: '14 2 14 8 20 8' }], ['line', { x1: 16, y1: 13, x2: 8, y2: 13 }], ['line', { x1: 16, y1: 17, x2: 8, y2: 17 }]],
    progress: [['polyline', { points: '22 12 18 12 15 21 9 3 6 12 2 12' }]],
    sources: [['circle', { cx: 12, cy: 12, r: 10 }], ['line', { x1: 2, y1: 12, x2: 22, y2: 12 }], ['path', { d: 'M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z' }]],
    posts: [['path', { d: 'M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z' }]],
    narratives: [['polygon', { points: '12 2 2 7 12 12 22 7 12 2' }], ['polyline', { points: '2 17 12 22 22 17' }], ['polyline', { points: '2 12 12 17 22 12' }]],
    coordination: [['path', { d: 'M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2' }], ['circle', { cx: 9, cy: 7, r: 4 }], ['path', { d: 'M23 21v-2a4 4 0 0 0-3-3.87' }], ['path', { d: 'M16 3.13a4 4 0 0 1 0 7.75' }]],
    origin: [['circle', { cx: 18, cy: 5, r: 3 }], ['circle', { cx: 6, cy: 12, r: 3 }], ['circle', { cx: 18, cy: 19, r: 3 }], ['line', { x1: 8.59, y1: 13.51, x2: 15.42, y2: 17.49 }], ['line', { x1: 15.41, y1: 6.51, x2: 8.59, y2: 10.49 }]],
    claims: [['path', { d: 'M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z' }], ['polyline', { points: '9 12 11 14 15 10' }]],
    exports: [['path', { d: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4' }], ['polyline', { points: '7 10 12 15 17 10' }], ['line', { x1: 12, y1: 15, x2: 12, y2: 3 }]],
    system: [['rect', { x: 2, y: 3, width: 20, height: 14, rx: 2 }], ['line', { x1: 8, y1: 21, x2: 16, y2: 21 }], ['line', { x1: 12, y1: 17, x2: 12, y2: 21 }]],
    light: [['circle', { cx: 12, cy: 12, r: 5 }], ['line', { x1: 12, y1: 1, x2: 12, y2: 3 }], ['line', { x1: 12, y1: 21, x2: 12, y2: 23 }], ['line', { x1: 4.22, y1: 4.22, x2: 5.64, y2: 5.64 }], ['line', { x1: 18.36, y1: 18.36, x2: 19.78, y2: 19.78 }], ['line', { x1: 1, y1: 12, x2: 3, y2: 12 }], ['line', { x1: 21, y1: 12, x2: 23, y2: 12 }], ['line', { x1: 4.22, y1: 19.78, x2: 5.64, y2: 18.36 }], ['line', { x1: 18.36, y1: 5.64, x2: 19.78, y2: 4.22 }]],
    dark: [['path', { d: 'M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z' }]],
  };
  const STANCES = ['assert', 'report', 'question', 'debunk'];
  const SIGNAL_LABELS = {
    near_duplicate_text: 'Near-duplicate text',
    near_simultaneous_posting: 'Near-simultaneous posting',
    similar_creation_dates: 'Similar account creation dates',
    reused_image: 'Reused image',
    amplification_loop: 'Amplification loop',
  };
  const KNOWN_PLATFORMS = ['x', 'telegram', 'youtube', 'linkedin', 'linkedin_ads'];
  // Human labels for source names (the badge keeps the raw name).
  const SOURCE_LABELS = { linkedin: 'LinkedIn posts' };
  function sourceLabel(name) { return SOURCE_LABELS[name] || name; }
  // LinkedIn posts card (cookie + walk settings); the warning is shown on the card.
  const LINKEDIN_KEYS = ['LINKEDIN_COOKIE', 'LINKEDIN_MAX_SCROLLS', 'LINKEDIN_BROWSER_HEADLESS',
    'LINKEDIN_VOYAGER_QUERY_ID', 'LINKEDIN_USER_AGENT'];
  const LINKEDIN_WARNING = 'Uses your own LinkedIn account. LinkedIn restricts accounts that ' +
    'automate searches; collection is paced and stops at any checkpoint.';
  const LINKEDIN_VOYAGER_WARNING = "Voyager calls LinkedIn's API directly: faster, but no " +
    'browser, so LinkedIn may notice automation sooner.';
  const LINKEDIN_PROVIDERS = [
    ['browser', 'browser: linkedin.com search page in a local browser (default)'],
    ['voyager', 'voyager: LinkedIn Voyager search API, called directly'],
  ];
  // Per-browser memory of the last 'Sources to scan' choice (a convenience only).
  const SRC_PREF_KEY = 'socmint.newrun.sources';
  const EXTRA_COLORS = ['#7a6a9a', '#3d8f7f', '#b0702a', '#5f6f2a', '#9a4a6a', '#4a7a9a'];
  const COORD_CAVEAT = 'Coordination signals show accounts behaving similarly in the collected ' +
    'sample. They do not by themselves establish inauthentic or malicious behaviour; check the ' +
    'benign explanations and the evidence posts before drawing conclusions.';

  // ------------------------------------------------------------------ DOM helpers
  /** Safe http(s) URL or null. Rejects javascript:, data:, relative and malformed URLs. */
  function safeUrl(u) {
    if (typeof u !== 'string') return null;
    const s = u.trim();
    if (!/^https?:\/\/[^\s]+$/i.test(s)) return null;
    try { const p = new URL(s); return (p.protocol === 'http:' || p.protocol === 'https:') ? p.href : null; }
    catch (e) { return null; }
  }

  function append(node, child) {
    if (child === null || child === undefined || child === false) return;
    if (Array.isArray(child)) { child.forEach((c) => append(node, c)); return; }
    node.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }

  /** Create an element. Text is always inserted as text nodes, never parsed as HTML. */
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    setAttrs(el, attrs);
    append(el, children);
    return el;
  }

  function setAttrs(el, attrs) {
    if (!attrs) return;
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') el.setAttribute('class', v);
      else if (k === 'text') el.textContent = String(v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (k === 'href' || k === 'src') {
        // Internal app links (#...) and same-origin API paths (/api/...) are allowed;
        // everything else must be http(s).
        const s = String(v);
        if (s.startsWith('#') || s.startsWith('/api/') || s.startsWith('blob:')) el.setAttribute(k, s);
        else { const u = safeUrl(s); if (u) el.setAttribute(k, u); }
      } else if (v === true) el.setAttribute(k, '');
      else el.setAttribute(k, String(v));
    }
  }

  const SVGNS = 'http://www.w3.org/2000/svg';
  function s(tag, attrs, ...children) {
    const el = document.createElementNS(SVGNS, tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v === null || v === undefined) continue;
        if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
        else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
        else el.setAttribute(k, String(v));
      }
    }
    append(el, children);
    return el;
  }

  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

  function pageHead(title, sub, action) {
    return h('div', { class: 'page-head' },
      h('div', { class: 'row' }, h('h1', null, title), action ? h('div', { class: 'spacer' }) : null, action || null),
      sub ? h('p', { class: 'sub' }, sub) : null);
  }

  function icon(name) {
    return s('svg', { viewBox: '0 0 24 24', 'aria-hidden': 'true', focusable: 'false' },
      (ICONS[name] || []).map(([tag, attrs]) => s(tag, attrs)));
  }

  function extLink(url, label) {
    const u = safeUrl(url);
    if (!u) return h('span', { class: 'muted' }, label || url || '');
    return h('a', { href: u, target: '_blank', rel: 'noopener noreferrer' }, label || u);
  }

  function enc(x) { return encodeURIComponent(String(x)); }
  function plural(n, w) {
    if (n === 1) return `${n} ${w}`;
    return `${n} ${/[^aeiou]y$/i.test(w) ? w.slice(0, -1) + 'ies' : w + 's'}`;
  }

  // Long text that must never be silently cut (status reasons): the first part is shown and
  // the rest opens in place, as in the TG Collector's error cells.
  function longText(text, max) {
    const s = String(text || '');
    const n = max || 240;
    if (s.length <= n) return s;
    const cut = s.lastIndexOf(' ', n) > n * 0.6 ? s.lastIndexOf(' ', n) : n;
    return h('details', { class: 'longtext' },
      h('summary', null, h('span', { class: 'preview' }, s.slice(0, cut) + '\u2026 '),
        h('span', { class: 'more' }, 'show full reason'), h('span', { class: 'less' }, 'show less')),
      h('div', { class: 'full' }, s));
  }
  // ---- Plain-language source status (socmint/status_text.py): the main reason first, then
  // what to do / when it resets; the technical reason stays one click away.
  const HEAD_ORDER = ['account', 'quota', 'rate_limit', 'error', 'setup', 'stopped', 'limit', 'coverage', 'none', 'ok'];
  const HEAD_TONE = { account: 'bad', quota: 'bad', rate_limit: 'bad', error: 'bad', setup: 'warn', stopped: 'warn', limit: 'warn', coverage: 'warn', none: 'muted', ok: 'ok' };
  const HEAD_LABEL = { account: 'Account check', quota: 'Quota used up', rate_limit: 'Rate limited', error: 'Error', setup: 'Not set up', stopped: 'Stopped by you', limit: 'Limit reached', coverage: 'Partial coverage', none: 'No results', ok: 'OK' };
  function isProblem(kind) { return kind && !['ok', 'none'].includes(kind); }
  /** The most important status of a source (several statuses for Telegram, etc.). */
  function worstStatus(list) {
    const rank = (st) => { const i = HEAD_ORDER.indexOf(st.headline_kind); return i < 0 ? HEAD_ORDER.length : i; };
    return (list || []).slice().sort((a, b) => rank(a) - rank(b))[0] || null;
  }
  /** Headline block: tag + plain sentence (+ action, + reset time) + optional details. */
  function statusHead(st, opts) {
    const o = opts || {};
    if (!st || (!st.headline && !st.reason)) return null;
    const kind = st.headline_kind || '';
    const tone = HEAD_TONE[kind] || 'muted';
    const text = st.headline || st.reason;
    return h('div', { class: `status-head tone-${tone}` },
      h('div', { class: 'sh-line' },
        kind && kind !== 'ok' ? h('span', { class: `chip ${tone === 'bad' ? 'bad' : tone === 'warn' ? 'warn' : 'skip'} sh-tag` }, HEAD_LABEL[kind] || kind) : null,
        h('span', { class: 'sh-text' }, text)),
      st.action ? h('div', { class: 'sh-action' }, st.action) : null,
      o.details !== false && st.headline && st.reason ? h('details', { class: 'sh-details' },
        h('summary', null, 'Technical details'), h('div', { class: 'small' }, st.reason)) : null);
  }

  /** Nav/section counts: numbers as '1,960'; preformatted labels ('6 / 20') as given. */
  function countLabel(c) { return typeof c === 'number' ? fmtNum(c) : String(c); }
  function fmtNum(n) { return (typeof n === 'number') ? n.toLocaleString('en-US') : '–'; }

  function fmtTs(iso) {
    if (!iso) return '–';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return String(iso);
    return d.toISOString().replace('T', ' ').slice(0, 16) + ' UTC';
  }
  function fmtDay(iso) { return iso ? String(iso).slice(0, 10) : '–'; }
  function fmtLag(sec) {
    if (typeof sec !== 'number' || !isFinite(sec)) return '–';
    const a = Math.abs(sec);
    const sign = sec < 0 ? '-' : '+';
    if (a < 60) return `${sign}${Math.round(a)}s`;
    if (a < 3600) return `${sign}${Math.round(a / 60)}m`;
    if (a < 86400) return `${sign}${(a / 3600).toFixed(1)}h`;
    return `${sign}${(a / 86400).toFixed(1)}d`;
  }

  function platformColor(src) {
    if (KNOWN_PLATFORMS.includes(src)) return `var(--p-${src})`;
    let hsh = 0;
    for (const ch of String(src || '')) hsh = (hsh * 31 + ch.charCodeAt(0)) >>> 0;
    return EXTRA_COLORS[hsh % EXTRA_COLORS.length];
  }
  // Platform marks (single-path, 24x24, from the CC0 simple-icons set), drawn in the
  // chip's text colour so they follow the light/dark theme. Inline: no network fetch.
  const PLATFORM_LOGOS = {
    x: 'M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932ZM17.61 20.644h2.039L6.486 3.24H4.298Z',
    telegram: 'M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.48.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z',
    youtube: 'M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z',
    linkedin: 'M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.73 1.636-1.555 3.368-1.555 3.601 0 4.267 2.37 4.267 4.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z',
  };
  PLATFORM_LOGOS.linkedin_ads = PLATFORM_LOGOS.linkedin;
  function platformLogo(src) {
    const d = PLATFORM_LOGOS[src];
    if (!d) return null;
    return s('svg', { class: 'plat-logo', viewBox: '0 0 24 24', 'aria-hidden': 'true', focusable: 'false' }, s('path', { d }));
  }
  function platformBadge(src) {
    // Known platforms take their text colour from the theme (--on-platform); the hashed
    // fallback colours are mid-dark in both themes, so white text is used on them.
    const style = KNOWN_PLATFORMS.includes(src) ? { background: platformColor(src) } : { background: platformColor(src), color: '#fff' };
    return h('span', { class: 'badge platform', style, title: sourceLabel(src) }, platformLogo(src), pname(src));
  }

  /** Analysable = usable for timelines/counts/origin (mirrors socmint.verify.analysable). */
  function analysable(p) {
    return !!(p && p.timestamp_verified && p.in_window !== false && p.published_at &&
      !isNaN(new Date(p.published_at).getTime()));
  }

  function stateClass(state) {
    const st = String(state || '');
    if (st === 'ok') return 'ok';
    if (st === 'partial' || st === 'completed_zero' || st === 'out_of_coverage') return 'warn';
    if (st.startsWith('skipped')) return 'skip';
    if (st === 'failed') return 'bad';
    if (st === 'collecting' || st === 'running') return 'run';
    return '';
  }
  function stateChip(state) {
    return h('span', { class: `chip ${stateClass(state)}` }, state || 'unknown');
  }

  function download(name, content, type) {
    const blob = new Blob([content], { type: type || 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = h('a', { href: url, download: name });
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 1000);
  }

  // ------------------------------------------------------------------ API
  async function api(path, opts) {
    const o = Object.assign({ headers: {} }, opts || {});
    if (o.body !== undefined && typeof o.body !== 'string') {
      o.body = JSON.stringify(o.body);
      o.headers['Content-Type'] = 'application/json';
    }
    const res = await fetch(path, o);
    let data = null;
    const txt = await res.text();
    try { data = txt ? JSON.parse(txt) : null; } catch (e) { data = txt; }
    if (!res.ok) {
      const detail = (data && typeof data === 'object' && data.detail) ? data.detail : (txt || res.statusText);
      const err = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
      err.status = res.status;
      throw err;
    }
    return data;
  }

  async function getMeta(force) {
    if (STATIC) return { read_only: true, sources: [], max_window_days: 30 };
    if (!S.meta || force) S.meta = await api('/api/meta');
    return S.meta;
  }

  async function getRun(id) {
    if (STATIC) return EMBEDDED || {};
    // Only finished bundles are cached. A bundle fetched while the run is still running
    // is partial (no narratives/clusters/claims yet); caching it would make findings look
    // absent after the run completes, so such bundles are always re-fetched.
    const cached = S.runs[id];
    if (cached && isFinalBundle(cached)) return cached;
    const bundle = await api(`/api/runs/${enc(id)}`);
    if (isFinalBundle(bundle)) S.runs[id] = bundle; else delete S.runs[id];
    return bundle;
  }

  // BEGIN isFinalBundle
  function isFinalBundle(bundle) {
    const st = bundle && bundle.run && bundle.run.status;
    return typeof st === 'string' && st !== '' && st !== 'running' && st !== 'pending' && st !== 'queued';
  }
  // END isFinalBundle

  function rawHref(runId, rawRef) {
    // Server mode only: raw payloads are served under /api/runs/<id>/raw/<path relative to raw/>.
    if (STATIC || !rawRef) return null;
    const rel = String(rawRef).replace(/^\/+/, '').replace(/^raw\//, '');
    if (rel.split('/').some((seg) => seg === '..')) return null;
    return `/api/runs/${enc(runId)}/raw/${rel.split('/').map(enc).join('/')}`;
  }

  // ------------------------------------------------------------------ shell & router
  const root = document.getElementById('app');
  const shell = {};

  /** SpiderSilk wordmark: cloned from the <template id="brand-logo"> that index.html and
   *  report.html both carry (a DOM clone, no HTML parsing, no network). Falls back to plain text. */
  function brandLogo() {
    const tpl = document.getElementById('brand-logo');
    const box = h('span', { class: 'brand-logo' });
    const svg = tpl && tpl.content ? tpl.content.querySelector('svg') : null;
    if (svg) box.appendChild(svg.cloneNode(true));
    else box.appendChild(document.createTextNode('SpiderSilk'));
    return box;
  }

  function themeButton() {
    const names = { system: 'System', light: 'Light', dark: 'Dark' };
    const btn = h('button', { type: 'button', class: 'theme-toggle' });
    const draw = () => {
      const t = getTheme();
      clear(btn);
      append(btn, [icon(t), h('span', { class: 'tlabel' }, 'Theme'), h('span', { class: 'tval' }, names[t])]);
      const next = THEMES[(THEMES.indexOf(t) + 1) % THEMES.length];
      btn.setAttribute('title', `Theme: ${names[t]}. Click for ${names[next]}.`);
      btn.setAttribute('aria-label', `Colour theme: ${names[t]}. Switch to ${names[next]}.`);
    };
    btn.addEventListener('click', () => {
      const t = getTheme();
      setTheme(THEMES[(THEMES.indexOf(t) + 1) % THEMES.length]);
      draw();
    });
    draw();
    return btn;
  }

  function buildShell(meta) {
    clear(root);
    const brand = h('a', { class: 'brand', href: STATIC ? '#/' : '#/runs', 'aria-label': 'SpiderSilk SOCMINT' },
      h('div', { class: 'brand-row' }, brandLogo(), h('span', { class: 'brand-div' }), h('span', { class: 'brand-word' }, 'SOCMINT')),
      h('div', { class: 'brand-sub' }, STATIC ? 'standalone report' : 'narrative & coordination'));
    shell.navLabel = h('div', { class: 'side-label' }, 'Menu');
    shell.nav = h('nav', { class: 'side-nav', 'aria-label': 'Main' });
    shell.runLabel = h('div', { class: 'side-label hidden' });
    shell.runNav = h('nav', { class: 'side-nav run-nav', 'aria-label': 'Run sections' });
    const badges = h('div', { class: 'badges' });
    if (STATIC) badges.appendChild(h('span', { class: 'sb ro', title: 'Data and code are in this file; post thumbnails still load from the platforms when online.' }, 'Standalone report'));
    else if (meta && meta.read_only) badges.appendChild(h('span', { class: 'sb ro' }, 'Read-only'));
    if (meta && meta.version) badges.appendChild(h('span', { class: 'sb ver' }, `v${meta.version}`));
    const foot = h('div', { class: 'side-foot' },
      h('div', { class: 'where' }, STATIC ? 'Local file' : 'Local · 127.0.0.1'),
      badges.childNodes.length ? badges : null,
      themeButton());
    const aside = h('aside', { class: 'sidebar' }, brand, shell.navLabel, shell.nav, shell.runLabel, shell.runNav, foot);
    shell.main = h('main', { id: 'main' });
    root.appendChild(h('div', { class: 'app' }, aside, shell.main));
    shell.modal = h('div', { class: 'modal-backdrop hidden', onclick: (e) => { if (e.target === shell.modal) closeModal(); } });
    root.appendChild(shell.modal);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });
  }

  function navLink(key, label, href, active, count) {
    return h('a', { href, title: label, class: active === key ? 'active' : null, 'aria-current': active === key ? 'page' : null },
      icon(key), h('span', { class: 'label' }, label),
      count !== undefined ? h('span', { class: 'n' }, countLabel(count)) : null);
  }

  function updateNav(active) {
    clear(shell.nav);
    updateRunNav(null);
    shell.navLabel.classList.toggle('hidden', STATIC);
    if (STATIC) return;
    const ro = S.meta && S.meta.read_only;
    const items = [['runs', 'Runs', '#/runs']];
    if (!ro) items.push(['new', 'New run', '#/new'], ['settings', 'Settings', '#/settings']);
    for (const [key, label, href] of items) shell.nav.appendChild(navLink(key, label, href, active));
  }

  /** Second sidebar group mirroring the open run's section tabs (null clears it). */
  function updateRunNav(ctx, active, counts) {
    clear(shell.runNav);
    if (!ctx) { shell.runLabel.classList.add('hidden'); return; }
    shell.runLabel.textContent = STATIC ? 'Report' : 'This run';
    shell.runLabel.setAttribute('title', ctx.run.topic || ctx.id);
    shell.runLabel.classList.remove('hidden');
    for (const [k, label] of TABS) {
      shell.runNav.appendChild(navLink(k, label, `#/run/${enc(ctx.id)}/${k}`, active, (counts || {})[k]));
    }
  }

  function parseHash() {
    const raw = location.hash.replace(/^#\/?/, '');
    return raw ? raw.split('/').map((x) => { try { return decodeURIComponent(x); } catch (e) { return x; } }) : [];
  }

  async function route() {
    if (S.timer) { clearTimeout(S.timer); S.timer = null; }
    closeModal();
    const seq = ++S.renderSeq;
    const parts = parseHash();
    const page = parts[0] || '';
    const main = clear(shell.main);
    try {
      if (STATIC) {
        updateNav('');
        const rid = (EMBEDDED && EMBEDDED.run && EMBEDDED.run.run_id) || 'report';
        const tab = page === 'run' ? parts[2] : null;
        await pageResults(main, rid, tab, seq, page === 'run' ? parts[3] : null);
        return;
      }
      const meta = await getMeta();
      const ro = meta.read_only;
      if (page === 'settings' && !ro) { updateNav('settings'); await pageSettings(main, seq); }
      else if (page === 'new' && !ro) { updateNav('new'); await pageNew(main, seq); }
      else if (page === 'progress' && parts[1]) { updateNav(''); await pageProgress(main, parts[1], seq); }
      else if (page === 'run' && parts[1]) { updateNav(''); await pageResults(main, parts[1], parts[2], seq, parts[3]); }
      else if (page === 'runs') { updateNav('runs'); await pageRuns(main, seq); }
      else if (meta.initial_run_id) { location.replace(`#/run/${enc(meta.initial_run_id)}`); }
      else { location.replace('#/runs'); }
    } catch (e) {
      if (seq !== S.renderSeq) return;
      main.appendChild(h('div', { class: 'banner bad' }, `Could not load this page: ${e.message || e}`));
    }
  }

  // ------------------------------------------------------------------ runs list
  async function pageRuns(main, seq) {
    const runs = await api('/api/runs');
    if (seq !== S.renderSeq) return;
    // Opening the runs list drops every cached bundle so that a run opened from here is
    // always fresh (a run may have finished in another tab or via the CLI).
    S.runs = {};
    main.appendChild(pageHead('Runs', 'Every investigation on this machine. Open a run to browse its posts, narratives, coordination, origin and claims, each linked to its evidence.',
      (S.meta && !S.meta.read_only) ? h('a', { class: 'btn primary', href: '#/new' }, icon('new'), 'New run') : null));
    if (!runs || !runs.length) {
      main.appendChild(h('div', { class: 'card' }, 'No runs yet. ',
        (S.meta && !S.meta.read_only) ? h('a', { href: '#/new' }, 'Start a new run.') : null));
      return;
    }
    const tbody = h('tbody');
    for (const r of runs) {
      const running = r.status === 'running';
      const href = running ? `#/progress/${enc(r.run_id)}` : `#/run/${enc(r.run_id)}`;
      const counts = r.counts || {};
      tbody.appendChild(h('tr', null,
        h('td', null, h('a', { href }, r.topic || r.run_id), h('div', { class: 'small muted mono' }, r.run_id)),
        h('td', null, `${fmtDay(r.start)} → ${fmtDay(r.end)}`),
        h('td', null, h('span', { class: `chip ${r.status === 'complete' ? 'ok' : r.status === 'failed' ? 'bad' : 'run'}` }, r.status || '?')),
        h('td', null, fmtNum(counts.posts)),
        h('td', null, h('div', { class: 'chips' }, (r.sources || []).map((x) => platformBadge(x)))),
        h('td', null, fmtTs(r.created_at)),
        h('td', null, r.path_kind ? h('span', { class: 'chip' }, r.path_kind) : ''),
        h('td', null, canDelete(r) ? deleteButton(r.run_id, r.topic, () => route()) : '')));
    }
    main.appendChild(h('div', { class: 'card table-wrap' }, h('table', null,
      h('thead', null, h('tr', null, ['Topic', 'Window', 'Status', 'Posts', 'Sources', 'Created', '', ''].map((t) => h('th', null, t)))),
      tbody)));
  }

  // Only runs inside the runs/ folder (path_kind 'run') can be deleted, never the bundled
  // sample, an external --run directory, a running run, or anything in read-only mode
  // (the server enforces the same rules).
  function canDelete(r) {
    return !STATIC && S.meta && !S.meta.read_only && r && r.path_kind === 'run'
      && r.status !== 'running';
  }

  /** Re-run only the analysis (no collection) on a finished run in the runs folder. */
  function canReanalyze(r) {
    return !STATIC && S.meta && !S.meta.read_only && r && r.path_kind === 'run'
      && r.status !== 'running';
  }

  function reanalyzeButton(runId) {
    const btn = h('button', { class: 'btn small', type: 'button', title: 'Re-run narratives, coordination, origin, claims and the summary on the saved posts' }, 'Re-analyse');
    btn.addEventListener('click', async (e) => {
      e.preventDefault();
      if (!window.confirm('Re-analyse this run?\n\nNo new posts are collected. Narratives, coordinated groups, ' +
        'origin, claims and the summary are rebuilt from the saved posts and replace the current analysis. ' +
        'This uses the configured LLM and evidence-search keys (small cost) and takes a few minutes.')) return;
      btn.disabled = true;
      btn.textContent = 'Starting…';
      try {
        await api(`/api/runs/${enc(runId)}/reanalyze`, { method: 'POST', body: {} });
        location.hash = `#/progress/${enc(runId)}`;
      } catch (err) {
        btn.disabled = false;
        btn.textContent = 'Re-analyse';
        window.alert(`Could not start the re-analysis: ${err.message || err}`);
      }
    });
    return btn;
  }

  function deleteButton(runId, topic, onDone) {
    const btn = h('button', { class: 'btn danger small', type: 'button' }, 'Delete');
    btn.addEventListener('click', async (e) => {
      e.preventDefault();
      e.stopPropagation();
      const label = topic ? `"${topic}" (${runId})` : runId;
      if (!window.confirm(`Delete run ${label}?\n\nThis permanently removes its posts, ` +
        'analysis, exports, report and raw payloads from disk. It cannot be undone.')) return;
      btn.disabled = true;
      btn.textContent = 'Deleting…';
      try {
        await api(`/api/runs/${enc(runId)}`, { method: 'DELETE' });
        delete S.runs[runId];
        // The GUI root opens initial_run_id; do not send it to a run that is gone.
        if (S.meta && S.meta.initial_run_id === runId) S.meta.initial_run_id = null;
        onDone();
      } catch (err) {
        btn.disabled = false;
        btn.textContent = 'Delete';
        window.alert(`Could not delete: ${err.message || err}`);
      }
    });
    return btn;
  }

  // ------------------------------------------------------------------ settings
  async function pageSettings(main, seq) {
    const [meta, settings] = await Promise.all([getMeta(true), api('/api/settings')]);
    if (seq !== S.renderSeq) return;
    const keyInfo = {};
    for (const k of (settings.keys || [])) keyInfo[k.key] = k;
    const cfg = settings.config || {};

    main.appendChild(pageHead('Settings',
      'Credentials are stored in your local config file, never in the repository. A source without ' +
      'its credentials is not an error: runs record it as skipped_no_credentials and explain why.'));

    const used = new Set();
    const grid = h('div', { class: 'grid' });
    for (const src of (meta.sources || [])) {
      (src.credential_keys || []).forEach((k) => used.add(k));
      if (src.name === 'x') {
        X_ALL_KEYS.forEach((k) => used.add(k));
        const provider = xProvider(cfg);
        const keys = X_PROVIDER_KEYS[provider] || src.credential_keys || [];
        grid.appendChild(settingsCard(src.name, src, keys, keyInfo, cfg, xProviderField(provider)));
        continue;
      }
      if (src.name === 'linkedin') {
        LINKEDIN_KEYS.forEach((k) => used.add(k));
        used.add('LINKEDIN_PROVIDER');
        grid.appendChild(settingsCard(sourceLabel(src.name), src, LINKEDIN_KEYS, keyInfo, cfg, linkedinNote(cfg)));
        continue;
      }
      grid.appendChild(settingsCard(sourceLabel(src.name), src, src.credential_keys || [], keyInfo, cfg));
    }
    const rest = Object.keys(keyInfo).filter((k) => !used.has(k));
    if (rest.length) grid.appendChild(settingsCard('LLM, evidence & other keys', null, rest, keyInfo, cfg));
    main.appendChild(grid);

    const xs = (meta.sources || []).find((x) => x.name === 'x');
    if (xs && xProvider(cfg) === 'web_session') main.appendChild(await xSessionBox(keyInfo, cfg, seq));
    const tg = (meta.sources || []).find((x) => x.name === 'telegram');
    if (tg) main.appendChild(await telegramBox());
  }

  function linkedinNote(cfg) {
    const current = (cfg && cfg.LINKEDIN_PROVIDER) || 'browser';
    const msg = h('div', { class: 'small' });
    const sel = h('select', {
      onchange: () => saveKeys({ LINKEDIN_PROVIDER: sel.value }, msg),
    }, LINKEDIN_PROVIDERS.map(([v, label]) => h('option', { value: v, selected: v === current }, label)));
    if (!LINKEDIN_PROVIDERS.some(([v]) => v === current)) {
      sel.appendChild(h('option', { value: current, selected: true }, `${current} (unknown)`));
    }
    return h('div', { class: 'small li-note' },
      h('div', { class: 'warn-line' }, LINKEDIN_WARNING),
      h('div', { class: 'warn-line' }, LINKEDIN_VOYAGER_WARNING),
      h('label', { class: 'field' }, h('span', null, 'LINKEDIN_PROVIDER'), sel), msg,
      h('div', { class: 'muted' },
        'LINKEDIN_COOKIE: the Cookie header of a logged-in linkedin.com tab (or the li_at value ' +
        'alone; voyager needs the full header with JSESSIONID). LINKEDIN_MAX_SCROLLS: 1-200 ' +
        '(browser: about 5 posts and 8 s per scroll; voyager: pages of 25 posts). ' +
        'LINKEDIN_BROWSER_HEADLESS: true or false (browser only). LINKEDIN_VOYAGER_QUERY_ID: ' +
        'LinkedIn rotates it; on HTTP 400/404 copy the current voyagerSearchDashClusters.* id ' +
        'from your browser DevTools > Network (empty restores the default). ' +
        'LINKEDIN_USER_AGENT: your own browser\'s user agent (empty = built-in desktop Chrome).'));
  }

  // run | skip | fail, from the readiness row (`outcome`), so a source that will FAIL (e.g. X
  // session authorized but no browser) is never shown as merely "skipped".
  function srcOutcome(src) {
    if (src && ['run', 'skip', 'fail'].includes(src.outcome)) return src.outcome;
    return src && src.ready ? 'run' : 'skip';
  }
  function outcomeChip(src) {
    const o = srcOutcome(src);
    const [cls, label] = { run: ['ok', 'ready'], skip: ['skip', 'will be skipped'], fail: ['bad', 'will fail'] }[o];
    return h('span', { class: `chip ${cls}` }, label);
  }
  function reasonText(src) {
    // Readiness reasons already start with "will skip:" / "will fail:"; the chip says that.
    return String((src && src.reason) || 'not ready').replace(/^will (skip|fail):\s*/, '');
  }

  function settingsCard(title, src, keys, keyInfo, cfg, extra) {
    const card = h('div', { class: 'card' });
    const head = h('div', { class: 'row' }, h('h3', null, title));
    if (src) {
      head.appendChild(outcomeChip(src));
      if (src.kind) head.appendChild(h('span', { class: 'chip' }, src.kind));
    }
    card.appendChild(head);
    if (src && src.reason) card.appendChild(h('div', { class: 'small muted' }, src.reason));
    // X web_session needs a logged-in session rather than a key (see the X session box).
    if (src && !src.needs_credentials && src.name !== 'x') card.appendChild(h('div', { class: 'small muted' }, 'No credentials needed.'));
    if (extra) card.appendChild(extra);
    const inputs = {};
    for (const key of keys) {
      const info = keyInfo[key] || { key, secret: /KEY|TOKEN|SECRET|PASS|HASH|COOKIE/i.test(key), set: !!cfg[key] };
      const masked = cfg[key] || '';
      const inp = h('input', {
        type: info.secret ? 'password' : 'text', autocomplete: 'off', spellcheck: 'false',
        placeholder: info.secret ? (info.set ? `set ${masked || '••••'}` : 'not set') : '',
      });
      if (!info.secret) inp.value = masked;
      inp.dataset.initial = info.secret ? '' : masked;
      inputs[key] = inp;
      const status = h('span', { class: `chip ${info.set ? 'ok' : 'skip'}` }, info.set ? (info.secret ? `set ${masked}` : 'set') : 'not set');
      const clearBtn = (info.secret && info.set)
        ? h('button', { type: 'button', class: 'link small', onclick: () => saveKeys({ [key]: '' }, msg) }, 'clear') : null;
      card.appendChild(h('label', { class: 'field' }, h('span', null, key, ' ', status, ' ', clearBtn), inp));
    }
    const msg = h('div', { class: 'small' });
    if (keys.length) {
      card.appendChild(h('button', {
        class: 'primary', type: 'button', onclick: () => {
          const body = {};
          for (const [k, inp] of Object.entries(inputs)) {
            const v = inp.value;
            if (inp.type === 'password' ? v !== '' : v !== inp.dataset.initial) body[k] = v;
          }
          if (!Object.keys(body).length) { msg.className = 'form-error'; msg.textContent = 'Nothing changed.'; return; }
          saveKeys(body, msg);
        },
      }, 'Save'));
    }
    card.appendChild(msg);
    return card;
  }

  async function saveKeys(body, msg) {
    try {
      await api('/api/settings', { method: 'POST', body });
      S.meta = null;
      msg.className = 'form-ok';
      msg.textContent = 'Saved.';
      setTimeout(route, 500);
    } catch (e) {
      msg.className = 'form-error';
      msg.textContent = `Save failed: ${e.message}`;
    }
  }

  // ------------------------------------------------------------------ X provider + session
  const X_PROVIDERS = [
    ['web_session', 'web_session: your own X account (browser login)'],
    ['twitterapi_io', 'twitterapi_io: TwitterAPI.io (paid API key)'],
    ['official', 'official: X API v2 (bearer token from developer.x.com, not from x.com)'],
  ];
  // Settings shown in the X card for each provider (cookies live in the X session box).
  const X_PROVIDER_KEYS = {
    web_session: ['X_MAX_SCROLLS', 'X_BROWSER_HEADLESS'],
    twitterapi_io: ['TWITTERAPI_IO_KEY'],
    official: ['X_BEARER_TOKEN'],
  };
  const X_ALL_KEYS = ['X_PROVIDER', 'X_AUTH_TOKEN', 'X_CT0', 'X_MAX_SCROLLS', 'X_BROWSER_HEADLESS',
    'TWITTERAPI_IO_KEY', 'X_BEARER_TOKEN'];
  const X_POLL_MS = 2000;

  function xProvider(cfg) {
    const v = String((cfg && cfg.X_PROVIDER) || 'web_session').trim().toLowerCase();
    return v || 'web_session';
  }

  function xProviderField(current) {
    const msg = h('div', { class: 'small' });
    const sel = h('select', {
      onchange: () => saveKeys({ X_PROVIDER: sel.value }, msg),
    }, X_PROVIDERS.map(([v, label]) => h('option', { value: v, selected: v === current }, label)));
    if (!X_PROVIDERS.some(([v]) => v === current)) {
      sel.appendChild(h('option', { value: current, selected: true }, `${current} (unknown)`));
    }
    return h('div', null, h('label', { class: 'field' }, h('span', null, 'X_PROVIDER'), sel), msg);
  }

  async function xSessionBox(keyInfo, cfg, seq) {
    const box = h('div', { class: 'card x-session' }, h('h3', null, 'X session'));
    box.appendChild(h('p', { class: 'small warn-note' },
      'Uses your own X account. Collection is paced and stops on rate limits; heavy use can get an account rate-limited or locked.'));
    box.appendChild(h('p', { class: 'small muted' },
      'You log in yourself in a normal browser window (CAPTCHA and 2FA included). SOCMINT never sees ' +
      'your password; it only keeps the resulting session on this machine.'));
    const body = h('div');
    box.appendChild(body);
    const stopPoll = () => { if (S.timer) { clearTimeout(S.timer); S.timer = null; } };
    let prevState = null;

    const draw = async (note) => {
      stopPoll();
      if (seq !== S.renderSeq) return; // navigated away: stop polling
      let st;
      try { st = await api('/api/x/status'); } catch (e) {
        if (seq !== S.renderSeq) return;
        clear(body);
        body.appendChild(h('div', { class: 'form-error' }, `Status unavailable: ${e.message}`));
        return;
      }
      if (seq !== S.renderSeq) return;
      st = st || {};
      if (prevState === 'waiting' && st.login_state !== 'waiting') {
        // Login finished (or failed): the X card above was drawn from the cached /api/meta
        // and would still say "will be skipped". Re-render the whole page from fresh meta.
        S.meta = null;
        route();
        return;
      }
      prevState = st.login_state || null;
      clear(body);
      const who = st.user ? `authorized as @${String(st.user).replace(/^@/, '')}` : 'authorized';
      body.appendChild(h('div', { class: 'row' },
        h('span', { class: `chip ${st.authorized ? 'ok' : 'skip'}` }, st.authorized ? who : 'not logged in'),
        st.source ? h('span', { class: 'chip' }, `source: ${st.source}`) : null,
        h('span', { class: `chip ${st.browser ? '' : 'warn'}` }, `browser: ${st.browser || 'not found'}`)));
      if (st.reason) body.appendChild(h('div', { class: 'small muted' }, st.reason));
      const msg = h('div', { class: 'small' });
      if (note) { msg.className = 'small form-ok'; msg.textContent = note; }

      if (st.login_state === 'waiting') {
        body.appendChild(h('div', { class: 'banner warn' },
          'Complete the login in the browser window that opened… This page updates when the login finishes.'));
        body.appendChild(msg);
        S.timer = setTimeout(() => draw(), X_POLL_MS);
        return;
      }
      if (st.login_state === 'failed' && !st.authorized) {
        msg.className = 'small form-error';
        msg.textContent = st.login_error
          ? `The last login attempt failed: ${st.login_error}`
          : 'The last login attempt did not finish (timed out, window closed or error). Try again.';
      }

      const actions = h('div', { class: 'row' });
      actions.appendChild(h('button', {
        type: 'button', class: st.authorized ? null : 'primary', onclick: async () => {
          try {
            const r = await api('/api/x/login', { method: 'POST', body: {} });
            if (r && r.ok === false) {
              msg.className = 'small form-error';
              msg.textContent = `Could not start the login: ${r.error || r.reason || 'unknown error'}`;
              return;
            }
            draw();
          } catch (e) { msg.className = 'small form-error'; msg.textContent = e.message; }
        },
      }, st.authorized ? 'Log in again (opens a browser window)' : 'Log in to X (opens a browser window)'));
      if (st.authorized || st.source) {
        actions.appendChild(h('button', {
          type: 'button', onclick: async () => {
            try {
              const r = await api('/api/x/logout', { method: 'POST', body: {} });
              S.meta = null;
              // x_web's own message says what actually happened (deleted, or nothing to
              // delete; pasted X_AUTH_TOKEN + X_CT0 remain) — never claim a deletion here.
              draw((r && (r.message || r.note)) || 'Log out finished.');
            } catch (e) { msg.className = 'small form-error'; msg.textContent = e.message; }
          },
        }, 'Log out'));
      }
      body.appendChild(actions);
      body.appendChild(msg);
      body.appendChild(xPasteCookies(keyInfo, cfg));
    };
    await draw();
    return box;
  }

  function xPasteCookies(keyInfo, cfg) {
    const msg = h('div', { class: 'small' });
    const inputs = {};
    const fields = ['X_AUTH_TOKEN', 'X_CT0'].map((key) => {
      const info = keyInfo[key] || { set: !!cfg[key] };
      const inp = h('input', {
        type: 'password', autocomplete: 'off', spellcheck: 'false',
        placeholder: info.set ? `set ${cfg[key] || '••••'}` : 'not set',
      });
      inputs[key] = inp;
      const clearBtn = info.set
        ? h('button', { type: 'button', class: 'link small', onclick: () => saveKeys({ [key]: '' }, msg) }, 'clear') : null;
      return h('label', { class: 'field' }, h('span', null, key, ' ',
        h('span', { class: `chip ${info.set ? 'ok' : 'skip'}` }, info.set ? 'set' : 'not set'), ' ', clearBtn), inp);
    });
    return h('details', { class: 'x-paste' },
      h('summary', null, 'Paste cookies instead'),
      h('p', { class: 'small muted' },
        'From a browser where you are logged in to x.com: developer tools → Application → Cookies → ' +
        'x.com, copy the values of auth_token and ct0. They are stored only in your local config.'),
      fields,
      h('button', {
        type: 'button', class: 'primary', onclick: () => {
          const body = {};
          for (const [k, inp] of Object.entries(inputs)) if (inp.value.trim() !== '') body[k] = inp.value.trim();
          if (!Object.keys(body).length) { msg.className = 'form-error'; msg.textContent = 'Nothing changed.'; return; }
          saveKeys(body, msg);
        },
      }, 'Save cookies'),
      msg);
  }

  async function telegramBox() {
    const box = h('div', { class: 'card' }, h('h3', null, 'Telegram login'));
    const body = h('div');
    box.appendChild(h('p', { class: 'small muted' },
      'Telegram uses a user session. Enter the phone number of the account, then the login code ' +
      'Telegram sends you (and the 2FA password if the account has one). The session stays on this machine.'));
    box.appendChild(body);
    const draw = async () => {
      clear(body);
      let st;
      try { st = await api('/api/telegram/status'); } catch (e) {
        body.appendChild(h('div', { class: 'form-error' }, `Status unavailable: ${e.message}`)); return;
      }
      body.appendChild(h('div', { class: 'row' },
        h('span', { class: `chip ${st.configured ? 'ok' : 'skip'}` }, st.configured ? 'API keys set' : 'API keys missing'),
        h('span', { class: `chip ${st.authorized ? 'ok' : 'skip'}` }, st.authorized ? 'authorized' : 'not logged in'),
        st.user ? h('span', null, `as ${st.user}`) : null,
        st.authorized ? h('span', { class: `chip ${st.premium ? 'ok' : ''}` }, st.premium ? 'Premium: global search available' : 'not Premium: channel search only') : null));
      if (st.reason) body.appendChild(h('div', { class: 'small muted' }, st.reason));
      const msg = h('div', { class: 'small' });
      if (st.authorized) {
        body.appendChild(h('button', {
          type: 'button', onclick: async () => {
            try { await api('/api/telegram/logout', { method: 'POST', body: {} }); draw(); }
            catch (e) { msg.className = 'form-error'; msg.textContent = e.message; }
          },
        }, 'Log out'));
        body.appendChild(msg);
        return;
      }
      if (!st.configured) { body.appendChild(h('div', { class: 'small muted' }, 'Save the Telegram API keys above first.')); return; }
      const phone = h('input', { type: 'text', placeholder: '+15551234567', autocomplete: 'off' });
      const code = h('input', { type: 'text', placeholder: 'login code', autocomplete: 'one-time-code' });
      const pw = h('input', { type: 'password', placeholder: '2FA password (if enabled)', autocomplete: 'off' });
      const step2 = h('div', { class: 'hidden' },
        h('label', { class: 'field' }, h('span', null, 'Code'), code),
        h('label', { class: 'field' }, h('span', null, '2FA password'), pw),
        h('button', {
          type: 'button', class: 'primary', onclick: async () => {
            try {
              const r = await api('/api/telegram/sign_in', { method: 'POST', body: { code: code.value.trim(), password: pw.value || undefined } });
              if (r && r.ok) { draw(); return; }
              msg.className = 'form-error';
              msg.textContent = (r && r.needs_password) ? 'This account has 2FA: enter the password and sign in again.' : `Sign-in failed: ${(r && r.error) || 'unknown error'}`;
            } catch (e) { msg.className = 'form-error'; msg.textContent = e.message; }
          },
        }, 'Sign in'));
      body.appendChild(h('label', { class: 'field' }, h('span', null, 'Phone'), phone));
      body.appendChild(h('button', {
        type: 'button', onclick: async () => {
          try {
            const r = await api('/api/telegram/send_code', { method: 'POST', body: { phone: phone.value.trim() } });
            if (r && r.ok) { step2.classList.remove('hidden'); msg.className = 'form-ok'; msg.textContent = 'Code sent. Check your Telegram app.'; }
            else { msg.className = 'form-error'; msg.textContent = `Could not send code: ${(r && r.error) || 'unknown error'}`; }
          } catch (e) { msg.className = 'form-error'; msg.textContent = e.message; }
        },
      }, 'Send code'));
      body.appendChild(step2);
      body.appendChild(msg);
    };
    await draw();
    return box;
  }

  // ------------------------------------------------------------------ new run
  function isoDate(d) { return d.toISOString().slice(0, 10); }

  async function pageNew(main, seq) {
    const meta = await getMeta(true);
    if (seq !== S.renderSeq) return;
    const maxDays = meta.max_window_days || 30;
    const presets = (Array.isArray(meta.window_presets) && meta.window_presets.length ? meta.window_presets : WINDOW_PRESETS)
      .filter((x) => x && x.id && x.days > 0 && x.limit > 0);
    const customLimit = meta.custom_limit || 500;
    // A custom range's default limit grows with its length (pipeline.custom_limit); a
    // limit the analyst typed is kept.
    const perDay = meta.custom_limit_per_day || 70;
    const limitMax = meta.custom_limit_max || 2000;
    const scaledLimit = (days) => Math.max(customLimit, Math.min(limitMax, perDay * Math.max(1, days || 1)));
    let limitTouched = false;
    const today = new Date();
    const startD = new Date(today.getTime() - 6 * 86400000);
    const topic = h('input', { type: 'text', placeholder: 'e.g. a protest, a product recall, a claim', autocomplete: 'off' });
    // Dates are UTC days; nothing after today can have posts (and platforms reject
    // searches that start in the future), so the pickers stop at today (UTC).
    const start = h('input', { type: 'date', value: isoDate(startD), max: isoDate(today) });
    const end = h('input', { type: 'date', value: isoDate(today), max: isoDate(today) });
    const limit = h('input', { type: 'number', min: '10', max: '5000', value: String(customLimit) });
    const match = h('select', { name: 'match' },
      h('option', { value: 'phrase' }, 'Exact phrase (recommended)'),
      h('option', { value: 'words' }, 'All words (broader)'));
    match.value = 'phrase';
    const winNote = h('div', { class: 'small muted' });
    const updateWin = () => {
      const days = windowDays(start.value, end.value);
      if (!limitTouched && days !== null && days >= 1) limit.value = String(scaledLimit(days));
      const todayIso = isoDate(new Date());
      const future = start.value > todayIso ? 'start' : end.value > todayIso ? 'end' : null;
      winNote.textContent = days === null ? 'Pick both dates.' :
        days < 1 ? 'End date is before start date.' :
          future === 'start' ? `The start date is after today (${todayIso}, UTC).` :
            future === 'end' ? `The end date is after today (${todayIso}, UTC); the window will end now.` :
              `${plural(days, 'day')} (maximum ${maxDays}).`;
      winNote.className = (days === null || days < 1 || days > maxDays || future === 'start') ? 'small form-error'
        : future === 'end' ? 'small warn-text' : 'small muted';
    };
    start.addEventListener('input', updateWin); end.addEventListener('input', updateWin); updateWin();

    // ---- Time window: preset radio group (last 1/3/7 days ending now, each with its own
    // per-source cap) or a custom start/end range, collapsed by default. Choosing either
    // deselects the other. Radio-group keyboard model: one tab stop, arrows move + select.
    const usable = presets.filter((x) => x.days <= maxDays);
    let win = (usable.find((x) => x.id === DEFAULT_WINDOW) || usable[usable.length - 1] || {}).id || 'custom';
    const winLine1 = h('div', { class: 'small win-what', id: 'win-what', 'aria-live': 'polite' });
    const winLine2 = h('div', { class: 'small muted win-cover' });
    const presetBtns = presets.map((x) => h('button', {
      type: 'button', role: 'radio', class: 'seg-opt', 'data-window': x.id,
      disabled: x.days > maxDays ? true : null,
      title: x.days > maxDays ? `Longer than the maximum window (${maxDays} days, Settings)` : null,
      onclick: () => setWin(x.id),
    }, presetName(x.days)));
    const radios = h('div', { class: 'seg win-seg', role: 'radiogroup', 'aria-labelledby': 'win-label', 'aria-describedby': 'win-what' }, presetBtns);
    const customPanel = h('div', { class: 'custom-range hidden', id: 'custom-range' },
      h('div', { class: 'row' },
        h('label', { class: 'field' }, h('span', null, 'Start date (UTC)'), start),
        h('label', { class: 'field' }, h('span', null, 'End date (UTC, inclusive)'), end),
        h('label', { class: 'field' }, h('span', null, 'Per-source post limit'), limit)),
      winNote);
    const customBtn = h('button', { type: 'button', class: 'link custom-toggle', 'aria-expanded': 'false', 'aria-controls': 'custom-range',
      onclick: () => setWin(win === 'custom' ? (usable.find((x) => x.id === DEFAULT_WINDOW) || usable[0] || { id: 'custom' }).id : 'custom') }, 'Custom range');
    const enabledBtns = () => presetBtns.filter((b) => !b.disabled);
    function setWin(id, focus) {
      win = id;
      const custom = id === 'custom';
      for (const b of presetBtns) {
        const on = b.getAttribute('data-window') === id;
        b.setAttribute('aria-checked', on ? 'true' : 'false');
        b.classList.toggle('on', on);
        b.tabIndex = on ? 0 : -1;
      }
      // with nothing checked (custom range), the first enabled preset keeps the tab stop
      if (custom && enabledBtns().length) enabledBtns()[0].tabIndex = 0;
      customPanel.classList.toggle('hidden', !custom);
      customBtn.setAttribute('aria-expanded', custom ? 'true' : 'false');
      customBtn.textContent = custom ? 'Use a preset instead' : 'Custom range';
      const p = presets.find((x) => x.id === id);
      const [a, b] = custom ? presetLines(null, parseInt(limit.value, 10) || customLimit, Date.now())
        : presetLines(p.days, p.limit, Date.now());
      winLine1.textContent = a; winLine2.textContent = b;
      if (focus) { const t = presetBtns.find((x) => x.getAttribute('data-window') === id); if (t) t.focus(); }
    }
    radios.addEventListener('keydown', (e) => {
      const keys = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
      const btns = enabledBtns();
      if (!btns.length) return;
      if (e.key in keys) {
        e.preventDefault();
        const i = btns.indexOf(document.activeElement);
        const next = btns[((i < 0 ? 0 : i + keys[e.key]) + btns.length) % btns.length];
        setWin(next.getAttribute('data-window'), true);
      } else if (e.key === 'Home' || e.key === 'End') {
        e.preventDefault();
        setWin(btns[e.key === 'Home' ? 0 : btns.length - 1].getAttribute('data-window'), true);
      }
    });
    limit.addEventListener('input', () => { limitTouched = true; if (win === 'custom') setWin('custom'); });
    // the scaled default limit follows the dates; refresh the summary line with it
    const onDates = () => { if (win === 'custom') setWin('custom'); };
    start.addEventListener('input', onDates); end.addEventListener('input', onDates);
    setWin(win);
    // The preset line names exact times ending 'now'; keep it current while the page is
    // open (the run itself uses the server's time at submit). Stops once the page is gone.
    const tick = setInterval(() => {
      if (!document.body.contains(winLine1)) { clearInterval(tick); return; }
      if (win !== 'custom') setWin(win);
    }, 30000);

    // ---- "Sources to scan" picker. Default: the analyst's last choice (this browser
    // only), else every source that is ready. Unready sources can still be ticked: the
    // run then records them as skipped with the reason.
    const allSrc = meta.sources || [];
    const known = new Set(allSrc.map((x) => x.name));
    let remembered = null;
    try {
      const v = JSON.parse(localStorage.getItem(SRC_PREF_KEY) || 'null');
      if (Array.isArray(v)) remembered = v.filter((n) => known.has(n));
    } catch (_e) { remembered = null; }
    if (remembered && !remembered.length) remembered = null;
    const checks = {};
    const srcCount = h('span', { class: 'small muted' });
    const srcWarn = h('div', { class: 'small' });
    const refreshSrc = () => {
      const on = allSrc.filter((x) => checks[x.name].checked);
      srcCount.textContent = `${on.length} of ${allSrc.length} selected`;
      const unready = on.filter((x) => !x.ready).map((x) => x.name);
      srcWarn.className = unready.length ? 'small warn-text' : 'small';
      srcWarn.textContent = unready.length
        ? `Not ready, will be recorded as skipped with the reason: ${unready.join(', ')}.` : '';
      if (on.length && msg.textContent.startsWith('Select at least one source')) msg.textContent = '';
      for (const x of allSrc) {
        const row = checks[x.name].closest('label');
        if (row) row.classList.toggle('selected', checks[x.name].checked);
      }
    };
    const srcList = h('div', { class: 'src-picker' });
    for (const src of allSrc) {
      const cb = h('input', { type: 'checkbox', name: 'source', value: src.name });
      cb.checked = remembered ? remembered.includes(src.name) : !!src.ready;
      cb.addEventListener('change', refreshSrc);
      checks[src.name] = cb;
      srcList.appendChild(h('label', { class: `check src-option ${src.ready ? '' : 'muted'}` }, cb,
        h('span', null, platformBadge(src.name), ' ',
          SOURCE_LABELS[src.name] ? h('span', { class: 'src-label' }, SOURCE_LABELS[src.name], ' ') : null,
          src.kind ? h('span', { class: 'chip' }, src.kind) : null, ' ',
          outcomeChip(src),
          src.ready ? null : h('div', { class: 'small' }, reasonText(src)))));
    }
    const setAll = (fn) => { for (const x of allSrc) checks[x.name].checked = fn(x); refreshSrc(); };
    const pickBtn = (label, fn) => h('button', { type: 'button', class: 'small',
      onclick: (e) => { e.preventDefault(); setAll(fn); } }, label);
    const srcTools = h('div', { class: 'row src-tools' },
      pickBtn('All', () => true), pickBtn('None', () => false), pickBtn('Ready only', (x) => !!x.ready),
      h('div', { class: 'spacer' }), srcCount);
    const msg = h('div');
    const go = h('button', { class: 'primary', type: 'submit' }, 'Start run');
    main.appendChild(pageHead('New run', 'Enter a topic, choose a time window (the last 1, 3 or 7 days, or a custom UTC date range of up to ' + maxDays + ' days), pick the sources to scan, and start. Progress and per-source status are shown live. ' +
      (meta.time_limit_s ? `Collection stops after ${Math.round(meta.time_limit_s / 60)} min (RUN_TIME_BUDGET_SECONDS in Settings); you can also stop the scan from the progress page, which discards the whole run.`
        : 'Runs have no time limit; you can stop the scan from the progress page at any time, which discards the whole run (no report is produced).')));
    const form = h('form', {
      class: 'card form-card', onsubmit: async (e) => {
        e.preventDefault();
        msg.className = 'form-error';
        const t = topic.value.trim();
        const custom = win === 'custom';
        const days = windowDays(start.value, end.value);
        if (!t) { msg.textContent = 'Enter a topic.'; return; }
        if (custom && (days === null || days < 1)) { msg.textContent = 'Choose a valid start and end date.'; return; }
        if (custom && days > maxDays) { msg.textContent = `The window is ${days} days; the maximum is ${maxDays}.`; return; }
        if (custom && start.value > isoDate(new Date())) { msg.textContent = 'The start date is in the future.'; return; }
        const sources = Object.keys(checks).filter((k) => checks[k].checked);
        if (!sources.length) { msg.textContent = 'Select at least one source to scan.'; return; }
        try { localStorage.setItem(SRC_PREF_KEY, JSON.stringify(sources)); } catch (_e) { /* private mode */ }
        const lim = parseInt(limit.value, 10);
        go.disabled = true;
        msg.className = 'muted'; msg.textContent = 'Starting…';
        try {
          const m = match.value === 'words' ? 'words' : 'phrase';
          const body = custom
            ? { topic: t, start: start.value, end: end.value, sources, limit: isFinite(lim) ? lim : customLimit, match: m }
            : { topic: t, window: win, sources, match: m };
          const r = await api('/api/runs', { method: 'POST', body });
          location.hash = `#/progress/${enc(r.run_id)}`;
        } catch (err) {
          go.disabled = false;
          msg.className = 'form-error';
          msg.textContent = err.status === 409 ? `Another run is in progress: ${err.message}` : `Could not start: ${err.message}`;
        }
      },
    },
      h('label', { class: 'field' }, h('span', null, 'Topic'), topic),
      h('div', { class: 'field win-field' },
        h('span', { class: 'field-label', id: 'win-label' }, 'Time window'),
        h('div', { class: 'row win-row' }, radios, customBtn),
        winLine1, winLine2,
        customPanel),
      h('label', { class: 'field match-field' }, h('span', null, 'Topic match'), match),
      h('div', { class: 'small muted' }, 'Multi-word topics are searched as an exact phrase unless you choose all words. Topics containing search operators (quotes, OR, -word, from:) are sent as written.'),
      h('h3', { style: { marginTop: '14px' } }, 'Sources to scan'),
      h('p', { class: 'small muted' }, 'Only the ticked sources are collected. Add keys in ',
        h('a', { href: '#/settings' }, 'Settings'), ' to make a source ready. Unticked sources are listed in the run log as not selected.'),
      srcTools,
      srcList,
      srcWarn,
      h('div', { class: 'row', style: { marginTop: '12px' } }, go),
      msg);
    main.appendChild(form);
    refreshSrc();
    topic.focus();
  }

  function windowDays(a, b) {
    if (!a || !b) return null;
    const da = Date.parse(`${a}T00:00:00Z`), db = Date.parse(`${b}T00:00:00Z`);
    if (isNaN(da) || isNaN(db)) return null;
    return Math.round((db - da) / 86400000) + 1;
  }

  // ------------------------------------------------------------------ progress
  async function pageProgress(main, id, seq) {
    main.appendChild(pageHead('Run progress', h('span', { class: 'mono' }, id)));
    const body = h('div');
    main.appendChild(body);
    let redirected = false;
    const tick = async () => {
      if (seq !== S.renderSeq) return;
      let p;
      try { p = await api(`/api/runs/${enc(id)}/progress`); } catch (e) {
        clear(body).appendChild(h('div', { class: 'banner bad' }, `Progress unavailable: ${e.message}`));
        S.timer = setTimeout(tick, 3000);
        return;
      }
      if (seq !== S.renderSeq) return;
      drawProgress(body, id, p || {});
      if (p && p.done) {
        delete S.runs[id];
        if (p.cancelled) {
          // The run was discarded; there is no run to open. Go back to the run list.
          if (!redirected) { redirected = true; S.timer = setTimeout(() => { location.hash = '#/'; }, 1800); }
          return;
        }
        if (!p.error && !redirected) { redirected = true; S.timer = setTimeout(() => { location.hash = `#/run/${enc(id)}`; }, 1500); }
        return;
      }
      S.timer = setTimeout(tick, 1500);
    };
    await tick();
  }

  // Ordered pipeline stages for the progress bar (socmint/pipeline.py). "collect" is the
  // long one; the bar fills as each later stage completes, and the active stage animates.
  const PROGRESS_STAGES = [
    ['collect', 'Collecting posts'], ['verify', 'Verifying timestamps'],
    ['narratives', 'Finding narratives'], ['coordination', 'Detecting coordination'],
    ['origin', 'Tracing origin & spread'], ['claims', 'Checking claims'],
    ['brief', 'Building the summary'], ['export', 'Exporting'], ['report', 'Writing the report'],
  ];

  function srcsForCount(p) { return p.sources || {}; }

  function progressBar(p, totalPosts) {
    let stage = String(p.stage || 'collect');
    if (stage === 'reanalyze') stage = 'narratives';      // re-analysis skips collection
    const cancelled = !!p.cancelled;
    let idx = PROGRESS_STAGES.findIndex((x) => x[0] === stage);
    if (idx < 0) idx = 0;                                   // 'starting' / unknown -> collecting
    const done = !!p.done;
    const n = PROGRESS_STAGES.length;
    const pct = cancelled ? 0 : done ? 100 : Math.round(((idx + 0.5) / n) * 100);
    const tone = cancelled ? 'warn' : p.error ? 'bad' : done ? 'ok' : '';
    const label = cancelled ? 'Scan stopped — nothing saved'
      : done ? (p.error ? 'Finished with an error' : 'Run complete')
        : PROGRESS_STAGES[idx][1];
    const detail = done ? '' : stage === 'collect' && totalPosts
      ? `${fmtNum(totalPosts)} posts so far` : `Step ${idx + 1} of ${n}`;
    const segs = PROGRESS_STAGES.map((x, i) => {
      const state = cancelled ? 'todo' : done || i < idx ? 'done' : i === idx ? 'active' : 'todo';
      return h('div', { class: `pb-seg ${state}`, title: x[1],
        style: { flex: x[0] === 'collect' ? '2 1 0' : '1 1 0' } });
    });
    if (cancelled) {
      return h('div', { class: 'progress-wrap tone-warn' },
        h('div', { class: 'pb-head' }, h('span', { class: 'pb-label' }, label)),
        h('div', { class: 'pb-track' }, segs));
    }
    return h('div', { class: `progress-wrap tone-${tone}` },
      h('div', { class: 'pb-head' },
        h('span', { class: 'pb-label' }, done ? label : `${label}…`),
        h('span', { class: 'pb-step small muted' },
          done ? '' : `${detail} · ${pct}%`)),
      h('div', { class: 'pb-track', role: 'progressbar', 'aria-valuenow': String(pct),
        'aria-valuemin': '0', 'aria-valuemax': '100',
        'aria-label': done ? label : `${label}, ${pct}%` }, segs));
  }

  function drawProgress(body, id, p) {
    const logEl = body.querySelector('.log');
    const atBottom = !logEl || (logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 4);
    clear(body);
    // Runs have no time limit by default, so the analyst ends a long scan here. Stop is a
    // hard cancel: collection halts and the whole run is discarded (no analysis, no report).
    let stopCtl = null;
    if (p.stopping && !p.done) {
      stopCtl = h('span', { class: 'chip warn' }, 'Stopping the scan…');
    } else if (p.can_stop && !p.done && !STATIC) {
      stopCtl = h('button', { class: 'btn danger small', type: 'button',
        title: 'Stop the scan now. Collection halts and the whole run is discarded — no analysis and no report are produced.' }, 'Stop scan');
      stopCtl.addEventListener('click', async () => {
        if (!window.confirm('Stop the scan now?\n\nThe whole run is discarded: collection stops at once and NO report is created. ' +
          'Nothing collected so far is kept. This cannot be undone.')) return;
        stopCtl.disabled = true;
        stopCtl.textContent = 'Stopping…';
        try {
          await api(`/api/runs/${enc(id)}/stop`, { method: 'POST', body: {} });
        } catch (err) {
          stopCtl.disabled = false;
          stopCtl.textContent = 'Stop scan';
          window.alert(`Could not stop the run: ${err.message || err}`);
        }
      });
    }
    const totalPosts = Object.values(srcsForCount(p)).reduce((a, r) => a + (Number(r.count) || 0), 0);
    body.appendChild(h('div', { class: 'card progress-card' },
      progressBar(p, totalPosts),
      h('div', { class: 'row pb-foot' },
        stopCtl || h('span'),
        h('div', { class: 'spacer' }),
        h('div', { class: 'pb-cost' }, h('span', { class: 'small muted' }, 'Running cost '),
          h('span', { class: 'cost' }, `$${Number(p.cost_usd || 0).toFixed(4)}`)))));
    if (p.error) body.appendChild(h('div', { class: 'banner bad' }, `Run error: ${p.error}`));
    const tbody = h('tbody');
    const srcs = p.sources || {};
    for (const name of Object.keys(srcs).sort()) {
      const r = srcs[name] || {};
      const states = String(r.state || 'pending').split(',').filter(Boolean);
      tbody.appendChild(h('tr', { class: r.headline || r.reason ? 'has-sub' : null }, h('td', null, platformBadge(name)),
        h('td', null, states.map((x) => [stateChip(x), ' '])),
        h('td', null, fmtNum(r.count)),
        h('td', null, typeof r.cost_usd === 'number' ? `$${r.cost_usd.toFixed(4)}` : '–')));
      // The plain-language reason gets its own full-width line (readable on a phone).
      if (r.headline || r.reason) {
        tbody.appendChild(h('tr', { class: 'sub-row' }, h('td', { colspan: '4', class: 'small reason-cell' },
          r.headline ? statusHead(r) : r.reason)));
      }
    }
    body.appendChild(h('div', { class: 'card table-wrap' }, h('h3', null, 'Sources'),
      Object.keys(srcs).length ? h('table', null,
        h('thead', null, h('tr', null, ['Source', 'State', 'Posts', 'Cost'].map((t) => h('th', null, t)))), tbody)
        : h('div', { class: 'muted' }, 'Waiting for sources to start…')));
    if (p.cancelled) {
      body.appendChild(h('div', { class: 'banner warn' },
        'You stopped the scan. The run was discarded — no posts were saved and no report was created. ',
        h('a', { href: '#/' }, 'Back to runs')));
    } else if (p.done) {
      body.appendChild(h('div', { class: `banner ${p.error ? 'warn' : ''}` },
        p.error ? 'The run finished with an error. Whatever was collected is still available: ' : 'Run complete. Opening results… ',
        h('a', { href: `#/run/${enc(id)}` }, 'Open results')));
    }
    const log = h('div', { class: 'log' }, (p.log || []).slice(-80).join('\n') || 'No log lines yet.');
    body.appendChild(h('div', { class: 'card' }, h('h3', null, 'Log'), log));
    if (atBottom) log.scrollTop = log.scrollHeight;
  }

  // ------------------------------------------------------------------ results
  function buildCtx(id, data) {
    const posts = Array.isArray(data.posts) ? data.posts : [];
    const postsById = new Map();
    posts.forEach((p) => { if (p && p.post_id) postsById.set(p.post_id, p); });
    const narratives = Array.isArray(data.narratives) ? data.narratives : [];
    const narrOf = new Map();
    const narrById = new Map();
    for (const n of narratives) {
      narrById.set(n.id, n);
      for (const pid of (n.member_post_ids || [])) {
        if (!narrOf.has(pid)) narrOf.set(pid, []);
        narrOf.get(pid).push(n.id);
      }
    }
    const run = data.run || {};
    const clusters = Array.isArray(data.clusters) ? data.clusters : [];
    const origin = Array.isArray(data.origin) ? data.origin : [];
    const claims = Array.isArray(data.claims) ? data.claims : [];
    const status = Array.isArray(data.source_status) ? data.source_status : [];
    // brief.json (run summary, ranking, coverage) is optional: older runs have none and
    // load_run gives {} then. Every view falls back to rules computed from the run files.
    const brief = (data.brief && typeof data.brief === 'object' && !Array.isArray(data.brief)) ? data.brief : {};
    const days = dayList(run.start, run.end);
    const ranks = rankNarratives(narratives, brief);
    const cov = coverageOf(status, posts, days);
    const clusterNarr = clusterNarratives(clusters, narrOf, brief);
    const narrClusters = new Map();
    const addNC = (nid, cid) => {
      if (!narrClusters.has(nid)) narrClusters.set(nid, []);
      if (!narrClusters.get(nid).includes(cid)) narrClusters.get(nid).push(cid);
    };
    Object.entries(clusterNarr).forEach(([cid, nids]) => nids.forEach((nid) => addNC(nid, cid)));
    ranks.forEach((r) => (r.cluster_ids || []).forEach((cid) => { if (clusters.some((c) => c.id === cid)) addNC(r.narrative_id, cid); }));
    const claimsByNarr = new Map();
    claims.forEach((c) => {
      const k = c.narrative_id || '';
      if (!claimsByNarr.has(k)) claimsByNarr.set(k, []);
      claimsByNarr.get(k).push(c);
    });
    return {
      id, data, posts, postsById, narratives, narrById, narrOf, clusters, origin, claims, status, run, brief, days, ranks,
      pathKind: data.path_kind || 'run',
      rankOf: new Map(ranks.map((r) => [r.narrative_id, r])),
      coverage: Array.isArray(brief.coverage) && brief.coverage.length ? brief.coverage : cov.notes,
      coverFrom: cov.from,
      clusterById: new Map(clusters.map((c) => [c.id, c])),
      clusterNarr, narrClusters, claimsByNarr,
      originByNarr: new Map(origin.map((t) => [t.narrative_id, t])),
      mainClusterIds: mainClusters(clusters, brief),
    };
  }

  function getFilter(id) {
    if (!S.filters[id]) S.filters[id] = { source: '', narrative: '', verifiedOnly: false, q: '', ids: null, idsLabel: '', sort: 'oldest', shown: 100 };
    return S.filters[id];
  }

  /** Jump to the Posts tab showing exactly these post ids (evidence for a finding). */
  function showPosts(ctx, ids, label) {
    const f = getFilter(ctx.id);
    Object.assign(f, { source: '', narrative: '', verifiedOnly: false, q: '', ids: Array.from(new Set(ids || [])), idsLabel: label || '', shown: 100 });
    const target = `#/run/${enc(ctx.id)}/posts`;
    if (location.hash === target) route(); else location.hash = target;
  }

  function pidButton(ctx, pid) {
    const p = ctx.postsById.get(pid);
    return h('button', { class: 'pid', type: 'button', title: p ? `${p.source} ${p.author_handle || ''}` : 'not in posts.jsonl', onclick: () => openPost(pid) }, pid);
  }
  function pidList(ctx, ids, max) {
    const list = ids || [];
    const lim = max || 12;
    const out = list.slice(0, lim).map((pid) => [pidButton(ctx, pid), ' ']);
    if (list.length > lim) out.push(h('button', { class: 'link small', type: 'button', onclick: () => showPosts(ctx, list, 'evidence posts') }, `+${list.length - lim} more`));
    return out;
  }
  function viewPostsBtn(ctx, ids, label, text) {
    const n = (ids || []).length;
    return h('button', { type: 'button', class: 'small', disabled: n === 0, onclick: () => showPosts(ctx, ids, label) }, text || `View ${plural(n, 'post')}`);
  }

  async function pageResults(main, id, tab, seq, sub) {
    let data;
    if (STATIC && !EMBEDDED) { main.appendChild(h('div', { class: 'banner bad' }, 'The embedded run data could not be read.')); return; }
    data = await getRun(id);
    if (seq !== S.renderSeq) return;
    const ctx = buildCtx(id, data || {});
    S.ctx = ctx;
    // #/run/<id>/narrative/<nid> is the per-narrative detail page (under Narratives).
    const detail = tab === 'narrative' && sub ? sub : null;
    const active = detail ? 'narratives' : (TABS.some(([k]) => k === tab) ? tab : 'summary');
    main.appendChild(active === 'summary' ? resultsHeader(ctx) : runStrip(ctx));
    const tabs = h('nav', { class: 'tabs mirrored', 'aria-label': 'Run sections' });
    const nMain = ctx.ranks.filter((r) => r.main).length;
    const ofTotal = (part, total) => (part && part < total ? `${part} / ${total}` : total);
    const counts = {
      sources: ctx.status.length, posts: ctx.posts.length, narratives: ofTotal(nMain, ctx.narratives.length),
      coordination: ofTotal(ctx.mainClusterIds.length, ctx.clusters.length),
      // Origin & spread opens on the main narratives' traces; their number is the count.
      origin: ctx.ranks.filter((r) => r.main && ctx.originByNarr.has(r.narrative_id)).length || ctx.origin.length,
      claims: ctx.claims.length,
    };
    updateRunNav(ctx, active, counts);
    for (const [k, label] of TABS) {
      tabs.appendChild(h('a', { href: `#/run/${enc(id)}/${k}`, class: k === active ? 'active' : null }, label,
        counts[k] !== undefined ? h('span', { class: 'count' }, countLabel(counts[k])) : null));
    }
    main.appendChild(tabs);
    if (active !== 'summary' && !detail) {
      const activeLabel = (TABS.find(([k]) => k === active) || [])[1];
      main.appendChild(h('div', { class: 'section-title' }, h('h1', null, activeLabel),
        counts[active] !== undefined ? h('span', { class: 'count' }, countLabel(counts[active])) : null));
    }
    const view = h('section', { class: `view view-${detail ? 'narrative' : active}` });
    main.appendChild(view);
    if (detail) { pageNarrative(view, ctx, detail); return; }
    const fn = { summary: tabSummary, sources: tabSources, posts: tabPosts, narratives: tabNarratives, coordination: tabCoordination, origin: tabOrigin, claims: tabClaims, exports: tabExports }[active];
    fn(view, ctx, sub);
  }

  // Topic-match mode recorded in run.json options; runs made before the option show nothing.
  // BEGIN matchLabel
  function matchLabel(r) {
    const m = r && r.options ? r.options.match : null;
    return m === 'phrase' ? 'exact phrase' : m === 'words' ? 'all words' : '';
  }
  // END matchLabel

  function durationLabel(sec) {
    if (typeof sec !== 'number' || !isFinite(sec) || sec <= 0) return '';
    const m = Math.floor(sec / 60);
    const s2 = Math.round(sec % 60);
    return m ? `${m} min ${s2} s` : `${s2} s`;
  }

  /** "analysis re-run 19 Nov 2026 17:46 UTC (took 3 min 37 s)" when run.json records a
   *  re-analysis (options.reanalyzed_at, timings.reanalyze_total); '' otherwise. The
   *  "took" beside the run date is the original collection run's duration. */
  function reanalysedLabel(r) {
    const at = r && r.options && r.options.reanalyzed_at;
    if (!at) return '';
    const took = durationLabel((r.timings || {}).reanalyze_total);
    return `analysis re-run ${fmtTs(at)}${took ? ` (took ${took})` : ''}`;
  }

  /** Full run header: the Summary page's title block. */
  function resultsHeader(ctx) {
    const r = ctx.run;
    const srcs = (r.sources || []).map(pname);
    const took = durationLabel((r.timings || {}).total);
    const card = h('div', { class: 'card run-header' },
      h('div', { class: 'eyebrow' }, STATIC ? 'Standalone report' : 'Run summary', ' · ', h('span', { class: 'mono rid' }, ctx.id)),
      h('div', { class: 'row head-row' }, h('h1', { dir: 'auto' }, `“${r.topic || ctx.id}”`),
        h('span', { class: 'h1-window' }, windowLabel(r.start, r.end)),
        h('span', { class: `chip ${r.status === 'complete' ? 'ok' : r.status === 'failed' ? 'bad' : 'run'}` }, r.status || 'unknown'),
        r.replay ? h('span', { class: 'chip' }, 'replay') : null,
        h('div', { class: 'spacer' }),
        h('a', { class: 'btn small', href: `#/run/${enc(ctx.id)}/exports` }, 'Export JSON / CSV / HTML'),
        canReanalyze({ path_kind: ctx.pathKind, status: r.status }) ? reanalyzeButton(ctx.id) : null,
        canDelete({ path_kind: ctx.pathKind, status: r.status })
          ? deleteButton(ctx.id, r.topic, () => { location.hash = '#/runs'; }) : null),
      h('div', { class: 'meta' },
        h('span', null, h('b', null, fmtNum(ctx.posts.length)), ' posts', srcs.length ? ` from ${joinNames(srcs)}` : ''),
        h('span', null, 'Window ', h('b', null, windowBounds(r)), ' (UTC)'),
        windowPresetLabel(r) ? h('span', { class: 'window-preset' }, windowPresetLabel(r)) : null,
        (r.options || {}).collection_stopped_by_analyst ? h('span', { class: 'chip warn',
          title: 'The analyst stopped collection before every source finished; see each source status for how far it got.' },
          'collection stopped early') : null,
        matchLabel(r) ? h('span', { class: 'match-mode' }, 'Topic match: ', h('b', null, matchLabel(r))) : null,
        r.created_at ? h('span', null, 'Run ', h('b', null, fmtTs(r.created_at)), took ? ` · took ${took}` : '') : null,
        reanalysedLabel(r) ? h('span', { class: 'reanalysed', title: 'The analysis (narratives, coordination, spread, claims, summary) was re-run on the stored posts; collection was not repeated.' },
          reanalysedLabel(r)) : null),
      h('div', { class: 'sample-note' }, 'Findings describe the collected sample only. Every figure links to the posts behind it.'));
    if (r.error) card.appendChild(h('div', { class: 'banner bad' }, `Run error: ${r.error}`));
    const notOk = ctx.status.filter((x) => x.state !== 'ok');
    // Problems that limit the results (quota, rate limit, account check, error, setup, a
    // stop), one plain line per source, main reason first.
    const SERIOUS = ['account', 'quota', 'rate_limit', 'error', 'setup', 'stopped'];
    const bySrc = new Map();
    for (const st of ctx.status) { if (!bySrc.has(st.source)) bySrc.set(st.source, []); bySrc.get(st.source).push(st); }
    const serious = Array.from(bySrc.values()).map(worstStatus).filter((w) => w && SERIOUS.includes(w.headline_kind));
    if (serious.length) {
      card.appendChild(h('div', { class: 'head-problems' }, serious.map((w) =>
        h('div', { class: 'head-problem' }, platformChip(w.source), statusHead(w, { details: false })))));
    }
    if (notOk.length) {
      card.appendChild(h('div', { class: 'small muted head-foot' },
        `${plural(notOk.length, 'source query')} did not complete normally. `,
        h('a', { href: `#/run/${enc(ctx.id)}/sources` }, 'See source status for the reasons.')));
    }
    return card;
  }

  /** Slim run line above every other section (topic, window, status, way back to Summary). */
  function runStrip(ctx) {
    const r = ctx.run;
    return h('div', { class: 'run-strip' },
      h('a', { class: 'topic', href: `#/run/${enc(ctx.id)}/summary`, dir: 'auto', title: 'Run summary' }, `“${r.topic || ctx.id}”`),
      h('span', { class: 'sep', 'aria-hidden': 'true' }),
      h('span', { class: 'tnum' }, windowLabel(r.start, r.end)),
      h('span', { class: 'sep', 'aria-hidden': 'true' }),
      h('span', { class: 'tnum' }, `${fmtNum(ctx.posts.length)} posts`),
      h('span', { class: `chip ${r.status === 'complete' ? 'ok' : r.status === 'failed' ? 'bad' : 'run'}` }, r.status || 'unknown'),
      STATIC ? h('span', { class: 'chip' }, 'standalone report') : null);
  }

  // ---- Sources
  function tabSources(view, ctx) {
    if (!ctx.status.length) {
      view.appendChild(h('div', { class: 'banner warn' }, 'No source status was recorded for this run, so empty results cannot be explained. Treat this run as incomplete.'));
      return;
    }
    view.appendChild(h('p', { class: 'muted' },
      'One row per source × query. Every empty source states why: completed with zero results, out of coverage, skipped (credentials or capability), partial (with the last cursor), or failed (with the error).'));
    const legend = h('div', { class: 'legend' },
      h('span', null, h('span', { class: 'chip ok' }, 'ok')),
      h('span', null, h('span', { class: 'chip warn' }, 'partial / completed_zero / out_of_coverage')),
      h('span', null, h('span', { class: 'chip skip' }, 'skipped_*')),
      h('span', null, h('span', { class: 'chip bad' }, 'failed')));
    view.appendChild(legend);
    // At a glance: one plain line per source, problems first.
    const bySrc = new Map();
    for (const st of ctx.status) { if (!bySrc.has(st.source)) bySrc.set(st.source, []); bySrc.get(st.source).push(st); }
    const glance = Array.from(bySrc.entries()).map(([src, list]) => {
      const w = worstStatus(list);
      const total = list.reduce((a, x) => a + (Number(x.count) || 0), 0);
      return { src, w, total };
    }).sort((a, b) => HEAD_ORDER.indexOf((a.w || {}).headline_kind) - HEAD_ORDER.indexOf((b.w || {}).headline_kind));
    view.appendChild(h('section', { class: 'card glance' }, h('h3', null, 'At a glance'),
      h('div', { class: 'glance-list' }, glance.map((g) => h('div', { class: 'glance-row' },
        platformChip(g.src),
        h('span', { class: 'glance-count' }, `${fmtNum(g.total)} posts`),
        g.w && isProblem(g.w.headline_kind) ? statusHead(g.w, { details: false })
          : h('div', { class: 'status-head tone-ok' }, h('span', { class: 'sh-text' }, g.total ? 'Collected normally.' : (g.w && g.w.headline) || 'No posts found.')))))));
    const tbody = h('tbody');
    for (const st of ctx.status) {
      const cls = stateClass(st.state);
      tbody.appendChild(h('tr', { class: `state-${cls || 'skip'}` },
        h('td', null, platformBadge(st.source)),
        h('td', { class: 'small query-cell' }, st.query || ''),
        h('td', null, stateChip(st.state)),
        h('td', { class: 'nowrap' }, fmtNum(st.count)),
        h('td', { class: 'small reason-cell' }, st.headline ? statusHead(st) : st.reason ? longText(st.reason, 260) : (st.state === 'ok' ? '' : h('span', { class: 'form-error' }, 'no reason recorded')),
          st.date_filtered === false ? h('div', { class: 'ts-note' }, 'Warning: this source could not filter by date on the server; results were bounded by platform timestamps afterwards and may be incomplete for the window.') : null),
        h('td', { class: 'mono small cursor-cell' }, st.cursor || ''),
        h('td', { class: 'small' }, st.date_filtered === false ? h('span', { class: 'chip warn' }, 'no') : 'yes'),
        h('td', { class: 'small nowrap' }, `${fmtNum(st.requests)} req.`,
          st.quota_units !== null && st.quota_units !== undefined ? h('div', { class: 'muted' }, `${st.quota_units} quota units`) : null,
          h('div', { class: 'muted' }, typeof st.cost_usd === 'number' ? `$${st.cost_usd.toFixed(4)}` : 'no cost')),
        h('td', { class: 'small nowrap', title: `${fmtTs(st.started_at)} → ${fmtTs(st.finished_at)}` }, shortTs(st.started_at), h('br'), shortTs(st.finished_at))));
    }
    view.appendChild(h('div', { class: 'card table-wrap' }, h('table', { class: 'tbl-sources' },
      h('thead', null, h('tr', null, ['Source', 'Query', 'State', 'Posts', 'Reason', 'Last cursor', 'Date filter', 'Requests / cost', 'Started / finished (UTC)'].map((t) => h('th', null, t)))),
      tbody)));
  }

  // ---- Posts
  function tabPosts(view, ctx) {
    const f = getFilter(ctx.id);
    const sources = Array.from(new Set(ctx.posts.map((p) => p.source))).sort();
    const srcSel = h('select', null, h('option', { value: '' }, 'All sources'), sources.map((x) => h('option', { value: x }, x)));
    srcSel.value = f.source;
    const narrSel = h('select', null, h('option', { value: '' }, 'All narratives'), h('option', { value: '__none' }, '(no narrative)'),
      ctx.narratives.map((n) => h('option', { value: n.id }, `${n.label || n.id} (${(n.member_post_ids || []).length})`)));
    narrSel.value = f.narrative;
    const vOnly = h('input', { type: 'checkbox' }); vOnly.checked = f.verifiedOnly;
    const q = h('input', { type: 'search', placeholder: 'Search text, author…' }); q.value = f.q;
    const sortSel = h('select', null, h('option', { value: 'oldest' }, 'Oldest first'), h('option', { value: 'newest' }, 'Newest first'), h('option', { value: 'engagement' }, 'Most engagement'));
    sortSel.value = f.sort;
    const idsBar = h('div');
    const list = h('div');
    const info = h('div', { class: 'small muted' });

    const apply = () => {
      f.source = srcSel.value; f.narrative = narrSel.value; f.verifiedOnly = vOnly.checked; f.q = q.value; f.sort = sortSel.value;
      drawList();
    };
    let deb = null;
    q.addEventListener('input', () => { clearTimeout(deb); deb = setTimeout(() => { f.shown = 100; apply(); }, 200); });
    [srcSel, narrSel, vOnly, sortSel].forEach((el) => el.addEventListener('change', () => { f.shown = 100; apply(); }));

    function drawList() {
      clear(idsBar);
      if (f.ids) {
        idsBar.appendChild(h('div', { class: 'banner' }, `Showing ${plural(f.ids.length, 'evidence post')}${f.idsLabel ? ` for: ${f.idsLabel}` : ''}. `,
          h('button', { class: 'link', type: 'button', onclick: () => { f.ids = null; f.idsLabel = ''; drawList(); } }, 'Show all posts')));
      }
      let rows = f.ids ? f.ids.map((pid) => ctx.postsById.get(pid) || { post_id: pid, _missing: true }) : ctx.posts.slice();
      const needle = f.q.trim().toLowerCase();
      rows = rows.filter((p) => {
        if (p._missing) return !f.source && !f.narrative && !f.verifiedOnly && !needle;
        if (f.source && p.source !== f.source) return false;
        if (f.narrative === '__none' && ctx.narrOf.has(p.post_id)) return false;
        if (f.narrative && f.narrative !== '__none' && !(ctx.narrOf.get(p.post_id) || []).includes(f.narrative)) return false;
        if (f.verifiedOnly && !analysable(p)) return false;
        if (needle) {
          const hay = `${p.text || ''} ${p.author_handle || ''} ${p.author_name || ''} ${p.post_id}`.toLowerCase();
          if (!hay.includes(needle)) return false;
        }
        return true;
      });
      const t = (p) => (analysable(p) || p.timestamp_verified) && p.published_at ? Date.parse(p.published_at) : NaN;
      const eng = (p) => { const m = p.metrics || {}; return ['likes', 'reposts', 'replies', 'quotes', 'shares', 'comments', 'forwards', 'reactions'].reduce((a, k) => a + (m[k] || 0), 0); };
      if (f.sort === 'engagement') rows.sort((a, b) => eng(b) - eng(a));
      else {
        const dir = f.sort === 'newest' ? -1 : 1;
        // Posts without a verified timestamp always sort last; they have no reliable time.
        rows.sort((a, b) => {
          const ta = t(a), tb = t(b);
          if (isNaN(ta) && isNaN(tb)) return 0;
          if (isNaN(ta)) return 1;
          if (isNaN(tb)) return -1;
          return dir * (ta - tb);
        });
      }
      info.textContent = `Showing ${Math.min(rows.length, f.shown)} of ${rows.length} matching posts (${ctx.posts.length} collected).`;
      clear(list);
      if (!rows.length) list.appendChild(h('div', { class: 'card muted' }, ctx.posts.length ? 'No posts match these filters.' : 'No posts were collected in this run. See the Sources tab for why each source returned nothing.'));
      rows.slice(0, f.shown).forEach((p) => list.appendChild(p._missing ? missingPost(p.post_id) : postCard(p, ctx)));
      if (rows.length > f.shown) {
        list.appendChild(h('button', { type: 'button', onclick: () => { f.shown += 100; drawList(); } }, `Show more (${rows.length - f.shown} remaining)`));
      }
    }

    view.appendChild(h('div', { class: 'filters' }, srcSel, narrSel, q, sortSel,
      h('label', { class: 'check' }, vOnly, h('span', null, 'Verified in-window timestamps only'))));
    view.appendChild(idsBar);
    view.appendChild(info);
    view.appendChild(list);
    drawList();
  }

  function missingPost(pid) {
    return h('div', { class: 'post' }, h('span', { class: 'mono' }, pid), ' ',
      h('span', { class: 'muted' }, 'is referenced by a finding but is not present in posts.jsonl.'));
  }

  function metricsEl(m) {
    const out = h('span', { class: 'metrics' });
    const labels = { views: 'views', likes: 'likes', replies: 'replies', reposts: 'reposts', quotes: 'quotes', shares: 'shares', comments: 'comments', forwards: 'forwards', reactions: 'reactions' };
    for (const [k, label] of Object.entries(labels)) {
      if (m && typeof m[k] === 'number') out.appendChild(h('span', null, h('b', null, fmtNum(m[k])), ` ${label}`));
    }
    if (!out.childNodes.length) out.appendChild(h('span', null, 'no metrics exposed'));
    return out;
  }

  function mediaEl(media) {
    const box = h('div', { class: 'media' });
    for (const m of (media || [])) {
      const thumb = safeUrl(m.thumbnail) || ((m.type === 'photo' || m.type === 'image') ? safeUrl(m.url) : null);
      const target = safeUrl(m.url) || thumb;
      if (thumb) {
        const img = h('img', { src: thumb, loading: 'lazy', referrerpolicy: 'no-referrer', alt: `${m.type || 'media'} attachment` });
        img.addEventListener('error', () => { (img.parentElement && img.parentElement.tagName === 'A' ? img.parentElement : img).classList.add('hidden'); });
        box.appendChild(target ? h('a', { href: target, target: '_blank', rel: 'noopener noreferrer' }, img) : img);
      } else if (target) {
        box.appendChild(h('a', { class: 'vid', href: target, target: '_blank', rel: 'noopener noreferrer' }, `${m.type || 'media'} ↗`));
      }
    }
    return box.childNodes.length ? box : null;
  }

  function relationsEl(ctx, rels) {
    if (!rels || !rels.length) return null;
    return h('ul', { class: 'relations' }, rels.map((r) => h('li', null,
      h('span', { class: 'badge direct' }, `direct · ${r.type}`), ' ',
      r.target_post_id ? (ctx.postsById.has(r.target_post_id) ? pidButton(ctx, r.target_post_id) : h('span', { class: 'mono small' }, `${r.target_post_id} (not collected)`)) : null, ' ',
      r.target_handle ? h('span', null, r.target_handle, ' ') : null,
      r.target_url ? extLink(r.target_url) : null)));
  }

  function textEl(p, full) {
    const t = p.text || '';
    const el = h('div', { class: 'text', dir: 'auto' }, t || h('span', { class: 'muted' }, '(no text)'));
    if (p.lang && /^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{1,8})*$/.test(p.lang)) el.setAttribute('lang', p.lang);
    if (full || t.length < 420) return el;
    el.classList.add('clamped');
    const btn = h('button', { class: 'link small', type: 'button', onclick: () => { const c = el.classList.toggle('clamped'); btn.textContent = c ? 'Show full text' : 'Show less'; } }, 'Show full text');
    return h('div', null, el, btn);
  }

  function timestampEl(p) {
    const wrap = h('span', null);
    if (p.published_at) wrap.appendChild(h('span', { title: `published_at ${p.published_at}` }, fmtTs(p.published_at)));
    else wrap.appendChild(h('span', { class: 'muted' }, 'no publication time'));
    if (!p.timestamp_verified) {
      wrap.appendChild(document.createTextNode(' '));
      wrap.appendChild(h('span', { class: 'badge unverified', title: p.timestamp_note || 'timestamp not verified' }, 'unverified timestamp'));
    } else if (p.in_window === false) {
      wrap.appendChild(document.createTextNode(' '));
      wrap.appendChild(h('span', { class: 'badge unverified' }, 'outside window'));
    }
    return wrap;
  }

  function postCard(p, ctx, opts) {
    const o = opts || {};
    const narrs = (ctx.narrOf.get(p.post_id) || []).map((nid) => ctx.narrById.get(nid)).filter(Boolean);
    const head = h('div', { class: 'head' },
      platformBadge(p.source),
      p.kind && p.kind !== 'post' ? h('span', { class: 'chip' }, p.kind) : null,
      h('span', { class: 'author', dir: 'auto' }, p.author_handle || p.author_name || p.author_id || 'unknown author'),
      p.author_name && p.author_handle ? h('span', { class: 'muted', dir: 'auto' }, p.author_name) : null,
      p.author_created_at ? h('span', { class: 'small muted' }, `account created ${fmtDay(p.author_created_at)}`) : null,
      h('span', { class: 'spacer' }),
      timestampEl(p));
    const card = h('article', { class: 'post' }, head);
    if (!p.timestamp_verified && p.timestamp_note && !o.compact) card.appendChild(h('div', { class: 'ts-note' }, p.timestamp_note));
    card.appendChild(textEl(p, false));
    if (!o.compact) {
      const media = mediaEl(p.media);
      if (media) card.appendChild(media);
    }
    const raw = rawHref(ctx.id, p.raw_ref);
    card.appendChild(h('div', { class: 'foot' },
      metricsEl(p.metrics),
      p.url ? extLink(p.url, 'Open on platform ↗') : h('span', null, 'no direct link'),
      raw ? h('a', { href: raw, target: '_blank', rel: 'noopener noreferrer' }, 'raw payload') : null,
      h('button', { class: 'pid', type: 'button', onclick: () => openPost(p.post_id) }, p.post_id),
      narrs.map((n) => h('span', { class: 'kw', title: n.claim_summary || '' }, n.label || n.id)),
      p.lang ? h('span', null, p.lang) : null));
    if (!o.compact) {
      const rel = relationsEl(ctx, p.relations);
      if (rel) card.appendChild(rel);
    }
    return card;
  }

  // ---- Post detail modal
  function openPost(pid) {
    const ctx = S.ctx;
    if (!ctx) return;
    const p = ctx.postsById.get(pid);
    const panel = h('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true', 'aria-label': `Post ${pid}` });
    panel.appendChild(h('button', { class: 'close', type: 'button', onclick: closeModal }, 'Close ✕'));
    panel.appendChild(h('h2', null, 'Post detail'));
    if (!p) {
      panel.appendChild(missingPost(pid));
    } else {
      const full = postCard(p, ctx);
      full.querySelector('.text.clamped') && full.querySelector('.text.clamped').classList.remove('clamped');
      panel.appendChild(full);
      const ad = p.ad_meta || null;
      const rows = [
        ['post_id', p.post_id], ['source', p.source], ['kind', p.kind], ['author_id', p.author_id],
        ['author_handle', p.author_handle], ['author_name', p.author_name], ['author_created_at', p.author_created_at],
        ['published_at (platform, UTC)', p.published_at], ['timestamp_verified', String(!!p.timestamp_verified)],
        ['timestamp_note', p.timestamp_note], ['in_window', String(p.in_window !== false)],
        ['collected_at', p.collected_at], ['lang', p.lang], ['query', p.query], ['raw_ref', p.raw_ref],
        ['duplicate_of', p.duplicate_of],
      ];
      if (ad) {
        rows.push(['ad run', `${ad.run_start || '?'} → ${ad.run_end || '?'}`], ['advertiser_id', ad.advertiser_id],
          ['regions', (ad.regions || []).join(', ')], ['impressions', ad.impressions_range]);
      }
      const dl = h('dl', { class: 'kv' });
      for (const [k, v] of rows) {
        if (v === null || v === undefined || v === '') continue;
        dl.appendChild(h('dt', null, k)); dl.appendChild(h('dd', null, String(v)));
      }
      if (ad && ad.landing_url) { dl.appendChild(h('dt', null, 'landing_url')); dl.appendChild(h('dd', null, extLink(ad.landing_url))); }
      if (p.url) { dl.appendChild(h('dt', null, 'url')); dl.appendChild(h('dd', null, extLink(p.url))); }
      panel.appendChild(h('div', { class: 'card' }, dl));

      // Where this post is used as evidence.
      const uses = [];
      (ctx.narrOf.get(pid) || []).forEach((nid) => { const n = ctx.narrById.get(nid); uses.push(`Narrative member: ${n ? n.label : nid}`); });
      ctx.clusters.forEach((c) => {
        if ((c.supporting_post_ids || []).includes(pid)) uses.push(`Coordination group ${c.id}: supporting post`);
        (c.signals || []).forEach((sg) => { if ((sg.evidence_post_ids || []).includes(pid)) uses.push(`Coordination group ${c.id}: evidence for ${SIGNAL_LABELS[sg.type] || sg.type}`); });
      });
      ctx.origin.forEach((tr) => {
        if ((tr.earliest_post_ids || []).includes(pid)) uses.push(`Earliest observed in collected data for narrative ${labelOf(ctx, tr.narrative_id)}`);
        const ne = (tr.edges || []).filter((e) => e.from === pid || e.to === pid).length;
        if (ne) uses.push(`${plural(ne, 'spread edge')} in narrative ${labelOf(ctx, tr.narrative_id)}`);
      });
      ctx.claims.forEach((c) => {
        STANCES.forEach((st) => { if (((c.stance_post_ids || {})[st] || []).includes(pid)) uses.push(`Claim "${c.text}": stance ${st}`); });
      });
      panel.appendChild(h('div', { class: 'card' }, h('h3', null, 'Used as evidence in'),
        uses.length ? h('ul', null, uses.map((u) => h('li', null, u))) : h('div', { class: 'muted' }, 'Not referenced by any finding.')));
    }
    clear(shell.modal).appendChild(panel);
    shell.modal.classList.remove('hidden');
    panel.querySelector('.close').focus();
  }
  function closeModal() { if (shell.modal) { shell.modal.classList.add('hidden'); clear(shell.modal); } }
  function labelOf(ctx, nid) { const n = ctx.narrById.get(nid); return n ? (n.label || nid) : (nid || '–'); }

  // ------------------------------------------------------------------ view helpers (pure)
  // No DOM use inside this block: tests/test_web_views.py runs it under node.
  // BEGIN viewHelpers
  const PLATFORM_NAMES = { x: 'X', telegram: 'Telegram', youtube: 'YouTube', linkedin: 'LinkedIn', linkedin_ads: 'LinkedIn Ads' };
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  function pname(src) { return PLATFORM_NAMES[src] || String(src || '?'); }

  /** 'YYYY-MM-DD' days from start to end inclusive (UTC); at most 400. */
  function dayList(start, end) {
    const a = Date.parse(`${String(start || '').slice(0, 10)}T00:00:00Z`);
    const b = Date.parse(`${String(end || '').slice(0, 10)}T00:00:00Z`);
    if (isNaN(a) || isNaN(b) || b < a) return [];
    const out = [];
    for (let t = a; t <= b && out.length < 400; t += 86400000) out.push(new Date(t).toISOString().slice(0, 10));
    return out;
  }
  /** UTC day of an ISO time, or null. */
  function dayOf(iso) {
    const t = Date.parse(iso || '');
    return isNaN(t) ? null : new Date(t).toISOString().slice(0, 10);
  }
  /** '16 Nov'; the year is added when it differs from refYear. */
  function shortDay(iso, refYear) {
    const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso || ''));
    if (!m) return '–';
    const base = `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]}`;
    return refYear && String(refYear) !== m[1] ? `${base} ${m[1]}` : base;
  }
  /** '16 Nov · 23:30' (UTC). */
  function shortTs(iso) {
    const t = Date.parse(iso || '');
    if (isNaN(t)) return '–';
    const z = new Date(t).toISOString();
    return `${shortDay(z)} · ${z.slice(11, 16)}`;
  }
  /** New Run window presets (owner decision 2026-11-20). Mirrors pipeline.WINDOW_PRESETS
   *  and PRESET_LIMITS; the page uses /api/meta's values and falls back to these. */
  const WINDOW_PRESETS = [{ id: '1d', days: 1, limit: 150 }, { id: '3d', days: 3, limit: 300 }, { id: '7d', days: 7, limit: 500 }];
  const DEFAULT_WINDOW = '3d';
  function presetName(days) { return days === 1 ? '1 day' : `${days} days`; }
  /** The exact window a preset would cover if started at nowMs: last N days ending now (UTC). */
  function presetSpan(days, nowMs) {
    const end = Math.floor(nowMs / 1000) * 1000;
    return { start: new Date(end - days * 86400000).toISOString().replace('.000Z', 'Z'),
      end: new Date(end).toISOString().replace('.000Z', 'Z') };
  }
  /** Two short lines under the window choice: [what the choice is, what each source covers].
   *  days = null for a custom range. Collection behaviour (socmint/collect): X and YouTube
   *  search every UTC day of the window separately; Telegram and LinkedIn posts walk newest
   *  first and stop at the cap; LinkedIn Ads lists ads running in the range. */
  function presetLines(days, limit, nowMs) {
    const cap = `up to ${Number(limit).toLocaleString('en-US')} posts per source`;
    let what;
    if (days) {
      const sp = presetSpan(days, nowMs);
      what = `Last ${days === 1 ? '24 hours' : `${days} days`}: ${shortTs(sp.start)} → ${shortTs(sp.end)} UTC, ${cap}.`;
    } else {
      what = `Whole UTC days from the start date to the end date, ${cap}.`;
    }
    const xy = days === 1 ? 'the whole 24 hours' : 'every day, searched day by day';
    return [what, `X and YouTube: ${xy}. Telegram and LinkedIn posts: most recent first, older posts only if the cap allows.`];
  }
  /** Run header label: '3-day window · up to 300 posts per source', 'Custom range · up to
   *  500 posts per source', or '' for runs saved before presets existed. */
  function windowPresetLabel(r) {
    const o = (r && r.options) || {};
    const lim = Number(o.limit);
    const cap = isFinite(lim) && lim > 0 ? ` · up to ${lim.toLocaleString('en-US')} posts per source` : '';
    const m = /^(\d+)d$/.exec(String(o.window_preset || ''));
    if (m) return `${m[1]}-day window${cap}`;
    if (o.window_preset === 'custom') return `Custom range${cap}`;
    return '';
  }
  /** The run window for the header: exact times for a preset run, dates otherwise. */
  function windowBounds(r) {
    const o = (r && r.options) || {};
    if (/^\d+d$/.test(String(o.window_preset || '')) && o.window_start_at && o.window_end_at) {
      return `${shortTs(o.window_start_at)} → ${shortTs(o.window_end_at)}`;
    }
    return windowLabel(r && r.start, r && r.end);
  }
  /** '13–19 Nov 2026', '28 Oct – 3 Nov 2026', '28 Dec 2025 – 3 Jan 2026'. */
  function windowLabel(start, end) {
    const a = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(start || ''));
    const b = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(end || ''));
    if (!a || !b) return `${start || '?'} – ${end || '?'}`;
    if (a[1] === b[1] && a[2] === b[2]) return `${Number(a[3])}–${Number(b[3])} ${MONTHS[Number(b[2]) - 1]} ${b[1]}`;
    if (a[1] === b[1]) return `${shortDay(start)} – ${shortDay(end)} ${b[1]}`;
    return `${shortDay(start)} ${a[1]} – ${shortDay(end)} ${b[1]}`;
  }
  /** 950, 3.5k, 209k, 1.2M. */
  function compactNum(n) {
    if (typeof n !== 'number' || !isFinite(n)) return '–';
    const a = Math.abs(n);
    const f = (v, u) => `${Math.abs(v) >= 10 ? Math.round(v) : Math.round(v * 10) / 10}${u}`;
    if (a >= 999500) return f(n / 1e6, 'M');
    if (a >= 1000) return f(n / 1e3, 'k');
    return String(Math.round(n));
  }
  /** Time after the first platform: '+15 min', '+22 h', '+3 d'. */
  function gapLabel(sec) {
    if (typeof sec !== 'number' || !isFinite(sec) || sec < 0) return '';
    if (sec < 3600) return `+${Math.max(1, Math.round(sec / 60))} min`;
    if (sec < 172800) return `+${Math.round(sec / 3600)} h`;
    return `+${Math.round(sec / 86400)} d`;
  }
  function truncate(text, max) {
    const s = String(text || '');
    return s.length > max ? `${s.slice(0, max - 1).trimEnd()}…` : s;
  }
  /** First non-empty line (a video title, a post's opening), at most max chars. */
  function firstLine(text, max) {
    const line = String(text || '').split('\n').map((x) => x.trim()).find(Boolean) || '';
    return truncate(line, max || 140);
  }
  /** 'A', 'A and B', 'A, B and C'. */
  function joinNames(list) {
    const l = (list || []).filter(Boolean);
    if (l.length < 2) return l.join('');
    return `${l.slice(0, -1).join(', ')} and ${l[l.length - 1]}`;
  }
  /** Pipeline notes in square brackets ("[Method: ...]") are split off the reasoning text.
   *  Citation markers like "[4]" stay in the text. */
  function splitRationale(text) {
    const notes = [];
    const body = String(text || '').replace(/\s*\[([^[\]]{3,})\]/g, (_m, inner) => { notes.push(inner.trim()); return ''; }).trim();
    return { body, notes };
  }
  /** First sentence, at most max chars. Initials ("U.S.") and common abbreviations do not end it. */
  function firstSentence(text, max) {
    const s = String(text || '').replace(/\s+/g, ' ').trim();
    const lim = max || 220;
    const re = /[.!?](?=\s+["“'([]?[A-Z0-9]|$)/g;
    let m;
    let end = -1;
    while ((m = re.exec(s))) {
      const word = s.slice(s.lastIndexOf(' ', m.index) + 1, m.index + 1);
      if (/^(?:[A-Za-z]\.)+$/.test(word) || /^(?:Mr|Mrs|Ms|Dr|St|Inc|Corp|Co|No|vs|Jr|Sr)\.$/i.test(word)) continue;
      end = m.index + 1;
      break;
    }
    let out = end > 0 ? s.slice(0, end) : s;
    if (out.length > lim) {
      const cut = out.lastIndexOf(' ', lim - 1);
      out = `${out.slice(0, cut > lim * 0.6 ? cut : lim - 1)}…`;
    }
    return out;
  }
  /** One sentence an analyst reads first: bottom_line, else the rationale's first sentence. */
  function claimBottomLine(c) {
    if (c && c.bottom_line) return String(c.bottom_line);
    return firstSentence(splitRationale(c && c.rationale).body);
  }
  /** Pipeline notes for "How this was checked": method_notes, else the bracketed notes. */
  function claimMethodNotes(c) {
    if (c && c.method_notes) return String(c.method_notes);
    return splitRationale(c && c.rationale).notes.join(' ');
  }
  /** Evidence for display: supports, then contradicts, then context (old events after the
   *  same kind). Items flagged "unrelated" (search misses) are never shown up front. */
  function arrangeEvidence(list, max) {
    const ev = (Array.isArray(list) ? list : []).filter(Boolean);
    const order = { supports: 0, contradicts: 1, context: 2 };
    const rank = (e) => (e.supports in order ? order[e.supports] : 2) + (e.flag === 'old_event' ? 0.5 : 0);
    const usable = ev.filter((e) => e.flag !== 'unrelated').map((e, i) => [e, i])
      .sort((a, b) => rank(a[0]) - rank(b[0]) || a[1] - b[1]).map(([e]) => e);
    const unrelated = ev.filter((e) => e.flag === 'unrelated');
    const lim = typeof max === 'number' ? max : 3;
    return {
      shown: usable.slice(0, lim),
      rest: usable.slice(lim).concat(unrelated),
      allUnrelated: ev.length > 0 && unrelated.length === ev.length,
    };
  }
  const SIGNAL_SHORT = {
    near_duplicate_text: 'Same text', near_simultaneous_posting: 'Posted together',
    similar_creation_dates: 'Created together', reused_image: 'Same image',
    amplification_loop: 'Amplify each other',
  };
  const SIGNAL_TYPES = Object.keys(SIGNAL_SHORT);
  /** The one number that carries a signal, parsed from its rule-built detail text
   *  (socmint/analyze/coordination.py); falls back to the evidence post count. */
  function signalKey(sg) {
    const d = String((sg && sg.detail) || '');
    const type = sg && sg.type;
    const n = ((sg && sg.evidence_post_ids) || []).length;
    const label = SIGNAL_SHORT[type] || String(type || 'signal');
    let m;
    if (type === 'near_duplicate_text' && (m = /^([\d,]+) posts by ([\d,]+) accounts.*?mean ([\d.]+)/.exec(d))) {
      return { label, key: `${m[1]} posts`, sub: `by ${m[2]} accounts, near-identical (similarity ${m[3]})` };
    }
    if (type === 'near_simultaneous_posting' && (m = /median lag ([\d.]+ ?(?:s|min|h|d))\b/.exec(d))) {
      const pairs = /^([\d,]+) same-content post pairs/.exec(d);
      const win = /within ([\d.]+ ?\w+) of each other/.exec(d);
      const np = pairs ? Number(pairs[1].replace(/,/g, '')) : null;
      return { label, key: m[1], sub: `median gap${np !== null ? ` · ${np.toLocaleString('en-US')} pairs` : ''}${win ? ` within ${win[1]}` : ''}` };
    }
    if (type === 'similar_creation_dates' && (m = /^(\d+) of (\d+) accounts.*?within ([\d.]+ ?\w+) of each other/.exec(d))) {
      const gap = /max gap ([\d.]+ ?\w+)/.exec(d);
      return { label, key: `${m[1]} / ${m[2]}`, sub: `accounts created within ${gap ? gap[1] : m[3]}` };
    }
    return { label, key: n ? `${n} posts` : '–', sub: d };
  }
  function narrSize(n) { return ((n && n.member_post_ids) || []).length; }
  /** Relevance group: 'threat' | 'related' | 'off_topic' | '' (not assessed). */
  function narrGroup(n, tags) {
    if (n && ['threat', 'related', 'off_topic'].includes(n.relevance)) return n.relevance;
    const t = tags || [];
    if (t.includes('off_topic')) return 'off_topic';
    if (t.includes('threat')) return 'threat';
    return '';
  }
  /** Severity scale (owner decision 2026-11-19), most severe first. */
  const SEVERITY_LEVELS = ['critical', 'high', 'medium', 'low'];
  const SEVERITY_LABEL = { critical: 'Critical', high: 'High', medium: 'Medium', low: 'Low' };
  /** A narrative's severity: the narrative's own field first, then the rank entry's; null
   *  for older runs, keyword labels and off-topic narratives. */
  function narrSeverity(n, r) {
    if (n && n.relevance === 'off_topic') return null;
    const v = (n && n.severity) || (r && r.severity) || null;
    return SEVERITY_LEVELS.includes(v) ? v : null;
  }
  function severityIndex(v) { const i = SEVERITY_LEVELS.indexOf(v); return i < 0 ? SEVERITY_LEVELS.length : i; }
  /** Ranked narratives: brief.narrative_ranks when present; otherwise (older runs) by
   *  severity (when rated) then post count, with the first 10 of at least 10 posts
   *  (never off-topic) as "main". */
  function rankNarratives(narratives, brief) {
    const list = Array.isArray(narratives) ? narratives : [];
    const byId = new Map(list.map((n) => [n.id, n]));
    const given = brief && Array.isArray(brief.narrative_ranks) ? brief.narrative_ranks : [];
    const out = [];
    const seen = new Set();
    given.slice().sort((a, b) => (Number(a.rank) || 1e9) - (Number(b.rank) || 1e9)).forEach((r) => {
      const n = r && byId.get(r.narrative_id);
      if (!n || seen.has(n.id)) return;
      seen.add(n.id);
      out.push(Object.assign({ score: 0, main: false, tags: [], reasons: [], cluster_ids: [], claim_ids: [], peak_day: null, peak_count: 0 },
        r, { group: narrGroup(n, r.tags), severity: narrSeverity(n, r), size: narrSize(n), fromBrief: true }));
    });
    let mainLeft = given.length ? 0 : 10;
    list.filter((n) => !seen.has(n.id))
      .sort((a, b) => severityIndex(narrSeverity(a)) - severityIndex(narrSeverity(b))
        || narrSize(b) - narrSize(a) || String(a.id).localeCompare(String(b.id), 'en', { numeric: true }))
      .forEach((n) => {
        const g = narrGroup(n, []);
        const sev = narrSeverity(n);
        // no brief: no coordination or claim context, so a 5-9 post critical/high
        // narrative cannot qualify (analyze/brief.py bucket c); >= 10 posts, by severity
        const big = narrSize(n) >= 10;
        const main = mainLeft > 0 && big && g !== 'off_topic';
        if (main) mainLeft -= 1;
        out.push({ narrative_id: n.id, score: narrSize(n), main, tags: [], reasons: [], cluster_ids: [], claim_ids: [],
          peak_day: null, peak_count: 0, group: g, severity: sev, size: narrSize(n), fromBrief: false });
      });
    out.forEach((r, i) => { r.rank = i + 1; });
    return out;
  }
  /** Busiest day of a volume_by_day map (earliest day wins a tie). */
  function peakOf(vbd) {
    let day = null;
    let count = 0;
    Object.keys(vbd || {}).sort().forEach((d) => { const v = Number(vbd[d]) || 0; if (v > count) { day = d; count = v; } });
    return { day, count };
  }
  const STATE_SEVERITY = ['ok', 'completed_zero', 'out_of_coverage', 'skipped_capability', 'skipped_no_credentials', 'partial', 'failed'];
  /** Per-source coverage notes (used when brief.json is missing) and, for sources that
   *  stopped early, the first day they reached: earlier days were not reached, not empty. */
  function coverageOf(status, posts, days) {
    const worst = new Map();
    (status || []).forEach((st) => {
      if (!st || !st.source) return;
      const cur = worst.get(st.source);
      if (cur === undefined || STATE_SEVERITY.indexOf(st.state) > STATE_SEVERITY.indexOf(cur)) worst.set(st.source, st.state);
    });
    const seen = {};
    const nonAd = new Set(); // sources with posts other than ads (ad dates are run dates, not a crawl order)
    (posts || []).forEach((p) => {
      if (p && p.kind !== 'ad') nonAd.add(p.source);
      if (!p || !p.timestamp_verified || p.in_window === false) return;
      const d = dayOf(p.published_at);
      if (d) (seen[p.source] = seen[p.source] || new Set()).add(d);
    });
    const total = days.length;
    const notes = [];
    const from = {};
    worst.forEach((state, src) => {
      const set = seen[src] || new Set();
      const inWin = days.filter((d) => set.has(d));
      const first = inWin[0] || null;
      const stopped = state === 'partial' || state === 'failed';
      let note;
      if (String(state).startsWith('skipped')) note = 'not collected (skipped)';
      else if (!inWin.length) note = state === 'failed' ? 'failed; no posts collected' : 'no posts in the window';
      else if (inWin.length === total) note = `all ${total} days covered`;
      else if (stopped && first > days[0]) {
        note = first === days[total - 1] ? 'newest day only; older days not reached' : `from ${shortDay(first)} only; older days not reached`;
      } else note = `posts on ${inWin.length} of ${total} days`;
      if (stopped && first && first > days[0] && nonAd.has(src)) from[src] = first;
      notes.push({ source: src, state, days_with_posts: inWin.length, days_total: total, note });
    });
    return { notes, from };
  }
  /** Posts many direct links start from (a video and its comments): origin hubs, else
   *  counted from the edges. */
  function hubsOf(tr, min) {
    if (tr && Array.isArray(tr.hubs) && tr.hubs.length) return tr.hubs.slice();
    const m = new Map();
    ((tr && tr.edges) || []).forEach((e) => {
      const hb = m.get(e.from) || { post_id: e.from, direct_out: 0, relations: {}, inferred_out: 0 };
      if (e.type === 'direct') {
        hb.direct_out += 1;
        const r = e.relation || 'link';
        hb.relations[r] = (hb.relations[r] || 0) + 1;
      } else hb.inferred_out += 1;
      m.set(e.from, hb);
    });
    return Array.from(m.values()).filter((hb) => hb.direct_out >= (min || 2))
      .sort((a, b) => b.direct_out - a.direct_out || String(a.post_id).localeCompare(String(b.post_id))).slice(0, 5);
  }
  const REL_WORDS = { reply: ['reply', 'replies'], quote: ['quote', 'quotes'], repost: ['repost', 'reposts'],
    forward: ['forward', 'forwards'], link: ['link', 'links'], mention: ['mention', 'mentions'] };
  /** {reply: 100} -> '100 comments' on YouTube, '100 replies' elsewhere. */
  function relationSummary(rels, src) {
    return Object.entries(rels || {}).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).map(([k, v]) => {
      const w = k === 'reply' && src === 'youtube' ? ['comment', 'comments'] : (REL_WORDS[k] || [k, `${k}s`]);
      return `${v} ${v === 1 ? w[0] : w[1]}`;
    }).join(', ');
  }
  function clusterSize(c) { return ((c && c.member_author_ids) || []).length || ((c && c.member_handles) || []).length; }
  /** Strongest groups: brief.main_cluster_ids, else groups of 5+ accounts (or the top 3). */
  function mainClusters(clusters, brief) {
    const list = Array.isArray(clusters) ? clusters : [];
    const ids = new Set(list.map((c) => c.id));
    if (brief && Array.isArray(brief.main_cluster_ids) && brief.main_cluster_ids.length) {
      return brief.main_cluster_ids.filter((id) => ids.has(id));
    }
    const sorted = list.slice().sort((a, b) => clusterSize(b) - clusterSize(a) || (b.signals || []).length - (a.signals || []).length);
    const big = sorted.filter((c) => clusterSize(c) >= 5);
    return (big.length ? big : sorted.slice(0, 3)).map((c) => c.id);
  }
  /** Narratives each group pushes: brief.cluster_narratives, else narratives holding at
   *  least 30% of the group's supporting posts (largest share first). */
  function clusterNarratives(clusters, narrOf, brief) {
    const given = brief && brief.cluster_narratives && typeof brief.cluster_narratives === 'object' ? brief.cluster_narratives : {};
    const out = {};
    (clusters || []).forEach((c) => {
      if (Array.isArray(given[c.id])) { out[c.id] = given[c.id].slice(); return; }
      const sup = c.supporting_post_ids || [];
      const cnt = new Map();
      sup.forEach((pid) => (narrOf.get(pid) || []).forEach((nid) => cnt.set(nid, (cnt.get(nid) || 0) + 1)));
      const need = Math.max(1, Math.ceil(sup.length * 0.3));
      out[c.id] = Array.from(cnt.entries()).filter(([, v]) => v >= need).sort((a, b) => b[1] - a[1]).map(([k]) => k);
    });
    return out;
  }
  function engagementOf(p) {
    const m = (p && p.metrics) || {};
    return ['likes', 'reposts', 'replies', 'quotes', 'shares', 'comments', 'forwards', 'reactions'].reduce((a, k) => a + (Number(m[k]) || 0), 0);
  }
  function viewsOf(p) { const v = p && p.metrics && p.metrics.views; return typeof v === 'number' ? v : -1; }
  /** Most-seen posts: by views, then engagement. */
  function topPosts(posts, k) {
    return (posts || []).filter(Boolean).slice()
      .sort((a, b) => (viewsOf(b) - viewsOf(a)) || (engagementOf(b) - engagementOf(a)) || String(a.post_id).localeCompare(String(b.post_id)))
      .slice(0, k);
  }
  /** Representative posts: the narrative's example_post_ids (closest to its centre,
   *  distinct accounts first), in stored order; most-seen posts fill up when fewer exist.
   *  Severity-rated cards show these first: the most-viewed members of a mixed narrative
   *  can be its least typical posts (sample N6: card dumps + two card-review videos). */
  function repPosts(ctx, n, k) {
    const out = [];
    const seen = new Set();
    ((n && n.example_post_ids) || []).forEach((pid) => {
      const p = ctx.postsById.get(pid);
      if (p && !seen.has(pid) && out.length < k) { out.push(p); seen.add(pid); }
    });
    return out;
  }
  /** "12 high · 9 medium · 60 low · 16 unrated" for a list of ranks (levels present only). */
  function severityCounts(ranks) {
    const c = {};
    (ranks || []).forEach((r) => { const k = r.severity || 'unrated'; c[k] = (c[k] || 0) + 1; });
    return SEVERITY_LEVELS.concat(['unrated']).filter((k) => c[k]).map((k) => `${Number(c[k]).toLocaleString('en-US')} ${k}`).join(' · ');
  }
  /** Platforms of THIS narrative (volume_by_platform > 0) that were not collected on
   *  `day` (their coverage starts later). Used to hatch only days a narrative's own
   *  platforms missed. */
  function gapSources(day, coverFrom, vbp) {
    const cf = coverFrom || {};
    return Object.keys(vbp || {}).filter((pl) => (Number(vbp[pl]) || 0) > 0 && cf[pl] && day < cf[pl]).sort();
  }
  /** Close-match first-seen time per platform (origin.platform_first_seen_strong), and
   *  the platforms whose recorded first post is only a loose match: their first-seen
   *  time differs from the close-match one, or they have no close match at all. */
  function looseFirstPlatforms(tr) {
    const strong = (tr && tr.platform_first_seen_strong) || {};
    const pfs = (tr && tr.platform_first_seen) || {};
    if (!Object.keys(strong).length) return [];
    return Object.keys(pfs).filter((pl) => strong[pl] !== pfs[pl]).sort();
  }
  /** Platforms in close-match first-seen order ([] when not computed). */
  function strongSeq(tr) {
    const strong = (tr && tr.platform_first_seen_strong) || {};
    const t = (pl) => { const v = Date.parse(strong[pl]); return Number.isNaN(v) ? Infinity : v; };
    return Object.keys(strong).sort((a, b) => (t(a) - t(b)) || (a < b ? -1 : a > b ? 1 : 0));
  }
  /** Hubs split by membership: a hub post outside the narrative is the parent of member
   *  posts (e.g. a video whose comments are in the narrative), not part of it. */
  function hubsWithMembership(hubs, memberIds) {
    const set = new Set(memberIds || []);
    return (hubs || []).map((hb) => Object.assign({}, hb, { member: set.has(hb.post_id) }));
  }
  /** Collapse blank lines and cut to `lines` lines with the ellipsis on the last line. */
  function compactLines(text, lines, maxChars) {
    const all = String(text || '').replace(/\r/g, '').split('\n').map((x) => x.trim()).filter(Boolean);
    let out = all.slice(0, lines || 3).join('\n');
    let cut = all.length > (lines || 3);
    if (maxChars && out.length > maxChars) { out = out.slice(0, maxChars - 1).trimEnd(); cut = true; }
    return cut ? `${out.replace(/[\s.,;:…]+$/, '')}…` : out;
  }
  /** Pruning trace for narrative nid from run.json options (narratives.TRACE_KEYS):
   *  {removed, moved, left, into, from, mixed}; null when nothing applies. splitMembers:
   *  member post IDs of the narrative the removed posts were split into (if any). */
  function pruneTrace(options, nid, splitMembers) {
    const o = options || {};
    const removed = ((o.pruned_post_ids || {})[nid]) || [];
    const splits = o.pruned_split_into || {};
    const into = splits[nid] || null;
    const from = Object.keys(splits).find((k) => splits[k] === nid) || null;
    const mixed = (o.narratives_mixed || {})[nid] || null;
    if (!removed.length && !from && !mixed) return null;
    const inSplit = new Set(into ? (splitMembers || []) : []);
    return { removed, moved: removed.filter((pid) => inSplit.has(pid)),
      left: removed.filter((pid) => !inSplit.has(pid)), into, from, mixed };
  }
  // END viewHelpers

  const VERDICTS = ['supported', 'disputed', 'unverified', 'contradicted'];
  const VERDICT_HELP = {
    supported: 'evidence from the window backs it', disputed: 'sources partly agree or conflict',
    unverified: 'nothing found that settles it', contradicted: 'evidence says otherwise',
  };
  const VERDICT_CHIP = { supported: 'run', disputed: 'warn', contradicted: 'risk', unverified: '' };
  function verdictOf(c) { return VERDICTS.includes(c && c.verdict) ? c.verdict : 'unverified'; }
  function validLang(l) { return l && /^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{1,8})*$/.test(l) ? l : null; }
  function narrHref(ctx, nid) { return `#/run/${enc(ctx.id)}/narrative/${enc(nid)}`; }

  function chip(cls, text, title) { return h('span', { class: `chip${cls ? ` ${cls}` : ''}`, title: title || null }, text); }
  function platformChip(src, text) {
    const known = KNOWN_PLATFORMS.includes(src);
    return h('span', {
      class: 'chip plat', title: sourceLabel(src),
      style: { background: platformColor(src), color: known ? 'var(--on-platform)' : '#fff' },
    }, platformLogo(src), text === undefined ? pname(src) : text);
  }
  function platformEntries(vbp) {
    return Object.entries(vbp || {}).filter(([, v]) => typeof v === 'number' && v > 0).sort((a, b) => b[1] - a[1]);
  }
  function linkButton(text, onclick, title) {
    return h('button', { type: 'button', class: 'link', onclick, title: title || null }, text);
  }

  /** One-line post: platform, author, opening text, views. Opens the post detail. */
  function postRow(p, opts) {
    const o = opts || {};
    const eng = engagementOf(p);
    const metric = viewsOf(p) >= 0 ? `${compactNum(viewsOf(p))} views` : (eng ? `${compactNum(eng)} interactions` : '');
    return h('button', { type: 'button', class: 'post-row', title: 'Open post detail', onclick: () => openPost(p.post_id) },
      platformChip(p.source),
      h('span', { class: 'pr-text', dir: 'auto' }, h('b', null, p.author_handle || p.author_name || 'unknown'),
        ` · “${firstLine(p.text, 160) || 'no text'}”`,
        o.time ? h('span', { class: 'muted' }, ` · ${shortTs(p.published_at)}`) : null),
      h('span', { class: 'pr-metric' }, metric));
  }
  /** Inline link to a post. An <a role=button>, not a <button>: buttons cannot break
   *  across lines, so a long reference left its full stop alone on the next line. */
  function postRef(p) {
    return h('a', { href: '#', role: 'button', class: 'inline-ref', dir: 'auto',
      onclick: (ev) => { ev.preventDefault(); openPost(p.post_id); } },
    `${p.author_handle || p.author_name || p.post_id} “${firstLine(p.text, 70)}”`);
  }

  /** Platform mix as one bar plus a legend line. */
  function mixBar(entries) {
    const total = entries.reduce((a, [, v]) => a + v, 0);
    if (!total) return null;
    return h('div', { class: 'mix-wrap' },
      h('div', { class: 'mix', role: 'img', 'aria-label': `By platform: ${entries.map(([pl, v]) => `${pname(pl)} ${v}`).join(', ')}` },
        entries.map(([pl, v]) => h('span', { title: `${pname(pl)}: ${v}`, style: { width: `${(v / total) * 100}%`, background: platformColor(pl) } }))),
      h('div', { class: 'mix-legend' }, entries.map(([pl, v]) => h('span', null, h('i', { style: { background: platformColor(pl) } }), `${pname(pl)} ${fmtNum(v)}`))));
  }

  function sparkline(vbd, days, hot) {
    const ds = days.length ? days : Object.keys(vbd || {}).sort();
    const vals = ds.map((d) => Number((vbd || {})[d]) || 0);
    const max = Math.max(1, ...vals);
    const W = 120;
    const H = 30;
    const x = (i) => (ds.length > 1 ? (i / (ds.length - 1)) * W : W / 2);
    const y = (v) => H - 3 - (v / max) * (H - 6);
    return s('svg', {
      class: `spark${hot ? ' hot' : ''}`, viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none', role: 'img',
      'aria-label': `Posts per day: ${ds.map((d, i) => `${shortDay(d)} ${vals[i]}`).join(', ')}`,
    }, s('polyline', { class: 'spark-line', points: vals.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' '), 'vector-effect': 'non-scaling-stroke' }));
  }

  /** "Telegram and LinkedIn were collected for 19 Nov only, so earlier days undercount them." */
  function coverageCaveat(ctx) {
    const ents = Object.entries(ctx.coverFrom || {});
    if (!ents.length) return '';
    const byDay = {};
    ents.forEach(([src, d]) => { (byDay[d] = byDay[d] || []).push(pname(src)); });
    const last = ctx.days[ctx.days.length - 1];
    const parts = Object.keys(byDay).sort().map((d) => `${joinNames(byDay[d])} ${byDay[d].length > 1 ? 'were' : 'was'} collected ${d === last ? `for ${shortDay(d)} only` : `from ${shortDay(d)} on`}`);
    return `${parts.join('; ')}, so earlier days undercount ${ents.length > 1 ? 'them' : 'it'} (not reached, not empty).`;
  }

  /** Severity badge (critical / high / medium / low); null for unrated narratives. */
  function sevBadge(sev, reason) {
    if (!SEVERITY_LEVELS.includes(sev)) return null;
    return h('span', { class: `chip sev sev-${sev}`, title: reason ? `${SEVERITY_LABEL[sev]} severity: ${reason}` : `${SEVERITY_LABEL[sev]} severity` }, SEVERITY_LABEL[sev]);
  }
  /** The labeller's one-line reason for the severity, shown under a narrative's label. */
  function sevReasonLine(n, sev) {
    const why = n && String(n.severity_reason || '').trim();
    if (!sev || !why) return null;
    return h('p', { class: 'sev-reason', dir: 'auto' }, h('span', { class: `sev-reason-k sev-k-${sev}` }, `${SEVERITY_LABEL[sev]} severity`), ` · ${why}`);
  }
  function isSevere(r) { return r && (r.severity === 'critical' || r.severity === 'high' || (!r.severity && r.group === 'threat')); }

  /** Narrative chips: severity, relevance, coordination, surging, claims by verdict. */
  function narrChips(ctx, nid, opts) {
    const r = ctx.rankOf.get(nid) || {};
    const n = ctx.narrById.get(nid) || {};
    const tags = new Set(r.tags || []);
    const out = [];
    const sev = narrSeverity(n, r);
    if (sev) out.push(sevBadge(sev, n.severity_reason || null));
    // with a severity the "threat-relevant" chip would repeat it; older runs keep it
    if (r.group === 'threat' && !sev) out.push(chip('risk', 'threat-relevant', n.relevance_reason || null));
    else if (r.group === 'off_topic') out.push(chip('', 'off-topic', n.relevance_reason || null));
    const cl = ctx.narrClusters.get(nid) || [];
    if (cl.length && !(opts && opts.noCoord)) out.push(chip('warn', `coordinated · ${plural(cl.length, 'group')}`));
    if (tags.has('surging')) out.push(chip('run', 'surging'));
    const vc = {};
    (ctx.claimsByNarr.get(nid) || []).forEach((c) => { const v = verdictOf(c); vc[v] = (vc[v] || 0) + 1; });
    ['contradicted', 'disputed', 'supported', 'unverified'].forEach((v) => {
      if (vc[v]) out.push(chip(VERDICT_CHIP[v], `${vc[v]} ${v} ${vc[v] === 1 ? 'claim' : 'claims'}`));
    });
    return out;
  }

  /** Earliest post in collected data for a narrative: {pid, post, source, at}. */
  function firstObserved(ctx, nid) {
    const tr = ctx.originByNarr.get(nid);
    const pid = tr && (tr.earliest_post_ids || [])[0];
    const p = pid && ctx.postsById.get(pid);
    if (p && p.published_at) return { pid, post: p, source: p.source, at: p.published_at, loose: tr.earliest_match === 'weak' };
    const n = ctx.narrById.get(nid);
    if (n && n.first_seen_at) return { pid: null, post: null, source: tr ? platformSeq(tr)[0] : null, at: n.first_seen_at };
    return null;
  }

  function verdictCounts(ctx) {
    const given = ctx.brief && ctx.brief.verdict_counts && typeof ctx.brief.verdict_counts === 'object' ? ctx.brief.verdict_counts : null;
    if (given && Object.keys(given).length) return given;
    const vc = {};
    ctx.claims.forEach((c) => { const v = verdictOf(c); vc[v] = (vc[v] || 0) + 1; });
    return vc;
  }

  /** A box that hides a list until asked; build() runs on first open. */
  function foldBox(title, sub, build, open) {
    const body = h('div', { class: 'fold-body' });
    const btn = h('button', { type: 'button', class: 'small', 'aria-expanded': 'false' }, 'Show');
    let shown = false;
    const toggle = () => {
      shown = !shown;
      btn.textContent = shown ? 'Hide' : 'Show';
      btn.setAttribute('aria-expanded', String(shown));
      clear(body);
      if (shown) append(body, build());
    };
    btn.addEventListener('click', toggle);
    const box = h('div', { class: 'fold' },
      h('div', { class: 'fold-head' }, h('div', { class: 'fold-title' }, h('strong', null, title), sub ? h('span', { class: 'k' }, sub) : null), btn),
      body);
    if (open) toggle();
    return box;
  }

  // ---- Summary
  function tabSummary(view, ctx) {
    const base = `#/run/${enc(ctx.id)}/`;
    const verified = ctx.posts.filter(analysable).length;
    const nMain = ctx.ranks.filter((r) => r.main).length;
    const nCl = ctx.mainClusterIds.length;
    const vc = verdictCounts(ctx);
    const kpi = (label, n, sub, tab) => h('a', { class: 'kpi-card', href: base + tab },
      h('span', { class: 'kpi-label' }, label), h('span', { class: 'kpi-num' }, fmtNum(n)), h('span', { class: 'kpi-sub' }, sub));
    view.appendChild(h('div', { class: 'kpi-grid' },
      kpi('Posts collected', ctx.posts.length, `${fmtNum(verified)} with verified in-window time · ${fmtNum(ctx.narrOf.size)} in narratives`, 'posts'),
      kpi('Main narratives', nMain, ctx.narratives.length > nMain ? `of ${fmtNum(ctx.narratives.length)} · ${fmtNum(ctx.narratives.length - nMain)} smaller or off-topic, collapsed` : 'every narrative formed', 'narratives'),
      kpi('Coordinated groups', nCl, ctx.clusters.length > nCl ? `of ${ctx.clusters.length} · ${ctx.clusters.length - nCl} smaller collapsed` : 'two or more signals agree in each', 'coordination'),
      kpi('Claims checked', ctx.claims.length, VERDICTS.filter((v) => vc[v]).map((v) => `${vc[v]} ${v}`).join(' · ') || 'none extracted', 'claims')));
    const derived = Array.isArray(ctx.data.derived_on_load) ? ctx.data.derived_on_load : [];
    if (derived.length) {
      view.appendChild(h('div', { class: 'banner info derived-note' },
        'This run was saved before the summary existed. The summary, evidence flags and claim bottom lines were built from its saved files when it was opened (by rule; verdicts unchanged). Narrative relevance and severity and the earliest-post match check need a re-analysis (the Re-analyse button for runs in the runs folder, or socmint reanalyze <run folder>).'));
    }
    view.appendChild(findingsCard(ctx));
    view.appendChild(coverageCard(ctx));
  }

  function fallbackFindings(ctx) {
    const out = [];
    ctx.ranks.filter((r) => r.main).slice(0, 3).forEach((r) => {
      const n = ctx.narrById.get(r.narrative_id);
      const plats = platformEntries(n.volume_by_platform).length;
      const pk = peakOf(n.volume_by_day);
      out.push({
        kind: 'narrative', narrative_ids: [n.id], cluster_ids: [], claim_ids: [], post_ids: n.member_post_ids || [],
        headline: `${n.label || n.id}: ${plural(narrSize(n), 'post')} on ${plural(plats, 'platform')}${pk.day ? `, peaking on ${shortDay(pk.day)} (${plural(pk.count, 'post')})` : ''}.`,
        detail: n.claim_summary || '',
      });
    });
    const c = ctx.clusterById.get(ctx.mainClusterIds[0]);
    if (c) {
      out.push({ kind: 'coordination', narrative_ids: [], cluster_ids: [c.id], claim_ids: [], post_ids: c.supporting_post_ids || [],
        headline: c.summary || clusterHeadline(c), detail: '' });
    }
    return out;
  }

  function findingsCard(ctx) {
    const given = ctx.brief && Array.isArray(ctx.brief.findings) && ctx.brief.findings.length ? ctx.brief.findings : null;
    const list = given || fallbackFindings(ctx);
    const card = h('section', { class: 'card findings' }, h('div', { class: 'card-head' }, h('h2', null, 'Key findings'),
      h('span', { class: 'k' }, given ? `${ctx.ranks.some((r) => r.severity) ? 'most severe first' : 'ranked by volume, spread across platforms and risk'}; every figure links to its posts`
        : 'largest narratives and strongest group (this run has no ranking file)')));
    if (!list.length) card.appendChild(h('p', { class: 'muted' }, 'Nothing to summarise: no narratives or coordinated groups were formed. See Sources for why.'));
    list.forEach((f, i) => card.appendChild(findingRow(f, i + 1, ctx)));
    return card;
  }

  function findingRow(f, i, ctx) {
    const base = `#/run/${enc(ctx.id)}`;
    const chips = [];
    const links = [];
    const nid = (f.narrative_ids || []).find((x) => ctx.narrById.has(x));
    const cids = (f.cluster_ids || []).filter((x) => ctx.clusterById.has(x));
    if (nid) {
      const n = ctx.narrById.get(nid);
      // A claims finding names its most notable claim's narrative only for the link: the
      // narrative's platform and tag chips would read as figures about the claims.
      if (f.kind !== 'claims') {
        platformEntries(n.volume_by_platform).forEach(([pl, v]) => chips.push(platformChip(pl, `${pname(pl)} ${fmtNum(v)}`)));
        chips.push(...narrChips(ctx, nid, { noCoord: cids.length > 0 }));
      }
      links.push(h('a', { class: 'more', href: narrHref(ctx, nid) }, 'Open narrative →'));
    }
    if (cids.length) {
      if (!nid) {
        const srcs = new Set();
        cids.forEach((id) => (ctx.clusterById.get(id).sources || []).forEach((x) => srcs.add(x)));
        srcs.forEach((x) => chips.push(platformChip(x)));
      }
      const accts = cids.reduce((a, id) => a + clusterSize(ctx.clusterById.get(id)), 0);
      chips.push(chip('warn', `${plural(cids.length, 'coordinated group')} · ${fmtNum(accts)} accounts`));
      links.push(h('a', { class: 'more', href: `${base}/coordination` }, cids.length > 1 ? 'Open groups →' : 'Open group →'));
    }
    const clids = (f.claim_ids || []).filter((x) => ctx.claims.some((c) => c.id === x));
    if (clids.length) links.push(h('a', { class: 'more', href: nid ? `${base}/claims/${enc(nid)}` : `${base}/claims` }, `${plural(clids.length, 'claim')} →`));
    if (f.kind === 'coverage') links.push(h('a', { class: 'more', href: `${base}/sources` }, 'Source details →'));
    const pids = f.post_ids || [];
    if (pids.length) links.push(linkButton(`View ${plural(pids.length, 'post')}`, () => showPosts(ctx, pids, f.headline)));
    return h('div', { class: 'finding' },
      h('div', { class: 'f-num' }, String(i)),
      h('div', { class: 'f-body' },
        h('div', { class: 'f-head', dir: 'auto' }, f.headline || ''),
        f.detail ? h('p', { class: 'f-detail', dir: 'auto' }, f.detail) : null,
        (chips.length || links.length) ? h('div', { class: 'f-meta' }, chips.length ? h('div', { class: 'chips' }, chips) : null,
          links.length ? h('div', { class: 'f-links' }, links) : null) : null));
  }

  function coverageCard(ctx) {
    const card = h('section', { class: 'card coverage' }, h('div', { class: 'card-head' }, h('h2', null, 'How complete is this picture?'),
      h('a', { class: 'more', href: `#/run/${enc(ctx.id)}/sources` }, 'Source details →')));
    if (!ctx.coverage.length) {
      card.appendChild(h('div', { class: 'banner warn' }, 'No source status was recorded for this run, so empty results cannot be explained. Treat this run as incomplete.'));
      return card;
    }
    card.appendChild(h('div', { class: 'cov-grid' }, ctx.coverage.map((cv) => {
      const w = worstStatus(ctx.status.filter((x) => x.source === cv.source));
      const prob = w && isProblem(w.headline_kind) && !['coverage', 'limit'].includes(w.headline_kind);
      return [
        platformChip(cv.source),
        h('span', { class: 'cov-note' }, prob ? statusHead(w, { details: false }) : (cv.note || '')),
        h('span', { class: 'cov-state' }, cv.state && cv.state !== 'ok' ? stateChip(cv.state) : null)];
    })));
    const cav = coverageCaveat(ctx);
    if (cav) card.appendChild(h('p', { class: 'k cov-caveat' }, cav));
    return card;
  }

  // ---- Narratives
  function narrState(id) {
    S.narr = S.narr || {};
    if (!S.narr[id]) S.narr[id] = { filter: 'main' };
    return S.narr[id];
  }

  function tabNarratives(view, ctx) {
    if (!ctx.narratives.length) {
      view.appendChild(h('div', { class: 'card muted' }, ctx.posts.length
        ? 'No narratives were formed: too few posts with text, or no group of similar posts large enough. All posts remain in the Posts tab.'
        : 'No posts were collected, so no narratives could be formed. See the Sources tab.'));
      return;
    }
    const st = narrState(ctx.id);
    const fromBrief = ctx.ranks.some((r) => r.fromBrief);
    const hasSev = ctx.ranks.some((r) => r.severity);
    view.appendChild(h('p', { class: 'lead' }, `${plural(ctx.narratives.length, 'narrative')} found. `,
      hasSev ? 'Ranked by severity for the topic (critical first, as rated by the labeller from the posts), then coordinated or disputed narratives, then size. '
        : fromBrief ? 'Ranked by posts, platforms, growth and risk. ' : 'Ranked by post count (this run has no ranking file). ',
      'Small and off-topic narratives are collapsed at the bottom, never removed. Daily volumes count only posts with a verified platform time inside the window.'));
    const groups = { main: (r) => r.main, off_topic: (r) => r.group === 'off_topic', all: () => true };
    SEVERITY_LEVELS.forEach((v) => { groups[v] = (r) => r.severity === v; });
    const hasOff = ctx.ranks.some((r) => r.group === 'off_topic');
    const pills = [['main', 'Main']].concat(hasSev ? SEVERITY_LEVELS.map((v) => [v, SEVERITY_LABEL[v]]) : [],
      hasOff ? [['off_topic', 'Off-topic']] : [], [['all', 'All']]);
    if (!pills.some(([k]) => k === st.filter)) st.filter = 'main';
    const bar = h('div', { class: 'pills', role: 'group', 'aria-label': 'Filter narratives' });
    const body = h('div');
    const draw = () => {
      clear(bar);
      pills.forEach(([k, label]) => bar.appendChild(h('button', {
        type: 'button', class: `pill${st.filter === k ? ' on' : ''}${SEVERITY_LEVELS.includes(k) ? ` pill-sev pill-${k}` : ''}`, 'aria-pressed': String(st.filter === k),
        onclick: () => { st.filter = k; draw(); },
      }, SEVERITY_LEVELS.includes(k) ? h('i', { class: `sev-dot sev-${k}`, 'aria-hidden': 'true' }) : null,
      label, h('span', { class: 'pill-n' }, fmtNum(ctx.ranks.filter(groups[k]).length)))));
      bar.appendChild(h('span', { class: 'k pills-sort' }, `Sorted by: ${hasSev ? 'Severity' : fromBrief ? 'Rank' : 'Posts'}`));
      clear(body);
      const sel = ctx.ranks.filter(groups[st.filter]);
      sel.filter((r) => r.main).forEach((r) => body.appendChild(narrativeCard(r, ctx)));
      const others = sel.filter((r) => !r.main);
      if (st.filter === 'main') {
        const rest = ctx.ranks.filter((r) => !r.main);
        const off = rest.filter((r) => r.group === 'off_topic');
        const small = rest.filter((r) => r.group !== 'off_topic');
        if (off.length) {
          const posts = off.reduce((a, r) => a + r.size, 0);
          body.appendChild(foldBox(`Off-topic · ${off.length}`,
            `${plural(posts, 'post')} that mention the topic only in passing: ${off.slice(0, 3).map((r) => truncate(labelOf(ctx, r.narrative_id), 40)).join(' · ')}`,
            () => narrList(ctx, off)));
        }
        if (small.length) {
          body.appendChild(foldBox(`Other narratives · ${fmtNum(small.length)}`,
            severityCounts(small), () => narrList(ctx, small)));
        }
      } else if (others.length) {
        body.appendChild(foldBox(sel.some((r) => r.main) ? `Other narratives in this view · ${others.length}` : `${plural(others.length, 'narrative')}`,
          'one line per narrative', () => narrList(ctx, others), true));
      }
      if (!sel.length) body.appendChild(h('div', { class: 'card muted' }, 'No narratives in this group.'));
    };
    view.appendChild(bar);
    view.appendChild(body);
    draw();
  }

  function narrativeCard(r, ctx) {
    const n = ctx.narrById.get(r.narrative_id);
    const members = n.member_post_ids || [];
    const href = narrHref(ctx, n.id);
    const chips = narrChips(ctx, n.id);
    const fo = firstObserved(ctx, n.id);
    if (fo) chips.push(chip(fo.loose ? 'warn' : '', `first observed: ${fo.source ? `${pname(fo.source)} · ` : ''}${shortDay(fo.at)}${fo.loose ? ' (loose match)' : ''}`,
      fo.loose ? 'The earliest collected post only loosely matches this narrative; see How it spread for the first close match' : 'First observed in collected data, not necessarily the origin'));
    const rep = r.severity ? repPosts(ctx, n, 2) : [];
    const top = rep.length ? rep : topPosts(members.map((pid) => ctx.postsById.get(pid)), 2);
    const claims = ctx.claimsByNarr.get(n.id) || [];
    const plats = platformEntries(n.volume_by_platform);
    const srcs = plats.length ? plats.map(([pl]) => pl) : Array.from(new Set(members.map((pid) => (ctx.postsById.get(pid) || {}).source).filter(Boolean)));
    const pk = peakOf(n.volume_by_day);
    const card = h('article', { class: `card ncard${isSevere(r) ? ' is-threat' : ''}${r.severity ? ` sev-edge-${r.severity}` : ''}` },
      h('div', { class: 'nc-rank', title: r.severity ? `Rank ${r.rank} (sorted by severity)` : `Rank ${r.rank}` }, String(r.rank)),
      h('div', { class: 'nc-body' },
        h('h2', { class: 'nc-title', dir: 'auto' }, h('a', { href }, n.label || n.id)),
        sevReasonLine(n, r.severity),
        n.claim_summary ? h('p', { class: 'nc-summary clamp-2', dir: 'auto', title: n.claim_summary }, n.claim_summary) : null,
        chips.length ? h('div', { class: 'chips' }, chips) : null,
        top.length ? h('div', { class: 'post-rows' }, top.map((p) => postRow(p))) : null,
        h('div', { class: 'nc-links' },
          h('a', { class: 'more', href }, `${plural(members.length, 'post')} →`),
          ctx.originByNarr.has(n.id) ? h('a', { class: 'more', href: `#/run/${enc(ctx.id)}/origin/${enc(n.id)}` }, 'How it spread →') : null,
          claims.length ? h('a', { class: 'more', href: `#/run/${enc(ctx.id)}/claims/${enc(n.id)}` }, `${plural(claims.length, 'claim')} →`) : null)),
      h('div', { class: 'nc-stats' },
        h('div', { class: 'nc-count' }, h('span', { class: 'big' }, fmtNum(members.length)), h('span', { class: 'k' }, `posts · ${plural(srcs.length, 'platform')}`)),
        pk.day ? sparkline(n.volume_by_day, ctx.days, isSevere(r)) : null,
        h('div', { class: 'k' }, pk.day ? `peak ${shortDay(pk.day)} · ${plural(pk.count, 'post')}`
          : `no posts with a verified in-window time${srcs.length ? ` (${srcs.map(pname).join(', ')})` : ''}`),
        mixBar(plats)));
    card.addEventListener('click', (e) => {
      if (e.target.closest('a, button, input, select, summary')) return;
      if (window.getSelection && String(window.getSelection())) return; // allow copying text
      location.hash = href;
    });
    return card;
  }

  /** One line per narrative, with search when long and paging by 50. */
  function narrList(ctx, ranks) {
    let q = '';
    let limit = 50;
    const rows = h('div', { class: 'nlist-rows', role: 'list' });
    const more = h('div', { class: 'nlist-more' });
    const search = ranks.length > 15 ? h('input', { type: 'search', class: 'nlist-search', placeholder: 'Search these narratives…', 'aria-label': 'Search narratives' }) : null;
    const draw = () => {
      const needle = q.trim().toLowerCase();
      const sel = ranks.filter((r) => {
        if (!needle) return true;
        const n = ctx.narrById.get(r.narrative_id) || {};
        return `${n.label || ''} ${n.claim_summary || ''} ${(n.keywords || []).join(' ')}`.toLowerCase().includes(needle);
      });
      clear(rows);
      clear(more);
      sel.slice(0, limit).forEach((r) => rows.appendChild(narrRow(ctx, r)));
      if (!sel.length) rows.appendChild(h('div', { class: 'muted small' }, 'No narrative matches.'));
      if (sel.length > limit) more.appendChild(h('button', { type: 'button', class: 'small', onclick: () => { limit += 50; draw(); } }, `Show more (${sel.length - limit} remaining)`));
    };
    if (search) search.addEventListener('input', () => { q = search.value; limit = 50; draw(); });
    draw();
    return h('div', { class: 'nlist' }, search,
      h('div', { class: 'nrow nrow-head', 'aria-hidden': 'true' }, h('span', null, '#'), h('span', null, 'Narrative'), h('span', null, 'Platforms'), h('span', null, 'Posts')),
      rows, more);
  }

  function narrRow(ctx, r) {
    const n = ctx.narrById.get(r.narrative_id);
    const plats = platformEntries(n.volume_by_platform);
    return h('a', { class: 'nrow', role: 'listitem', href: narrHref(ctx, n.id), title: n.claim_summary || null },
      h('span', { class: 'nrow-rank' }, String(r.rank)),
      h('span', { class: 'nrow-label', dir: 'auto' }, n.label || n.id),
      h('span', { class: 'nrow-plats', title: plats.map(([pl, v]) => `${pname(pl)} ${v}`).join(', ') },
        plats.map(([pl]) => h('i', { class: 'dot', style: { background: platformColor(pl) } }))),
      h('span', { class: 'nrow-n' }, fmtNum(narrSize(n))));
  }

  /** Outlier pruning trace from run.json options (narratives.TRACE_KEYS): posts the
   *  labeller judged not to fit this narrative's label, where they went, and narratives
   *  kept whole because the labeller flagged too many posts. null when nothing applies
   *  (older runs, keyword labels, nothing flagged). */
  function pruningCard(ctx, nid) {
    const o = (ctx.run && ctx.run.options) || {};
    const into0 = (o.pruned_split_into || {})[nid];
    const tr = pruneTrace(o, nid, into0 ? (ctx.narrById.get(into0) || {}).member_post_ids : null);
    if (!tr) return null;
    const { removed, moved, left, into, from, mixed } = tr;
    const card = h('section', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, 'Posts that did not fit the label')));
    const nlink = (id) => h('a', { href: narrHref(ctx, id) }, `${id}${ctx.narrById.get(id) ? ` “${ctx.narrById.get(id).label || id}”` : ''}`);
    if (mixed) {
      const k = `${mixed.flagged} of the ${plural(mixed.shown, 'post')} it was shown`;
      card.appendChild(h('p', null, mixed.reason === 'size'
        ? `The labeller judged ${k} not to fit this label. Removing ${mixed.flagged === 1 ? 'it' : 'them'} would leave fewer than 3 posts, so the narrative was kept whole: read its posts before relying on the label.`
        : `The labeller judged ${k} not to fit this label. That is too many to remove, so the narrative was kept whole: treat it as mixed, and read its posts before relying on the label.`));
      const fl = (mixed.flagged_post_ids || []).map((pid) => ctx.postsById.get(pid)).filter(Boolean);
      if (fl.length) {
        card.appendChild(h('p', { class: 'k' }, `The ${fl.length === 1 ? 'post' : 'posts'} the labeller flagged:`));
        card.appendChild(h('div', { class: 'post-rows' }, fl.slice(0, 10).map((p) => postRow(p, { time: true }))));
      }
    }
    if (from) {
      card.appendChild(h('p', null, 'These posts were grouped into ', nlink(from), ' at first. The labeller judged that they did not fit that label, so they were split off into this narrative.'));
    }
    if (removed.length) {
      const what = [];
      if (moved.length) what.push(h('span', null, `${plural(moved.length, 'post')} now form `, nlink(into)));
      if (left.length) what.push(h('span', null, `${plural(left.length, 'post')} ${left.length === 1 ? 'belongs' : 'belong'} to no narrative (still in the posts table and exports)`));
      card.appendChild(h('p', null, `${plural(removed.length, 'post')} removed as not fitting this label: `, what.reduce((a, x, i) => (i ? a.concat(['; ', x]) : [x]), []), '. They are not counted in the volumes above.'));
      const rows = removed.map((pid) => ctx.postsById.get(pid)).filter(Boolean);
      const shown = rows.slice(0, 10);
      card.appendChild(h('div', { class: 'post-rows' }, shown.map((p) => postRow(p, { time: true }))));
      if (removed.length > shown.length) card.appendChild(linkButton(`All ${plural(removed.length, 'removed post')} →`, () => showPosts(ctx, removed, `posts removed from ${nid}`)));
    }
    return card;
  }

  /** Per-narrative detail page (#/run/<id>/narrative/<nid>). */
  function pageNarrative(view, ctx, nid) {
    const base = `#/run/${enc(ctx.id)}`;
    view.appendChild(h('a', { class: 'back', href: `${base}/narratives` }, '← All narratives'));
    const n = ctx.narrById.get(nid);
    if (!n) { view.appendChild(h('div', { class: 'banner warn' }, `Narrative ${nid} is not in this run.`)); return; }
    const r = ctx.rankOf.get(nid) || {};
    const members = n.member_post_ids || [];
    const mp = members.map((pid) => ctx.postsById.get(pid)).filter(Boolean);
    const plats = platformEntries(n.volume_by_platform);
    const fo = firstObserved(ctx, nid);
    const pk = peakOf(n.volume_by_day);
    const unv = mp.filter((p) => !analysable(p)).length;
    const views = mp.reduce((a, p) => a + Math.max(0, viewsOf(p)), 0);
    const chips = narrChips(ctx, nid);
    const kpi = (label, val, sub) => h('div', { class: 'kpi' }, h('span', { class: 'k' }, label), h('b', null, val), h('span', { class: 'k' }, sub));
    view.appendChild(h('section', { class: 'card' },
      h('div', { class: 'k eyebrow-k' }, `Narrative${r.rank ? ` #${r.rank}` : ''} · ${n.id} · ${n.label_method === 'llm' ? 'LLM label' : 'keyword label'}`),
      h('h1', { class: 'detail-title', dir: 'auto' }, n.label || n.id),
      sevReasonLine(n, narrSeverity(n, r)),
      chips.length ? h('div', { class: 'chips' }, chips) : null,
      n.claim_summary ? h('p', { class: 'detail-summary', dir: 'auto' }, n.claim_summary) : null,
      h('div', { class: 'kpis' },
        kpi('Posts', fmtNum(members.length), unv ? `${unv} without verified time` : 'all with verified time'),
        kpi('Platforms', String(plats.length), plats.length ? `${pname(plats[0][0])} leads` : 'none'),
        kpi('First observed', fo ? shortDay(fo.at) : '–', fo ? `${fo.source ? `${pname(fo.source)} · ` : ''}${String(fmtTs(fo.at)).slice(11)}${fo.loose ? ' · loose match' : ''}` : 'in collected data'),
        kpi('Peak day', pk.day ? shortDay(pk.day) : '–', pk.day ? plural(pk.count, 'post') : 'no verified volume'),
        views > 0 ? kpi('Views', compactNum(views), 'sum where platforms show views') : null),
      h('h3', null, 'Posts per day (UTC)'),
      dayBars(n, ctx),
      mixBar(plats),
      n.unverified_member_count ? h('p', { class: 'k' }, `${plural(n.unverified_member_count, 'member post')} without a verified in-window time ${n.unverified_member_count === 1 ? 'is' : 'are'} listed but not counted per day.`) : null));

    const prunedCard = pruningCard(ctx, nid);
    if (prunedCard) view.appendChild(prunedCard);

    const rep = repPosts(ctx, n, 5);
    const top = topPosts(mp, 4);
    if (rep.length) {
      view.appendChild(h('section', { class: 'card' },
        h('div', { class: 'card-head' }, h('h2', null, 'Representative posts'), linkButton(`All ${plural(members.length, 'post')} →`, () => showNarrativePosts(ctx, nid))),
        h('p', { class: 'k' }, 'The posts closest to what this narrative says, from different accounts where possible.'),
        h('div', { class: 'post-rows' }, rep.map((p) => postRow(p, { time: true })))));
    }
    view.appendChild(h('section', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', null, 'Most-seen posts'), rep.length ? null : linkButton(`All ${plural(members.length, 'post')} →`, () => showNarrativePosts(ctx, nid))),
      rep.length ? h('p', { class: 'k' }, 'By views, then engagement. In a mixed narrative these can be its least typical posts.') : null,
      top.length ? h('div', { class: 'post-rows' }, top.map((p) => postRow(p, { time: true }))) : h('p', { class: 'muted' }, 'No member posts found.')));

    const tr = ctx.originByNarr.get(nid);
    const spread = h('section', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, 'How it spread'),
      tr ? h('a', { class: 'more', href: `${base}/origin/${enc(nid)}` }, 'Full spread view →') : null));
    if (tr) {
      const pid = (tr.earliest_post_ids || [])[0];
      const p = pid && ctx.postsById.get(pid);
      spread.appendChild(h('div', { class: 'row-head' }, h('h3', null, 'First observed in collected data'),
        tr.earliest_match === 'weak' ? chip('warn', 'loose match: review') : null));
      spread.appendChild(p ? h('div', { class: 'post-rows' }, postRow(p, { time: true })) : h('p', { class: 'muted' }, 'None recorded.'));
      if (tr.first_strong_post_id && tr.first_strong_post_id !== pid && ctx.postsById.get(tr.first_strong_post_id)) {
        spread.appendChild(h('h3', null, 'First post that clearly carries the narrative'));
        spread.appendChild(h('div', { class: 'post-rows' }, postRow(ctx.postsById.get(tr.first_strong_post_id), { time: true })));
      }
      if (platformSeq(tr).length) spread.appendChild(pathSteps(ctx, tr, n));
    } else spread.appendChild(h('p', { class: 'muted' }, 'No spread trace was produced for this narrative.'));
    view.appendChild(spread);

    const claims = ctx.claimsByNarr.get(nid) || [];
    view.appendChild(h('section', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', null, `Claims · ${claims.length}`), claims.length ? h('a', { class: 'more', href: `${base}/claims/${enc(nid)}` }, 'Open in Claims →') : null),
      claims.length ? claims.map((c) => claimCard(c, ctx, { compact: true })) : h('p', { class: 'muted' }, 'No checkable claims were extracted from this narrative.')));

    const cids = ctx.narrClusters.get(nid) || [];
    view.appendChild(h('section', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', null, `Coordinated groups · ${cids.length}`), cids.length ? h('a', { class: 'more', href: `${base}/coordination` }, 'Open Coordination →') : null),
      cids.length ? groupTable(ctx, cids.map((id) => ctx.clusterById.get(id)).filter(Boolean)) : h('p', { class: 'muted' }, 'No coordinated group’s posts fall in this narrative.')));

    view.appendChild(h('section', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', null, 'All posts'), h('button', { type: 'button', class: 'primary small', onclick: () => showNarrativePosts(ctx, nid) }, `Browse all ${plural(members.length, 'post')}`)),
      (n.keywords || []).length ? h('div', { class: 'chips' }, h('span', { class: 'k' }, 'Keywords'), n.keywords.map((k) => h('span', { class: 'kw', dir: 'auto' }, k))) : null));
  }

  /** Posts per day as bars; days a partially collected platform did not reach are hatched. */
  function dayBars(n, ctx) {
    const days = ctx.days.length ? ctx.days : Object.keys(n.volume_by_day || {}).sort();
    if (!days.length) return h('p', { class: 'muted small' }, 'No verified, in-window posts to chart.');
    const vbd = n.volume_by_day || {};
    const vals = days.map((d) => Number(vbd[d]) || 0);
    const max = Math.max(1, ...vals);
    const pk = peakOf(vbd);
    const cols = { gridTemplateColumns: `repeat(${days.length}, minmax(0, 1fr))` };
    const dense = days.length > 14;
    const every = dense ? Math.ceil(days.length / 8) : 1;
    const bars = h('div', { class: `daybars${dense ? ' dense' : ''}`, style: cols, role: 'img',
      'aria-label': `Posts per day: ${days.map((d, i) => `${shortDay(d)} ${vals[i]}`).join(', ')}` });
    const labels = h('div', { class: `daylabels${dense ? ' dense' : ''}`, style: cols, 'aria-hidden': 'true' });
    let anyGap = false;
    days.forEach((d, i) => {
      const gaps = gapSources(d, ctx.coverFrom, n.volume_by_platform);
      if (gaps.length) anyGap = true;
      bars.appendChild(h('div', {
        class: `daybar${d === pk.day ? ' peak' : ''}${gaps.length ? ' gap' : ''}`,
        title: `${shortDay(d)}: ${plural(vals[i], 'post')}${gaps.length ? ` · not collected that day: ${gaps.map(pname).join(', ')}` : ''}`,
      }, h('span', { class: 'fill', style: { height: vals[i] ? `${Math.max(3, (vals[i] / max) * 100)}%` : '0' } })));
      const show = d === pk.day || i % every === 0;
      labels.appendChild(h('span', { class: d === pk.day ? 'pk' : null }, show ? (d === pk.day ? `${Number(d.slice(8))} · ${vals[i]}` : String(Number(d.slice(8)))) : ''));
    });
    // The caption only when one of this narrative's platforms missed a day.
    const cav = anyGap ? coverageCaveat(ctx) : '';
    return h('div', { class: 'daychart' }, bars, labels,
      cav ? h('p', { class: 'k' }, h('span', { class: 'hatch-key', 'aria-hidden': 'true' }), cav) : null);
  }

  function showNarrativePosts(ctx, nid) {
    const f = getFilter(ctx.id);
    Object.assign(f, { source: '', narrative: nid, verifiedOnly: false, q: '', ids: null, idsLabel: '', shown: 100 });
    const target = `#/run/${enc(ctx.id)}/posts`;
    if (location.hash === target) route(); else location.hash = target;
  }

  // ---- Coordination
  function clusterHeadline(c) {
    const srcs = (c.sources || []).map(pname);
    const sigs = (c.signals || []).map((sg) => (SIGNAL_SHORT[sg.type] || sg.type).toLowerCase());
    return `${fmtNum(clusterSize(c))} ${srcs.length ? `${joinNames(srcs)} ` : ''}accounts acted alike: ${joinNames(sigs)}.`;
  }
  const GENERIC_BENIGN = /do not by themselves prove/i;

  function tabCoordination(view, ctx) {
    view.appendChild(h('p', { class: 'lead' }, 'Accounts that behaved alike in the collected posts. A group is shown only when at least two different signals agree. ',
      'Acting alike is not proof of inauthentic behaviour: check the benign explanation on each group.'));
    if (!ctx.clusters.length) {
      view.appendChild(h('div', { class: 'card' }, h('h3', null, 'No coordinated groups flagged'),
        h('p', { class: 'muted' }, 'A group is only flagged when at least two independent signal types agree (for example near-duplicate text and near-simultaneous posting). ' +
          'No set of accounts in this sample met that bar. This is a statement about the collected sample, not proof that no coordination exists.')));
      return;
    }
    const mains = ctx.mainClusterIds.map((id) => ctx.clusterById.get(id)).filter(Boolean);
    const mainSet = new Set(mains.map((c) => c.id));
    const rest = ctx.clusters.filter((c) => !mainSet.has(c.id)).sort((a, b) => clusterSize(b) - clusterSize(a) || (b.signals || []).length - (a.signals || []).length);
    const nCards = Math.min(3, mains.length);
    mains.slice(0, nCards).forEach((c, i) => view.appendChild(clusterCard(c, ctx, i === 0)));
    if (mains.length > nCards || rest.length) {
      const card = h('section', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, 'Other groups'), h('span', { class: 'k' }, 'ranked by size and signals')));
      if (mains.length > nCards) card.appendChild(groupTable(ctx, mains.slice(nCards)));
      if (rest.length) {
        const sizes = rest.map(clusterSize);
        card.appendChild(foldBox(plural(rest.length, 'smaller group'), `of ${Math.min(...sizes)}–${Math.max(...sizes)} accounts each`, () => groupTable(ctx, rest), !mains.length));
      }
      view.appendChild(card);
    }
    view.appendChild(h('p', { class: 'k sig-legend' }, 'Signals: ',
      SIGNAL_TYPES.map((t) => h('span', null, h('i', { class: 'sd' }), SIGNAL_SHORT[t].toLowerCase())),
      ' · Scores and raw post IDs are in the export and behind “View posts”.'));
    view.appendChild(h('div', { class: 'caveat' }, COORD_CAVEAT));
  }

  function clusterCard(c, ctx, strongest) {
    const sigs = c.signals || [];
    const nids = ctx.clusterNarr[c.id] || [];
    const lead = ctx.postsById.get((c.supporting_post_ids || [])[0]);
    const handles = (c.member_handles || []).length ? c.member_handles : (c.member_author_ids || []);
    const specific = (c.benign_explanations || []).filter((b) => !GENERIC_BENIGN.test(b));
    const card = h('article', { class: 'card ccard' });
    card.appendChild(h('div', { class: 'card-head' },
      h('div', { class: 'chips' }, (c.sources || []).map((x) => platformChip(x)),
        chip('warn', `${sigs.length} signals agree`), h('span', { class: 'k' }, `Group ${c.id}${strongest ? ' · strongest' : ''}`)),
      nids[0] && ctx.narrById.has(nids[0]) ? h('a', { class: 'more', href: narrHref(ctx, nids[0]) }, `Pushes: ${truncate(labelOf(ctx, nids[0]), 56)} →`) : null));
    card.appendChild(h('h2', { class: 'ccard-title' }, c.summary || clusterHeadline(c)));
    card.appendChild(h('div', { class: 'sig-grid' }, sigs.map((sg) => {
      const k = signalKey(sg);
      return h('div', { class: 'sig', title: sg.detail || null },
        h('span', { class: 'k sig-label' }, k.label), h('b', { class: 'sig-key' }, k.key), h('span', { class: 'k' }, k.sub));
    })));
    if (lead) {
      card.appendChild(h('div', { class: 'quote' },
        h('span', { class: 'k' }, `The message they shared${lead.lang ? ` (language: ${lead.lang}, original kept)` : ' (original kept)'}`),
        h('div', { class: 'quote-text', dir: 'auto', lang: validLang(lead.lang) }, compactLines(lead.text, 3, 360) || '(no text)'),
        h('span', { class: 'k' }, 'Example post by ', linkButton(lead.author_handle || lead.post_id, () => openPost(lead.post_id)), ` · ${shortTs(lead.published_at)}`)));
    }
    const acc = h('div', { class: 'accounts' });
    const drawAcc = (all) => {
      clear(acc);
      (all ? handles : handles.slice(0, 6)).forEach((x) => acc.appendChild(h('span', { class: 'acc', dir: 'auto' }, x)));
      if (!all && handles.length > 6) acc.appendChild(linkButton(`+${handles.length - 6} more`, () => drawAcc(true)));
    };
    drawAcc(false);
    card.appendChild(h('div', { class: 'acc-block' }, h('span', { class: 'k' }, `Accounts · ${fmtNum(clusterSize(c))}`), acc));
    card.appendChild(h('div', { class: 'benign' }, h('strong', null, 'Benign explanation to rule out: '),
      specific.length ? specific.join(' ') : 'syndication, a shared newsroom, official reposting or one legitimate operator running several channels can produce the same pattern. Check the posts before drawing conclusions.'));
    const sup = c.supporting_post_ids || [];
    card.appendChild(h('div', { class: 'actions' },
      viewPostsBtn(ctx, sup, `group ${c.id}`, `View ${plural(sup.length, 'post')}`),
      h('button', { type: 'button', class: 'small', onclick: () => download(`group-${c.id}.json`, JSON.stringify({
        cluster: c, narratives: nids, posts: sup.map((pid) => ctx.postsById.get(pid) || { post_id: pid, missing: true }),
      }, null, 2)) }, 'Export group (JSON)')));
    return card;
  }

  function sigDots(c) {
    const fired = new Set((c.signals || []).map((sg) => sg.type));
    return h('span', { class: 'sigdots', title: (c.signals || []).map((sg) => SIGNAL_LABELS[sg.type] || sg.type).join(', ') },
      SIGNAL_TYPES.filter((t) => fired.has(t) || ['near_duplicate_text', 'near_simultaneous_posting', 'similar_creation_dates'].includes(t))
        .map((t) => h('i', { class: fired.has(t) ? null : 'off' })));
  }

  function groupTable(ctx, list) {
    const tbody = h('tbody');
    list.forEach((c) => {
      const lead = ctx.postsById.get((c.supporting_post_ids || [])[0]);
      const nid = (ctx.clusterNarr[c.id] || [])[0];
      const specific = (c.benign_explanations || []).filter((b) => !GENERIC_BENIGN.test(b));
      tbody.appendChild(h('tr', null,
        h('td', { class: 'nowrap g-id' }, h('div', { class: 'chips' }, h('strong', null, c.id), (c.sources || []).map((x) => platformChip(x)))),
        h('td', { class: 'num g-n', 'data-label': 'accounts' }, fmtNum(clusterSize(c))),
        h('td', { class: 'nowrap g-sig', 'data-label': 'signals' }, sigDots(c), h('span', { class: 'tnum' }, String((c.signals || []).length))),
        h('td', { class: 'what' }, h('div', { class: 'clamp-2', dir: 'auto' }, lead ? firstLine(lead.text, 160) : '–'),
          specific.length ? h('div', { class: 'k' }, specific[0]) : null),
        h('td', { class: 'g-narr' }, nid && ctx.narrById.has(nid) ? h('a', { href: narrHref(ctx, nid), dir: 'auto' }, truncate(labelOf(ctx, nid), 48)) : h('span', { class: 'muted' }, '–')),
        h('td', { class: 'nowrap g-act' }, viewPostsBtn(ctx, c.supporting_post_ids, `group ${c.id}`, 'View posts'))));
    });
    return h('div', { class: 'table-wrap' }, h('table', { class: 'tbl tbl-groups' },
      h('thead', null, h('tr', null, [['Group'], ['Accounts', 'num'], ['Signals'], ['What they posted'], ['Narrative'], ['']].map(([t, cls]) => h('th', { class: cls || null }, t)))),
      tbody));
  }

  // ---- Origin & spread
  function tabOrigin(view, ctx, sub) {
    view.appendChild(h('p', { class: 'lead' }, 'Where each main narrative was first observed in the collected data (not necessarily its true origin) and how it moved. ',
      'Direct links (quote, reply, repost, forward, URL) come from platform data; inferred links are only similar content plus timing and may be wrong.'));
    if (!ctx.origin.length) {
      view.appendChild(h('div', { class: 'card muted' }, ctx.narratives.length ? 'No spread traces were produced for these narratives.' : 'No narratives, so no spread traces. See the Narratives tab.'));
      return;
    }
    const withTrace = ctx.ranks.filter((r) => ctx.originByNarr.has(r.narrative_id));
    const mains = withTrace.filter((r) => r.main);
    const nid = sub && ctx.originByNarr.has(sub) ? sub
      : ((mains[0] || withTrace[0] || {}).narrative_id || ctx.origin[0].narrative_id);
    const opts = (mains.length ? mains : withTrace.slice(0, 10)).slice();
    if (!opts.some((r) => r.narrative_id === nid)) opts.unshift(ctx.rankOf.get(nid) || { narrative_id: nid, rank: '–' });
    const sel = h('select', { 'aria-label': 'Narrative', onchange: () => { location.hash = `#/run/${enc(ctx.id)}/origin/${enc(sel.value)}`; } },
      opts.map((r) => h('option', { value: r.narrative_id, selected: r.narrative_id === nid },
        `${r.rank} · ${r.severity ? `${SEVERITY_LABEL[r.severity]} · ` : ''}${truncate(labelOf(ctx, r.narrative_id), 60)} (${fmtNum(narrSize(ctx.narrById.get(r.narrative_id)))})`)));
    const selRank = ctx.rankOf.get(nid) || {};
    view.appendChild(h('div', { class: 'toolbar' }, h('label', { class: 'inline-field' }, h('span', { class: 'k' }, 'Narrative'), sel),
      sevBadge(selRank.severity, (ctx.narrById.get(nid) || {}).severity_reason || null),
      h('a', { class: 'more', href: narrHref(ctx, nid) }, 'Open narrative →')));
    const tr = ctx.originByNarr.get(nid);
    const n = ctx.narrById.get(nid) || { id: nid, member_post_ids: [], volume_by_platform: {}, volume_by_day: {} };
    view.appendChild(firstObservedCard(ctx, tr));
    const seq = platformSeq(tr);
    const move = h('section', { class: 'card' }, h('h2', null, 'How it moved across platforms'));
    if (seq.length) move.appendChild(pathSteps(ctx, tr, n));
    const cav = coverageCaveat(ctx);
    if (cav && seq.some((pl) => ctx.coverFrom[pl])) move.appendChild(h('p', { class: 'k' }, `${cav} The narrative may have arrived there earlier.`));
    move.appendChild(laneGrid(ctx, n, tr));
    view.appendChild(move);
    view.appendChild(momentsCard(ctx, tr, n));
    view.appendChild(linksCard(ctx, tr, ctx.origin.indexOf(tr), n));
  }

  function originPost(ctx, pid) {
    const p = ctx.postsById.get(pid);
    if (!p) return missingPost(pid);
    const meta = [
      p.author_name && p.author_handle ? h('span', { dir: 'auto' }, p.author_name) : null,
      h('span', { class: 'tnum' }, `${shortTs(p.published_at)} UTC`),
      viewsOf(p) >= 0 ? h('span', { class: 'tnum' }, `${compactNum(viewsOf(p))} views`) : null,
    ].filter(Boolean);
    return h('div', { class: 'opost' },
      h('div', { class: 'opost-head' }, platformChip(p.source),
        h('strong', { dir: 'auto' }, p.author_handle || p.author_name || 'unknown author'),
        h('span', { class: 'k opost-meta' }, meta.flatMap((m, i) => (i ? [' · ', m] : [m]))),
        !p.timestamp_verified ? h('span', { class: 'badge unverified' }, 'unverified timestamp') : null),
      h('div', { class: 'opost-text clamp-3', dir: 'auto', lang: validLang(p.lang) }, p.text || '(no text)'),
      h('div', { class: 'opost-foot' }, p.url ? extLink(p.url, `Open on ${pname(p.source)} ↗`) : h('span', { class: 'muted' }, 'no direct link'),
        h('button', { class: 'pid', type: 'button', onclick: () => openPost(pid) }, pid)));
  }

  function firstObservedCard(ctx, tr) {
    const pid = (tr.earliest_post_ids || [])[0];
    const card = h('section', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, 'First observed in collected data'),
      tr.earliest_match === 'weak' ? chip('warn', 'loose match: review', 'The earliest post only loosely matches the narrative') : null));
    card.appendChild(pid ? originPost(ctx, pid) : h('p', { class: 'muted' }, 'None recorded.'));
    if (tr.earliest_match === 'weak') {
      card.appendChild(h('p', { class: 'note-warn' }, `This post only loosely matches the narrative${typeof tr.earliest_similarity === 'number' ? ` (similarity ${tr.earliest_similarity.toFixed(2)})` : ''}. ` +
        'Read it before treating it as the start' + (tr.first_strong_post_id && tr.first_strong_post_id !== pid ? '; the first post that clearly carries the narrative is below.' : '.')));
    }
    if (tr.first_strong_post_id && tr.first_strong_post_id !== pid) {
      card.appendChild(h('h3', null, 'First post that clearly carries the narrative'));
      card.appendChild(originPost(ctx, tr.first_strong_post_id));
    }
    const more = (tr.earliest_post_ids || []).slice(1);
    if (more.length) card.appendChild(h('div', { class: 'small muted' }, 'Also among the earliest: ', pidList(ctx, more, 6)));
    // Evidence first, method after: one short note, the full method text on request.
    const excl = /(\d+) member post\(s\) excluded/.exec(tr.note || '');
    card.appendChild(h('p', { class: 'k opost-note' }, 'Earliest in the collected data, not necessarily the true origin.',
      excl ? ` ${plural(Number(excl[1]), 'member post')} ${Number(excl[1]) === 1 ? 'was' : 'were'} left out (no verified time inside the window, e.g. ads from a longer ads lookback, or duplicates).` : ''));
    if (tr.note) {
      card.appendChild(h('details', { class: 'method-note' }, h('summary', null, 'How this was worked out'),
        h('p', { class: 'k' }, tr.note)));
    }
    return card;
  }

  /** Platform steps in first-seen order with time, gap, volume and coverage. */
  function pathSteps(ctx, tr, n) {
    const seq = platformSeq(tr);
    const pfs = tr.platform_first_seen || {};
    const t0 = Date.parse(pfs[seq[0]]);
    const vbp = n.volume_by_platform || {};
    const largest = platformEntries(vbp)[0];
    const last = ctx.days[ctx.days.length - 1];
    const loose = looseFirstPlatforms(tr);
    const strong = tr.platform_first_seen_strong || {};
    const wrap = h('div', { class: 'path', role: 'list', 'aria-label': 'Platforms in the order the narrative was first seen' });
    seq.forEach((pl, i) => {
      if (i) wrap.appendChild(h('span', { class: 'arrow', 'aria-hidden': 'true' }, '→'));
      const bits = [i === 0 ? 'first post' : gapLabel((Date.parse(pfs[pl]) - t0) / 1000)];
      if (vbp[pl]) bits.push(`${plural(vbp[pl], 'post')}${largest && largest[0] === pl && seq.length > 1 ? ', the largest share' : ''}`);
      const cf = ctx.coverFrom[pl];
      if (cf) bits.push(cf === last ? `collected for ${shortDay(cf)} only` : `collected from ${shortDay(cf)}`);
      const isLoose = loose.includes(pl);
      wrap.appendChild(h('div', { class: `step${isLoose ? ' loose' : ''}`, role: 'listitem' },
        h('div', { class: 'chips' }, platformChip(pl), isLoose ? chip('warn', 'loose match', 'The first post on this platform only loosely matches the narrative') : null),
        h('strong', { class: 'tnum' }, shortTs(pfs[pl])), h('span', { class: 'k' }, bits.filter(Boolean).join(' · ')),
        isLoose ? h('span', { class: 'k' }, strong[pl] ? `first close match ${shortTs(strong[pl])}` : 'no close match on this platform') : null));
    });
    return wrap;
  }

  /** One lane per platform, one cell per day; darker = more posts, hatched = not collected. */
  function laneGrid(ctx, n, tr) {
    const days = ctx.days;
    const counts = {};
    (n.member_post_ids || []).forEach((pid) => {
      const p = ctx.postsById.get(pid);
      if (!p || !analysable(p)) return;
      const d = dayOf(p.published_at);
      counts[p.source] = counts[p.source] || {};
      counts[p.source][d] = (counts[p.source][d] || 0) + 1;
    });
    const seq = platformSeq(tr);
    const lanes = seq.filter((pl) => counts[pl]).concat(Object.keys(counts).filter((pl) => !seq.includes(pl)));
    if (!days.length || !lanes.length) return h('p', { class: 'muted small' }, 'No posts with verified in-window times to lay out.');
    const max = Math.max(1, ...lanes.map((pl) => Math.max(...Object.values(counts[pl]))));
    const pk = peakOf(n.volume_by_day);
    const every = days.length > 14 ? Math.ceil(days.length / 8) : 1;
    const grid = h('div', { class: 'lanes', style: { gridTemplateColumns: `minmax(64px, 104px) repeat(${days.length}, minmax(0, 1fr))` },
      role: 'img', 'aria-label': `Daily posts per platform: ${lanes.map((pl) => `${pname(pl)} ${Object.entries(counts[pl]).map(([d, v]) => `${shortDay(d)} ${v}`).join(', ')}`).join('; ')}` });
    grid.appendChild(h('span'));
    days.forEach((d, i) => grid.appendChild(h('span', { class: 'lane-day' }, i % every === 0 || d === pk.day ? String(Number(d.slice(8))) : '')));
    lanes.forEach((pl) => {
      grid.appendChild(h('span', { class: 'lane-name' }, pname(pl)));
      days.forEach((d) => {
        const v = counts[pl][d] || 0;
        const gap = ctx.coverFrom[pl] && d < ctx.coverFrom[pl];
        grid.appendChild(h('span', {
          class: `cell${gap ? ' gap' : ''}${v ? ' on' : ''}`, title: `${pname(pl)} · ${shortDay(d)}: ${gap ? 'not collected' : plural(v, 'post')}`,
          style: v ? { background: platformColor(pl), opacity: String(0.35 + (0.65 * v) / max) } : null,
        }));
      });
    });
    if (pk.day) {
      grid.appendChild(h('span', { class: 'lane-name' }, 'Peak'));
      days.forEach((d) => grid.appendChild(h('span', { class: `cell mark${d === pk.day ? ' on' : ''}`, title: d === pk.day ? `Peak: ${shortDay(d)}, ${plural(pk.count, 'post')}` : null })));
    }
    return h('div', { class: 'lanes-wrap' }, grid,
      h('p', { class: 'k' }, 'Darker cells = more posts that day. ', Object.keys(ctx.coverFrom).length ? [h('span', { class: 'hatch-key', 'aria-hidden': 'true' }), 'Hatched = platform not collected that day. '] : null, 'Only posts with a verified platform time are placed.'));
  }

  function momentsCard(ctx, tr, n) {
    const items = [];
    const pfs = tr.platform_first_seen || {};
    const mp = (n.member_post_ids || []).map((pid) => ctx.postsById.get(pid)).filter((p) => p && analysable(p))
      .sort((a, b) => Date.parse(a.published_at) - Date.parse(b.published_at));
    const loose = looseFirstPlatforms(tr);
    platformSeq(tr).forEach((pl, i) => {
      const first = mp.find((p) => p.source === pl);
      const at = pfs[pl] || (first && first.published_at);
      const isLoose = loose.includes(pl) || (i === 0 && tr.earliest_match === 'weak');
      items.push({ t: Date.parse(at), when: shortTs(at), body: [
        i === 0 ? `First observed post, on ${pname(pl)}` : `First post on ${pname(pl)}`,
        first ? [': ', postRef(first)] : '.',
        isLoose ? h('span', { class: 'muted' }, ' (loose match)') : null] });
    });
    // Hubs are posts many direct links start from; one outside the narrative is the
    // parent of member posts (e.g. a video whose comments are members), so say so.
    hubsWithMembership(hubsOf(tr).filter((hb) => hb.direct_out >= 3), n.member_post_ids).slice(0, 3).forEach((hb) => {
      const p = ctx.postsById.get(hb.post_id);
      if (!p) return;
      const t = Date.parse(p.published_at);
      items.push({ t: isNaN(t) ? Infinity : t, when: shortTs(p.published_at), body: [
        postRef(p), hb.member ? null : h('span', { class: 'muted' }, ' (parent post, not in this narrative)'),
        viewsOf(p) >= 0 ? ` · ${compactNum(viewsOf(p))} views · ` : ' · ',
        linkButton(`${relationSummary(hb.relations, p.source) || plural(hb.direct_out, 'direct link')} collected`,
          () => showPosts(ctx, directTargets(tr, hb.post_id), `direct links from ${hb.post_id}`))] });
    });
    const pk = peakOf(n.volume_by_day);
    if (pk.day) items.push({ t: Date.parse(`${pk.day}T23:59:59Z`), when: shortDay(pk.day), body: [h('strong', null, 'Peak day'), `: ${plural(pk.count, 'post')}.`] });
    items.sort((a, b) => a.t - b.t);
    return h('section', { class: 'card' }, h('h2', null, 'Key moments'),
      items.length ? h('ol', { class: 'moments' }, items.map((it) => h('li', { class: 'moment' },
        h('span', { class: 'when' }, it.when), h('span', { class: 'mdot', 'aria-hidden': 'true' }), h('div', { class: 'mbody', dir: 'auto' }, it.body))))
        : h('p', { class: 'muted' }, 'No dated moments: no posts with verified times.'));
  }

  function directTargets(tr, pid) {
    return (tr.edges || []).filter((e) => e.type === 'direct' && e.from === pid).map((e) => e.to);
  }

  function linksCard(ctx, tr, k, n) {
    const edges = tr.edges || [];
    const nd = edges.filter((e) => e.type === 'direct').length;
    const ni = edges.length - nd;
    const card = h('section', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, 'Links between posts'),
      h('div', { class: 'legend' }, h('span', null, h('i', { class: 'edge-mark' }), 'direct (platform data)'),
        h('span', null, h('i', { class: 'edge-mark inf' }), 'inferred (similar text + timing)'))));
    if (!edges.length) {
      card.appendChild(h('p', { class: 'muted' }, 'No links: no direct links between collected posts and no inferred sequence met the similarity threshold.'));
      return card;
    }
    const hubs = hubsWithMembership(hubsOf(tr).slice(0, 3), (n || {}).member_post_ids);
    const hubIds = new Set(hubs.map((x) => x.post_id));
    const tbody = h('tbody');
    hubs.forEach((hb) => {
      const p = ctx.postsById.get(hb.post_id);
      tbody.appendChild(h('tr', null,
        h('td', null, h('div', { class: 'cell-post' }, p ? platformChip(p.source) : h('span', { class: 'chip' }, '?'),
          h('span', { class: 'cp-text' }, p ? postRef(p) : pidButton(ctx, hb.post_id),
            hb.member ? null : h('span', { class: 'k' }, ' parent post, not in this narrative')))),
        h('td', { class: 'nowrap' }, h('i', { class: 'edge-mark' }), linkButton(relationSummary(hb.relations, p && p.source) || plural(hb.direct_out, 'link'),
          () => showPosts(ctx, directTargets(tr, hb.post_id), `direct links from ${hb.post_id}`))),
        h('td', { class: 'nowrap' }, hb.inferred_out ? [h('i', { class: 'edge-mark inf' }), plural(hb.inferred_out, 'similar later post')] : '–')));
    });
    const otherD = edges.filter((e) => e.type === 'direct' && !hubIds.has(e.from));
    const otherI = edges.filter((e) => e.type !== 'direct' && !hubIds.has(e.from));
    if (otherD.length || otherI.length) {
      const rels = {};
      otherD.forEach((e) => { const r = e.relation || 'link'; rels[r] = (rels[r] || 0) + 1; });
      tbody.appendChild(h('tr', null, h('td', null, chip('', hubs.length ? 'other posts' : 'all posts')),
        h('td', null, otherD.length ? [h('i', { class: 'edge-mark' }), relationSummary(rels)] : '–'),
        h('td', null, otherI.length ? [h('i', { class: 'edge-mark inf' }), `${plural(otherI.length, 'similar post')} that came later`] : '–')));
    }
    card.appendChild(h('div', { class: 'table-wrap' }, h('table', { class: 'tbl' },
      h('thead', null, h('tr', null, ['Post', 'Direct links from it', 'Inferred follow-ons'].map((t) => h('th', null, t)))), tbody)));
    const holder = h('div');
    let open = '';
    const toggle = (what, btn, build) => {
      open = open === what ? '' : what;
      card.querySelectorAll('.links-actions button').forEach((b) => b.setAttribute('aria-expanded', 'false'));
      clear(holder);
      if (open) { btn.setAttribute('aria-expanded', 'true'); holder.appendChild(build()); }
    };
    const bTable = h('button', { type: 'button', class: 'small', 'aria-expanded': 'false' }, `Show all ${plural(edges.length, 'link')} (${nd} direct, ${ni} inferred)`);
    const bGraph = h('button', { type: 'button', class: 'small', 'aria-expanded': 'false' }, 'Show the link graph');
    bTable.addEventListener('click', () => toggle('table', bTable, () => edgeTable(tr, ctx)));
    bGraph.addEventListener('click', () => toggle('graph', bGraph, () => timeline(tr, ctx, k)));
    card.appendChild(h('div', { class: 'actions links-actions' }, bTable, bGraph));
    card.appendChild(holder);
    return card;
  }

  // BEGIN platformSeq
  // Platform order for an origin trace. platform_sequence defaults to [] in the schema
  // (and [] is truthy in JS), so an empty sequence falls back to platform_first_seen
  // ordered by its first-seen timestamps (first seen in collected data, not "origin").
  function platformSeq(tr) {
    if (tr && Array.isArray(tr.platform_sequence) && tr.platform_sequence.length) return tr.platform_sequence;
    const pfs = (tr && tr.platform_first_seen) || {};
    const t = (pl) => { const v = Date.parse(pfs[pl]); return Number.isNaN(v) ? Infinity : v; };
    return Object.keys(pfs).sort((a, b) => (t(a) - t(b)) || (a < b ? -1 : a > b ? 1 : 0));
  }
  // END platformSeq

  function timeline(tr, ctx, k) {
    const edges = tr.edges || [];
    const ids = new Set(tr.earliest_post_ids || []);
    edges.forEach((e) => { ids.add(e.from); ids.add(e.to); });
    const pts = [], unplotted = [];
    ids.forEach((pid) => {
      const p = ctx.postsById.get(pid);
      if (p && analysable(p)) pts.push({ pid, p, t: Date.parse(p.published_at) });
      else unplotted.push(pid);
    });
    if (!pts.length) return h('div', { class: 'small muted' }, 'No posts with verified timestamps to plot.');
    const lanes = [];
    platformSeq(tr).forEach((pl) => { if (!lanes.includes(pl)) lanes.push(pl); });
    pts.forEach(({ p }) => { if (!lanes.includes(p.source)) lanes.push(p.source); });
    let t0 = Math.min(...pts.map((x) => x.t)), t1 = Math.max(...pts.map((x) => x.t));
    if (t1 - t0 < 3600000) { t0 -= 1800000; t1 += 1800000; }
    const L = 110, R = 24, laneH = 46, top = 16, W = 940;
    const H = top + lanes.length * laneH + 28;
    const xOf = (t) => L + ((t - t0) / (t1 - t0)) * (W - L - R);
    const yOf = (src) => top + lanes.indexOf(src) * laneH + laneH / 2;
    const svg = s('svg', { class: 'timeline', viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Spread timeline' });
    const mid = `m${k}`;
    svg.appendChild(s('defs', null,
      s('marker', { id: `${mid}d`, viewBox: '0 0 10 10', refX: 15, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, s('path', { d: 'M0,0 L10,5 L0,10 z', style: { fill: 'var(--direct)' } })),
      s('marker', { id: `${mid}i`, viewBox: '0 0 10 10', refX: 15, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, s('path', { d: 'M0,0 L10,5 L0,10 z', style: { fill: 'var(--inferred)' } }))));
    lanes.forEach((pl) => {
      const y = yOf(pl);
      svg.appendChild(s('line', { class: 'lane', x1: L, x2: W - R, y1: y, y2: y }));
      svg.appendChild(s('text', { x: 8, y: y + 4 }, pl));
    });
    for (let i = 0; i <= 4; i++) {
      const t = t0 + ((t1 - t0) * i) / 4;
      const x = xOf(t);
      svg.appendChild(s('text', { x, y: H - 8, 'text-anchor': i === 0 ? 'start' : i === 4 ? 'end' : 'middle' }, new Date(t).toISOString().slice(5, 16).replace('T', ' ')));
    }
    const pos = new Map(pts.map((x) => [x.pid, { x: xOf(x.t), y: yOf(x.p.source) }]));
    // Edges first so nodes sit on top. Direct = solid, inferred = dashed.
    edges.forEach((e) => {
      const a = pos.get(e.from), b = pos.get(e.to);
      if (!a || !b) return;
      const direct = e.type === 'direct';
      let d;
      if (Math.abs(a.y - b.y) < 1) {
        const lift = Math.min(30, 10 + Math.abs(b.x - a.x) / 6);
        d = `M${a.x},${a.y} Q${(a.x + b.x) / 2},${a.y - lift} ${b.x},${b.y}`;
      } else d = `M${a.x},${a.y} L${b.x},${b.y}`;
      svg.appendChild(s('path', { d, class: direct ? 'edge-direct' : 'edge-inferred', 'marker-end': `url(#${mid}${direct ? 'd' : 'i'})` },
        s('title', null, `${e.type}: ${e.from} → ${e.to} (${e.relation || e.basis || ''}, ${fmtLag(e.lag_seconds)})`)));
    });
    const earliest = new Set(tr.earliest_post_ids || []);
    pts.forEach(({ pid, p }) => {
      const { x, y } = pos.get(pid);
      svg.appendChild(s('circle', {
        cx: x, cy: y, r: earliest.has(pid) ? 7 : 5.5, class: `node${earliest.has(pid) ? ' earliest' : ''}`, tabindex: 0,
        style: { fill: platformColor(p.source) }, onclick: () => openPost(pid),
        onkeydown: (ev) => { if (ev.key === 'Enter') openPost(pid); },
      }, s('title', null, `${pid} · ${p.author_handle || ''} · ${fmtTs(p.published_at)}${earliest.has(pid) ? ' · earliest observed in collected data' : ''}`)));
    });
    const legend = h('div', { class: 'legend' },
      h('span', null, h('i', { class: 'line', style: { borderColor: 'var(--direct)' } }), 'direct link (platform data)'),
      h('span', null, h('i', { class: 'line dashed', style: { borderColor: 'var(--inferred)' } }), 'inferred (similar content + timing)'),
      h('span', null, 'larger outlined node = earliest observed in collected data; click a node for the post'));
    return h('div', null, h('div', { class: 'timeline-wrap' }, svg), legend,
      unplotted.length ? h('div', { class: 'small muted' }, `${plural(unplotted.length, 'post')} without a verified in-window timestamp not plotted: `, pidList(ctx, unplotted, 6)) : null);
  }

  function edgeTable(tr, ctx) {
    const edges = (tr.edges || []).slice().sort((a, b) => (a.type === b.type ? 0 : a.type === 'direct' ? -1 : 1));
    if (!edges.length) return h('div', { class: 'small muted' }, 'No spread edges: no direct links between collected posts and no inferred sequence met the similarity threshold.');
    const tbody = h('tbody');
    edges.forEach((e) => {
      tbody.appendChild(h('tr', null,
        h('td', null, h('span', { class: `badge ${e.type === 'direct' ? 'direct' : 'inferred'}` }, e.type)),
        h('td', null, pidButton(ctx, e.from)),
        h('td', null, pidButton(ctx, e.to)),
        h('td', { class: 'small' }, e.relation ? h('b', null, e.relation, ' ') : null, e.basis || ''),
        h('td', { class: 'small' }, typeof e.similarity === 'number' ? e.similarity.toFixed(2) : '–'),
        h('td', { class: 'small' }, fmtLag(e.lag_seconds))));
    });
    const nd = edges.filter((e) => e.type === 'direct').length;
    return h('div', null, h('h3', { style: { marginTop: '12px' } }, `Edges (${nd} direct, ${edges.length - nd} inferred)`),
      h('div', { class: 'table-wrap' }, h('table', null,
        h('thead', null, h('tr', null, ['Type', 'From (earlier)', 'To (later)', 'Basis', 'Similarity', 'Lag'].map((t) => h('th', null, t)))), tbody)));
  }

  // ---- Claims
  function claimState(id) {
    S.claimView = S.claimView || {};
    if (!S.claimView[id]) S.claimView[id] = { by: 'narrative', verdict: '' };
    return S.claimView[id];
  }

  function tabClaims(view, ctx, sub) {
    view.appendChild(h('p', { class: 'lead' }, 'Factual claims taken from the narratives and checked against evidence found for the same dates. ',
      'The tool never rules a claim true or false on its own: verdicts are supported, contradicted, disputed or unverified, each citing what was found. ',
      'Posts that report or debunk a claim never count as support.'));
    if (!ctx.claims.length) {
      view.appendChild(h('div', { class: 'card muted' }, 'No checkable claims were extracted from this sample.'));
      return;
    }
    const st = claimState(ctx.id);
    const onlyN = sub && ctx.narrById.has(sub) ? sub : '';
    const vc = {};
    ctx.claims.forEach((c) => { const v = verdictOf(c); vc[v] = (vc[v] || 0) + 1; });
    const tally = h('div', { class: 'tally', role: 'group', 'aria-label': 'Filter by verdict' });
    const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Group claims' });
    const status = h('span', { class: 'k' });
    const body = h('div');
    const draw = () => {
      clear(tally);
      VERDICTS.forEach((v) => tally.appendChild(h('button', {
        type: 'button', class: 'tally-card', 'aria-pressed': String(st.verdict === v),
        onclick: () => { st.verdict = st.verdict === v ? '' : v; draw(); },
      }, h('span', { class: `verdict ${v}` }, v), h('b', null, fmtNum(vc[v] || 0)), h('span', { class: 'k' }, VERDICT_HELP[v]))));
      clear(seg);
      [['narrative', 'By narrative'], ['verdict', 'By verdict']].forEach(([k, label]) => seg.appendChild(h('button', {
        type: 'button', class: st.by === k ? 'on' : null, 'aria-pressed': String(st.by === k), onclick: () => { st.by = k; draw(); },
      }, label)));
      const list = ctx.claims.filter((c) => (!onlyN || c.narrative_id === onlyN) && (!st.verdict || verdictOf(c) === st.verdict));
      status.textContent = `Showing ${plural(list.length, 'claim')}${onlyN ? ` from “${truncate(labelOf(ctx, onlyN), 50)}”` : ''}${st.verdict ? `, verdict ${st.verdict}` : ''}.`;
      clear(body);
      if (!list.length) { body.appendChild(h('div', { class: 'card muted' }, 'No claims match.')); return; }
      if (st.by === 'verdict') {
        VERDICTS.forEach((v) => {
          const cs = list.filter((c) => verdictOf(c) === v);
          if (!cs.length) return;
          body.appendChild(h('h2', { class: 'group-head' }, h('span', { class: `verdict ${v}` }, v), h('span', { class: 'k' }, plural(cs.length, 'claim'))));
          cs.forEach((c) => body.appendChild(claimCard(c, ctx)));
        });
        return;
      }
      const byN = new Map();
      list.forEach((c) => { const key = c.narrative_id && ctx.narrById.has(c.narrative_id) ? c.narrative_id : ''; if (!byN.has(key)) byN.set(key, []); byN.get(key).push(c); });
      const order = Array.from(byN.keys()).sort((a, b) => ((ctx.rankOf.get(a) || {}).rank || 1e9) - ((ctx.rankOf.get(b) || {}).rank || 1e9));
      const isMain = (nid) => !!(ctx.rankOf.get(nid) || {}).main;
      const groupEl = (nid) => {
        const r = ctx.rankOf.get(nid);
        const out = [h('h2', { class: 'group-head' },
          r ? h('span', { class: 'gr-rank' }, `#${r.rank}`) : null,
          nid ? h('a', { href: narrHref(ctx, nid), dir: 'auto' }, labelOf(ctx, nid)) : 'Not tied to a narrative',
          h('span', { class: 'k' }, plural(byN.get(nid).length, 'claim')))];
        byN.get(nid).forEach((c) => out.push(claimCard(c, ctx, { compact: true })));
        return out;
      };
      const shownNow = onlyN ? order : order.filter(isMain);
      const later = onlyN ? [] : order.filter((nid) => !isMain(nid));
      shownNow.forEach((nid) => append(body, groupEl(nid)));
      if (later.length) {
        const n = later.reduce((a, nid) => a + byN.get(nid).length, 0);
        const lvc = {};
        later.forEach((nid) => byN.get(nid).forEach((c) => { const v = verdictOf(c); lvc[v] = (lvc[v] || 0) + 1; }));
        body.appendChild(foldBox(`${plural(n, 'more claim')}`, `from smaller narratives · ${VERDICTS.filter((v) => lvc[v]).map((v) => `${lvc[v]} ${v}`).join(' · ')}`,
          () => later.map(groupEl), !shownNow.length));
      }
    };
    view.appendChild(tally);
    view.appendChild(h('div', { class: 'toolbar' }, seg, h('span', { class: 'spacer' }), status,
      onlyN ? h('a', { class: 'more', href: `#/run/${enc(ctx.id)}/claims` }, 'Show all claims') : null));
    view.appendChild(body);
    draw();
  }

  function evRow(e, refYear) {
    const kind = e.flag === 'old_event' ? 'old' : e.flag === 'unrelated' ? 'unrelated'
      : (['supports', 'contradicts', 'context'].includes(e.supports) ? e.supports : 'context');
    const label = { supports: 'Supports', contradicts: 'Contradicts', context: 'Context', old: 'Old event', unrelated: 'Unrelated' }[kind];
    const date = e.published_at ? shortDay(e.published_at, refYear) : (e.published_raw || '–');
    const tip = kind === 'old' ? 'Published before the claim’s window: about an earlier event, so it cannot count as support.'
      : kind === 'unrelated' ? 'Shares none of the claim’s key terms: a search miss.' : (e.note || null);
    return h('div', { class: `ev ev-${kind}` },
      h('span', { class: 'ev-tag', title: tip }, label),
      h('span', { class: 'ev-src', dir: 'auto' }, (e.publisher || e.source) ? h('span', { class: 'ev-pub' }, `${e.publisher || e.source} · `) : null, extLink(e.url, e.title || e.url)),
      h('span', { class: 'ev-date', title: e.published_raw ? `as given by the provider: ${e.published_raw}` : null }, date));
  }

  function stanceLine(c, ctx) {
    const stances = c.stances || {};
    const spids = c.stance_post_ids || {};
    const total = STANCES.reduce((a, st) => a + (Number(stances[st]) || 0), 0);
    if (!total) return h('div', { class: 'stance-line k' }, 'No collected posts were classified for this claim.');
    return h('div', { class: 'stance-line' },
      h('span', { class: 'k', title: 'Posts that report or debunk a claim never count as support.' }, `How ${plural(total, 'post')} treat it:`),
      STANCES.map((st) => h('button', {
        type: 'button', class: 'stance-pill', disabled: !(spids[st] || []).length,
        onclick: () => showPosts(ctx, spids[st] || [], `posts that ${st} "${c.text}"`),
      }, h('i', { class: `sw stance-${st}`, 'aria-hidden': 'true' }), h('b', null, String(stances[st] || 0)), ` ${st}`)));
  }

  function claimCard(c, ctx, opts) {
    const o = opts || {};
    const verdict = verdictOf(c);
    const conf = Math.max(0, Math.min(1, Number(c.confidence) || 0));
    const pct = Math.round(conf * 100);
    const ev = arrangeEvidence(c.evidence, 3);
    const nEv = (c.evidence || []).length;
    const refYear = String(ctx.run.start || '').slice(0, 4);
    const r = c.narrative_id && ctx.rankOf.get(c.narrative_id);
    const { body } = splitRationale(c.rationale);
    const notes = claimMethodNotes(c);
    const moreParts = [];
    if (ev.rest.length) moreParts.push(plural(ev.rest.length, 'more source'));
    if (body) moreParts.push('full reasoning');
    if (notes || c.method) moreParts.push('how this was checked');
    const related = c.related_post_ids || [];
    const details = (moreParts.length || related.length) ? h('details', { class: 'claim-more' },
      h('summary', null, moreParts.join(' · ') || 'related posts'),
      ev.rest.length ? h('div', { class: 'ev-list' }, ev.rest.map((e) => evRow(e, refYear))) : null,
      body ? h('div', null, h('h4', null, 'Full reasoning'), h('p', { dir: 'auto' }, body)) : null,
      (notes || c.method) ? h('div', null, h('h4', null, 'How this was checked'),
        h('p', { class: 'k' }, [c.method ? `Method: ${c.method}.` : '', notes].filter(Boolean).join(' '))) : null,
      related.length ? h('div', { class: 'actions' }, viewPostsBtn(ctx, related, `claim "${c.text}"`, `View ${plural(related.length, 'related post')}`)) : null) : null;
    return h('article', { class: 'card claim' },
      h('div', { class: 'claim-side' },
        h('span', { class: `verdict ${verdict}` }, verdict),
        h('span', { class: 'k tnum' }, pct ? `confidence ${pct}%` : 'no confidence'),
        h('span', { class: 'confbar', role: 'img', 'aria-label': `confidence ${pct}%` }, h('span', { style: { width: `${pct}%` } })),
        !o.compact && r ? h('a', { class: 'k', href: narrHref(ctx, c.narrative_id), title: labelOf(ctx, c.narrative_id) }, `narrative #${r.rank}`) : null),
      h('div', { class: 'claim-main' },
        h('h3', { class: 'claim-text', dir: 'auto' }, c.text || '(no claim text)'),
        claimBottomLine(c) ? h('p', { class: 'bottom-line', dir: 'auto' }, claimBottomLine(c)) : null,
        ev.allUnrelated ? h('div', { class: 'flag' }, h('strong', null, 'Evidence search missed: '),
          `${nEv === 1 ? 'the only result does' : `none of the ${nEv} results do`} not mention the claim’s key terms. Worth a manual search.`) : null,
        !nEv ? h('div', { class: 'flag neutral' }, 'No external evidence was retrieved; the verdict cannot rest on anything beyond the collected posts.') : null,
        ev.shown.length ? h('div', { class: 'ev-list' }, ev.shown.map((e) => evRow(e, refYear))) : null,
        stanceLine(c, ctx),
        details));
  }

  // ---- Exports
  function tabExports(view, ctx) {
    if (STATIC) {
      const d = ctx.data;
      const bundle = JSON.stringify(d, null, 2);
      const jsonl = (d.posts || []).map((p) => JSON.stringify(p)).join('\n') + ((d.posts || []).length ? '\n' : '');
      const base = (ctx.run.run_id || 'socmint-run').replace(/[^A-Za-z0-9._-]+/g, '_');
      const items = [
        [`${base}.json`, 'Full run (all sections)', bundle, 'application/json'],
        ['posts.jsonl', 'Posts (JSON Lines)', jsonl, 'application/x-ndjson'],
      ];
      ['narratives', 'clusters', 'origin', 'claims', 'source_status', 'run'].forEach((k) => {
        items.push([`${k}.json`, k, JSON.stringify(d[k] === undefined ? [] : d[k], null, 2), 'application/json']);
      });
      view.appendChild(h('div', { class: 'card' }, h('h3', null, 'Download embedded data'),
        h('p', { class: 'small muted' }, 'This report embeds the run data. CSV files (posts.csv, narratives.csv, ...) are next to this report in the run folder.'),
        h('div', { class: 'row' }, items.map(([name, label, content, type]) => h('button', { type: 'button', onclick: () => download(name, content, type) }, label)))));
      return;
    }
    const files = [
      ['posts.csv', 'Posts (CSV)'], ['posts.jsonl', 'Posts (JSON Lines)'], ['narratives.json', 'Narratives (JSON)'],
      ['narratives.csv', 'Narratives (CSV)'], ['clusters.json', 'Coordination (JSON)'], ['clusters.csv', 'Coordination (CSV)'],
      ['origin.json', 'Origin & spread (JSON)'], ['claims.json', 'Claims (JSON)'], ['claims.csv', 'Claims (CSV)'],
      ['source_status.json', 'Source status (JSON)'], ['source_status.csv', 'Source status (CSV)'], ['run.json', 'Run metadata (JSON)'],
    ];
    view.appendChild(h('div', { class: 'card' }, h('h3', null, 'Report & archive'),
      h('div', { class: 'row' },
        // ?inline=true makes the server send Content-Disposition: inline so the report renders in the new tab.
        h('a', { class: 'btn primary', href: `/api/runs/${enc(ctx.id)}/files/report.html?inline=true`, target: '_blank', rel: 'noopener' }, 'Open standalone report (report.html)'),
        h('a', { class: 'btn', href: `/api/runs/${enc(ctx.id)}/files/report.html`, download: 'report.html' }, 'Download report.html'),
        h('a', { class: 'btn', href: `/api/runs/${enc(ctx.id)}/export.zip`, download: `${ctx.id}.zip` }, 'Everything (export.zip)')),
      h('p', { class: 'small muted' }, 'report.html is self-contained and opens offline. CSV files are UTF-8 with BOM; cells beginning with = + - @ are prefixed with a quote to prevent spreadsheet formula injection (the JSON files are unmodified).')));
    view.appendChild(h('div', { class: 'card' }, h('h3', null, 'Individual files'),
      h('div', { class: 'row' }, files.map(([name, label]) => h('a', { class: 'btn', href: `/api/runs/${enc(ctx.id)}/files/${enc(name)}`, download: name }, label)))));
  }

  // ------------------------------------------------------------------ boot
  async function boot() {
    let meta = null;
    if (!STATIC) {
      try { meta = await getMeta(); } catch (e) {
        buildShell(null);
        shell.main.appendChild(h('div', { class: 'banner bad' }, `Cannot reach the local server: ${e.message}`));
        return;
      }
    }
    buildShell(meta);
    window.addEventListener('hashchange', route);
    route();
  }
  boot();
})();
