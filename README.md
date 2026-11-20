# SOCMINT Narrative & Coordination Module

A local tool that takes a **topic + date range** and investigates what is being said about it
across social platforms, with the evidence behind every finding. Enter a topic and window,
pick your sources, and it produces:

1. **Posts** in range — account, text, media, timestamp, engagement, direct link.
2. **Narratives** — what is being claimed, grouped, with volume over time and per platform.
3. **Coordinated accounts** — groups acting together, each explained by the signals that fired
   (near-identical text, near-simultaneous posting, shared creation dates, reused images,
   mutual amplification) and benign explanations.
4. **Origin & spread** — the earliest posts in the collected data and how a narrative moved
   across platforms, with **direct** links (reply/repost/quote/forward) kept separate from
   **inferred** sequences.
5. **Misinformation assessment** — claims with cited evidence, a verdict
   (`supported | contradicted | disputed | unverified`), confidence and rationale.
6. **Exports** — JSON and CSV, a browser dashboard, and a standalone HTML report.

Everything runs on your machine. No data leaves it except the API calls the sources and the
LLM require.

---

## Requirements

- **Python 3.11+** (the launcher re-execs with 3.11/3.12/3.13 from your PATH if needed).
- **macOS, Linux or Windows.**
- **Google Chrome** installed — required only for the X and LinkedIn collectors, which drive
  your own logged-in browser session.
- Network access on first run (to install dependencies and download a ~100 MB multilingual
  embedding model). After that, replay mode works fully offline.

All Python dependencies are installed automatically into a local `.venv/` on first launch;
you do not run `pip` yourself.

---

## Quick start

```bash
python run.py
```

This checks your Python version, creates/uses `.venv/`, installs dependencies, creates a
local config file, and opens the GUI in your browser at `http://127.0.0.1:<port>`. A second
launch skips install and opens straight to the GUI.

**Try it with no keys and no network** (opens the bundled sample run read-only):

```bash
python run.py --run data/sample_run     # or:  socmint replay data/sample_run
```

---

## API keys & accounts you need

Credentials are entered and saved in the **GUI → Settings** page (stored in a local,
git-ignored config file — never in the repo). A missing credential simply downgrades that
source to "skipped"; the run still completes and the status panel explains every gap.

| Source / feature | What you need | Where to get it | Required? |
|---|---|---|---|
| **X (Twitter)** | Your own logged-in **x.com** session (default provider `web_session`). Log in once from Settings in a local browser, or paste the `auth_token` and `ct0` cookies. (TwitterAPI.io or the official API can be selected instead.) | Your x.com account | For X results |
| **Telegram** | **`TELEGRAM_API_ID`** + **`TELEGRAM_API_HASH`**, then a one-time login code entered in the GUI. A **Premium** account enables global topic search; without it the tool uses seed/discovered channels. | my.telegram.org → API development tools | For Telegram results |
| **YouTube** | **`YOUTUBE_API_KEY`** (Data API v3). | Google Cloud Console → enable *YouTube Data API v3* → create an API key | For YouTube results |
| **LinkedIn** | **`LINKEDIN_COOKIE`** — the `Cookie` header (or `li_at` value) of your own logged-in linkedin.com session. | Your linkedin.com account (browser dev tools) | For LinkedIn results |
| **Analysis (LLM)** | **`ANTHROPIC_API_KEY`** — narrative labels, claim extraction and stance (model `claude-haiku-5-5`). | console.anthropic.com | Recommended (without it you get keyword labels) |
| **Claim evidence** | **`SERPER_API_KEY`** (serper.dev) and/or **`GOOGLE_FACTCHECK_API_KEY`**. | serper.dev; Google Cloud Console (Fact Check Tools API) | Optional (improves verdicts) |

**Account notes**
- The X and LinkedIn collectors use *your own* logged-in accounts. Collection is paced and
  stops at once on any rate limit, checkpoint or challenge; heavy use can get an account
  rate-limited, so use an account you are comfortable using for research.
- Telegram is **read-only**: the tool never joins, messages, buys or downloads from a channel.
- Keys live only in your local config (`~/.socmint/config.json` or a project `.env`), which is
  git-ignored. They are never printed in logs or committed.

---

## Usage

The GUI is the control surface — enter keys, pick sources, set topic + window, start a run,
watch live progress, browse results and export, all in the browser.

A thin CLI mirrors it for automation:

```bash
socmint doctor [--prefetch]     # show what will run / be skipped, and cache the embedding model
socmint run --topic "Acme Bank" --window 7d [--sources x,telegram,youtube,linkedin] [--match phrase|words]
socmint run --topic "Acme Bank" --start 2026-11-01 --end 2026-11-07 [--limit N]
socmint replay <run_dir>        # open a saved run read-only (no keys, no network)
socmint report <run_dir>        # rebuild exports + report.html from a saved run (offline)
socmint reanalyze <run_dir>     # re-run only the analysis on a saved run's posts
socmint delete <run_dir>        # permanently delete a run
```

Window: presets of the last 1, 3 or 7 days, or a custom UTC range up to 30 days. A run has no
time limit by default; you can stop it from the progress page (which discards the run).

---

## Output

Each run is written to its own directory (`runs/<run_id>/`) containing `posts.jsonl`,
`posts.csv`, `narratives`, `clusters`, `origin`, `claims`, `source_status`, a `brief`, a
self-contained `report.html`, and the raw platform responses under `raw/`. The same data is
browsable in the GUI dashboard. A bundled example lives in `data/sample_run/`.

---

## Project layout

```
run.py                 one-command launcher (standard library only)
socmint/               the package
  collect/             per-source collectors (x, telegram, youtube, linkedin)
  analyze/             narratives, coordination, origin, claims, brief
  server/, web/        FastAPI server and the browser GUI
data/sample_run/       bundled offline demo run
docs/                  methodology, decisions, AI usage, evaluation, platform notes
tests/                 pytest suite
```

---

## Documentation

The written methodology, design decisions, AI-usage and verification notes, evaluation
(real false positives / negatives), and platform analysis are provided **separately**
alongside this submission.

---

## Security & privacy

- Secrets, sessions and collected personal data are never committed: `.env*`, `*.session`,
  `sessions/`, `cookies*` and `runs/` are git-ignored, and a gitleaks pre-commit hook scans
  staged changes.
- Payment-card and credential data in collected posts is masked at write time.
- The server only ever listens on `127.0.0.1`.
