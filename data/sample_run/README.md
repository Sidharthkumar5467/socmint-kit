# Sample run (bundled for offline replay)

A real run, so the GUI and `socmint replay data/sample_run` work with no keys and no network
on first launch. The collected posts are unedited, and the analysis is the run's own (no
later re-analysis).

- Command: `socmint run --topic "bank of america" --window 7d --sources x,telegram,youtube,linkedin --match phrase`
  (topic match: exact phrase; LinkedIn posts via the `voyager` provider).
- Collected 2026-11-20 (UTC), window 2026-11-13 → 2026-11-20, from four sources:
  - X: the owner's session, headless browser.
  - Telegram: Premium global search **plus** per-channel date-bounded search of the channels
    found by topic discovery and the bundled leak/underground list (searched first, so the
    targeted channels are not starved; global search runs last with the remaining cap).
  - YouTube: Data API v3.
  - LinkedIn posts: the owner's session (Voyager content search).
  - (LinkedIn Ads was removed from the tool; it is not in this run.)
- Result: 2,287 posts (2,255 in scope), 269 narratives, 32 coordination clusters, 30 claims.
  Collection and analysis took about 6 minutes.
- Coverage, as each source's status reports it:
  - X, YouTube and LinkedIn posts: posts on all 8 days.
  - Telegram: global search has no date filter, so it balances the days client-side within
    its page limit; the discovered and leak/underground channels are searched by date (most
    returned nothing for this topic this week — a true negative, not an error).
- Coordination clusters fired on near-duplicate text, near-simultaneous posting and similar
  account-creation dates. The reused-image signal is computed (avatars/thumbnails are
  perceptually hashed) but did not fire this week — no two accounts shared an image.
- Claims cite serper.dev news results and Google Fact Check. Only items dated in or near the
  window may count as support.
- `raw/` holds the platform responses each post's `raw_ref` points to. For LinkedIn posts it
  holds per-item extracts, because LinkedIn's responses carry the collecting account's own
  data.

## Provenance

- Analysis is the run's own (no `reanalyze`). Model `claude-haiku-5-5`, 241 calls,
  670,657 input and 104,579 output tokens. `llm_log.jsonl` keeps every call.
- Payment card and credential data in post text is masked at write time: card numbers keep
  the first 6 and last 4 digits (`BIN******last4`), a CVV next to one becomes `***`, and a
  leaked password next to an email/handle is masked while the identifier is kept. (This
  week's collection contained no card dumps, so no card masking was needed; the masking runs
  regardless.) Platform IDs, numbers and file names are not touched, and every `.json` /
  `.jsonl` file still parses.
- The run was checked for every configured credential value, the collecting accounts'
  identity and local file paths. None were present.
- Media URLs point to the platforms, so images and thumbnails do not load offline.
