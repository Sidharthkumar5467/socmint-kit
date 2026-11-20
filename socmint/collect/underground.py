"""Known underground / leak Telegram channels and the vocabulary that finds new ones.

Owner decision 2026-11-20: the Telegram collector can look for data-leak, infostealer and
breach channels and search THOSE for the topic, instead of only channels whose name matches
the topic. A hit — an underground channel selling or discussing data about the subject — is
far more valuable than general chatter.

Two inputs:
  * a bundled list of known PUBLIC underground channels (data/underground_channels.json),
    extracted from deepdarkCTI (https://github.com/fastfire/deepdarkCTI, GPL-3.0; see
    NOTICE). Invite-only (`t.me/+...`) and offline/expired entries were removed, because
    the tool never joins a channel and can only read public ones.
  * VOCAB: leak/infostealer terms searched in Telegram's own directory each run, so newly
    created channels (not in the static list) are found too.

The list is attribution-only reference data, not evidence: every post the tool keeps still
comes from a live, date-bounded search of the channel, with its own platform timestamp.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data" / "underground_channels.json"

# Leak / infostealer vocabulary for Telegram's directory search (contacts.search matches
# channel names and usernames). High-signal terms that name a channel's purpose, not every
# term a post might contain; kept short because each is one directory request.
VOCAB: tuple[str, ...] = (
    "logs", "stealer logs", "combolist", "database leak", "leaks", "breach",
    "cloud logs", "cracked", "fullz", "rdp access",
)

# Substrings that confirm a directory match is really a leak/underground channel (its name
# or username contains one), so "logs" matching @EpsteinDocuments is dropped.
LEAK_TERMS: tuple[str, ...] = (
    "log", "leak", "breach", "stealer", "combo", "cloud", "dump", "cracked", "crack",
    "fraud", "carding", "card", "fullz", "rdp", "database", "dbase", "market", "cvv",
    "sniff", "clouded", "redline", "lumma", "raccoon", "vidar", "pwn", "exploit",
)


def looks_underground(*texts: str | None) -> bool:
    """True if any text (channel title / username) contains a leak term."""
    blob = " ".join(t.lower() for t in texts if t)
    return any(term in blob for term in LEAK_TERMS)


# Categories most likely to carry brand-relevant leak data, searched first under the cap.
CATEGORY_WEIGHT = {"infostealer": 0, "market": 1, "forum": 2, "ransomware": 3,
                   "threat_actor": 4}


def _cat(c: dict) -> int:
    return min((CATEGORY_WEIGHT.get(k, 9) for k in c.get("categories") or []), default=9)


@lru_cache(maxsize=1)
def curated_channels() -> tuple[dict, ...]:
    """Bundled known public underground channels, ordered for search under a cap.

    Round-robin across categories (one infostealer, one market, one forum, … then the next
    of each), so a capped run covers a SPREAD of channel types rather than filling entirely
    with one category. Infostealer channels often post credentials as file attachments the
    tool does not download, while forums and markets describe sales in text, so a mix is
    more likely to yield a text match for the topic.

    Each dict: username, status, label, categories. Empty if the data file is missing (the
    feature then relies on directory discovery alone)."""
    try:
        data = json.loads(_DATA.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    rows = [c for c in data.get("channels", []) if c.get("username")]
    buckets: dict[int, list[dict]] = {}
    for c in sorted(rows, key=lambda c: c["username"].lower()):
        buckets.setdefault(_cat(c), []).append(c)
    ordered: list[dict] = []
    for i in range(max((len(b) for b in buckets.values()), default=0)):
        for w in sorted(buckets):
            if i < len(buckets[w]):
                ordered.append(buckets[w][i])
    return tuple(ordered)


def curated_usernames() -> set[str]:
    """Lowercased usernames of the bundled channels (fast membership test)."""
    return {c["username"].lower() for c in curated_channels()}
