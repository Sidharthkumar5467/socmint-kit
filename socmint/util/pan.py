"""Mask payment card numbers (PANs) in text we write to disk (owner decision 2026-11-19).

Card-dump channels post full card numbers with expiry and CVV. The run files are the
evidence we hand on (posts.jsonl, posts.csv, the HTML report, raw/ responses), so card
data is masked once, where posts are written by the pipeline, and in raw payloads as they
are saved. The evidence keeps what an analyst needs: the BIN (first 6 digits, which names
the issuing bank), the last 4, the expiry, and the link to the original post.

  5275170000003226|08|2030|956   ->  527517******3226|08|2030|***
  5275 1700 0000 3226            ->  5275 17** **** 3226

A digit run is treated as a card number only when ALL of these hold, so tweet IDs, phone
numbers, order numbers and timestamps are left alone:
- 13 to 19 digits, optionally grouped by single spaces or dashes;
- not glued to letters, '+', '/', '=', '.', '#', '-' or '_' (URLs, IDs, decimals, phones);
- not inside a URL;
- an issuer prefix of a card network with a length that network issues (Visa 4: 13/16,
  Mastercard 51-55 / 2221-2720: 16, Amex 34/37: 15, Diners 30/36/38/39: 14-16,
  Discover/UnionPay/JCB/Maestro 6 and 35: 16-19), so 19-digit X status IDs never match;
- a valid Luhn check digit.
The CVV is masked only next to a masked card: in "PAN|MM|YY(YY)|CVV" style lines and after
a "CVV"/"CVC" label in the same text.

JSON is never masked as serialised text (that corrupted raw Telegram responses: 19-digit
numeric ids such as "id", "access_hash" and "document_id" starting with 6 pass the
Discover/UnionPay rules and were replaced unquoted). `mask_json` walks the decoded object
and masks ONLY string values: never numbers, never dict keys, and never a string that is
entirely an identifier (all digits, "scheme:digits" like "mtproto:604...", or the value of
an ID field: id, *_id, access_hash, document_id, file_reference, hash). Card numbers in
real posts appear inside free text, which is still masked.

Credentials (owner decision 2026-11-20, for underground/leak channels): leak and
infostealer posts paste `email:password`, `url:login:password` and labelled `PASS:` lines.
The SECRET is masked and the identifier kept, so an analyst sees WHOSE credential leaked
(email, domain, login) without the working password:

  victim@acme.com:Hunter2!        ->  victim@acme.com:***
  https://mail.acme.com | user42 | S3cr3t   ->  https://mail.acme.com | user42 | ***
  PASSWORD: correct-horse         ->  PASSWORD: ***

Only a value that follows an email / URL / host identifier or an explicit PASS label is
masked, so ordinary text with colons (times, ratios, "http://x") is left alone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# a run of digit groups separated by single spaces or dashes; candidates are cut from it
_RUN_RE = re.compile(r"\d+(?:[ \-]\d+)*")
_BAD_BEFORE = set("+/=.#-_")
_ESCAPES = ("\\n", "\\r", "\\t")  # JSON-escaped whitespace in raw files
# card lengths to try, most common first (a run "PAN MM YYYY" must yield the 16-digit PAN)
_LENGTHS = (16, 15, 19, 14, 13, 18, 17)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# right after a PAN: |MM|YY|CVV  or  |MM/YY|CVV  (separators | : ; , / and spaces)
_TAIL_RE = re.compile(
    r"(?P<pre>\s*[|:;,]\s*(?:0?[1-9]|1[0-2])\s*[|/\-]\s*(?:\d{4}|\d{2})\s*[|:;,]\s*)"
    r"(?P<cvv>\d{3,4})(?!\d)")
_CVV_LABEL_RE = re.compile(r"(?i)(\b(?:cvv2?|cvc2?|cid|csc)\b\s*[:=\-]?\s*)(\d{3,4})(?!\d)")

# --- credentials (leak / infostealer posts). Mask the secret, keep the identifier.
MASK_SECRET = "***"
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}")
# A line's left side identifies the account: an email, or a URL/host (so "time 10:30" and
# "ratio 3:2" never match). The secret is the last colon/semicolon/pipe-separated field.
_CRED_URL = r"(?:https?://|www\.)[^\s:|;]+(?::\d{2,5})?[^\s:|;]*"
_CRED_HOST = r"(?:[A-Za-z0-9\-]+\.)+[A-Za-z]{2,24}"
_CRED_EMAILP = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}"
# identifier [sep (user)] sep secret   e.g. url:user:pass, email:pass, host|user|pass
_COMBO_RE = re.compile(
    rf"(?P<id>{_CRED_URL}|{_CRED_EMAILP}|{_CRED_HOST})"
    r"(?P<mid>\s*[:|;]\s*[^\s:|;]{1,120})?"
    r"(?P<sep>\s*[:|;]\s*)(?P<secret>[^\s:|;]{1,120})\s*$")
# labelled password: "PASS:" / "PASSWORD =" / "pwd:" (not "passport")
_PASS_LABEL_RE = re.compile(r"(?i)(?P<lbl>\b(?:pass(?:word)?|pwd)\b\s*[:=]\s*)(?P<secret>\S[^\n]*)$")


def _mask_credentials_line(line: str) -> str:
    """Mask the secret in one line of free text (idempotent)."""
    m = _PASS_LABEL_RE.search(line)
    if m and m.group("secret").strip() != MASK_SECRET:
        return line[: m.start("secret")] + MASK_SECRET
    m = _COMBO_RE.search(line.rstrip())
    if m and m.group("secret") != MASK_SECRET:
        idv = m.group("id")
        # Mask only a real credential shape: an email (`email:pass`), a URL
        # (`https://host:user:pass`), or a host WITH a user field (`host|user|pass`). A bare
        # `word.word:token` (e.g. a log label "claims.stance:N4") is not masked.
        if "@" in idv or "/" in idv or m.group("mid"):
            # Keep the user field (mid) if present; mask only the trailing secret.
            return line[: m.start("secret")] + MASK_SECRET + line[m.end("secret"):]
    return line


def mask_credentials(text: str) -> str:
    """Mask passwords/secrets in email:pass, url:user:pass and labelled PASS lines."""
    if not text:
        return text
    low = text.lower()
    if ":" not in text and "|" not in text and ";" not in text \
            and "pass" not in low and "pwd" not in low:
        return text
    return "\n".join(_mask_credentials_line(ln) for ln in text.split("\n"))


def mask_free_text(text: str) -> str:
    """Everything we mask in a human-written string: card numbers then credentials."""
    return mask_credentials(mask_pans(text))


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _network_ok(d: str) -> bool:
    """Issuer prefix and a length that network actually issues. Lengths are kept tight on
    purpose: 19-digit platform IDs (X status IDs, Telegram hashes) must not pass."""
    n, two, four = len(d), int(d[:2]), int(d[:4])
    if d[0] == "4":  # Visa
        return n in (13, 16)
    if 51 <= two <= 55 or 2221 <= four <= 2720:  # Mastercard
        return n == 16
    if two in (34, 37):  # Amex
        return n == 15
    if two in (30, 36, 38, 39):  # Diners
        return 14 <= n <= 16
    if d[0] == "6" or two == 35:  # Discover, UnionPay, JCB, Maestro
        return 16 <= n <= 19
    return False


def is_pan(candidate: str) -> bool:
    d = re.sub(r"\D", "", candidate)
    return 13 <= len(d) <= 19 and _network_ok(d) and luhn_ok(d)


def _mask_digits(s: str) -> str:
    """Keep the first 6 and last 4 digits; every other digit -> '*'; separators kept."""
    n = sum(c.isdigit() for c in s)
    out, i = [], 0
    for c in s:
        if c.isdigit():
            out.append(c if i < 6 or i >= n - 4 else "*")
            i += 1
        else:
            out.append(c)
    return "".join(out)


def mask_pans(text: str) -> str:
    """Return `text` with card numbers (and CVVs next to them) masked. Idempotent."""
    if not text or not any(c.isdigit() for c in text):
        return text
    urls = [m.span() for m in _URL_RE.finditer(text)]

    def in_url(a: int, b: int) -> bool:
        return any(s <= a and b <= e for s, e in urls)

    out: list[str] = []
    pos = 0
    hit = False
    for run in _RUN_RE.finditer(text):
        rs, re_ = run.span()
        before = text[rs - 1] if rs else ""
        after = text[re_] if re_ < len(text) else ""
        if (before.isalpha() and text[max(0, rs - 2):rs] not in _ESCAPES) or (
                before and before in _BAD_BEFORE) or after.isalpha() or after in ("/", "_"):
            continue  # glued to a word, URL, ID, decimal or phone number
        groups = [(g.start(), g.end()) for g in re.finditer(r"\d+", run.group())]
        i = 0
        while i < len(groups):
            found = None
            for want in _LENGTHS:
                n = 0
                for j in range(i, len(groups)):
                    n += groups[j][1] - groups[j][0]
                    if n >= want:
                        break
                if n != want:
                    continue
                a, b = rs + groups[i][0], rs + groups[j][1]
                if not in_url(a, b) and is_pan(text[a:b]):
                    found = (a, b, j)
                    break
            if found is None:
                i += 1
                continue
            a, b, j = found
            hit = True
            out.append(text[pos:a])
            out.append(_mask_digits(text[a:b]))
            pos = b
            tail = _TAIL_RE.match(text, b)
            if tail:
                out.append(tail.group("pre") + "*" * len(tail.group("cvv")))
                pos = tail.end()
            i = j + 1
            if pos > rs + (groups[i][0] if i < len(groups) else len(run.group())):
                break  # the CVV tail consumed the rest of this run
    if not hit:
        return text
    out.append(text[pos:])
    masked = "".join(out)
    return _CVV_LABEL_RE.sub(lambda m: m.group(1) + "*" * len(m.group(2)), masked)


# dict keys whose string values are identifiers, never free text
_ID_KEYS = frozenset({"id", "access_hash", "document_id", "file_reference", "hash"})
# free-text keys: an all-digit string here is still masked (a post that is only a card)
_TEXT_KEYS = frozenset({"text", "message", "caption", "content", "body", "description",
                        "title", "snippet", "prompt", "output", "system", "full_text"})
_ID_STRING_RE = re.compile(r"[A-Za-z][\w.\-]*:\d+")  # e.g. "mtproto:6046556857407906435"


def _is_id_key(key: str | None) -> bool:
    if not key:
        return False
    k = key.lower()
    return k in _ID_KEYS or k.endswith("_id") or k.endswith("_ids")


def _is_identifier(value: str, key: str | None) -> bool:
    if _is_id_key(key):
        return True
    v = value.strip()
    if v.isdigit() and (key or "").lower() not in _TEXT_KEYS:
        return True
    return bool(_ID_STRING_RE.fullmatch(v))


def mask_json(obj: Any, _key: str | None = None) -> Any:
    """Return a copy of a decoded JSON value with card numbers masked in string values
    only. Numbers, booleans, dict keys and identifier strings are left untouched."""
    if isinstance(obj, str):
        return obj if _is_identifier(obj, _key) else mask_free_text(obj)
    if isinstance(obj, dict):
        return {k: mask_json(v, k if isinstance(k, str) else None) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        # a list under an ID key ("ids": [...]) holds identifiers too
        return [mask_json(v, _key) for v in obj]
    return obj


def _json_indent(text: str) -> int | None:
    """Best guess of the indent a JSON document was written with (None = compact)."""
    m = re.match(r"\s*[\[{]\n( +)", text)
    return len(m.group(1)) if m else None


def mask_json_text(text: str) -> str:
    """Mask a serialised JSON document by walking its decoded value; re-serialised only
    when something changed. Text that is not JSON is masked as plain text."""
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return mask_pans(text)
    masked = mask_json(obj)
    if masked == obj:
        return text
    out = json.dumps(masked, ensure_ascii=False, indent=_json_indent(text))
    return out + ("\n" if text.endswith("\n") else "")


def mask_jsonl_text(text: str) -> str:
    """mask_json_text() for every line of a JSON Lines document."""
    lines = text.split("\n")
    return "\n".join(mask_json_text(ln) if ln.strip() else ln for ln in lines)


def mask_text_for(name: str, text: str) -> str:
    """Mask `text` according to the file name it is stored under (.json/.jsonl are
    walked as JSON, everything else is masked as text)."""
    low = name.lower()
    if low.endswith(".jsonl") or low.endswith(".ndjson"):
        return mask_jsonl_text(text)
    if low.endswith(".json"):
        return mask_json_text(text)
    return mask_free_text(text)


def dumps_masked(obj: Any, **kw: Any) -> str:
    """json.dumps(mask_json(obj)) -- the safe way to write a JSON payload."""
    kw.setdefault("ensure_ascii", False)
    default = kw.pop("default", None) or str
    plain = json.loads(json.dumps(obj, ensure_ascii=False, default=default))
    return json.dumps(mask_json(plain), **kw)


def mask_post(post):  # socmint.schema.Post (untyped here to keep this module import-light)
    """Mask card numbers in a Post's free-text fields in place; returns the post."""
    post.text = mask_free_text(post.text or "")
    return post


def mask_file(path: Path) -> bool:
    """Mask card numbers in a text file on disk, in place. Returns True if it changed.
    Binary files (not UTF-8) are left alone."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    masked = mask_text_for(path.name, raw)
    if masked == raw:
        return False
    tmp = path.with_name(path.name + ".masktmp")
    tmp.write_text(masked, encoding="utf-8")
    tmp.replace(path)
    return True


def mask_tree(root: Path) -> int:
    """mask_file() every regular file under `root`; returns how many files changed."""
    if not root.is_dir():
        return 0
    return sum(1 for p in sorted(root.rglob("*")) if p.is_file() and mask_file(p))


__all__ = ["mask_pans", "mask_json", "mask_json_text", "mask_jsonl_text", "mask_text_for",
           "dumps_masked", "mask_post", "mask_file", "mask_tree", "is_pan", "luhn_ok"]
