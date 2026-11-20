"""UTC time helpers. All stored timestamps are ISO8601 UTC with a trailing 'Z'."""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from email.utils import parsedate_to_datetime


def now_utc() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        raise ValueError("naive datetime; refuse to guess a timezone")
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def now_iso() -> str:
    return iso(now_utc())  # type: ignore[return-value]


def parse_ts(value: str | int | float | datetime | None) -> datetime | None:
    """Parse a platform timestamp into tz-aware UTC.

    Accepts ISO8601 (with Z or offset), RFC 2822 (e.g. X's 'Wed Oct 10 20:19:24 +0000
    2018' is handled separately below), and unix epoch seconds. Returns None when the
    value carries no timezone information, because guessing would make it unverifiable.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else None
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1e12:  # milliseconds
            v /= 1000.0
        return datetime.fromtimestamp(v, tz=UTC)
    s = str(value).strip()
    if s.isdigit():
        return parse_ts(int(s))
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(UTC) if dt.tzinfo else None
    except ValueError:
        pass
    # Twitter legacy format: "Wed Oct 10 20:19:24 +0000 2018"
    try:
        dt = datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
        return dt.astimezone(UTC)
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(s)
        return dt.astimezone(UTC) if dt and dt.tzinfo else None
    except (TypeError, ValueError):
        return None


def day_start(d: date) -> datetime:
    return datetime.combine(d, time(0, 0, 0), tzinfo=UTC)


def day_end(d: date) -> datetime:
    return datetime.combine(d, time(23, 59, 59), tzinfo=UTC)


def parse_date(s: str) -> date:
    return date.fromisoformat(s.strip()[:10])
