"""Logging with secret redaction. Use `get_logger(__name__)` everywhere."""

from __future__ import annotations

import logging

from socmint.config import redact


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        record.msg = redact(msg)
        record.args = ()
        return True


_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(RedactingFilter())
    root = logging.getLogger("socmint")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    # third-party HTTP libraries log full URLs (which can contain keys) at INFO
    for noisy in ("httpx", "httpcore", "telethon", "urllib3", "googleapiclient"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name if name.startswith("socmint") else f"socmint.{name}")
