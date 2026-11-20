"""Telegram login helpers for the GUI (synchronous API over Telethon).

Telegram login is a two-step flow (phone -> code sent to the user's Telegram app ->
code [+ 2FA password]). Between steps the SAME Telethon client must stay alive, so this
module owns one background thread running a dedicated asyncio event loop; every public
function submits a coroutine to it and waits for the result.

The session file lives in ~/.socmint/sessions/ (outside the repo, see
socmint.collect.telegram.session_path). Errors are redacted before being returned.
The client is disconnected after every completed step except `send_code`, so the
collector can open the same SQLite session file without a lock conflict.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Coroutine
from typing import Any

from socmint.collect.telegram import session_path
from socmint.config import has, redact
from socmint.util.log import get_logger

log = get_logger(__name__)

STEP_TIMEOUT = 60.0  # seconds a single GUI step may take


class _LoopThread:
    """One daemon thread + event loop, created lazily and shared by all calls."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._thread is None or not self._thread.is_alive():
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(target=self._loop.run_forever,
                                                name="socmint-telegram-auth", daemon=True)
                self._thread.start()
            return self._loop

    def run(self, coro: Coroutine[Any, Any, Any], timeout: float = STEP_TIMEOUT) -> Any:
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop())
        return fut.result(timeout=timeout)


_runner = _LoopThread()
# Pending login state, only touched from the runner's loop.
_pending: dict[str, Any] = {"client": None, "phone": None, "hash": None, "session": None}


def _make_client(cfg: dict[str, str]) -> Any:
    """Create a Telethon client bound to the session file. Overridden in tests."""
    from telethon import TelegramClient

    path = session_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    return TelegramClient(str(path), int(cfg["TELEGRAM_API_ID"]), cfg["TELEGRAM_API_HASH"],
                          receive_updates=False)


def _err(e: BaseException, cfg: dict[str, str]) -> str:
    name = type(e).__name__
    friendly = {
        "PhoneNumberInvalidError": "invalid phone number (use international format, e.g. +44...)",
        "PhoneCodeInvalidError": "incorrect code",
        "PhoneCodeExpiredError": "code expired - request a new one",
        "PasswordHashInvalidError": "incorrect 2FA password",
        "ApiIdInvalidError": "TELEGRAM_API_ID / TELEGRAM_API_HASH rejected by Telegram",
        "PhoneNumberBannedError": "this phone number is banned by Telegram",
    }
    if name == "FloodWaitError":
        return f"Telegram rate-limited this request; retry in {getattr(e, 'seconds', '?')}s"
    return redact(friendly.get(name) or f"{name}: {e}", cfg)


async def _disconnect(client: Any) -> None:
    try:
        await client.disconnect()
    except Exception:  # noqa: BLE001
        pass


async def _drop_pending() -> None:
    c = _pending.get("client")
    _pending.update({"client": None, "phone": None, "hash": None, "session": None})
    if c is not None:
        await _disconnect(c)


def _user_label(me: Any) -> str | None:
    if me is None:
        return None
    if getattr(me, "username", None):
        return f"@{me.username}"
    name = " ".join(p for p in (getattr(me, "first_name", None),
                                getattr(me, "last_name", None)) if p)
    return name or str(getattr(me, "id", "")) or None


# ======================================================================== public API


def status(cfg: dict[str, str]) -> dict[str, Any]:
    """Is Telegram configured and is the session authorized? Never raises."""
    path = session_path(cfg)
    out: dict[str, Any] = {"configured": has(cfg, "TELEGRAM_API_ID", "TELEGRAM_API_HASH"),
                           "authorized": False, "user": None, "premium": None,
                           "session_path": str(path), "reason": ""}
    if not out["configured"]:
        out["reason"] = "TELEGRAM_API_ID and TELEGRAM_API_HASH are not set"
        return out
    if not path.is_file():
        out["reason"] = "not logged in (no session file) - send a login code"
        return out

    async def _check() -> None:
        # Reuse a pending client if a login is in progress on this same session.
        client = _pending["client"] if _pending.get("session") == str(path) else None
        own = client is None
        client = client or _make_client(cfg)
        try:
            if not client.is_connected():
                await client.connect()
            if await client.is_user_authorized():
                me = await client.get_me()
                out.update(authorized=True, user=_user_label(me),
                           premium=bool(getattr(me, "premium", False)),
                           reason="authorized")
            else:
                out["reason"] = "session file exists but is not authorized - log in again"
        finally:
            if own:
                await _disconnect(client)

    try:
        _runner.run(_check())
    except Exception as e:  # noqa: BLE001
        out["reason"] = f"could not check session: {_err(e, cfg)}"
    return out


def send_code(cfg: dict[str, str], phone: str) -> dict[str, Any]:
    """Step 1: ask Telegram to send a login code to the user's Telegram app."""
    if not has(cfg, "TELEGRAM_API_ID", "TELEGRAM_API_HASH"):
        return {"ok": False, "error": "set TELEGRAM_API_ID and TELEGRAM_API_HASH first"}
    phone = (phone or "").strip().replace(" ", "")
    if not phone:
        return {"ok": False, "error": "phone number is required"}

    async def _send() -> dict[str, Any]:
        await _drop_pending()
        client = _make_client(cfg)
        await client.connect()
        if await client.is_user_authorized():
            await _disconnect(client)
            return {"ok": True, "error": None, "already_authorized": True}
        try:
            sent = await client.send_code_request(phone)
        except BaseException:
            await _disconnect(client)
            raise
        _pending.update({"client": client, "phone": phone,
                         "hash": getattr(sent, "phone_code_hash", None),
                         "session": str(session_path(cfg))})
        return {"ok": True, "error": None}

    try:
        return _runner.run(_send())
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": _err(e, cfg)}


def sign_in(cfg: dict[str, str], code: str, password: str | None = None) -> dict[str, Any]:
    """Step 2: submit the code (and the 2FA password when the account has one)."""

    async def _sign() -> dict[str, Any]:
        from telethon.errors import SessionPasswordNeededError

        client = _pending.get("client")
        if client is None:
            return {"ok": False, "needs_password": False, "premium": None,
                    "error": "no login in progress - send a code first"}
        try:
            if code and code.strip():
                try:
                    await client.sign_in(_pending["phone"], code.strip(),
                                         phone_code_hash=_pending["hash"])
                except SessionPasswordNeededError:
                    if not password:
                        return {"ok": False, "needs_password": True, "premium": None,
                                "error": "this account has two-step verification - "
                                         "enter its password"}
                    await client.sign_in(password=password)
            elif password:
                await client.sign_in(password=password)
            else:
                return {"ok": False, "needs_password": False, "premium": None,
                        "error": "enter the code Telegram sent"}
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "PhoneCodeExpiredError":
                await _drop_pending()
            return {"ok": False, "needs_password": False, "premium": None,
                    "error": _err(e, cfg)}
        me = await client.get_me()
        premium = bool(getattr(me, "premium", False))
        log.info("telegram login complete for %s (premium=%s)", _user_label(me), premium)
        await _drop_pending()  # disconnect so the collector can open the session file
        return {"ok": True, "needs_password": False, "premium": premium, "error": None,
                "user": _user_label(me)}

    try:
        return _runner.run(_sign())
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "needs_password": False, "premium": None, "error": _err(e, cfg)}


def logout(cfg: dict[str, str]) -> dict[str, Any]:
    """Log the session out on Telegram's side (if possible) and delete the file."""
    path = session_path(cfg)

    async def _out() -> None:
        await _drop_pending()
        if path.is_file() and has(cfg, "TELEGRAM_API_ID", "TELEGRAM_API_HASH"):
            client = _make_client(cfg)
            try:
                await client.connect()
                if await client.is_user_authorized():
                    await client.log_out()  # also removes the session file
            finally:
                await _disconnect(client)

    error = None
    try:
        _runner.run(_out())
    except Exception as e:  # noqa: BLE001
        error = _err(e, cfg)
    for p in (path, path.with_name(path.name + "-journal")):
        try:
            p.unlink(missing_ok=True)
        except OSError as e:
            error = error or redact(str(e), cfg)
    return {"ok": not path.exists(), "error": error}
