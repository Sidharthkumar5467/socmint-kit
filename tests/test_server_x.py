"""X web_session routes and /api/meta readiness (socmint/server/app.py).

A fake socmint.collect.x_web is injected into sys.modules: no browser, no network.
"""

from __future__ import annotations

import sys
import types

import pytest
from fastapi.testclient import TestClient

from socmint import doctor
from socmint.config import save_config
from socmint.server.app import create_app
from socmint.server.jobs import JobManager

COOKIE = "authtoken-SECRETVALUE-0123456789abcdef"


def client_for(app) -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def fake_x_web(*, authorized: bool = False, user: str | None = None,
               source: str | None = None, login_state: str = "idle",
               reason: str = "", browser: str | None = "chrome",
               login_error: str | None = None) -> types.ModuleType:
    mod = types.ModuleType("socmint.collect.x_web")
    mod.calls = []  # type: ignore[attr-defined]

    def session_status(cfg):
        mod.calls.append("status")
        return {"configured": authorized, "authorized": authorized, "user": user,
                "source": source, "session_path": "/tmp/x_storage_state.json",
                "browser": browser, "login_state": login_state, "reason": reason}

    def start_login(cfg, timeout_s: int = 300):
        mod.calls.append("login")
        return {"ok": True, "error": None}

    def logout(cfg):
        mod.calls.append("logout")
        return {"ok": True, "message": "saved X browser session deleted"}

    def get_login_state():
        return {"state": login_state, "error": login_error}

    def ensure_browser():
        mod.calls.append("ensure_browser")
        return {"ok": True, "browser": "chrome", "message": "Google Chrome found"}

    mod.session_status = session_status  # type: ignore[attr-defined]
    mod.start_login = start_login  # type: ignore[attr-defined]
    mod.logout = logout  # type: ignore[attr-defined]
    mod.ensure_browser = ensure_browser  # type: ignore[attr-defined]
    mod.login_state = get_login_state  # type: ignore[attr-defined]
    return mod


def install(monkeypatch, mod: types.ModuleType | None) -> None:
    import socmint.collect as collect_pkg

    if mod is None:
        # Simulate "module not importable": a None entry makes `import` raise ImportError.
        monkeypatch.setitem(sys.modules, "socmint.collect.x_web", None)
        monkeypatch.delattr(collect_pkg, "x_web", raising=False)
        return
    monkeypatch.setitem(sys.modules, "socmint.collect.x_web", mod)
    monkeypatch.setattr(collect_pkg, "x_web", mod, raising=False)


@pytest.fixture
def client() -> TestClient:
    return client_for(create_app())


@pytest.fixture
def ro_client() -> TestClient:
    return client_for(create_app(read_only=True))


@pytest.fixture(autouse=True)
def _browser_found(monkeypatch):
    monkeypatch.setattr(doctor, "x_browser_status", lambda: ("chrome", "Google Chrome found"))


# ---------------------------------------------------------------------- routes
def test_status_login_logout_with_fake_module(monkeypatch, client: TestClient) -> None:
    mod = fake_x_web(authorized=True, user="analyst", source="browser_login")
    install(monkeypatch, mod)
    st = client.get("/api/x/status").json()
    assert st["authorized"] is True and st["user"] == "analyst"
    assert st["source"] == "browser_login" and st["login_state"] == "idle"
    assert client.post("/api/x/login").json() == {"ok": True, "error": None}
    assert client.post("/api/x/logout").json()["ok"] is True
    assert mod.calls == ["status", "login", "logout"]


def test_logout_passes_x_web_message_through(monkeypatch, client: TestClient) -> None:
    # Regression: the server used to add its own "browser session removed" note even when
    # nothing was deleted. Only x_web's own message (what actually happened) is returned.
    install(monkeypatch, fake_x_web(authorized=True, source="pasted_cookies"))
    save_config({"X_AUTH_TOKEN": COOKIE, "X_CT0": "ct0-SECRETVALUE-987654"})
    out = client.post("/api/x/logout").json()
    assert out == {"ok": True, "message": "saved X browser session deleted"}
    assert COOKIE not in str(out)


def test_status_carries_redacted_login_error(monkeypatch, client: TestClient) -> None:
    # Regression: the reason a login failed (e.g. no browser -> run --prefetch) never
    # reached the GUI because session_status only has the bare login_state string.
    save_config({"X_CT0": "ct0-SECRETVALUE-987654"})
    install(monkeypatch, fake_x_web(
        login_state="failed",
        login_error="no browser: run socmint doctor --prefetch; ct0-SECRETVALUE-987654"))
    st = client.get("/api/x/status").json()
    assert st["login_state"] == "failed"
    assert "--prefetch" in st["login_error"]
    assert "ct0-SECRETVALUE-987654" not in str(st)


def test_status_login_error_is_none_without_failure(monkeypatch, client: TestClient) -> None:
    install(monkeypatch, fake_x_web())
    assert client.get("/api/x/status").json()["login_error"] is None


class _BusyJobs(JobManager):
    """A JobManager that reports an active run and refuses to start another."""

    def busy(self) -> bool:
        return True

    def start(self, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("a run must not start")


class _IdleJobs(JobManager):
    def start(self, **kw):
        raise AssertionError("a run must not start while an X login is waiting")


def test_login_refused_while_a_run_is_active(monkeypatch) -> None:
    # Regression: login + collection must never drive the owner's X account in two
    # browsers at once (and must not rewrite the session file a run is reading).
    mod = fake_x_web(authorized=True)
    install(monkeypatch, mod)
    c = client_for(create_app(jobs=_BusyJobs()))
    r = c.post("/api/x/login")
    assert r.status_code == 409 and "run is in progress" in r.text
    assert "login" not in mod.calls  # no browser was opened


def test_run_with_x_refused_while_login_waiting(monkeypatch) -> None:
    install(monkeypatch, fake_x_web(login_state="waiting"))
    c = client_for(create_app(jobs=_IdleJobs()))
    r = c.post("/api/runs", json={"topic": "t", "start": "2026-11-01", "end": "2026-11-10",
                                  "sources": ["x"]})
    assert r.status_code == 409 and "X login is in progress" in r.text


@pytest.mark.parametrize("method,path", [("get", "/api/x/status"), ("post", "/api/x/login"),
                                         ("post", "/api/x/logout")])
def test_x_routes_blocked_in_read_only(monkeypatch, ro_client: TestClient, method, path) -> None:
    mod = fake_x_web()
    install(monkeypatch, mod)
    r = getattr(ro_client, method)(path)
    assert r.status_code == 403
    assert mod.calls == []  # nothing ran: no browser was opened


def test_x_routes_when_module_missing(monkeypatch, client: TestClient) -> None:
    install(monkeypatch, None)
    st = client.get("/api/x/status")
    assert st.status_code == 200
    body = st.json()
    assert body["configured"] is False and body["authorized"] is False
    assert "unavailable" in body["reason"]
    assert client.post("/api/x/login").json()["ok"] is False


def test_x_route_missing_function_is_reported(monkeypatch, client: TestClient) -> None:
    mod = fake_x_web()
    del mod.start_login
    install(monkeypatch, mod)
    out = client.post("/api/x/login").json()
    assert out["ok"] is False and "AttributeError" in out["reason"]


def test_x_route_errors_are_redacted(monkeypatch, client: TestClient) -> None:
    save_config({"X_AUTH_TOKEN": COOKIE})
    mod = fake_x_web()

    def boom(cfg, timeout_s: int = 300):
        raise RuntimeError(f"browser crashed with cookie auth_token={COOKIE}")

    mod.start_login = boom  # type: ignore[attr-defined]
    install(monkeypatch, mod)
    r = client.post("/api/x/login")
    assert r.status_code == 400
    assert COOKIE not in r.text and "REDACTED" in r.text


def test_status_reason_is_redacted(monkeypatch, client: TestClient) -> None:
    save_config({"X_CT0": "ct0-SECRETVALUE-987654"})
    install(monkeypatch, fake_x_web(reason="cookie ct0-SECRETVALUE-987654 expired"))
    body = client.get("/api/x/status").json()
    assert "ct0-SECRETVALUE-987654" not in str(body)


# ------------------------------------------------------------------- meta readiness
def x_row(client: TestClient) -> dict:
    rows = {s["name"]: s for s in client.get("/api/meta").json()["sources"]}
    return rows["x"]


def test_meta_web_session_authorized_is_ready(monkeypatch, client: TestClient) -> None:
    install(monkeypatch, fake_x_web(authorized=True, user="analyst", source="browser_login"))
    row = x_row(client)
    if row["kind"] is None:
        pytest.skip("X collector not importable")
    assert row["ready"] is True and row["provider"] == "web_session"
    assert "@analyst" in row["reason"]


def test_meta_web_session_authorized_without_browser_is_fail(monkeypatch,
                                                              client: TestClient) -> None:
    # Regression: the collector returns `failed` (no browser), so readiness must say
    # "fail", not merely "skip" (the GUI chip reads `outcome`).
    install(monkeypatch, fake_x_web(authorized=True, source="pasted_cookies", browser=None))
    monkeypatch.setattr(doctor, "x_browser_status", lambda: (None, "no browser: install Chrome"))
    row = x_row(client)
    if row["kind"] is None:
        pytest.skip("X collector not importable")
    assert row["ready"] is False and row["outcome"] == "fail"
    assert row["reason"].startswith("will fail")


def test_meta_web_session_not_authorized_is_skipped(monkeypatch, client: TestClient) -> None:
    install(monkeypatch, fake_x_web(authorized=False))
    row = x_row(client)
    if row["kind"] is None:
        pytest.skip("X collector not importable")
    assert row["ready"] is False
    assert "log in to X from Settings or paste X_AUTH_TOKEN + X_CT0" in row["reason"]


def test_meta_web_session_module_missing_is_not_ready(monkeypatch, client: TestClient) -> None:
    install(monkeypatch, None)
    row = x_row(client)
    assert row["ready"] is False


def test_meta_other_provider_uses_key_rules(monkeypatch, client: TestClient) -> None:
    mod = fake_x_web(authorized=True)
    install(monkeypatch, mod)
    save_config({"X_PROVIDER": "twitterapi_io"})
    row = x_row(client)
    if row["kind"] is None:
        pytest.skip("X collector not importable")
    assert row["ready"] is False and "TWITTERAPI_IO_KEY" in row["reason"]
    assert "status" not in mod.calls  # web session not consulted for other providers


# ----------------------------------------------------------------------- settings
def test_settings_validates_x_provider(client: TestClient) -> None:
    assert client.post("/api/settings", json={"X_PROVIDER": "scraper"}).status_code == 400
    r = client.post("/api/settings", json={"X_PROVIDER": "official"})
    assert r.status_code == 200 and r.json()["config"]["X_PROVIDER"] == "official"
    assert client.post("/api/settings", json={"X_MAX_SCROLLS": "0"}).status_code == 400
    assert client.post("/api/settings", json={"X_BROWSER_HEADLESS": "maybe"}).status_code == 400


# ------------------------------------------------- against the REAL x_web module
# Regression: every other test uses a hand-written fake; these pin the routes to the real
# module's return shapes. No browser is launched (_open_browser is patched) and the
# session files live in the tmp SOCMINT_HOME set by conftest.
@pytest.fixture
def real_x_web(monkeypatch):
    import importlib

    pytest.importorskip("playwright")
    mod = importlib.import_module("socmint.collect.x_web")
    # Fresh login state per test (module global, restored afterwards).
    monkeypatch.setattr(mod, "_login", {"state": "idle", "error": None, "thread": None})
    monkeypatch.setattr(mod, "_detect_browser", lambda: "chrome")

    def no_browser(**kw):
        raise mod.NoBrowserError("no browser in tests")

    monkeypatch.setattr(mod, "_open_browser", no_browser)
    return mod


def _write_state(mod, expires: float) -> None:
    import json
    import time as _t

    path = mod.session_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cookies": [
        {"name": "auth_token", "value": "doc-example-not-a-real-token", "domain": ".x.com",
         "path": "/", "expires": _t.time() + expires, "httpOnly": True, "secure": True,
         "sameSite": "None"}], "origins": []}), encoding="utf-8")


def test_real_status_no_session(real_x_web, client: TestClient) -> None:
    st = client.get("/api/x/status").json()
    assert st["authorized"] is False and st["configured"] is False
    assert st["source"] is None and st["login_state"] == "idle"
    assert st["login_error"] is None and st["browser"] == "chrome"
    assert "log in" in st["reason"]


def test_real_status_pasted_cookies_and_logout(real_x_web, client: TestClient) -> None:
    save_config({"X_AUTH_TOKEN": COOKIE, "X_CT0": "ct0-SECRETVALUE-987654"})
    st = client.get("/api/x/status").json()
    assert st["authorized"] is True and st["source"] == "pasted_cookies"
    assert COOKIE not in str(st)
    out = client.post("/api/x/logout").json()
    assert out["ok"] is True
    # Nothing was deleted and pasted cookies remain: the message must say exactly that.
    assert out["message"].startswith("no saved X browser session")
    assert "remain" in out["message"] and "note" not in out
    assert COOKIE not in str(out)


def test_real_status_saved_session_then_logout(real_x_web, client: TestClient) -> None:
    _write_state(real_x_web, expires=86400)
    st = client.get("/api/x/status").json()
    assert st["authorized"] is True and st["source"] == "browser_login"
    assert "doc-example-not-a-real-token" not in str(st)
    out = client.post("/api/x/logout").json()
    assert out["ok"] is True and out["message"] == "saved X browser session deleted"
    assert not real_x_web.session_file().exists()
    assert client.get("/api/x/status").json()["authorized"] is False


def test_real_status_expired_session_not_authorized(real_x_web, client: TestClient) -> None:
    _write_state(real_x_web, expires=-3600)
    st = client.get("/api/x/status").json()
    assert st["authorized"] is False and st["configured"] is True
    assert "expired" in st["reason"]


def test_real_login_failure_reason_reaches_status(real_x_web, client: TestClient) -> None:
    r = client.post("/api/x/login").json()
    assert r == {"ok": True, "error": None}
    t = real_x_web._login["thread"]
    t.join(timeout=10)
    assert not t.is_alive()
    st = client.get("/api/x/status").json()
    assert st["login_state"] == "failed"
    assert st["login_error"] == real_x_web.NO_BROWSER and "--prefetch" in st["login_error"]
