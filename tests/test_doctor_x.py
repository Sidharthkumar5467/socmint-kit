"""`socmint doctor` X rows for X_PROVIDER=web_session. No browser, no network."""

from __future__ import annotations

import sys
import types

import pytest

from socmint import doctor

COOKIE = "authtoken-SECRETVALUE-0123456789abcdef"
CT0 = "ct0value-SECRETVALUE-fedcba987654"


def fake_x_web(**status) -> types.ModuleType:
    mod = types.ModuleType("socmint.collect.x_web")
    mod.ensured = 0  # type: ignore[attr-defined]
    base = {"configured": False, "authorized": False, "user": None, "source": None,
            "session_path": "/tmp/s.json", "browser": None, "login_state": "idle", "reason": ""}

    def session_status(cfg):
        return {**base, **status}

    def ensure_browser():
        mod.ensured += 1
        return {"ok": True, "browser": "chromium", "message": "installed Playwright Chromium"}

    mod.session_status = session_status  # type: ignore[attr-defined]
    mod.ensure_browser = ensure_browser  # type: ignore[attr-defined]
    return mod


def install(monkeypatch, mod) -> None:
    import socmint.collect as collect_pkg

    monkeypatch.setitem(sys.modules, "socmint.collect.x_web", mod)
    if mod is None:
        monkeypatch.delattr(collect_pkg, "x_web", raising=False)
    else:
        monkeypatch.setattr(collect_pkg, "x_web", mod, raising=False)


def report_text(cfg: dict) -> str:
    rep = doctor.collect_report(cfg, check_telegram=False, check_model=False)
    return doctor.format_report(rep)


@pytest.fixture(autouse=True)
def _chrome(monkeypatch):
    monkeypatch.setattr(doctor, "x_browser_status", lambda: ("chrome", "Google Chrome found"))


def x_row(cfg: dict) -> dict:
    return {r["name"]: r for r in doctor.source_readiness(cfg)}["x"]


def test_web_session_authorized_browser_login(monkeypatch) -> None:
    install(monkeypatch, fake_x_web(configured=True, authorized=True, user="analyst",
                                    source="browser_login"))
    cfg = {"X_PROVIDER": "web_session"}
    row = x_row(cfg)
    if row["kind"] is None:
        pytest.skip("X collector not importable")
    assert row["ready"] is True
    out = report_text(cfg)
    assert "X provider: web_session" in out
    assert "X session: authorized as @analyst (source: browser_login)" in out
    assert "X browser: chrome" in out


def test_web_session_not_authorized_will_skip(monkeypatch) -> None:
    install(monkeypatch, fake_x_web())
    cfg = {"X_PROVIDER": "web_session"}
    row = x_row(cfg)
    if row["kind"] is None:
        pytest.skip("X collector not importable")
    assert row["ready"] is False and "skipped_no_credentials" in row["reason"]
    assert row["outcome"] == "skip"
    out = report_text(cfg)
    assert "NO  X session: not authorized" in out
    assert "log in to X from Settings or paste X_AUTH_TOKEN + X_CT0" in out


def test_web_session_pasted_cookies_never_printed(monkeypatch) -> None:
    install(monkeypatch, fake_x_web(configured=True, authorized=True, source="pasted_cookies",
                                    reason=f"cookie auth_token={COOKIE}"))
    cfg = {"X_PROVIDER": "web_session", "X_AUTH_TOKEN": COOKIE, "X_CT0": CT0}
    out = report_text(cfg)
    assert "source: pasted_cookies" in out
    assert COOKIE not in out and CT0 not in out
    assert COOKIE[-6:] not in out


def test_web_session_without_browser_is_not_ready(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "x_browser_status",
                        lambda: (None, "no browser: install Google Chrome"))
    install(monkeypatch, fake_x_web(configured=True, authorized=True, source="browser_login"))
    row = x_row({"X_PROVIDER": "web_session"})
    if row["kind"] is None:
        pytest.skip("X collector not importable")
    assert row["ready"] is False and "no browser" in row["reason"]
    # Regression: the collector returns `failed` here, so the status column and the
    # readiness outcome must say "will fail", not "will skip".
    assert row["outcome"] == "fail"
    out = report_text({"X_PROVIDER": "web_session"})
    assert "NO  X browser: none" in out
    x_line = next(ln for ln in out.splitlines() if ln.startswith("x "))
    assert "will fail" in x_line.split("x_session=")[0]
    assert "will skip" not in x_line


def test_outcome_column_matches_readiness(monkeypatch) -> None:
    install(monkeypatch, fake_x_web())
    rows = doctor.source_readiness({"X_PROVIDER": "web_session"})
    for r in rows:
        assert r["outcome"] in ("run", "skip", "fail")
        assert (r["outcome"] == "run") == bool(r["ready"])
    assert doctor.row_outcome({"ready": True}) == "run"  # rows without `outcome`
    assert doctor.row_outcome({"ready": False}) == "skip"


def test_web_session_module_missing(monkeypatch) -> None:
    install(monkeypatch, None)
    st = doctor.x_session_status({})
    assert st["authorized"] is False and "unavailable" in st["reason"]
    assert doctor.ensure_x_browser({})[0] is None


def test_other_provider_shows_provider_only(monkeypatch) -> None:
    install(monkeypatch, fake_x_web(authorized=True))
    out = report_text({"X_PROVIDER": "official"})
    assert "X provider: official" in out and "X session" not in out


def test_prefetch_calls_ensure_browser_for_web_session(monkeypatch, capsys) -> None:
    mod = fake_x_web()
    install(monkeypatch, mod)
    monkeypatch.setattr(doctor, "load_config", lambda: {"X_PROVIDER": "web_session"})
    fake_embed = types.ModuleType("socmint.analyze.embed")
    fake_embed.prefetch_model = lambda cfg: (True, "model cached")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "socmint.analyze.embed", fake_embed)
    monkeypatch.setattr(doctor, "collect_report", lambda cfg: {})
    monkeypatch.setattr(doctor, "format_report", lambda rep: "")
    assert doctor.main(prefetch=True) == 0
    assert mod.ensured == 1
    assert "X browser: chromium: installed Playwright Chromium" in capsys.readouterr().out


def test_prefetch_skips_browser_for_api_provider(monkeypatch, capsys) -> None:
    mod = fake_x_web()
    install(monkeypatch, mod)
    monkeypatch.setattr(doctor, "load_config", lambda: {"X_PROVIDER": "twitterapi_io"})
    fake_embed = types.ModuleType("socmint.analyze.embed")
    fake_embed.prefetch_model = lambda cfg: (True, "model cached")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "socmint.analyze.embed", fake_embed)
    monkeypatch.setattr(doctor, "collect_report", lambda cfg: {})
    monkeypatch.setattr(doctor, "format_report", lambda rep: "")
    doctor.main(prefetch=True)
    assert mod.ensured == 0 and "not needed" in capsys.readouterr().out


def test_find_chrome_uses_filesystem_only(monkeypatch, tmp_path) -> None:
    fake = tmp_path / "chrome"
    fake.write_text("")
    monkeypatch.setattr(doctor, "CHROME_PATHS", (str(fake),))
    assert doctor.find_chrome() == str(fake)
    monkeypatch.setattr(doctor, "CHROME_PATHS", (str(tmp_path / "missing"),))
    monkeypatch.setattr(doctor.shutil, "which", lambda exe: None)
    assert doctor.find_chrome() is None


def test_playwright_chromium_detection(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
    assert doctor.playwright_chromium_installed() is False
    (tmp_path / "chromium-1140").mkdir()
    assert doctor.playwright_chromium_installed() is True
