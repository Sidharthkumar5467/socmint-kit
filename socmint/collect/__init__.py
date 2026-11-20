"""Collector registry.

Built-in sources are registered by name. Owner-supplied plugins register through the
`socmint.collectors` entry-point group (value: "package.module:CollectorClass"), or via
the SOCMINT_PLUGINS env var ("pkg.mod:Class,pkg2.mod:Class"). Each must implement
`socmint.interfaces.Collector`.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable

from socmint.interfaces import Collector
from socmint.util.log import get_logger

log = get_logger(__name__)

# LinkedIn Ads was removed from the active source set (owner decision 2026-11-20): the Ad
# Library added noise (local promos, undated ads that never reach the timelines) and is not
# needed. The collector module (collect.linkedin_ads) stays on disk but is no longer
# registered, so it is never offered in the GUI/CLI or run by default.
BUILTIN: dict[str, str] = {
    "x": "socmint.collect.x:XCollector",
    "telegram": "socmint.collect.telegram:TelegramCollector",
    "youtube": "socmint.collect.youtube:YouTubeCollector",
    "linkedin": "socmint.collect.linkedin_posts:LinkedInPostsCollector",
}
DEFAULT_SOURCES: tuple[str, ...] = ("x", "telegram", "youtube", "linkedin")


def _load(spec: str) -> Callable[[], Collector]:
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls)


def plugin_specs() -> dict[str, str]:
    specs: dict[str, str] = {}
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="socmint.collectors"):
            specs[ep.name] = ep.value
    except Exception as e:  # noqa: BLE001
        log.warning("could not read collector entry points: %s", e)
    for item in filter(None, (os.environ.get("SOCMINT_PLUGINS") or "").split(",")):
        mod_cls = item.strip()
        name = mod_cls.rsplit(":", 1)[-1].lower().removesuffix("collector") or mod_cls
        specs[name] = mod_cls
    return specs


def available() -> dict[str, str]:
    return {**BUILTIN, **plugin_specs()}


def get_collector(name: str) -> Collector:
    specs = available()
    if name not in specs:
        raise KeyError(f"unknown source '{name}'. Known: {', '.join(sorted(specs))}")
    return _load(specs[name])()
