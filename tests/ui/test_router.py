from __future__ import annotations

import importlib


def test_router_imports_pages():
    # Smoke test: ensure page modules are importable.
    pages = [
        "ui.pages.dashboard",
        "ui.pages.live_monitoring",
        "ui.pages.analytics",
        "ui.pages.history",
        "ui.pages.evaluation",
        "ui.pages.dataset",
        "ui.pages.training",
        "ui.pages.models",
        "ui.pages.alerts",
        "ui.pages.system_health",
        "ui.pages.settings",
        "ui.pages.user_management",
        "ui.pages.placeholder",
    ]
    for p in pages:
        mod = importlib.import_module(p)
        assert hasattr(mod, "render_page")

