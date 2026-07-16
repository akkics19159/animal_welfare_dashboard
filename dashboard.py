from __future__ import annotations

from pathlib import Path
from datetime import datetime

import streamlit as st

from ui.api_client import ApiClient
from ui.sidebar import render_sidebar
from ui.state import ensure_session_defaults
from ui.theme import apply_theme


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_HISTORY_PATH = BASE_DIR / "welfare_analysis_history.csv"
DEFAULT_VIDEO_PATH = BASE_DIR / "data" / "video" / "puppy crying.mp4"


def _get_page_module(selected_page: str) -> str:
    return {
        "Dashboard": "ui.pages.dashboard",
        "Live Monitoring": "ui.pages.live_monitoring",
        "Analytics": "ui.pages.analytics",
        "History": "ui.pages.history",
        "Evaluation": "ui.pages.evaluation",
        "Dataset": "ui.pages.dataset",
        "Training": "ui.pages.training",
        "Models": "ui.pages.models",
        "Alerts": "ui.pages.alerts",
        "System Health": "ui.pages.system_health",
        "Settings": "ui.pages.settings",
        "User Management": "ui.pages.user_management",
    }.get(selected_page, "ui.pages.placeholder")


def _render_footer() -> None:
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #64748b; font-size: 0.85em; font-weight: 500;'>
            🔍 Sentient Being Welfare Monitoring Platform v2026.1
            <br/>Modular UI Architecture | REST API Layer | Real-Time MLOps Engine
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_shell_header(backend_online: bool) -> None:
    status_class = "ok" if backend_online else "warn"
    status_label = "Backend Online" if backend_online else "Fallback Mode"
    now_label = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(
        f"""
        <div class='app-shell-header'>
            <div class='app-shell-row'>
                <div class='app-shell-title'>Sentient Welfare Monitoring Platform</div>
                <div class='app-shell-meta'>
                    <span class='status-chip {status_class}'>{status_label}</span>
                    <span>Updated {now_label}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_theme()
    ensure_session_defaults()

    api = ApiClient("http://127.0.0.1:8000")
    backend_online = api.online()
    if not backend_online:
        # Best-effort fallback probe to reduce false offline state.
        try:
            _ = api.health()
            backend_online = True
        except Exception:
            backend_online = False
    st.session_state.backend_online = backend_online

    _render_shell_header(backend_online)

    selected_page = render_sidebar()
    page_mod_name = _get_page_module(selected_page)

    import importlib

    mod = importlib.import_module(page_mod_name)
    mod.render_page(
        api=api,
        backend_online=backend_online,
        default_history_path=str(DEFAULT_HISTORY_PATH),
        default_video_path=DEFAULT_VIDEO_PATH,
    )

    _render_footer()


if __name__ == "__main__":
    main()

