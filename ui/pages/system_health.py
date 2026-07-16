from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from ui.api_client import ApiClient
from ui.components_live.system_health_widgets import render_device_status, render_system_health_kpis
from ui.state import get_global_date_range, filter_records_by_date_range


def render_page(api: ApiClient, backend_online: bool, default_history_path: str, default_video_path: str) -> None:
    st.markdown("<h1 class='main-title'>🔧 System Health</h1>", unsafe_allow_html=True)
    st.markdown(
        "Live resource and device health widgets sourced from `/api/health`. "
        "FPS may be unavailable depending on backend payload."
    )

    if not backend_online:
        st.caption("Backend health check was inconclusive. Trying `/api/health` directly...")

    try:
        health = api.health()
    except Exception as e:
        st.warning("API backend offline: system health is not available.")
        st.error(f"Failed to load system health: {e}")
        return

    if not isinstance(health, dict):
        st.info("No health payload available.")
        return
    global_start, global_end = get_global_date_range()
    health_rows = filter_records_by_date_range(
        [health],
        global_start,
        global_end,
        date_fields=("timestamp",),
        include_if_missing=False,
    )
    if not health_rows:
        st.info("No system health snapshot in the selected global date range.")
        return
    st.caption(f"Global date range: {global_start} → {global_end}")
    health = health_rows[0]

    # CPU/RAM/GPU/Disk/Temperature/Inference throughput/FPS
    render_system_health_kpis(health)

    st.divider()

    # Camera/Microphone/Sensor status
    render_device_status(health)
