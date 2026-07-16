from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import requests
import psutil
import streamlit as st

from ui.api_client import ApiClient
from ui.components import kpi_card, welfare_timeline_chart
from ui.state import DashboardConfig, get_config


BASE_DIR = __file__


def _load_local_history(default_history_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(default_history_path)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "welfare_score", "video_score", "audio_score", "sensor_score", "probability"])


def render_page(api: ApiClient, backend_online: bool, default_history_path: str, default_video_path: str) -> None:
    cfg = get_config()

    st.markdown("<h1 class='main-title'>🔍 Platform Executive Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Real-time summary of sentient welfare index, active alerts, and resource health status.")
    st.divider()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    welfare_avg = 100.0
    active_alerts_count = 0
    cpu_usage = psutil.cpu_percent()
    active_model = "welfare-fusion-v1"

    if backend_online:
        try:
            health = api.health()
            cpu_usage = health.get("cpu", cpu_usage)

            alerts = api.get("/api/alerts")
            active_alerts_count = len([a for a in alerts if not a.get("acknowledged")])

            models = api.get("/api/models")
            active_model = next((m["model_version"] for m in models if m.get("active")), active_model)

            analytics = api.get("/api/analytics")
            welfare_avg = (1.0 - analytics.get("aggregates", {}).get("mean_probability", 0.0)) * 100.0
        except Exception:
            pass
    else:
        history_df = _load_local_history(default_history_path)
        if len(history_df) > 0:
            welfare_avg = float(history_df["welfare_score"].mean())

    with kpi1:
        kpi_card(f"{welfare_avg:.1f}%", "Welfare Score Average")
    with kpi2:
        kpi_card(str(active_alerts_count), "Unresolved Alerts")
    with kpi3:
        kpi_card(f"{cpu_usage}%", "Server CPU Usage")
    with kpi4:
        kpi_card(active_model, "Active Model Registry")

    st.divider()
    st.subheader("⚠️ Critical Alert Panel")

    if backend_online:
        try:
            alerts = api.get("/api/alerts")
            from ui.components import render_alerts

            render_alerts(alerts)
        except Exception:
            st.error("Failed to load alerts from REST server.")
    else:
        st.info("Direct alerts only available via API backend connection.")

    st.divider()
    st.subheader("📈 Real-time Welfare Trends")

    history_df = _load_local_history(default_history_path)
    welfare_timeline_chart(history_df)

