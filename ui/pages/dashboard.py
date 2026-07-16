from __future__ import annotations

import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import requests
import psutil
import streamlit as st

from ui.api_client import ApiClient
from ui.components import kpi_card, welfare_timeline_chart
from ui.state import DashboardConfig, get_config, get_global_date_range, filter_records_by_date_range


BASE_DIR = __file__


def _load_local_history(default_history_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(default_history_path)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "welfare_score", "video_score", "audio_score", "sensor_score", "probability"])


def _load_history_consistent(api: ApiClient, backend_online: bool, default_history_path: str) -> pd.DataFrame:
    """Canonical run-history loader used for dashboard consistency.

    Source of truth order:
    1) Backend /api/history when reachable.
    2) Local CSV fallback.
    """
    if backend_online:
        try:
            rows = api.get("/api/history")
            if isinstance(rows, list):
                df = pd.DataFrame(rows)
                if not df.empty:
                    return df
        except Exception:
            pass
    return _load_local_history(default_history_path)


def _filter_history_by_global_range(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    if df.empty:
        return df
    dt_col = None
    for candidate in ["timestamp", "time", "datetime", "created_at"]:
        if candidate in df.columns:
            dt_col = candidate
            break
    if dt_col is None:
        return df
    dfi = df.copy()
    dfi[dt_col] = pd.to_datetime(dfi[dt_col], errors="coerce")
    dfi = dfi.dropna(subset=[dt_col])
    if dfi.empty:
        return dfi
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return dfi[(dfi[dt_col] >= start_dt) & (dfi[dt_col] <= end_dt)]


def render_page(api: ApiClient, backend_online: bool, default_history_path: str, default_video_path: str) -> None:
    cfg = get_config()

    st.markdown("<h1 class='main-title'>🔍 Platform Executive Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Real-time summary of sentient welfare index, active alerts, and resource health status.")
    st.divider()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    global_start, global_end = get_global_date_range()

    history_df = _load_history_consistent(api, backend_online, default_history_path)
    history_df = _filter_history_by_global_range(history_df, global_start, global_end)

    welfare_avg = 100.0
    active_alerts_count = 0
    cpu_usage = psutil.cpu_percent()
    active_model = "welfare-fusion-v1"

    if backend_online:
        try:
            health = api.health()
            cpu_usage = health.get("cpu", cpu_usage)

            alerts = filter_records_by_date_range(
                api.get("/api/alerts"),
                global_start,
                global_end,
                date_fields=("timestamp", "created_at"),
                include_if_missing=False,
            )
            active_alerts_count = len([a for a in alerts if not a.get("acknowledged")])

            models = api.get("/api/models")
            active_model = next((m["model_version"] for m in models if m.get("active")), active_model)

            if "welfare_score" in history_df.columns and len(history_df) > 0:
                welfare_avg = float(pd.to_numeric(history_df["welfare_score"], errors="coerce").dropna().mean())
            elif "probability" in history_df.columns and len(history_df) > 0:
                mean_prob = float(pd.to_numeric(history_df["probability"], errors="coerce").dropna().mean())
                welfare_avg = (1.0 - mean_prob) * 100.0
        except Exception:
            pass
    else:
        if len(history_df) > 0:
            if "welfare_score" in history_df.columns:
                welfare_avg = float(pd.to_numeric(history_df["welfare_score"], errors="coerce").dropna().mean())
            elif "probability" in history_df.columns:
                mean_prob = float(pd.to_numeric(history_df["probability"], errors="coerce").dropna().mean())
                welfare_avg = (1.0 - mean_prob) * 100.0

    with kpi1:
        kpi_card(f"{welfare_avg:.1f}%", "Welfare Score Average")
    with kpi2:
        kpi_card(str(active_alerts_count), "Unresolved Alerts")
    with kpi3:
        kpi_card(f"{cpu_usage}%", "Server CPU Usage")
    with kpi4:
        kpi_card(active_model, "Active Model Registry")
    st.caption(f"Global date range: {global_start} → {global_end}")

    st.divider()
    st.subheader("⚠️ Critical Alert Panel")

    if backend_online:
        try:
            alerts = filter_records_by_date_range(
                api.get("/api/alerts"),
                global_start,
                global_end,
                date_fields=("timestamp", "created_at"),
                include_if_missing=False,
            )
            from ui.components import render_alerts

            render_alerts(alerts)
        except Exception:
            st.error("Failed to load alerts from REST server.")
    else:
        st.info("Direct alerts only available via API backend connection.")

    st.divider()
    st.subheader("📈 Real-time Welfare Trends")

    welfare_timeline_chart(history_df)
