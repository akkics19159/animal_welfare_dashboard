from __future__ import annotations

from typing import Any, Iterable, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def kpi_card(value: Any, label: str) -> None:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-label'>{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_class(severity: str) -> str:
    if severity == "Critical":
        return "sev-critical"
    if severity == "Warning":
        return "sev-warning"
    return "sev-info"


def render_alerts(alerts: List[dict]) -> None:
    unack = [a for a in alerts if not a.get("acknowledged")]
    if not unack:
        st.success("🟢 No active alerts. All monitoring points are normal.")
        return

    for alert in unack:
        sev_class = severity_class(alert.get("severity", "Info"))
        st.markdown(
            f"""
            <div class='{sev_class}'>
                <strong>[{alert.get('severity').upper()}] {alert.get('source')}</strong>: {alert.get('message')}
                <br/><span style='font-size:0.8rem; opacity:0.8;'>Triggered: {alert.get('timestamp')}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def welfare_timeline_chart(history_df: pd.DataFrame) -> None:
    if history_df.empty:
        st.info("No historical trends generated yet. Run a Live Monitoring check.")
        return

    df = history_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["welfare_score"],
            mode="lines+markers",
            name="Welfare Score (%)",
            line=dict(color="#8b5cf6", width=3),
            fill="tozeroy",
        )
    )
    fig.update_layout(
        title="Real-Time Welfare Status Curve",
        xaxis_title="Time",
        yaxis_title="Score (%)",
        height=350,
        hovermode="x unified",
        template="plotly_dark",
    )
    st.plotly_chart(fig, width="stretch")

