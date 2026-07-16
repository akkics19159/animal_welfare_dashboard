from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_datetime_col(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    for c in ["timestamp", "time", "datetime", "created_at"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if any(k in c.lower() for k in ["timestamp", "time", "date"]):
            return c
    return None


def render_page(api, backend_online: bool, default_history_path: str, default_video_path: str) -> None:
    st.markdown("<h1 class='main-title'>📚 History</h1>", unsafe_allow_html=True)
    st.markdown("Historical welfare/inference records and trend visualization. Backend source: `/api/history`.")

    if not backend_online:
        st.caption("Backend health check was inconclusive. Trying `/api/history` directly...")

    try:
        raw = api.get("/api/history")
    except Exception as e:
        st.warning("API backend offline: History page requires `/api/history`.")
        st.error(f"Failed to load history: {e}")
        return

    rows: List[Dict[str, Any]] = raw if isinstance(raw, list) else []
    if not rows:
        st.info("No history records available.")
        return

    df = pd.DataFrame(rows)
    dt_col = _safe_datetime_col(df)
    if dt_col is None:
        st.error("History dataset has no recognizable timestamp column.")
        return

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])
    if df.empty:
        st.info("No valid history records after parsing timestamps.")
        return

    # Optional behaviour/risk filters when columns exist.
    if "behaviour" in df.columns:
        opts = sorted(df["behaviour"].dropna().astype(str).unique().tolist())
        if opts:
            selected_b = st.sidebar.multiselect("Behaviour", options=opts, default=opts)
            df = df[df["behaviour"].astype(str).isin(selected_b)]
    if "risk_level" in df.columns:
        ropts = sorted(df["risk_level"].dropna().astype(str).unique().tolist())
        if ropts:
            selected_r = st.sidebar.multiselect("Risk level", options=ropts, default=ropts)
            df = df[df["risk_level"].astype(str).isin(selected_r)]

    # Filters
    st.sidebar.header("History filters")
    min_date = df[dt_col].dt.date.min()
    max_date = df[dt_col].dt.date.max()
    start_date = st.sidebar.date_input("Start date", value=min_date)
    end_date = st.sidebar.date_input("End date", value=max_date)
    if start_date > end_date:
        st.sidebar.error("Start date must be <= end date")
        return

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    mask = (df[dt_col] >= start_dt) & (df[dt_col] <= end_dt)
    df_f = df.loc[mask].copy()

    if df_f.empty:
        st.info("No history data for the selected date range.")
        return

    # KPIs
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    prob_col = "probability" if "probability" in df_f.columns else None
    welfare_col = "welfare_score" if "welfare_score" in df_f.columns else None

    avg_prob = _safe_float(df_f[prob_col].mean(), None) if prob_col else None
    max_prob = _safe_float(df_f[prob_col].max(), None) if prob_col else None
    critical_runs = int((df_f[prob_col] >= 0.7).sum()) if prob_col else 0
    avg_welfare = _safe_float(df_f[welfare_col].mean(), None) if welfare_col else None

    with k1:
        st.metric("Runs", str(len(df_f)))
    with k2:
        st.metric("Avg probability", f"{avg_prob:.3f}" if avg_prob is not None else "N/A")
    with k3:
        st.metric("Max probability", f"{max_prob:.3f}" if max_prob is not None else "N/A")
    with k4:
        st.metric("Critical alerts", str(critical_runs))

    if avg_welfare is not None:
        st.caption(f"Avg welfare score (from history): {avg_welfare:.1f}%")

    # Trend chart: welfare_score if available else probability
    st.divider()
    st.subheader("Trend")

    fig = go.Figure()
    if welfare_col and welfare_col in df_f.columns:
        fig.add_trace(
            go.Scatter(
                x=df_f[dt_col],
                y=pd.to_numeric(df_f[welfare_col], errors="coerce"),
                mode="lines+markers",
                name="Welfare score (%)",
                line=dict(width=3, color="#8b5cf6"),
            )
        )
        fig.update_yaxes(title_text="Welfare score (%)")
    elif prob_col:
        fig.add_trace(
            go.Scatter(
                x=df_f[dt_col],
                y=pd.to_numeric(df_f[prob_col], errors="coerce"),
                mode="lines+markers",
                name="Risk probability",
                line=dict(width=3, color="#fb7185"),
            )
        )
        fig.update_yaxes(title_text="Probability")
    else:
        st.info("No welfare_score or probability columns found to render trend.")
        return

    fig.update_layout(
        height=380,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    # Table view
    st.divider()
    st.subheader("Records")
    show_cols = []
    for c in [dt_col, "welfare_score", "probability", "latency_ms", "video_score", "audio_score", "sensor_score"]:
        if c in df_f.columns:
            show_cols.append(c)

    for c in [
        "behaviour",
        "behaviour_probability",
        "behaviour_confidence",
        "behaviour_duration",
        "behaviour_transition",
        "behaviour_stability",
        "occupancy",
        "species",
        "risk_level",
    ]:
        if c in df_f.columns:
            show_cols.append(c)

    st.dataframe(df_f[show_cols].sort_values(by=dt_col, ascending=False).head(200), width="stretch", hide_index=True)


