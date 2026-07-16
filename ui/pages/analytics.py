from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _to_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)


def _load_local_history(default_history_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(default_history_path)
    except Exception:
        df = pd.DataFrame(columns=["timestamp", "welfare_score", "video_score", "audio_score", "sensor_score", "probability"])

    if "timestamp" in df.columns:
        df["timestamp"] = _to_datetime_series(df["timestamp"])
    return df


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _guess_datetime_col(df: pd.DataFrame) -> Optional[str]:
    if "timestamp" in df.columns:
        return "timestamp"
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower():
            return c
    return None


def _filter_by_date_range(df: pd.DataFrame, datetime_col: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    if df.empty:
        return df
    dfi = df.copy()
    dfi[datetime_col] = _to_datetime_series(dfi[datetime_col])
    dfi = dfi.dropna(subset=[datetime_col])
    start_dt = dt.datetime.combine(start, dt.time.min)
    end_dt = dt.datetime.combine(end, dt.time.max)
    return dfi[(dfi[datetime_col] >= start_dt) & (dfi[datetime_col] <= end_dt)]


def _render_plot(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title=title,
        height=350,
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, width="stretch")


def _pick_series(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_sorted(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    if datetime_col in df.columns:
        return df.sort_values(by=datetime_col)
    return df


def _render_daily_trends(df: pd.DataFrame, datetime_col: str) -> None:
    welfare_col = _pick_series(df, ["welfare_score", "welfare", "welfare_percent"])
    prob_col = _pick_series(df, ["probability", "welfare_probability", "mean_probability"])

    if welfare_col is None and prob_col is None:
        st.info("Daily trends: no welfare/probability columns found in the selected dataset.")
        return

    d = df.copy()
    d["_day"] = d[datetime_col].dt.date

    fig = go.Figure()
    if welfare_col is not None:
        day_w = d.groupby("_day")[welfare_col].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=day_w["_day"],
                y=day_w[welfare_col],
                mode="lines+markers",
                name="Welfare (avg)",
                line=dict(width=3, color="#8b5cf6"),
            )
        )
    if prob_col is not None:
        day_p = d.groupby("_day")[prob_col].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=day_p["_day"],
                y=day_p[prob_col],
                mode="lines+markers",
                name="Probability (avg)",
                line=dict(width=3, color="#22c55e"),
            )
        )

    _render_plot(fig, "Daily trends")


def _render_weekly_trends(df: pd.DataFrame, datetime_col: str) -> None:
    welfare_col = _pick_series(df, ["welfare_score", "welfare", "welfare_percent"])
    prob_col = _pick_series(df, ["probability", "welfare_probability", "mean_probability"])

    if welfare_col is None and prob_col is None:
        st.info("Weekly trends: no welfare/probability columns found in the selected dataset.")
        return

    d = df.copy()
    d["_week"] = d[datetime_col].dt.to_period("W").dt.start_time

    fig = go.Figure()
    if welfare_col is not None:
        w = d.groupby("_week")[welfare_col].mean().reset_index()
        fig.add_trace(go.Bar(x=w["_week"], y=w[welfare_col], name="Welfare (avg)", marker_color="#8b5cf6"))

    if prob_col is not None:
        p = d.groupby("_week")[prob_col].mean().reset_index()
        fig.add_trace(go.Bar(x=p["_week"], y=p[prob_col], name="Probability (avg)", marker_color="#22c55e"))

    fig.update_layout(barmode="group")
    _render_plot(fig, "Weekly trends")


def _render_monthly_trends(df: pd.DataFrame, datetime_col: str) -> None:
    welfare_col = _pick_series(df, ["welfare_score", "welfare", "welfare_percent"])
    prob_col = _pick_series(df, ["probability", "welfare_probability", "mean_probability"])

    if welfare_col is None and prob_col is None:
        st.info("Monthly trends: no welfare/probability columns found in the selected dataset.")
        return

    d = df.copy()
    d["_month"] = d[datetime_col].dt.to_period("M").dt.start_time

    fig = go.Figure()
    if welfare_col is not None:
        m = d.groupby("_month")[welfare_col].mean().reset_index()
        fig.add_trace(go.Scatter(x=m["_month"], y=m[welfare_col], mode="lines+markers", name="Welfare (avg)", line=dict(width=3, color="#8b5cf6")))

    if prob_col is not None:
        p = d.groupby("_month")[prob_col].mean().reset_index()
        fig.add_trace(go.Scatter(x=p["_month"], y=p[prob_col], mode="lines+markers", name="Probability (avg)", line=dict(width=3, color="#22c55e")))

    _render_plot(fig, "Monthly trends")


def _render_species_distribution(df: pd.DataFrame, datetime_col: str) -> None:
    species_col = _pick_series(df, ["species", "species_name", "animal_species", "class_name"])
    if species_col is None:
        st.info("Species distribution: no species column found in the selected dataset.")
        return

    d = df.copy()
    dist = d[species_col].value_counts().reset_index()
    dist.columns = ["species", "count"]

    fig = px.bar(dist, x="species", y="count", color="species", title="")
    fig.update_layout(height=350, template="plotly_dark", margin=dict(l=10, r=10, t=50, b=10))
    fig.update_traces(showlegend=False)
    fig.update_layout(title="Species distribution")
    st.plotly_chart(fig, width="stretch")


def _render_occupancy(df: pd.DataFrame, datetime_col: str) -> None:
    occ_col = _pick_series(df, ["occupancy", "occupancy_rate", "animal_count", "count"])
    if occ_col is None:
        st.info("Occupancy: no occupancy/count column found in the selected dataset.")
        return

    d = df.copy()
    d = d.dropna(subset=[occ_col])
    if d.empty:
        st.info("Occupancy: occupancy column exists but has no valid values in the selected date range.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d[datetime_col],
            y=_safe_numeric(d, occ_col),
            mode="lines",
            name="Occupancy",
            line=dict(color="#38bdf8", width=3),
        )
    )
    _render_plot(fig, "Occupancy")


def _render_behaviour_frequency(df: pd.DataFrame, datetime_col: str) -> None:
    behaviour_col = _pick_series(df, ["behaviour", "behavior", "behavior_label", "behaviour_label"])
    if behaviour_col is None:
        st.info("Behaviour frequency: no behaviour column found in the selected dataset.")
        return

    d = df.copy()
    d["_bucket_week"] = d[datetime_col].dt.to_period("W").dt.start_time
    if d["_bucket_week"].isna().all():
        st.info("Behaviour frequency: unable to bucket data into weekly intervals.")
        return

    counts = d.groupby(["_bucket_week", behaviour_col]).size().reset_index(name="count")
    if counts.empty:
        st.info("Behaviour frequency: no behaviour events found for the selected date range.")
        return

    fig = px.bar(
        counts,
        x="_bucket_week",
        y="count",
        color=behaviour_col,
        barmode="stack",
        title="",
    )
    fig.update_layout(height=380, template="plotly_dark", margin=dict(l=10, r=10, t=50, b=10))
    fig.update_layout(title="Behaviour frequency (weekly buckets)")
    st.plotly_chart(fig, width="stretch")


def _render_distress_timeline(df: pd.DataFrame, datetime_col: str) -> None:
    distress_col = _pick_series(df, ["distress", "is_distressed", "distress_detected"])
    prob_col = _pick_series(df, ["distress_probability", "distress_prob", "distress_score", "probability"])

    if distress_col is None and prob_col is None:
        st.info("Distress timeline: no distress columns found in the selected dataset.")
        return

    d = df.copy()
    d["_day"] = d[datetime_col].dt.date

    fig = go.Figure()

    if distress_col is not None:
        # Convert to boolean-ish
        bool_s = d[distress_col]
        if bool_s.dtype != bool:
            bool_s = bool_s.astype(str).str.lower().isin(["true", "1", "yes", "y"])
        tmp = d.copy()
        tmp["_distress_bool"] = bool_s
        daily_counts = tmp.groupby("_day")["_distress_bool"].mean().reset_index(name="distress_rate")
        fig.add_trace(
            go.Scatter(
                x=daily_counts["_day"],
                y=daily_counts["distress_rate"],
                mode="lines+markers",
                name="Distress rate",
                line=dict(color="#fb7185", width=3),
            )
        )

    if prob_col is not None and prob_col in d.columns:
        daily_p = d.groupby("_day")[prob_col].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=daily_p["_day"],
                y=_safe_numeric(daily_p, prob_col),
                mode="lines+markers",
                name="Distress probability (avg)",
                line=dict(color="#f97316", width=3),
            )
        )

    _render_plot(fig, "Distress timeline")


def _render_sensor_history(df: pd.DataFrame, datetime_col: str) -> None:
    sensor_cols = [c for c in df.columns if any(k in c.lower() for k in ["sensor", "temp", "temperature", "humidity", "motion", "accel", "gyroscope", "pressure"])]
    if not sensor_cols:
        st.info("Sensor history: no sensor-like columns found in the selected dataset.")
        return

    # Pick up to 5 for readability
    sensor_cols = sorted(sensor_cols)
    sensor_cols = sensor_cols[:5]

    fig = go.Figure()
    for c in sensor_cols:
        fig.add_trace(go.Scatter(x=df[datetime_col], y=_safe_numeric(df, c), mode="lines", name=c))

    _render_plot(fig, "Sensor history")


def _render_inference_statistics(df: pd.DataFrame, datetime_col: str) -> None:
    # Try common names
    latency_col = _pick_series(df, ["latency_ms", "inference_latency_ms", "latency", "inference_latency"])
    fps_col = _pick_series(df, ["fps", "frames_per_second"])

    if latency_col is None and fps_col is None:
        st.info("Inference statistics: no inference timing/FPS columns found in the selected dataset.")
        return

    d = df.copy()
    d["_day"] = d[datetime_col].dt.date

    fig = go.Figure()

    if latency_col is not None:
        daily = d.groupby("_day")[latency_col].mean().reset_index()
        fig.add_trace(go.Bar(x=daily["_day"], y=_safe_numeric(daily, latency_col), name="Avg latency (ms)", marker_color="#a78bfa"))

    if fps_col is not None:
        daily2 = d.groupby("_day")[fps_col].mean().reset_index()
        fig.add_trace(go.Scatter(x=daily2["_day"], y=_safe_numeric(daily2, fps_col), mode="lines+markers", name="Avg FPS", line=dict(width=3, color="#34d399")))

    fig.update_layout(barmode="group")
    _render_plot(fig, "Inference statistics")


def _render_model_confidence_trends(df: pd.DataFrame, datetime_col: str) -> None:
    conf_col = _pick_series(df, ["model_confidence", "confidence", "probability", "mean_probability"])
    if conf_col is None:
        st.info("Model confidence trends: no confidence/probability columns found.")
        return

    d = df.copy()
    d["_day"] = d[datetime_col].dt.date
    daily = d.groupby("_day")[conf_col].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["_day"], y=_safe_numeric(daily, conf_col), mode="lines+markers", name="Confidence (avg)", line=dict(width=3, color="#60a5fa")))
    _render_plot(fig, "Model confidence trends")


def _render_behaviour_transition_trends(df: pd.DataFrame, datetime_col: str) -> None:
    trans_col = _pick_series(df, ["behaviour_transition", "behavior_transition"])
    if trans_col is None:
        st.info("Behaviour transitions: no transition column found.")
        return

    d = df.copy()
    d[trans_col] = d[trans_col].fillna("steady").astype(str)
    d["_day"] = d[datetime_col].dt.date
    counts = d.groupby(["_day", trans_col]).size().reset_index(name="count")
    if counts.empty:
        st.info("Behaviour transitions: no data in selected range.")
        return

    fig = px.bar(counts, x="_day", y="count", color=trans_col, barmode="stack", title="")
    fig.update_layout(height=380, template="plotly_dark", margin=dict(l=10, r=10, t=50, b=10))
    fig.update_layout(title="Behaviour transitions (daily)")
    st.plotly_chart(fig, width="stretch")


def _render_behaviour_stability(df: pd.DataFrame, datetime_col: str) -> None:
    stability_col = _pick_series(df, ["behaviour_stability", "behavior_stability"])
    if stability_col is None:
        st.info("Behaviour stability: no stability column found.")
        return

    d = df.copy()
    d["_day"] = d[datetime_col].dt.date
    daily = d.groupby("_day")[stability_col].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["_day"],
            y=_safe_numeric(daily, stability_col),
            mode="lines+markers",
            name="Avg stability",
            line=dict(width=3, color="#f59e0b"),
        )
    )
    _render_plot(fig, "Behaviour stability")


def _render_risk_timeline(df: pd.DataFrame, datetime_col: str) -> None:
    prob_col = _pick_series(df, ["probability", "welfare_probability", "mean_probability"])
    risk_col = _pick_series(df, ["risk_level", "risk", "severity"])

    if prob_col is None and risk_col is None:
        st.info("Risk timeline: no probability/risk columns found.")
        return

    d = df.copy()
    d["_day"] = d[datetime_col].dt.date
    fig = go.Figure()

    if prob_col is not None:
        p = d.groupby("_day")[prob_col].mean().reset_index()
        fig.add_trace(go.Scatter(x=p["_day"], y=_safe_numeric(p, prob_col), mode="lines+markers", name="Avg risk probability", line=dict(width=3, color="#ef4444")))

    if risk_col is not None:
        mapped = d[risk_col].astype(str).str.lower().map({"low": 1, "normal": 1, "moderate": 2, "warning": 2, "high": 3, "critical": 3}).fillna(1)
        tmp = d.copy()
        tmp["_risk_numeric"] = mapped
        r = tmp.groupby("_day")["_risk_numeric"].mean().reset_index()
        fig.add_trace(go.Scatter(x=r["_day"], y=r["_risk_numeric"], mode="lines+markers", name="Risk level index", line=dict(width=2, color="#f97316"), yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Risk index"))

    _render_plot(fig, "Risk timeline")


def _render_trajectory_heatmap(df: pd.DataFrame) -> None:
    occ_col = _pick_series(df, ["occupancy", "occupancy_rate", "animal_count", "count"])
    prob_col = _pick_series(df, ["probability", "welfare_probability", "mean_probability"])
    if occ_col is None or prob_col is None:
        st.info("Trajectory heatmap proxy: requires occupancy and probability columns.")
        return

    d = df.copy()
    x = pd.to_numeric(d[occ_col], errors="coerce")
    y = pd.to_numeric(d[prob_col], errors="coerce")
    valid = ~(x.isna() | y.isna())
    if valid.sum() == 0:
        st.info("Trajectory heatmap proxy: no valid rows.")
        return

    fig = px.density_heatmap(
        x=x[valid],
        y=y[valid],
        nbinsx=16,
        nbinsy=12,
        labels={"x": occ_col, "y": prob_col},
        title="",
        color_continuous_scale="YlOrRd",
    )
    fig.update_layout(height=380, template="plotly_dark", margin=dict(l=10, r=10, t=50, b=10), title="Trajectory/occupancy heatmap (proxy)")
    st.plotly_chart(fig, width="stretch")


def _render_alert_trends_from_df(alerts_df: pd.DataFrame, datetime_col: str) -> None:
    if alerts_df.empty:
        st.info("Alert trends: no alert data available.")
        return

    severity_col = _pick_series(alerts_df, ["severity", "level"])
    if severity_col is None:
        st.info("Alert trends: alert severity column not found.")
        return

    alerts_df = alerts_df.copy()
    alerts_df[datetime_col] = _to_datetime_series(alerts_df[datetime_col])
    alerts_df = alerts_df.dropna(subset=[datetime_col])
    if alerts_df.empty:
        st.info("Alert trends: unable to parse alert timestamps.")
        return

    alerts_df["_day"] = alerts_df[datetime_col].dt.date
    counts = alerts_df.groupby(["_day", severity_col]).size().reset_index(name="count")

    if counts.empty:
        st.info("Alert trends: no alert events in the selected date range.")
        return

    fig = px.bar(counts, x="_day", y="count", color=severity_col, barmode="stack")
    fig.update_layout(height=380, template="plotly_dark", margin=dict(l=10, r=10, t=50, b=10))
    fig.update_layout(title="Alert trends (daily) ")
    st.plotly_chart(fig, width="stretch")


def _try_backend_load(api: Any, backend_online: bool, start: dt.date, end: dt.date) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Best-effort: never fails hard. Returns (history_df, alerts_df)."""
    if not backend_online:
        return pd.DataFrame(), None

    history_df = pd.DataFrame()
    alerts_df: Optional[pd.DataFrame] = None

    # History / analytics endpoints might differ across deployments.
    # We'll try a small set of plausible URLs.
    candidates = ["/api/analytics", "/api/history", "/api/trends", "/api/observations"]

    for path in candidates:
        try:
            resp = api.get(path)
            # Many backends return {"rows": [...]} or direct list
            if isinstance(resp, dict):
                rows = resp.get("rows") or resp.get("data") or resp.get("items") or resp.get("history")
                if rows is not None:
                    history_df = pd.DataFrame(rows)
                else:
                    # Sometimes analytics aggregates only
                    history_df = pd.DataFrame(resp)
            elif isinstance(resp, list):
                history_df = pd.DataFrame(resp)

            if not history_df.empty:
                break
        except Exception:
            continue

    # Alerts endpoint
    try:
        alerts_resp = api.get("/api/alerts")
        if isinstance(alerts_resp, list):
            alerts_df = pd.DataFrame(alerts_resp)
        elif isinstance(alerts_resp, dict):
            rows = alerts_resp.get("rows") or alerts_resp.get("data")
            if rows is not None:
                alerts_df = pd.DataFrame(rows)
    except Exception:
        alerts_df = None

    # Filter by date if possible
    dt_col_hist = _guess_datetime_col(history_df) if not history_df.empty else None
    if history_df is not None and not history_df.empty and dt_col_hist is not None:
        history_df = _filter_by_date_range(history_df, dt_col_hist, start, end)

    if alerts_df is not None and not alerts_df.empty:
        dt_col_alert = _guess_datetime_col(alerts_df) or ("timestamp" if "timestamp" in alerts_df.columns else None)
        if dt_col_alert is not None:
            alerts_df = _filter_by_date_range(alerts_df, dt_col_alert, start, end)

    return history_df, alerts_df


def render_page(api, backend_online: bool, default_history_path: str, default_video_path: str) -> None:
    st.markdown("<h1 class='main-title'>📈 Analytics</h1>", unsafe_allow_html=True)
    st.markdown("Interactive analytics dashboard with filtering and date ranges. Charts are best-effort from backend analytics/history; otherwise fall back to the local history CSV.")

    # ---- Filters ----
    today = dt.date.today()
    default_start = today - dt.timedelta(days=30)

    with st.sidebar:
        st.header("Filters")
        start_date = st.date_input("Start date", value=default_start)
        end_date = st.date_input("End date", value=today)

        if start_date > end_date:
            st.error("Start date must be <= end date")

        species_col_candidates = ["species", "species_name", "animal_species", "class_name"]
        occupancy_col_candidates = ["occupancy", "occupancy_rate", "animal_count", "count"]
        behavior_col_candidates = ["behaviour", "behavior", "behavior_label", "behaviour_label"]

        species_filter: List[str] = []
        behavior_filter: List[str] = []
        occupancy_filter: List[str] = []

        # These dropdowns are populated after data load; keep placeholders for stable layout.
        st.caption("Filters for species/behaviour/occupancy are populated once data is loaded.")

        show_daily = st.checkbox("Daily trends", True, key="analytics_show_daily")
        show_weekly = st.checkbox("Weekly trends", True, key="analytics_show_weekly")
        show_monthly = st.checkbox("Monthly trends", True, key="analytics_show_monthly")
        show_species = st.checkbox("Species distribution", True, key="analytics_show_species")
        show_occupancy = st.checkbox("Occupancy", True, key="analytics_show_occupancy")
        show_behaviour = st.checkbox("Behaviour frequency", True, key="analytics_show_behaviour")
        show_distress = st.checkbox("Distress timeline", True, key="analytics_show_distress")
        show_sensor_history = st.checkbox("Sensor history", True, key="analytics_show_sensor_history")
        show_inference_stats = st.checkbox("Inference statistics", True, key="analytics_show_inference_stats")
        show_confidence = st.checkbox("Model confidence trends", True, key="analytics_show_confidence")
        show_behaviour_transitions = st.checkbox("Behaviour transitions", True, key="analytics_show_behaviour_transitions")
        show_behaviour_stability = st.checkbox("Behaviour stability", True, key="analytics_show_behaviour_stability")
        show_risk_timeline = st.checkbox("Risk timeline", True, key="analytics_show_risk_timeline")
        show_trajectory_heatmap = st.checkbox("Trajectory heatmap", True, key="analytics_show_trajectory_heatmap")
        show_alerts = st.checkbox("Alert trends", True, key="analytics_show_alerts")

    # Persist toggles for chart rendering in the main body
    show_all_requested = st.sidebar.checkbox("Show all requested analytics sections", True, key="analytics_show_all_requested")


    if start_date > end_date:
        return

    # ---- Load data (best-effort backend) ----
    history_df, alerts_df = _try_backend_load(api, backend_online, start_date, end_date)

    if history_df.empty:
        history_df = _load_local_history(default_history_path)
        dt_col = _guess_datetime_col(history_df) or "timestamp"
        history_df = _filter_by_date_range(history_df, dt_col, start_date, end_date)

    history_df = _ensure_sorted(history_df, _guess_datetime_col(history_df) or "timestamp")
    datetime_col = _guess_datetime_col(history_df) or ("timestamp" if "timestamp" in history_df.columns else None)

    if datetime_col is None:
        st.warning("Analytics: could not determine a timestamp column in the history dataset.")
        return

    # ---- Additional filters based on available columns ----
    # Species
    species_col = _pick_series(history_df, ["species", "species_name", "animal_species", "class_name"])
    if species_col is not None:
        species_options = sorted(history_df[species_col].dropna().astype(str).unique().tolist())
        if species_options:
            species_filter = st.sidebar.multiselect("Species", options=species_options, default=species_options)
            history_df = history_df[history_df[species_col].astype(str).isin(species_filter)]

    # Behaviour
    behaviour_col = _pick_series(history_df, ["behaviour", "behavior", "behavior_label", "behaviour_label"])
    if behaviour_col is not None:
        behavior_options = sorted(history_df[behaviour_col].dropna().astype(str).unique().tolist())
        if behavior_options:
            behavior_filter = st.sidebar.multiselect("Behaviour", options=behavior_options, default=behavior_options)
            history_df = history_df[history_df[behaviour_col].astype(str).isin(behavior_filter)]

    # Occupancy (bucket/values): only if occupancy is categorical-like. Otherwise skip.
    occ_col = _pick_series(history_df, ["occupancy", "occupancy_rate", "animal_count", "count"])
    if occ_col is not None:
        # Try categorical unique values; if too many, skip.
        vals = history_df[occ_col].dropna().unique().tolist()
        try:
            vals_sorted = sorted([str(v) for v in vals])
        except Exception:
            vals_sorted = [str(v) for v in vals]

        if 0 < len(vals_sorted) <= 20:
            occupancy_filter = st.sidebar.multiselect("Occupancy values", options=vals_sorted, default=vals_sorted)
            history_df = history_df[history_df[occ_col].astype(str).isin(occupancy_filter)]

    if history_df.empty:
        st.info("No rows available after applying filters for the selected date range.")
        return

    # ---- Sections ----
    st.divider()

    left, right = st.columns([1.05, 1])
    with left:
        # quick KPI-like summary (best-effort)
        welfare_col = _pick_series(history_df, ["welfare_score", "welfare", "welfare_percent"])
        prob_col = _pick_series(history_df, ["probability", "welfare_probability", "mean_probability"])

        kpi_a, kpi_b, kpi_c, kpi_d = st.columns(4)
        welfare_avg = float(_safe_numeric(history_df, welfare_col).mean()) if welfare_col is not None else None
        prob_avg = float(_safe_numeric(history_df, prob_col).mean()) if prob_col is not None else None
        n_rows = int(len(history_df))

        with kpi_a:
            if welfare_avg is not None:
                st.metric("Avg welfare", f"{welfare_avg:.1f}%")
            else:
                st.metric("Avg welfare", "N/A")
        with kpi_b:
            if prob_avg is not None:
                st.metric("Avg probability", f"{prob_avg:.3f}")
            else:
                st.metric("Avg probability", "N/A")
        with kpi_c:
            st.metric("Rows", str(n_rows))
        with kpi_d:
            st.metric("Range", f"{start_date} → {end_date}")

    with right:
        st.caption("Data source: " + ("Backend" if backend_online else "Local CSV fallback"))

    st.divider()

    if show_all_requested:
        # keep the toggles but allow a single gate
        pass

    # Daily trends
    if st.session_state.get("show_daily", True):
        if show_daily:
            _render_daily_trends(history_df, datetime_col)

    # Weekly trends
    if st.session_state.get("show_weekly", True):
        if show_weekly:
            _render_weekly_trends(history_df, datetime_col)

    # Monthly trends
    if st.session_state.get("show_monthly", True):
        if show_monthly:
            _render_monthly_trends(history_df, datetime_col)

    # Species distribution
    if show_species:
        st.divider()
        _render_species_distribution(history_df, datetime_col)

    # Occupancy
    if show_occupancy:
        st.divider()
        _render_occupancy(history_df, datetime_col)

    # Behaviour frequency
    if show_behaviour:
        st.divider()
        _render_behaviour_frequency(history_df, datetime_col)

    # Distress timeline
    if show_distress:
        st.divider()
        _render_distress_timeline(history_df, datetime_col)

    # Sensor history
    if show_sensor_history:
        st.divider()
        _render_sensor_history(history_df, datetime_col)

    # Inference statistics
    if show_inference_stats:
        st.divider()
        _render_inference_statistics(history_df, datetime_col)

    # Model confidence trends
    if show_confidence:
        st.divider()
        _render_model_confidence_trends(history_df, datetime_col)

    # Behaviour transitions
    if show_behaviour_transitions:
        st.divider()
        _render_behaviour_transition_trends(history_df, datetime_col)

    # Behaviour stability
    if show_behaviour_stability:
        st.divider()
        _render_behaviour_stability(history_df, datetime_col)

    # Risk timeline
    if show_risk_timeline:
        st.divider()
        _render_risk_timeline(history_df, datetime_col)

    # Trajectory heatmap proxy
    if show_trajectory_heatmap:
        st.divider()
        _render_trajectory_heatmap(history_df)

    # Alert trends
    if show_alerts:
        st.divider()
        if alerts_df is not None and not alerts_df.empty:
            alert_dt_col = _guess_datetime_col(alerts_df) or ("timestamp" if "timestamp" in alerts_df.columns else None)
            if alert_dt_col is None:
                st.info("Alert trends: could not determine alert timestamp column.")
            else:
                _render_alert_trends_from_df(alerts_df, alert_dt_col)
        else:
            st.info("Alert trends: no alert data available for the selected date range.")

