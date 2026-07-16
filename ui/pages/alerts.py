from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from ui.state import get_global_date_range, filter_records_by_date_range


def _safe_list(x: Any) -> List[Dict[str, Any]]:
    return x if isinstance(x, list) else []


def _severity_sort_key(sev: str) -> int:
    # Info < Warning < Critical
    if sev == "Critical":
        return 2
    if sev == "Warning":
        return 1
    return 0


def render_page(api, backend_online: bool, default_history_path: str, default_video_path: str) -> None:
    st.markdown("<h1 class='main-title'>🔔 Alerts</h1>", unsafe_allow_html=True)
    st.markdown(
        "Configure and triage model/system welfare alerts. Alerts are sourced from the backend `/api/alerts`."
    )

    if not backend_online:
        st.caption("Backend health check was inconclusive. Trying `/api/alerts` directly...")

    try:
        alerts_raw = api.get("/api/alerts")
    except Exception as e:
        st.warning("API backend offline. Alerts page requires `/api/alerts`.")
        st.error(f"Failed to load alerts: {e}")
        return

    alerts: List[Dict[str, Any]] = _safe_list(alerts_raw)
    global_start, global_end = get_global_date_range()
    alerts = filter_records_by_date_range(
        alerts,
        global_start,
        global_end,
        date_fields=("timestamp", "created_at"),
        include_if_missing=False,
    )

    if not alerts:
        st.info("No alerts available in the selected global date range.")
        return

    st.caption(f"Global date range: {global_start} → {global_end}")
    unresolved_count = len([a for a in alerts if not a.get("acknowledged")])
    st.caption(f"Unresolved alerts: **{unresolved_count}**")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_info = st.checkbox("Info", True, key="alert_filter_info")
    with col2:
        show_warning = st.checkbox("Warning", True, key="alert_filter_warning")
    with col3:
        show_critical = st.checkbox("Critical", True, key="alert_filter_critical")

    allowed_sev = set()
    if show_info:
        allowed_sev.add("Info")
    if show_warning:
        allowed_sev.add("Warning")
    if show_critical:
        allowed_sev.add("Critical")

    # Source filter (optional)
    sources = sorted({str(a.get("source") or "Unknown") for a in alerts})
    source_sel = st.multiselect("Sources", options=sources, default=sources, key="alert_filter_sources")

    filtered = [
        a
        for a in alerts
        if (a.get("severity") in allowed_sev)
        and (str(a.get("source") or "Unknown") in set(source_sel))
    ]

    # Unacknowledged focus toggle
    only_unack = st.checkbox("Show only unresolved", True, key="alert_filter_only_unack")
    if only_unack:
        filtered = [a for a in filtered if not a.get("acknowledged")]

    # Sorting
    filtered = sorted(
        filtered,
        key=lambda x: ( _severity_sort_key(str(x.get("severity") or "Info")), str(x.get("timestamp") or "")),
        reverse=True,
    )

    st.divider()

    # Render alerts list with acknowledge actions
    st.subheader("Alert Center")

    if not filtered:
        st.info("No alerts match the current filters.")
        return

    for idx, a in enumerate(filtered):
        alert_id = str(a.get("id") or "")
        severity = str(a.get("severity") or "Info")
        category = str(a.get("category") or severity)
        message = str(a.get("message") or "")
        timestamp = str(a.get("timestamp") or "")
        source = str(a.get("source") or "Unknown")
        acknowledged = bool(a.get("acknowledged"))
        details = a.get("details") or {}

        header = f"[{severity}] {category}"
        with st.container():
            if severity == "Critical":
                st.error(header)
            elif severity == "Warning":
                st.warning(header)
            else:
                st.info(header)

            st.write(message)
            st.caption(f"Source: {source} • {timestamp}")

            if details:
                with st.expander("View evidence/details", expanded=False):
                    st.json(details)

            # Acknowledge action only when unresolved
            if not acknowledged and alert_id:
                ack_btn = st.button(
                    f"Acknowledge {alert_id}",
                    key=f"ack_{idx}_{alert_id}",
                    type="primary",
                )
                if ack_btn:
                    try:
                        api.post("/api/alerts/acknowledge", {"alert_id": alert_id})
                        st.success(f"Acknowledged alert {alert_id}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to acknowledge alert {alert_id}: {e}")

            if acknowledged:
                st.caption("Status: ✅ Acknowledged")

    st.divider()
    st.caption("Acknowledgements are persisted by the backend via `/api/alerts/acknowledge`.")

