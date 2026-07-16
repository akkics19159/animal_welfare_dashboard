from __future__ import annotations

import streamlit as st

from ui.state import DashboardConfig, get_config, set_config


def _role_permissions() -> dict:
    return {
        "Administrator": [
            "Dashboard",
            "Live Monitoring",
            "History",
            "Analytics",
            "Evaluation",
            "Training",
            "Models",
            "Dataset",
            "Alerts",
            "System Health",
            "Settings",
            "User Management",
        ],
        "Researcher": [
            "Dashboard",
            "Live Monitoring",
            "History",
            "Analytics",
            "Evaluation",
            "Training",
            "Dataset",
            "System Health",
        ],
        "Operator": [
            "Dashboard",
            "Live Monitoring",
            "History",
            "Analytics",
            "Alerts",
            "System Health",
        ],
        "Viewer": ["Dashboard", "Live Monitoring", "History", "Analytics", "System Health"],
    }


def render_sidebar() -> str:
    """Render sidebar controls and return selected page name."""
    with st.sidebar:
        st.markdown(
            """
            <div style='padding: 8px 2px 4px 2px;'>
                <div style='display:flex;align-items:center;gap:10px;'>
                    <div style='width:34px;height:34px;border-radius:10px;background:#dbeafe;color:#1d4ed8;display:flex;align-items:center;justify-content:center;font-weight:800;'>SW</div>
                    <div>
                        <div style='font-weight:800;font-size:1.02rem;color:#ffffff;'>Sentient Welfare</div>
                        <div style='font-size:0.76rem;opacity:0.86;'>Enterprise Monitoring Platform</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        backend_online = False
        try:
            # Safe best-effort; API client is created in dashboard router.
            backend_online = st.session_state.get("backend_online", False)
        except Exception:
            backend_online = False

        st.markdown(
            (
                "<span style='display:inline-flex;align-items:center;gap:6px;background:rgba(16,185,129,0.16);border:1px solid rgba(110,231,183,0.45);border-radius:999px;padding:4px 10px;color:#d1fae5;font-weight:700;font-size:0.78rem;'>● API Backend Online</span>"
                if backend_online
                else "<span style='display:inline-flex;align-items:center;gap:6px;background:rgba(244,63,94,0.16);border:1px solid rgba(253,164,175,0.45);border-radius:999px;padding:4px 10px;color:#ffe4e6;font-weight:700;font-size:0.78rem;'>○ Local Fallback Mode (API Offline)</span>"
            ),
            unsafe_allow_html=True,
        )

        st.divider()

        st.subheader("👤 Account Role")
        selected_role = st.selectbox(
            "Current Role:",
            ["Administrator", "Researcher", "Operator", "Viewer"],
            key="role_select",
        )
        st.session_state.role = selected_role

        allowed_pages = _role_permissions().get(selected_role, ["Dashboard", "Live Monitoring"])

        st.divider()
        st.subheader("🧭 Navigation Modules")
        selected_page = st.radio("Select Module:", allowed_pages)

        st.divider()
        st.subheader("⚡ Configuration Tuner")
        cfg = get_config()
        cfg.video_weight = st.slider("Video Weight", 0.0, 1.0, float(cfg.video_weight), 0.1)
        cfg.audio_weight = st.slider("Audio Weight", 0.0, 1.0, float(cfg.audio_weight), 0.1)
        cfg.sensor_weight = st.slider("Sensor Weight", 0.0, 1.0, float(cfg.sensor_weight), 0.1)
        cfg.ontology_strength = st.slider("Ontology Strength", 0.0, 1.0, float(cfg.ontology_strength), 0.1)
        cfg.use_simulated_sensors = st.checkbox("Simulate Sensors", bool(cfg.use_simulated_sensors), key="sidebar_simulate_sensors")
        set_config(cfg)

        return selected_page

