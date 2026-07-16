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
        st.image("https://img.icons8.com/color/96/000000/dog-heart.png", width=64)
        st.markdown("<h2 style='margin-bottom:5px;'>Welfare Platform</h2>", unsafe_allow_html=True)

        backend_online = False
        try:
            # Safe best-effort; API client is created in dashboard router.
            backend_online = st.session_state.get("backend_online", False)
        except Exception:
            backend_online = False

        st.markdown(
            (
                "<span style='color:#10b981; font-weight:bold;'>● API Backend Online</span>"
                if backend_online
                else "<span style='color:#f43f5e; font-weight:bold;'>○ Local Fallback Mode (API Offline)</span>"
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

