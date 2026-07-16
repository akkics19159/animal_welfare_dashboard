from __future__ import annotations

import streamlit as st
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DashboardConfig:
    video_weight: float = 0.4
    audio_weight: float = 0.4
    sensor_weight: float = 0.2
    ontology_strength: float = 0.6
    use_simulated_sensors: bool = True


DEFAULT_API_URL = "http://127.0.0.1:8000"


def ensure_session_defaults() -> None:
    """Initialize session state keys used across pages."""
    if "last_analysis" not in st.session_state:
        st.session_state.last_analysis = None
    if "role" not in st.session_state:
        st.session_state.role = "Administrator"
    if "config" not in st.session_state:
        st.session_state.config = DashboardConfig()


def get_config() -> DashboardConfig:
    ensure_session_defaults()
    return st.session_state.config


def set_config(cfg: DashboardConfig) -> None:
    st.session_state.config = cfg


def get_api_url() -> str:
    # Can be extended later for env var overrides.
    return DEFAULT_API_URL

