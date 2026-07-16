from __future__ import annotations

import streamlit as st


def render_page(api, backend_online: bool, default_history_path: str, default_video_path: str) -> None:
    st.markdown("<h1 class='main-title'>⚙️ Settings</h1>", unsafe_allow_html=True)
    st.info("Settings UI is not implemented yet in this architecture refactor milestone.")

