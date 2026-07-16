from __future__ import annotations
import streamlit as st


def render_page(title: str) -> None:
    st.markdown(f"<h1 class='main-title'>{title}</h1>", unsafe_allow_html=True)
    st.info("Not implemented in this refactor milestone.")

