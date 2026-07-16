from __future__ import annotations

import streamlit as st


def kpi_card_live(value, label: str) -> None:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-label'>{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

