from __future__ import annotations

from typing import Dict, List

import streamlit as st


def _role_permissions() -> Dict[str, List[str]]:
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


def render_page(api, backend_online: bool, default_history_path: str, default_video_path: str) -> None:
    st.markdown("<h1 class='main-title'>👥 User Management</h1>", unsafe_allow_html=True)

    st.markdown(
        "Role-based access control (RBAC) UI. Backend authentication/authorization is not implemented in this refactor; "
        "this page provides permission mapping and simulated role selection."
    )

    roles = list(_role_permissions().keys())
    selected = st.selectbox("Select role for preview", roles, index=roles.index(st.session_state.get("role", "Administrator")) if st.session_state.get("role", None) in roles else 0)
    st.caption(f"Previewing permissions for: **{selected}**")

    perms = _role_permissions().get(selected, [])

    st.divider()
    st.subheader("Permissions")

    if not perms:
        st.info("No permissions configured for this role.")
        return

    # Render as two-column chips
    left, right = st.columns(2)
    midpoint = (len(perms) + 1) // 2
    for col, items in zip((left, right), (perms[:midpoint], perms[midpoint:])):
        with col:
            for p in items:
                st.write(f"• {p}")

    st.divider()
    st.subheader("Role-based permissions model")
    st.code(
        """\
- Administrator: full access
- Researcher: experiments + evaluation + training read/write
- Operator: live monitoring + alerts triage + history/analytics
- Viewer: read-only monitoring and analytics
"""
    )


