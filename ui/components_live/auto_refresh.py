from __future__ import annotations

import streamlit as st


def maybe_autorefresh(enabled: bool, interval_seconds: int, key: str) -> None:
    """Trigger periodic reruns when enabled.

    Uses Streamlit's built-in autorefresh mechanism when available.
    Does not touch inference logic; only reruns the UI.
    """

    if not enabled:
        return

    # Streamlit provides st_autorefresh in newer versions; fall back gracefully.
    try:
        from streamlit_autorefresh import st_autorefresh  # type: ignore

        st_autorefresh(interval=interval_seconds * 1000, limit=None, key=key)
        return
    except Exception:
        pass

    # Fallback: best-effort using st.session_state timestamp.
    now = st.session_state.get("_live_now", 0)
    st.session_state["_live_now"] = now + 1

    # Minimal fallback: rely on user interaction to rerun if autorefresh module isn't present.
    st.caption("Auto-refresh module not available; rerun will occur on interaction.")

