from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from ui.api_client import ApiClient

from ui.components_live.evaluation_widgets import (
    render_calibration,
    render_confusion_matrix,
    render_latency_memory,
    render_kpis_for_metrics,
    render_model_comparison,
    render_roc_curve,
)


def _pick_latest_experiment(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not history:
        return {}

    # history entries appear as an ordered list; latest is last.
    # Do not compute/alter evaluation logic.
    return history[-1] if isinstance(history[-1], dict) else {}


def render_page(api: ApiClient, backend_online: bool, default_history_path: str, default_video_path: str) -> None:
    st.markdown("<h1 class='main-title'>📊 Evaluation</h1>", unsafe_allow_html=True)
    st.markdown(
        "Evaluation results dashboard. Widgets are rendered from backend `/api/evaluation` experiment history. "
        "Metric computation logic is not modified."
    )

    if not backend_online:
        st.caption("Backend health check was inconclusive. Trying `/api/evaluation` directly...")

    try:
        history = api.get("/api/evaluation")
    except Exception as e:
        st.warning("API backend offline: evaluation history is not available.")
        st.error(f"Failed to load evaluation history: {e}")
        return

    if not isinstance(history, list) or not history:
        st.info("No evaluation experiments found.")
        return

    # Select experiment
    names = []
    by_name: Dict[str, Dict[str, Any]] = {}
    for entry in history:
        if not isinstance(entry, dict):
            continue
        cfg = entry.get("config") or {}
        name = entry.get("name") or cfg.get("model") or "unknown"
        # Keep unique keys
        key = str(name)
        if key in by_name:
            key = f"{key}_{len(by_name)}"
        by_name[key] = entry
        names.append(key)

    selected_key = st.selectbox("Select evaluation run", options=names, index=len(names) - 1)
    selected = by_name[selected_key]

    metrics = selected.get("metrics") or {}

    # Header / summary
    st.divider()
    st.subheader("Core metrics")
    render_kpis_for_metrics(metrics)

    # Confusion matrix
    st.divider()
    st.subheader("Confusion Matrix")
    render_confusion_matrix(metrics)

    # ROC
    st.divider()
    st.subheader("ROC")
    render_roc_curve(metrics)

    # Calibration
    st.divider()
    st.subheader("Calibration")
    render_calibration(metrics)

    # Latency / Memory
    st.divider()
    st.subheader("Latency & Memory")
    render_latency_memory(metrics)

    # Model comparison
    st.divider()
    render_model_comparison(history)

