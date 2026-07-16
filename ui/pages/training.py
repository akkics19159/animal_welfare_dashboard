from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _render_learning_curves_from_history(history: List[Dict[str, Any]]) -> None:
    # The current API returns training.history as run summaries (final metrics),
    # while the active training status is epoch-by-epoch simulated.
    # We'll render what we can: for each run, show final loss/acc and validation.

    if not history:
        st.info("No training history available yet.")
        return

    # Build minimal tables
    rows = []
    for h in history:
        m = h.get("metrics") or {}
        cfg = h.get("config") or {}
        rows.append(
            {
                "experiment_id": h.get("experiment_id"),
                "pipeline_type": cfg.get("pipeline_type") or cfg.get("model_name"),
                "epochs": m.get("epochs_completed"),
                "final_loss": m.get("final_loss"),
                "final_val_loss": m.get("final_val_loss"),
                "final_accuracy": m.get("final_accuracy"),
                "final_val_accuracy": m.get("final_val_accuracy"),
            }
        )

    st.subheader("Past training runs (final metrics)")
    st.dataframe(rows, width="stretch", hide_index=True)


def _render_active_status(status: Dict[str, Any]) -> None:
    is_training = bool(status.get("is_training", False))
    pipeline_type = status.get("pipeline_type") or ""

    current_epoch = _safe_int(status.get("current_epoch"), 0)
    total_epochs = _safe_int(status.get("total_epochs"), 1)
    progress = _safe_float(status.get("progress"), 0.0)

    loss = _safe_float(status.get("loss"), 0.0)
    val_loss = _safe_float(status.get("val_loss"), 0.0)
    acc = _safe_float(status.get("accuracy"), 0.0)
    val_acc = _safe_float(status.get("val_accuracy"), 0.0)

    eta_seconds = _safe_int(status.get("eta_seconds"), 0)
    gpu_usage = _safe_float(status.get("gpu_usage"), 0.0)

    st.divider()
    st.subheader("Current training status")

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        st.metric("Pipeline", pipeline_type if pipeline_type else "N/A")
    with top2:
        st.metric("Epoch", f"{current_epoch}/{total_epochs}")
    with top3:
        st.metric("GPU utilization", f"{gpu_usage:.1f}%")
    with top4:
        st.metric("ETA (sec)", str(eta_seconds) if is_training else "N/A")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Loss", f"{loss:.4f}")
        st.metric("Val Loss", f"{val_loss:.4f}")
    with c2:
        st.metric("Accuracy", f"{acc:.4f}")
        st.metric("Val Accuracy", f"{val_acc:.4f}")

    st.divider()
    st.subheader("Progress")
    st.progress(min(max(progress, 0.0), 1.0))
    st.caption("Read-only status from `/api/training`. Training is simulated on this backend.")


def render_page(api, backend_online: bool, default_history_path: str, default_video_path: str) -> None:
    st.markdown("<h1 class='main-title'>🏋️ Training</h1>", unsafe_allow_html=True)

    st.info("This page is read-only: no training/retraining actions are provided. "
             "Model switches are allowed only from the Models page by configuration.")

    if not backend_online:
        st.caption("Backend health check was inconclusive. Trying `/api/training` directly...")

    # Load active training status + history
    try:
        resp = api.get("/api/training")
    except Exception as e:
        st.warning("API backend offline. Training status/history requires `/api/training`.")
        st.error(f"Failed to load training data: {e}")
        return

    status = (resp or {}).get("status") or {}
    history = (resp or {}).get("history") or []

    if status:
        _render_active_status(status)
    else:
        st.info("No active training status available.")

    st.divider()
    _render_learning_curves_from_history(history)

