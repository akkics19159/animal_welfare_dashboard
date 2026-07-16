from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


def _safe_str(x: Any, default: str = "") -> str:
    return str(x) if x is not None else default


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _render_models_table(models: List[Dict[str, Any]]) -> None:
    if not models:
        st.info("No models found in registry.")
        return

    rows = []
    for m in models:
        rows.append(
            {
                "model_version": m.get("model_version"),
                "name": m.get("name"),
                "type": m.get("type"),
                "dataset_version": m.get("dataset_version"),
                "active": m.get("active"),
                "accuracy": m.get("accuracy"),
                "f1_score": m.get("f1_score"),
                "latency_ms": m.get("latency_ms"),
                "memory_mb": m.get("memory_mb"),
            }
        )

    st.subheader("Model versions")
    st.dataframe(rows, width="stretch", hide_index=True)


def render_page(api, backend_online: bool, default_history_path: str, default_video_path: str) -> None:
    st.markdown("<h1 class='main-title'>📦 Model Registry</h1>", unsafe_allow_html=True)

    if not backend_online:
        st.caption("Backend health check was inconclusive. Trying `/api/models` directly...")

    try:
        models = api.get("/api/models")
    except Exception as e:
        st.warning("API backend offline. Model registry requires `/api/models`.")
        st.error(f"Failed to load model registry: {e}")
        return

    models = models if isinstance(models, list) else []
    if not models:
        st.info("No models present in registry.")
        return

    active = next((m for m in models if m.get("active")), None)

    st.divider()
    st.subheader("Active model")
    if active:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Active version", _safe_str(active.get("model_version"), "N/A"))
        with c2:
            st.metric("Accuracy", str(active.get("accuracy")) if active.get("accuracy") is not None else "N/A")
        with c3:
            st.metric("F1", str(active.get("f1_score")) if active.get("f1_score") is not None else "N/A")
        with c4:
            st.metric("Deployment", "Deployed (Active)" if active.get("active") else "Not deployed")
    else:
        st.info("No active model is flagged in registry.")

    st.divider()
    _render_models_table(models)

    # ---------------- Configuration-driven switching ----------------
    st.divider()
    st.subheader("Switch active model (configuration only)")

    options = [str(m.get("model_version")) for m in models if m.get("model_version")]
    options = sorted(set(options))

    if "selected_model_version" not in st.session_state:
        st.session_state.selected_model_version = _safe_str(active.get("model_version")) if active else (options[0] if options else "")

    selected = st.selectbox(
        "Select model version to activate",
        options=options,
        index=options.index(st.session_state.selected_model_version) if st.session_state.selected_model_version in options else 0,
        key="model_version_select",
    )

    # Keep session state in sync
    st.session_state.selected_model_version = selected

    # Read-only semantics: this is the only allowed action.
    colA, colB = st.columns([1, 1])
    with colA:
        st.caption("This does not retrain models; it only flips the active flag via `/api/models/active`.")
    with colB:
        apply_disabled = not bool(selected)
        if st.button("Apply / Activate", disabled=apply_disabled):
            try:
                api.post("/api/models/active", {"model_version": selected})
                st.success(f"Activated model: {selected}")
                st.rerun()
            except Exception as e:
                st.error(f"Activation failed: {e}")

