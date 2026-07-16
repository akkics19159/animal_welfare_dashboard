from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_performance_panel(data: Dict[str, Any]) -> None:
    # Inference latency
    latency_ms = data.get("latency_ms")
    if latency_ms is None:
        latency_ms = data.get("inference_latency_ms")
    if latency_ms is None:
        latency_ms = data.get("processing_latency_ms")

    # FPS
    fps = data.get("fps")
    if fps is None:
        fps = data.get("inference_fps")
    if fps is None:
        fps = data.get("inference_throughput")

    # Model
    model = data.get("model_version") or data.get("active_model")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if latency_ms is None:
            st.metric("Inference latency", "n/a")
        else:
            try:
                st.metric("Inference latency", f"{float(latency_ms):.1f} ms")
            except Exception:
                st.metric("Inference latency", str(latency_ms))

    with c2:
        if fps is None:
            st.metric("FPS", "n/a")
        else:
            st.metric("FPS", str(fps))

    with c3:
        st.metric("Model currently running", str(model) if model else "n/a")

