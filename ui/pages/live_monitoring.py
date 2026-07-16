from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient
from ui.state import get_config

# IMPORTANT: Do not modify AI inference logic.
from video_module import analyze_video_behavior
from audio_module import capture_audio, detect_distress, extract_audio_file_features
from sensors import read_sensors
from multimodal_engine import analyze_combined

from ui.components_live import (
    maybe_autorefresh,
    render_audio_events_panel,
    render_detections_table,
    render_live_camera_view,
    render_meta_kpis_grid,
    render_performance_panel,
    render_sensor_values_panel,
    render_welfare_risk_panel,
    render_alert_center_panel,
    render_explainability_panel,
)



def _run_inference(api: ApiClient, backend_online: bool, cfg: Any, default_video_path: str) -> Dict[str, Any]:
    """Keep inference calls exactly as the original UI used."""

    # Video
    video_result = analyze_video_behavior(source=0, confidence=0.25)

    # Audio
    audio_result: Dict[str, Any] = {"distress": False, "score": 0.0, "error": None}
    try:
        audio_features = capture_audio(duration=3)
        audio_result = detect_distress(audio_features)
    except Exception as exc:
        # Fallback: analyze the current video stream audio when microphone is unavailable.
        try:
            audio_features = extract_audio_file_features(str(default_video_path))
            audio_result = detect_distress(audio_features)
            audio_result["error"] = f"Microphone unavailable, used video audio fallback: {exc}"
        except Exception as fallback_exc:
            audio_result["error"] = f"{exc}; fallback failed: {fallback_exc}"

    # Sensors
    sensor_result = read_sensors(use_simulation=cfg.use_simulated_sensors)

    # Fallback to sample if no detections
    if len(video_result.get("detections", [])) == 0:
        video_result = analyze_video_behavior(source=str(default_video_path), confidence=0.25)
        try:
            audio_features = extract_audio_file_features(str(default_video_path))
            audio_result = detect_distress(audio_features)
        except Exception as exc:
            audio_result["error"] = str(exc)

    weights = {
        "video_score": cfg.video_weight,
        "audio_score": cfg.audio_weight,
        "sensor_score": cfg.sensor_weight,
    }

    if backend_online:
        payload = {
            "video_result": video_result,
            "audio_result": audio_result,
            "sensor_result": sensor_result,
            "ontology_strength": cfg.ontology_strength,
            "weights": weights,
        }
        return api.post("/api/inference", payload)

    return analyze_combined(
        video_result,
        audio_result,
        sensor_result,
        ontology_strength=cfg.ontology_strength,
        weights=weights,
    )


def _get_active_model(api: ApiClient, backend_online: bool) -> str:
    if not backend_online:
        return "welfare-fusion-v1"

    try:
        models = api.get("/api/models")
        return next((m["model_version"] for m in models if m.get("active")), "welfare-fusion-v1")
    except Exception:
        return "welfare-fusion-v1"


def render_page(api: ApiClient, backend_online: bool, default_history_path: str, default_video_path: str) -> None:
    cfg = get_config()

    st.markdown("<h1 class='main-title'>📹 Live Monitoring</h1>", unsafe_allow_html=True)
    st.markdown("Live monitoring UI with best-effort visualization from inference metadata.")
    st.caption("Auto-refresh re-runs the UI; inference logic remains unchanged.")
    st.divider()

    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        run_btn = st.button("▶️ RUN LIVE INFERENCE", key="live_run_btn")

        auto_refresh_enabled = st.checkbox("Enable auto refresh", False, key="live_auto_refresh_enabled")
        refresh_seconds = st.slider("Auto refresh (seconds)", 1, 30, 5, 1, key="live_refresh_seconds")

        # Trigger rerun periodically (UI rerun; inference is performed when no cached data exists or when user hits RUN)
        # For real streaming, backend would need a dedicated endpoint. Here we keep inference calls untouched.
        maybe_autorefresh(auto_refresh_enabled, refresh_seconds, key="live_monitoring_autorefresh")

        st.divider()
        st.subheader("Model currently running")
        st.write(f"`{_get_active_model(api, backend_online)}`")

        st.divider()
        st.subheader("Behaviour status")
        # Behaviour status is best-effort derived from video_result
        if st.session_state.get("last_analysis"):
            combined_last = (st.session_state.last_analysis.get("combined") or {})
            video_result = (st.session_state.last_analysis.get("video_result") or {})
            agitated = bool(video_result.get("agitated", False))
            motion_score = float(video_result.get("motion_score", 0.0) or 0.0)
            behaviour = str(combined_last.get("behaviour") or video_result.get("behaviour") or ("Agitated" if agitated else "Calm"))
            behaviour_prob = float(combined_last.get("behaviour_probability", video_result.get("behaviour_probability", 0.0)) or 0.0)
            behaviour_stability = float(combined_last.get("behaviour_stability", video_result.get("behaviour_stability", 0.0)) or 0.0)
            st.write(f"Motion score: `{motion_score:.3f}`")
            st.write(f"Behaviour: `{behaviour}`")
            st.write(f"Behaviour probability: `{behaviour_prob:.3f}`")
            st.write(f"Behaviour stability: `{behaviour_stability:.3f}`")
        else:
            st.write("Click RUN LIVE INFERENCE to begin.")

        st.divider()
        # Audio + sensors shown on right for better scan path.

    # Execute inference on button press OR when auto-refresh triggers and last_analysis missing
    if run_btn or (auto_refresh_enabled and not st.session_state.get("last_analysis")):
        with st.spinner("Running live inference (video + audio + sensors)..."):
            combined = _run_inference(
                api,
                backend_online,
                cfg,
                default_video_path=str(default_video_path),
            )

            # Preserve a stable session contract.
            st.session_state.last_analysis = {
                "welfare_score": (1.0 - float(combined.get("probability", 0.0) or 0.0)) * 100.0,
                "probability": combined.get("probability", 0.0),
                "behaviour": combined.get("behaviour"),
                "behaviour_probability": combined.get("behaviour_probability"),
                "behaviour_confidence": combined.get("behaviour_confidence"),
                "behaviour_duration": combined.get("behaviour_duration"),
                "behaviour_history": combined.get("behaviour_history", []),
                "behaviour_transition": combined.get("behaviour_transition"),
                "behaviour_stability": combined.get("behaviour_stability"),
                "video_result": combined.get("video_result", {}),
                "audio_result": combined.get("audio_result", {}),
                "sensor_result": combined.get("sensor_result", {}),
                "combined": combined,
                **({"latency_ms": combined.get("latency_ms")} if combined.get("latency_ms") is not None else {}),
            }

    data = st.session_state.get("last_analysis")
    if not data:
        st.info("Click RUN LIVE INFERENCE to start camera/audio/sensor fusion.")
        return

    combined = data["combined"]

    # Unify data structure expected by widgets
    data_to_render: Dict[str, Any] = {
        **combined,
        "video_result": data.get("video_result", {}) or {},
        "audio_result": data.get("audio_result", {}) or {},
        "sensor_result": data.get("sensor_result", {}) or {},
    }

    # ---- Right column: all required panels ----
    with col_right:
        # 1) Live camera feed (with YOLO + tracked IDs overlay if xyxy available)
        render_live_camera_view(data_to_render)

        st.divider()

        # 2) Sentient / species / occupancy / behaviour / confidence grid
        render_meta_kpis_grid(data_to_render)

        st.divider()

        # 3) Welfare score + Risk level + Confidence
        render_welfare_risk_panel(data_to_render)

        st.divider()

        # 4) Inference latency + FPS + model
        render_performance_panel(
            {
                **data_to_render,
                "model_version": data_to_render.get("model_version") or data_to_render.get("active_model"),
            }
        )

        st.divider()

        # 5) Annotated detections / YOLO boxes / tracked IDs
        render_detections_table(data_to_render)

        st.divider()

        # 6) Audio events
        render_audio_events_panel(data_to_render)

        st.divider()

        # 7) Sensor values
        render_sensor_values_panel(data_to_render)

    # 8) Alert Center + Explainability panels (UI-only)
    from ui.components_live import render_alert_center_panel, render_explainability_panel

    st.divider()
    with st.expander("Alert Center", expanded=False):
        render_alert_center_panel(data_to_render)

    with st.expander("Explainability Panel", expanded=False):
        render_explainability_panel(data_to_render)



