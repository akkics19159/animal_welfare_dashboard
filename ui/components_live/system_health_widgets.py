from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from ui.components_live.kpi_card_live import kpi_card_live


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_str(x: Any, default: str = "n/a") -> str:
    if x is None:
        return default
    return str(x)


def render_system_health_kpis(health: Dict[str, Any]) -> None:
    cpu = _safe_float(health.get("cpu"))
    ram = _safe_float(health.get("ram"))
    disk = _safe_float(health.get("disk"))

    gpu_obj = health.get("gpu") or {}
    gpu_usage = _safe_float(gpu_obj.get("usage"))
    temp = _safe_float(gpu_obj.get("temp"))
    gpu_name = gpu_obj.get("name")

    inf_tp = health.get("inference_throughput")
    pipeline = health.get("pipeline") or {}
    model_status = health.get("model_status") or {}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card_live(f"{cpu:.1f}%" if cpu is not None else "n/a", "CPU")
    with c2:
        kpi_card_live(f"{ram:.1f}%" if ram is not None else "n/a", "RAM")
    with c3:
        kpi_card_live(f"{disk:.1f}%" if disk is not None else "n/a", "Disk")
    with c4:
        kpi_card_live(f"{inf_tp}" if inf_tp is not None else "n/a", "Inference throughput")

    st.divider()

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        kpi_card_live(f"{gpu_usage:.1f}%" if gpu_usage is not None else "n/a", "GPU usage")
    with c6:
        kpi_card_live(f"{temp:.1f}°C" if temp is not None else "n/a", "Temperature")
    with c7:
        kpi_card_live(_safe_str(gpu_name, "n/a"), "GPU")

    # FPS isn't returned by /api/health; keep it as best-effort.
    fps = health.get("fps")
    with c8:
        kpi_card_live(_safe_str(fps if fps is not None else pipeline.get("current_fps"), "n/a"), "FPS")

    st.divider()

    c9, c10, c11, c12 = st.columns(4)
    with c9:
        kpi_card_live(_safe_str(pipeline.get("status"), "n/a"), "Pipeline status")
    with c10:
        kpi_card_live(_safe_str(pipeline.get("current_input_type"), "n/a"), "Input source type")
    with c11:
        kpi_card_live(_safe_str(model_status.get("active_model"), "n/a"), "Active model")
    with c12:
        lat = _safe_float(pipeline.get("inference_latency_ms"))
        kpi_card_live(f"{lat:.2f} ms" if lat is not None else "n/a", "Inference latency")

    st.caption(
        f"Input source: {_safe_str(pipeline.get('current_input_source'), 'n/a')} | "
        f"Thread count: {_safe_str(pipeline.get('thread_count'), 'n/a')}"
    )


def render_device_status(health: Dict[str, Any]) -> None:
    devices = health.get("devices") or {}
    camera = devices.get("camera")
    microphone = devices.get("microphone")
    sensor_array = devices.get("sensor_array")

    st.subheader("Devices & sensors")

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card_live(_safe_str(camera, "Not Detected"), "Camera status")
    with c2:
        kpi_card_live(_safe_str(microphone, "Unavailable"), "Microphone status")
    with c3:
        kpi_card_live(_safe_str(sensor_array, "n/a"), "Sensor status")


def render_system_health(health: Dict[str, Any]) -> None:
    st.subheader("Resource health")
    render_system_health_kpis(health)
    render_device_status(health)

