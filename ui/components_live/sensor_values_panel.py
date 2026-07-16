from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_sensor_values_panel(data: Dict[str, Any]) -> None:
    sensor_result = data.get("sensor_result", {}) or {}

    st.subheader("Sensor values")

    # Common keys (best-effort)
    keys = ["temp", "temperature", "humidity", "heart_rate", "heartRate", "oxygen", "co2"]
    rows = []
    for k in keys:
        if k in sensor_result:
            rows.append((k, sensor_result.get(k)))

    # If no known keys found, show all.
    if not rows and sensor_result:
        for k, v in sensor_result.items():
            rows.append((k, v))

    if not rows:
        st.info("No sensor values available.")
        return

    cols = st.columns(2)
    for i, (k, v) in enumerate(rows):
        with cols[i % 2]:
            if isinstance(v, (int, float)):
                st.write(f"**{k}**: `{v:.3f}`")
            else:
                st.write(f"**{k}**: `{v}`")

