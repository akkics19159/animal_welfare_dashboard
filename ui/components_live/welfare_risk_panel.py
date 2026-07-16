from __future__ import annotations

from typing import Any, Dict

import plotly.graph_objects as go
import streamlit as st


def _compute_risk_level(prob: float, warning: float = 0.4, critical: float = 0.7) -> str:
    if prob >= critical:
        return "CRITICAL"
    if prob >= warning:
        return "WARNING"
    return "NORMAL"


def render_welfare_risk_panel(data: Dict[str, Any]) -> None:
    probability = data.get("probability")
    if probability is None:
        probability = (1.0 - float(data.get("welfare_probability", 0.0)))

    try:
        prob_f = float(probability) if probability is not None else 0.0
    except Exception:
        prob_f = 0.0

    welfare_score = float(data.get("welfare_score", (1.0 - prob_f) * 100.0) or ((1.0 - prob_f) * 100.0))
    risk = str(data.get("risk_level", _compute_risk_level(prob_f))).upper()
    severity = str(data.get("severity", "low")).upper()
    urgency = str(data.get("urgency", "routine")).upper()
    agreement_score = float(data.get("agreement_score", 0.0) or 0.0)

    confidence = data.get("confidence")
    if confidence is None:
        confidence = data.get("model_confidence")

    try:
        conf_f = float(confidence) if confidence is not None else None
    except Exception:
        conf_f = None

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=welfare_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Welfare score (0-100)"},
            gauge={
                "axis": {"range": [None, 100]},
                "steps": [
                    {"range": [0, 40], "color": "rgba(239, 68, 68, 0.25)"},
                    {"range": [40, 70], "color": "rgba(245, 158, 11, 0.22)"},
                    {"range": [70, 100], "color": "rgba(16, 185, 129, 0.22)"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 70},
            },
        )
    )
    fig.update_layout(height=220, template="plotly_dark", margin=dict(t=30, b=0, l=10, r=10))

    st.subheader("Welfare score & risk level")
    st.plotly_chart(fig, width="stretch")
    st.write(f"Risk level: ` {risk} `")
    st.write(f"Severity: ` {severity} `")
    st.write(f"Urgency: ` {urgency} `")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Probability", f"{prob_f:.3f}")
    with col2:
        st.metric("Confidence", "n/a" if conf_f is None else f"{conf_f:.3f}")
    with col3:
        st.metric("Agreement", f"{agreement_score:.3f}")

    temporal = data.get("temporal_welfare_state", {}) or {}
    timeline = temporal.get("timeline", []) if isinstance(temporal, dict) else []
    if timeline:
        ts = [str(item.get("timestamp")) for item in timeline]
        ys = [float(item.get("score", 0.0) or 0.0) for item in timeline]
        tfig = go.Figure()
        tfig.add_trace(go.Scatter(x=ts, y=ys, mode="lines+markers", name="Welfare timeline"))
        tfig.update_layout(height=220, template="plotly_dark", margin=dict(t=30, b=20, l=10, r=10), title="Temporal Welfare Timeline")
        st.plotly_chart(tfig, width="stretch")

