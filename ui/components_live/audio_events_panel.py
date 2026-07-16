from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def render_audio_events_panel(data: Dict[str, Any]) -> None:
    audio_result = data.get("audio_result", {}) or {}

    st.subheader("Audio events")

    # Best-effort common fields
    distress = audio_result.get("distress", audio_result.get("audio_distress", False))
    score = audio_result.get("score", audio_result.get("audio_distress_probability", 0.0))

    distress_bool = bool(distress)
    try:
        score_f = float(score)
    except Exception:
        score_f = 0.0

    st.write(f"Distress event: `{ 'Yes' if distress_bool else 'No' }`")
    st.write(f"Audio distress score: `{score_f:.3f}`")

    event_type = audio_result.get("event_type", "unknown_audio")
    confidence = float(audio_result.get("confidence", audio_result.get("audio_confidence", 0.0)) or 0.0)
    quality = float(audio_result.get("audio_quality", 0.0) or 0.0)
    temporal_consistency = float(audio_result.get("temporal_consistency", 0.0) or 0.0)
    non_distress_p = float(audio_result.get("non_distress_probability", 0.0) or 0.0)
    suppressed_reason = audio_result.get("suppressed_reason")

    st.write(f"Audio event type: `{event_type}`")
    st.write(f"Confidence: `{confidence:.3f}`")
    st.write(f"Audio quality: `{quality:.3f}`")
    st.write(f"Temporal consistency: `{temporal_consistency:.3f}`")
    st.write(f"Non-distress probability: `{non_distress_p:.3f}`")
    if suppressed_reason:
        st.warning(f"Suppression: {suppressed_reason}")
    else:
        st.caption("Suppression: none")

    # If pipeline returns event list
    events: List[Dict[str, Any]] = audio_result.get("events") or []
    if events:
        st.divider()
        for e in events:
            st.info(f"{e.get('label') or 'event'} — {e.get('timestamp') or ''} (score={e.get('score', 0.0)})")

    if audio_result.get("error"):
        st.warning(f"Audio inference error: {audio_result.get('error')}")

