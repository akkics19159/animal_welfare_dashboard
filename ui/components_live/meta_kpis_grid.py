from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from .kpi_card_live import kpi_card_live


def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


def render_meta_kpis_grid(data: Dict[str, Any]) -> None:
    video_result = data.get("video_result", {}) or {}
    detections = video_result.get("detections", []) or []

    # Sentient-being count (best-effort): number of tracks/ids if present; else raw detections count.
    if video_result.get("tracks"):
        sentient_count = _safe_len(video_result.get("tracks"))
    else:
        sentient_count = _safe_len(detections)

    # Species count (best-effort): unique detected classes.
    species_set = set()
    for d in detections:
        cls = d.get("final_species") or d.get("species") or d.get("class")
        if cls:
            species_set.add(str(cls))
    species_count = len(species_set)

    # Occupancy: if backend provides explicit occupancy use it; else sentient_count.
    occupancy = data.get("occupancy")
    if occupancy is None:
        occupancy = sentient_count

    behaviour_status = video_result.get("behaviour") or video_result.get("behaviour_status")
    if behaviour_status is None:
        agitated = bool(video_result.get("agitated", False))
        behaviour_status = "Agitated" if agitated else "Calm"

    confidence = data.get("confidence")
    if confidence is None:
        confidence = video_result.get("confidence")
    try:
        confidence_f = float(confidence) if confidence is not None else 0.0
    except Exception:
        confidence_f = 0.0

    agreement = data.get("agreement_score", 0.0)
    try:
        agreement_f = float(agreement)
    except Exception:
        agreement_f = 0.0

    if detections:
        det_agreement = [float(d.get("agreement_score", 0.0) or 0.0) for d in detections]
        if det_agreement:
            agreement_f = max(agreement_f, sum(det_agreement) / max(1, len(det_agreement)))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card_live(str(sentient_count), "Sentient count")
    with c2:
        kpi_card_live(str(species_count), "Species count")
    with c3:
        kpi_card_live(str(occupancy), "Occupancy")
    with c4:
        kpi_card_live(str(behaviour_status), "Behaviour status")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        kpi_card_live(f"{confidence_f:.3f}", "Confidence")
    with c6:
        kpi_card_live(str(len(detections)), "Detections")
    with c7:
        kpi_card_live(str(video_result.get("motion_score", 0.0)), "Motion score")
    with c8:
        kpi_card_live(f"{agreement_f:.3f}", "Agreement score")

