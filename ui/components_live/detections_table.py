from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def render_detections_table(data: Dict[str, Any]) -> None:
    video_result = data.get("video_result", {}) or {}
    detections = video_result.get("detections", []) or []

    st.subheader("Annotated detections / YOLO boxes")
    st.caption("Best-effort detection metadata returned by the inference pipeline.")

    if not detections:
        st.info("No detections returned for the current run.")
        return

    df_rows = []
    for i, d in enumerate(detections):
        track_id = d.get("track_id") or d.get("id") or i
        df_rows.append(
            {
                "id": track_id,
                "class": d.get("class"),
                "detector_label": d.get("detector_label"),
                "classifier_label": d.get("classifier_label"),
                "final_species": d.get("final_species") or d.get("species"),
                "species_confidence": d.get("species_confidence"),
                "temporal_confidence": d.get("temporal_confidence"),
                "agreement_score": d.get("agreement_score"),
                "top5_predictions": d.get("top5_predictions") or d.get("top_5_predictions"),
                "classification_history": d.get("classification_history"),
                "confidence": d.get("confidence"),
                "xyxy": d.get("xyxy"),
            }
        )

    df = pd.DataFrame(df_rows)
    st.dataframe(df, width="stretch", hide_index=True)

