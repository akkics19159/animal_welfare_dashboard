from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st


ALERT_HISTORY_PATH = "alerts.json"


@dataclass
class AlertItem:
    key: str
    category: str
    severity: str  # Info/Warning/Critical
    message: str
    details: Dict[str, Any]
    timestamp: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_alert_history() -> List[Dict[str, Any]]:
    try:
        with open(ALERT_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and isinstance(data.get("history"), list):
                return data["history"]
    except FileNotFoundError:
        return []
    except Exception:
        # Corrupt file -> start fresh rather than crash UI.
        return []
    return []


def _save_alert_history(history: List[Dict[str, Any]]) -> None:
    # Persist as a simple list for compatibility.
    with open(ALERT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _infer_confidence(data: Dict[str, Any]) -> Optional[float]:
    # Best-effort common keys.
    for k in ("confidence", "model_confidence", "reasoning_confidence"):
        if k in data and data.get(k) is not None:
            try:
                return float(data.get(k))
            except Exception:
                pass
    prob = data.get("probability")
    if prob is not None:
        try:
            # Convert probability into a pseudo-confidence when confidence is missing.
            p = float(prob)
            return max(0.0, min(1.0, abs(p - 0.5) * 2.0))
        except Exception:
            return None
    return None


def _detect_missing_modalities(data: Dict[str, Any]) -> List[str]:
    missing = []
    video_result = data.get("video_result") or {}
    audio_result = data.get("audio_result") or {}
    sensor_result = data.get("sensor_result") or {}

    if video_result.get("error"):
        missing.append("video")
    if audio_result.get("error"):
        missing.append("audio")
    if sensor_result.get("error"):
        missing.append("sensor")

    # Also check for empty results.
    if not video_result.get("detections") and not video_result.get("tracks"):
        # do not force; only informational unless an explicit missing signal exists
        pass

    return missing


def generate_alerts(data: Dict[str, Any]) -> List[AlertItem]:
    combined = data

    video_result = combined.get("video_result") or {}
    audio_result = combined.get("audio_result") or {}
    sensor_result = combined.get("sensor_result") or {}

    alerts: List[AlertItem] = []

    # Shared confidence / uncertainty gate
    conf = _infer_confidence(combined)

    # Camera failure (video error)
    if video_result.get("error") or video_result.get("missing"):
        alerts.append(
            AlertItem(
                key="camera_failure",
                category="Camera failure",
                severity="Critical",
                message="Camera/video inference failed or returned an error.",
                details={"error": video_result.get("error"), "missing": video_result.get("missing")},
                timestamp=_now_iso(),
            )
        )

    # Audio failure
    if audio_result.get("error"):
        alerts.append(
            AlertItem(
                key="audio_failure",
                category="Audio failure",
                severity="Critical",
                message="Audio inference failed or returned an error.",
                details={"error": audio_result.get("error")},
                timestamp=_now_iso(),
            )
        )

    # Sensor failure
    if sensor_result.get("error"):
        alerts.append(
            AlertItem(
                key="sensor_failure",
                category="Sensor failure",
                severity="Critical",
                message="Sensor readings failed or returned an error.",
                details={"error": sensor_result.get("error")},
                timestamp=_now_iso(),
            )
        )

    # Missing modality (explicit)
    missing_modalities = _detect_missing_modalities(combined)
    if missing_modalities:
        sev = "Critical" if len(missing_modalities) >= 2 else "Warning"
        alerts.append(
            AlertItem(
                key="missing_modality",
                category="Missing modality",
                severity=sev,
                message=f"Missing modalities detected: {', '.join(missing_modalities)}",
                details={"missing_modalities": missing_modalities},
                timestamp=_now_iso(),
            )
        )

    # Distress (from audio)
    audio_distress = bool(audio_result.get("distress") or audio_result.get("audio_distress", False))
    audio_distress_score = audio_result.get("score", audio_result.get("audio_distress_probability"))
    if audio_distress:
        alerts.append(
            AlertItem(
                key="distress",
                category="Distress",
                severity="Critical",
                message="Distress signal detected in audio.",
                details={"distress": audio_distress, "score": audio_distress_score},
                timestamp=_now_iso(),
            )
        )

    # Crowding (from occupancy/sentient count)
    occupancy = combined.get("occupancy")
    if occupancy is None:
        # fall back to video detections count
        occupancy = len(video_result.get("detections") or [])

    try:
        occ_f = float(occupancy)
    except Exception:
        occ_f = 0.0

    # Heuristic thresholds (UI-only; does not touch inference)
    if occ_f >= 12:
        alerts.append(
            AlertItem(
                key="crowding",
                category="Crowding",
                severity="Critical",
                message=f"High occupancy/crowding detected (occupancy={occ_f}).",
                details={"occupancy": occ_f},
                timestamp=_now_iso(),
            )
        )
    elif occ_f >= 7:
        alerts.append(
            AlertItem(
                key="crowding",
                category="Crowding",
                severity="Warning",
                message=f"Moderate crowding detected (occupancy={occ_f}).",
                details={"occupancy": occ_f},
                timestamp=_now_iso(),
            )
        )

    # High welfare risk (from welfare_probability / probability)
    prob = combined.get("probability")
    welfare_probability = combined.get("welfare_probability")
    risk_prob = welfare_probability if welfare_probability is not None else prob
    risk_prob_f = None
    if risk_prob is not None:
        try:
            risk_prob_f = float(risk_prob)
        except Exception:
            risk_prob_f = None

    if risk_prob_f is not None:
        if risk_prob_f >= 0.7:
            alerts.append(
                AlertItem(
                    key="high_welfare_risk",
                    category="High welfare risk",
                    severity="Critical",
                    message=f"High welfare risk (risk={risk_prob_f:.3f}).",
                    details={"welfare_risk": risk_prob_f},
                    timestamp=_now_iso(),
                )
            )
        elif risk_prob_f >= 0.4:
            alerts.append(
                AlertItem(
                    key="high_welfare_risk",
                    category="High welfare risk",
                    severity="Warning",
                    message=f"Elevated welfare risk (risk={risk_prob_f:.3f}).",
                    details={"welfare_risk": risk_prob_f},
                    timestamp=_now_iso(),
                )
            )

    # Low confidence
    if conf is not None:
        if conf < 0.35:
            alerts.append(
                AlertItem(
                    key="low_confidence",
                    category="Low confidence",
                    severity="Warning",
                    message=f"Low confidence in assessment (confidence={conf:.3f}).",
                    details={"confidence": conf},
                    timestamp=_now_iso(),
                )
            )

    # De-duplicate by key within a run (keep highest severity)
    severity_rank = {"Info": 0, "Warning": 1, "Critical": 2}
    best_by_key: Dict[str, AlertItem] = {}
    for a in alerts:
        if a.key not in best_by_key:
            best_by_key[a.key] = a
        else:
            if severity_rank[a.severity] > severity_rank[best_by_key[a.key].severity]:
                best_by_key[a.key] = a

    return list(best_by_key.values())


def _severity_to_streamlit(sev: str):
    if sev == "Critical":
        return st.error
    if sev == "Warning":
        return st.warning
    return st.info


def render_alert_center_panel(data: Dict[str, Any]) -> None:
    st.subheader("Alert Center")

    # Generate alerts based on UI-visible metadata.
    alerts = generate_alerts(data)

    # Render severity filter
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        show_info = st.checkbox("Info", True, key="alert_filter_info")
    with filter_col2:
        show_warning = st.checkbox(
            "Warning", True, key="alert_filter_warning"
        )
    with filter_col3:
        show_critical = st.checkbox(
            "Critical", True, key="alert_filter_critical"
        )

    sev_allowed = set()
    if show_info:
        sev_allowed.add("Info")
    if show_warning:
        sev_allowed.add("Warning")
    if show_critical:
        sev_allowed.add("Critical")

    # Store alert history (append-only).
    store_history = st.checkbox(
        "Store alert history", True, key="store_alert_history"
    )

    if store_history and alerts:
        history = _load_alert_history()
        for a in alerts:
            history.append(
                {
                    "key": a.key,
                    "category": a.category,
                    "severity": a.severity,
                    "message": a.message,
                    "details": a.details,
                    "timestamp": a.timestamp,
                }
            )
        _save_alert_history(history)

    # Filtered render
    visible = [a for a in alerts if a.severity in sev_allowed]

    if not visible:
        st.success("No alerts for this run.")
    else:
        for a in sorted(visible, key=lambda x: {"Info": 0, "Warning": 1, "Critical": 2}[x.severity], reverse=True):
            streamlit_printer = _severity_to_streamlit(a.severity)
            streamlit_printer(f"[{a.severity}] {a.category}: {a.message}")
            if a.details:
                with st.expander("Details", expanded=False):
                    st.json(a.details)
            st.caption(f"{a.timestamp}")

    st.divider()
    st.markdown("### Alert history")

    # History view (basic, best-effort)
    history = _load_alert_history()
    if not history:
        st.info("No alert history saved yet.")
        return

    # Use same filters on history
    history_visible = [h for h in history if h.get("severity") in sev_allowed]
    if not history_visible:
        st.info("No matching alerts in history for the current filter.")
        return

    # Show only last N
    history_visible = history_visible[-50:]
    st.dataframe(
        [
            {
                "timestamp": h.get("timestamp"),
                "severity": h.get("severity"),
                "category": h.get("category"),
                "message": h.get("message"),
            }
            for h in history_visible
        ],
        width="stretch",
        hide_index=True,
    )

