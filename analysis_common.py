"""Shared analysis helpers used by the dashboard and batch scripts.

This module centralizes the field names and row construction so the
dashboard, sample analysis script, and multimodal batch runner stay
in sync.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from multimodal_engine import analyze_combined


DEFAULT_WEIGHTS = {"video_score": 0.4, "audio_score": 0.4, "sensor_score": 0.2}


def build_flags(video_result: Dict[str, Any], audio_result: Dict[str, Any], sensor_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build the normalized flag dictionaries used across the project."""
    video_flags = {
        "sentient_present": len(video_result.get("detections", [])) > 0,
        "sentient_count": len(video_result.get("detections", [])),
        "motion_agitation": bool(video_result.get("agitated", False)),
        "motion_score": float(video_result.get("motion_score", 0.0)),
        "missing": bool(video_result.get("error")),
    }

    audio_flags = {
        "audio_distress": bool(audio_result.get("distress", False)),
        "score": float(audio_result.get("score", 0.0)),
        "missing": bool(audio_result.get("error")),
    }

    sensor_flags = {
        "temp_high": sensor_result.get("temp") is not None and sensor_result.get("temp") > 30,
        "humidity_extreme": sensor_result.get("humidity") is not None
        and (sensor_result.get("humidity") < 35 or sensor_result.get("humidity") > 75),
        "heart_rate_extreme": sensor_result.get("heart_rate") is not None
        and (sensor_result.get("heart_rate") < 55 or sensor_result.get("heart_rate") > 110),
        "missing": bool(sensor_result.get("error")),
    }

    return {"video": video_flags, "audio": audio_flags, "sensor": sensor_flags}


def run_analysis_bundle(
    video_result: Dict[str, Any],
    audio_result: Dict[str, Any],
    sensor_result: Dict[str, Any],
    ontology_strength: float = 0.6,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute flags and the combined multimodal result in one call."""
    flags = build_flags(video_result, audio_result, sensor_result)
    combined = analyze_combined(
        video_result,
        audio_result,
        sensor_result,
        ontology_strength=ontology_strength,
        weights=weights or DEFAULT_WEIGHTS,
    )
    return {"flags": flags, "combined": combined}


def build_canonical_report_row(
    source: str,
    combined: Dict[str, Any],
    annotated_frame: Optional[str] = None,
    audio_clip: Optional[str] = None,
    audio_error: Optional[str] = None,
    audio_clip_error: Optional[str] = None,
    sensor_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the standard report row shape used everywhere in the codebase."""
    return {
        "video": source,
        "probability": combined["probability"],
        "raw_probability": combined["raw_probability"],
        "video_score": combined["modality_scores"]["video_score"],
        "audio_score": combined["modality_scores"]["audio_score"],
        "sensor_score": combined["modality_scores"]["sensor_score"],
        "annotated_frame": annotated_frame,
        "audio_clip": audio_clip,
        "welfare_reasons": " | ".join(combined["explanations"]),
        "audio_error": audio_error,
        "audio_clip_error": audio_clip_error,
        "sensor_note": sensor_note,
    }


def build_history_record(timestamp: str, welfare_score: float, combined: Dict[str, Any]) -> Dict[str, Any]:
    """Build the history row stored by the dashboard."""
    return {
        "timestamp": timestamp,
        "welfare_score": welfare_score,
        "video_score": combined["modality_scores"]["video_score"],
        "audio_score": combined["modality_scores"]["audio_score"],
        "sensor_score": combined["modality_scores"]["sensor_score"],
        "probability": combined["probability"],
    }


def detect_sensor_note(use_simulated_sensors: bool, sensor_result: Dict[str, Any]) -> str:
    """Return a consistent sensor provenance note."""
    base = "simulated (use_simulation=True)" if use_simulated_sensors else "real sensor mode requested"
    error = sensor_result.get("error")
    if error:
        return f"{base}; error: {error}"
    return base
