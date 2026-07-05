"""Compatibility wrapper for the new welfare reasoning engine.

The legacy ontology interface is preserved so the rest of the pipeline can keep
calling evaluate_rules() and ontology_penalty(). Internally, those calls now flow
through a configuration-driven hybrid reasoning engine.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from welfare_reasoning.engine import WelfareReasoningEngine
from welfare_reasoning.types import EvidenceFact, ReasoningInput


_ENGINE = WelfareReasoningEngine()

ONTOLOGY_RULES = [
    ("audio_distress", True, 0.6, "Vocal distress patterns detected"),
    ("motion_agitation", True, 0.5, "Agitated movement detected in video"),
    ("temp_high", True, 0.4, "High temperature may indicate stress"),
    ("heart_rate_extreme", True, 0.5, "Abnormal heart rate indicates physiological stress"),
    ("sentient_present", None, 0.0, "Presence of sentient being (enables other rules)"),
]


def _legacy_flags_to_reasoning_input(video_flags: dict, audio_flags: dict, sensor_flags: dict) -> ReasoningInput:
    evidence: List[EvidenceFact] = []
    video_flags = video_flags or {}
    audio_flags = audio_flags or {}
    sensor_flags = sensor_flags or {}

    if audio_flags.get("audio_distress") or audio_flags.get("distress"):
        evidence.append(EvidenceFact(name="distress_vocalization", value=True, weight=0.9, source="audio", explanation="Distress vocalization detected"))
    if video_flags.get("motion_agitation") or video_flags.get("pacing"):
        evidence.append(EvidenceFact(name="pacing", value=True, weight=0.8, source="vision", explanation="Agitated movement detected"))
    if video_flags.get("limping"):
        evidence.append(EvidenceFact(name="limping", value=True, weight=0.8, source="vision", explanation="Limping detected"))
    if sensor_flags.get("heart_rate_extreme"):
        evidence.append(EvidenceFact(name="heart_rate_extreme", value=True, weight=0.9, source="sensor", explanation="Abnormal heart rate"))
    if sensor_flags.get("temp_high"):
        evidence.append(EvidenceFact(name="temp_high", value=True, weight=0.7, source="sensor", explanation="Elevated temperature"))
    if video_flags.get("sleeping"):
        evidence.append(EvidenceFact(name="sleeping", value=True, weight=0.5, source="vision", explanation="Animal appears to be sleeping"))
    if video_flags.get("running"):
        evidence.append(EvidenceFact(name="running", value=True, weight=0.8, source="vision", explanation="Animal appears to be running"))
    if video_flags.get("sentient_present"):
        evidence.append(EvidenceFact(name="sentient_present", value=True, weight=0.2, source="vision", explanation="Animal present"))

    context = {
        "time_of_day": video_flags.get("time_of_day", "day"),
        "environment": video_flags.get("environment", "outdoor"),
        "crowding": video_flags.get("crowding", "low"),
        "occupancy": video_flags.get("sentient_count") if video_flags.get("sentient_count") is not None else (1 if video_flags.get("sentient_present") else 0),
    }

    return ReasoningInput(
        species=str(video_flags.get("species", "unknown")).lower() or "unknown",
        evidence=evidence,
        fusion_confidence=float(video_flags.get("fusion_confidence", 0.7)),
        sensor_reliability=float(sensor_flags.get("sensor_reliability", 0.7)),
        audio_quality=float(audio_flags.get("audio_quality", 0.7)),
        vision_quality=float(video_flags.get("vision_quality", 0.7)),
        prediction_uncertainty=float(video_flags.get("prediction_uncertainty", 0.2)),
        context=context,
    )


def evaluate_rules(video_flags: dict, audio_flags: dict, sensor_flags: dict) -> Tuple[float, List[str]]:
    """Evaluate welfare reasoning rules and return an ontology score plus explanation strings."""
    input_data = _legacy_flags_to_reasoning_input(video_flags, audio_flags, sensor_flags)
    result = _ENGINE.reason(input_data)
    explanations = [
        f"{item.label}: {item.detail}" for item in result.evidence
    ]
    explanations.extend(result.contradictions)
    return round(result.final_assessment_score, 4), explanations


def ontology_penalty(predicted_prob: float, ontology_score: float, strength: float = 0.5):
    """Compute a penalty to apply when prediction contradicts welfare reasoning."""
    adjusted = predicted_prob * (1 - strength) + ontology_score * strength
    return max(0.0, min(1.0, adjusted))
