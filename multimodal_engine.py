"""Compatibility wrapper for the new multimodal fusion engine."""

from typing import Dict

from fusion_engine.engine import MultimodalFusionEngine
from fusion_engine.types import FeatureVector, FusionInput
from ontology import evaluate_rules, ontology_penalty


_ENGINE = MultimodalFusionEngine()


def compute_modality_scores(video_flags: Dict, audio_flags: Dict, sensor_flags: Dict):
    """Backward-compatible scoring helper for the legacy interface."""
    motion = float(video_flags.get("motion_score", 0.0))
    agitated = 1.0 if video_flags.get("motion_agitation") else 0.0
    present = 1.0 if video_flags.get("sentient_present") else 0.0
    video_score = min(1.0, 0.6 * agitated + 0.4 * motion + 0.2 * present)

    audio_score = float(audio_flags.get("score", 0.0)) if audio_flags else 0.0

    s_score = 0.0
    if sensor_flags.get("temp_high"):
        s_score += 0.4
    if sensor_flags.get("humidity_extreme"):
        s_score += 0.2
    if sensor_flags.get("heart_rate_extreme"):
        s_score += 0.5
    sensor_score = min(1.0, s_score)

    return {
        "video_score": video_score,
        "audio_score": audio_score,
        "sensor_score": sensor_score,
    }


def fuse_scores(scores: Dict, weights=None):
    """Backward-compatible weighted sum wrapper."""
    if weights is None:
        weights = {"video_score": 0.4, "audio_score": 0.4, "sensor_score": 0.2}
    total = 0.0
    weighted = 0.0
    for k, w in weights.items():
        total += w
        weighted += scores.get(k, 0.0) * w
    prob = weighted / total if total > 0 else 0.0
    return max(0.0, min(1.0, prob))


def analyze_combined(video_result: Dict, audio_result: Dict, sensor_result: Dict, ontology_strength=0.5, weights=None):
    """Produce a combined suffering probability and textual explanation using the new fusion engine."""
    video_flags = {
        "sentient_present": len(video_result.get("detections", [])) > 0,
        "sentient_count": len(video_result.get("detections", [])),
        "motion_agitation": video_result.get("agitated", False),
        "motion_score": video_result.get("motion_score", 0.0),
        "missing": bool(video_result.get("error")),
    }

    audio_flags = {
        "audio_distress": bool(audio_result.get("distress", False)),
        "score": float(audio_result.get("score", 0.0)),
        "missing": bool(audio_result.get("error")),
    }

    sensor_flags = {
        "temp_high": sensor_result.get("temp") is not None and sensor_result.get("temp") > 30,
        "humidity_extreme": sensor_result.get("humidity") is not None and (sensor_result.get("humidity") < 35 or sensor_result.get("humidity") > 75),
        "heart_rate_extreme": sensor_result.get("heart_rate") is not None and (sensor_result.get("heart_rate") < 55 or sensor_result.get("heart_rate") > 110),
        "missing": bool(sensor_result.get("error")),
    }

    modality_scores = compute_modality_scores(video_flags, audio_flags, sensor_flags)
    raw_prob = fuse_scores(modality_scores, weights=weights)

    ontology_score, ontology_explanations = evaluate_rules(video_flags, audio_flags, sensor_flags)
    adjusted_prob = ontology_penalty(raw_prob, ontology_score, strength=ontology_strength)

    vision_feature = FeatureVector(
        modality="vision",
        timestamp=0.0,
        values={"motion_score": video_flags.get("motion_score", 0.0), "agitated": 1.0 if video_flags.get("motion_agitation") else 0.0},
        confidence=0.8,
        quality_score=0.8,
        version="1.0",
    )
    audio_feature = FeatureVector(
        modality="audio",
        timestamp=0.0,
        values={"audio_score": float(audio_flags.get("score", 0.0))},
        confidence=0.8,
        quality_score=0.8,
        version="1.0",
    )
    sensor_feature = FeatureVector(
        modality="sensors",
        timestamp=0.0,
        values={"temp_high": 1.0 if sensor_flags.get("temp_high") else 0.0, "humidity_extreme": 1.0 if sensor_flags.get("humidity_extreme") else 0.0, "heart_rate_extreme": 1.0 if sensor_flags.get("heart_rate_extreme") else 0.0},
        confidence=0.8,
        quality_score=0.8,
        version="1.0",
    )

    output = _ENGINE.fuse(FusionInput(vision=vision_feature, audio=audio_feature, sensors=sensor_feature, timestamp=0.0))
    output.welfare_probability = adjusted_prob
    output.confidence = max(output.confidence, output.welfare_probability)

    explanations = []
    explanations.append(f"Raw fused probability: {raw_prob:.3f}")
    explanations.append(f"Ontology score: {ontology_score:.3f}")
    explanations.extend(ontology_explanations)
    explanations.append(f"Adjusted probability: {adjusted_prob:.3f}")

    return {
        "probability": adjusted_prob,
        "raw_probability": raw_prob,
        "modality_scores": modality_scores,
        "ontology_score": ontology_score,
        "explanations": explanations,
        "fusion_output": output,
    }
