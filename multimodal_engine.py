"""Compatibility wrapper for the new multimodal fusion engine."""

from typing import Any, Dict, List

from fusion_engine.engine import MultimodalFusionEngine
from fusion_engine.types import FeatureVector, FusionInput
from ontology import evaluate_rules_detailed, ontology_penalty
from behaviour_recognition import BehaviourRecognitionEngine


_ENGINE = MultimodalFusionEngine()
_BEHAVIOUR_ENGINE = BehaviourRecognitionEngine()


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


def _risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "high"
    if prob >= 0.45:
        return "moderate"
    return "low"


def _severity(prob: float) -> str:
    if prob >= 0.8:
        return "critical"
    if prob >= 0.6:
        return "elevated"
    return "low"


def _urgency(prob: float) -> str:
    if prob >= 0.8:
        return "immediate"
    if prob >= 0.5:
        return "soon"
    return "routine"


def _calculate_agreement(modality_scores: Dict[str, float]) -> float:
    vals = [float(modality_scores.get(k, 0.0) or 0.0) for k in ("video_score", "audio_score", "sensor_score")]
    if len(vals) <= 1:
        return 1.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = var ** 0.5
    return max(0.0, min(1.0, 1.0 - 2.0 * std))


def _false_positive_suppression(video_result: Dict[str, Any], audio_result: Dict[str, Any]) -> Dict[str, Any]:
    non_distress_audio = float(audio_result.get("non_distress_probability", 0.0) or 0.0)
    event_type = str(audio_result.get("event_type", "unknown_audio") or "unknown_audio").lower()
    visual_normal_flags = {
        "mating",
        "courtship",
        "play",
        "feeding",
        "drinking",
        "grooming",
        "parent_offspring_communication",
        "social_communication",
        "exploration",
        "curiosity",
        "resting",
        "sleeping",
    }

    present_visual_normals = [f for f in visual_normal_flags if bool(video_result.get(f, False))]
    suppressed = False
    reason = None
    suppression_factor = 1.0

    if present_visual_normals and (non_distress_audio >= 0.55 or event_type in visual_normal_flags):
        suppressed = True
        suppression_factor = 0.45
        reason = (
            "Suppressed distress contribution due to normal behaviour + high-energy audio: "
            + ", ".join(sorted(present_visual_normals))
        )

    return {
        "suppressed": suppressed,
        "suppression_factor": suppression_factor,
        "reason": reason,
        "visual_non_distress": present_visual_normals,
    }


def analyze_combined(video_result: Dict, audio_result: Dict, sensor_result: Dict, ontology_strength=0.5, weights=None):
    """Produce a combined suffering probability and textual explanation using the new fusion engine."""
    tracks = video_result.get("tracks", []) or []
    for tr in tracks:
        final_species = tr.get("final_species") or tr.get("species") or "unknown"
        tr["final_species"] = final_species
        tr["species"] = final_species

    species_candidates = [
        str(t.get("final_species") or t.get("species") or "").strip()
        for t in tracks
        if str(t.get("final_species") or t.get("species") or "").strip()
    ]
    primary_species = species_candidates[0] if species_candidates else "unknown"

    video_flags = {
        "sentient_present": len(video_result.get("detections", [])) > 0,
        "sentient_count": int(video_result.get("occupancy", len(video_result.get("tracks", [])) or len(video_result.get("detections", [])))),
        "motion_agitation": video_result.get("agitated", False),
        "motion_score": video_result.get("motion_score", 0.0),
        "species": primary_species,
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

    agreement_score = _calculate_agreement(modality_scores)

    ontology_score, ontology_explanations, ontology_details = evaluate_rules_detailed(video_flags, audio_flags, sensor_flags)
    adjusted_prob = ontology_penalty(raw_prob, ontology_score, strength=ontology_strength)

    suppression = _false_positive_suppression(video_result, audio_result)
    if suppression["suppressed"]:
        adjusted_prob = float(max(0.0, min(1.0, adjusted_prob * suppression["suppression_factor"])))

    behaviour_result = _BEHAVIOUR_ENGINE.analyze(
        video_result=video_result,
        audio_result=audio_result,
        sensor_result=sensor_result,
    )

    # Keep behaviour fields available for legacy and dashboard consumers.
    video_result["behaviour"] = behaviour_result.behaviour
    video_result["behaviour_probability"] = behaviour_result.behaviour_probability
    video_result["behaviour_confidence"] = behaviour_result.behaviour_confidence
    video_result["behaviour_duration"] = behaviour_result.behaviour_duration
    video_result["behaviour_history"] = behaviour_result.behaviour_history
    video_result["behaviour_transition"] = behaviour_result.behaviour_transition
    video_result["behaviour_stability"] = behaviour_result.behaviour_stability

    if tracks:
        avg_speed = sum(float(((t.get("velocity") or [0.0, 0.0])[0] ** 2 + (t.get("velocity") or [0.0, 0.0])[1] ** 2) ** 0.5) for t in tracks) / len(tracks)
        avg_acc = sum(float(((t.get("acceleration") or [0.0, 0.0])[0] ** 2 + (t.get("acceleration") or [0.0, 0.0])[1] ** 2) ** 0.5) for t in tracks) / len(tracks)
        avg_dwell = sum(float(t.get("dwell_time", 0.0) or 0.0) for t in tracks) / len(tracks)
        avg_tracking_conf = sum(float(t.get("tracking_confidence", 0.0) or 0.0) for t in tracks) / len(tracks)
    else:
        avg_speed = 0.0
        avg_acc = 0.0
        avg_dwell = 0.0
        avg_tracking_conf = 0.0

    vision_feature = FeatureVector(
        modality="vision",
        timestamp=0.0,
        values={
            "motion_score": video_flags.get("motion_score", 0.0),
            "agitated": 1.0 if video_flags.get("motion_agitation") else 0.0,
            "occupancy": float(video_result.get("occupancy", 0.0) or 0.0),
            "velocity": float(avg_speed),
            "acceleration": float(avg_acc),
            "trajectory_density": float(video_result.get("average_occupancy", 0.0) or 0.0),
            "dwell_time": float(avg_dwell),
            "tracking_confidence": float(avg_tracking_conf),
        },
        confidence=0.8,
        quality_score=0.8,
        version="1.0",
    )
    audio_feature = FeatureVector(
        modality="audio",
        timestamp=0.0,
        values={
            "audio_score": float(audio_flags.get("score", 0.0)),
            "distress_probability": float(audio_result.get("distress_probability", audio_flags.get("score", 0.0)) or 0.0),
            "confidence": float(audio_result.get("confidence", audio_result.get("audio_confidence", 0.0)) or 0.0),
            "non_distress_probability": float(audio_result.get("non_distress_probability", 0.0) or 0.0),
            "event_duration": float(audio_result.get("event_duration", 0.0) or 0.0),
            "audio_quality": float(audio_result.get("audio_quality", 0.0) or 0.0),
            "temporal_consistency": float(audio_result.get("temporal_consistency", 0.0) or 0.0),
        },
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
    output.agreement_score = float(max(output.agreement_score, agreement_score))
    output.confidence = max(output.confidence, 0.5 * output.welfare_probability + 0.5 * output.agreement_score)

    prediction_uncertainty = float(max(0.0, min(1.0, 1.0 - output.confidence)))
    missing_modality_impact = float(output.unified_representation.get("missing_modality_impact", 0.0) or 0.0)
    calibration_score = float(output.calibration_score or output.confidence)

    temporal_welfare_state = {
        "trend": float(output.temporal_context.get("trend", 0.0) if isinstance(output.temporal_context, dict) else 0.0),
        "timeline": output.temporal_context.get("welfare_timeline", []) if isinstance(output.temporal_context, dict) else [],
        "state": ontology_details.get("temporal_welfare_state", {}).get("state", "stable"),
    }

    conflicts: List[str] = []
    if suppression["suppressed"]:
        conflicts.append("Vision indicates normal behavior while audio was high-energy; suppression applied")
    if sensor_flags.get("temp_high") and not video_flags.get("motion_agitation") and not audio_flags.get("audio_distress"):
        conflicts.append("Sensor abnormality observed while behavior appears normal")
    if audio_flags.get("audio_distress") and not video_flags.get("sentient_present"):
        conflicts.append("Audio distress detected but no visible sentient being")

    explanations = []
    explanations.append(f"Raw fused probability: {raw_prob:.3f}")
    explanations.append(f"Ontology score: {ontology_score:.3f}")
    explanations.extend(ontology_explanations)
    explanations.extend(conflicts)
    if suppression["reason"]:
        explanations.append(str(suppression["reason"]))
    explanations.append(f"Adjusted probability: {adjusted_prob:.3f}")

    risk_level = _risk_level(adjusted_prob)
    severity = _severity(adjusted_prob)
    urgency = _urgency(adjusted_prob)

    evidence_summary = {
        "primary_evidence": ontology_details.get("evidence_summary", {}).get("primary_evidence", []),
        "supporting_evidence": ontology_details.get("evidence_summary", {}).get("supporting_evidence", []),
        "contradictory_evidence": ontology_details.get("evidence_summary", {}).get("contradictory_evidence", []),
        "conflicts": conflicts,
    }

    suppressed_evidence = list(ontology_details.get("suppressed_evidence", []))
    if suppression["suppressed"]:
        suppressed_evidence.append(
            {
                "name": "normal_behavior_suppression",
                "source": "multimodal_conflict_resolver",
                "reason": suppression["reason"],
            }
        )

    reasoning_trace = list(ontology_details.get("reasoning_trace", []))
    reasoning_trace.append(f"Computed multimodal agreement={agreement_score:.3f}")
    reasoning_trace.append(f"Final welfare probability={adjusted_prob:.3f}; risk={risk_level}")

    explainability = {
        "triggered_rules": ontology_details.get("triggered_rules", []),
        "evidence": ontology_details.get("evidence_summary", {}).get("primary_evidence", [])
        + ontology_details.get("evidence_summary", {}).get("supporting_evidence", []),
        "feature_contribution": output.feature_contributions,
        "modality_contribution": output.reliability_scores,
        "attention_weights": output.unified_representation.get("attention_weights", {}),
        "confidence": float(output.confidence),
        "contradictions": ontology_details.get("contradictions", []) + conflicts,
        "alternative_explanations": ontology_details.get("alternative_explanations", []),
        "recommended_human_action": bool(ontology_details.get("urgency", "routine") == "immediate" or prediction_uncertainty > 0.4),
        "summary": ontology_details.get("summary", ""),
    }

    return {
        "probability": adjusted_prob,
        "raw_probability": raw_prob,
        "modality_scores": modality_scores,
        "ontology_score": ontology_score,
        "explanations": explanations,
        "fusion_output": output,
        "welfare_score": float((1.0 - adjusted_prob) * 100.0),
        "risk_level": risk_level,
        "severity": severity,
        "urgency": urgency,
        "confidence": float(output.confidence),
        "prediction_uncertainty": prediction_uncertainty,
        "agreement_score": float(max(output.agreement_score, agreement_score)),
        "evidence_summary": evidence_summary,
        "suppressed_evidence": suppressed_evidence,
        "reasoning_trace": reasoning_trace,
        "temporal_welfare_state": temporal_welfare_state,
        "calibration_score": calibration_score,
        "missing_modality_impact": missing_modality_impact,
        "behaviour": behaviour_result.behaviour,
        "behaviour_probability": behaviour_result.behaviour_probability,
        "behaviour_confidence": behaviour_result.behaviour_confidence,
        "behaviour_duration": behaviour_result.behaviour_duration,
        "behaviour_history": behaviour_result.behaviour_history,
        "behaviour_transition": behaviour_result.behaviour_transition,
        "behaviour_stability": behaviour_result.behaviour_stability,
        "behaviour_timeline": behaviour_result.behaviour_timeline,
        "behaviour_distribution": behaviour_result.distribution,
        "explainability": explainability,
        "video_result": video_result,
        "audio_result": audio_result,
        "sensor_result": sensor_result,
    }
