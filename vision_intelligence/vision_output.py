from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _bbox_from_track(tr) -> Optional[List[float]]:
    bbox = getattr(tr, "bbox_xyxy", None)
    if bbox is None:
        return None
    return [float(v) for v in bbox]


def extend_vision_output(
    *,
    legacy_video_result: Dict[str, Any],
    tracks: Dict[int, any],
    occupancy: int,
    counting: Optional[Dict[str, Any]] = None,
    visibility_scores: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """Extend existing vision output contract while preserving existing fields.

    Adds per-track fields and aggregate occupancy stats.
    """

    video_result = dict(legacy_video_result)

    # Per-track list for UI and dashboard
    track_items: List[Dict[str, Any]] = []
    for tid, tr in tracks.items():
        bbox = _bbox_from_track(tr)
        traj = getattr(tr, "trajectory", []) or []
        velocity = getattr(tr, "velocity_xy", (0.0, 0.0))
        acceleration = getattr(tr, "acceleration_xy", (0.0, 0.0))
        direction = getattr(tr, "direction_deg", None)
        direction_label = getattr(tr, "direction_label", "stationary")

        tracking_conf = getattr(tr, "tracking_confidence", None)
        detection_conf = getattr(tr, "detection_confidence", None)

        entry_time = getattr(tr, "entry_time", None)
        exit_time = getattr(tr, "exit_time", None)
        dwell_time = getattr(tr, "dwell_time", 0.0)
        track_lifetime = getattr(tr, "track_lifetime_sec", dwell_time)
        distance_travelled = float(getattr(tr, "distance_travelled", 0.0) or 0.0)
        stationary_duration = float(getattr(tr, "stationary_duration", 0.0) or 0.0)
        movement_density = float(getattr(tr, "movement_density", 0.0) or 0.0)

        visibility_score = (
            (visibility_scores or {}).get(tid)
            if visibility_scores is not None
            else float(getattr(tr, "visibility_score", 0.0) or 0.0)
        )

        track_items.append(
            {
                "track_id": int(tid),
                "species": getattr(tr, "class_name", "animal"),
                "detector_label": getattr(tr, "detector_label", getattr(tr, "class_name", "animal")),
                "classifier_label": getattr(tr, "classifier_label", getattr(tr, "class_name", "animal")),
                "final_species": getattr(tr, "final_species", getattr(tr, "class_name", "animal")),
                "species_group": getattr(tr, "species_group", "unknown"),
                "species_confidence": float(getattr(tr, "species_confidence", getattr(tr, "detection_confidence", 0.0)) or 0.0),
                "classifier_confidence": float(getattr(tr, "classifier_confidence", 0.0) or 0.0),
                "classification_confidence": float(getattr(tr, "classification_confidence", 0.0) or 0.0),
                "temporal_confidence": float(getattr(tr, "temporal_confidence", getattr(tr, "classification_confidence", 0.0)) or 0.0),
                "agreement_score": float(getattr(tr, "agreement_score", 0.0) or 0.0),
                "embedding_distance": float(getattr(tr, "embedding_distance", 1.0) or 1.0),
                "top5_predictions": getattr(tr, "top5_predictions", getattr(tr, "top_5_predictions", [])),
                "top_5_predictions": getattr(tr, "top_5_predictions", getattr(tr, "top5_predictions", [])),
                "classification_history": getattr(tr, "classification_history", []),
                "species_embedding": getattr(tr, "species_embedding", []),
                "bbox": bbox,
                "confidence": float(detection_conf) if detection_conf is not None else None,
                "tracking_confidence": float(tracking_conf) if tracking_conf is not None else None,
                "trajectory": traj,
                "velocity": [float(velocity[0]), float(velocity[1])],
                "acceleration": [float(acceleration[0]), float(acceleration[1])],
                "direction": direction,
                "direction_label": direction_label,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "dwell_time": float(dwell_time) if dwell_time is not None else 0.0,
                "track_lifetime": float(track_lifetime) if track_lifetime is not None else 0.0,
                "distance_travelled": distance_travelled,
                "stationary_duration": stationary_duration,
                "movement_density": movement_density,
                "occupancy": int(occupancy),
                "visibility_score": float(visibility_score),
                "age_frames": int(getattr(tr, "age_frames", 0) or 0),
                "occlusion_state": "visible" if getattr(tr, "lost_frames", 0) == 0 else "occluded",
                "occlusion_status": bool(getattr(tr, "lost_frames", 0) > 0),
                "recovered_status": bool(getattr(tr, "was_occluded", False) and getattr(tr, "lost_frames", 0) == 0),
                "lost_reacquired_state": "reacquired" if getattr(tr, "reacquired", False) else "lost" if getattr(tr, "lost_frames", 0) > 0 else "active",
            }
        )

    # Add aggregate keys for fusion compatibility
    video_result.setdefault("tracks", [])
    video_result["tracks"] = track_items

    # Also extend detections list for UI overlay, but keep original detections fields
    # legacy_video_result['detections'] is a list already; augment matching entries by track_id.
    dets = video_result.get("detections", []) or []
    for det in dets:
        tid = det.get("track_id") or det.get("id")
        if tid is None:
            continue
        tid = int(tid)
        tr = tracks.get(tid)
        if tr is None:
            continue
        det["species"] = getattr(tr, "class_name", det.get("class"))
        det["detector_label"] = getattr(tr, "detector_label", det.get("class"))
        det["classifier_label"] = getattr(tr, "classifier_label", det.get("class"))
        det["final_species"] = getattr(tr, "final_species", getattr(tr, "class_name", det.get("class")))
        det["species_group"] = getattr(tr, "species_group", "unknown")
        det["species_confidence"] = float(getattr(tr, "species_confidence", det.get("confidence", 0.0)) or 0.0)
        det["classifier_confidence"] = float(getattr(tr, "classifier_confidence", 0.0) or 0.0)
        det["temporal_confidence"] = float(getattr(tr, "temporal_confidence", getattr(tr, "classification_confidence", 0.0)) or 0.0)
        det["agreement_score"] = float(getattr(tr, "agreement_score", 0.0) or 0.0)
        det["classification_history"] = getattr(tr, "classification_history", [])
        det["top5_predictions"] = getattr(tr, "top5_predictions", getattr(tr, "top_5_predictions", []))
        det["top_5_predictions"] = getattr(tr, "top_5_predictions", getattr(tr, "top5_predictions", []))
        det["embedding_distance"] = float(getattr(tr, "embedding_distance", 1.0) or 1.0)
        det["species_embedding"] = getattr(tr, "species_embedding", [])
        det["bbox"] = _bbox_from_track(tr)
        det["tracking_confidence"] = getattr(tr, "tracking_confidence", det.get("confidence"))
        det["trajectory"] = getattr(tr, "trajectory", [])
        det["velocity"] = [float(getattr(tr, "velocity_xy", (0.0, 0.0))[0]), float(getattr(tr, "velocity_xy", (0.0, 0.0))[1])]
        det["acceleration"] = [float(getattr(tr, "acceleration_xy", (0.0, 0.0))[0]), float(getattr(tr, "acceleration_xy", (0.0, 0.0))[1])]
        det["direction"] = getattr(tr, "direction_deg", None)
        det["direction_label"] = getattr(tr, "direction_label", "stationary")
        det["entry_time"] = getattr(tr, "entry_time", None)
        det["exit_time"] = getattr(tr, "exit_time", None)
        det["dwell_time"] = float(getattr(tr, "dwell_time", 0.0) or 0.0)
        det["track_lifetime"] = float(getattr(tr, "track_lifetime_sec", getattr(tr, "dwell_time", 0.0)) or 0.0)
        det["distance_travelled"] = float(getattr(tr, "distance_travelled", 0.0) or 0.0)
        det["stationary_duration"] = float(getattr(tr, "stationary_duration", 0.0) or 0.0)
        det["movement_density"] = float(getattr(tr, "movement_density", 0.0) or 0.0)
        det["occupancy"] = int(occupancy)
        det["visibility_score"] = float(getattr(tr, "visibility_score", 0.0) or 0.0)

    if counting:
        video_result["occupancy"] = int(counting.get("current_occupancy", occupancy))
        video_result["total_unique_individuals"] = int(counting.get("total_unique_individuals", 0))
        video_result["species_wise_count"] = counting.get("species_wise_count", {})
        video_result["region_wise_count"] = counting.get("region_wise_count", {})
        video_result["entry_count"] = int(counting.get("entry_count", 0))
        video_result["exit_count"] = int(counting.get("exit_count", 0))
        video_result["maximum_occupancy"] = int(counting.get("maximum_occupancy", occupancy))
        video_result["average_occupancy"] = float(counting.get("average_occupancy", float(occupancy)))
        video_result["average_dwell_time"] = float(counting.get("average_dwell_time", 0.0) or 0.0)
        video_result["longest_dwell_time"] = float(counting.get("longest_dwell_time", 0.0) or 0.0)
        video_result["region_dwell_time"] = counting.get("region_dwell_time", {})
        video_result["region_movement_stats"] = counting.get("region_movement_stats", {})

    return video_result

