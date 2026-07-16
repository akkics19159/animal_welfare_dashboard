from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .interfaces import BehaviourModelMetadata, BehaviourObservation, BehaviourRecognizerPlugin


NORMAL_BEHAVIOURS = {
    "walking",
    "standing",
    "running",
    "sitting",
    "resting",
    "sleeping",
    "eating",
    "drinking",
    "feeding",
    "grooming",
    "playing",
    "exploring",
    "social_interaction",
    "parent_offspring_interaction",
    "courtship",
    "mating",
    "curiosity",
    "environmental_observation",
}

ABNORMAL_BEHAVIOURS = {
    "repeated_circling",
    "escape_attempt",
    "panic",
    "aggression",
    "fighting",
    "continuous_vocal_distress",
    "collapse",
    "limping",
    "isolation",
    "overcrowding",
    "long_inactivity",
    "abnormal_pacing",
    "self_injury",
}


class TemporalBehaviourRecognizerPlugin(BehaviourRecognizerPlugin):
    """Placeholder plugin for Video Swin / TimeSformer / SlowFast / X3D backends.

    This plugin is intentionally optional. If dependencies are unavailable,
    the engine falls back to rule-based methods.
    """

    def __init__(self, model_name: str, priority: int):
        super().__init__(BehaviourModelMetadata(name=model_name, priority=priority))

    def available(self) -> bool:
        return False

    def predict(self, track: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {}


class RuleBasedBehaviourRecognizerPlugin(BehaviourRecognizerPlugin):
    def __init__(self):
        super().__init__(BehaviourModelMetadata(name="pose_trajectory_rule_engine", priority=5))

    def available(self) -> bool:
        return True

    def predict(self, track: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        speed = float(track.get("speed", 0.0) or 0.0)
        acceleration = float(track.get("acceleration_mag", 0.0) or 0.0)
        occupancy = int(context.get("occupancy", 0) or 0)
        dwell = float(track.get("dwell_time", 0.0) or 0.0)
        direction_label = str(track.get("direction_label", "stationary") or "stationary")
        audio_event = str(context.get("audio_event", "") or "").lower()

        behaviour = "unknown_behaviour"
        prob = 0.45

        if speed >= 120:
            behaviour, prob = "running", 0.82
        elif speed >= 35:
            behaviour, prob = "walking", 0.78
        elif speed < 3 and dwell >= 20:
            behaviour, prob = "long_inactivity", 0.8
        elif speed < 3 and dwell >= 8:
            behaviour, prob = "resting", 0.74
        elif acceleration >= 180:
            behaviour, prob = "panic", 0.79
        elif direction_label in {"left", "right"} and acceleration >= 90 and speed >= 80:
            behaviour, prob = "escape_attempt", 0.77

        if occupancy >= 8:
            behaviour, prob = "overcrowding", max(prob, 0.8)
        if occupancy == 1 and speed < 5:
            behaviour, prob = "isolation", max(prob, 0.75)
        if audio_event in {"distress_vocalization", "panic"} and speed >= 60:
            behaviour, prob = "continuous_vocal_distress", max(prob, 0.83)
        if audio_event in {"play", "social_communication"} and speed >= 20:
            behaviour, prob = "playing", max(prob, 0.72)

        return {
            "behaviour": behaviour,
            "behaviour_probability": float(max(0.0, min(1.0, prob))),
            "behaviour_confidence": float(max(0.0, min(1.0, prob * 0.95 + 0.03))),
        }


class TrajectoryOnlyFallbackPlugin(BehaviourRecognizerPlugin):
    def __init__(self):
        super().__init__(BehaviourModelMetadata(name="trajectory_only_fallback", priority=6))

    def available(self) -> bool:
        return True

    def predict(self, track: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        speed = float(track.get("speed", 0.0) or 0.0)
        if speed >= 80:
            b = "running"
            p = 0.7
        elif speed >= 20:
            b = "walking"
            p = 0.68
        else:
            b = "standing"
            p = 0.65
        return {
            "behaviour": b,
            "behaviour_probability": float(p),
            "behaviour_confidence": float(p * 0.9),
        }


@dataclass
class BehaviourEngineResult:
    behaviour: str = "unknown_behaviour"
    behaviour_probability: float = 0.0
    behaviour_confidence: float = 0.0
    behaviour_duration: float = 0.0
    behaviour_history: List[Dict[str, Any]] = field(default_factory=list)
    behaviour_transition: str = "steady"
    behaviour_stability: float = 0.0
    behaviour_timeline: List[Dict[str, Any]] = field(default_factory=list)
    distribution: Dict[str, int] = field(default_factory=dict)


class BehaviourRecognitionEngine:
    def __init__(self):
        self.plugins: List[BehaviourRecognizerPlugin] = [
            TemporalBehaviourRecognizerPlugin("video_swin_transformer", 1),
            TemporalBehaviourRecognizerPlugin("timesformer", 2),
            TemporalBehaviourRecognizerPlugin("slowfast", 3),
            TemporalBehaviourRecognizerPlugin("x3d", 4),
            RuleBasedBehaviourRecognizerPlugin(),
            TrajectoryOnlyFallbackPlugin(),
        ]
        self._history_by_track: Dict[int, List[Dict[str, Any]]] = {}

    def analyze(self, *, video_result: Dict[str, Any], audio_result: Dict[str, Any], sensor_result: Dict[str, Any]) -> BehaviourEngineResult:
        tracks = list(video_result.get("tracks", []) or [])
        occupancy = int(video_result.get("occupancy", len(tracks)) or len(tracks))
        audio_event = str(audio_result.get("event_type", "unknown_audio") or "unknown_audio")

        if not tracks:
            return BehaviourEngineResult(
                behaviour="unclassified",
                behaviour_probability=0.0,
                behaviour_confidence=0.0,
                behaviour_duration=0.0,
                behaviour_history=[],
                behaviour_transition="steady",
                behaviour_stability=0.0,
                behaviour_timeline=[],
                distribution={},
            )

        now_items: List[Dict[str, Any]] = []
        distribution: Dict[str, int] = {}
        for tr in tracks:
            tid = int(tr.get("track_id", -1))
            vel = tr.get("velocity") or [0.0, 0.0]
            acc = tr.get("acceleration") or [0.0, 0.0]
            speed = float((float(vel[0]) ** 2 + float(vel[1]) ** 2) ** 0.5)
            acc_mag = float((float(acc[0]) ** 2 + float(acc[1]) ** 2) ** 0.5)

            track_ctx = {
                "speed": speed,
                "acceleration_mag": acc_mag,
                "dwell_time": float(tr.get("dwell_time", 0.0) or 0.0),
                "direction_label": tr.get("direction_label", "stationary"),
            }
            global_ctx = {
                "occupancy": occupancy,
                "audio_event": audio_event,
                "sensor": sensor_result,
            }

            pred = self._predict_with_fallback(track_ctx, global_ctx)
            behaviour = str(pred.get("behaviour", "unknown_behaviour"))
            prob = float(pred.get("behaviour_probability", 0.0) or 0.0)
            conf = float(pred.get("behaviour_confidence", 0.0) or 0.0)

            hist = self._history_by_track.setdefault(tid, [])
            prev_b = hist[-1]["behaviour"] if hist else behaviour
            transition = f"{prev_b}->{behaviour}" if prev_b != behaviour else "steady"
            hist.append(
                {
                    "track_id": tid,
                    "behaviour": behaviour,
                    "probability": prob,
                    "confidence": conf,
                    "transition": transition,
                }
            )
            if len(hist) > 120:
                del hist[:-120]

            recent_labels = [h["behaviour"] for h in hist[-20:]]
            same_count = sum(1 for b in recent_labels if b == behaviour)
            stability = float(same_count / max(1, len(recent_labels)))
            duration = float(len([b for b in reversed(recent_labels) if b == behaviour]))

            tr["behaviour"] = behaviour
            tr["behaviour_probability"] = prob
            tr["behaviour_confidence"] = conf
            tr["behaviour_duration"] = duration
            tr["behaviour_history"] = recent_labels
            tr["behaviour_transition"] = transition
            tr["behaviour_stability"] = stability

            now_items.append(
                {
                    "track_id": tid,
                    "species": tr.get("final_species") or tr.get("species", "unknown"),
                    "behaviour": behaviour,
                    "behaviour_probability": prob,
                    "behaviour_confidence": conf,
                    "behaviour_duration": duration,
                    "behaviour_history": recent_labels,
                    "behaviour_transition": transition,
                    "behaviour_stability": stability,
                }
            )
            distribution[behaviour] = distribution.get(behaviour, 0) + 1

        primary = max(now_items, key=lambda x: x["behaviour_probability"]) if now_items else {
            "behaviour": "unclassified",
            "behaviour_probability": 0.0,
            "behaviour_confidence": 0.0,
            "behaviour_duration": 0.0,
            "behaviour_transition": "steady",
            "behaviour_stability": 0.0,
        }

        timeline = []
        for tid, hist in self._history_by_track.items():
            for idx, h in enumerate(hist[-20:]):
                timeline.append(
                    {
                        "track_id": tid,
                        "step": idx,
                        "behaviour": h["behaviour"],
                        "probability": h["probability"],
                        "confidence": h["confidence"],
                    }
                )

        return BehaviourEngineResult(
            behaviour=primary["behaviour"],
            behaviour_probability=float(primary["behaviour_probability"]),
            behaviour_confidence=float(primary["behaviour_confidence"]),
            behaviour_duration=float(primary["behaviour_duration"]),
            behaviour_history=now_items,
            behaviour_transition=str(primary["behaviour_transition"]),
            behaviour_stability=float(primary["behaviour_stability"]),
            behaviour_timeline=timeline,
            distribution=distribution,
        )

    def _predict_with_fallback(self, track_ctx: Dict[str, Any], global_ctx: Dict[str, Any]) -> Dict[str, Any]:
        for plugin in sorted(self.plugins, key=lambda p: p.metadata.priority):
            if not plugin.available():
                continue
            pred = plugin.predict(track_ctx, global_ctx)
            if pred and pred.get("behaviour"):
                return pred
        return {
            "behaviour": "unclassified",
            "behaviour_probability": 0.0,
            "behaviour_confidence": 0.0,
        }
