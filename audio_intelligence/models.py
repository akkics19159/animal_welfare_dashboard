from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .config import AudioProcessingConfig
from .features import AudioFeatureExtractor
from .interfaces import BaseDistressClassifier, BaseSoundClassifier, BaseSpeciesClassifier


logger = logging.getLogger(__name__)


class RuleBasedSoundClassifier(BaseSoundClassifier):
    """Rule-based fallback sound classifier for environments without YAMNet."""

    def __init__(self, config: Optional[AudioProcessingConfig] = None):
        super().__init__()
        self.config = config or AudioProcessingConfig()
        self.features = AudioFeatureExtractor()

    def classify(self, signal: Any, sample_rate: int, **kwargs) -> List[Dict[str, Any]]:
        if signal is None:
            return [{"label": "silence", "confidence": 1.0, "timestamp": 0.0}]
        signal = np.asarray(signal, dtype=np.float32)
        features = self.features.extract(signal, sample_rate)
        energy = features.rms_energy
        pitch = features.pitch
        tempo = features.tempo
        if energy < 1e-3:
            label = "silence"
            confidence = 0.98
        elif tempo > 145 and energy < 0.03:
            label = "play"
            confidence = 0.70
        elif 120 <= tempo <= 170 and 160 <= pitch <= 280 and energy < 0.05:
            label = "mating"
            confidence = 0.66
        elif energy > 0.06 and pitch < 160:
            label = "feeding"
            confidence = 0.62
        elif 180 <= pitch <= 320 and energy < 0.04:
            label = "social_communication"
            confidence = 0.65
        elif pitch > 250:
            label = "barking"
            confidence = 0.72
        elif energy > 0.08:
            label = "animal_vocalization"
            confidence = 0.68
        else:
            label = "background_noise"
            confidence = 0.55
        return [{"label": label, "confidence": float(confidence), "timestamp": 0.0}]


class RuleBasedSpeciesClassifier(BaseSpeciesClassifier):
    """Species-aware classifier that infers likely species from sound class cues."""

    def classify(self, signal: Any, sample_rate: int, sound_predictions: Optional[List[Dict[str, Any]]] = None, **kwargs) -> List[Dict[str, Any]]:
        labels = [item.get("label", "") for item in (sound_predictions or [])]
        label_text = " ".join(labels).lower()
        if "barking" in label_text or "dog" in label_text:
            return [{"label": "dog", "confidence": 0.8}]
        if "meow" in label_text or "cat" in label_text:
            return [{"label": "cat", "confidence": 0.8}]
        if "cow" in label_text:
            return [{"label": "cow", "confidence": 0.8}]
        if "horse" in label_text:
            return [{"label": "horse", "confidence": 0.8}]
        if "speech" in label_text:
            return [{"label": "human", "confidence": 0.7}]
        if "bird" in label_text:
            return [{"label": "bird", "confidence": 0.7}]
        return [{"label": "unknown", "confidence": 0.4}]


class RuleBasedDistressClassifier(BaseDistressClassifier):
    """Rule-based distress model that provides interpretable probabilities for downstream fusion."""

    def __init__(self, config: Optional[AudioProcessingConfig] = None):
        super().__init__()
        self.config = config or AudioProcessingConfig()
        self.features = AudioFeatureExtractor()

    def classify(self, signal: Any, sample_rate: int, sound_predictions: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Dict[str, Any]:
        signal = np.asarray(signal, dtype=np.float32)
        features = self.features.extract(signal, sample_rate)
        energy = features.rms_energy
        pitch = features.pitch
        tempo = features.tempo
        base = 0.15
        if energy > 0.05:
            base += 0.2
        if pitch > 250:
            base += 0.2
        if tempo > 120:
            base += 0.1
        distress = min(0.98, max(0.0, base))
        pain = min(0.98, max(0.0, distress * 0.8))
        fear = min(0.98, max(0.0, distress * 0.75))
        panic = min(0.98, max(0.0, distress * 0.6))
        aggression = min(0.98, max(0.0, distress * 0.5))
        confidence = min(0.98, 0.55 + min(0.35, distress))
        return {
            "distress_probability": float(distress),
            "pain_probability": float(pain),
            "fear_probability": float(fear),
            "panic_probability": float(panic),
            "aggression_probability": float(aggression),
            "confidence": float(confidence),
            "audio_quality": float(min(1.0, max(0.0, energy / 0.08))),
            "event_duration": float(features.duration),
            "event_type": "distress_vocalization" if distress >= 0.55 else "undetermined",
        }


class VocalizationFilter:
    """Rejects vocalizations that should not influence welfare risk."""

    def __init__(self, config: Optional[AudioProcessingConfig] = None):
        self.config = config or AudioProcessingConfig()

    def filter(self, sound_predictions: List[Dict[str, Any]], species_predictions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        labels = [str(pred.get("label", "")).lower() for pred in sound_predictions]
        reasons = []
        for label in labels:
            if label in self.config.rejected_sound_labels or label in self.config.non_distress_labels:
                reasons.append(f"Rejected non-distress sound: {label}")
        accepted = len(reasons) == 0 and len(labels) > 0
        if not labels:
            reasons.append("No vocalization detected")
            accepted = False
        return {
            "accepted": bool(accepted),
            "reasons": reasons,
            "filtered_labels": labels,
            "species": [pred.get("label") for pred in (species_predictions or [])],
        }


class NonDistressFilter:
    """Reduces false positives by suppressing normal behavioural sounds."""

    def __init__(self, config: Optional[AudioProcessingConfig] = None):
        self.config = config or AudioProcessingConfig()

    def evaluate(self, sound_predictions: List[Dict[str, Any]], distress_probability: float) -> Dict[str, Any]:
        if not sound_predictions:
            return {
                "non_distress_probability": 0.0,
                "suppressed": False,
                "suppressed_reason": None,
                "event_type": "unknown_audio",
                "label": "unknown",
            }

        label = str(sound_predictions[0].get("label", "unknown")).lower()
        conf = float(sound_predictions[0].get("confidence", 0.0) or 0.0)

        normal_map = {
            "mating": "mating_vocalization",
            "courtship": "courtship_behaviour",
            "play": "play_vocalization",
            "feeding": "feeding_sound",
            "grooming": "grooming_sound",
            "parent_offspring": "parent_offspring_communication",
            "social_communication": "social_communication",
            "exploration": "exploration",
            "curiosity": "curiosity",
            "speech": "normal_communication",
            "environmental": "normal_environmental_sound",
            "background_noise": "normal_environmental_sound",
            "music": "normal_environmental_sound",
            "wind": "normal_environmental_sound",
            "rain": "normal_environmental_sound",
        }

        event_type = normal_map.get(label, "distress_vocalization" if distress_probability >= 0.55 else "unknown_audio")
        non_distress_probability = conf if label in normal_map else max(0.0, 1.0 - distress_probability)

        suppressed = False
        reason = None
        if label in normal_map and non_distress_probability >= self.config.non_distress_suppression_threshold:
            if distress_probability < self.config.strong_distress_override_threshold:
                suppressed = True
                reason = f"Suppressed likely non-distress event: {event_type}"

        return {
            "non_distress_probability": float(min(1.0, max(0.0, non_distress_probability))),
            "suppressed": bool(suppressed),
            "suppressed_reason": reason,
            "event_type": event_type,
            "label": label,
        }


class TemporalAudioAnalyzer:
    """Analyzes temporal distress patterns over a rolling context."""

    def __init__(self, config: Optional[AudioProcessingConfig] = None):
        self.config = config or AudioProcessingConfig()

    def analyze(self, distress_probabilities: List[float], segment_durations: Optional[List[float]] = None) -> Dict[str, Any]:
        if not distress_probabilities:
            return {
                "pattern": "continuous_normal_behaviour",
                "persistent_distress": False,
                "repeated_distress": False,
                "escalating_distress": False,
                "intermittent_distress": False,
                "duration": 0.0,
                "repetition": 0,
                "rhythm": 0.0,
                "escalation": 0.0,
                "persistence": 0.0,
                "event_frequency": 0.0,
                "temporal_consistency": 0.0,
            }

        threshold = self.config.temporal_distress_threshold
        if all(prob >= threshold for prob in distress_probabilities):
            pattern = "persistent_distress"
        elif distress_probabilities[-1] >= threshold and (distress_probabilities[-1] - distress_probabilities[0]) >= self.config.temporal_escalation_delta:
            pattern = "escalating_distress"
        elif sum(1 for prob in distress_probabilities if prob >= threshold) >= self.config.temporal_min_events_for_repetition:
            pattern = "repeated_distress"
        elif any(prob >= threshold for prob in distress_probabilities) and any(prob < 0.3 for prob in distress_probabilities):
            pattern = "intermittent_distress"
        else:
            pattern = "continuous_normal_behaviour"

        durations = segment_durations or []
        total_duration = float(sum(durations)) if durations else float(len(distress_probabilities))
        repetition = int(sum(1 for prob in distress_probabilities if prob >= threshold))
        diffs = np.diff(np.asarray(distress_probabilities, dtype=np.float32)) if len(distress_probabilities) >= 2 else np.asarray([], dtype=np.float32)
        escalation = float(max(0.0, np.mean(np.maximum(diffs, 0.0)))) if diffs.size else 0.0
        rhythm = float(np.std(diffs)) if diffs.size else 0.0
        persistence = float(np.mean(distress_probabilities))
        event_frequency = float(repetition / max(1e-6, total_duration))
        temporal_consistency = float(max(0.0, min(1.0, 1.0 - min(1.0, rhythm))))

        return {
            "pattern": pattern,
            "persistent_distress": pattern == "persistent_distress",
            "repeated_distress": pattern == "repeated_distress",
            "escalating_distress": pattern == "escalating_distress",
            "intermittent_distress": pattern == "intermittent_distress",
            "duration": total_duration,
            "repetition": repetition,
            "rhythm": rhythm,
            "escalation": escalation,
            "persistence": persistence,
            "event_frequency": event_frequency,
            "temporal_consistency": temporal_consistency,
        }
