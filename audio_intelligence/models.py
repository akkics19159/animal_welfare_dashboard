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
        if energy < 1e-3:
            label = "silence"
            confidence = 0.98
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


class TemporalAudioAnalyzer:
    """Analyzes temporal distress patterns over a rolling context."""

    def analyze(self, distress_probabilities: List[float]) -> Dict[str, Any]:
        if not distress_probabilities:
            return {"pattern": "continuous_normal_behaviour", "persistent_distress": False, "repeated_distress": False, "escalating_distress": False, "intermittent_distress": False}

        if all(prob >= 0.7 for prob in distress_probabilities):
            pattern = "persistent_distress"
        elif distress_probabilities[-1] >= 0.7 and distress_probabilities[-1] > distress_probabilities[0]:
            pattern = "escalating_distress"
        elif sum(1 for prob in distress_probabilities if prob >= 0.7) >= 2:
            pattern = "repeated_distress"
        elif any(prob >= 0.7 for prob in distress_probabilities) and any(prob < 0.3 for prob in distress_probabilities):
            pattern = "intermittent_distress"
        else:
            pattern = "continuous_normal_behaviour"

        return {
            "pattern": pattern,
            "persistent_distress": pattern == "persistent_distress",
            "repeated_distress": pattern == "repeated_distress",
            "escalating_distress": pattern == "escalating_distress",
            "intermittent_distress": pattern == "intermittent_distress",
        }
