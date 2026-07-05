from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from .types import FeatureStoreEntry, FeatureVector, ValidationReport


class FeatureStore:
    """In-memory feature store for latest features, history, and metadata."""

    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.latest: Dict[str, FeatureStoreEntry] = {}
        self.history: Dict[str, Deque[FeatureStoreEntry]] = {}
        self.metadata: Dict[str, Dict[str, object]] = {}

    def add(self, feature: FeatureVector, validation: ValidationReport) -> None:
        entry = FeatureStoreEntry(timestamp=feature.timestamp, modality=feature.modality, vector=feature, validation=validation)
        self.latest[feature.modality] = entry
        self.history.setdefault(feature.modality, deque(maxlen=self.max_history)).append(entry)
        self.metadata[feature.modality] = {
            "version": feature.version,
            "confidence": feature.confidence,
            "quality_score": feature.quality_score,
            "health_status": feature.health_status,
        }

    def get_latest(self, modality: str) -> Optional[FeatureStoreEntry]:
        return self.latest.get(modality)

    def get_history(self, modality: str) -> List[FeatureStoreEntry]:
        return list(self.history.get(modality, deque()))
