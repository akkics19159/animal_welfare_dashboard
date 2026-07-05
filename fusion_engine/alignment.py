from __future__ import annotations

from typing import Dict, Optional

from .types import FeatureVector, FusionInput


class FeatureAligner:
    """Aligns modalities to a common timestamp using configurable tolerance."""

    def __init__(self, sync_tolerance_seconds: float = 0.5):
        self.sync_tolerance_seconds = sync_tolerance_seconds

    def align(self, fusion_input: FusionInput) -> FusionInput:
        aligned = FusionInput(timestamp=fusion_input.timestamp, metadata=fusion_input.metadata.copy())
        aligned.vision = self._align_single(fusion_input.vision, fusion_input.timestamp)
        aligned.audio = self._align_single(fusion_input.audio, fusion_input.timestamp)
        aligned.sensors = self._align_single(fusion_input.sensors, fusion_input.timestamp)
        return aligned

    def _align_single(self, feature: Optional[FeatureVector], reference_timestamp: Optional[float]) -> Optional[FeatureVector]:
        if feature is None or reference_timestamp is None:
            return feature
        if abs(feature.timestamp - reference_timestamp) <= self.sync_tolerance_seconds:
            return feature
        return FeatureVector(
            modality=feature.modality,
            timestamp=reference_timestamp,
            values=feature.values,
            confidence=feature.confidence,
            quality_score=feature.quality_score,
            latency_ms=feature.latency_ms,
            health_status=feature.health_status,
            version=feature.version,
            metadata=feature.metadata,
        )
