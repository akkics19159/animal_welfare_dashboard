from __future__ import annotations

from typing import Dict, List, Optional

from .types import FeatureVector


class TemporalContextBuilder:
    """Builds rolling temporal context for the fusion engine."""

    def __init__(self, window_seconds: float = 30.0):
        self.window_seconds = window_seconds

    def build(self, history: List[FeatureVector], reference_timestamp: Optional[float] = None) -> Dict[str, object]:
        if not history:
            return {"window_seconds": self.window_seconds, "samples": 0, "average_confidence": 0.0}
        if reference_timestamp is None:
            reference_timestamp = history[-1].timestamp
        relevant = [item for item in history if reference_timestamp - item.timestamp <= self.window_seconds]
        if not relevant:
            relevant = history[-1:]
        average_confidence = sum(item.confidence for item in relevant) / len(relevant)
        return {
            "window_seconds": self.window_seconds,
            "samples": len(relevant),
            "average_confidence": average_confidence,
            "timestamps": [item.timestamp for item in relevant],
            "modalities": [item.modality for item in relevant],
        }
