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
        # Timeline proxy uses mean value magnitude per sample to support temporal welfare analysis.
        timeline = []
        for item in relevant:
            vals = [float(v) for v in item.values.values()] if item.values else [0.0]
            timeline.append(
                {
                    "timestamp": item.timestamp,
                    "modality": item.modality,
                    "score": float(sum(vals) / max(1, len(vals))),
                    "confidence": float(item.confidence),
                }
            )

        trend = 0.0
        if len(timeline) >= 2:
            trend = float(timeline[-1]["score"] - timeline[0]["score"])

        return {
            "window_seconds": self.window_seconds,
            "samples": len(relevant),
            "average_confidence": average_confidence,
            "timestamps": [item.timestamp for item in relevant],
            "modalities": [item.modality for item in relevant],
            "welfare_timeline": timeline,
            "trend": trend,
        }
