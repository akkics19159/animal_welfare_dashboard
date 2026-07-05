from __future__ import annotations

from typing import Any, Dict, List

from .knowledge import KnowledgeBase
from .types import ReasoningInput


class ContextEngine:
    """Evaluates contextual pressures that influence welfare reasoning."""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

    def evaluate(self, input_data: ReasoningInput) -> Dict[str, Any]:
        context = input_data.context or {}
        factors: List[str] = []
        score = 0.0

        time_of_day = str(context.get("time_of_day", "day")).lower()
        if time_of_day in {"night", "dawn", "dusk"}:
            score += 0.08
            factors.append("night_period")

        crowding = str(context.get("crowding", "low")).lower()
        if crowding in {"high", "dense", "packed"}:
            score += 0.12
            factors.append("high_crowding")

        environment = str(context.get("environment", "outdoor")).lower()
        if environment == "indoor":
            score += 0.05
            factors.append("indoor_environment")

        occupancy = context.get("occupancy", 1)
        if occupancy is not None and occupancy <= 0:
            score += 0.1
            factors.append("no_detected_animals")

        sensor_quality = input_data.sensor_reliability
        if sensor_quality < 0.4:
            score -= 0.08
            factors.append("poor_sensor_quality")

        audio_quality = input_data.audio_quality
        if audio_quality < 0.4:
            score -= 0.06
            factors.append("poor_audio_quality")

        history = input_data.history or []
        if history:
            score += 0.03
            factors.append("historical_context")

        return {
            "context_score": max(0.0, min(0.35, score)),
            "factors": factors,
            "context": context,
        }
