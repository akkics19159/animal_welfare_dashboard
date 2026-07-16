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
        elif occupancy is not None and occupancy >= 8:
            score += 0.12
            factors.append("extreme_overcrowding")
        elif occupancy is not None and occupancy == 1:
            score += 0.06
            factors.append("continuous_isolation")

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

        agreement = float(max(0.0, min(1.0, input_data.agreement_score)))
        if agreement >= 0.75:
            score += 0.06
            factors.append("high_multimodal_agreement")
        elif agreement < 0.35:
            score -= 0.05
            factors.append("low_multimodal_agreement")

        temporal_state = self._temporal_state(input_data)
        if temporal_state["state"] in {"escalating", "persistent_distress"}:
            score += 0.08
            factors.append("temporal_escalation")
        elif temporal_state["state"] == "recovering":
            score -= 0.04
            factors.append("temporal_recovery")

        return {
            "context_score": max(0.0, min(0.35, score)),
            "factors": factors,
            "context": context,
            "temporal_welfare_state": temporal_state,
        }

    def _temporal_state(self, input_data: ReasoningInput) -> Dict[str, Any]:
        history = input_data.history or []
        if len(history) < 2:
            return {"state": "insufficient_history", "trend": 0.0, "event_frequency": 0.0}

        probs = [float(h.get("probability", 0.0) or 0.0) for h in history if isinstance(h, dict)]
        if len(probs) < 2:
            return {"state": "insufficient_history", "trend": 0.0, "event_frequency": 0.0}

        trend = probs[-1] - probs[0]
        high_events = sum(1 for p in probs if p >= 0.65)
        event_frequency = float(high_events / max(1, len(probs)))

        state = "stable"
        if trend >= 0.2 and event_frequency >= 0.4:
            state = "escalating"
        elif event_frequency >= 0.7:
            state = "persistent_distress"
        elif trend <= -0.15:
            state = "recovering"

        return {
            "state": state,
            "trend": float(trend),
            "event_frequency": event_frequency,
            "history_samples": len(probs),
        }
