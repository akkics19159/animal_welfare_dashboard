from __future__ import annotations

from typing import Any, Dict, List

from .knowledge import KnowledgeBase
from .types import EvidenceItem, ReasoningInput, ReasoningResult


class ConfidenceAwareInferenceEngine:
    """Combines evidence, confidence and context into a welfare assessment."""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

    def infer(self, input_data: ReasoningInput, rule_result: Dict[str, Any], context_result: Dict[str, Any], contradictions: List[str]) -> ReasoningResult:
        support_score = float(rule_result.get("support_score", 0.0))
        context_score = float(context_result.get("context_score", 0.0))

        quality_factor = (
            0.25 * max(0.0, min(1.0, input_data.fusion_confidence))
            + 0.25 * max(0.0, min(1.0, input_data.sensor_reliability))
            + 0.25 * max(0.0, min(1.0, input_data.audio_quality))
            + 0.25 * max(0.0, min(1.0, input_data.vision_quality))
        )
        uncertainty_penalty = max(0.0, min(1.0, input_data.prediction_uncertainty))
        confidence = max(0.05, min(0.99, quality_factor * (1.0 - uncertainty_penalty)))

        if contradictions:
            confidence *= 0.9

        base_score = min(1.0, support_score + context_score)
        final_score = min(1.0, max(0.0, base_score * confidence + 0.05 * context_score))
        if contradictions:
            final_score = min(1.0, final_score + 0.05)

        assessment_label = "low_welfare_concern"
        if final_score >= 0.75:
            assessment_label = "high_welfare_concern"
        elif final_score >= 0.45:
            assessment_label = "moderate_welfare_concern"

        evidence_items = [
            EvidenceItem(
                label=fact.name,
                score=float(fact.weight) * (1.0 - uncertainty_penalty),
                detail=fact.explanation or f"{fact.name} observed",
                source=fact.source,
            )
            for fact in input_data.evidence
        ]

        alternative_explanations = [
            "The same evidence could reflect normal behavior in a low-stress environment",
            "Sensor or audio noise may be inflating the distress signal",
        ]
        if contradictions:
            alternative_explanations.append("Contradictions suggest manual review is warranted")

        recommended_review = bool(contradictions) or confidence < 0.6 or final_score > 0.7 and uncertainty_penalty > 0.3

        return ReasoningResult(
            final_assessment_score=final_score,
            assessment_label=assessment_label,
            reasoning_confidence=confidence,
            evidence=evidence_items,
            triggered_rules=rule_result.get("triggered_rules", []),
            ignored_rules=rule_result.get("ignored_rules", []),
            contradictions=contradictions,
            alternative_explanations=alternative_explanations,
            recommended_human_review=recommended_review,
            summary=self._build_summary(final_score, assessment_label, contradictions),
        )

    def _build_summary(self, score: float, label: str, contradictions: List[str]) -> str:
        base = f"Assessment={label}; score={score:.2f}"
        if contradictions:
            base += "; contradictions detected"
        return base
