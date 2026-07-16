from __future__ import annotations

import math
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
        decision_cfg = self.knowledge_base.get_decision_config() or {}

        # Bayesian-inspired evidence accumulation in log-odds space.
        prior = float(decision_cfg.get("prior_risk", 0.15))
        prior = min(0.95, max(0.01, prior))
        odds = prior / (1.0 - prior)

        suppressed_evidence: List[Dict[str, Any]] = []
        primary_evidence: List[Dict[str, Any]] = []
        contradictory_evidence: List[Dict[str, Any]] = []
        supporting_evidence: List[Dict[str, Any]] = []
        reasoning_trace: List[str] = []

        non_distress_behaviours = {
            "mating",
            "courtship",
            "play",
            "feeding",
            "drinking",
            "grooming",
            "parent_offspring_communication",
            "normal_social_vocalisation",
            "exploration",
            "curiosity",
            "resting",
            "sleeping",
        }
        suppression_strength = float(decision_cfg.get("suppression_strength", 0.7))

        for fact in input_data.evidence:
            name = str(fact.name).lower()
            val = fact.value
            is_active = bool(val) if isinstance(val, bool) else float(val or 0.0) > 0.0
            base_weight = min(2.5, max(0.0, float(fact.weight)))

            if not is_active:
                continue

            lr = 1.0 + base_weight
            suppressed = False
            if name in non_distress_behaviours:
                lr = max(1.0, lr * (1.0 - suppression_strength))
                suppressed = True

            odds *= lr
            detail = {
                "name": fact.name,
                "source": fact.source,
                "weight": base_weight,
                "explanation": fact.explanation,
                "suppressed": suppressed,
            }
            if suppressed:
                suppressed_evidence.append({**detail, "reason": "configured_false_positive_reduction"})
                reasoning_trace.append(f"Suppressed evidence {fact.name} due to non-distress behavior policy")
            else:
                supporting_evidence.append(detail)
                if base_weight >= 0.75:
                    primary_evidence.append(detail)
                reasoning_trace.append(f"Accumulated evidence {fact.name} with likelihood ratio {lr:.2f}")

        bayesian_prob = odds / (1.0 + odds)
        bayesian_prob = min(1.0, max(0.0, bayesian_prob))

        quality_factor = (
            0.25 * max(0.0, min(1.0, input_data.fusion_confidence))
            + 0.25 * max(0.0, min(1.0, input_data.sensor_reliability))
            + 0.25 * max(0.0, min(1.0, input_data.audio_quality))
            + 0.25 * max(0.0, min(1.0, input_data.vision_quality))
        )
        # Agreement and calibration increase confidence, missing modality impact decreases it.
        quality_factor = min(
            1.0,
            max(
                0.0,
                quality_factor * (0.8 + 0.2 * max(0.0, min(1.0, input_data.agreement_score)))
                * (0.85 + 0.15 * max(0.0, min(1.0, input_data.calibration_score)))
                * (1.0 - max(0.0, min(0.8, input_data.missing_modality_impact))),
            ),
        )
        uncertainty_penalty = max(0.0, min(1.0, input_data.prediction_uncertainty))
        confidence = max(0.05, min(0.99, quality_factor * (1.0 - uncertainty_penalty)))

        if contradictions:
            confidence *= 0.9
            contradictory_evidence.extend(
                {
                    "name": "constraint_conflict",
                    "source": "reasoning_constraint",
                    "weight": 0.0,
                    "explanation": c,
                }
                for c in contradictions
            )
            reasoning_trace.append("Applied contradiction penalty to confidence")

        # Confidence-weighted evidence fusion + Bayesian accumulation + rule support.
        weighted_evidence = min(1.0, max(0.0, support_score + context_score))
        blend_rule = float(decision_cfg.get("blend_rule", 0.35))
        blend_bayes = float(decision_cfg.get("blend_bayes", 0.40))
        blend_conf = float(decision_cfg.get("blend_conf", 0.25))
        blend_sum = max(1e-6, blend_rule + blend_bayes + blend_conf)
        blend_rule /= blend_sum
        blend_bayes /= blend_sum
        blend_conf /= blend_sum
        final_score = (
            blend_rule * weighted_evidence
            + blend_bayes * bayesian_prob
            + blend_conf * (weighted_evidence * confidence)
        )
        final_score = min(1.0, max(0.0, final_score))
        if contradictions:
            final_score = min(1.0, max(0.0, final_score * 0.95))

        # Species-aware configurable thresholds.
        species = str(input_data.species or "unknown").lower()
        s_cfg = (self.knowledge_base.get_species_thresholds() or {}).get(species, {})
        high_thr = float(s_cfg.get("high", decision_cfg.get("high_threshold", 0.75)))
        med_thr = float(s_cfg.get("moderate", decision_cfg.get("moderate_threshold", 0.45)))

        assessment_label = "low_welfare_concern"
        if final_score >= high_thr:
            assessment_label = "high_welfare_concern"
        elif final_score >= med_thr:
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

        risk_level = "low"
        severity = "low"
        urgency = "routine"
        if final_score >= high_thr:
            risk_level = "high"
            severity = "critical"
            urgency = "immediate"
        elif final_score >= med_thr:
            risk_level = "moderate"
            severity = "elevated"
            urgency = "soon"

        temporal_state = context_result.get("temporal_welfare_state", {}) or {}
        reasoning_trace.append(
            f"Final score {final_score:.3f} computed from weighted={weighted_evidence:.3f}, bayesian={bayesian_prob:.3f}, confidence={confidence:.3f}"
        )

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
            welfare_score=float((1.0 - final_score) * 100.0),
            risk_level=risk_level,
            severity=severity,
            urgency=urgency,
            prediction_uncertainty=uncertainty_penalty,
            agreement_score=float(max(0.0, min(1.0, input_data.agreement_score))),
            evidence_summary={
                "primary_evidence": primary_evidence,
                "supporting_evidence": supporting_evidence,
                "contradictory_evidence": contradictory_evidence,
            },
            suppressed_evidence=suppressed_evidence,
            reasoning_trace=reasoning_trace,
            temporal_welfare_state=temporal_state,
        )

    def _build_summary(self, score: float, label: str, contradictions: List[str]) -> str:
        base = f"Assessment={label}; score={score:.2f}"
        if contradictions:
            base += "; contradictions detected"
        return base
