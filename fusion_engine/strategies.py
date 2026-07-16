from __future__ import annotations

import math
from typing import Dict, List

from .interfaces import FusionStrategy
from .types import FusionInput, FusionOutput, FeatureVector


class WeightedFeatureFusion(FusionStrategy):
    """Baseline fusion strategy using modality-weighted feature aggregation."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__("weighted_feature_fusion")
        self.weights = weights or {"vision": 0.45, "audio": 0.35, "sensors": 0.20}

    def fuse(self, fusion_input: FusionInput) -> FusionOutput:
        features = []
        contributions = {}
        reliability = {}
        for modality in ("vision", "audio", "sensors"):
            feature = getattr(fusion_input, modality, None)
            if feature is None:
                continue
            features.append(feature)
            weight = self.weights.get(modality, 0.0)
            contributions[modality] = weight
            reliability[modality] = min(1.0, max(0.0, feature.confidence * feature.quality_score))

        if not features:
            return FusionOutput(fusion_strategy_used=self.name)

        merged_values = {}
        modality_scores: Dict[str, float] = {}
        for feature in features:
            vals: List[float] = []
            for key, value in feature.values.items():
                merged_values[key] = merged_values.get(key, 0.0) + float(value) * self.weights.get(feature.modality, 0.0)
                vals.append(float(value))
            modality_scores[feature.modality] = min(1.0, max(0.0, sum(vals) / max(1, len(vals))))

        weighted_evidence = min(1.0, max(0.0, sum(merged_values.values()) / max(1, len(merged_values))))
        bayes_prob = self._bayesian_probability(modality_scores, reliability)
        welfare_probability = min(1.0, max(0.0, 0.55 * weighted_evidence + 0.45 * bayes_prob))
        agreement_score = self._agreement(modality_scores)
        conf = sum(reliability.values()) / max(1, len(reliability))
        return FusionOutput(
            unified_representation={
                "features": merged_values,
                "modalities": [feature.modality for feature in features],
                "modality_scores": modality_scores,
            },
            welfare_probability=welfare_probability,
            confidence=conf,
            uncertainty=max(0.0, 1.0 - conf),
            feature_contributions=contributions,
            reliability_scores=reliability,
            fusion_strategy_used=self.name,
            agreement_score=agreement_score,
            modality_confidence=reliability,
        )

    def _bayesian_probability(self, modality_scores: Dict[str, float], reliability: Dict[str, float]) -> float:
        if not modality_scores:
            return 0.0
        prior = 0.15
        odds = prior / (1.0 - prior)
        for modality, score in modality_scores.items():
            rel = float(reliability.get(modality, 0.5))
            lr = 1.0 + max(0.0, min(1.0, score)) * (0.5 + rel)
            odds *= lr
        return min(1.0, max(0.0, odds / (1.0 + odds)))

    def _agreement(self, modality_scores: Dict[str, float]) -> float:
        vals = list(modality_scores.values())
        if len(vals) <= 1:
            return 1.0 if vals else 0.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)
        return float(max(0.0, min(1.0, 1.0 - std * 2.0)))


class EarlyFusion(FusionStrategy):
    def __init__(self):
        super().__init__("early_fusion")

    def fuse(self, fusion_input: FusionInput) -> FusionOutput:
        values = {}
        for modality in ("vision", "audio", "sensors"):
            feature = getattr(fusion_input, modality, None)
            if feature is None:
                continue
            for key, value in feature.values.items():
                values[key] = values.get(key, 0.0) + float(value)
        welfare_probability = min(1.0, max(0.0, sum(values.values()) / max(1, len(values))))
        return FusionOutput(unified_representation={"features": values}, welfare_probability=welfare_probability, fusion_strategy_used=self.name)


class LateFusion(FusionStrategy):
    def __init__(self):
        super().__init__("late_fusion")

    def fuse(self, fusion_input: FusionInput) -> FusionOutput:
        probs = []
        for modality in ("vision", "audio", "sensors"):
            feature = getattr(fusion_input, modality, None)
            if feature is None:
                continue
            probs.append(float(feature.confidence))
        probability = sum(probs) / max(1, len(probs)) if probs else 0.0
        return FusionOutput(welfare_probability=probability, confidence=probability, fusion_strategy_used=self.name)


class AttentionFusion(FusionStrategy):
    def __init__(self):
        super().__init__("attention_fusion")

    def fuse(self, fusion_input: FusionInput) -> FusionOutput:
        modalities = []
        for modality in ("vision", "audio", "sensors"):
            feature = getattr(fusion_input, modality, None)
            if feature is None:
                continue
            reliability = min(1.0, max(0.0, feature.confidence * feature.quality_score))
            if feature.health_status != "healthy":
                reliability *= 0.5
            modalities.append((modality, feature, reliability))

        if not modalities:
            return FusionOutput(fusion_strategy_used=self.name)

        total_reliability = sum(rel for _, _, rel in modalities)
        if total_reliability <= 0:
            total_reliability = 1.0

        weighted_values = {}
        modality_scores: Dict[str, float] = {}
        contribution_sum = {}
        for modality, feature, reliability in modalities:
            weight = reliability / total_reliability
            contribution_sum[modality] = weight
            vals: List[float] = []
            for key, value in feature.values.items():
                weighted_values[key] = weighted_values.get(key, 0.0) + float(value) * weight
                vals.append(float(value))
            modality_scores[modality] = min(1.0, max(0.0, sum(vals) / max(1, len(vals))))

        weighted_prob = min(1.0, max(0.0, sum(weighted_values.values()) / max(1, len(weighted_values))))
        bayes_prob = WeightedFeatureFusion()._bayesian_probability(modality_scores, {m: r for m, _, r in modalities})
        probability = min(1.0, max(0.0, 0.5 * weighted_prob + 0.5 * bayes_prob))
        agreement_score = WeightedFeatureFusion()._agreement(modality_scores)
        confidence = sum(rel for _, _, rel in modalities) / len(modalities)
        return FusionOutput(
            unified_representation={
                "features": weighted_values,
                "attention_weights": contribution_sum,
                "modality_scores": modality_scores,
            },
            welfare_probability=probability,
            confidence=confidence,
            uncertainty=max(0.0, 1.0 - confidence),
            feature_contributions=contribution_sum,
            reliability_scores={modality: rel for modality, _, rel in modalities},
            fusion_strategy_used=self.name,
            agreement_score=agreement_score,
            modality_confidence={modality: rel for modality, _, rel in modalities},
        )
