from __future__ import annotations

import math
from typing import Dict

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
        for feature in features:
            for key, value in feature.values.items():
                merged_values[key] = merged_values.get(key, 0.0) + float(value) * self.weights.get(feature.modality, 0.0)

        welfare_probability = min(1.0, max(0.0, sum(merged_values.values()) / max(1, len(merged_values))))
        return FusionOutput(
            unified_representation={"features": merged_values, "modalities": [feature.modality for feature in features]},
            welfare_probability=welfare_probability,
            confidence=sum(reliability.values()) / max(1, len(reliability)),
            uncertainty=max(0.0, 1.0 - sum(reliability.values()) / max(1, len(reliability))),
            feature_contributions=contributions,
            reliability_scores=reliability,
            fusion_strategy_used=self.name,
        )


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
        contribution_sum = {}
        for modality, feature, reliability in modalities:
            weight = reliability / total_reliability
            contribution_sum[modality] = weight
            for key, value in feature.values.items():
                weighted_values[key] = weighted_values.get(key, 0.0) + float(value) * weight

        probability = min(1.0, max(0.0, sum(weighted_values.values()) / max(1, len(weighted_values))))
        return FusionOutput(
            unified_representation={"features": weighted_values, "attention_weights": contribution_sum},
            welfare_probability=probability,
            confidence=sum(rel for _, _, rel in modalities) / len(modalities),
            uncertainty=max(0.0, 1.0 - (sum(rel for _, _, rel in modalities) / len(modalities))),
            feature_contributions=contribution_sum,
            reliability_scores={modality: rel for modality, _, rel in modalities},
            fusion_strategy_used=self.name,
        )
