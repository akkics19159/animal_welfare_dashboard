from __future__ import annotations

import logging
from typing import Dict, Optional

from .alignment import FeatureAligner
from .calibration import ConfidenceCalibrator, UncertaintyEstimator
from .context import TemporalContextBuilder
from .interfaces import FusionPluginRegistry, FusionStrategy
from .store import FeatureStore
from .strategies import AttentionFusion, EarlyFusion, LateFusion, WeightedFeatureFusion
from .types import FeatureVector, FusionInput, FusionOutput
from .validation import FeatureValidator


logger = logging.getLogger(__name__)


class MultimodalFusionEngine:
    """Research-grade multimodal fusion engine with validation, alignment, store, temporal context and calibration."""

    def __init__(
        self,
        validator: Optional[FeatureValidator] = None,
        aligner: Optional[FeatureAligner] = None,
        feature_store: Optional[FeatureStore] = None,
        temporal_builder: Optional[TemporalContextBuilder] = None,
        strategy: Optional[FusionStrategy] = None,
        plugin_registry: Optional[FusionPluginRegistry] = None,
        calibrator: Optional[ConfidenceCalibrator] = None,
        uncertainty_estimator: Optional[UncertaintyEstimator] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        self.validator = validator or FeatureValidator()
        self.aligner = aligner or FeatureAligner()
        self.feature_store = feature_store or FeatureStore()
        self.temporal_builder = temporal_builder or TemporalContextBuilder()
        self.plugin_registry = plugin_registry or FusionPluginRegistry()
        self.calibrator = calibrator or ConfidenceCalibrator()
        self.uncertainty_estimator = uncertainty_estimator or UncertaintyEstimator()
        self.logger = logger_instance or logger
        self.strategy = strategy or self._default_strategy()
        self._register_default_plugins()

    def fuse(self, fusion_input: FusionInput, strategy_name: Optional[str] = None) -> FusionOutput:
        aligned_input = self.aligner.align(fusion_input)
        validation_report = {}
        for modality in ("vision", "audio", "sensors"):
            feature = getattr(aligned_input, modality, None)
            report = self.validator.validate(feature)
            validation_report[modality] = report
            if feature is not None:
                self.feature_store.add(feature, report)

        strategy = self._resolve_strategy(strategy_name)
        output = strategy.fuse(aligned_input)
        output.validation_report = validation_report
        output.reliability_scores = self._build_reliability_scores(aligned_input, output)
        output = self.calibrator.calibrate(output, output.reliability_scores)
        output = self.uncertainty_estimator.estimate(output, output.reliability_scores)

        history = []
        for modality in ("vision", "audio", "sensors"):
            latest = self.feature_store.get_latest(modality)
            if latest is not None:
                history.append(latest.vector)
        output.temporal_context = self.temporal_builder.build(history, reference_timestamp=aligned_input.timestamp)
        output.unified_representation = output.unified_representation or {}
        output.unified_representation["temporal_context"] = output.temporal_context
        output.unified_representation["validation"] = {name: report.__dict__ for name, report in validation_report.items()}
        output.unified_representation["welfare_features"] = {
            "behavior": output.unified_representation.get("features", {}),
            "species": output.unified_representation.get("features", {}),
            "occupancy": output.unified_representation.get("features", {}),
            "distress_probabilities": output.unified_representation.get("features", {}),
            "physiological_trends": output.unified_representation.get("features", {}),
            "motion_statistics": output.unified_representation.get("features", {}),
            "temporal_descriptors": output.temporal_context,
            "confidence_values": output.reliability_scores,
            "quality_scores": output.reliability_scores,
        }
        return output

    def _build_reliability_scores(self, fusion_input: FusionInput, output: FusionOutput) -> Dict[str, float]:
        reliability = {}
        for modality in ("vision", "audio", "sensors"):
            feature = getattr(fusion_input, modality, None)
            if feature is None:
                continue
            reliability[modality] = min(1.0, max(0.0, feature.confidence * feature.quality_score))
            if feature.health_status != "healthy":
                reliability[modality] *= 0.5
        return reliability

    def _resolve_strategy(self, strategy_name: Optional[str]) -> FusionStrategy:
        if strategy_name is None:
            return self.strategy
        plugin = self.plugin_registry.get(strategy_name)
        if plugin is not None:
            return plugin
        raise KeyError(f"Unknown fusion strategy: {strategy_name}")

    def _register_default_plugins(self) -> None:
        self.plugin_registry.register("weighted_feature_fusion", WeightedFeatureFusion())
        self.plugin_registry.register("early_fusion", EarlyFusion())
        self.plugin_registry.register("late_fusion", LateFusion())
        self.plugin_registry.register("attention_fusion", AttentionFusion())

    def _default_strategy(self) -> FusionStrategy:
        return AttentionFusion()
