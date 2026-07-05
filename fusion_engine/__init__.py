"""Research-grade multimodal fusion engine for welfare monitoring."""

from .interfaces import FusionStrategy, FusionPluginRegistry
from .types import FeatureVector, ModalityFeature, FusionInput, FusionOutput, ValidationReport, FeatureStoreEntry
from .validation import FeatureValidator
from .alignment import FeatureAligner
from .store import FeatureStore
from .context import TemporalContextBuilder
from .strategies import WeightedFeatureFusion, EarlyFusion, LateFusion, AttentionFusion
from .calibration import ConfidenceCalibrator, UncertaintyEstimator
from .engine import MultimodalFusionEngine

__all__ = [
    "FusionStrategy",
    "FusionPluginRegistry",
    "FeatureVector",
    "ModalityFeature",
    "FusionInput",
    "FusionOutput",
    "ValidationReport",
    "FeatureStoreEntry",
    "FeatureValidator",
    "FeatureAligner",
    "FeatureStore",
    "TemporalContextBuilder",
    "WeightedFeatureFusion",
    "EarlyFusion",
    "LateFusion",
    "AttentionFusion",
    "ConfidenceCalibrator",
    "UncertaintyEstimator",
    "MultimodalFusionEngine",
]
