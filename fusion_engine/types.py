from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModalityFeature:
    modality: str
    timestamp: float
    values: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    quality_score: float = 0.0
    latency_ms: float = 0.0
    health_status: str = "healthy"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVector:
    modality: str
    timestamp: float
    values: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    quality_score: float = 0.0
    latency_ms: float = 0.0
    health_status: str = "healthy"
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionInput:
    vision: Optional[FeatureVector] = None
    audio: Optional[FeatureVector] = None
    sensors: Optional[FeatureVector] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    modality: str = "unknown"
    version_check: bool = True


@dataclass
class FeatureStoreEntry:
    timestamp: float
    modality: str
    vector: FeatureVector
    validation: ValidationReport


@dataclass
class FusionOutput:
    unified_representation: Dict[str, Any] = field(default_factory=dict)
    welfare_probability: float = 0.0
    confidence: float = 0.0
    uncertainty: float = 0.0
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    reliability_scores: Dict[str, float] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    fusion_strategy_used: str = "unknown"
    model_version: str = "1.0"
    calibration_score: float = 0.0
    prediction_reliability: float = 0.0
    confidence_interval: Dict[str, float] = field(default_factory=dict)
    validation_report: Dict[str, ValidationReport] = field(default_factory=dict)
