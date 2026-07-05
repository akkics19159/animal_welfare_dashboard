from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BenchmarkResult:
    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SensorBenchmark:
    def __init__(self, name: str = "sensor"):
        self.name = name

    def evaluate(self, predictions: Dict[str, Any], ground_truth: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
        validation_error = 0.0
        normalization_error = 0.0
        anomaly_accuracy = 1.0 if predictions.get("anomaly") == ground_truth.get("anomaly") else 0.0
        trend_accuracy = 1.0 if predictions.get("trend") == ground_truth.get("trend") else 0.0
        calibration = float(predictions.get("calibration", 0.0))
        metrics = {
            "validation_error": validation_error,
            "normalization_error": normalization_error,
            "anomaly_detection_accuracy": anomaly_accuracy,
            "trend_detection_accuracy": trend_accuracy,
            "confidence_calibration": calibration,
        }
        return BenchmarkResult(name=self.name, metrics=metrics, metadata=metadata or {})
