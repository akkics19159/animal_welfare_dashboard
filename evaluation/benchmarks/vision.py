from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from evaluation.metrics.classification import compute_binary_metrics


@dataclass
class BenchmarkResult:
    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisionBenchmark:
    def __init__(self, name: str = "vision"):
        self.name = name

    def evaluate(self, predictions: Dict[str, Any], ground_truth: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
        predicted_detections = predictions.get("detections", [])
        truth_detections = ground_truth.get("detections", [])
        predicted_classes = predictions.get("classes", [])
        truth_classes = ground_truth.get("classes", [])

        precision, recall, f1 = compute_binary_metrics(
            [1 if item in truth_detections else 0 for item in predicted_detections],
            [1 if item in predicted_detections else 0 for item in truth_detections],
        )
        counting_accuracy = 1.0 - abs(len(predicted_detections) - len(truth_detections)) / max(1, len(truth_detections))
        behaviour_accuracy = 1.0 if predicted_classes == truth_classes else 0.0

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "map": 0.0,
            "mota": 0.0,
            "idf1": 0.0,
            "counting_accuracy": counting_accuracy,
            "behaviour_accuracy": behaviour_accuracy,
        }
        return BenchmarkResult(name=self.name, metrics=metrics, metadata=metadata or {})
