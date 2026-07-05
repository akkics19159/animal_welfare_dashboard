from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from evaluation.metrics.calibration import compute_calibration_metrics
from evaluation.metrics.classification import compute_binary_metrics


@dataclass
class BenchmarkResult:
    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AudioBenchmark:
    def __init__(self, name: str = "audio"):
        self.name = name

    def evaluate(self, predictions: Sequence[int], targets: Sequence[int], scores: Optional[Sequence[float]] = None, metadata: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
        precision, recall, f1 = compute_binary_metrics(targets, predictions)
        scores = scores if scores is not None else [0.5 for _ in targets]
        calibration = compute_calibration_metrics(scores, targets)
        tp = sum(1 for t, p in zip(targets, predictions) if int(t) == 1 and int(p) == 1)
        fp = sum(1 for t, p in zip(targets, predictions) if int(t) == 0 and int(p) == 1)
        fn = sum(1 for t, p in zip(targets, predictions) if int(t) == 1 and int(p) == 0)
        fpr = fp / max(1, fp + sum(1 for t in targets if int(t) == 0))
        fnr = fn / max(1, fn + sum(1 for t in targets if int(t) == 1))
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "brier_score": calibration["brier_score"],
        }
        return BenchmarkResult(name=self.name, metrics=metrics, metadata=metadata or {})
