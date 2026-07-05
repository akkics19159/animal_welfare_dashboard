from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BenchmarkResult:
    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FusionBenchmark:
    def __init__(self, name: str = "fusion"):
        self.name = name

    def evaluate(self, strategy_name: str, accuracy: float, calibration: float, latency_ms: float, robustness: float, metadata: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
        metrics = {
            "accuracy": accuracy,
            "calibration": calibration,
            "latency_ms": latency_ms,
            "robustness_to_missing_modalities": robustness,
        }
        return BenchmarkResult(name=strategy_name, metrics=metrics, metadata=metadata or {})
