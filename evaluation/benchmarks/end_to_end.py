from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BenchmarkResult:
    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EndToEndBenchmark:
    def __init__(self, name: str = "end_to_end"):
        self.name = name

    def evaluate(self, accuracy: float, latency_ms: float, throughput: float, memory_mb: float, cpu_percent: float, gpu_percent: float, metadata: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
        metrics = {
            "overall_welfare_prediction_accuracy": accuracy,
            "latency_ms": latency_ms,
            "throughput": throughput,
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "gpu_percent": gpu_percent,
        }
        return BenchmarkResult(name=self.name, metrics=metrics, metadata=metadata or {})
