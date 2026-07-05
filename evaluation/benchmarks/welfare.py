from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BenchmarkResult:
    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WelfareReasoningBenchmark:
    def __init__(self, name: str = "welfare_reasoning"):
        self.name = name

    def evaluate(self, rule_coverage: float, constraint_detection: float, contradiction_detection: float, explanation_completeness: float, reasoning_confidence: float, human_agreement: float, metadata: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
        metrics = {
            "rule_coverage": rule_coverage,
            "constraint_detection": constraint_detection,
            "contradiction_detection": contradiction_detection,
            "explanation_completeness": explanation_completeness,
            "reasoning_confidence": reasoning_confidence,
            "human_agreement": human_agreement,
        }
        return BenchmarkResult(name=self.name, metrics=metrics, metadata=metadata or {})
