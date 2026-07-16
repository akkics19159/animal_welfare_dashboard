from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceFact:
    name: str
    value: Any
    weight: float = 1.0
    source: str = "unknown"
    explanation: Optional[str] = None


@dataclass
class ReasoningInput:
    species: str = "unknown"
    evidence: List[EvidenceFact] = field(default_factory=list)
    fusion_confidence: float = 0.5
    sensor_reliability: float = 0.5
    audio_quality: float = 0.5
    vision_quality: float = 0.5
    prediction_uncertainty: float = 0.5
    agreement_score: float = 0.5
    missing_modality_impact: float = 0.0
    calibration_score: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvidenceItem:
    label: str
    score: float
    detail: str
    source: str


@dataclass
class ReasoningResult:
    final_assessment_score: float
    assessment_label: str
    reasoning_confidence: float
    evidence: List[EvidenceItem] = field(default_factory=list)
    triggered_rules: List[str] = field(default_factory=list)
    ignored_rules: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    alternative_explanations: List[str] = field(default_factory=list)
    recommended_human_review: bool = False
    summary: str = ""
    welfare_score: float = 0.0
    risk_level: str = "low"
    severity: str = "low"
    urgency: str = "routine"
    prediction_uncertainty: float = 0.5
    agreement_score: float = 0.5
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    suppressed_evidence: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    temporal_welfare_state: Dict[str, Any] = field(default_factory=dict)
