from __future__ import annotations

from typing import Any, Dict, List

from .knowledge import KnowledgeBase
from .types import EvidenceFact, ReasoningInput


class ConstraintValidator:
    """Detects impossible or conflicting welfare situations."""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

    def validate(self, input_data: ReasoningInput) -> List[str]:
        evidence_map = {fact.name: fact for fact in input_data.evidence}
        contradictions: List[str] = []
        for constraint in self.knowledge_base.get_constraints():
            if self._is_triggered(constraint, evidence_map, input_data):
                contradictions.append(constraint.get("description", constraint.get("name", "constraint")))
        return contradictions

    def _is_triggered(self, constraint: Dict[str, Any], evidence_map: Dict[str, EvidenceFact], input_data: ReasoningInput) -> bool:
        conditions = constraint.get("conditions", [])
        triggered = True
        for condition in conditions:
            field = condition.get("field")
            expected_value = condition.get("value")
            value = self._resolve_value(field, input_data, evidence_map)
            if value != expected_value:
                triggered = False
                break
        return triggered

    def _resolve_value(self, field: str, input_data: ReasoningInput, evidence_map: Dict[str, EvidenceFact]) -> Any:
        if field in {"time_of_day", "environment", "crowding", "occupancy"}:
            return input_data.context.get(field)
        fact = evidence_map.get(field)
        if fact is None:
            return None
        return fact.value
