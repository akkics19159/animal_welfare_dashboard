from __future__ import annotations

from typing import Any, Dict, List, Optional

from .knowledge import KnowledgeBase
from .types import EvidenceFact, ReasoningInput


class RuleEngine:
    """Modular rule engine supporting AND/OR/NOT/threshold/weighted/nested rules."""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

    def evaluate(self, input_data: ReasoningInput) -> Dict[str, Any]:
        evidence_map = {fact.name: fact for fact in input_data.evidence}
        triggered: List[str] = []
        ignored: List[str] = []
        total_support = 0.0
        total_weight = 0.0
        detail_items: List[Dict[str, Any]] = []

        for rule in self.knowledge_base.get_reasoning_rules():
            result = self._evaluate_rule(rule, evidence_map, input_data)
            total_weight += float(rule.get("weight", 0.0))
            if result["satisfied"]:
                triggered.append(rule.get("name", "rule"))
                total_support += float(result.get("support", 0.0))
                detail_items.append({"name": rule.get("name"), "support": result.get("support", 0.0)})
            else:
                ignored.append(rule.get("name", "rule"))

        support_score = min(1.0, total_support / max(1.0, total_weight)) if total_weight > 0 else 0.0
        return {
            "support_score": support_score,
            "triggered_rules": triggered,
            "ignored_rules": ignored,
            "rule_details": detail_items,
        }

    def _evaluate_rule(self, rule: Dict[str, Any], evidence_map: Dict[str, EvidenceFact], input_data: ReasoningInput) -> Dict[str, Any]:
        op = str(rule.get("op", "exists")).lower()
        weight = float(rule.get("weight", 0.0))
        if op in {"and", "or", "not", "weighted"}:
            child_results = [self._evaluate_rule(child, evidence_map, input_data) for child in rule.get("rules", [])]
            if op == "and":
                satisfied = all(item["satisfied"] for item in child_results)
                support = sum(item.get("support", 0.0) for item in child_results)
            elif op == "or":
                satisfied = any(item["satisfied"] for item in child_results)
                support = max(item.get("support", 0.0) for item in child_results) if child_results else 0.0
            elif op == "not":
                child = child_results[0] if child_results else {"satisfied": False, "support": 0.0}
                satisfied = not child["satisfied"]
                support = weight if satisfied else 0.0
            else:
                satisfied = False
                support = 0.0
                total = 0.0
                for item in child_results:
                    if item["satisfied"]:
                        support += item.get("support", 0.0)
                        total += 1
                satisfied = total > 0
                support = support / max(1, total) if total else 0.0
            return {"satisfied": satisfied, "support": support if satisfied else 0.0}

        field = rule.get("field")
        if field is None:
            return {"satisfied": False, "support": 0.0}

        if op == "exists":
            fact = evidence_map.get(field)
            value = rule.get("value", True)
            satisfied = bool(fact is not None and fact.value == value)
            support = weight if satisfied else 0.0
            return {"satisfied": satisfied, "support": support}

        if op == "threshold":
            value = self._resolve_value(field, input_data, evidence_map)
            direction = rule.get("direction", "gte")
            threshold = float(rule.get("value", 0.0))
            if direction == "lt":
                satisfied = value < threshold
            else:
                satisfied = value >= threshold
            return {"satisfied": satisfied, "support": weight if satisfied else 0.0}

        if op == "eq":
            value = self._resolve_value(field, input_data, evidence_map)
            satisfied = value == rule.get("value")
            return {"satisfied": satisfied, "support": weight if satisfied else 0.0}

        return {"satisfied": False, "support": 0.0}

    def _resolve_value(self, field: str, input_data: ReasoningInput, evidence_map: Dict[str, EvidenceFact]) -> Any:
        if field == "species":
            return input_data.species.lower()
        if field in {"fusion_confidence", "sensor_reliability", "audio_quality", "vision_quality", "prediction_uncertainty"}:
            return getattr(input_data, field)
        if field in {"time_of_day", "environment", "crowding", "occupancy"}:
            return input_data.context.get(field)
        fact = evidence_map.get(field)
        if fact is None:
            return None
        return fact.value
