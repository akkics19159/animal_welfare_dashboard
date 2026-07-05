from __future__ import annotations

from typing import Any, Dict, List, Optional

from .constraints import ConstraintValidator
from .context import ContextEngine
from .inference import ConfidenceAwareInferenceEngine
from .interfaces import ReasoningPlugin, ReasoningPluginRegistry
from .knowledge import KnowledgeBase
from .rules import RuleEngine
from .types import EvidenceFact, ReasoningInput, ReasoningResult


class WelfareReasoningEngine:
    """Hybrid welfare reasoning engine with knowledge, context, rules, constraints, and confidence-aware inference."""

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        context_engine: Optional[ContextEngine] = None,
        rule_engine: Optional[RuleEngine] = None,
        constraint_validator: Optional[ConstraintValidator] = None,
        inference_engine: Optional[ConfidenceAwareInferenceEngine] = None,
        plugin_registry: Optional[ReasoningPluginRegistry] = None,
    ):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.context_engine = context_engine or ContextEngine(self.knowledge_base)
        self.rule_engine = rule_engine or RuleEngine(self.knowledge_base)
        self.constraint_validator = constraint_validator or ConstraintValidator(self.knowledge_base)
        self.inference_engine = inference_engine or ConfidenceAwareInferenceEngine(self.knowledge_base)
        self.plugin_registry = plugin_registry or ReasoningPluginRegistry()
        self._register_default_plugins()

    def reason(self, input_data: ReasoningInput) -> ReasoningResult:
        context_result = self.context_engine.evaluate(input_data)
        rule_result = self.rule_engine.evaluate(input_data)
        contradictions = self.constraint_validator.validate(input_data)
        result = self.inference_engine.infer(input_data, rule_result, context_result, contradictions)

        for plugin in self.plugin_registry._registry.values():
            plugin_result = plugin.execute(input_data, {"context": context_result, "rules": rule_result, "contradictions": contradictions})
            if plugin_result is not None:
                result = self._merge_results(result, plugin_result)

        return result

    def _merge_results(self, base: ReasoningResult, override: ReasoningResult) -> ReasoningResult:
        return ReasoningResult(
            final_assessment_score=override.final_assessment_score if override.final_assessment_score else base.final_assessment_score,
            assessment_label=override.assessment_label or base.assessment_label,
            reasoning_confidence=override.reasoning_confidence if override.reasoning_confidence else base.reasoning_confidence,
            evidence=override.evidence or base.evidence,
            triggered_rules=override.triggered_rules or base.triggered_rules,
            ignored_rules=override.ignored_rules or base.ignored_rules,
            contradictions=override.contradictions or base.contradictions,
            alternative_explanations=override.alternative_explanations or base.alternative_explanations,
            recommended_human_review=override.recommended_human_review or base.recommended_human_review,
            summary=override.summary or base.summary,
        )

    def _register_default_plugins(self) -> None:
        class DefaultPlugin(ReasoningPlugin):
            def __init__(self):
                super().__init__("default_reasoning_plugin")

            def execute(self, input_data: ReasoningInput, state: Dict[str, Any]) -> ReasoningResult:
                return ReasoningResult(final_assessment_score=0.0, assessment_label="", reasoning_confidence=0.0)

        self.plugin_registry.register("default_reasoning_plugin", DefaultPlugin())
