import unittest

from welfare_reasoning.engine import WelfareReasoningEngine
from welfare_reasoning.types import ReasoningInput, EvidenceFact


class WelfareReasoningEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = WelfareReasoningEngine()

    def test_rule_evaluation_supports_nested_rules(self):
        input_data = ReasoningInput(
            species="dog",
            evidence=[
                EvidenceFact(name="distress_vocalization", value=True, weight=0.9, source="audio"),
                EvidenceFact(name="limping", value=True, weight=0.8, source="vision"),
            ],
            fusion_confidence=0.9,
            sensor_reliability=0.85,
            audio_quality=0.9,
            vision_quality=0.85,
            prediction_uncertainty=0.15,
            context={"time_of_day": "night", "environment": "indoor", "crowding": "low"},
        )

        result = self.engine.reason(input_data)
        self.assertGreater(result.final_assessment_score, 0.5)
        self.assertTrue(any(rule == "distress_vocalization_rule" for rule in result.triggered_rules))

    def test_constraint_validation_detects_conflicts(self):
        input_data = ReasoningInput(
            species="dog",
            evidence=[
                EvidenceFact(name="sleeping", value=True, weight=0.9, source="vision"),
                EvidenceFact(name="running", value=True, weight=0.9, source="vision"),
            ],
            fusion_confidence=0.8,
            sensor_reliability=0.9,
            audio_quality=0.8,
            vision_quality=0.9,
            prediction_uncertainty=0.1,
        )

        result = self.engine.reason(input_data)
        self.assertTrue(result.contradictions)

    def test_evidence_aggregation_collects_structured_evidence(self):
        input_data = ReasoningInput(
            species="cat",
            evidence=[
                EvidenceFact(name="distress_vocalization", value=True, weight=0.9, source="audio"),
                EvidenceFact(name="pacing", value=True, weight=0.8, source="vision"),
            ],
            fusion_confidence=0.8,
            sensor_reliability=0.7,
            audio_quality=0.85,
            vision_quality=0.8,
            prediction_uncertainty=0.2,
        )

        result = self.engine.reason(input_data)
        self.assertTrue(result.evidence)
        self.assertTrue(any(item.label == "distress_vocalization" for item in result.evidence))

    def test_context_reasoning_uses_context_to_adjust_score(self):
        base_input = ReasoningInput(
            species="cow",
            evidence=[EvidenceFact(name="pacing", value=True, weight=0.8, source="vision")],
            fusion_confidence=0.8,
            sensor_reliability=0.8,
            audio_quality=0.8,
            vision_quality=0.8,
            prediction_uncertainty=0.1,
            context={"time_of_day": "day", "environment": "outdoor", "crowding": "low"},
        )
        stressed_context_input = ReasoningInput(
            species="cow",
            evidence=[EvidenceFact(name="pacing", value=True, weight=0.8, source="vision")],
            fusion_confidence=0.8,
            sensor_reliability=0.8,
            audio_quality=0.8,
            vision_quality=0.8,
            prediction_uncertainty=0.1,
            context={"time_of_day": "night", "environment": "indoor", "crowding": "high"},
        )

        base_result = self.engine.reason(base_input)
        stressed_result = self.engine.reason(stressed_context_input)
        self.assertGreater(stressed_result.final_assessment_score, base_result.final_assessment_score)

    def test_confidence_propagation_reduces_reasoning_confidence(self):
        high_confidence_input = ReasoningInput(
            species="horse",
            evidence=[EvidenceFact(name="limping", value=True, weight=0.8, source="vision")],
            fusion_confidence=0.95,
            sensor_reliability=0.95,
            audio_quality=0.9,
            vision_quality=0.95,
            prediction_uncertainty=0.05,
        )
        low_confidence_input = ReasoningInput(
            species="horse",
            evidence=[EvidenceFact(name="limping", value=True, weight=0.8, source="vision")],
            fusion_confidence=0.4,
            sensor_reliability=0.3,
            audio_quality=0.2,
            vision_quality=0.3,
            prediction_uncertainty=0.8,
        )

        high_result = self.engine.reason(high_confidence_input)
        low_result = self.engine.reason(low_confidence_input)
        self.assertGreater(high_result.reasoning_confidence, low_result.reasoning_confidence)

    def test_contradiction_detection_detects_no_animal_with_distress_audio(self):
        input_data = ReasoningInput(
            species="bird",
            evidence=[
                EvidenceFact(name="distress_vocalization", value=True, weight=0.9, source="audio"),
            ],
            fusion_confidence=0.8,
            sensor_reliability=0.8,
            audio_quality=0.9,
            vision_quality=0.2,
            prediction_uncertainty=0.1,
            context={"occupancy": 0},
        )

        result = self.engine.reason(input_data)
        self.assertTrue(any("no detected animals" in contradiction for contradiction in result.contradictions))


if __name__ == "__main__":
    unittest.main()
