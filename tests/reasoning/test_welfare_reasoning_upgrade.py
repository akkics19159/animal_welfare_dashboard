import unittest

from welfare_reasoning.engine import WelfareReasoningEngine
from welfare_reasoning.types import EvidenceFact, ReasoningInput


class WelfareReasoningUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.engine = WelfareReasoningEngine()

    def test_species_thresholds_and_temporal_state(self):
        inp = ReasoningInput(
            species="dog",
            evidence=[
                EvidenceFact(name="distress_vocalization", value=True, weight=0.9, source="audio"),
                EvidenceFact(name="pacing", value=True, weight=0.8, source="vision"),
                EvidenceFact(name="heart_rate_extreme", value=True, weight=0.9, source="sensor"),
            ],
            fusion_confidence=0.85,
            sensor_reliability=0.8,
            audio_quality=0.9,
            vision_quality=0.8,
            prediction_uncertainty=0.1,
            agreement_score=0.8,
            context={"occupancy": 1, "crowding": "low", "time_of_day": "night", "environment": "indoor"},
            history=[{"probability": 0.35}, {"probability": 0.55}, {"probability": 0.75}],
        )
        res = self.engine.reason(inp)
        self.assertGreater(res.final_assessment_score, 0.1)
        self.assertIn("state", res.temporal_welfare_state)

    def test_false_positive_suppression_in_reasoning(self):
        inp = ReasoningInput(
            species="cat",
            evidence=[
                EvidenceFact(name="mating", value=True, weight=0.8, source="vision"),
                EvidenceFact(name="distress_vocalization", value=True, weight=0.5, source="audio"),
            ],
            fusion_confidence=0.8,
            sensor_reliability=0.8,
            audio_quality=0.8,
            vision_quality=0.8,
            prediction_uncertainty=0.2,
            agreement_score=0.7,
        )
        res = self.engine.reason(inp)
        self.assertTrue(any(item.get("suppressed") for item in res.suppressed_evidence))

    def test_conflicts_reduce_confidence(self):
        clean = ReasoningInput(
            species="horse",
            evidence=[EvidenceFact(name="limping", value=True, weight=0.9, source="vision")],
            fusion_confidence=0.9,
            sensor_reliability=0.9,
            audio_quality=0.9,
            vision_quality=0.9,
            prediction_uncertainty=0.1,
            agreement_score=0.9,
        )
        conflicted = ReasoningInput(
            species="horse",
            evidence=[
                EvidenceFact(name="sleeping", value=True, weight=0.8, source="vision"),
                EvidenceFact(name="running", value=True, weight=0.8, source="vision"),
            ],
            fusion_confidence=0.9,
            sensor_reliability=0.9,
            audio_quality=0.9,
            vision_quality=0.9,
            prediction_uncertainty=0.1,
            agreement_score=0.9,
        )
        r1 = self.engine.reason(clean)
        r2 = self.engine.reason(conflicted)
        self.assertLessEqual(r2.reasoning_confidence, r1.reasoning_confidence)


if __name__ == "__main__":
    unittest.main()
