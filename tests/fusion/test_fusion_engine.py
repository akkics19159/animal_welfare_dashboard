import unittest

from fusion_engine.alignment import FeatureAligner
from fusion_engine.calibration import ConfidenceCalibrator, UncertaintyEstimator
from fusion_engine.context import TemporalContextBuilder
from fusion_engine.engine import MultimodalFusionEngine
from fusion_engine.store import FeatureStore
from fusion_engine.strategies import AttentionFusion, EarlyFusion, LateFusion, WeightedFeatureFusion
from fusion_engine.types import FeatureVector, FusionInput, FusionOutput
from fusion_engine.validation import FeatureValidator


class FusionEngineTests(unittest.TestCase):
    def test_feature_validation(self):
        validator = FeatureValidator()
        feature = FeatureVector(modality="vision", timestamp=1.0, values={"motion": 0.8}, confidence=0.9, quality_score=0.8, version="1.0")
        report = validator.validate(feature)
        self.assertTrue(report.valid)

    def test_alignment(self):
        aligner = FeatureAligner(sync_tolerance_seconds=0.5)
        fusion_input = FusionInput(
            vision=FeatureVector(modality="vision", timestamp=1.0, values={"motion": 0.8}),
            audio=FeatureVector(modality="audio", timestamp=1.2, values={"energy": 0.3}),
            sensors=FeatureVector(modality="sensors", timestamp=1.0, values={"temp": 0.2}),
            timestamp=1.0,
        )
        aligned = aligner.align(fusion_input)
        self.assertEqual(aligned.vision.timestamp, 1.0)
        self.assertEqual(aligned.audio.timestamp, 1.2)

    def test_store(self):
        store = FeatureStore(max_history=5)
        feature = FeatureVector(modality="vision", timestamp=1.0, values={"motion": 0.7}, confidence=0.8, quality_score=0.9, version="1.0")
        store.add(feature, FeatureValidator().validate(feature))
        self.assertEqual(store.get_latest("vision").modality, "vision")

    def test_weighted_strategy(self):
        strategy = WeightedFeatureFusion()
        output = strategy.fuse(FusionInput(
            vision=FeatureVector(modality="vision", timestamp=1.0, values={"motion": 0.8}, confidence=0.9, quality_score=0.9),
            audio=FeatureVector(modality="audio", timestamp=1.0, values={"distress": 0.6}, confidence=0.8, quality_score=0.8),
        ))
        self.assertGreaterEqual(output.welfare_probability, 0.0)

    def test_attention_strategy(self):
        strategy = AttentionFusion()
        output = strategy.fuse(FusionInput(
            vision=FeatureVector(modality="vision", timestamp=1.0, values={"motion": 0.8}, confidence=0.9, quality_score=0.9),
            audio=FeatureVector(modality="audio", timestamp=1.0, values={"distress": 0.6}, confidence=0.7, quality_score=0.4),
        ))
        self.assertIn("attention_weights", output.unified_representation)

    def test_confidence_calibration(self):
        calibrator = ConfidenceCalibrator()
        output = FusionInput(vision=FeatureVector(modality="vision", timestamp=1.0, values={"motion": 0.8}))
        self.assertIsNotNone(output)

    def test_uncertainty_estimation(self):
        estimator = UncertaintyEstimator()
        output = FusionOutput(welfare_probability=0.5, confidence=0.8)
        output = estimator.estimate(output, {"vision": 0.8, "audio": 0.6})
        self.assertGreaterEqual(output.uncertainty, 0.0)

    def test_end_to_end_fusion(self):
        engine = MultimodalFusionEngine()
        output = engine.fuse(FusionInput(
            vision=FeatureVector(modality="vision", timestamp=1.0, values={"motion": 0.8}, confidence=0.9, quality_score=0.9),
            audio=FeatureVector(modality="audio", timestamp=1.0, values={"distress": 0.7}, confidence=0.8, quality_score=0.8),
            sensors=FeatureVector(modality="sensors", timestamp=1.0, values={"temp": 0.2}, confidence=0.7, quality_score=0.8),
            timestamp=1.0,
        ))
        self.assertGreaterEqual(output.welfare_probability, 0.0)
        self.assertIn("validation", output.unified_representation)


if __name__ == "__main__":
    unittest.main()
