import json
import os
import tempfile
import unittest

from evaluation.metrics.classification import compute_binary_metrics
from evaluation.metrics.calibration import compute_calibration_metrics
from evaluation.benchmarks.vision import VisionBenchmark
from evaluation.benchmarks.audio import AudioBenchmark
from evaluation.reports.reporting import ReportGenerator
from evaluation.experiments.tracker import ExperimentTracker


class EvaluationFrameworkTests(unittest.TestCase):
    def test_binary_metrics_are_correct(self):
        precision, recall, f1 = compute_binary_metrics([1, 0, 1, 1], [1, 1, 0, 1])
        self.assertAlmostEqual(precision, 0.6666666667, places=5)
        self.assertAlmostEqual(recall, 0.6666666667, places=5)
        self.assertAlmostEqual(f1, 0.6666666667, places=5)

    def test_calibration_metrics_return_expected_values(self):
        metrics = compute_calibration_metrics([0.9, 0.2, 0.8], [1, 0, 1])
        self.assertIn("brier_score", metrics)
        self.assertIn("ece", metrics)
        self.assertGreaterEqual(metrics["brier_score"], 0.0)

    def test_vision_benchmark_returns_expected_keys(self):
        benchmark = VisionBenchmark()
        result = benchmark.evaluate(
            predictions={"detections": [1, 1, 0], "classes": ["dog", "dog", "cat"]},
            ground_truth={"detections": [1, 1, 1], "classes": ["dog", "dog", "cat"]},
        )
        self.assertIn("precision", result.metrics)
        self.assertIn("recall", result.metrics)
        self.assertIn("counting_accuracy", result.metrics)

    def test_audio_benchmark_returns_expected_keys(self):
        benchmark = AudioBenchmark()
        result = benchmark.evaluate(
            predictions=[1, 0, 1],
            targets=[1, 0, 0],
            scores=[0.9, 0.2, 0.8],
        )
        self.assertIn("precision", result.metrics)
        self.assertIn("roc_auc", result.metrics)
        self.assertIn("pr_auc", result.metrics)

    def test_report_generation_creates_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(output_dir=temp_dir)
            metrics = {"accuracy": 0.91, "f1": 0.88}
            report_path = generator.write_markdown_report("demo", metrics)
            json_path = generator.write_json_report("demo", metrics)
            self.assertTrue(os.path.exists(report_path))
            self.assertTrue(os.path.exists(json_path))
            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["experiment_name"], "demo")

    def test_experiment_tracker_persists_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = ExperimentTracker(storage_path=os.path.join(temp_dir, "experiments.json"))
            tracker.log_experiment("exp1", {"accuracy": 0.9}, {"model": "demo"})
            history = tracker.load_history()
            self.assertEqual(len(history), 1)


if __name__ == "__main__":
    unittest.main()
