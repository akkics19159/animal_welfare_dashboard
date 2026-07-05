import os
import tempfile
import unittest

from dataset.registry.dataset_registry import DatasetRegistry
from dataset.versioning.versioning import DatasetVersion
from dataset.validation.validator import DatasetValidator
from dataset.labeling.labels import LabelManager
from dataset.augmentation.vision import VisionAugmenter
from dataset.augmentation.audio import AudioAugmenter
from dataset.augmentation.sensors import SensorAugmenter
from training.config.training_config import TrainingConfig
from training.pipelines.vision import VisionTrainingPipeline
from training.registry.model_registry import ModelRegistry
from training.tracking.experiment_tracker import ExperimentTracker


class DatasetTrainingFrameworkTests(unittest.TestCase):
    def test_dataset_registry_registers_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = DatasetRegistry(storage_path=os.path.join(temp_dir, "datasets.json"))
            entry = registry.register("ds1", "v1", {"species": "dog"})
            self.assertEqual(entry["dataset_id"], "ds1")

    def test_dataset_version_serializes(self):
        version = DatasetVersion(dataset_id="ds1", version="v1", description="demo", species="dog", num_samples=10)
        payload = version.to_dict()
        self.assertEqual(payload["dataset_id"], "ds1")
        self.assertEqual(payload["num_samples"], 10)

    def test_validator_flags_missing_files(self):
        validator = DatasetValidator()
        result = validator.validate([{"id": "1", "path": "missing.mp4", "media_type": "video", "annotations": {}, "timestamp": 0.0}])
        self.assertFalse(result["valid"])

    def test_label_manager_supports_multilabel(self):
        manager = LabelManager()
        self.assertTrue(manager.supports_multi_label())
        self.assertTrue(manager.validate_annotation({"distress": ["yes"], "species": "dog"}))

    def test_augmenters_return_augmented_samples(self):
        self.assertIn("augmentations", VisionAugmenter().augment({"id": "1"}))
        self.assertIn("augmentations", AudioAugmenter().augment({"id": "1"}))
        self.assertIn("augmentations", SensorAugmenter().augment({"id": "1"}))

    def test_training_config_and_pipeline(self):
        config = TrainingConfig(model_name="demo", architecture="cnn", epochs=3)
        pipeline = VisionTrainingPipeline(config=config)
        result = pipeline.train(dataset=None)
        self.assertEqual(result["status"], "trained")

    def test_model_registry_and_experiment_tracker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(storage_path=os.path.join(temp_dir, "models.json"))
            tracker = ExperimentTracker(storage_path=os.path.join(temp_dir, "experiments.json"))
            entry = registry.register("m1", {"architecture": "demo"})
            logged = tracker.log("exp1", {"accuracy": 0.9}, {"seed": 42})
            self.assertEqual(entry["model_version"], "m1")
            self.assertEqual(logged["experiment_id"], "exp1")


if __name__ == "__main__":
    unittest.main()
