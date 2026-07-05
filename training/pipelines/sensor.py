from __future__ import annotations

from typing import Any, Dict, Optional

from training.config.training_config import TrainingConfig


class SensorTrainingPipeline:
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig(model_name="sensor-baseline", architecture="sensor-model")

    def train(self, dataset: Any) -> Dict[str, Any]:
        return {
            "status": "trained",
            "model_name": self.config.model_name,
            "architecture": self.config.architecture,
            "dataset_version": self.config.dataset_version,
            "random_seed": self.config.random_seed,
        }
