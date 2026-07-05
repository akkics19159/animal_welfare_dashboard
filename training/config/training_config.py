from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrainingConfig:
    model_name: str = "baseline"
    architecture: str = "mlp"
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 0.001
    random_seed: int = 42
    dataset_version: str = "v1"
    validation_split: float = 0.2
    test_split: float = 0.1
    cv_folds: int = 3
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
