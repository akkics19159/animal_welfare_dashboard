from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EvaluationDataset:
    name: str
    version: str
    samples: List[Dict[str, Any]]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("dataset must contain at least one sample")
