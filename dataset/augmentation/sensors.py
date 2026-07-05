from __future__ import annotations

import random
from typing import Any, Dict, List


class SensorAugmenter:
    def __init__(self):
        self.operations = ["gaussian_noise", "interpolation", "temporal_jitter", "missing_value_simulation"]

    def augment(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        augmented = dict(sample)
        augmented["augmentations"] = random.sample(self.operations, k=min(2, len(self.operations)))
        return augmented
