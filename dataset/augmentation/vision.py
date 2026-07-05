from __future__ import annotations

import random
from typing import Any, Dict, List


class VisionAugmenter:
    def __init__(self):
        self.operations = ["random_crop", "flip", "brightness", "blur", "noise", "rotation", "occlusion"]

    def augment(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        augmented = dict(sample)
        augmented["augmentations"] = random.sample(self.operations, k=min(2, len(self.operations)))
        return augmented
