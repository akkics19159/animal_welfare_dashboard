from __future__ import annotations

import random
from typing import Any, Dict, List


class AudioAugmenter:
    def __init__(self):
        self.operations = ["pitch_shift", "time_stretch", "background_noise", "gain", "spec_augment"]

    def augment(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        augmented = dict(sample)
        augmented["augmentations"] = random.sample(self.operations, k=min(2, len(self.operations)))
        return augmented
