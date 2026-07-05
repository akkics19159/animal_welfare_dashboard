from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


class DatasetValidator:
    def __init__(self):
        self.errors: List[str] = []

    def validate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.errors = []
        for sample in samples:
            path = sample.get("path")
            if path and not Path(path).exists():
                self.errors.append(f"missing file: {path}")
            if sample.get("media_type") in {"video", "audio", "image"} and not sample.get("path"):
                self.errors.append("media sample missing path")
            if sample.get("annotations") is None:
                self.errors.append(f"annotation missing for {sample.get('id', 'unknown')}")
            if sample.get("timestamp") is None:
                self.errors.append(f"timestamp missing for {sample.get('id', 'unknown')}")
        return {"valid": not self.errors, "errors": self.errors}
