from __future__ import annotations

from typing import Any, Dict, List, Optional


class LabelManager:
    def __init__(self):
        self.supported_labels = [
            "species",
            "behaviour",
            "distress",
            "pain",
            "fear",
            "aggression",
            "occupancy",
            "pose",
            "health",
            "environment",
        ]

    def create_label(self, name: str, values: Optional[List[str]] = None) -> Dict[str, Any]:
        return {"name": name, "values": values or []}

    def validate_annotation(self, annotation: Dict[str, Any]) -> bool:
        if not annotation:
            return False
        for key, value in annotation.items():
            if key not in self.supported_labels:
                return False
        return True

    def supports_multi_label(self) -> bool:
        return True
