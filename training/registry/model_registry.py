from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelRegistry:
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "training/registry/models.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def register(self, model_version: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        registry = self.load_registry()
        entry = {"model_version": model_version, **metadata}
        registry[model_version] = entry
        self.storage_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        return entry

    def load_registry(self) -> Dict[str, Any]:
        if not self.storage_path.exists():
            return {}
        return json.loads(self.storage_path.read_text(encoding="utf-8"))
