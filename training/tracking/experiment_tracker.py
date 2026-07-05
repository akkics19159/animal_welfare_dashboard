from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentTracker:
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "training/tracking/experiments.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, experiment_id: str, metrics: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        history = self.load_history()
        entry = {"experiment_id": experiment_id, "metrics": metrics, "config": config}
        history.append(entry)
        self.storage_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return entry

    def load_history(self) -> List[Dict[str, Any]]:
        if not self.storage_path.exists():
            return []
        return json.loads(self.storage_path.read_text(encoding="utf-8"))
