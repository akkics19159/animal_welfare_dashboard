from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class ExperimentTracker:
    def __init__(self, storage_path: str | None = None):
        self.storage_path = Path(storage_path or "evaluation/experiments/history.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def log_experiment(self, name: str, metrics: Dict[str, Any], config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        entry = {
            "name": name,
            "metrics": metrics,
            "config": config or {},
        }
        history = self.load_history()
        history.append(entry)
        self.storage_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return entry

    def load_history(self) -> List[Dict[str, Any]]:
        if not self.storage_path.exists():
            return []
        return json.loads(self.storage_path.read_text(encoding="utf-8"))
