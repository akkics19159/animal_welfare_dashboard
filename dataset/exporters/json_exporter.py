from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class JsonExporter:
    def __init__(self, path: str):
        self.path = Path(path)

    def save(self, payload: List[Dict[str, Any]]) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(self.path)
