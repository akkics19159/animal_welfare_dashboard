from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class JsonImporter:
    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        return [payload]
