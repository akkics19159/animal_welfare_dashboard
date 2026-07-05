from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


class CsvExporter:
    def __init__(self, path: str):
        self.path = Path(path)

    def save(self, payload: List[Dict[str, Any]]) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not payload:
            self.path.write_text("", encoding="utf-8")
            return str(self.path)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload[0].keys()))
            writer.writeheader()
            writer.writerows(payload)
        return str(self.path)
