from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


class CsvImporter:
    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]
