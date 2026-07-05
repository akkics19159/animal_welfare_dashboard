from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Any


def save_plot_data(output_path: str, series: Dict[str, List[float]]) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(series, handle, indent=2)
    return str(path)
