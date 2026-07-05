from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any


class ReportGenerator:
    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir or "evaluation/reports/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_markdown_report(self, experiment_name: str, metrics: Dict[str, Any]) -> str:
        path = self.output_dir / f"{experiment_name}.md"
        lines = [f"# {experiment_name}", "", "## Metrics", ""]
        lines.extend([f"- {key}: {value}" for key, value in metrics.items()])
        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path)

    def write_csv_report(self, experiment_name: str, metrics: Dict[str, Any]) -> str:
        path = self.output_dir / f"{experiment_name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
            writer.writeheader()
            for key, value in metrics.items():
                writer.writerow({"metric": key, "value": value})
        return str(path)

    def write_json_report(self, experiment_name: str, metrics: Dict[str, Any]) -> str:
        path = self.output_dir / f"{experiment_name}.json"
        payload = {"experiment_name": experiment_name, "metrics": metrics}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)
