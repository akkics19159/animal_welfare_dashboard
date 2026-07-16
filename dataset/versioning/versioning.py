from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class DatasetVersion:
    dataset_id: str
    version: str
    creation_date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""
    species: str = "unknown"
    num_samples: int = 0
    annotation_status: str = "pending"
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "creation_date": self.creation_date,
            "description": self.description,
            "species": self.species,
            "num_samples": self.num_samples,
            "annotation_status": self.annotation_status,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }
