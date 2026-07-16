from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BehaviourModelMetadata:
    name: str
    version: str = "1.0"
    priority: int = 0


@dataclass
class BehaviourObservation:
    track_id: int
    species: str
    behaviour: str
    behaviour_probability: float
    behaviour_confidence: float
    behaviour_duration: float
    behaviour_history: List[str] = field(default_factory=list)
    behaviour_transition: str = "steady"
    behaviour_stability: float = 0.0


class BehaviourRecognizerPlugin(ABC):
    def __init__(self, metadata: Optional[BehaviourModelMetadata] = None):
        self.metadata = metadata or BehaviourModelMetadata(name=self.__class__.__name__)

    @abstractmethod
    def available(self) -> bool:
        pass

    @abstractmethod
    def predict(self, track: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        pass
