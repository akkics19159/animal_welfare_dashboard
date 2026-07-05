from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelMetadata:
    name: str
    version: str = "1.0"
    capabilities: tuple[str, ...] = ()


class BaseSoundClassifier(ABC):
    """Plugin interface for general sound classification."""

    def __init__(self, metadata: Optional[ModelMetadata] = None):
        self.metadata = metadata or ModelMetadata(name=self.__class__.__name__)

    @abstractmethod
    def classify(self, signal: Any, sample_rate: int, **kwargs) -> List[Dict[str, Any]]:
        """Return a list of predictions with label/confidence/timestamp keys."""


class BaseSpeciesClassifier(ABC):
    """Plugin interface for species-aware vocalization analysis."""

    def __init__(self, metadata: Optional[ModelMetadata] = None):
        self.metadata = metadata or ModelMetadata(name=self.__class__.__name__)

    @abstractmethod
    def classify(self, signal: Any, sample_rate: int, sound_predictions: Optional[List[Dict[str, Any]]] = None, **kwargs) -> List[Dict[str, Any]]:
        """Return a list of species predictions."""


class BaseDistressClassifier(ABC):
    """Plugin interface for distress estimation."""

    def __init__(self, metadata: Optional[ModelMetadata] = None):
        self.metadata = metadata or ModelMetadata(name=self.__class__.__name__)

    @abstractmethod
    def classify(self, signal: Any, sample_rate: int, sound_predictions: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Dict[str, Any]:
        """Return distress probabilities and confidence."""


class AudioPluginRegistry:
    """Very small registry for runtime plugin selection."""

    def __init__(self):
        self._registry: Dict[str, Any] = {}

    def register(self, name: str, instance: Any) -> None:
        self._registry[name] = instance

    def get(self, name: str, default: Optional[Any] = None) -> Any:
        return self._registry.get(name, default)
