from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .types import FusionInput, FusionOutput


class FusionStrategy(ABC):
    """Plugin interface for multimodal fusion strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fuse(self, fusion_input: FusionInput) -> FusionOutput:
        """Fuse a standardized multimodal input into a unified output."""


class FusionPluginRegistry:
    """Small registry for selecting fusion strategies by configuration."""

    def __init__(self):
        self._registry: Dict[str, FusionStrategy] = {}

    def register(self, name: str, strategy: FusionStrategy) -> None:
        self._registry[name] = strategy

    def get(self, name: str) -> Optional[FusionStrategy]:
        return self._registry.get(name)
