from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .types import ReasoningInput, ReasoningResult


class ReasoningPlugin(ABC):
    """Extension point for future reasoning backends such as OWL, probabilistic graphs, or LLM explanation."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, input_data: ReasoningInput, state: Dict[str, Any]) -> ReasoningResult:
        raise NotImplementedError


class ReasoningPluginRegistry:
    def __init__(self):
        self._registry: Dict[str, ReasoningPlugin] = {}

    def register(self, name: str, plugin: ReasoningPlugin) -> None:
        self._registry[name] = plugin

    def get(self, name: str) -> Optional[ReasoningPlugin]:
        return self._registry.get(name)
