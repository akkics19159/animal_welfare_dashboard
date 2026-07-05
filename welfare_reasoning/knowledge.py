from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class KnowledgeBase:
    """Configuration-driven knowledge base for welfare reasoning."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        if config is None:
            config_path = config_path or str(Path(__file__).resolve().parent / "config" / "knowledge_base.json")
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        self.config = config

    def get_species_profile(self, species: str) -> Dict[str, Any]:
        return self.config.get("species_profiles", {}).get(species.lower(), {})

    def get_behaviour(self, behaviour: str) -> Dict[str, Any]:
        return self.config.get("behaviour_library", {}).get(behaviour.lower(), {})

    def get_indicator(self, indicator: str) -> Dict[str, Any]:
        return self.config.get("welfare_indicators", {}).get(indicator.lower(), {})

    def get_sensor_rules(self) -> List[Dict[str, Any]]:
        return self.config.get("sensor_rules", [])

    def get_audio_rules(self) -> List[Dict[str, Any]]:
        return self.config.get("audio_rules", [])

    def get_contextual_knowledge(self) -> List[Dict[str, Any]]:
        return self.config.get("contextual_knowledge", [])

    def get_reasoning_rules(self) -> List[Dict[str, Any]]:
        return self.config.get("reasoning_rules", [])

    def get_constraints(self) -> List[Dict[str, Any]]:
        return self.config.get("constraints", [])
