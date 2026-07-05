from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AudioProcessingConfig:
    """Configuration for preprocessing and segmentation."""

    target_sample_rate: int = 16000
    target_dbfs: float = -20.0
    low_cutoff_hz: float = 80.0
    high_cutoff_hz: float = 8000.0
    silence_threshold_db: float = -40.0
    vad_energy_threshold: float = 0.01
    vad_min_duration: float = 0.05
    window_duration: float = 0.5
    overlap: float = 0.5
    min_segment_duration: float = 0.1
    noise_floor_db: float = -50.0
    enable_noise_reduction: bool = True
    enable_vad: bool = True
    enable_silence_removal: bool = True
    log_level: str = "INFO"
    rejected_sound_labels: tuple[str, ...] = (
        "speech",
        "music",
        "vehicle",
        "wind",
        "rain",
        "silence",
        "background_noise",
    )
    non_distress_labels: tuple[str, ...] = (
        "speech",
        "music",
        "vehicle",
        "wind",
        "rain",
        "feeding",
        "play",
        "social_bonding",
    )
    species_priority: tuple[str, ...] = (
        "dog",
        "cat",
        "cow",
        "horse",
        "goat",
        "sheep",
        "bird",
        "human",
    )


@dataclass
class AudioPipelineConfig:
    """Configuration for the end-to-end audio intelligence pipeline."""

    processing: AudioProcessingConfig = field(default_factory=AudioProcessingConfig)
    sound_classifier_name: str = "rule-based"
    species_classifier_name: str = "rule-based"
    distress_classifier_name: str = "rule-based"
    enable_temporal_analysis: bool = True
    enable_vocalization_filter: bool = True
    enable_feature_logging: bool = False
    logger_name: str = "audio_intelligence"
    model_cache_dir: Optional[str] = None
