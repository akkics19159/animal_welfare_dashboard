from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import AudioProcessingConfig


@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    signal: np.ndarray
    duration: float
    signal_quality: float
    species_confidence: float = 0.0


class AudioSegmenter:
    """Creates overlapping windows for temporal audio analysis."""

    def __init__(self, config: Optional[AudioProcessingConfig] = None, window_duration: float = 0.5, overlap: float = 0.5):
        self.config = config or AudioProcessingConfig()
        self.window_duration = window_duration or self.config.window_duration
        self.overlap = overlap if overlap >= 0 else self.config.overlap

    def segment(self, signal: np.ndarray, sample_rate: int) -> List[AudioSegment]:
        if signal.size == 0:
            return []

        window_samples = max(1, int(self.window_duration * sample_rate))
        hop_samples = max(1, int(window_samples * (1 - self.overlap)))
        segments: List[AudioSegment] = []
        for start in range(0, len(signal) - window_samples + 1, hop_samples):
            chunk = signal[start:start + window_samples]
            duration = len(chunk) / sample_rate
            if duration < self.config.min_segment_duration:
                continue
            quality = float(np.sqrt(np.mean(np.square(chunk))))
            segments.append(
                AudioSegment(
                    start_time=start / sample_rate,
                    end_time=(start + len(chunk)) / sample_rate,
                    signal=chunk.astype(np.float32),
                    duration=duration,
                    signal_quality=min(1.0, quality / 0.2),
                )
            )
        return segments
