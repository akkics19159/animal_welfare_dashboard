from __future__ import annotations

from typing import List, Tuple

import numpy as np


class VoiceActivityDetector:
    """Simple energy-based VAD that ignores silence and background noise."""

    def __init__(self, threshold: float = 0.01, min_duration: float = 0.05):
        self.threshold = threshold
        self.min_duration = min_duration

    def detect(self, signal: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        if signal.size == 0:
            return []
        frame_size = max(1, int(0.02 * sample_rate))
        hop_size = max(1, int(0.01 * sample_rate))
        energies = []
        for start in range(0, len(signal) - frame_size + 1, hop_size):
            frame = signal[start:start + frame_size]
            energy = float(np.sqrt(np.mean(np.square(frame))))
            energies.append((start, energy))

        active = []
        for start, energy in energies:
            if energy >= self.threshold:
                active.append(start)

        if not active:
            return []

        segments = []
        start = active[0]
        current = active[0]
        for idx in active[1:]:
            if idx - current <= hop_size:
                current = idx
            else:
                segments.append((start, current))
                start = idx
                current = idx
        segments.append((start, current))

        result = []
        for s, e in segments:
            duration = (e - s) / sample_rate
            if duration >= self.min_duration:
                result.append((s / sample_rate, e / sample_rate))
        return result
