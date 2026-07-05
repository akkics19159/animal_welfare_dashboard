from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np

from .config import AudioProcessingConfig


logger = logging.getLogger(__name__)


@dataclass
class ProcessedAudio:
    signal: np.ndarray
    sample_rate: int
    original_sample_rate: int
    quality_score: float
    removed_silence: bool


class AudioPreprocessor:
    """Normalizes, filters, and cleans audio before downstream analysis."""

    def __init__(self, config: Optional[AudioProcessingConfig] = None):
        self.config = config or AudioProcessingConfig()
        self.logger = logger

    def preprocess(self, signal: np.ndarray, sample_rate: int) -> ProcessedAudio:
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)

        if signal.size == 0:
            raise ValueError("Audio signal is empty")

        original_sr = int(sample_rate)
        signal = self.normalize_sample_rate(signal, original_sr, self.config.target_sample_rate)
        signal = self.normalize_loudness(signal)
        if self.config.enable_noise_reduction:
            signal = self.noise_suppression(signal)
        if self.config.enable_silence_removal:
            signal = self.remove_silence(signal, self.config.target_sample_rate)
        signal = self.bandpass_filter(signal, self.config.target_sample_rate)

        quality_score = self._quality_score(signal)
        return ProcessedAudio(
            signal=signal.astype(np.float32),
            sample_rate=self.config.target_sample_rate,
            original_sample_rate=original_sr,
            quality_score=float(quality_score),
            removed_silence=self.config.enable_silence_removal,
        )

    def normalize_sample_rate(self, signal: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
        if sample_rate == target_rate:
            return signal
        return librosa.resample(signal, orig_sr=sample_rate, target_sr=target_rate)

    def normalize_loudness(self, signal: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(np.square(signal)))
        if rms <= 1e-8:
            return signal
        target_rms = 10 ** (self.config.target_dbfs / 20.0)
        return signal * (target_rms / rms)

    def bandpass_filter(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            from scipy.signal import butter, filtfilt
        except Exception:
            return signal

        nyquist = 0.5 * sample_rate
        low = self.config.low_cutoff_hz / nyquist
        high = self.config.high_cutoff_hz / nyquist
        if low <= 0 or high >= 1:
            return signal
        b, a = butter(2, [low, high], btype="bandpass")
        return filtfilt(b, a, signal)

    def noise_suppression(self, signal: np.ndarray) -> np.ndarray:
        if signal.size < 2:
            return signal
        noise_floor = np.percentile(np.abs(signal), 10)
        threshold = max(noise_floor, 1e-4)
        return np.where(np.abs(signal) < threshold, 0.0, signal)

    def remove_silence(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        if signal.size < 2:
            return signal
        intervals = librosa.effects.split(signal, top_db=self.config.silence_threshold_db)
        if len(intervals) == 0:
            return signal
        chunks = [signal[start:end] for start, end in intervals]
        return np.concatenate(chunks).astype(np.float32) if chunks else signal

    def _quality_score(self, signal: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(signal))))
        if rms <= 1e-8:
            return 0.0
        return min(1.0, rms / 0.2)
