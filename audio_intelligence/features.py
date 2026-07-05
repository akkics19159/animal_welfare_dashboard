from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import librosa
import numpy as np


@dataclass
class AudioFeatures:
    log_mel_spectrogram: List[float]
    mfcc: List[float]
    chroma: List[float]
    spectral_contrast: List[float]
    tonnetz: List[float]
    rms_energy: float
    pitch: float
    tempo: float
    mel_energy: float
    duration: float


class AudioFeatureExtractor:
    """Computes acoustic features expected by downstream classifiers."""

    def extract(self, signal: np.ndarray, sample_rate: int) -> AudioFeatures:
        if signal.size == 0:
            return AudioFeatures([], [], [], [], [], 0.0, 0.0, 0.0, 0.0, 0.0)

        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
        log_mel = np.mean(librosa.power_to_db(mel_spectrogram, ref=np.max), axis=1).tolist()
        mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13), axis=1).tolist()
        chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sample_rate), axis=1).tolist()
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=signal, sr=sample_rate), axis=1).tolist()
        tonnetz = np.mean(librosa.feature.tonnetz(y=signal, sr=sample_rate), axis=1).tolist()
        rms_energy = float(np.mean(librosa.feature.rms(y=signal)))
        pitches, magnitudes = librosa.piptrack(y=signal, sr=sample_rate)
        pitch = float(np.nanmean(pitches[magnitudes > 0])) if np.any(magnitudes > 0) else 0.0
        tempo = float(librosa.beat.tempo(y=signal, sr=sample_rate)[0]) if signal.size > 0 else 0.0
        mel_energy = float(np.mean(mel_spectrogram))
        duration = float(len(signal) / sample_rate)

        return AudioFeatures(
            log_mel_spectrogram=log_mel,
            mfcc=mfcc,
            chroma=chroma,
            spectral_contrast=spectral_contrast,
            tonnetz=tonnetz,
            rms_energy=rms_energy,
            pitch=pitch,
            tempo=tempo,
            mel_energy=mel_energy,
            duration=duration,
        )

    def to_embedding(self, features: AudioFeatures) -> List[float]:
        embedding = []
        embedding.extend(features.log_mel_spectrogram)
        embedding.extend(features.mfcc)
        embedding.extend(features.chroma)
        embedding.extend(features.spectral_contrast)
        embedding.extend(features.tonnetz)
        embedding.extend([features.rms_energy, features.pitch, features.tempo, features.mel_energy, features.duration])
        return [float(x) for x in embedding]
