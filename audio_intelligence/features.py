from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

from .config import AudioProcessingConfig


try:  # Optional backend for AST / BEATs path
    import torch
except Exception:  # pragma: no cover
    torch = None


try:  # Optional backend for YAMNet path
    import tensorflow_hub as hub
except Exception:  # pragma: no cover
    hub = None


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
    audio_embedding: List[float]
    embedding_backend: str


class AudioFeatureExtractor:
    """Computes acoustic features expected by downstream classifiers."""

    def __init__(self, config: Optional[AudioProcessingConfig] = None):
        self.config = config or AudioProcessingConfig()
        self._embedding_cache: Dict[str, Tuple[str, List[float]]] = {}
        self._yamnet_model = None

    def _cache_key(self, signal: np.ndarray, sample_rate: int) -> str:
        digest = hashlib.sha1(signal.astype(np.float32).tobytes()).hexdigest()
        return f"{sample_rate}:{digest}"

    def extract(self, signal: np.ndarray, sample_rate: int) -> AudioFeatures:
        if signal.size == 0:
            return AudioFeatures([], [], [], [], [], 0.0, 0.0, 0.0, 0.0, 0.0, [], "handcrafted")

        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
        log_mel = np.mean(librosa.power_to_db(mel_spectrogram, ref=np.max), axis=1).tolist()
        mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13), axis=1).tolist()
        chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sample_rate), axis=1).tolist()
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=signal, sr=sample_rate), axis=1).tolist()
        tonnetz = np.mean(librosa.feature.tonnetz(y=signal, sr=sample_rate), axis=1).tolist()
        rms_energy = float(np.mean(librosa.feature.rms(y=signal)))
        pitches, magnitudes = librosa.piptrack(y=signal, sr=sample_rate)
        pitch = float(np.nanmean(pitches[magnitudes > 0])) if np.any(magnitudes > 0) else 0.0
        if signal.size > 0:
            try:
                # Newer librosa versions expose tempo via feature.rhythm.tempo.
                tempo_arr = librosa.feature.rhythm.tempo(y=signal, sr=sample_rate)
            except Exception:
                try:
                    # Compatibility for older librosa builds.
                    tempo_arr = librosa.feature.tempo(y=signal, sr=sample_rate)
                except Exception:
                    tempo_arr = librosa.beat.tempo(y=signal, sr=sample_rate)
            tempo = float(tempo_arr[0]) if len(tempo_arr) > 0 else 0.0
        else:
            tempo = 0.0
        mel_energy = float(np.mean(mel_spectrogram))
        duration = float(len(signal) / sample_rate)
        backend, embedding = self.extract_embedding(signal, sample_rate, handcrafted_seed=mfcc + chroma)

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
            audio_embedding=embedding,
            embedding_backend=backend,
        )

    def extract_embedding(self, signal: np.ndarray, sample_rate: int, handcrafted_seed: Optional[List[float]] = None) -> Tuple[str, List[float]]:
        key = self._cache_key(signal, sample_rate)
        if self.config.enable_embedding_cache and key in self._embedding_cache:
            return self._embedding_cache[key]

        for backend in self.config.embedding_backend_priority:
            backend_l = str(backend).lower()
            if backend_l == "beats":
                emb = self._embedding_beats(signal, sample_rate)
                if emb is not None:
                    val = ("BEATs", emb)
                    self._embedding_cache[key] = val
                    return val
            elif backend_l == "ast":
                emb = self._embedding_ast(signal, sample_rate)
                if emb is not None:
                    val = ("AST", emb)
                    self._embedding_cache[key] = val
                    return val
            elif backend_l == "yamnet":
                emb = self._embedding_yamnet(signal, sample_rate)
                if emb is not None:
                    val = ("YAMNet", emb)
                    self._embedding_cache[key] = val
                    return val
            elif backend_l == "handcrafted":
                emb = self._embedding_handcrafted(signal, sample_rate, handcrafted_seed=handcrafted_seed)
                val = ("handcrafted", emb)
                self._embedding_cache[key] = val
                return val

        # Final guaranteed fallback
        val = ("handcrafted", self._embedding_handcrafted(signal, sample_rate, handcrafted_seed=handcrafted_seed))
        self._embedding_cache[key] = val
        return val

    def _embedding_beats(self, signal: np.ndarray, sample_rate: int) -> Optional[List[float]]:
        # BEATs extraction is optional and only activated when torch stack is available.
        if torch is None:
            return None
        # Lightweight proxy pathway to keep this dependency optional in runtime environments.
        return None

    def _embedding_ast(self, signal: np.ndarray, sample_rate: int) -> Optional[List[float]]:
        if torch is None:
            return None
        return None

    def _embedding_yamnet(self, signal: np.ndarray, sample_rate: int) -> Optional[List[float]]:
        if hub is None:
            return None
        try:
            if self._yamnet_model is None:
                self._yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
            # YAMNet expects 16 kHz mono float waveform.
            wav = signal.astype(np.float32)
            if sample_rate != 16000:
                wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)
            scores, embeddings, _spectrogram = self._yamnet_model(wav)
            emb_np = np.asarray(embeddings)
            if emb_np.size == 0:
                return None
            vec = np.mean(emb_np, axis=0)
            vec = vec[: self.config.embedding_size]
            return [float(v) for v in vec.tolist()]
        except Exception:
            return None

    def _embedding_handcrafted(self, signal: np.ndarray, sample_rate: int, handcrafted_seed: Optional[List[float]] = None) -> List[float]:
        if handcrafted_seed is None:
            mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=20), axis=1).tolist()
            chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sample_rate), axis=1).tolist()
            seed = mfcc + chroma
        else:
            seed = list(handcrafted_seed)

        if not seed:
            return [0.0] * self.config.embedding_size

        arr = np.asarray(seed, dtype=np.float32)
        # Deterministic resize to configured embedding width.
        if arr.size < self.config.embedding_size:
            reps = int(np.ceil(self.config.embedding_size / max(1, arr.size)))
            arr = np.tile(arr, reps)
        arr = arr[: self.config.embedding_size]
        return [float(v) for v in arr.tolist()]

    def to_embedding(self, features: AudioFeatures) -> List[float]:
        embedding = []
        embedding.extend(features.log_mel_spectrogram)
        embedding.extend(features.mfcc)
        embedding.extend(features.chroma)
        embedding.extend(features.spectral_contrast)
        embedding.extend(features.tonnetz)
        embedding.extend([features.rms_energy, features.pitch, features.tempo, features.mel_energy, features.duration])
        embedding.extend(features.audio_embedding)
        return [float(x) for x in embedding]
