from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf


logger = logging.getLogger(__name__)


@dataclass
class AudioMetadata:
    source_type: str
    file_path: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration: Optional[float] = None
    bit_depth: Optional[int] = None
    format: Optional[str] = None
    valid: bool = True


@dataclass
class AudioInput:
    signal: np.ndarray
    sample_rate: int
    metadata: AudioMetadata


class AudioIngestionService:
    """Ingests audio from files, videos, or microphones and validates metadata."""

    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        self.logger = logger_instance or logger

    def capture_microphone(self, duration: float = 3.0, sample_rate: int = 22050) -> AudioInput:
        try:
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError(f"Audio capture failed: {exc}") from exc

        signal = audio.flatten().astype(np.float32)
        if signal.size == 0:
            raise ValueError("No audio captured from microphone")
        metadata = AudioMetadata(
            source_type="microphone",
            sample_rate=sample_rate,
            channels=1,
            duration=float(len(signal)) / sample_rate,
            bit_depth=16,
            format="wav",
            valid=True,
        )
        return AudioInput(signal=signal, sample_rate=sample_rate, metadata=metadata)

    def load_file(self, file_path: str, sample_rate: Optional[int] = None) -> AudioInput:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(file_path)

        suffix = path.suffix.lower()
        if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4a"}:
            signal, sr = self._load_video_audio(path, sample_rate=sample_rate)
            metadata = self._build_metadata(path, signal, sr, source_type="video")
            return AudioInput(signal=signal, sample_rate=sr, metadata=metadata)

        signal, sr = librosa.load(str(path), sr=sample_rate, mono=True)
        metadata = self._build_metadata(path, signal, sr, source_type="file")
        return AudioInput(signal=signal, sample_rate=sr, metadata=metadata)

    def validate_metadata(self, metadata: AudioMetadata) -> AudioMetadata:
        if metadata.sample_rate is None or metadata.sample_rate <= 0:
            metadata.valid = False
            metadata.sample_rate = 16000
        if metadata.channels is None or metadata.channels <= 0:
            metadata.valid = False
            metadata.channels = 1
        if metadata.duration is None or metadata.duration <= 0:
            metadata.valid = False
        if metadata.bit_depth is None:
            metadata.bit_depth = 16
        return metadata

    def _build_metadata(self, path: Path, signal: np.ndarray, sample_rate: int, source_type: str) -> AudioMetadata:
        try:
            info = sf.info(str(path))
            channels = int(getattr(info, "channels", 1))
            bit_depth = self._infer_bit_depth(info.subtype if hasattr(info, "subtype") else None)
        except Exception:
            channels = 1
            bit_depth = 16
        metadata = AudioMetadata(
            source_type=source_type,
            file_path=str(path),
            sample_rate=int(sample_rate),
            channels=channels,
            duration=float(len(signal)) / sample_rate,
            bit_depth=bit_depth,
            format=path.suffix.lower().lstrip("."),
            valid=True,
        )
        return self.validate_metadata(metadata)

    def _infer_bit_depth(self, subtype: Optional[str]) -> int:
        if not subtype:
            return 16
        subtype = subtype.upper()
        if "32" in subtype:
            return 32
        if "24" in subtype:
            return 24
        return 16

    def _load_video_audio(self, path: Path, sample_rate: Optional[int] = None) -> tuple[np.ndarray, int]:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            cmd = [ffmpeg, "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate or 16000), tmp_path]
            try:
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                signal, sr = librosa.load(tmp_path, sr=sample_rate, mono=True)
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            return signal.astype(np.float32), int(sr)

        try:
            import moviepy
            from moviepy import VideoFileClip

            clip = VideoFileClip(str(path))
            audio = clip.audio
            if audio is None:
                raise RuntimeError("No audio track found")
            signal, sr = audio.to_soundarray(fps=sample_rate or 16000, nbytes=2)
            return np.asarray(signal, dtype=np.float32), int(sr)
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError(f"Unable to extract audio from video: {exc}") from exc
