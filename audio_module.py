import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional

import librosa
import numpy as np
import sounddevice as sd

try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except Exception:
    MOVIEPY_AVAILABLE = False

from audio_intelligence.config import AudioPipelineConfig, AudioProcessingConfig
from audio_intelligence.ingestion import AudioIngestionService
from audio_intelligence.pipeline import AudioIntelligencePipeline

logger = logging.getLogger(__name__)


def ffmpeg_available():
    """Return path to ffmpeg if available on PATH, otherwise None."""
    return shutil.which("ffmpeg")


def _compute_acoustic_features(signal: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    energy = float(np.mean(librosa.feature.rms(y=signal)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=signal)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=sample_rate)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)))
    return {
        "mfcc": mfcc,
        "energy": energy,
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
    }


def _run_pipeline(signal: np.ndarray, sample_rate: int, source_type: str = "microphone", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ingestion = AudioIngestionService(logger_instance=logger)
    pipeline = AudioIntelligencePipeline(
        config=AudioPipelineConfig(processing=AudioProcessingConfig(target_sample_rate=sample_rate)),
        logger_instance=logger,
    )
    from audio_intelligence.ingestion import AudioInput, AudioMetadata

    audio_input = AudioInput(
        signal=signal.astype(np.float32),
        sample_rate=sample_rate,
        metadata=AudioMetadata(source_type=source_type, sample_rate=sample_rate, channels=1, duration=float(len(signal)) / sample_rate, bit_depth=16, format="wav", valid=True),
    )
    result = pipeline.analyze(audio_input)
    feature_dict = _compute_acoustic_features(signal, sample_rate)
    feature_dict.update(
        {
            "distress_probability": float(result.summary.distress_probability),
            "emotion_probabilities": result.feature_vector.emotion_probabilities[-1] if result.feature_vector.emotion_probabilities else {},
            "audio_confidence": float(result.summary.confidence),
            "species": result.summary.detected_species,
            "sound_classes": result.summary.detected_sounds,
            "filtered_sounds": result.summary.filtered_sounds,
            "temporal_pattern": result.summary.temporal_pattern,
            "processing_latency_ms": float(result.processing_latency_ms),
            "segments": result.raw_segments,
            "feature_embeddings": result.feature_vector.feature_embeddings,
            "distress": result.summary.distress_probability >= 0.4,
            "audio_distress": result.summary.distress_probability >= 0.4,
        }
    )
    return feature_dict


def capture_audio(duration=5, sr=22050):
    """Record audio and return a structured summary for distress detection."""
    try:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()
    except Exception as exc:
        raise RuntimeError(f"Audio capture failed: {exc}") from exc

    audio = audio.flatten().astype(np.float32)
    if audio.size == 0:
        raise ValueError("No audio captured from the microphone")
    result = _run_pipeline(audio, sr, source_type="microphone")
    result.update({
        "duration": duration,
        "sr": sr,
    })
    return result


def detect_distress(audio_features, energy_threshold=0.02, zcr_threshold=0.12,
                    centroid_threshold=1500.0, bandwidth_threshold=1200.0,
                    distress_score_threshold=0.4):
    """Estimate distress from audio using traditional acoustic cues and the new pipeline outputs."""
    if "distress_probability" in audio_features:
        probability = float(audio_features.get("distress_probability", 0.0))
        distress = probability >= distress_score_threshold
        return {
            "distress": distress,
            "audio_distress": distress,
            "score": probability,
            "energy": float(audio_features.get("energy", 0.0)),
            "zcr": float(audio_features.get("zcr", 0.0)),
            "centroid": float(audio_features.get("centroid", 0.0)),
            "bandwidth": float(audio_features.get("bandwidth", 0.0)),
            "audio_confidence": float(audio_features.get("audio_confidence", 0.0)),
            "species": audio_features.get("species", []),
            "sound_classes": audio_features.get("sound_classes", []),
            "filtered_sounds": audio_features.get("filtered_sounds", []),
            "temporal_pattern": audio_features.get("temporal_pattern", "continuous_normal_behaviour"),
            "processing_latency_ms": float(audio_features.get("processing_latency_ms", 0.0)),
        }

    energy = float(audio_features.get("energy", 0.0))
    zcr = float(audio_features.get("zcr", 0.0))
    centroid = float(audio_features.get("centroid", 0.0))
    bandwidth = float(audio_features.get("bandwidth", 0.0))

    score = 0.0
    if energy > energy_threshold:
        score += 0.35
    if zcr > zcr_threshold:
        score += 0.25
    if centroid > centroid_threshold:
        score += 0.2
    if bandwidth > bandwidth_threshold:
        score += 0.2

    distress = score >= distress_score_threshold
    return {
        "distress": distress,
        "audio_distress": distress,
        "score": score,
        "energy": energy,
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
    }


def extract_audio_file_features(file_path, sr=22050):
    """Extract audio features and a structured audio intelligence summary from a file or video."""
    ingestion = AudioIngestionService(logger_instance=logger)
    audio_input = ingestion.load_file(file_path, sample_rate=sr)
    feature_dict = _run_pipeline(audio_input.signal, audio_input.sample_rate, source_type=audio_input.metadata.source_type)
    feature_dict.update({"duration": float(len(audio_input.signal)) / audio_input.sample_rate, "sr": audio_input.sample_rate})
    return feature_dict


def save_loudest_audio_segment(file_path, out_wav_path, sr=22050, segment_duration=3.0):
    """Extract the loudest segment of `segment_duration` seconds from file_path and save to out_wav_path."""
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
    except Exception:
        if MOVIEPY_AVAILABLE:
            try:
                clip = VideoFileClip(file_path)
                clip.audio.write_audiofile(out_wav_path, logger=None)
                return out_wav_path
            except Exception:
                pass

        ff = ffmpeg_available()
        if ff:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                cmd = [ff, '-y', '-i', file_path, '-ar', str(sr), '-ac', '1', tmp_path]
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                y, _ = librosa.load(tmp_path, sr=sr, mono=True)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            except Exception as exc:
                logger.exception("ffmpeg conversion failed in save_loudest_audio_segment")
                raise RuntimeError("Failed to load audio and ffmpeg conversion failed") from exc
        else:
            raise RuntimeError("Failed to load audio and no fallback available (moviepy or ffmpeg)")

    hop_length = 512
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length, n_fft=frame_length)
    window_frames = int(np.round(segment_duration * sr / hop_length)) if hop_length > 0 else 1
    if window_frames <= 0:
        window_frames = 1

    if len(rms) <= window_frames:
        start_time = 0.0
        end_time = min(segment_duration, librosa.get_duration(y=y, sr=sr))
    else:
        max_idx = int(np.argmax(np.convolve(rms, np.ones(window_frames), mode='valid')))
        start_time = max(0.0, times[max_idx])
        end_time = start_time + segment_duration

    ff = ffmpeg_available()
    if ff:
        try:
            cmd = [ff, '-y', '-i', file_path, '-ss', str(start_time), '-t', str(segment_duration), '-ar', str(sr), '-ac', '1', '-acodec', 'pcm_s16le', out_wav_path]
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out_wav_path
        except Exception:
            logger.exception("ffmpeg trim failed; falling back to moviepy if available")

    if MOVIEPY_AVAILABLE:
        clip = VideoFileClip(file_path).subclip(start_time, min(end_time, librosa.get_duration(filename=file_path)))
        clip.audio.write_audiofile(out_wav_path, logger=None)
        return out_wav_path

    raise RuntimeError("moviepy or ffmpeg required to save audio segments but neither is available")
