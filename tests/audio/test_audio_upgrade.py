from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from audio_intelligence.config import AudioPipelineConfig, AudioProcessingConfig
from audio_intelligence.ingestion import AudioIngestionService, AudioInput, AudioMetadata
from audio_intelligence.models import NonDistressFilter, RuleBasedDistressClassifier
from audio_intelligence.pipeline import AudioIntelligencePipeline
from audio_module import detect_distress


def _tone(sr: int = 16000, freq: float = 440.0, sec: float = 1.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(sr * sec), dtype=np.float32) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def test_microphone_capture_path_with_mock(monkeypatch):
    def _fake_rec(samples, samplerate, channels):
        return np.ones((samples, channels), dtype=np.float32) * 0.02

    monkeypatch.setattr("audio_intelligence.ingestion.sd.rec", _fake_rec)
    monkeypatch.setattr("audio_intelligence.ingestion.sd.wait", lambda: None)

    svc = AudioIngestionService()
    data = svc.capture_microphone(duration=0.5, sample_rate=16000)

    assert data.metadata.source_type == "microphone"
    assert data.signal.size > 0
    assert data.sample_rate == 16000


def test_pipeline_outputs_required_extended_audio_fields():
    sig = _tone(freq=620.0, sec=1.2, amp=0.18)
    pipeline = AudioIntelligencePipeline(
        config=AudioPipelineConfig(processing=AudioProcessingConfig(target_sample_rate=16000))
    )
    inp = AudioInput(
        signal=sig,
        sample_rate=16000,
        metadata=AudioMetadata(source_type="microphone", sample_rate=16000, channels=1, duration=1.2, bit_depth=16, format="wav", valid=True),
    )

    result = pipeline.analyze(inp)
    summary = result.summary

    assert isinstance(summary.audio_embedding, list)
    assert summary.embedding_backend
    assert isinstance(summary.event_type, str)
    assert 0.0 <= summary.distress_probability <= 1.0
    assert 0.0 <= summary.confidence <= 1.0
    assert 0.0 <= summary.audio_quality <= 1.0
    assert 0.0 <= summary.temporal_consistency <= 1.0
    assert 0.0 <= summary.non_distress_probability <= 1.0
    assert summary.event_duration >= 0.0


def test_non_distress_suppression_categories():
    f = NonDistressFilter(AudioProcessingConfig())
    for label in ["mating", "play", "feeding", "grooming"]:
        out = f.evaluate([{"label": label, "confidence": 0.9}], distress_probability=0.45)
        assert out["suppressed"] is True
        assert out["suppressed_reason"] is not None
        assert out["non_distress_probability"] >= 0.55


def test_distress_not_suppressed_when_strong_evidence():
    f = NonDistressFilter(AudioProcessingConfig())
    out = f.evaluate([{"label": "play", "confidence": 0.95}], distress_probability=0.95)
    assert out["suppressed"] is False


def test_detect_distress_contract_contains_new_fields():
    payload = {
        "distress_probability": 0.7,
        "energy": 0.03,
        "zcr": 0.2,
        "centroid": 1900.0,
        "bandwidth": 1400.0,
        "confidence": 0.8,
        "audio_quality": 0.7,
        "temporal_consistency": 0.6,
        "non_distress_probability": 0.2,
        "suppressed_reason": None,
        "event_duration": 1.5,
        "event_type": "distress_vocalization",
        "audio_embedding": [0.1, 0.2, 0.3],
    }
    out = detect_distress(payload)
    for key in [
        "distress",
        "score",
        "distress_probability",
        "confidence",
        "audio_quality",
        "temporal_consistency",
        "non_distress_probability",
        "suppressed_reason",
        "event_duration",
        "event_type",
        "audio_embedding",
    ]:
        assert key in out


def test_live_monitoring_uses_video_audio_fallback_on_microphone_failure(monkeypatch):
    import ui.pages.live_monitoring as lm

    monkeypatch.setattr(lm, "analyze_video_behavior", lambda source=0, confidence=0.25: {"detections": [{"class": "dog", "confidence": 0.9, "xyxy": [1, 2, 10, 20]}], "motion_score": 0.1, "agitated": False, "error": None})
    monkeypatch.setattr(lm, "capture_audio", lambda duration=3: (_ for _ in ()).throw(RuntimeError("mic unavailable")))
    monkeypatch.setattr(lm, "extract_audio_file_features", lambda path: {"distress_probability": 0.33, "energy": 0.01, "zcr": 0.05, "centroid": 1000.0, "bandwidth": 600.0, "confidence": 0.7, "audio_quality": 0.8, "temporal_consistency": 0.6, "non_distress_probability": 0.7, "suppressed_reason": "Suppressed likely non-distress event: play_vocalization", "event_duration": 1.2, "event_type": "play_vocalization", "audio_embedding": [0.1, 0.2]})
    monkeypatch.setattr(lm, "read_sensors", lambda use_simulation=True: {"temp": 25.0, "humidity": 60.0, "heart_rate": 90, "error": None})
    monkeypatch.setattr(lm, "analyze_combined", lambda video_result, audio_result, sensor_result, ontology_strength=0.6, weights=None: {"probability": 0.2, "video_result": video_result, "audio_result": audio_result, "sensor_result": sensor_result})

    cfg = SimpleNamespace(
        use_simulated_sensors=True,
        ontology_strength=0.6,
        video_weight=0.4,
        audio_weight=0.4,
        sensor_weight=0.2,
    )

    res = lm._run_inference(api=None, backend_online=False, cfg=cfg, default_video_path="dummy.mp4")
    audio = res.get("audio_result", {})
    assert "Microphone unavailable" in str(audio.get("error", ""))
    assert "event_type" in audio
    assert "non_distress_probability" in audio
