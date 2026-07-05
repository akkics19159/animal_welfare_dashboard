from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import AudioPipelineConfig
from .features import AudioFeatureExtractor, AudioFeatures
from .ingestion import AudioInput, AudioMetadata
from .interfaces import BaseDistressClassifier, BaseSoundClassifier, BaseSpeciesClassifier, AudioPluginRegistry
from .models import RuleBasedDistressClassifier, RuleBasedSoundClassifier, RuleBasedSpeciesClassifier, TemporalAudioAnalyzer, VocalizationFilter
from .preprocessing import AudioPreprocessor, ProcessedAudio
from .segmentation import AudioSegment, AudioSegmenter
from .vad import VoiceActivityDetector


logger = logging.getLogger(__name__)


@dataclass
class AudioFeatureVector:
    timestamps: List[float] = field(default_factory=list)
    species: List[str] = field(default_factory=list)
    general_sound_class: List[str] = field(default_factory=list)
    distress_probability: List[float] = field(default_factory=list)
    emotion_probabilities: List[Dict[str, float]] = field(default_factory=list)
    confidence: List[float] = field(default_factory=list)
    signal_quality: List[float] = field(default_factory=list)
    temporal_descriptors: Dict[str, Any] = field(default_factory=dict)
    feature_embeddings: List[List[float]] = field(default_factory=list)


@dataclass
class AudioSummary:
    detected_species: List[str]
    detected_sounds: List[str]
    filtered_sounds: List[str]
    distress_probability: float
    confidence: float
    temporal_pattern: str
    reasons: List[str] = field(default_factory=list)


@dataclass
class AudioAnalysisResult:
    metadata: AudioMetadata
    processed_audio: Optional[ProcessedAudio]
    segments: List[AudioSegment]
    feature_vector: AudioFeatureVector
    summary: AudioSummary
    processing_latency_ms: float
    raw_segments: List[Dict[str, Any]] = field(default_factory=list)


class AudioIntelligencePipeline:
    """End-to-end audio intelligence pipeline for distress-aware multimodal analysis."""

    def __init__(
        self,
        config: Optional[AudioPipelineConfig] = None,
        preprocessor: Optional[AudioPreprocessor] = None,
        vad: Optional[VoiceActivityDetector] = None,
        segmenter: Optional[AudioSegmenter] = None,
        feature_extractor: Optional[AudioFeatureExtractor] = None,
        sound_classifier: Optional[BaseSoundClassifier] = None,
        species_classifier: Optional[BaseSpeciesClassifier] = None,
        distress_classifier: Optional[BaseDistressClassifier] = None,
        vocalization_filter: Optional[VocalizationFilter] = None,
        temporal_analyzer: Optional[TemporalAudioAnalyzer] = None,
        plugin_registry: Optional[AudioPluginRegistry] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        self.config = config or AudioPipelineConfig()
        self.preprocessor = preprocessor or AudioPreprocessor(self.config.processing)
        self.vad = vad or VoiceActivityDetector(self.config.processing.vad_energy_threshold, self.config.processing.vad_min_duration)
        self.segmenter = segmenter or AudioSegmenter(self.config.processing)
        self.feature_extractor = feature_extractor or AudioFeatureExtractor()
        self.sound_classifier = sound_classifier or RuleBasedSoundClassifier(self.config.processing)
        self.species_classifier = species_classifier or RuleBasedSpeciesClassifier()
        self.distress_classifier = distress_classifier or RuleBasedDistressClassifier(self.config.processing)
        self.vocalization_filter = vocalization_filter or VocalizationFilter(self.config.processing)
        self.temporal_analyzer = temporal_analyzer or TemporalAudioAnalyzer()
        self.plugin_registry = plugin_registry or AudioPluginRegistry()
        self.logger = logger_instance or logger

    def analyze(self, audio_input: AudioInput) -> AudioAnalysisResult:
        start = time.perf_counter()
        processed_audio = self.preprocessor.preprocess(audio_input.signal, audio_input.sample_rate)
        segments = self._build_segments(processed_audio)

        feature_vector = AudioFeatureVector()
        raw_segments = []
        distress_probabilities = []
        sound_labels = []
        filtered_labels = []
        species_labels = []
        confidences = []
        signal_quality = []
        embeddings = []

        for segment in segments:
            features = self.feature_extractor.extract(segment.signal, processed_audio.sample_rate)
            sound_predictions = self.sound_classifier.classify(segment.signal, processed_audio.sample_rate)
            species_predictions = self.species_classifier.classify(segment.signal, processed_audio.sample_rate, sound_predictions=sound_predictions)
            filter_result = self.vocalization_filter.filter(sound_predictions, species_predictions) if self.config.processing.enable_vad else {"accepted": True, "reasons": [], "filtered_labels": []}
            distress_result = self.distress_classifier.classify(segment.signal, processed_audio.sample_rate, sound_predictions=sound_predictions)
            if filter_result.get("accepted", True):
                distress_probabilities.append(distress_result.get("distress_probability", 0.0))
            else:
                distress_probabilities.append(0.0)

            sound_labels.extend([pred.get("label") for pred in sound_predictions])
            filtered_labels.extend(filter_result.get("filtered_labels", []))
            species_labels.extend([pred.get("label") for pred in species_predictions])
            confidences.append(distress_result.get("confidence", 0.0))
            signal_quality.append(segment.signal_quality)
            embeddings.append(self.feature_extractor.to_embedding(features))
            feature_vector.timestamps.append(segment.start_time)
            feature_vector.species.append(species_predictions[0].get("label", "unknown") if species_predictions else "unknown")
            feature_vector.general_sound_class.append(sound_predictions[0].get("label", "silence") if sound_predictions else "silence")
            feature_vector.distress_probability.append(float(distress_result.get("distress_probability", 0.0)))
            feature_vector.emotion_probabilities.append(
                {
                    "distress": distress_result.get("distress_probability", 0.0),
                    "pain": distress_result.get("pain_probability", 0.0),
                    "fear": distress_result.get("fear_probability", 0.0),
                    "panic": distress_result.get("panic_probability", 0.0),
                    "aggression": distress_result.get("aggression_probability", 0.0),
                }
            )
            feature_vector.confidence.append(float(distress_result.get("confidence", 0.0)))
            feature_vector.signal_quality.append(float(segment.signal_quality))
            feature_vector.feature_embeddings.append(self.feature_extractor.to_embedding(features))
            raw_segments.append(
                {
                    "timestamp": segment.start_time,
                    "duration": segment.duration,
                    "sound_class": sound_predictions[0].get("label", "silence") if sound_predictions else "silence",
                    "species": species_predictions[0].get("label", "unknown") if species_predictions else "unknown",
                    "distress_probability": distress_result.get("distress_probability", 0.0),
                    "accepted": filter_result.get("accepted", True),
                    "reasons": filter_result.get("reasons", []),
                    "confidence": distress_result.get("confidence", 0.0),
                }
            )

        temporal_summary = self.temporal_analyzer.analyze(distress_probabilities) if self.config.enable_temporal_analysis else {"pattern": "continuous_normal_behaviour"}
        summary_reasons = []
        for item in raw_segments:
            for reason in item.get("reasons", []):
                summary_reasons.append(reason)

        summary = AudioSummary(
            detected_species=list(dict.fromkeys(species_labels)),
            detected_sounds=list(dict.fromkeys(sound_labels)),
            filtered_sounds=list(dict.fromkeys(filtered_labels)),
            distress_probability=float(np.mean(distress_probabilities)) if distress_probabilities else 0.0,
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            temporal_pattern=temporal_summary.get("pattern", "continuous_normal_behaviour"),
            reasons=summary_reasons,
        )
        feature_vector.temporal_descriptors = temporal_summary
        processing_latency_ms = (time.perf_counter() - start) * 1000.0
        return AudioAnalysisResult(
            metadata=audio_input.metadata,
            processed_audio=processed_audio,
            segments=segments,
            feature_vector=feature_vector,
            summary=summary,
            processing_latency_ms=processing_latency_ms,
            raw_segments=raw_segments,
        )

    def _build_segments(self, processed_audio: ProcessedAudio) -> List[AudioSegment]:
        if self.config.processing.enable_vad:
            vad_regions = self.vad.detect(processed_audio.signal, processed_audio.sample_rate)
            if vad_regions:
                segments = []
                for start, end in vad_regions:
                    chunk = processed_audio.signal[int(start * processed_audio.sample_rate):int(end * processed_audio.sample_rate)]
                    if chunk.size == 0:
                        continue
                    duration = len(chunk) / processed_audio.sample_rate
                    segments.append(
                        AudioSegment(
                            start_time=start,
                            end_time=end,
                            signal=chunk.astype(np.float32),
                            duration=duration,
                            signal_quality=min(1.0, float(np.sqrt(np.mean(np.square(chunk)))) / 0.2),
                        )
                    )
                if segments:
                    return segments
        return self.segmenter.segment(processed_audio.signal, processed_audio.sample_rate)
