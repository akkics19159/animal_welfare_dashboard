"""Modern audio intelligence subsystem for multimodal welfare monitoring."""

from .config import AudioProcessingConfig, AudioPipelineConfig
from .interfaces import BaseSoundClassifier, BaseSpeciesClassifier, BaseDistressClassifier, AudioPluginRegistry
from .ingestion import AudioIngestionService, AudioMetadata, AudioInput
from .preprocessing import AudioPreprocessor, ProcessedAudio
from .vad import VoiceActivityDetector
from .segmentation import AudioSegmenter, AudioSegment
from .features import AudioFeatureExtractor, AudioFeatures
from .models import RuleBasedSoundClassifier, RuleBasedSpeciesClassifier, RuleBasedDistressClassifier, VocalizationFilter, TemporalAudioAnalyzer
from .pipeline import AudioIntelligencePipeline, AudioAnalysisResult, AudioFeatureVector, AudioSummary

__all__ = [
    "AudioProcessingConfig",
    "AudioPipelineConfig",
    "BaseSoundClassifier",
    "BaseSpeciesClassifier",
    "BaseDistressClassifier",
    "AudioPluginRegistry",
    "AudioIngestionService",
    "AudioMetadata",
    "AudioInput",
    "AudioPreprocessor",
    "ProcessedAudio",
    "VoiceActivityDetector",
    "AudioSegmenter",
    "AudioSegment",
    "AudioFeatureExtractor",
    "AudioFeatures",
    "RuleBasedSoundClassifier",
    "RuleBasedSpeciesClassifier",
    "RuleBasedDistressClassifier",
    "VocalizationFilter",
    "TemporalAudioAnalyzer",
    "AudioIntelligencePipeline",
    "AudioAnalysisResult",
    "AudioFeatureVector",
    "AudioSummary",
]
