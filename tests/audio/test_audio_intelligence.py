import unittest
import numpy as np

from audio_intelligence.config import AudioProcessingConfig
from audio_intelligence.preprocessing import AudioPreprocessor
from audio_intelligence.vad import VoiceActivityDetector
from audio_intelligence.segmentation import AudioSegmenter
from audio_intelligence.models import RuleBasedSoundClassifier, RuleBasedDistressClassifier, VocalizationFilter, TemporalAudioAnalyzer


class AudioIntelligenceTests(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.signal = np.sin(2 * np.pi * 440 * np.arange(self.sr) / self.sr).astype(np.float32)
        self.noisy_signal = np.concatenate([
            np.zeros(2000, dtype=np.float32),
            self.signal[2000:6000],
            np.zeros(2000, dtype=np.float32),
        ])

    def test_preprocessing_normalizes_and_reduces_silence(self):
        cfg = AudioProcessingConfig(target_sample_rate=16000, target_dbfs=-20.0)
        preprocessor = AudioPreprocessor(cfg)
        processed = preprocessor.preprocess(self.noisy_signal, self.sr)
        self.assertEqual(processed.sample_rate, 16000)
        self.assertGreater(processed.signal.size, 0)
        self.assertGreater(np.max(np.abs(processed.signal)), 0.0)

    def test_vad_detects_voice_activity(self):
        detector = VoiceActivityDetector()
        segments = detector.detect(self.signal, self.sr)
        self.assertTrue(len(segments) > 0)
        self.assertTrue(all(end > start for start, end in segments))

    def test_segmentation_returns_overlapping_windows(self):
        segmenter = AudioSegmenter(window_duration=0.1, overlap=0.5)
        segments = segmenter.segment(self.signal, self.sr)
        self.assertTrue(len(segments) >= 2)
        self.assertTrue(all(seg.duration > 0 for seg in segments))

    def test_sound_classification_returns_predictions(self):
        classifier = RuleBasedSoundClassifier()
        predictions = classifier.classify(self.signal, self.sr)
        self.assertTrue(len(predictions) > 0)
        self.assertTrue(any(pred["label"] for pred in predictions))

    def test_vocalization_filter_rejects_non_distress_calls(self):
        filter_service = VocalizationFilter()
        result = filter_service.filter([{"label": "speech"}, {"label": "barking"}], [])
        self.assertFalse(result["accepted"])
        self.assertTrue(len(result["reasons"]) > 0)

    def test_distress_classifier_returns_probabilities(self):
        classifier = RuleBasedDistressClassifier()
        result = classifier.classify(self.signal, self.sr)
        self.assertIn("distress_probability", result)
        self.assertIn("confidence", result)
        self.assertGreaterEqual(result["confidence"], 0.0)

    def test_temporal_analysis_identifies_pattern(self):
        analyzer = TemporalAudioAnalyzer()
        probs = [0.1, 0.8, 0.9, 0.85, 0.8]
        summary = analyzer.analyze(probs)
        self.assertIn("pattern", summary)
        self.assertTrue(summary["pattern"])


if __name__ == "__main__":
    unittest.main()
