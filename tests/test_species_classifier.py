from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from vision_intelligence.species_classifier import SpeciesClassifier


class _MockBackend:
    def __init__(self, rows):
        self.rows = rows
        self.priority = 1
        self.name = "mock_backend"

    def available(self):
        return True

    def classify_batch(self, crops_bgr, taxonomy):
        return [list(self.rows) for _ in crops_bgr]


class SpeciesClassifierTests(unittest.TestCase):
    def _mk_classifier(self, rows, threshold=0.42):
        return SpeciesClassifier(
            backends=[_MockBackend(rows)],
            low_confidence_threshold=threshold,
            temporal_window=30,
            species_switch_margin=0.08,
        )

    def _frame(self):
        return np.full((180, 240, 3), 127, dtype=np.uint8)

    def _tracks(self, cls="person", conf=0.55):
        return {
            1: SimpleNamespace(
                bbox_xyxy=[20.0, 20.0, 140.0, 140.0],
                class_name=cls,
                detection_confidence=conf,
                lost_frames=0,
            )
        }

    def test_dog_vs_person_prefers_classifier(self):
        rows = [
            {"species": "Dog", "group": "mammal", "confidence": 0.82, "distance": 0.1},
            {"species": "Human", "group": "human", "confidence": 0.14, "distance": 0.4},
        ]
        clf = self._mk_classifier(rows)
        tracks = self._tracks(cls="person", conf=0.35)
        clf.classify_tracks(frame_bgr=self._frame(), tracks=tracks)
        self.assertEqual(tracks[1].detector_label, "person")
        self.assertEqual(tracks[1].classifier_label, "Dog")
        self.assertEqual(tracks[1].final_species, "Dog")
        self.assertEqual(tracks[1].class_name, "Dog")

    def test_cat_vs_dog(self):
        rows = [
            {"species": "Cat", "group": "mammal", "confidence": 0.77, "distance": 0.2},
            {"species": "Dog", "group": "mammal", "confidence": 0.2, "distance": 0.7},
        ]
        clf = self._mk_classifier(rows)
        tracks = self._tracks(cls="dog", conf=0.5)
        clf.classify_tracks(frame_bgr=self._frame(), tracks=tracks)
        self.assertEqual(tracks[1].classifier_label, "Cat")
        self.assertEqual(tracks[1].class_name, "Cat")

    def test_horse_vs_cow_uses_detector_when_classifier_weak(self):
        rows = [
            {"species": "Cow", "group": "mammal", "confidence": 0.18, "distance": 0.3},
            {"species": "Horse", "group": "mammal", "confidence": 0.16, "distance": 0.35},
        ]
        clf = self._mk_classifier(rows)
        tracks = self._tracks(cls="horse", conf=0.9)
        clf.classify_tracks(frame_bgr=self._frame(), tracks=tracks)
        self.assertEqual(tracks[1].class_name, "Horse")

    def test_bird_vs_chicken(self):
        rows = [
            {"species": "Chicken", "group": "bird", "confidence": 0.71, "distance": 0.2},
            {"species": "Bird", "group": "bird", "confidence": 0.2, "distance": 0.7},
        ]
        clf = self._mk_classifier(rows)
        tracks = self._tracks(cls="bird", conf=0.4)
        clf.classify_tracks(frame_bgr=self._frame(), tracks=tracks)
        self.assertEqual(tracks[1].class_name, "Chicken")

    def test_low_confidence_uses_detector_mapping(self):
        rows = [
            {"species": "Dog", "group": "mammal", "confidence": 0.25, "distance": 0.8},
            {"species": "Cat", "group": "mammal", "confidence": 0.2, "distance": 0.9},
        ]
        clf = self._mk_classifier(rows, threshold=0.5)
        tracks = self._tracks(cls="dog", conf=0.6)
        clf.classify_tracks(frame_bgr=self._frame(), tracks=tracks)
        self.assertEqual(tracks[1].class_name, "Dog")

    def test_person_low_confidence_remains_human(self):
        rows = [
            {"species": "Human", "group": "human", "confidence": 0.2, "distance": 0.9},
            {"species": "Dog", "group": "mammal", "confidence": 0.19, "distance": 0.91},
        ]
        clf = self._mk_classifier(rows, threshold=0.5)
        tracks = self._tracks(cls="person", conf=0.65)
        clf.classify_tracks(frame_bgr=self._frame(), tracks=tracks)
        self.assertEqual(tracks[1].class_name, "Human")

    def test_temporal_smoothing_prevents_oscillation(self):
        frame = self._frame()
        tracks = self._tracks(cls="person", conf=0.55)

        clf = SpeciesClassifier(backends=[_MockBackend([{"species": "Dog", "group": "mammal", "confidence": 0.85, "distance": 0.1}])], temporal_window=30, species_switch_margin=0.10)
        clf.classify_tracks(frame_bgr=frame, tracks=tracks)

        clf._backend = _MockBackend([{"species": "Human", "group": "human", "confidence": 0.56, "distance": 0.12}])
        clf.classify_tracks(frame_bgr=frame, tracks=tracks)

        clf._backend = _MockBackend([{"species": "Dog", "group": "mammal", "confidence": 0.83, "distance": 0.13}])
        clf.classify_tracks(frame_bgr=frame, tracks=tracks)

        # Switch should be resistant unless confidence margin is exceeded.
        self.assertEqual(tracks[1].class_name, "Dog")
        self.assertTrue(len(tracks[1].classification_history) >= 3)

    def test_multiple_tracks_are_classified_independently(self):
        clf = self._mk_classifier(
            [{"species": "Dog", "group": "mammal", "confidence": 0.81, "distance": 0.2}]
        )
        tracks = {
            1: SimpleNamespace(bbox_xyxy=[20.0, 20.0, 80.0, 80.0], class_name="person", detection_confidence=0.4, lost_frames=0),
            2: SimpleNamespace(bbox_xyxy=[100.0, 20.0, 160.0, 80.0], class_name="cat", detection_confidence=0.6, lost_frames=0),
        }
        clf.classify_tracks(frame_bgr=self._frame(), tracks=tracks)

        self.assertEqual(tracks[1].final_species, "Dog")
        self.assertEqual(tracks[2].final_species, "Dog")

    def test_occluded_tracks_skip_reclassification(self):
        clf = self._mk_classifier(
            [{"species": "Cat", "group": "mammal", "confidence": 0.78, "distance": 0.2}]
        )
        tracks = self._tracks(cls="dog", conf=0.6)
        clf.classify_tracks(frame_bgr=self._frame(), tracks=tracks)
        first_species = tracks[1].final_species

        tracks[1].lost_frames = 2
        clf.classify_tracks(frame_bgr=self._frame(), tracks=tracks)
        self.assertEqual(tracks[1].final_species, first_species)

    def test_backend_unavailable_falls_back_to_yolo_mapping(self):
        class _UnavailableBackend:
            name = "broken"
            priority = 1

            def available(self):
                return False

        clf = SpeciesClassifier(backends=[_UnavailableBackend()])
        tracks = self._tracks(cls="person", conf=0.7)
        clf.classify_tracks(frame_bgr=self._frame(), tracks=tracks)
        self.assertEqual(clf.backend_name, "yolo_fallback")
        self.assertEqual(tracks[1].final_species, "Human")

    def test_predict_interface_returns_required_payload(self):
        clf = self._mk_classifier(
            [{"species": "Cow", "group": "mammal", "confidence": 0.84, "distance": 0.1}]
        )
        pred = clf.predict(self._frame(), detector_label="cow", detector_confidence=0.6, track_id=7)
        self.assertEqual(pred.species, "Cow")
        self.assertGreaterEqual(pred.confidence, 0.0)
        self.assertLessEqual(pred.confidence, 1.0)
        self.assertTrue(len(pred.top5_predictions) >= 1)


if __name__ == "__main__":
    unittest.main()
