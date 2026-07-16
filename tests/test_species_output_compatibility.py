from __future__ import annotations

import unittest

from vision_intelligence.vision_output import extend_vision_output


class _T:
    def __init__(self):
        self.class_name = "Dog"
        self.detector_label = "Human"
        self.classifier_label = "Dog"
        self.species_group = "mammal"
        self.species_confidence = 0.82
        self.classifier_confidence = 0.82
        self.classification_confidence = 0.84
        self.temporal_confidence = 0.79
        self.agreement_score = 0.31
        self.embedding_distance = 0.22
        self.top5_predictions = [{"species": "Dog", "confidence": 0.82}]
        self.top_5_predictions = [{"species": "Dog", "confidence": 0.82}]
        self.classification_history = ["Dog", "Dog"]
        self.species_embedding = [0.1, 0.2, 0.3]
        self.bbox_xyxy = [1.0, 2.0, 3.0, 4.0]
        self.tracking_confidence = 0.9
        self.detection_confidence = 0.8
        self.trajectory = []
        self.velocity_xy = (0.0, 0.0)
        self.acceleration_xy = (0.0, 0.0)
        self.direction_deg = None
        self.direction_label = "stationary"
        self.entry_time = None
        self.exit_time = None
        self.dwell_time = 0.0
        self.track_lifetime_sec = 0.0
        self.visibility_score = 1.0
        self.age_frames = 1
        self.lost_frames = 0
        self.was_occluded = False
        self.reacquired = False


class SpeciesOutputCompatibilityTests(unittest.TestCase):
    def test_extend_output_includes_species_metadata(self):
        out = extend_vision_output(
            legacy_video_result={"detections": [{"class": "person", "confidence": 0.8, "xyxy": [1, 2, 3, 4], "track_id": 1}]},
            tracks={1: _T()},
            occupancy=1,
            counting={"current_occupancy": 1, "total_unique_individuals": 1, "species_wise_count": {"Dog": 1}, "region_wise_count": {}, "entry_count": 1, "exit_count": 0, "maximum_occupancy": 1, "average_occupancy": 1.0},
            visibility_scores={1: 1.0},
        )

        t = out["tracks"][0]
        self.assertEqual(t["final_species"], "Dog")
        self.assertIn("classifier_label", t)
        self.assertIn("agreement_score", t)
        self.assertIn("top5_predictions", t)
        self.assertIn("temporal_confidence", t)

        d = out["detections"][0]
        self.assertEqual(d["final_species"], "Dog")
        self.assertIn("species_confidence", d)
        self.assertIn("top5_predictions", d)


if __name__ == "__main__":
    unittest.main()
