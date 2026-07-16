import unittest

from multimodal_engine import analyze_combined


class SpeciesFusionReasoningCompatibilityTests(unittest.TestCase):
    def test_combined_inference_accepts_species_enriched_tracks(self):
        video_result = {
            "detections": [{"class": "person", "confidence": 0.6, "xyxy": [1, 2, 80, 120], "track_id": 1}],
            "tracks": [
                {
                    "track_id": 1,
                    "species": "Dog",
                    "final_species": "Dog",
                    "detector_label": "Human",
                    "classifier_label": "Dog",
                    "species_confidence": 0.83,
                    "agreement_score": 0.33,
                    "velocity": [1.0, 0.0],
                    "acceleration": [0.0, 0.0],
                    "dwell_time": 3.0,
                    "direction_label": "right",
                }
            ],
            "occupancy": 1,
            "motion_score": 0.2,
            "agitated": False,
            "error": None,
        }
        audio_result = {"distress": False, "score": 0.2, "event_type": "play", "error": None}
        sensor_result = {"temp": 27.0, "humidity": 55.0, "heart_rate": 85, "error": None}

        out = analyze_combined(video_result, audio_result, sensor_result, ontology_strength=0.6)

        self.assertIn("probability", out)
        self.assertIn("video_result", out)
        self.assertIsInstance(out["video_result"], dict)


if __name__ == "__main__":
    unittest.main()
