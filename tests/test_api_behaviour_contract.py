import unittest

from fastapi.testclient import TestClient

from api_server import app


class APIBehaviourContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_inference_contains_behaviour_contract(self):
        payload = {
            "video_result": {
                "detections": [{"class": "dog", "confidence": 0.9, "xyxy": [10, 20, 100, 200]}],
                "tracks": [
                    {
                        "track_id": 7,
                        "velocity": [50.0, 0.0],
                        "acceleration": [5.0, 0.0],
                        "dwell_time": 2.0,
                        "direction_label": "right",
                        "species": "dog",
                    }
                ],
                "occupancy": 1,
                "motion_score": 0.2,
                "agitated": False,
                "error": None,
            },
            "audio_result": {"distress": False, "score": 0.2, "event_type": "play", "error": None},
            "sensor_result": {"temp": 26.5, "humidity": 55.0, "heart_rate": 88, "error": None},
            "ontology_strength": 0.6,
            "weights": {"video_score": 0.4, "audio_score": 0.4, "sensor_score": 0.2},
        }
        r = self.client.post("/api/inference", json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.json()

        for key in [
            "behaviour",
            "behaviour_probability",
            "behaviour_confidence",
            "behaviour_duration",
            "behaviour_history",
            "behaviour_transition",
            "behaviour_stability",
            "behaviour_timeline",
            "behaviour_distribution",
        ]:
            self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
