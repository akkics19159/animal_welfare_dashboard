import unittest

from behaviour_recognition import BehaviourRecognitionEngine


class BehaviourEngineTests(unittest.TestCase):
    def test_engine_returns_required_contract_fields(self):
        engine = BehaviourRecognitionEngine()
        video_result = {
            "occupancy": 3,
            "tracks": [
                {
                    "track_id": 1,
                    "velocity": [45.0, 5.0],
                    "acceleration": [2.0, 1.0],
                    "dwell_time": 3.0,
                    "direction_label": "right",
                    "species": "dog",
                }
            ],
        }
        audio_result = {"event_type": "social_communication"}
        sensor_result = {"temp": 28.0, "humidity": 50.0}

        out = engine.analyze(video_result=video_result, audio_result=audio_result, sensor_result=sensor_result)

        self.assertIsInstance(out.behaviour, str)
        self.assertIn("behaviour", video_result["tracks"][0])
        self.assertIn("behaviour_probability", video_result["tracks"][0])
        self.assertIn("behaviour_confidence", video_result["tracks"][0])
        self.assertIn("behaviour_duration", video_result["tracks"][0])
        self.assertIn("behaviour_history", video_result["tracks"][0])
        self.assertIn("behaviour_transition", video_result["tracks"][0])
        self.assertIn("behaviour_stability", video_result["tracks"][0])
        self.assertIsInstance(out.behaviour_timeline, list)


if __name__ == "__main__":
    unittest.main()
