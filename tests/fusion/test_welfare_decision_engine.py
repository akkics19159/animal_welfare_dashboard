import unittest

from multimodal_engine import analyze_combined


class WelfareDecisionEngineTests(unittest.TestCase):
    def test_true_distress_increases_risk(self):
        video_result = {
            "detections": [{"class": "dog", "confidence": 0.9, "xyxy": [1, 2, 10, 20]}],
            "agitated": True,
            "motion_score": 0.8,
            "pacing": True,
            "occupancy": 1,
        }
        audio_result = {
            "distress": True,
            "score": 0.85,
            "distress_probability": 0.85,
            "event_type": "distress_vocalization",
            "non_distress_probability": 0.1,
            "audio_quality": 0.8,
            "confidence": 0.8,
        }
        sensor_result = {"temp": 35.0, "humidity": 82.0, "heart_rate": 130, "error": None}

        out = analyze_combined(video_result, audio_result, sensor_result)
        self.assertGreaterEqual(float(out["probability"]), 0.45)
        self.assertIn(out["risk_level"], {"moderate", "high"})
        self.assertIn("evidence_summary", out)

    def test_mating_false_positive_suppression(self):
        video_result = {
            "detections": [{"class": "cat", "confidence": 0.8, "xyxy": [1, 2, 10, 20]}],
            "agitated": False,
            "motion_score": 0.2,
            "mating": True,
            "occupancy": 2,
        }
        audio_result = {
            "distress": True,
            "score": 0.8,
            "distress_probability": 0.8,
            "event_type": "mating",
            "non_distress_probability": 0.9,
            "audio_quality": 0.8,
            "confidence": 0.8,
        }
        sensor_result = {"temp": 25.0, "humidity": 55.0, "heart_rate": 80, "error": None}

        out = analyze_combined(video_result, audio_result, sensor_result)
        self.assertLess(float(out["probability"]), 0.6)
        self.assertTrue(len(out.get("suppressed_evidence", [])) > 0)

    def test_play_feeding_grooming_suppression(self):
        for behavior, event_type in [("play", "play"), ("feeding", "feeding"), ("grooming", "grooming")]:
            video_result = {
                "detections": [{"class": "dog", "confidence": 0.8, "xyxy": [1, 2, 10, 20]}],
                behavior: True,
                "motion_score": 0.3,
                "occupancy": 2,
            }
            audio_result = {
                "distress": True,
                "score": 0.75,
                "distress_probability": 0.75,
                "event_type": event_type,
                "non_distress_probability": 0.88,
                "audio_quality": 0.8,
                "confidence": 0.8,
            }
            sensor_result = {"temp": 25.0, "humidity": 50.0, "heart_rate": 82, "error": None}
            out = analyze_combined(video_result, audio_result, sensor_result)
            self.assertLess(float(out["probability"]), 0.65)

    def test_contradictory_evidence_is_explained(self):
        video_result = {
            "detections": [],
            "agitated": False,
            "motion_score": 0.1,
            "occupancy": 0,
        }
        audio_result = {
            "distress": True,
            "score": 0.8,
            "distress_probability": 0.8,
            "event_type": "distress_vocalization",
            "non_distress_probability": 0.1,
            "audio_quality": 0.9,
            "confidence": 0.9,
        }
        sensor_result = {"temp": 22.0, "humidity": 40.0, "heart_rate": 78, "error": None}

        out = analyze_combined(video_result, audio_result, sensor_result)
        text = " | ".join(out.get("explanations", []))
        self.assertIn("no visible sentient being", text.lower())

    def test_output_contract_additive_fields_present(self):
        out = analyze_combined(
            {"detections": [{"class": "dog", "confidence": 0.9, "xyxy": [1, 2, 10, 20]}], "motion_score": 0.3},
            {"distress": False, "score": 0.2, "event_type": "social_communication", "non_distress_probability": 0.7},
            {"temp": 25.0, "humidity": 50.0, "heart_rate": 80, "error": None},
        )
        for k in [
            "welfare_score",
            "risk_level",
            "severity",
            "urgency",
            "confidence",
            "prediction_uncertainty",
            "agreement_score",
            "evidence_summary",
            "suppressed_evidence",
            "reasoning_trace",
            "temporal_welfare_state",
        ]:
            self.assertIn(k, out)


if __name__ == "__main__":
    unittest.main()
