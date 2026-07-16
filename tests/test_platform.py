import json
import unittest
from fastapi.testclient import TestClient

# Import the API application
from api_server import app, load_json_db, ALERTS_JSON, MODELS_JSON, DATASETS_JSON

class PlatformIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("cpu", data)
        self.assertIn("ram", data)
        self.assertIn("disk", data)
        self.assertIn("devices", data)

    def test_alerts_endpoint(self):
        # Verify active alerts list gets fetched
        response = self.client.get("/api/alerts")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_dataset_registry_endpoint(self):
        response = self.client.get("/api/dataset")
        self.assertEqual(response.status_code, 200)
        datasets = response.json()
        self.assertIsInstance(datasets, list)
        if len(datasets) > 0:
            self.assertIn("dataset_id", datasets[0])

    def test_models_registry_endpoint(self):
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        models = response.json()
        self.assertIsInstance(models, list)
        if len(models) > 0:
            self.assertIn("model_version", models[0])

    def test_switch_active_model(self):
        # Fetch initial models
        response = self.client.get("/api/models")
        models = response.json()
        if len(models) > 0:
            version_to_test = models[0]["model_version"]
            # Switch
            switch_response = self.client.post("/api/models/active", json={"model_version": version_to_test})
            self.assertEqual(switch_response.status_code, 200)
            self.assertIn("activated", switch_response.json()["message"])

    def test_inference_pipeline_execution(self):
        # Mock inputs
        payload = {
            "video_result": {
                "detections": [{"class": "dog", "confidence": 0.89, "xyxy": [10, 20, 100, 200]}],
                "motion_score": 0.12,
                "agitated": False,
                "error": None
            },
            "audio_result": {
                "distress": True,
                "score": 0.65,
                "error": None
            },
            "sensor_result": {
                "temp": 32.5,
                "humidity": 60.0,
                "heart_rate": 115,
                "error": None
            },
            "ontology_strength": 0.6
        }
        
        response = self.client.post("/api/inference", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("probability", data)
        self.assertIn("raw_probability", data)
        self.assertIn("modality_scores", data)
        self.assertIn("ontology_score", data)
        self.assertIn("explanations", data)

if __name__ == "__main__":
    unittest.main()
