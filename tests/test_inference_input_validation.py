import unittest

from fastapi.testclient import TestClient

from api_server import app


class InferenceValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def _base_payload(self):
        return {
            'video_result': {'detections': [], 'motion_score': 0.0, 'agitated': False, 'error': None},
            'audio_result': {'distress': False, 'score': 0.0, 'error': None},
            'sensor_result': {'temp': 25.0, 'humidity': 50.0, 'heart_rate': 80, 'error': None},
            'ontology_strength': 0.6,
            'weights': {'video_score': 0.4, 'audio_score': 0.4, 'sensor_score': 0.2},
        }

    def test_rejects_invalid_ontology_strength(self):
        payload = self._base_payload()
        payload['ontology_strength'] = 1.5
        r = self.client.post('/api/inference', json=payload)
        self.assertEqual(r.status_code, 400)

    def test_rejects_non_positive_weight_sum(self):
        payload = self._base_payload()
        payload['weights'] = {'video_score': 0.0, 'audio_score': 0.0, 'sensor_score': 0.0}
        r = self.client.post('/api/inference', json=payload)
        self.assertEqual(r.status_code, 400)


if __name__ == '__main__':
    unittest.main()
