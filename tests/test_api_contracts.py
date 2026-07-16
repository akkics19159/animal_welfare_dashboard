import unittest

from fastapi.testclient import TestClient

from api_server import app


class APIContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_contract(self):
        r = self.client.get('/api/health')
        self.assertEqual(r.status_code, 200)
        data = r.json()
        for k in ['cpu', 'ram', 'disk', 'devices', 'gpu', 'inference_throughput']:
            self.assertIn(k, data)

    def test_inference_returns_required_keys(self):
        payload = {
            'video_result': {
                'detections': [{'class': 'dog', 'confidence': 0.9, 'xyxy': [10, 20, 100, 200]}],
                'motion_score': 0.12,
                'agitated': False,
                'error': None,
            },
            'audio_result': {'distress': True, 'score': 0.65, 'error': None},
            'sensor_result': {'temp': 32.5, 'humidity': 60.0, 'heart_rate': 115, 'error': None},
            'ontology_strength': 0.6,
            'weights': {'video_score': 0.4, 'audio_score': 0.4, 'sensor_score': 0.2},
        }
        r = self.client.post('/api/inference', json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        # Contract minimum for UI integration
        self.assertIn('probability', data)
        self.assertIn('modality_scores', data)
        self.assertIn('explanations', data)
        self.assertIn('latency_ms', data)


if __name__ == '__main__':
    unittest.main()

