import unittest

from fastapi.testclient import TestClient

from api_server import app


class HealthExtendedContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_has_pipeline_and_model_status(self):
        r = self.client.get('/api/health')
        self.assertEqual(r.status_code, 200)
        data = r.json()

        self.assertIn('pipeline', data)
        self.assertIn('memory', data)
        self.assertIn('storage', data)
        self.assertIn('model_status', data)

        pipeline = data['pipeline']
        for key in [
            'status',
            'current_input_source',
            'current_input_type',
            'average_fps',
            'current_fps',
            'frame_latency_ms',
            'inference_latency_ms',
            'thread_count',
            'queue_sizes',
        ]:
            self.assertIn(key, pipeline)


if __name__ == '__main__':
    unittest.main()
