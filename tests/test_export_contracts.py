import unittest

from fastapi.testclient import TestClient

from api_server import app


class ExportContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def _check_supported_export_endpoint(self, url):
        # Ensure every format returns 200 and basic keys exist.
        for fmt in ['json', 'csv', 'markdown', 'pdf']:
            r = self.client.get(f'{url}?format={fmt}')
            self.assertEqual(r.status_code, 200, msg=f'Failed: {url} fmt={fmt}')
            data = r.json()
            self.assertEqual(data.get('format'), fmt)

    def test_export_history_formats(self):
        self._check_supported_export_endpoint('/api/export/history')

    def test_export_alerts_formats(self):
        self._check_supported_export_endpoint('/api/export/alerts')

    def test_export_dataset_formats(self):
        self._check_supported_export_endpoint('/api/export/dataset')

    def test_export_models_formats(self):
        self._check_supported_export_endpoint('/api/export/models')


if __name__ == '__main__':
    unittest.main()

