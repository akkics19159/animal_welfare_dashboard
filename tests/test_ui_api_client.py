import unittest
from unittest.mock import MagicMock, patch

from ui.api_client import ApiClient


class ApiClientTests(unittest.TestCase):
    def test_online_uses_health_request_successfully(self):
        client = ApiClient("http://127.0.0.1:8000")
        response = MagicMock()
        response.status_code = 200

        with patch.object(client._session, "request", return_value=response) as mock_req:
            self.assertTrue(client.online())
            mock_req.assert_called()

    def test_online_caches_recent_result(self):
        client = ApiClient("http://127.0.0.1:8000")
        response = MagicMock()
        response.status_code = 200

        with patch.object(client._session, "request", return_value=response) as mock_req:
            self.assertTrue(client.online())
            self.assertTrue(client.online())
            self.assertEqual(mock_req.call_count, 1)


if __name__ == "__main__":
    unittest.main()
