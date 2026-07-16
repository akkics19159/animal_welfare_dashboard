from __future__ import annotations

import time
import requests


class ApiClient:
    def __init__(self, api_url: str = "http://127.0.0.1:8000"):
        self.api_url = api_url.rstrip("/")
        self._last_online_check_at = 0.0
        self._last_online_value = False
        self._session = requests.Session()

    def _request(self, method: str, path: str, *, timeout: float, **kwargs):
        return self._session.request(method, f"{self.api_url}{path}", timeout=timeout, **kwargs)

    def health(self) -> dict:
        r = self._request("GET", "/api/health", timeout=3.0)
        r.raise_for_status()
        return r.json()

    def get(self, path: str):
        r = self._request("GET", path, timeout=5.0)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, payload: dict):
        r = self._request("POST", path, timeout=10.0, json=payload)
        r.raise_for_status()
        return r.json()

    def online(self) -> bool:
        now = time.time()
        if now - self._last_online_check_at < 3.0:
            return self._last_online_value

        try:
            r = self._request("GET", "/api/health", timeout=3.0)
            self._last_online_value = r.status_code == 200
        except Exception:
            self._last_online_value = False

        self._last_online_check_at = now
        return self._last_online_value

