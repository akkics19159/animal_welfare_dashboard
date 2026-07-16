from __future__ import annotations

import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import requests


class ApiClient:
    def __init__(self, api_url: str = "http://127.0.0.1:8000", run_token: str = ""):
        self.api_url = api_url.rstrip("/")
        self._last_online_check_at = 0.0
        self._last_online_value = False
        self._session = requests.Session()
        self._run_token = run_token

    def _request(self, method: str, path: str, *, timeout: float, **kwargs):
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.setdefault("Cache-Control", "no-cache, no-store, must-revalidate")
        headers.setdefault("Pragma", "no-cache")
        headers.setdefault("Expires", "0")
        url_path = self._with_run_token(path)
        return self._session.request(method, f"{self.api_url}{url_path}", timeout=timeout, headers=headers, **kwargs)

    def _with_run_token(self, path: str) -> str:
        if not self._run_token:
            return path
        parts = urlsplit(path)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["_run"] = str(self._run_token)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

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
