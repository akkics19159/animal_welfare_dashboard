# Deployment Guide

Last validated: 2026-07-16

## Prerequisites
- Python 3.8+
- FFmpeg installed and accessible on PATH (recommended for audio support)

## Local Development
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Start backend:
```bash
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```
3. Start dashboard:
```bash
streamlit run dashboard.py --server.port 8501
```
4. Open: http://localhost:8501

## One-command launcher
Use `run_platform.py` to start both services (backend + Streamlit) together.

## Production Considerations
- **Process supervision**: Use a process manager (e.g., systemd, PM2 equivalent for Python, Docker Compose, or Kubernetes).
- **Health checks**: Monitor `GET /api/health`.
- **State**: This deployment uses file-based persistence (CSV/JSON). For multi-instance production, move to a shared datastore.
- **CORS/auth**: This codebase does not provide authentication middleware. Add auth before exposing externally.
- **Exports**: Export endpoints return JSON-safe payloads (CSV base64 + markdown). Frontend should decode base64 when downloading files.

## Production Hardening Rollout Checklist
1. Confirm webcam-first behavior on target host and verify automatic fallback to the newest valid file under `data/video`.
2. Ensure `logs/` is writable so rotating structured logs can be persisted to `logs/backend.log`.
3. Validate `/api/health` includes pipeline telemetry fields (`average_fps`, `current_fps`, `frame_latency_ms`, `inference_latency_ms`, `current_input_source`).
4. Run regression tests before release:
```bash
python -m pytest tests/test_vision_acquisition_and_tracking.py tests/test_vision_contracts_unit.py tests/test_api_contracts.py tests/test_api_exports.py tests/test_platform.py tests/test_behaviour_engine.py tests/test_api_behaviour_contract.py tests/test_health_extended_contract.py tests/test_inference_input_validation.py tests/test_ui_pages_smoke.py
```
5. Verify exports (`/api/export/history`, `/api/export/alerts`, `/api/export/analytics`) in `csv`, `json`, and `markdown` formats.
6. Validate API input rejection behavior:
	- Missing required inference fields -> HTTP 422
	- Invalid ontology strength or export format -> HTTP 400
7. Run full regression gate:
```bash
python -m pytest -q
```
Expected release-candidate baseline: all tests pass.

## Operational Notes
- History CSV reading is resilient to malformed legacy rows and skips bad lines instead of failing endpoint responses.
- JSON stores are written atomically to reduce corruption risk during unexpected shutdowns.
- Inference input validation enforces positive weight sum and ontology strength bounds `[0,1]`.


