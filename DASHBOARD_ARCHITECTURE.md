# Dashboard Architecture (2026)

## Overview
The dashboard is a Streamlit web UI that orchestrates multimodal inference through a FastAPI REST backend. The UI is split into page modules (navigation targets) and reusable live components (widgets).

- **UI**: `dashboard.py`, `ui/`
- **Backend API**: `api_server.py`
- **Core AI logic (do not modify)**: `multimodal_engine.py`, `video_module.py`, `audio_module.py`, `sensors.py`, `ontology.py`, `welfare_reasoning/*`

## Request flow (Live Monitoring)
1. User clicks **RUN LIVE INFERENCE** (or auto-refresh triggers rerun).
2. `ui/pages/live_monitoring.py` calls `_run_inference(...)`.
3. If API backend is online: `_run_inference` POSTs to `POST /api/inference` with payload:
   - `video_result`, `audio_result`, `sensor_result`
   - `ontology_strength`
   - `weights` (fusion weights)
4. Backend:
   - calls `analyze_combined(...)`
   - returns probability, modality scores, explanations, and metadata
   - writes legacy CSV history (`welfare_analysis_history.csv`)
   - triggers alert heuristics and persists alerts to `alerts.json`

## Navigation + UI module wiring
- Sidebar module: `ui/sidebar.py`
- Page routing: `dashboard.py` maps selected page name to `ui.pages.<page>` and calls `render_page(...)`.

`dashboard.py` mapping:
- Dashboard → `ui.pages.dashboard`
- Live Monitoring → `ui.pages.live_monitoring`
- Analytics → `ui.pages.analytics`
- History → `ui.pages.history`
- Evaluation → `ui.pages.evaluation`
- Dataset → `ui.pages.dataset`
- Training → `ui.pages.training`
- Models → `ui.pages.models`
- Alerts → `ui.pages.alerts`
- System Health → `ui.pages.system_health`
- Settings → `ui.pages.settings`
- User Management → `ui.pages.user_management`

## UI pages
Pages live under `ui/pages/` and must expose `render_page(api, backend_online, default_history_path, default_video_path)`.

Not implemented / placeholder pages:
- Alerts, History, Settings, User Management (currently return informative placeholders but remain importable).

## API surface (high level)
### Core
- `POST /api/inference`
- `GET /api/health`
- `GET /api/history`
- `GET /api/analytics`
- `GET /api/training`
- `POST /api/training/run`
- `POST /api/training/stop`
- `GET /api/evaluation`
- `GET /api/dataset`
- `GET /api/models`
- `POST /api/models/active`
- `GET /api/alerts`
- `POST /api/alerts/acknowledge`

### Logs
- `GET /api/logs/{category}`
- `POST /api/logs/clear`

### Exports
- `GET /api/export/history`
- `GET /api/export/analytics`
- `GET /api/export/evaluation`
- `GET /api/export/training`
- `GET /api/export/dataset`
- `GET /api/export/models`
- `GET /api/export/alerts`
- `GET /api/export/health`
- `GET /api/export/charts/analytics`

Exports support `format=csv|json|markdown` and return graceful PDF fallbacks when PDF generation is unavailable.

## Data persistence (file-based)
- History CSV: `welfare_analysis_history.csv`
- Alerts: `alerts.json`
- Datasets registry: `dataset/registry/datasets.json`
- Models registry: `training/registry/models.json`
- Evaluation history: `evaluation/experiments/history.json`
- Training tracking: `training/tracking/experiments.json`

## Quality & UX rules
- Backend calls must be wrapped with `try/except` and show a user-facing error/empty state.
- Loading indicators should be used while waiting for backend responses.
- No changes to AI inference logic are required for UI integration.

