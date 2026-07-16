# User Manual

## What you’re running
This is a multimodal welfare monitoring platform:
- Video analysis
- Audio distress detection
- Sensor (simulated by default)
- Fusion + ontology-regularized welfare probability

## Start the platform
### Option A: Launcher
Run:
```bash
python run_platform.py
```
Then open: http://localhost:8501

### Option B: Manual
Backend:
```bash
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```
Dashboard:
```bash
streamlit run dashboard.py --server.port 8501
```

## Navigate the dashboard
Use the **sidebar** to choose a module:
- Dashboard
- Live Monitoring
- Analytics
- Evaluation
- Dataset
- Training
- Models
- Alerts (placeholder)
- System Health
- Settings (placeholder)
- User Management (placeholder)

## Run live inference
1. Go to **Live Monitoring**.
2. Click **RUN LIVE INFERENCE**.
3. (Optional) enable **auto refresh**.

Results are shown as:
- Welfare score / probability
- Video/audio/sensor contributions
- Detections overlay and metadata widgets
- Alerts panel (if produced)

## Analytics
- Configure date range and optional filters.
- Charts are best-effort based on backend analytics/history or local CSV fallback.

## Models
- You can switch the active model via the registry.
- This toggles the `active` flag in the backend model registry.

## Exports (API support)
Backend supports export endpoints for history/alerts/evaluation/training/dataset/models/analytics/health.

Formats supported:
- `json`
- `csv` (base64 payload)
- `markdown`
- `pdf` (graceful fallback)

(Frontend download UI is not fully implemented; use API endpoints directly.)

## Troubleshooting
- If audio fails, install FFmpeg and ensure it’s on PATH.
- If backend is offline, pages use local CSV fallback where possible.


