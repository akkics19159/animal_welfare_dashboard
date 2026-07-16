# API Documentation

Base URL: `http://127.0.0.1:8000`

## Conventions
- Responses are JSON.
- Errors use standard FastAPI `HTTPException` payloads: `{ "detail": "..." }`.
- Many UI endpoints rely on stable keys (e.g., `probability`, `modality_scores`, `explanations`).

## Inference
### `POST /api/inference`
Payload (JSON):
```json
{
  "video_result": { "...": "..." },
  "audio_result": { "...": "..." },
  "sensor_result": { "...": "..." },
  "ontology_strength": 0.6,
  "weights": {
    "video_score": 0.4,
    "audio_score": 0.4,
    "sensor_score": 0.2
  }
}
```
Response keys (best-effort contract):
- `probability` (float)
- `raw_probability` (may exist depending on inference output)
- `modality_scores` (object: `video_score`, `audio_score`, `sensor_score`)
- `ontology_score` (optional)
- `explanations` (object/list)
- `latency_ms` (float, added by API)
- `timestamp` (ISO string, added by API)

## Health
### `GET /api/health`
Response keys (best-effort contract):
- `cpu`, `ram`, `disk` (numbers)
- `gpu` (object: `name`, `usage`, `temp`)
- `devices` (object: `camera`, `microphone`, `sensor_array`)
- `inference_throughput` (string)
- `timestamp` (ISO string)

## History & analytics
### `GET /api/history`
Response: list of history records (from CSV).

### `GET /api/analytics`
Response:
```json
{
  "trends": [ {"date_hour": "...", "probability": 0.0, "video_score": 0.0, "audio_score": 0.0, "sensor_score": 0.0} ],
  "aggregates": {"mean_probability": 0.0, "max_probability": 0.0, "total_runs": 0, "critical_runs": 0}
}
```

## Training
### `GET /api/training`
Response:
- `status`: current simulated training status
- `history`: list of training runs

### `POST /api/training/run`
Payload:
```json
{"pipeline_type": "fusion", "epochs": 10}
```
Response:
- `{ "message": "Training pipeline started successfully in background." }`

### `POST /api/training/stop`
Response:
- `{ "message": "Training pipeline stop command issued." }`

## Evaluation
### `GET /api/evaluation`
Response: list of evaluation experiment records.

## Registries
### `GET /api/dataset`
Response: list of dataset entries (values from `dataset/registry/datasets.json`).

### `GET /api/models`
Response: list of model entries (values from `training/registry/models.json`).

### `POST /api/models/active`
Payload:
```json
{"model_version": "welfare-fusion-v1"}
```
Response:
- `{ "message": "Successfully activated model '...'." }`

## Alerts
### `GET /api/alerts`
Response: list of alert objects.

### `POST /api/alerts/acknowledge`
Payload:
```json
{"alert_id": "..."}
```
Response:
- `{ "message": "Acknowledged alert ..." }`

## Logs
### `GET /api/logs/{category}`
Categories supported (in-memory): `audit`, `inference`, `alert`, `training`, `model`.

### `POST /api/logs/clear`
Response: `{ "message": "Logs cleared successfully." }`

## Exports
All export endpoints use `format` query param.

### Common formats
- `json` → structured JSON with `rows`/data
- `csv` → base64-encoded CSV string under `csv_base64`
- `markdown` → markdown table under `markdown`
- `pdf` → graceful fallback containing `markdown_fallback` and a `note`

### History
- `GET /api/export/history?format=csv|json|markdown|pdf`

### Analytics
- `GET /api/export/analytics?format=...`
- `GET /api/export/charts/analytics?type=welfare_timeline`
  - returns `plotly_json` suitable for frontend rendering

### Others
- `GET /api/export/evaluation`
- `GET /api/export/training`
- `GET /api/export/dataset`
- `GET /api/export/models`
- `GET /api/export/alerts`
- `GET /api/export/health`


