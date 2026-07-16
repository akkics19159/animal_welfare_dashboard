# Operations Guide

## Purpose
Operational runbook for production-style execution without changing architecture or public contracts.

## Startup

1. Backend
- Command: python -m uvicorn api_server:app --host 127.0.0.1 --port 8000

2. Dashboard
- Command: streamlit run dashboard.py --server.port 8501

3. Health check
- Endpoint: GET /api/health

## Input Source Policy (Must Remain Unchanged)

Priority order:
1. Laptop webcam
2. Latest valid local video in data/video
3. Graceful no-source mode

## Daily Checks

1. Verify backend log is rotating in logs/backend.log.
2. Check GET /api/health for:
- source_type
- current_fps
- inference_latency_ms
- average_fps
- memory usage
3. Verify /api/inference returns 200 on valid payload.
4. Verify /api/history and /api/analytics return 200.

## Incident Handling

1. Camera unavailable
- Confirm no-source graceful mode is active.
- Confirm fallback local video source discovery in data/video.

2. Audio capture failures
- Confirm microphone permissions.
- Verify ffmpeg availability for media fallback workflows.

3. Inference failures
- Check structured error logs in backend.log.
- Inspect request payload for schema/weight validation errors.

## Log Categories

- inference
- alert
- training
- model
- audit
- export
- health

## Release Gate Checklist

1. Run full regression: python -m pytest -q
2. Validate key API endpoints and invalid input behavior.
3. Run short reliability soak for multimodal inference loops.
4. Confirm dashboard loads and renders major pages.
5. Confirm export endpoints return valid status codes.

## Rollback Plan

1. Preserve previous tagged release artifacts.
2. Revert to prior deployment package.
3. Validate /api/health and /api/inference after rollback.
