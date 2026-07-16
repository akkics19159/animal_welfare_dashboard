# Operations Runbook

## Startup

1. Activate environment.
2. Start API server.
3. Launch dashboard.

## Input Source Behavior

- System attempts webcam first.
- If webcam unavailable, it automatically selects newest valid video under data/video.
- If no source is available, API/UI remain responsive and emit no-source diagnostics.

## Health Monitoring

Use /api/health for:
- device status
- pipeline status
- input source details
- latency/FPS metrics
- memory/storage/thread telemetry

## Log Locations

- Structured backend logs: logs/backend.log
- API in-memory category logs: /api/logs/{category}

## Recovery Playbook

1. Camera unavailable:
- Verify camera permissions and device lock.
- System should auto-fallback to local video.

2. Export errors:
- Check history CSV integrity.
- System now skips malformed rows, but investigate root corruption source.

3. Elevated latency:
- Verify current input source and device status from /api/health.
- Check CPU/RAM pressure and thread counts.

## Incident Response

- Capture /api/health snapshot.
- Export relevant logs from logs/backend.log.
- Export history/alerts via /api/export endpoints.
