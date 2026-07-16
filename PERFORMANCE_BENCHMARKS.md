# Performance Benchmarks

Date: 2026-07-07
Environment: Windows, local .venv, CPU-first runtime conditions observed.

## Sample Benchmark Results

Collected via local benchmark script and API probes:

- vision_ms: 8494.06
- audio_ms: 0.01
- fusion_ms: 0.58
- health_api_ms: 1195.77
- inference_api_ms: 26.45
- memory_delta_mb: 188.45
- vision_fps: 0.12

API statuses:
- /api/health: 200
- /api/inference: 200

## Interpretation

- End-to-end inference API latency is low for pure fusion/reasoning path.
- Vision latency is dominant under current local capture conditions.
- Health endpoint includes source probing, which adds measurable overhead.

## Exposed Runtime Metrics

The health endpoint now exposes:
- average_fps
- current_fps
- frame_latency_ms
- inference_latency_ms
- thread_count
- queue_sizes
- process/system memory
- storage health
- current input source and pipeline status

## Optimization Opportunities

- Add startup warmup for detector path.
- Cache camera probe metadata for health endpoint with short TTL.
- Introduce periodic async sampling for health metrics to avoid synchronous probe cost.
