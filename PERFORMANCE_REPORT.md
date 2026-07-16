# Performance Report

Date: 2026-07-16
Environment: Windows, Python 3.14.5, CPU-only runtime (no CUDA device available)

## Measurement Method

- Vision profiling via repeated calls to analyze_video_behavior
- Fusion profiling via repeated calls to analyze_combined
- Process metrics via psutil

## Results

### Vision
- Average latency: 1696.772 ms
- P95 latency: 1148.109 ms
- Average FPS: 0.77

### Fusion
- Average latency: 0.2457 ms
- P95 latency: 0.3406 ms

### Resource Usage
- CPU sample start: 8.4%
- CPU sample end: 39.4%
- RAM start: 255.37 MB
- RAM end: 483.37 MB
- RAM delta: 228.0 MB
- Thread count: 48
- Disk used: 143.53 GB
- Disk free: 68.68 GB
- GPU available: False
- GPU memory allocated: 0.0 MB
- GPU memory reserved: 0.0 MB

## Interpretation

1. Fusion and reasoning path is low-latency.
2. Vision path dominates end-to-end latency in CPU-only mode.
3. For real-time production throughput, GPU acceleration is recommended.

## Actions Already Applied

1. Reduced memory pressure in heatmap generation by removing per-frame accumulation list.
2. Reduced repeated audio pipeline initialization overhead through runtime caching.
3. Kept API contracts and architecture unchanged.
