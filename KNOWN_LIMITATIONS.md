# Known Limitations

1. CPU-only environments have limited real-time vision throughput; measured FPS in this environment is below interactive real-time expectations.
2. GPU telemetry is best-effort unless NVML/vendor-specific monitoring is integrated.
3. Long-duration soak and high-concurrency stress are not fully automated in CI yet.
4. Health endpoint performs live source probing, which can increase response latency.
5. Audio capture quality depends on host hardware and OS permissions.
6. A transient Windows native access violation stack trace was observed once in a targeted mixed test path (pyarrow/sklearn import chain), while tests still completed; binary dependency stability should be monitored.
