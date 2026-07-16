# Final Architecture (Hardened)

Validated release-candidate hardening update: 2026-07-16

The architecture remains unchanged and modular. This pass introduced hardening enhancements only.

## Core Flow

1. Input Source Manager
- Webcam-first probe and health check
- Automatic fallback to newest readable local video

2. Vision Intelligence
- Detection -> Tracking -> Counting -> Trajectory analytics
- Extended vision contract fields passed downstream

3. Audio + Sensors
- Audio distress and event metadata
- Sensor readings integrated as modality evidence

4. Fusion + Welfare Reasoning
- Multimodal fusion and calibrated welfare probability
- Explainability and agreement/conflict metadata

5. Behaviour Layer
- Plugin fallback behavior inference, additive fields only

6. Serving Layer
- FastAPI inference/history/analytics/exports/health/model/dataset/training/evaluation endpoints

7. UI Layer
- Streamlit dashboard pages consuming API contracts

## Hardening Additions

- Structured rotating logs
- Atomic JSON persistence
- Thread-safe history appends
- Resilient history parsing
- Expanded system health telemetry
- Input validation for inference payloads
- Audio pipeline runtime reuse cache (no API changes)
- Heatmap memory optimization in legacy helper path
- Species recognition staged pipeline stabilization maintained
