# Vision Species Migration Report

Date: 2026-07-07

## Summary

The vision subsystem was upgraded with a dedicated species classification stage while preserving existing architecture and interfaces.

## Changes

1. Added species classifier module with prioritized model backends and automatic fallback.
2. Added configuration-driven taxonomy.
3. Added detector-classifier fusion with agreement scoring.
4. Added temporal species smoothing by track ID.
5. Added unknown-species handling for low-confidence predictions.
6. Extended vision output and dashboard tables with additive species metadata.
7. Upgraded detector default model preference to YOLO11 with YOLOv8 fallback.

## Compatibility

- No pipeline redesign.
- No tracking/counting replacement.
- Existing dashboard/fusion/reasoning paths remain compatible.
- Added fields are additive and optional for downstream consumers.

## Verification Scope

- Unit tests for species classifier decision cases.
- Output compatibility tests for new species fields.
- Existing vision/API/UI tests retained.

## Verification Snapshot

- Test suite: 23 passed (species + vision + API + UI compatibility)
- Multi-video probe: 1 local video available at runtime (`data/video/puppy crying.mp4`)
- Average classification latency: 1.955 ms
- Species consistency: 1.000
- Misclassification proxy (detector vs final species disagreement rate): 0.000
- Process memory during probe: 427.25 MB
- CPU usage during probe: 50.7%
- GPU utilization: not directly collected by the probe script (health endpoint provides simulated GPU metrics)

## Remaining Limitations

- Full BioCLIP/SigLIP2/EVA-CLIP backend activation depends on optional model/runtime dependencies.
- Runtime verification used only currently available local video media in `data/video`.
- Misclassification proxy is an operational indicator, not ground-truth labeled accuracy.
