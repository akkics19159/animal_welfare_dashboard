# Classifier Migration Report

## Summary
Migrated species recognition from detector-trust mode to staged classification with temporal and confidence calibration.

## Implemented Changes
- Reworked `vision_intelligence/species_classifier.py` into plugin-based architecture.
- Added backend order support: OpenCLIP ViT-H-14 -> OpenCLIP ViT-L-14 -> SigLIP2 -> YOLO fallback.
- Added `predict(image) -> SpeciesPrediction` interface.
- Added per-track temporal stabilization (30-frame weighted history + switch margin).
- Added calibrated species confidence from detector/classifier/embedding/temporal/agreement signals.
- Extended `TrackState` in `vision_intelligence/tracking.py` with species contract fields.
- Extended `vision_intelligence/vision_output.py` with additive species metadata fields.
- Updated `behaviour_recognition/engine.py` to consume `final_species`.
- Updated `multimodal_engine.py` to normalize and reason from `final_species`.
- Updated `api_server.py` history species logging to prefer `final_species`.
- Updated taxonomy in `vision_intelligence/species_taxonomy.json` to include required species + `Other`.

## Validation
Executed targeted tests:
- `tests/test_species_classifier.py`
- `tests/test_species_output_compatibility.py`
- `tests/test_species_fusion_reasoning_compat.py`
- `tests/test_behaviour_engine.py`

Result: 14 passed.

## Backward Compatibility
- Existing fields preserved.
- New fields are additive.
- Webcam-first and local-video fallback workflow unchanged.

## Remaining Limitations
- OpenCLIP/SigLIP2 availability depends on optional runtime dependencies and model assets.
- Full field validation under real webcam/video scenarios requires runtime hardware/data verification.
- Benchmark metrics (accuracy/F1/FPS/GPU utilization) require evaluation runs over curated validation data.
