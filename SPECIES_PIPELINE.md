# Species Pipeline

## Runtime Behavior
1. Input Source Manager keeps webcam-first policy.
2. YOLO detects objects and provides detector labels.
3. Tracker assigns stable track ids.
4. Species classifier processes active track crops in batch.
5. Temporal verification stabilizes per-track species.
6. Confidence calibration computes final species confidence.
7. Vision output emits additive species metadata.

## Performance Notes
- Backend model loads once.
- Text embeddings are cached in backend.
- Crop classification is batched for active tracks.
- GPU is used when available; CPU fallback is automatic.

## Fallback Path
If OpenCLIP/SigLIP2 backends are unavailable:
- YOLO-label fallback backend maps detector labels to taxonomy species.
- Pipeline still returns complete output schema.

## Downstream Consumption
- Behaviour engine reads `final_species`.
- Multimodal engine normalizes tracks so `species == final_species`.
- API history records final species, not detector class.
