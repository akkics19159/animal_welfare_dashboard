# Species Classification Architecture

This upgrade adds a dedicated species classification stage to the existing vision pipeline without redesigning architecture.

Flow:

Input Source -> YOLO Detector -> Tracker -> Object Crop -> Species Classifier -> Behavior Recognition -> Fusion

## Key Principles

- Detector answers location.
- Classifier answers species identity.
- Detector and classifier are fused into a final species result.
- Temporal smoothing stabilizes track-level species over time.
- Unknown classes are returned when confidence is low.

## Backward Compatibility

- Existing detections/tracks contracts are preserved.
- Added fields are additive: detector_label, classifier_label, final_species, species_confidence, agreement_score, classification_history.

## Model Priority

1. BioCLIP
2. SigLIP2
3. EVA-CLIP
4. CLIP ViT-L/14
5. Heuristic fallback when no CLIP backend is available

The module auto-selects the first available backend and does not fail hard when a model is unavailable.
