# Species Classifier

## Purpose
The species subsystem resolves final species labels independently from YOLO detector labels.

Pipeline stages:
1. Detection (YOLO boxes/confidence/initial label)
2. Tracking (track id and temporal continuity)
3. Species recognition (crop-only classification)
4. Temporal verification (30-frame weighted voting)
5. Confidence calibration
6. Final species emission

## Public Interface
Primary class: `vision_intelligence/species_classifier.py::SpeciesClassifier`

Required interface:
- `predict(image, detector_label='unknown', detector_confidence=0.0, track_id=None) -> SpeciesPrediction`
- `classify_tracks(frame_bgr, tracks) -> float` (latency in ms)

`SpeciesPrediction` fields:
- `species`
- `confidence`
- `top5_predictions`
- `feature_embedding`
- `embedding_distance`

## Backend Plugins
Backend selection is plugin-based and ordered by configuration priority:
1. `openclip_vith14` (OpenCLIP ViT-H-14)
2. `openclip_vitl14` (OpenCLIP ViT-L-14)
3. `siglip2`
4. `yolo_fallback`

Backend is loaded once and cached for process lifetime.

## Device Policy
- Use GPU if available.
- Fallback to CPU automatically.
- Never crash if classifier dependencies are unavailable.

## Taxonomy
Species set is configured in `vision_intelligence/species_taxonomy.json` and includes:
Human, Dog, Cat, Cow, Horse, Goat, Sheep, Pig, Bird, Rabbit, Monkey, Bear, Elephant, Zebra, Giraffe, Camel, Buffalo, Chicken, Duck, Deer, Other.

## Output Contract (Added Fields)
Track and detection outputs include:
- `detector_label`
- `classifier_label`
- `final_species`
- `species_confidence`
- `classifier_confidence`
- `agreement_score`
- `embedding_distance`
- `top5_predictions`
- `classification_history`
- `species_embedding`
- `temporal_confidence`

Legacy fields remain unchanged for backward compatibility.
