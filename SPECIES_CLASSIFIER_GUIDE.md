# Species Classifier Guide

## Module

- vision_intelligence/species_classifier.py
- taxonomy config: vision_intelligence/species_taxonomy.json

## Outputs per track

- detector_label
- classifier_label
- final_species
- species_group
- species_confidence
- classifier_confidence
- classification_confidence
- agreement_score
- embedding_distance
- top_5_predictions
- classification_history

## Runtime Behavior

- Lazy-loads classifier backend on first inference.
- Caches selected backend for reuse.
- Supports GPU if backend supports it and GPU is available.
- Falls back to CPU and/or heuristic backend automatically.

## Integration

- Vision pipeline updates TrackState.class_name to smoothed final species.
- Counting, behavior, fusion, and welfare consume improved species without interface changes.
