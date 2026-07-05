# Audio Intelligence Migration Report

## Overview

The legacy audio subsystem was a single heuristic module that performed recording, feature extraction, distress scoring, and audio export in one place. The new subsystem replaces that path with a modular pipeline that supports ingestion, validation, preprocessing, VAD, segmentation, feature extraction, general sound classification, species-aware analysis, non-distress filtering, distress classification, and temporal analysis.

## Files Added or Modified

### Modified
- [audio_module.py](audio_module.py) — preserved the existing public API while routing analysis through the new pipeline.

### Added
- [audio_intelligence/__init__.py](audio_intelligence/__init__.py)
- [audio_intelligence/config.py](audio_intelligence/config.py)
- [audio_intelligence/interfaces.py](audio_intelligence/interfaces.py)
- [audio_intelligence/ingestion.py](audio_intelligence/ingestion.py)
- [audio_intelligence/preprocessing.py](audio_intelligence/preprocessing.py)
- [audio_intelligence/vad.py](audio_intelligence/vad.py)
- [audio_intelligence/segmentation.py](audio_intelligence/segmentation.py)
- [audio_intelligence/features.py](audio_intelligence/features.py)
- [audio_intelligence/models.py](audio_intelligence/models.py)
- [audio_intelligence/pipeline.py](audio_intelligence/pipeline.py)
- [tests/audio/test_audio_intelligence.py](tests/audio/test_audio_intelligence.py)

## Old Audio Pipeline

1. Capture audio or load a file.
2. Extract hand-crafted features.
3. Apply fixed thresholds.
4. Produce a single distress score.

## New Audio Pipeline

1. Ingest from microphone, file, or video audio.
2. Validate sample rate, channels, duration, and bit depth.
3. Normalize and denoise the signal.
4. Apply VAD to isolate speech-like or vocal activity.
5. Segment into overlapping windows.
6. Extract acoustic features and embeddings.
7. Classify general sound events.
8. Infer species-aware vocalization context.
9. Filter out non-distress sounds.
10. Estimate distress, pain, fear, panic, and aggression.
11. Produce a temporal pattern summary.

## Architectural Decisions

- Plugin-based interfaces were introduced so sound classifiers, species classifiers, and distress classifiers can be swapped without changing the pipeline.
- Dependency injection is used for the preprocessing, VAD, segmentation, and model components.
- Structured logging and configuration-driven thresholds replace hard-coded scoring heuristics.
- The existing audio module API remains compatible so the rest of the system can continue to call the previous functions.

## Advantages

- Easier replacement of models and classification strategies.
- Better separation between feature extraction and classification logic.
- Support for future transformer-based models and audio foundation models.
- More structured outputs for the fusion layer and dashboard.

## Future Extension Points

- Replace the default rule-based models with YAMNet, AST, or PANNs.
- Add species-specific vocalization models.
- Plug in real-time streaming backends.
- Add multi-channel and multi-microphone support.
- Store audio embeddings and analyses for research experiments.
