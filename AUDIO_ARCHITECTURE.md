# Audio Architecture

## Scope
This document describes the upgraded in-place Audio Intelligence subsystem used by the multimodal welfare platform.

## Design Goals
- Preserve existing API entry points (`audio_module.capture_audio`, `audio_module.extract_audio_file_features`, `audio_module.detect_distress`).
- Support microphone-first ingestion with video-audio fallback in dashboard flows.
- Reduce false alarms through configurable non-distress suppression.
- Keep model interfaces modular for future pretrained model swaps and fine-tuning.

## Modules
1. Ingestion
   - `audio_intelligence/ingestion.py`
   - Microphone capture, audio-file loading, and video-audio extraction.

2. Preprocessing
   - `audio_intelligence/preprocessing.py`
   - Resampling, loudness normalization, optional noise reduction, silence removal, band-pass filtering.

3. Segmentation + VAD
   - `audio_intelligence/vad.py`
   - `audio_intelligence/segmentation.py`

4. Feature Extraction + Embeddings
   - `audio_intelligence/features.py`
   - Handcrafted features + embedding backend priority:
     - BEATs (if available)
     - AST (if available)
     - YAMNet (if available)
     - handcrafted fallback (always available)

5. Classification
   - `audio_intelligence/models.py`
   - Sound class, species, distress classifier, non-distress filter, temporal analyzer.

6. Orchestration
   - `audio_intelligence/pipeline.py`
   - Produces segment-level and summary-level outputs with backward-compatible fields.

7. API Compatibility Wrapper
   - `audio_module.py`
   - Keeps legacy output keys and extends with richer audio evidence.

## Compatibility
- Existing fields such as `distress`, `audio_distress`, `score`, `energy`, `zcr`, `centroid`, `bandwidth` are preserved.
- Additional fields are additive only.
