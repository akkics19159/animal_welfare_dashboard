# Audio Pipeline

## Processing Flow
1. Ingest audio from microphone or file/video source.
2. Preprocess signal:
   - resample
   - loudness normalize
   - noise suppression (optional)
   - silence removal (optional)
   - band-pass filter
3. Detect voice activity and build analysis segments.
4. Extract handcrafted descriptors and embeddings.
5. Classify:
   - general sound class
   - species (best effort)
   - distress probability + confidence
6. Apply non-distress suppression for known normal behaviours.
7. Run temporal analysis across segments.
8. Emit backward-compatible result with extended evidence fields.

## Input Modes
- Live microphone.
- Audio track extracted from video/audio files.

## Fallback
- Dashboard live monitoring now attempts video-audio analysis when microphone initialization fails.

## Output Highlights
- `distress_probability`
- `confidence`
- `audio_quality`
- `temporal_consistency`
- `non_distress_probability`
- `suppressed_reason`
- `event_type`
- `audio_embedding`
