# Audio Flow

## Runtime Path
1. `capture_audio(...)` or `extract_audio_file_features(...)` in `audio_module.py`.
2. Build `AudioInput` and call `AudioIntelligencePipeline.analyze(...)`.
3. Preprocess + segment + feature extract + classify + suppress + temporal aggregate.
4. Return enriched but backward-compatible audio result.
5. `detect_distress(...)` maps to fusion/dashboard-friendly contract.

## Fusion-Relevant Outputs
- `score` / `distress_probability`
- `confidence`
- `non_distress_probability`
- `event_duration`
- `audio_quality`
- `temporal_consistency`

## Reasoning-Relevant Outputs
- `event_type`
- `suppressed_reason`
- segment-level evidence in `segments`
