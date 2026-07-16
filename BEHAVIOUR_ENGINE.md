# Behaviour Engine

## Architecture

The behaviour engine is implemented in behaviour_recognition/engine.py as a plugin-based recognizer.

- Plugins expose available() and predict(track, context).
- Plugins are prioritized and evaluated in order.
- The first valid prediction is used.
- Per-track temporal memory derives behaviour history, transition, duration, and stability.

## Integration

multimodal_engine.analyze_combined calls BehaviourRecognitionEngine.analyze and injects additive behaviour outputs into both:

- top-level combined response
- video_result for legacy UI compatibility

No existing Vision, Audio, Tracking, Fusion, or Welfare logic is replaced.
