# Behaviour Recognition

This module extends the existing multimodal platform with additive behaviour inference fields while preserving all existing contracts.

## Output Fields

- behaviour
- behaviour_probability
- behaviour_confidence
- behaviour_duration
- behaviour_history
- behaviour_transition
- behaviour_stability
- behaviour_timeline
- behaviour_distribution

All fields are additive and backward compatible.

## Fallback Chain

The engine follows this order:

1. Video Swin Transformer (optional)
2. TimeSformer (optional)
3. SlowFast (optional)
4. X3D (optional)
5. Pose plus trajectory rule engine
6. Trajectory-only fallback

When advanced temporal models are unavailable, the rule-based and trajectory fallbacks continue delivering behaviour signals.
