# Multimodal Decision Flow

1. Gather modality outputs from vision/audio/sensors.
2. Compute legacy-compatible modality scores.
3. Run fusion engine (confidence-weighted + Bayesian-inspired evidence accumulation).
4. Compute modality agreement and missing-modality impact.
5. Run welfare reasoning engine (rules, context, contradictions, species thresholds, temporal context).
6. Apply false-positive suppression for known normal behaviours.
7. Resolve conflicts and generate explainability artifacts.
8. Emit final welfare contract fields and preserve legacy outputs.
