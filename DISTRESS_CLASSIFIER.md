# Distress Classifier

## Current Implementation
- Class: `RuleBasedDistressClassifier`
- File: `audio_intelligence/models.py`

## Outputs
- `distress_probability`
- `pain_probability`
- `fear_probability`
- `panic_probability`
- `aggression_probability`
- `confidence`
- `audio_quality`
- `event_duration`
- `event_type`

## Notes
- The classifier is intentionally modular through `BaseDistressClassifier`.
- A pretrained/fine-tuned model can replace the current rule-based implementation without changing pipeline contracts.
