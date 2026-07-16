# Non-Distress Filter

## Objective
Reduce false positives by suppressing likely normal behavioural vocalizations.

## Implementation
- Class: `NonDistressFilter`
- File: `audio_intelligence/models.py`

## Supported Normal Categories
- mating vocalization
- courtship behaviour
- play vocalization
- feeding sound
- grooming sound
- parent-offspring communication
- social communication
- exploration
- curiosity
- normal environmental sound

## Behavior
- Computes `non_distress_probability`.
- Emits `event_type` and optional `suppressed_reason`.
- Suppresses distress contribution when non-distress confidence is high and distress evidence is not strong.
- Strong distress evidence can override suppression.

## Configuration
- `non_distress_suppression_strength`
- `non_distress_suppression_threshold`
- `strong_distress_override_threshold`
- `allow_cross_modal_override`
