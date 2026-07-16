# Temporal Classification

## Objective
Reduce species jitter and transient mislabels under occlusion, blur, low light, and fast motion.

## Per-Track State
Each track maintains:
- rolling history of last 30 predictions
- confidence per prediction
- final species history for observability

## Weighted Voting
For history index `i` in `1..N`, recency weight is:

`w_i = i / N`

Weighted vote score for species `s`:

`V(s) = sum(w_i * c_i)` over history entries with species `s`.

Temporal winner is species with max vote score.

## Species Switch Gate
Let:
- `S_prev` = previous final species
- `C_prev` = previous confidence
- `S_new` = candidate species from current classification
- `C_new` = candidate confidence
- `delta` = configurable margin

Switch only if:
- `S_new != S_prev`
- `C_new > C_prev + delta`

Otherwise keep previous species.

## Temporal Confidence
Temporal confidence is derived from weighted confidence accumulation and exposed as:
- `temporal_confidence`

This value is included in final confidence calibration.
