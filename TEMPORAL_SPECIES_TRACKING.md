# Temporal Species Tracking

Temporal species tracking prevents rapid species oscillation for the same track ID.

## Strategy

- Keep a bounded history buffer per track ID.
- Append each frame-level species prediction with confidence.
- Apply confidence-weighted, recency-aware voting.
- Emit the dominant species as the smoothed final species.

## Outcome

Tracks are stabilized against frame-to-frame noise, reducing patterns like:

Dog -> Person -> Dog -> Person

for the same individual.

## Unknown Handling

If confidence remains below threshold, output Unknown Animal or Unknown Human and keep temporal history for later recovery.
