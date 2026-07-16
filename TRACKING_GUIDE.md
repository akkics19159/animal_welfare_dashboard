# Tracking Guide

## Tracker Policy
- Primary tracker: BoT-SORT
- Fallback tracker: ByteTrack
- Final safety fallback: internal IoU/centroid tracker
- DeepSORT is not used.

## Implementation
- File: `vision_intelligence/tracking.py`
- Runtime behavior:
  - attempts BoT-SORT initialization first
  - then ByteTrack
  - then resilient internal tracking if dependencies are unavailable

## Per-Track Schema
Each active individual keeps persistent state:
- `track_id`
- `class_name` (species)
- `bbox_xyxy`
- `detection_confidence`
- `tracking_confidence`
- `age_frames`
- `entry_time`, `exit_time`, `dwell_time`
- `track_lifetime_sec`
- `trajectory` (time-stamped centroid history)
- `velocity_xy`, `acceleration_xy`
- `direction_deg`, `direction_label`
- `visibility_score`
- `lost_frames`, `was_occluded`, `reacquired`

## Occlusion and Recovery
- Occlusion is inferred from unmatched update cycles (`lost_frames > 0`).
- Recovery occurs when a previously lost target re-matches.
- `was_occluded` records historical recovery context.

## Contract Extension Output
Tracking state is emitted through `vision_output.extend_vision_output(...)` as:
- track list under `tracks`
- enriched detection objects (best effort track-linked)

