# Counting Guide

## Persistent Counting Metrics
Implemented in `vision_intelligence/counting.py`:
- Current Occupancy
- Unique Individuals (lifetime)
- Species-wise Count
- Region-wise Count
- Entry Count
- Exit Count
- Maximum Occupancy
- Average Occupancy

## ROI and Line Support
Counter accepts optional geometry config:
- regions: polygon zones
- entry lines: signed line-crossing counts
- exit lines: signed line-crossing counts

Generated aggregates:
- `region_wise_count`
- `region_dwell_time`
- `region_movement_stats`

## Counting Semantics
- Visible individual: `lost_frames == 0`
- If entry/exit lines are configured:
  - line crossings determine entry/exit counts
- Else:
  - visibility transitions determine entry/exit counts

## Output Compatibility
Counting statistics are additive and included in the existing vision payload.
Legacy consumers remain unaffected.

