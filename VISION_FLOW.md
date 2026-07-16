# Vision Flow

## Single-Cycle Execution (`run_once`)
1. UI/backend calls `video_module.analyze_video_behavior(...)`
2. Delegates to `VisionPipeline.run_once(...)`
3. Input Source Manager performs strict source resolution:
  - webcam health probe
  - recursive local fallback (newest valid video)
  - standardized no-source error if none
4. Frame sample batch collected (`max_frames` bounded by pipeline limits)
5. YOLO detection runs (batch-capable, last-frame output used for current cycle)
6. Tracker updates identity state (BoT-SORT/ByteTrack/fallback)
7. Trajectory analytics computed per track
8. Intelligent counter computes occupancy, unique counts, ROI/line metrics
9. Behavior flags derived (overcrowding, isolation, long inactivity, escape attempt, repeated circling, abnormal movement)
10. Output extension preserves legacy fields and appends advanced fields

## Streaming Execution (`run_stream`)
- Continuous source frames are processed one by one.
- Source switching is handled by Input Source Manager when failures occur.

## Output Groups
- Legacy: detection and motion flags for existing consumers
- Tracking: persistent IDs, confidence, occlusion/recovery, lifetime
- Motion: velocity, acceleration, direction, distance, movement density
- Counting: occupancy, unique, species-wise, region-wise, entry/exit, dwell
- Source metadata: source type and selected source path/index

