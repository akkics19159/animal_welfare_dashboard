# Vision Intelligence Subsystem

This folder contains the modernized vision subsystem used by `video_module.py`.

Key components:
- `input_source_manager.py`: webcam-first sourcing with fallback to newest valid local video.
- `yolo_detector.py`: YOLOv8+ detection with device selection and caching.
- `tracking.py`: BoT-SORT primary + ByteTrack fallback; falls back to a lightweight tracker if unavailable.
- `trajectory.py`: trajectory analytics (distance, speed, stationary duration, smoothness).
- `counting.py`: intelligent counting using persistent track visibility.
- `vision_output.py`: extends legacy output contract without removing existing keys.
- `vision_pipeline.py`: orchestrates full flow for a single analysis run.

