# Vision Architecture

## Objective
Deliver a production-ready Vision Acquisition + Vision Intelligence subsystem for multimodal welfare monitoring while preserving all legacy interfaces used by dashboard, fusion, and reasoning layers.

## Design Principles
- Webcam-first source priority with autonomous fallback and no user intervention.
- Centralized capture control: all frame acquisition is routed through Input Source Manager.
- Backward compatibility first: legacy keys remain intact, new fields are additive.
- Plugin-friendly evolution path for pose estimation and behavior recognition.

## Runtime Architecture
1. Acquisition Layer
   - `vision_intelligence/input_source_manager.py`
   - Responsibilities:
     - webcam discovery and health validation (open/read/responsive/FPS)
     - recursive local video discovery (`.mp4`, `.avi`, `.mov`, `.mkv`, `.mpeg`, `.webm`)
     - corrupted video skip + next-valid fallback
     - frame streaming + source switching

2. Detection Layer
   - `vision_intelligence/yolo_detector.py`
   - YOLO retained and loaded once per process via cache.
   - Auto GPU selection (`cuda`) with CPU fallback.
   - Supports single-frame and batch prediction paths.

3. Tracking Layer
   - `vision_intelligence/tracking.py`
   - Tracker strategy order:
     - primary: BoT-SORT (if available)
     - fallback: ByteTrack (if available)
     - safe fallback: internal IoU/centroid tracker
   - Outputs persistent track states with identity continuity.

4. Analytics Layer
   - `vision_intelligence/trajectory.py`: velocity, acceleration, direction, distance, stationary time, movement density.
   - `vision_intelligence/counting.py`: occupancy, unique individuals, species-wise, region-wise, entry/exit, max/avg occupancy.

5. Contract Layer
   - `vision_intelligence/vision_output.py`
   - Extends output payload with track-level and aggregate features without removing legacy fields.

6. Orchestration Layer
   - `vision_intelligence/vision_pipeline.py`
   - `run_once()` for compatibility.
   - `run_stream()` for continuous monitoring.

## Compatibility Mapping
- Entry point remains `video_module.analyze_video_behavior(...)`.
- Legacy keys retained:
  - `detections`, `motion_score`, `agitated`, `motion_agitation`, `sentient_present`, `frame_count`, `error`
- Extended keys include:
  - `tracks`, `occupancy`, `entry_count`, `exit_count`, `species_wise_count`, `region_wise_count`, movement and dwell features.

## Integration Notes
- `multimodal_engine.py` now consumes enriched vision features (`occupancy`, velocity/acceleration, dwell, tracking confidence) while retaining old scoring behavior.
- API camera health probe now uses Input Source Manager, removing direct `cv2.VideoCapture` calls outside the manager.

