# Input Source Manager

## Mandatory Priority Logic
1. Priority 1: default webcam (`index=0`)
   - open success
   - at least one frame received
   - repeated frame responsiveness
   - minimum acceptable FPS
   - if healthy: webcam is used exclusively

2. Priority 2: local videos (automatic fallback)
   - recursive search under `C:\Users\Akash\Downloads\animal_welfare_dashboard\data\video`
   - formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.mpeg`, `.webm`
   - sort by newest modified timestamp
   - skip corrupted/unreadable files
   - choose next valid candidate automatically

3. Priority 3: no valid source
   - emits standardized message:
     - `No valid video source available.`
     - `Please connect a webcam or place a supported video inside ...`
   - application remains operational
   - no crash propagation

## Public APIs
- `probe_source()`
  - returns active capture + metadata + optional error.
- `read_frame_once()`
  - one-shot frame read under strict source priority.
- `get_frame_generator(max_frames=...)`
  - streaming generator with automatic source switching on failure.
- `get_frame_generator_for_source(source, ...)`
  - legacy compatibility stream for explicit source requests.

## Health and Recovery
- Webcam health probe includes responsiveness and FPS checks.
- Stream mode performs periodic source health verification.
- On read failure, manager attempts immediate failover.

## Centralization Rule
All `cv2.VideoCapture` usage in this project is routed through:
- `vision_intelligence/input_source_manager.py`

