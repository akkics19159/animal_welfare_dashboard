# TODO_VISION_INTELLIGENCE

## Phase 1 — Input + Contract foundations
- [ ] Harden `InputSourceManager` corruption detection + fallback priority + no-crash behavior.
- [ ] Ensure no other module touches `cv2.VideoCapture` (verify by search).

## Phase 2 — Tracking modernization
- [x] Implement BoT-SORT primary + ByteTrack configurable fallback + centroid fallback (no DeepSORT).

- [ ] Ensure `TrackState` + output contract includes all required tracking fields.
- [ ] Add occlusion / lost / reacquired state logic.

## Phase 3 — Vision pipeline modernization
- [ ] Add continuous `run_stream()` mode while keeping `run_once()` for compatibility.
- [ ] Compute FPS, occupancy, counting, and motion stats incrementally in streaming.

## Phase 4 — Output contract + fusion/reasoning compatibility
- [ ] Extend `extend_vision_output()` to include required per-track + aggregate keys without removing legacy fields.
- [ ] Ensure fusion/reasoning compatibility keys are exposed.

## Phase 5 — Counting + trajectory analytics
- [ ] Ensure counting accuracy uses persistent tracks and dwell analytics.
- [ ] Ensure trajectory analytics per Track ID includes distance, speed, smoothness, stationary/stationary duration.

## Phase 6 — Tests + documentation
- [ ] Add unit tests: webcam priority, fallback selection, corrupted video skipping, persistent IDs, counting, trajectory, occlusion recovery.
- [ ] Add integration tests: no input, webcam unavailable with local video, smoke contract compatibility.
- [ ] Update docs: VISION_ARCHITECTURE.md, INPUT_SOURCE_MANAGER.md, TRACKING_GUIDE.md, COUNTING_GUIDE.md, VISION_FLOW.md, VISION_MIGRATION_REPORT.md.

## Phase 7 — Verification
- [ ] Run `pytest` and fix any contract-test failures.

