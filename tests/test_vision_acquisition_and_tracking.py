from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vision_intelligence.counting import IntelligentCounter
from vision_intelligence.input_source_manager import InputSourceManager, NO_SOURCE_ERROR
from vision_intelligence.tracking import TrackerPipeline


class _FakeCapture:
    def __init__(self, frames, opened=True, fps=15.0, width=640, height=480):
        self._frames = list(frames)
        self._opened = opened
        self._fps = fps
        self._width = width
        self._height = height

    def isOpened(self):
        return bool(self._opened)

    def read(self):
        if not self._opened or not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def get(self, prop_id):
        # cv2 constants used by InputSourceManager
        if prop_id == 5:  # CAP_PROP_FPS
            return self._fps
        if prop_id == 3:  # CAP_PROP_FRAME_WIDTH
            return self._width
        if prop_id == 4:  # CAP_PROP_FRAME_HEIGHT
            return self._height
        return 0.0

    def release(self):
        self._opened = False


def _frame(value: int = 0):
    return np.full((24, 24, 3), value, dtype=np.uint8)


def test_webcam_priority_over_local_videos(tmp_path, monkeypatch):
    video_dir = tmp_path / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    newest = video_dir / "newest.mp4"
    newest.write_bytes(b"x")

    captures = {
        0: _FakeCapture([_frame(1), _frame(2), _frame(3)], opened=True),
        str(newest): _FakeCapture([_frame(9)], opened=True),
    }

    def _factory(source):
        return captures.get(source, _FakeCapture([], opened=False))

    monkeypatch.setattr("vision_intelligence.input_source_manager.cv2.VideoCapture", _factory)

    mgr = InputSourceManager(local_video_dir=str(video_dir), min_acceptable_fps=1.0)
    cap, meta, err = mgr.probe_source()

    assert err is None
    assert cap is not None
    assert meta is not None
    assert meta.source_type == "webcam"
    assert meta.source == "0"


def test_fallback_skips_corrupted_and_uses_next_valid(tmp_path, monkeypatch):
    video_dir = tmp_path / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    corrupted = video_dir / "z_latest.mp4"
    valid = video_dir / "a_older.mp4"
    corrupted.write_bytes(b"bad")
    valid.write_bytes(b"ok")

    # Ensure ordering by mtime: corrupted is newer than valid.
    os.utime(str(valid), (1_000_000_000, 1_000_000_000))
    os.utime(str(corrupted), (1_100_000_000, 1_100_000_000))

    def _factory(source):
        if source == 0:
            return _FakeCapture([], opened=False)
        if source == str(corrupted):
            return _FakeCapture([], opened=False)
        if source == str(valid):
            return _FakeCapture([_frame(7), _frame(8)], opened=True)
        return _FakeCapture([], opened=False)

    monkeypatch.setattr("vision_intelligence.input_source_manager.cv2.VideoCapture", _factory)

    mgr = InputSourceManager(local_video_dir=str(video_dir), min_acceptable_fps=1.0)
    cap, meta, err = mgr.probe_source()

    assert err is None
    assert cap is not None
    assert meta is not None
    assert meta.source_type == "video"
    assert Path(meta.source).name == "a_older.mp4"


def test_no_source_returns_expected_error(tmp_path, monkeypatch):
    video_dir = tmp_path / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "vision_intelligence.input_source_manager.cv2.VideoCapture",
        lambda _source: _FakeCapture([], opened=False),
    )

    mgr = InputSourceManager(local_video_dir=str(video_dir), min_acceptable_fps=1.0)
    frame, meta, err = mgr.read_frame_once()

    assert frame is None
    assert meta is None
    assert err == NO_SOURCE_ERROR


def test_tracker_persistent_id_and_occlusion_recovery():
    tracker = TrackerPipeline(frame_rate=10.0, max_lost_frames=3, iou_match_threshold=0.2, try_botsort=False, try_bytetrack=False)

    # Initial detection creates track id 1.
    trk1 = tracker.update(None, [{"xyxy": [10, 10, 20, 20], "class": "dog", "confidence": 0.9}], timestamp=1.0)
    assert 1 in trk1

    # Small movement keeps same track id.
    trk2 = tracker.update(None, [{"xyxy": [11, 10, 21, 20], "class": "dog", "confidence": 0.88}], timestamp=1.1)
    assert 1 in trk2

    # Occlusion frame.
    trk3 = tracker.update(None, [], timestamp=1.2)
    assert trk3[1].lost_frames == 1

    # Reappearance near previous position should recover same track.
    trk4 = tracker.update(None, [{"xyxy": [12, 10, 22, 20], "class": "dog", "confidence": 0.9}], timestamp=1.3)
    assert 1 in trk4
    assert trk4[1].lost_frames == 0
    assert trk4[1].was_occluded is True
    assert len(trk4[1].trajectory) >= 3


def test_counting_accuracy_and_trajectory_metrics():
    counter = IntelligentCounter()

    t1 = SimpleNamespace(
        class_name="dog",
        lost_frames=0,
        bbox_xyxy=[0.0, 0.0, 10.0, 10.0],
        dwell_time=2.0,
        velocity_xy=(1.0, 0.0),
        acceleration_xy=(0.2, 0.0),
        movement_density=0.7,
    )
    t2 = SimpleNamespace(
        class_name="cat",
        lost_frames=0,
        bbox_xyxy=[20.0, 20.0, 30.0, 30.0],
        dwell_time=1.0,
        velocity_xy=(0.5, 0.1),
        acceleration_xy=(0.1, 0.0),
        movement_density=0.4,
    )

    s1 = counter.update({1: t1})
    s2 = counter.update({1: t1, 2: t2})
    s3 = counter.update({2: t2})

    assert s1.current_occupancy == 1
    assert s2.current_occupancy == 2
    assert s3.current_occupancy == 1
    assert s3.total_unique_individuals == 2
    assert s3.entry_count == 2
    assert s3.exit_count == 1
    assert s3.maximum_occupancy == 2
    assert s3.average_occupancy > 1.0


def test_video_module_contract_compatibility():
    import video_module

    res = video_module.analyze_video_behavior(source=0, confidence=0.1)
    assert isinstance(res, dict)
    for key in ["detections", "motion_score", "agitated", "motion_agitation", "sentient_present", "frame_count", "error"]:
        assert key in res

    for track in res.get("tracks", []) or []:
        for k in [
            "track_id",
            "species",
            "bbox",
            "confidence",
            "tracking_confidence",
            "trajectory",
            "velocity",
            "acceleration",
            "direction",
            "entry_time",
            "exit_time",
            "dwell_time",
            "occupancy",
            "visibility_score",
        ]:
            assert k in track
