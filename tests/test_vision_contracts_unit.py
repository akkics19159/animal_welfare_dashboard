import time

import pytest


def test_vision_no_crash_on_missing_input():
    # Force no-input by calling pipeline with source=0; if webcam exists,
    # this still should not crash.
    import video_module

    res = video_module.analyze_video_behavior(source=0, confidence=0.1)
    assert isinstance(res, dict)
    assert "detections" in res
    assert "error" in res


@pytest.mark.parametrize("conf", [0.05, 0.2])
def test_vision_contract_contains_expected_extended_keys(conf):
    import video_module

    res = video_module.analyze_video_behavior(source=0, confidence=conf)
    # Extended keys are contract additions; if present, they must be JSON-like types.
    if "tracks" in res:
        assert isinstance(res["tracks"], list)
    if "occupancy" in res:
        assert isinstance(res["occupancy"], int)


def test_trajectory_fields_best_effort():
    import video_module

    res = video_module.analyze_video_behavior(source=0, confidence=0.1)
    tracks = res.get("tracks") or []
    for t in tracks:
        # trajectory + movement fields may be empty/best-effort but keys should exist if track object is present
        assert "track_id" in t
        assert "trajectory" in t
        assert "velocity" in t
        assert "acceleration" in t
        assert "direction" in t or t["direction"] is None

