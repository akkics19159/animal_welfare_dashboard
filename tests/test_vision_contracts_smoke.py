import pytest


def test_vision_output_contract_smoke():
    # Smoke test: ensure analyze_video_behavior returns legacy keys
    import video_module

    res = video_module.analyze_video_behavior(source=0, confidence=0.2)
    assert isinstance(res, dict)

    # Legacy keys expected by fusion/reasoning/UI
    for k in [
        "detections",
        "motion_score",
        "agitated",
        "motion_agitation",
        "sentient_present",
        "frame_count",
        "error",
    ]:
        assert k in res

    assert isinstance(res.get("detections"), list)


def test_vision_extended_fields_optional():
    import video_module

    res = video_module.analyze_video_behavior(source=0, confidence=0.2)

    # Extended fields are best-effort; verify absence doesn't break API.
    # If present, they must be JSON-serializable-like.
    if "tracks" in res:
        assert isinstance(res["tracks"], list)

    # occupancy is expected to be present by the new contract
    if "occupancy" in res:
        assert isinstance(res["occupancy"], int)


