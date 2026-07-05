import os
import json
from pathlib import Path
from video_module import analyze_video_behavior
from reports import generate_report
from audio_module import extract_audio_file_features, detect_distress
from sensors import read_sensors
from analysis_common import build_flags, run_analysis_bundle, build_canonical_report_row, detect_sensor_note


BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "data" / "video" / "puppy crying.mp4"
REPORT_FILE = BASE_DIR / "sample_analysis_report.csv"


def main():
    print(f"Analyzing video: {VIDEO_PATH}")
    if not VIDEO_PATH.exists():
        print("ERROR: sample video not found at:", VIDEO_PATH)
        return

    video_result = analyze_video_behavior(source=str(VIDEO_PATH))

    # Audio analysis from the file
    try:
        audio_features = extract_audio_file_features(str(VIDEO_PATH))
        audio_result = detect_distress(audio_features)
    except Exception as exc:
        audio_features = None
        audio_result = {"distress": False, "score": 0.0, "error": str(exc)}

    # Simulated sensors for offline analysis
    sensors = read_sensors(use_simulation=True)

    bundle = run_analysis_bundle(
        video_result,
        audio_result,
        sensors,
        ontology_strength=0.6,
    )
    video_flags = bundle["flags"]["video"]
    audio_flags = bundle["flags"]["audio"]
    sensor_flags = bundle["flags"]["sensor"]

    welfare_score = (1.0 - bundle["combined"]["probability"]) * 100.0

    summary = {
        "video": str(VIDEO_PATH),
        "probability": bundle["combined"]["probability"],
        "raw_probability": bundle["combined"]["raw_probability"],
        "video_score": bundle["combined"]["modality_scores"]["video_score"],
        "audio_score": bundle["combined"]["modality_scores"]["audio_score"],
        "sensor_score": bundle["combined"]["modality_scores"]["sensor_score"],
        "welfare_score": welfare_score,
        "temp": sensors["temp"],
        "humidity": sensors["humidity"],
        "heart_rate": sensors["heart_rate"],
        "audio_error": audio_result.get("error"),
        "sensor_note": detect_sensor_note(True, {"error": None}),
        "reasons": "; ".join(bundle["combined"]["explanations"]),
    }

    print("Analysis result summary:")
    print(json.dumps(summary, indent=2))

    report = [
        build_canonical_report_row(
            source=str(VIDEO_PATH),
            combined=bundle["combined"],
            audio_error=audio_result.get("error"),
            sensor_note=detect_sensor_note(True, {"error": None}),
        )
    ]
    generate_report(report, filename=str(REPORT_FILE))
    print(f"Report written to {REPORT_FILE}")


if __name__ == "__main__":
    main()
