import os
import json
from video_module import analyze_video_behavior, annotate_frame
from audio_module import extract_audio_file_features, detect_distress, save_loudest_audio_segment
from sensors import read_sensors
from reports import generate_report
from multimodal_engine import analyze_combined
import pathlib


VIDEO_PATH = os.path.join("data", "video", "puppy crying.mp4")
REPORT_FILE = "sample_multimodal_report.csv"


def main():
    print(f"Multimodal analyzing video: {VIDEO_PATH}")
    if not os.path.exists(VIDEO_PATH):
        print("ERROR: sample video not found at:", VIDEO_PATH)
        return

    video_result = analyze_video_behavior(source=VIDEO_PATH)
    audio_error = None
    try:
        audio_features = extract_audio_file_features(VIDEO_PATH)
        audio_result = detect_distress(audio_features)
    except Exception as exc:
        audio_error = str(exc)
        audio_result = {"distress": False, "score": 0.0, "error": audio_error}

    sensors = read_sensors(use_simulation=True)

    # Save an annotated frame with detections
    out_dir = pathlib.Path("analysis_outputs")
    out_dir.mkdir(exist_ok=True)
    annotated_frame_path = None
    audio_clip_path = None
    if video_result.get("detections"):
        try:
            # fallback: annotate last frame from analyze_video_behavior output if available
            # For simplicity, re-run capture to get a frame
            try:
                # reuse capture_video to get a single frame
                from video_module import capture_video
                frame = capture_video(VIDEO_PATH)
            except Exception:
                frame = None

            if frame is not None:
                annotated_frame_path = str(out_dir / "annotated_frame.jpg")
                annotate_frame(frame, video_result.get("detections", []), annotated_frame_path)
        except Exception:
            annotated_frame_path = None

    # Save the loudest audio clip for human verification
    audio_clip_error = None
    try:
        audio_clip_path = str(out_dir / "loudest_segment.wav")
        save_loudest_audio_segment(VIDEO_PATH, audio_clip_path)
    except Exception as exc:
        audio_clip_path = None
        audio_clip_error = str(exc)

    combined = analyze_combined(video_result, audio_result, sensors, ontology_strength=0.6)

    summary = {
        "video": VIDEO_PATH,
        "probability": combined["probability"],
        "raw_probability": combined["raw_probability"],
        "video_score": combined["modality_scores"]["video_score"],
        "audio_score": combined["modality_scores"]["audio_score"],
        "sensor_score": combined["modality_scores"]["sensor_score"],
        "annotated_frame": annotated_frame_path,
        "audio_clip": audio_clip_path,
        "welfare_reasons": " | ".join(combined["explanations"]),
        "audio_error": audio_error,
        "audio_clip_error": audio_clip_error,
        "sensor_note": "simulated (use_simulation=True)",
    }

    print(json.dumps(summary, indent=2))
    generate_report([summary], filename=REPORT_FILE)
    print(f"Saved multimodal report to {REPORT_FILE}")


if __name__ == "__main__":
    main()
