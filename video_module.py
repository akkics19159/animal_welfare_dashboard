import cv2
import numpy as np
import os

# Try importing YOLO (Ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


SENTIENT_CLASSES = {
    "person",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "bird",
    "bear",
    "zebra",
    "giraffe",
    "elephant",
    "mouse",
    "rabbit",
    "lion",
    "tiger",
    "monkey",
    "pig",
    "goat",
}

DEFAULT_YOLO_MODEL = "yolov8n.pt"
FRAME_CAPTURE_COUNT = 12
MOTION_DIFF_THRESHOLD = 30

# Cache for YOLO models to prevent repeated loading
_MODEL_CACHE = {}

# -----------------------------
# Capture Functions
# -----------------------------
# -----------------------------

def capture_video(source=0):
    """
    Capture a single frame from webcam or video file.
    source = 0 → webcam
    source = "path/to/video.mp4" → video file
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")

    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame

    raise ValueError("Could not read a frame from the video source")


def capture_video_frames(source=0, frame_count=FRAME_CAPTURE_COUNT):
    """Capture a small burst of consecutive frames for motion analysis."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")

    frames = []
    count = 0
    while count < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1

    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames captured for motion analysis")

    return frames


def estimate_motion_score(frames):
    """Estimate motion intensity from consecutive frames.
    Returns a normalized score between 0 and 1.
    """
    if len(frames) < 2:
        return 0.0

    motion_values = []
    for index in range(1, len(frames)):
        prev_gray = cv2.cvtColor(frames[index - 1], cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, next_gray)
        _, thresh = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        motion_pixels = int(np.count_nonzero(thresh))
        motion_values.append(motion_pixels)

    frame_area = frames[0].shape[0] * frames[0].shape[1]
    avg_motion = float(np.mean(motion_values)) / frame_area
    return min(max(avg_motion, 0.0), 1.0)


def process_video_for_heatmap(source):
    """
    Process an entire video (or webcam stream) to generate a movement heatmap.
    Returns the final heatmap image.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")

    frames = []
    heatmap = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if heatmap is None:
            heatmap = gray
        else:
            heatmap += gray
    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames captured for heatmap")

    if heatmap is None:
        raise ValueError("Heatmap could not be initialized")

    heatmap = np.sum(frames, axis=0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)


# -----------------------------
# Detection Functions
# -----------------------------
# -----------------------------

def detect_any_species(frame, model_path=DEFAULT_YOLO_MODEL, confidence=0.25):
    """
    Detect humans, mammals, birds, and other sentient beings.
    Uses YOLO when available, otherwise falls back to basic human face detection.
    """
    detections = []
    if YOLO_AVAILABLE:
        try:
            model = YOLO(model_path)
            results = model(frame)
            if model_path not in _MODEL_CACHE:
                _MODEL_CACHE[model_path] = YOLO(model_path)
            model = _MODEL_CACHE[model_path]
            results = model(frame, verbose=False)  # Suppress console log noise
            for r in results:
                for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                    name = model.names[int(cls)]
                    if name in SENTIENT_CLASSES and float(conf) >= confidence:
                        detections.append(
                            {
                                "class": name,
                                "confidence": float(conf),
                                "xyxy": [float(v) for v in box.tolist()],
                            }
                        )
            return detections
        except Exception:
            detections = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return [
        {
            "class": "person",
            "confidence": 0.8,
            "xyxy": [int(x), int(y), int(x + w), int(y + h)],
        }
        for (x, y, w, h) in faces
    ]


def detect_animals(frame, **kwargs):
    """
    Wrapper expected by the dashboard.
    Returns detections for humans and other sentient beings.
    """
    return detect_any_species(frame, **kwargs)


def analyze_video_behavior(source=0, model_path=DEFAULT_YOLO_MODEL, confidence=0.25):
    """Analyze visible behavior and motion to estimate stress risk."""
    try:
        frames = capture_video_frames(source)
        frame = frames[0]
        detections = detect_any_species(frame, model_path=model_path, confidence=confidence)
        motion_score = estimate_motion_score(frames)
        return {
            "detections": detections,
            "motion_score": motion_score,
            "agitated": motion_score >= 0.08,
            "motion_agitation": motion_score >= 0.08,  # Standardized key for ontology
            "sentient_present": len(detections) > 0,   # Logic to enable other rules
            "frame_count": len(frames),
            "error": None,
        }
    except Exception as exc:
        return {
            "detections": [],
            "motion_score": 0.0,
            "agitated": False,
            "motion_agitation": False,
            "sentient_present": False,
            "frame_count": 0,
            "error": str(exc),
        }


def detect_animals_with_cascade(frame, cascade="haarcascade_frontalface_default.xml"):
    """
    Fallback detection using Haar cascades.
    Works for humans (faces) or if you have species-specific cascades.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + cascade)
    animals = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return animals


def annotate_frame(frame, detections, out_path):
    """Draw bounding boxes and labels on `frame` and save to `out_path`.

    `detections` expected to be a list of dicts with keys: 'xyxy', 'class', 'confidence'.
    """
    img = frame.copy()
    for det in detections:
        xy = det.get('xyxy')
        if not xy or len(xy) < 4:
            continue
        x1, y1, x2, y2 = map(int, xy[:4])
        label = f"{det.get('class', 'obj')}:{det.get('confidence', 0.0):.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(out_path, img)
    return out_path


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    video_path = r"C:\Users\Akash\Downloads\animal_welfare_dashboard\data\video\sample.mp4"
    # Use relative paths for portability
    video_path = os.path.join("data", "video", "sample.mp4")
    output_heatmap = os.path.join("data", "video", "movement_heatmap.jpg")

    frame = capture_video(video_path)
    heatmap = process_video_for_heatmap(video_path)
    cv2.imwrite(r"C:\Users\Akash\Downloads\animal_welfare_dashboard\data\video\movement_heatmap.jpg", heatmap)
    cv2.imwrite(output_heatmap, heatmap)

    detections = detect_any_species(frame)
    print("Detections:", detections)

    humans = detect_animals_with_cascade(frame)
    print("Detected humans:", len(humans))
