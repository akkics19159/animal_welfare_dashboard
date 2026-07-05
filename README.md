# Animal Welfare Dashboard

A multimodal detection system for identifying signs of suffering in sentient beings (humans, animals, birds) using video, audio, and sensor data.

## Features

- **Webcam-first analysis**: Automatically captures from webcam; falls back to sample video if idle
- **Multimodal fusion**: Combines video detection, audio distress analysis, and sensor readings
- **Ontology-regularized scoring**: Explainable suffering probability using rule-based regularization
- **Annotated outputs**: Saves annotated frames with detection overlays and audio clips for review
- **CSV reporting**: Generates timestamped welfare reports with diagnostic information

## Architecture

```
├── video_module.py       → Video capture, motion scoring, YOLO detection, heatmaps
├── audio_module.py       → Live/file audio capture, distress detection, segment extraction
├── sensors.py            → Simulated sensor readings (temperature, humidity, heart rate)
├── multimodal_engine.py  → Late fusion + ontology-based regularization
├── ontology.py           → Suffering indicator rules and explanations
├── dashboard.py          → Streamlit UI (webcam-first, live analysis)
├── analyze_multimodal.py → Batch analysis runner for sample videos
└── reports.py            → CSV report generation
```

## Installation

### Prerequisites
- Python 3.8+
- FFmpeg (required for robust audio extraction)

### Step 1: Install Python dependencies

```bash
pip install -r requirements.txt
```

**Note**: The following packages are key:
- `ultralytics` → YOLOv8 for object detection (auto-downloads weights)
- `librosa` + audio backends → Audio feature extraction
- `moviepy` → Video/audio processing fallback
- `streamlit` → Web UI
- `opencv-python` → Video processing

### Step 2: Install FFmpeg (CRITICAL for audio support)

#### Windows (Recommended approach)
1. Download from: https://www.gyan.dev/ffmpeg/builds/
   - Choose "git-full" (includes all libraries and tools)
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your system PATH:
   - Right-click "This PC" → Properties → Advanced system settings → Environment Variables
   - Edit `PATH` and add the FFmpeg bin directory
4. Verify installation:
   ```bash
   ffmpeg -version
   ```

#### macOS
```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### Step 3: Verify Setup
Run the dependency check:
```bash
python check_dependencies.py
```

This will verify FFmpeg, audio backends, and other critical libraries.

## Usage

### Option 1: Interactive Streamlit Dashboard (Recommended for live analysis)
```bash
streamlit run dashboard.py
```
- Opens a local web UI (usually http://localhost:8501)
- Starts with webcam; falls back to sample video if idle for >2s
- Live updates with suffering probability, video/audio scores, and sensor readings
- Saves reports to `welfare_report.csv`

### Option 2: Batch Analysis (for video files)
```bash
python analyze_multimodal.py
```
Analyzes `data/video/puppy crying.mp4` and saves:
- `sample_multimodal_report.csv` → Full report with probability, modality scores, diagnostic info
- `analysis_outputs/annotated_frame.jpg` → Detection overlay on a representative frame
- `analysis_outputs/loudest_segment.wav` → Loudest 3-second audio clip (if audio extraction succeeds)

### Option 3: Video-Only Analysis (no audio/sensors)
```bash
python analyze_sample.py
```
Produces `sample_analysis_report.csv` with video features and motion scoring.

## Report Output

Each report row contains:
- `video` → Path to analyzed video
- `probability` → Adjusted suffering probability (0–1, regularized by ontology)
- `raw_probability` → Raw fused score before ontology adjustment
- `video_score` → Video-based welfare concern (motion, detections)
- `audio_score` → Audio-based welfare concern (distress acoustic features)
- `sensor_score` → Sensor-based welfare concern (temperature, heart rate, humidity)
- `annotated_frame` → Path to frame with detection overlays
- `audio_clip` → Path to loudest audio segment (null if extraction fails)
- `welfare_reasons` → Human-readable explanation of the probability
- `audio_error` → Error message if audio extraction failed (diagnostic)
- `audio_clip_error` → Error message if audio segment save failed (diagnostic)
- `sensor_note` → Notes on sensor source (e.g., "simulated")
- `checked_at` → ISO timestamp of analysis

**Example**:
```csv
video,probability,raw_probability,video_score,audio_score,sensor_score,annotated_frame,audio_clip,welfare_reasons,audio_error,audio_clip_error,sensor_note,checked_at
data\video\puppy crying.mp4,0.1857,0.1643,0.2107,0.0,0.4,analysis_outputs\annotated_frame.jpg,null,"Raw fused probability: 0.164 | Ontology score: 0.200 | High temperature may indicate stress | Adjusted probability: 0.186","Could not load audio from data\video\puppy crying.mp4: (moviepy not available and ffmpeg not found)","Failed to load audio and no fallback available (moviepy or ffmpeg)","simulated (use_simulation=True)",2026-05-31T10:15:30.123456
```

## Troubleshooting

### Audio extraction fails
**Error**: "Could not load audio from ... (moviepy not available and ffmpeg not found)"

**Solutions**:
1. Install FFmpeg (see "Step 2" above)
2. Verify FFmpeg is on PATH: `ffmpeg -version`
3. Re-run analysis

### No detections in video
**Cause**: YOLOv8 model may be downloading on first run, or the object is not a common class.

**Solutions**:
- Check `annotated_frame.jpg` to see what was detected
- Verify the video file is readable: `ffmpeg -i <video_file>`

### Streamlit dashboard won't start
**Cause**: Port 8501 in use or missing dependencies.

**Solutions**:
```bash
# Run on a different port
streamlit run dashboard.py --server.port 8502
# Or reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Sensor score is always 0.0
**Cause**: Sensors are simulated by default (random values within normal range).

**Solution**: To integrate real sensors, edit `sensors.py` and set `use_simulation=False` with actual sensor APIs.

## Configuration Tuning

Edit the following files to customize behavior:

### Video sensitivity (`video_module.py`)
- `IDLE_MOTION_THRESHOLD` → Reduce to detect smaller motions
- Motion agitation thresholds in `analyze_video_behavior()`

### Audio sensitivity (`audio_module.py`)
- `energy_threshold`, `zcr_threshold`, `centroid_threshold` in `detect_distress()`
- Lower thresholds = more sensitive to distress sounds

### Fusion weights (`multimodal_engine.py`)
- Adjust `weights` dict in `fuse_scores()` to prioritize video/audio/sensors
- Change `ontology_strength` in `analyze_combined()` for more/less rule-based regularization

### Ontology rules (`ontology.py`)
- Add domain-specific rules to `ONTOLOGY_RULES` for higher accuracy

## Next Steps for Production

- [ ] Collect labeled training data (videos/audio with suffering annotations)
- [ ] Fine-tune audio classifier (e.g., PANNs) on distress sounds
- [ ] Add pose estimation (MediaPipe/OpenPose) for body language analysis
- [ ] Integrate action recognition (SlowFast/X3D) for behavioral patterns
- [ ] Human-in-the-loop verification workflow
- [ ] Privacy & ethics review (consent, data retention, deployment policy)
- [ ] Edge deployment (Docker/K8s) or cloud integration (AWS/GCP)

## License

[Your license here]

## Support

For issues or questions, please refer to the error messages in the CSV report's `audio_error` and `audio_clip_error` fields, or run `python check_dependencies.py` to diagnose the environment.
