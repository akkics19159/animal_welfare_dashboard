# Quick Start Guide

## 1-Minute Setup

### Already have Python dependencies installed? Just need FFmpeg.

**Windows (5 minutes)**:
1. Download: https://www.gyan.dev/ffmpeg/builds/
   - Click "git-full" (direct link recommended)
2. Extract the ZIP to `C:\ffmpeg`
3. Add to PATH:
   - Open Windows Search → "Environment Variables"
   - Click "Edit the system environment variables"
   - Click "Environment Variables..." button
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and add: `C:\ffmpeg\bin`
   - Click OK → OK → OK
4. Verify: Open Command Prompt and type:
   ```bash
   ffmpeg -version
   ```
   (If this works, you're done!)

**macOS**:
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get install ffmpeg
```

### 2. Run the dependency check:
```bash
python check_dependencies.py
```
Should show all green ✓ checks.

### 3. Start the dashboard:
```bash
streamlit run dashboard.py
```
Opens at http://localhost:8501

### 4. Test with a sample video:
```bash
python analyze_multimodal.py
```
Generates `sample_multimodal_report.csv` with full analysis.

---

## What You Get

After FFmpeg is installed, the system will:

✓ **Detect suffering** via video (YOLO detections + motion analysis)  
✓ **Analyze audio** (emotion in cries/barks, acoustic features)  
✓ **Read sensors** (temperature, heart rate, humidity)  
✓ **Generate reports** (CSV with probability, explanations, annotated frames, audio clips)  
✓ **Explain decisions** (why probability is X%, not a black box)

---

## Current Issue (Why audio_score is 0.0)

The report shows:
```
audio_score: 0.0
audio_error: "FFmpeg not found in PATH"
audio_clip_error: "Failed to load audio and no fallback available"
```

**Why?** FFmpeg is needed to extract audio from `.mp4` files.  
**Fix?** Install FFmpeg using the steps above, then re-run.

---

## Verify FFmpeg is Accessible

After installation, open Command Prompt / Terminal and type:
```bash
ffmpeg -version
```

You should see version info. If not, FFmpeg isn't on PATH yet (repeat step 3 of Windows setup or reinstall on macOS/Linux).

---

## Next: Run Full Analysis

Once FFmpeg is installed:

```bash
python check_dependencies.py  # Verify all green ✓
python analyze_multimodal.py  # Full analysis with audio
```

Report will include:
- `audio_clip`: Path to loudest 3-second audio segment
- `audio_score`: Non-zero if distress detected
- `welfare_reasons`: Full explanation

---

## Stuck?

1. Run `python check_dependencies.py` and share the output
2. Check `sample_multimodal_report.csv` → `audio_error` column (most informative)
3. Verify FFmpeg: `ffmpeg -version` in terminal
4. Re-run: `python analyze_multimodal.py`

