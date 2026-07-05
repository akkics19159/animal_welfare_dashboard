# ⚡ Dashboard Quick Start

## Launch Dashboard (30 seconds)

```bash
cd C:\Users\Akash\Downloads\animal_welfare_dashboard
streamlit run dashboard.py
```

**Opens at**: http://localhost:8501

---

## First Steps

1. **See Live Metrics** (top of page)
   - Welfare Score, Probability, Video/Audio Scores
   - Updates after each analysis run

2. **Run Analysis** (large blue button)
   - Click "▶️ RUN FULL WELFARE CHECK"
   - Automatically analyzes video + audio + sensors
   - Falls back to sample video if webcam empty

3. **View Results**
   - **Summary tab**: Alert level, modality breakdown, explanation
   - **Trends tab**: Historical graphs over time
   - **Details tab**: Breakdown of each modality
   - **History tab**: Full table of all analyses

4. **Customize Settings** (sidebar)
   - Adjust weights (video/audio/sensor importance)
   - Change thresholds (critical/warning levels)
   - Toggle simulated sensors
   - Enable raw score debugging

5. **Export Data** (📥 Export button)
   - Download all history as CSV
   - Use in Excel/Tableau/BI tools

---

## What You'll See

### Alert Levels
- 🔴 **CRITICAL** (Prob > 0.7): Immediate intervention needed
- 🟠 **WARNING** (Prob > 0.4): Close monitoring recommended
- 🟢 **NORMAL** (Prob ≤ 0.4): Stable welfare

### Charts (in Trends tab)
- Red line shows suffering probability over time
- Multi-color lines show video/audio/sensor contributions
- All charts are interactive (zoom, hover, export)

### Data Persistence
- All analyses automatically saved to `welfare_analysis_history.csv`
- No manual export needed to build trends
- Click "Clear History" in sidebar to reset

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Dashboard won't start | `pip install streamlit plotly` |
| No camera detected | Ensure camera connected and enabled |
| Audio score always 0 | Install FFmpeg (see QUICKSTART.md) |
| Import errors | Run `python check_dependencies.py` |

---

## Key Features Highlighted

✅ **Real-time monitoring** - Live metrics update  
✅ **Historical tracking** - CSV persistence  
✅ **Advanced analytics** - Plotly charts  
✅ **Customizable settings** - Adjust weights/thresholds  
✅ **Multi-modality** - Video + Audio + Sensors  
✅ **Explainability** - AI-generated explanations  
✅ **Export ready** - Download as CSV  
✅ **Error handling** - Graceful fallbacks  

---

## Default Configuration

```
Video Weight:        0.4 (40%)
Audio Weight:        0.4 (40%)
Sensor Weight:       0.2 (20%)
Ontology Strength:   0.6 (60% rule-based regularization)
Critical Threshold:  0.7
Warning Threshold:   0.4
Sensors:             Simulated (random values)
```

Adjust these in the sidebar for your use case.

---

## Sample Workflow

```
1. Launch dashboard
   ↓
2. Run welfare check
   ↓
3. See alert level (green/orange/red)
   ↓
4. Check Details tab for modality breakdown
   ↓
5. Go to Trends tab to see history
   ↓
6. Export CSV for reporting
   ↓
7. Adjust settings and re-run
```

---

**Dashboard is ready to use!** 🚀
