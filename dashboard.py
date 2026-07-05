"""
Advanced Sentient Being Welfare Monitoring Dashboard - 2026 Edition
Multimodal suffering detection with real-time monitoring, analytics, and explainability.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import csv

from video_module import analyze_video_behavior

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_HISTORY_PATH = BASE_DIR / "welfare_analysis_history.csv"
DEFAULT_VIDEO_PATH = BASE_DIR / "data" / "video" / "puppy crying.mp4"
from audio_module import capture_audio, detect_distress, extract_audio_file_features, ffmpeg_available
from sensors import read_sensors
from multimodal_engine import analyze_combined
from reports import generate_report

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="🔍 Welfare Dashboard 2026",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern look
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .alert-high {
        background-color: #ff6b6b;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
    }
    .alert-medium {
        background-color: #ffa500;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
    }
    .alert-low {
        background-color: #51cf66;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_analysis_history(filepath=DEFAULT_HISTORY_PATH):
    """Load historical analysis data."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_columns = ['timestamp', 'welfare_score', 'video_score', 'audio_score', 'sensor_score', 'probability']
    if path.exists():
        df = pd.read_csv(path)
        for col in expected_columns:
            if col not in df.columns:
                df[col] = np.nan
        return df[expected_columns]
    return pd.DataFrame(columns=expected_columns)


def save_analysis_record(record, filepath=DEFAULT_HISTORY_PATH):
    """Append analysis record to history."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(record)

def get_alert_level(probability):
    """Determine alert level based on probability."""
    if probability > 0.7:
        return "🔴 CRITICAL", "alert-high"
    elif probability > 0.4:
        return "🟠 WARNING", "alert-medium"
    else:
        return "🟢 NORMAL", "alert-low"

def format_timestamp():
    """Get formatted current timestamp."""
    return datetime.now().isoformat()

# ============================================================================
# SIDEBAR - SYSTEM STATUS & SETTINGS
# ============================================================================

with st.sidebar:
    st.title("⚙️ System Control Panel")
    
    # System Diagnostics
    st.subheader("📊 System Status")
    col1, col2 = st.columns(2)
    
    ffmpeg_path = ffmpeg_available()
    ffmpeg_status = "✓" if ffmpeg_path else "✗"
    with col1:
        st.metric("FFmpeg", ffmpeg_status)
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        camera_ok = cap.isOpened()
        cap.release()
        camera_status = "✓" if camera_ok else "✗"
    except:
        camera_status = "✗"
    
    with col2:
        st.metric("Camera", camera_status)
    
    if not ffmpeg_path:
        st.warning("⚠️ FFmpeg not found - audio processing disabled")
    
    # Settings
    st.divider()
    st.subheader("⚡ Analysis Settings")
    
    video_weight = st.slider("Video Weight", 0.0, 1.0, 0.4, 0.1)
    audio_weight = st.slider("Audio Weight", 0.0, 1.0, 0.4, 0.1)
    sensor_weight = st.slider("Sensor Weight", 0.0, 1.0, 0.2, 0.1)
    ontology_strength = st.slider("Ontology Strength", 0.0, 1.0, 0.6, 0.1)
    
    # Thresholds
    st.divider()
    st.subheader("🎚️ Alert Thresholds")
    
    critical_threshold = st.slider("Critical Threshold", 0.5, 1.0, 0.7, 0.05)
    warning_threshold = st.slider("Warning Threshold", 0.2, 0.8, 0.4, 0.05)
    
    # Data Options
    st.divider()
    st.subheader("💾 Data & Export")
    
    use_simulated_sensors = st.checkbox("Use Simulated Sensors", True)
    show_raw_scores = st.checkbox("Show Raw Scores", False)
    
    if st.button("🗑️ Clear History", key="clear_history"):
        if DEFAULT_HISTORY_PATH.exists():
            DEFAULT_HISTORY_PATH.unlink()
            st.success("History cleared!")

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.title("🔍 Sentient Being Welfare Monitoring Dashboard")
st.markdown("**Advanced Multimodal Analysis System - 2026 Edition**")

# Load historical data
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

history_df = load_analysis_history(DEFAULT_HISTORY_PATH)

# ============================================================================
# TOP METRICS ROW
# ============================================================================

st.subheader("📈 Live Metrics")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

if len(history_df) > 0:
    latest = history_df.iloc[-1]
    with metric_col1:
        st.metric("Current Welfare Score", f"{latest['welfare_score']:.1f}%", delta=None)
    with metric_col2:
        st.metric("Probability", f"{latest['probability']:.2%}", delta=None)
    with metric_col3:
        st.metric("Video Score", f"{latest['video_score']:.2f}")
    with metric_col4:
        st.metric("Audio Score", f"{latest['audio_score']:.2f}")
else:
    with metric_col1:
        st.metric("Current Welfare Score", "—", delta=None)
    with metric_col2:
        st.metric("Probability", "—", delta=None)
    with metric_col3:
        st.metric("Video Score", "—")
    with metric_col4:
        st.metric("Audio Score", "—")

# ============================================================================
# MAIN ACTION BUTTON
# ============================================================================

col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])

with col_btn1:
    if st.button("▶️ RUN FULL WELFARE CHECK", key="main_button", help="Analyze video, audio, and sensors"):
        with st.spinner("🔄 Analyzing..."):
            # Webcam analysis
            st.info("📹 Analyzing live camera feed...")
            video_result = analyze_video_behavior(source=0, confidence=0.25)
            audio_result = {"distress": False, "score": 0.0, "error": None}
            
            # Audio capture attempt
            try:
                audio_features = capture_audio(duration=3)
                audio_result = detect_distress(audio_features)
            except Exception as exc:
                audio_result["error"] = str(exc)
                st.warning(f"🔊 Microphone access limited: {str(exc)[:50]}...")
            
            # Sensor readings
            sensor_result = read_sensors(use_simulation=use_simulated_sensors)
            
            # Fallback to sample if needed
            if len(video_result.get("detections", [])) == 0:
                st.info("📹 Falling back to sample video analysis...")
                video_result = analyze_video_behavior(source=str(DEFAULT_VIDEO_PATH), confidence=0.25)
                try:
                    audio_features = extract_audio_file_features(str(DEFAULT_VIDEO_PATH))
                    audio_result = detect_distress(audio_features)
                except Exception as exc:
                    audio_result["error"] = str(exc)
            
            # Prepare flags for analysis
            video_flags = {
                "sentient_present": len(video_result.get("detections", [])) > 0,
                "sentient_count": len(video_result.get("detections", [])),
                "motion_agitation": video_result.get("agitated", False),
                "motion_score": video_result.get("motion_score", 0.0),
                "missing": bool(video_result.get("error")),
            }
            
            audio_flags = {
                "audio_distress": bool(audio_result.get("distress", False)),
                "score": float(audio_result.get("score", 0.0)),
                "missing": bool(audio_result.get("error")),
            }
            
            sensor_flags = {
                "temp_high": sensor_result.get("temp") is not None and sensor_result.get("temp") > 30,
                "humidity_extreme": sensor_result.get("humidity") is not None and (
                    sensor_result.get("humidity") < 35 or sensor_result.get("humidity") > 75
                ),
                "heart_rate_extreme": sensor_result.get("heart_rate") is not None and (
                    sensor_result.get("heart_rate") < 55 or sensor_result.get("heart_rate") > 110
                ),
                "missing": bool(sensor_result.get("error")),
            }
            
            # Prepare weights from sidebar
            weights = {
                "video_score": video_weight,
                "audio_score": audio_weight,
                "sensor_score": sensor_weight
            }

            # Multimodal fusion
            combined = analyze_combined(
                video_result, audio_result, sensor_result, 
                ontology_strength=ontology_strength, weights=weights
            )
            
            # Calculate welfare score (0-100 scale)
            welfare_score = (1.0 - combined["probability"]) * 100
            
            # Record history
            record = {
                "timestamp": format_timestamp(),
                "welfare_score": welfare_score,
                "video_score": combined["modality_scores"]["video_score"],
                "audio_score": combined["modality_scores"]["audio_score"],
                "sensor_score": combined["modality_scores"]["sensor_score"],
                "probability": combined["probability"],
            }
            save_analysis_record(record)
            
            # Display alert
            alert_text, alert_class = get_alert_level(combined["probability"])
            st.markdown(f'<div class="{alert_class}">{alert_text}</div>', unsafe_allow_html=True)
            
            # Store in session for display
            st.session_state.last_analysis = {
                "welfare_score": welfare_score,
                "probability": combined["probability"],
                "video_flags": video_flags,
                "audio_flags": audio_flags,
                "sensor_flags": sensor_flags,
                "sensor_result": sensor_result,
                "combined": combined,
                "video_result": video_result,
                "audio_result": audio_result,
            }
            
            st.success("✅ Analysis complete!")

with col_btn2:
    if st.button("📊 Refresh", key="refresh"):
        st.rerun()

with col_btn3:
    if st.button("📥 Export", key="export"):
        if len(history_df) > 0:
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"welfare_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

st.divider()

# ============================================================================
# TABBED INTERFACE
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Summary", "📈 Trends", "🔍 Details", "⚙️ Diagnostics", "📚 History"])

# ============================================================================
# TAB 1: SUMMARY
# ============================================================================

with tab1:
    if "last_analysis" in st.session_state:
        data = st.session_state.last_analysis
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Welfare Assessment")
            st.metric("Welfare Score", f"{data['welfare_score']:.1f}%")
            st.metric("Suffering Probability", f"{data['probability']:.1%}")
            
            # Alert classification
            if data['probability'] > critical_threshold:
                st.error(f"🔴 CRITICAL - Immediate intervention required!")
            elif data['probability'] > warning_threshold:
                st.warning(f"🟠 WARNING - Close monitoring recommended")
            else:
                st.success(f"🟢 NORMAL - Welfare indicators are stable")
        
        with col2:
            st.subheader("📊 Modality Scores")
            modality_data = pd.DataFrame({
                'Modality': ['Video', 'Audio', 'Sensor'],
                'Score': [
                    data['combined']['modality_scores']['video_score'],
                    data['combined']['modality_scores']['audio_score'],
                    data['combined']['modality_scores']['sensor_score'],
                ]
            })
            fig = px.bar(modality_data, x='Modality', y='Score', 
                         color='Score', color_continuous_scale='RdYlGn_r',
                         range_y=[0, 1])
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, width="stretch")
        
        st.subheader("💭 Analysis Explanation")
        for reason in data['combined']['explanations']:
            st.info(f"→ {reason}")
        
        if data['video_flags']['sentient_present']:
            st.subheader("🎬 Detection Summary")
            st.write(f"✓ **Sentient beings detected**: {data['video_flags']['sentient_count']}")
            st.write(f"✓ **Motion score**: {data['video_flags']['motion_score']:.2f}")
            st.write(f"✓ **Motion agitation**: {'Yes' if data['video_flags']['motion_agitation'] else 'No'}")
    else:
        st.info("👈 Run a welfare check to see the summary")

# ============================================================================
# TAB 2: TRENDS & ANALYTICS
# ============================================================================

with tab2:
    if len(history_df) > 0:
        st.subheader("📈 Historical Trends")
        
        # Convert timestamp to datetime
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        # Probability trend
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=history_df['timestamp'],
            y=history_df['probability'],
            mode='lines+markers',
            name='Suffering Probability',
            line=dict(color='#ff6b6b', width=3),
            fill='tozeroy'
        ))
        fig_trend.update_layout(
            title="Suffering Probability Over Time",
            xaxis_title="Timestamp",
            yaxis_title="Probability",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_trend, width="stretch")
        
        # Modality scores over time
        fig_modality = go.Figure()
        fig_modality.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['video_score'], 
                                         mode='lines', name='Video', line=dict(width=2)))
        fig_modality.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['audio_score'], 
                                         mode='lines', name='Audio', line=dict(width=2)))
        fig_modality.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['sensor_score'], 
                                         mode='lines', name='Sensor', line=dict(width=2)))
        fig_modality.update_layout(
            title="Modality Scores Over Time",
            xaxis_title="Timestamp",
            yaxis_title="Score",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_modality, width="stretch")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Probability", f"{history_df['probability'].mean():.2%}")
        with col2:
            st.metric("Max Probability", f"{history_df['probability'].max():.2%}")
        with col3:
            st.metric("Min Probability", f"{history_df['probability'].min():.2%}")
        with col4:
            st.metric("Std Deviation", f"{history_df['probability'].std():.2%}")
    else:
        st.info("📈 No historical data yet. Run analyses to build trends.")

# ============================================================================
# TAB 3: DETAILED ANALYSIS
# ============================================================================

with tab3:
    if "last_analysis" in st.session_state:
        data = st.session_state.last_analysis
        
        st.subheader("🎬 Video Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Sentient Present**: {'✓ Yes' if data['video_flags']['sentient_present'] else '✗ No'}")
            st.write(f"**Count**: {data['video_flags']['sentient_count']}")
            st.write(f"**Motion Score**: {data['video_flags']['motion_score']:.3f}")
        with col2:
            st.write(f"**Motion Agitation**: {'✓ Yes' if data['video_flags']['motion_agitation'] else '✗ No'}")
            st.write(f"**Video Missing**: {'✓ Yes' if data['video_flags']['missing'] else '✗ No'}")
            if data['video_result'].get('error'):
                st.error(f"Error: {data['video_result']['error']}")
        
        st.divider()
        
        st.subheader("🔊 Audio Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Distress Detected**: {'✓ Yes' if data['audio_flags']['audio_distress'] else '✗ No'}")
            st.write(f"**Audio Score**: {data['audio_flags']['score']:.3f}")
        with col2:
            st.write(f"**Audio Missing**: {'✓ Yes' if data['audio_flags']['missing'] else '✗ No'}")
            if data['audio_result'].get('error'):
                st.error(f"Error: {data['audio_result']['error']}")
        
        st.divider()
        
        st.subheader("📊 Sensor Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            if data['sensor_result'].get('temp'):
                temp = data['sensor_result']['temp']
                st.write(f"**Temperature**: {temp:.1f}°C")
                st.write(f"High (>30°C): {data['sensor_flags']['temp_high']}")
            else:
                st.write("**Temperature**: N/A")
        with col2:
            if data['sensor_result'].get('humidity'):
                hum = data['sensor_result']['humidity']
                st.write(f"**Humidity**: {hum:.1f}%")
                st.write(f"Extreme: {data['sensor_flags']['humidity_extreme']}")
            else:
                st.write("**Humidity**: N/A")
        with col3:
            if data['sensor_result'].get('heart_rate'):
                hr = data['sensor_result']['heart_rate']
                st.write(f"**Heart Rate**: {hr:.0f} bpm")
                st.write(f"Extreme: {data['sensor_flags']['heart_rate_extreme']}")
            else:
                st.write("**Heart Rate**: N/A")
        
        if show_raw_scores:
            st.divider()
            st.subheader("🔬 Raw Scores (Debug)")
            st.json({
                "raw_probability": data['combined']['raw_probability'],
                "ontology_score": data['combined']['ontology_score'],
                "video_flags": data['video_flags'],
                "audio_flags": data['audio_flags'],
                "sensor_flags": data['sensor_flags'],
            })
    else:
        st.info("👈 Run a welfare check to see detailed analysis")

# ============================================================================
# TAB 4: DIAGNOSTICS
# ============================================================================

with tab4:
    st.subheader("🔧 System Diagnostics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Component Status**")
        try:
            import cv2
            st.write("✓ OpenCV (cv2)")
        except:
            st.write("✗ OpenCV (cv2)")
        
        try:
            import librosa
            st.write("✓ Librosa")
        except:
            st.write("✗ Librosa")
        
        try:
            import sounddevice
            st.write("✓ Sounddevice")
        except:
            st.write("✗ Sounddevice")
    
    with col2:
        st.write("**System Tools**")
        ffmpeg_ok = ffmpeg_available()
        st.write(f"{'✓' if ffmpeg_ok else '✗'} FFmpeg")
        
        try:
            cap = cv2.VideoCapture(0)
            cam_ok = cap.isOpened()
            cap.release()
            st.write(f"{'✓' if cam_ok else '✗'} Camera")
        except:
            st.write("✗ Camera")
    
    st.divider()
    st.subheader("📋 Latest Analysis Data")
    if "last_analysis" in st.session_state:
        data = st.session_state.last_analysis
        st.json({
            "timestamp": format_timestamp(),
            "video_score": data['combined']['modality_scores']['video_score'],
            "audio_score": data['combined']['modality_scores']['audio_score'],
            "sensor_score": data['combined']['modality_scores']['sensor_score'],
            "probability": data['combined']['probability'],
            "raw_probability": data['combined']['raw_probability'],
        })
    else:
        st.info("No data yet")

# ============================================================================
# TAB 5: HISTORY
# ============================================================================

with tab5:
    st.subheader("📚 Analysis History")
    if len(history_df) > 0:
        st.dataframe(history_df, width="stretch", height=400)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyses", len(history_df))
        with col2:
            st.metric("Average Probability", f"{history_df['probability'].mean():.2%}")
        with col3:
            st.metric("High Risk Events (>0.7)", len(history_df[history_df['probability'] > 0.7]))
    else:
        st.info("📊 No historical data. Run welfare checks to build history.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.8em;'>
    <p>🔍 Sentient Being Welfare Monitoring Dashboard v2026</p>
    <p>Multimodal Analysis | Real-Time Monitoring | AI-Powered Insights</p>
</div>
""", unsafe_allow_html=True)
