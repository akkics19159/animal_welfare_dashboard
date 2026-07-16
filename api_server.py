import os
import json
import time
import psutil
import pandas as pd
import threading
import traceback
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import shutil
import logging
from logging.handlers import RotatingFileHandler
import uuid
import csv
import base64
import plotly




# Import legacy components
from multimodal_engine import analyze_combined
from sensors import read_sensors
from audio_module import ffmpeg_available
from vision_intelligence.input_source_manager import InputSourceManager


_INPUT_SOURCE_MANAGER = InputSourceManager()

app = FastAPI(
    title="Sentient Being Welfare Monitoring API Platform",
    description="Production-grade MLOps monitoring and inference endpoints.",
    version="2026.1"
)

BASE_DIR = Path(__file__).resolve().parent
HISTORY_CSV = BASE_DIR / "welfare_analysis_history.csv"
ALERTS_JSON = BASE_DIR / "alerts.json"
MODELS_JSON = BASE_DIR / "training/registry/models.json"
DATASETS_JSON = BASE_DIR / "dataset/registry/datasets.json"
EVAL_HISTORY_JSON = BASE_DIR / "evaluation/experiments/history.json"
TRAIN_TRACKING_JSON = BASE_DIR / "training/tracking/experiments.json"
LOGS_DIR = BASE_DIR / "logs"
BACKEND_LOG_FILE = LOGS_DIR / "backend.log"

# Global training simulation state
TRAINING_STATUS = {
    "is_training": False,
    "pipeline_type": "",
    "current_epoch": 0,
    "total_epochs": 10,
    "progress": 0.0,
    "loss": 0.0,
    "val_loss": 0.0,
    "accuracy": 0.0,
    "val_accuracy": 0.0,
    "gpu_usage": 0.0,
    "eta_seconds": 0
}

# Audit & general log buffers (API-visible). Kept for backwards compatibility.
LOGS = {
    "audit": [],
    "inference": [],
    "alert": [],
    "training": [],
    "model": []
}

# Structured logging (JSON events) for backend logging.
LOGGER = logging.getLogger("welfare_api")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            fmt="%(message)s"
        )
    )
    _file_handler = RotatingFileHandler(
        BACKEND_LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    _file_handler.setFormatter(logging.Formatter(fmt="%(message)s"))
    LOGGER.addHandler(_handler)
    LOGGER.addHandler(_file_handler)

REQUEST_ID_HEADER = "X-Request-Id"
_JSON_LOCK = threading.RLock()
_HISTORY_LOCK = threading.RLock()
_PIPELINE_METRICS = {
    "inference_latency_ms": deque(maxlen=300),
    "frame_latency_ms": deque(maxlen=300),
}
_LAST_INPUT_SOURCE = {
    "source_type": "unknown",
    "source": None,
    "updated_at": None,
}

def _new_request_id() -> str:
    return uuid.uuid4().hex

def _emit_structured_log(category: str, event: str, message: str, *, level: str = "INFO", extra: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None):
    payload: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "category": category,
        "event": event,
        "message": message,
    }
    if request_id:
        payload["request_id"] = request_id
    if extra:
        payload["extra"] = extra
    LOGGER.info(json.dumps(payload, ensure_ascii=False))


def _safe_json_atomic_write(file_path: Path, data: Any):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    serialized = json.dumps(data, indent=2)
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(serialized)
    os.replace(tmp_path, file_path)


def _append_history_record(record: Dict[str, Any]):
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([record])
    with _HISTORY_LOCK:
        if HISTORY_CSV.exists():
            df_new.to_csv(HISTORY_CSV, mode="a", header=False, index=False, encoding="utf-8")
        else:
            df_new.to_csv(HISTORY_CSV, index=False, encoding="utf-8")


def _recent_avg(values: deque) -> float:
    if not values:
        return 0.0
    return float(sum(values) / max(1, len(values)))


def _safe_history_rows() -> int:
    if not HISTORY_CSV.exists():
        return 0
    try:
        return int(len(pd.read_csv(HISTORY_CSV)))
    except Exception:
        return 0


def _active_model_version() -> str:
    db = load_json_db(MODELS_JSON, {})
    if not isinstance(db, dict):
        return "welfare-fusion-v1"
    for model in db.values():
        if isinstance(model, dict) and model.get("active"):
            return str(model.get("model_version") or "welfare-fusion-v1")
    return "welfare-fusion-v1"


def _append_api_log(category: str, message: str, *, request_id: Optional[str] = None, event: str = "log"):
    timestamp = datetime.now().isoformat()
    entry: Dict[str, Any] = {"timestamp": timestamp, "message": message, "event": event}
    if request_id:
        entry["request_id"] = request_id
    LOGS[category].append(entry)
    # Maintain maximum of 500 logs per category in memory
    if len(LOGS[category]) > 500:
        LOGS[category].pop(0)


class InferenceInput(BaseModel):
    video_result: Dict[str, Any]
    audio_result: Dict[str, Any]
    sensor_result: Dict[str, Any]
    ontology_strength: Optional[float] = 0.6
    weights: Optional[Dict[str, float]] = None

class ActiveModelRequest(BaseModel):
    model_version: str

class AlertAcknowledgeRequest(BaseModel):
    alert_id: str

class TrainingRequest(BaseModel):
    pipeline_type: str
    epochs: int

def log_event(category: str, message: str):
    # Backwards compatible helper (keeps `/api/logs/{category}` working)
    _append_api_log(category, message, event="log")
    _emit_structured_log(category=category, event="log", message=message, level="INFO")


# Helper functions for database load/save
def load_json_db(file_path: Path, default: Any) -> Any:
    if not file_path.exists():
        return default
    try:
        with _JSON_LOCK:
            return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        _emit_structured_log(
            category="storage",
            event="read_error",
            message=f"Failed reading JSON store: {file_path.name}",
            level="ERROR",
            extra={"error": str(e)},
        )
        return default

def save_json_db(file_path: Path, data: Any):
    with _JSON_LOCK:
        _safe_json_atomic_write(file_path, data)

def check_alert_heuristics(
    result: Dict[str, Any],
    video_input: Dict[str, Any],
    audio_input: Dict[str, Any],
    sensor_input: Dict[str, Any]
):
    alerts = load_json_db(ALERTS_JSON, [])
    prob = result.get("probability", 0.0)
    
    # 1. High Welfare Risk
    if prob >= 0.7:
        alerts.append({
            "id": f"alert_risk_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "severity": "Critical",
            "source": "Reasoning Engine",
            "message": f"Critical Welfare Risk: Suffering probability is {prob:.2%}",
            "acknowledged": False
        })
        log_event("alert", f"Critical alert generated: Suffering probability {prob:.2%}")
    elif prob >= 0.4:
        alerts.append({
            "id": f"alert_risk_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "severity": "Warning",
            "source": "Reasoning Engine",
            "message": f"Warning Welfare Risk: Suffering probability is {prob:.2%}",
            "acknowledged": False
        })
        log_event("alert", f"Warning alert generated: Suffering probability {prob:.2%}")

    # 2. Camera/Mic/Sensor Failures
    if video_input.get("error"):
        alerts.append({
            "id": f"alert_cam_fail_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "severity": "Critical",
            "source": "System Health",
            "message": f"Camera Failure: {video_input.get('error')}",
            "acknowledged": False
        })
    if audio_input.get("error"):
        alerts.append({
            "id": f"alert_mic_fail_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "severity": "Warning",
            "source": "System Health",
            "message": f"Audio Capture Limit/Failure: {audio_input.get('error')}",
            "acknowledged": False
        })
    if sensor_input.get("error"):
        alerts.append({
            "id": f"alert_sensor_fail_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "severity": "Warning",
            "source": "System Health",
            "message": f"Sensor Failure: {sensor_input.get('error')}",
            "acknowledged": False
        })

    # 3. Crowding / High Occupancy
    detections = video_input.get("detections", [])
    if len(detections) > 3:
        alerts.append({
            "id": f"alert_crowd_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "severity": "Warning",
            "source": "Vision Module",
            "message": f"Crowding Alert: {len(detections)} sentient beings detected in view",
            "acknowledged": False
        })

    # 4. Repeated distress
    if audio_input.get("distress"):
        # Check if the last history record also had distress
        had_recent = False
        if HISTORY_CSV.exists():
            try:
                df = pd.read_csv(HISTORY_CSV)
                if len(df) > 0 and df.iloc[-1].get("audio_score", 0.0) > 0.4:
                    had_recent = True
            except:
                pass
        if had_recent:
            alerts.append({
                "id": f"alert_distress_{int(time.time())}",
                "timestamp": datetime.now().isoformat(),
                "severity": "Critical",
                "source": "Audio Intelligence",
                "message": "Repeated Vocal Distress Patterns Detected",
                "acknowledged": False
            })

    save_json_db(ALERTS_JSON, alerts)

def execute_training_pipeline(pipeline_type: str, epochs: int):
    global TRAINING_STATUS
    TRAINING_STATUS["is_training"] = True
    TRAINING_STATUS["pipeline_type"] = pipeline_type
    TRAINING_STATUS["total_epochs"] = epochs
    TRAINING_STATUS["current_epoch"] = 0
    TRAINING_STATUS["progress"] = 0.0
    
    import random
    loss = 0.72
    val_loss = 0.78
    acc = 0.55
    val_acc = 0.52
    
    for epoch in range(1, epochs + 1):
        if not TRAINING_STATUS["is_training"]:  # Interrupted
            break
        time.sleep(1.0)  # Simulate 1 epoch training duration
        
        loss -= random.uniform(0.04, 0.08)
        val_loss -= random.uniform(0.03, 0.07)
        acc += random.uniform(0.03, 0.06)
        val_acc += random.uniform(0.02, 0.05)
        
        loss = max(0.12, loss)
        val_loss = max(0.15, val_loss)
        acc = min(0.96, acc)
        val_acc = min(0.93, val_acc)
        
        TRAINING_STATUS["current_epoch"] = epoch
        TRAINING_STATUS["progress"] = epoch / epochs
        TRAINING_STATUS["loss"] = round(loss, 4)
        TRAINING_STATUS["val_loss"] = round(val_loss, 4)
        TRAINING_STATUS["accuracy"] = round(acc, 4)
        TRAINING_STATUS["val_accuracy"] = round(val_acc, 4)
        TRAINING_STATUS["gpu_usage"] = round(random.uniform(75, 96), 1)
        TRAINING_STATUS["eta_seconds"] = epochs - epoch
        
        log_event("training", f"Epoch {epoch}/{epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f} - acc: {acc:.4f}")
        
    TRAINING_STATUS["is_training"] = False
    
    # Save the run metrics to experiment log
    history = load_json_db(TRAIN_TRACKING_JSON, [])
    history.append({
        "experiment_id": f"run_{pipeline_type.lower()}_{int(time.time())}",
        "metrics": {
            "final_loss": TRAINING_STATUS["loss"],
            "final_val_loss": TRAINING_STATUS["val_loss"],
            "final_accuracy": TRAINING_STATUS["accuracy"],
            "final_val_accuracy": TRAINING_STATUS["val_accuracy"],
            "epochs_completed": epochs,
            "total_duration_sec": float(epochs)
        },
        "config": {
            "pipeline_type": pipeline_type,
            "epochs": epochs,
            "learning_rate": 0.0005,
            "optimizer": "AdamW"
        }
    })
    save_json_db(TRAIN_TRACKING_JSON, history)
    log_event("training", f"Training completed successfully for pipeline {pipeline_type}")


# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.post("/api/inference")
def run_inference(payload: InferenceInput):
    request_id = _new_request_id()
    t_start = time.time()
    try:
        if payload.weights is not None:
            wsum = float(sum(float(v) for v in payload.weights.values()))
            if wsum <= 0:
                raise HTTPException(status_code=400, detail="weights must have positive sum")
        if payload.ontology_strength is not None and not (0.0 <= float(payload.ontology_strength) <= 1.0):
            raise HTTPException(status_code=400, detail="ontology_strength must be in [0, 1]")

        result = analyze_combined(
            payload.video_result,
            payload.audio_result,
            payload.sensor_result,
            ontology_strength=payload.ontology_strength,
            weights=payload.weights
        )
        latency = round((time.time() - t_start) * 1000, 2)
        _PIPELINE_METRICS["inference_latency_ms"].append(float(latency))
        _PIPELINE_METRICS["frame_latency_ms"].append(float(latency))
        
        # Append latency metric
        result["latency_ms"] = latency
        result["timestamp"] = datetime.now().isoformat()
        src_type = ((result.get("video_result") or {}).get("source_type") or "unknown")
        src = ((result.get("video_result") or {}).get("source"))
        _LAST_INPUT_SOURCE["source_type"] = src_type
        _LAST_INPUT_SOURCE["source"] = src
        _LAST_INPUT_SOURCE["updated_at"] = result["timestamp"]
        
        # Save record to legacy CSV history if active
        record = {
            "timestamp": result["timestamp"],
            "welfare_score": (1.0 - result["probability"]) * 100.0,
            "video_score": result["modality_scores"]["video_score"],
            "audio_score": result["modality_scores"]["audio_score"],
            "sensor_score": result["modality_scores"]["sensor_score"],
            "probability": result["probability"],
            "behaviour": result.get("behaviour", "unknown_behaviour"),
            "behaviour_probability": result.get("behaviour_probability", 0.0),
            "behaviour_confidence": result.get("behaviour_confidence", 0.0),
            "behaviour_duration": result.get("behaviour_duration", 0.0),
            "behaviour_transition": result.get("behaviour_transition", "steady"),
            "behaviour_stability": result.get("behaviour_stability", 0.0),
            "occupancy": int((result.get("video_result", {}) or {}).get("occupancy", len((result.get("video_result", {}) or {}).get("tracks", []) or [])) or 0),
            "species": (
                ((result.get("video_result", {}) or {}).get("tracks", [{}]) or [{}])[0].get("final_species")
                or ((result.get("video_result", {}) or {}).get("detections", [{}]) or [{}])[0].get("final_species")
                or ((result.get("video_result", {}) or {}).get("detections", [{}]) or [{}])[0].get("species")
                or "unknown"
            ),
            "risk_level": result.get("risk_level", "low"),
        }
        
        # Write to local CSV
        _append_history_record(record)

        # Check heuristics for automated alert triggers
        check_alert_heuristics(result, payload.video_result, payload.audio_result, payload.sensor_result)
        
        # Log event
        _emit_structured_log(
            category="inference",
            event="completed",
            message=f"Multimodal inference completed in {latency}ms",
            level="INFO",
            request_id=request_id,
            extra={
                "probability": float(result.get("probability", 0.0) or 0.0),
                "input_source": src,
                "source_type": src_type,
                "behaviour": result.get("behaviour"),
                "risk_level": result.get("risk_level"),
                "inference_time_ms": latency,
            },
        )
        log_event("inference", f"Multimodal inference completed in {latency}ms. Prob: {result['probability']:.3f}")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        _emit_structured_log(
            category="inference",
            event="failure",
            message="Inference pipeline crash",
            level="ERROR",
            request_id=request_id,
            extra={"error": str(exc), "trace": traceback.format_exc()},
        )
        log_event("inference", f"Inference pipeline crash: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/history")
def get_history():
    if not HISTORY_CSV.exists():
        return []
    try:
        df = _read_history_df()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read history CSV: {e}")

@app.get("/api/analytics")
def get_analytics():
    if not HISTORY_CSV.exists():
        return {"trends": [], "aggregates": {}}
    try:
        df = _read_history_df()
        if df.empty:
            return {"trends": [], "aggregates": {}}
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Aggregate statistics
        mean_prob = float(df['probability'].mean())
        max_prob = float(df['probability'].max())
        total_runs = int(len(df))
        critical_runs = int(len(df[df['probability'] >= 0.7]))

        behaviour_distribution: Dict[str, int] = {}
        transition_distribution: Dict[str, int] = {}
        avg_behaviour_stability = 0.0
        if 'behaviour' in df.columns:
            behaviour_distribution = {
                str(k): int(v)
                for k, v in df['behaviour'].fillna('unknown_behaviour').value_counts().to_dict().items()
            }
        if 'behaviour_transition' in df.columns:
            transition_distribution = {
                str(k): int(v)
                for k, v in df['behaviour_transition'].fillna('steady').value_counts().to_dict().items()
            }
        if 'behaviour_stability' in df.columns:
            avg_behaviour_stability = float(pd.to_numeric(df['behaviour_stability'], errors='coerce').fillna(0.0).mean())
        
        # Formulate a simplified hourly/daily trend list
        df['date_hour'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:00')
        trend = df.groupby('date_hour').agg({
            'probability': 'mean',
            'video_score': 'mean',
            'audio_score': 'mean',
            'sensor_score': 'mean'
        }).reset_index().to_dict(orient="records")

        trend_rows = trend
        behaviour_timeline = []
        if 'behaviour' in df.columns:
            bt = (
                df.groupby(['date_hour', 'behaviour'])
                .size()
                .reset_index(name='count')
                .to_dict(orient='records')
            )
            behaviour_timeline = bt

        return {
            "trends": trend,
            "behaviour_timeline": behaviour_timeline,
            "aggregates": {
                "mean_probability": mean_prob,
                "max_probability": max_prob,
                "total_runs": total_runs,
                "critical_runs": critical_runs,
                "avg_behaviour_stability": avg_behaviour_stability,
            },
            "behaviour_distribution": behaviour_distribution,
            "behaviour_transition_distribution": transition_distribution,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics failed: {e}")

@app.get("/api/training")
def get_training():
    history = load_json_db(TRAIN_TRACKING_JSON, [])
    return {
        "status": TRAINING_STATUS,
        "history": history
    }

@app.post("/api/training/run")
def start_training(req: TrainingRequest, bg_tasks: BackgroundTasks):
    if TRAINING_STATUS["is_training"]:
        raise HTTPException(status_code=400, detail="A training run is already in progress.")
    bg_tasks.add_task(execute_training_pipeline, req.pipeline_type, req.epochs)
    log_event("audit", f"Admin initiated training run for pipeline {req.pipeline_type} with {req.epochs} epochs")
    return {"message": "Training pipeline started successfully in background."}

@app.post("/api/training/stop")
def stop_training():
    if not TRAINING_STATUS["is_training"]:
        raise HTTPException(status_code=400, detail="No training run in progress to stop.")
    TRAINING_STATUS["is_training"] = False
    log_event("audit", "Admin force stopped the active training pipeline.")
    return {"message": "Training pipeline stop command issued."}

@app.get("/api/evaluation")
def get_evaluation():
    history = load_json_db(EVAL_HISTORY_JSON, [])
    return history

@app.get("/api/dataset")
def get_dataset():
    db = load_json_db(DATASETS_JSON, {})
    return list(db.values())

@app.get("/api/models")
def get_models():
    db = load_json_db(MODELS_JSON, {})
    return list(db.values())

@app.post("/api/models/active")
def switch_active_model(req: ActiveModelRequest):
    db = load_json_db(MODELS_JSON, {})
    if req.model_version not in db:
        raise HTTPException(status_code=404, detail=f"Model version '{req.model_version}' not found in registry.")
    
    # Update active flag
    for mv in db:
        db[mv]["active"] = (mv == req.model_version)
    
    save_json_db(MODELS_JSON, db)
    log_event("model", f"Model registry switched active model to: {req.model_version}")
    log_event("audit", f"Admin switched active model to: {req.model_version}")
    return {"message": f"Successfully activated model '{req.model_version}'."}

@app.get("/api/alerts")
def get_alerts():
    alerts = load_json_db(ALERTS_JSON, [])
    return alerts

@app.post("/api/alerts/acknowledge")
def acknowledge_alert(req: AlertAcknowledgeRequest):

    alerts = load_json_db(ALERTS_JSON, [])
    found = False
    for a in alerts:
        if a["id"] == req.alert_id:
            a["acknowledged"] = True
            found = True
            log_event("alert", f"Alert acknowledged: {req.alert_id}")
            break
    if not found:
        raise HTTPException(status_code=404, detail="Alert ID not found")
    save_json_db(ALERTS_JSON, alerts)
    return {"message": f"Acknowledged alert {req.alert_id}"}

@app.get("/api/health")
def get_health():
    # Basic system resource checks
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    disk_root = BASE_DIR.anchor or str(BASE_DIR)
    disk = psutil.disk_usage(disk_root).percent

    # Check physical hardware statuses
    mic_ok = bool(ffmpeg_available() or shutil.which("ffmpeg"))

    camera_ok = False
    cap = None
    source_meta: Dict[str, Any] = {}
    try:
        cap, meta, err = _INPUT_SOURCE_MANAGER.probe_source()
        camera_ok = bool(cap is not None and meta is not None and meta.source_type == "webcam")
        source_meta = {
            "source_type": getattr(meta, "source_type", None),
            "source": getattr(meta, "source", None),
            "error": err,
        }
        if meta is not None:
            _LAST_INPUT_SOURCE["source_type"] = getattr(meta, "source_type", "unknown")
            _LAST_INPUT_SOURCE["source"] = getattr(meta, "source", None)
            _LAST_INPUT_SOURCE["updated_at"] = datetime.now().isoformat()
    except Exception as exc:
        camera_ok = False
        source_meta = {"source_type": None, "source": None, "error": str(exc)}
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    vm = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    try:
        process_memory_mb = float(process.memory_info().rss / (1024 * 1024))
    except Exception:
        process_memory_mb = 0.0

    avg_inference_latency = _recent_avg(_PIPELINE_METRICS["inference_latency_ms"])

    payload = {
        "cpu": cpu,
        "ram": ram,
        "disk": disk,
        "gpu": {
            "name": "NVIDIA GeForce RTX 4070 (Simulated)",
            "usage": round(psutil.cpu_percent() * 0.4, 1),
            "temp": round(45 + cpu * 0.2, 1)
        },
        "devices": {
            "camera": "Connected" if camera_ok else "Not Detected",
            "microphone": "Available" if mic_ok else "Unavailable",
            "sensor_array": "Online (Simulation Mode)"
        },
        "inference_throughput": f"{(1000.0 / avg_inference_latency):.2f} frame/sec" if avg_inference_latency > 0 else "n/a",
        "pipeline": {
            "status": "healthy" if camera_ok or _LAST_INPUT_SOURCE.get("source") else "degraded",
            "current_input_source": _LAST_INPUT_SOURCE.get("source"),
            "current_input_type": _LAST_INPUT_SOURCE.get("source_type"),
            "camera_probe": source_meta,
            "average_fps": float((1000.0 / avg_inference_latency) if avg_inference_latency > 0 else 0.0),
            "current_fps": float((1000.0 / _PIPELINE_METRICS["inference_latency_ms"][-1]) if _PIPELINE_METRICS["inference_latency_ms"] and _PIPELINE_METRICS["inference_latency_ms"][-1] > 0 else 0.0),
            "frame_latency_ms": float(_PIPELINE_METRICS["frame_latency_ms"][-1]) if _PIPELINE_METRICS["frame_latency_ms"] else 0.0,
            "inference_latency_ms": float(_PIPELINE_METRICS["inference_latency_ms"][-1]) if _PIPELINE_METRICS["inference_latency_ms"] else 0.0,
            "model_loading_time_ms": 0.0,
            "queue_sizes": {"inference": 0, "dashboard": 0},
            "thread_count": int(process.num_threads()),
        },
        "memory": {
            "system_percent": ram,
            "system_available_mb": float(vm.available / (1024 * 1024)),
            "process_rss_mb": process_memory_mb,
        },
        "storage": {
            "disk_percent": disk,
            "history_csv_exists": bool(HISTORY_CSV.exists()),
            "history_rows": _safe_history_rows(),
        },
        "model_status": {
            "active_model": _active_model_version(),
            "version": app.version,
        },
        "timestamp": datetime.now().isoformat()
    }

    # structured log
    _emit_structured_log(category="health", event="read", message="Health check", level="INFO")
    return payload


@app.get("/api/logs/{category}")
def get_logs(category: str):

    if category not in LOGS:
        raise HTTPException(status_code=404, detail="Log category not found")
    return LOGS[category]

@app.post("/api/logs/clear")
def clear_logs():
    for cat in LOGS:
        LOGS[cat].clear()
    log_event("audit", "Admin cleared system logs.")
    return {"message": "Logs cleared successfully."}


# ==========================================
# EXPORT REST API ENDPOINTS (CSV/JSON/Markdown/PDF/Charts)
# ==========================================


def _coerce_format(fmt: Optional[str], default: str) -> str:
    if not fmt:
        return default
    return str(fmt).lower().strip()


def _as_markdown_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join(["---"] * len(keys)) + " |"
    body_lines = []
    for r in rows:
        body_lines.append("| " + " | ".join([str(r.get(k, "")) for k in keys]) + " |")
    return "\n".join([header, sep] + body_lines)


def _read_history_df() -> pd.DataFrame:
    if not HISTORY_CSV.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_CSV)
    except Exception:
        # Handle legacy rows with differing column counts without crashing exports.
        return pd.read_csv(HISTORY_CSV, engine="python", on_bad_lines="skip")


@app.get("/api/export/history")
def export_history(format: Optional[str] = "csv"):
    fmt = _coerce_format(format, "csv")
    _emit_structured_log(category="export", event="history", message=f"Export history format={fmt}")

    df = _read_history_df()
    if df.empty:
        return {"rows": [], "format": fmt, "message": "No history data available."}

    rows = df.to_dict(orient="records")

    if fmt == "json":
        return {"format": "json", "rows": rows}

    if fmt == "markdown":
        return {"format": "markdown", "markdown": _as_markdown_table(rows)}

    if fmt == "csv":
        # Keep response JSON-safe (base64 CSV)
        csv_bytes = df.to_csv(index=False, encoding="utf-8").encode("utf-8")
        csv_b64 = base64.b64encode(csv_bytes).decode("ascii")
        return {"format": "csv", "csv_base64": csv_b64}

    if fmt == "pdf":
        # Graceful fallback: return markdown + note.
        md = _as_markdown_table(rows)
        return {"format": "pdf", "markdown_fallback": md, "note": "PDF generation not available in this environment."}

    raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")


@app.get("/api/export/alerts")
def export_alerts(format: Optional[str] = "json"):
    fmt = _coerce_format(format, "json")
    _emit_structured_log(category="export", event="alerts", message=f"Export alerts format={fmt}")

    alerts = load_json_db(ALERTS_JSON, [])

    if fmt == "json":
        return {"format": "json", "rows": alerts}
    if fmt == "markdown":
        return {"format": "markdown", "markdown": _as_markdown_table(alerts)}
    if fmt == "csv":
        if not alerts:
            return {"format": "csv", "csv_base64": ""}
        df = pd.DataFrame(alerts)
        csv_b64 = base64.b64encode(df.to_csv(index=False, encoding="utf-8").encode("utf-8")).decode("ascii")
        return {"format": "csv", "csv_base64": csv_b64}
    if fmt == "pdf":
        md = _as_markdown_table(alerts)
        return {"format": "pdf", "markdown_fallback": md, "note": "PDF generation not available in this environment."}

    raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")


@app.get("/api/export/evaluation")
def export_evaluation(format: Optional[str] = "json"):
    fmt = _coerce_format(format, "json")
    _emit_structured_log(category="export", event="evaluation", message=f"Export evaluation format={fmt}")

    history = load_json_db(EVAL_HISTORY_JSON, [])

    if fmt == "json":
        return {"format": "json", "experiments": history}
    if fmt == "markdown":
        # flatten summary rows
        rows = []
        for e in history:
            if not isinstance(e, dict):
                continue
            metrics = e.get("metrics") or {}
            cfg = e.get("config") or {}
            rows.append({
                "name": e.get("name"),
                "model": cfg.get("model"),
                "dataset": cfg.get("dataset"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "brier_score": metrics.get("brier_score"),
                "ece": metrics.get("ece"),
                "latency": metrics.get("latency"),
                "memory": metrics.get("memory"),
            })
        return {"format": "markdown", "markdown": _as_markdown_table(rows)}
    if fmt == "csv":
        rows = []
        for e in history:
            if not isinstance(e, dict):
                continue
            metrics = e.get("metrics") or {}
            cfg = e.get("config") or {}
            rows.append({
                "name": e.get("name"),
                "model": cfg.get("model"),
                "dataset": cfg.get("dataset"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "brier_score": metrics.get("brier_score"),
                "ece": metrics.get("ece"),
                "latency": metrics.get("latency"),
                "memory": metrics.get("memory"),
            })
        df = pd.DataFrame(rows)
        csv_b64 = base64.b64encode(df.to_csv(index=False, encoding="utf-8").encode("utf-8")).decode("ascii")
        return {"format": "csv", "csv_base64": csv_b64}
    if fmt == "pdf":
        md = export_evaluation("markdown").get("markdown", "")
        return {"format": "pdf", "markdown_fallback": md, "note": "PDF generation not available in this environment."}

    raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")


@app.get("/api/export/training")
def export_training(format: Optional[str] = "json"):
    fmt = _coerce_format(format, "json")
    _emit_structured_log(category="export", event="training", message=f"Export training format={fmt}")

    history = load_json_db(TRAIN_TRACKING_JSON, [])

    if fmt == "json":
        return {"format": "json", "runs": history, "status": TRAINING_STATUS}

    # Normalize to rows
    rows: List[Dict[str, Any]] = []
    for r in history:
        if not isinstance(r, dict):
            continue
        metrics = r.get("metrics") or {}
        cfg = r.get("config") or {}
        rows.append({
            "experiment_id": r.get("experiment_id"),
            "pipeline_type": cfg.get("pipeline_type"),
            "epochs": cfg.get("epochs"),
            "final_loss": metrics.get("final_loss"),
            "final_val_loss": metrics.get("final_val_loss"),
            "final_accuracy": metrics.get("final_accuracy"),
            "final_val_accuracy": metrics.get("final_val_accuracy"),
        })

    if fmt == "markdown":
        return {"format": "markdown", "markdown": _as_markdown_table(rows)}
    if fmt == "csv":
        df = pd.DataFrame(rows)
        csv_b64 = base64.b64encode(df.to_csv(index=False, encoding="utf-8").encode("utf-8")).decode("ascii")
        return {"format": "csv", "csv_base64": csv_b64}
    if fmt == "pdf":
        md = _as_markdown_table(rows)
        return {"format": "pdf", "markdown_fallback": md, "note": "PDF generation not available in this environment."}

    raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")


@app.get("/api/export/dataset")
def export_dataset(format: Optional[str] = "json"):
    fmt = _coerce_format(format, "json")
    _emit_structured_log(category="export", event="dataset", message=f"Export dataset format={fmt}")

    datasets = load_json_db(DATASETS_JSON, {})
    rows = list(datasets.values()) if isinstance(datasets, dict) else []

    if fmt == "json":
        return {"format": "json", "datasets": rows}
    if fmt == "markdown":
        # dataset metadata is nested; summarize
        flat = []
        for d in rows:
            meta = d.get("metadata") or {}
            flat.append({
                "dataset_id": d.get("dataset_id"),
                "version": d.get("version"),
                "samples": meta.get("samples"),
                "validation_status": meta.get("validation_status"),
                "annotation_progress": meta.get("annotation_progress"),
                "quality_score": meta.get("quality_score"),
                "created_at": meta.get("created_at"),
            })
        return {"format": "markdown", "markdown": _as_markdown_table(flat)}
    if fmt == "csv":
        flat = []
        for d in rows:
            meta = d.get("metadata") or {}
            flat.append({
                "dataset_id": d.get("dataset_id"),
                "version": d.get("version"),
                "samples": meta.get("samples"),
                "validation_status": meta.get("validation_status"),
                "annotation_progress": meta.get("annotation_progress"),
                "quality_score": meta.get("quality_score"),
                "created_at": meta.get("created_at"),
            })
        df = pd.DataFrame(flat)
        csv_b64 = base64.b64encode(df.to_csv(index=False, encoding="utf-8").encode("utf-8")).decode("ascii")
        return {"format": "csv", "csv_base64": csv_b64}
    if fmt == "pdf":
        md = export_dataset("markdown").get("markdown", "")
        return {"format": "pdf", "markdown_fallback": md, "note": "PDF generation not available in this environment."}

    raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")


@app.get("/api/export/models")
def export_models(format: Optional[str] = "json"):
    fmt = _coerce_format(format, "json")
    _emit_structured_log(category="export", event="models", message=f"Export models format={fmt}")

    models = load_json_db(MODELS_JSON, {})
    rows = list(models.values()) if isinstance(models, dict) else []

    if fmt == "json":
        return {"format": "json", "models": rows}

    if fmt in ("markdown", "csv", "pdf"):
        # summarize
        flat = []
        for m in rows:
            flat.append({
                "model_version": m.get("model_version"),
                "name": m.get("name"),
                "type": m.get("type"),
                "dataset_version": m.get("dataset_version"),
                "accuracy": m.get("accuracy"),
                "f1_score": m.get("f1_score"),
                "latency_ms": m.get("latency_ms"),
                "memory_mb": m.get("memory_mb"),
                "active": m.get("active"),
            })
        if fmt == "markdown":
            return {"format": "markdown", "markdown": _as_markdown_table(flat)}
        if fmt == "csv":
            df = pd.DataFrame(flat)
            csv_b64 = base64.b64encode(df.to_csv(index=False, encoding="utf-8").encode("utf-8")).decode("ascii")
            return {"format": "csv", "csv_base64": csv_b64}
        if fmt == "pdf":
            md = _as_markdown_table(flat)
            return {"format": "pdf", "markdown_fallback": md, "note": "PDF generation not available in this environment."}

    raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")


@app.get("/api/export/analytics")
def export_analytics(format: Optional[str] = "json"):
    fmt = _coerce_format(format, "json")
    _emit_structured_log(category="export", event="analytics", message=f"Export analytics format={fmt}")

    analytics = get_analytics()
    if fmt == "json":
        return {"format": "json", "analytics": analytics}
    if fmt == "markdown":
        return {"format": "markdown", "markdown": json.dumps(analytics, indent=2)}
    if fmt == "csv":
        # flatten aggregates only
        agg = analytics.get("aggregates") or {}
        df = pd.DataFrame([{k: v for k, v in agg.items()}])
        csv_b64 = base64.b64encode(df.to_csv(index=False, encoding="utf-8").encode("utf-8")).decode("ascii")
        return {"format": "csv", "csv_base64": csv_b64}
    if fmt == "pdf":
        return {"format": "pdf", "markdown_fallback": json.dumps(analytics, indent=2), "note": "PDF generation not available."}

    raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")


@app.get("/api/export/health")
def export_health(format: Optional[str] = "json"):
    fmt = _coerce_format(format, "json")
    health = get_health()
    if fmt == "json":
        return {"format": "json", "health": health}
    if fmt == "markdown":
        return {"format": "markdown", "markdown": json.dumps(health, indent=2)}
    if fmt == "pdf":
        return {"format": "pdf", "markdown_fallback": json.dumps(health, indent=2), "note": "PDF generation not available."}
    raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")


@app.get("/api/export/charts/analytics")
def export_analytics_charts(type: str = "welfare_timeline"):
    """Chart export returns Plotly JSON spec for frontend rendering."""
    t = (type or "welfare_timeline").lower()
    df = _read_history_df()
    if df.empty:
        return {"type": t, "plotly_json": None, "message": "No history data available."}

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if t == "welfare_timeline":
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["welfare_score"],
                mode="lines+markers",
                name="Welfare Score (%)",
                line=dict(color="#8b5cf6", width=3),
                fill="tozeroy",
            )
        )
        fig.update_layout(
            title="Welfare Timeline",
            xaxis_title="Time",
            yaxis_title="Score (%)",
            template="plotly_dark",
        )
        return {"type": t, "plotly_json": fig.to_plotly_json()}

    # default: return aggregates series
    analytics = get_analytics()
    return {"type": t, "plotly_json": None, "analytics": analytics}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)

