from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


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


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: List[float]  # [x1,y1,x2,y2]


class YoloDetector:
    """YOLOv8+ detector with model caching and device selection."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        sentient_classes: Optional[set] = None,
    ):
        self.model_path = model_path
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.sentient_classes = sentient_classes or SENTIENT_CLASSES

        self._model = None
        self._device = self._auto_device()

    _MODEL_CACHE: Dict[Tuple[str, str], any] = {}
    _CACHE_LOCK = threading.Lock()

    def _auto_device(self) -> str:
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _load_model(self):
        if YOLO is None:
            raise RuntimeError("ultralytics is not available; install ultralytics for YOLO detection")
        if self._model is None:
            candidates = [self.model_path, "yolo11n.pt", "yolov8n.pt"]
            seen = set()
            unique_candidates = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    unique_candidates.append(c)

            loaded = None
            with self._CACHE_LOCK:
                for cand in unique_candidates:
                    cache_key = (cand, self._device)
                    cached = self._MODEL_CACHE.get(cache_key)
                    if cached is not None:
                        loaded = cached
                        self.model_path = cand
                        break
                    try:
                        cached = YOLO(cand)
                        self._MODEL_CACHE[cache_key] = cached
                        loaded = cached
                        self.model_path = cand
                        break
                    except Exception:
                        continue
            if loaded is None:
                raise RuntimeError("Failed to load YOLO model. Tried: " + ", ".join(unique_candidates))
            self._model = loaded
        return self._model

    @torch.no_grad()
    def detect(self, frame) -> List[Detection]:
        model = self._load_model()

        # ultralytics runs on device arg via model.predict
        results = model.predict(
            source=frame,
            verbose=False,
            conf=self.confidence,
            iou=self.iou,
            device=0 if self._device == "cuda" else "cpu",
        )

        dets: List[Detection] = []
        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
            boxes = r.boxes
            if boxes.xyxy is None or len(boxes.xyxy) == 0:
                continue

            # boxes: xyxy (n,4), conf (n,), cls (n,)
            for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                class_idx = int(cls.item()) if hasattr(cls, "item") else int(cls)
                conf_val = float(conf.item()) if hasattr(conf, "item") else float(conf)
                name = model.names.get(class_idx, str(class_idx)) if hasattr(model, "names") else str(class_idx)
                if name in self.sentient_classes:
                    dets.append(
                        Detection(
                            class_name=name,
                            confidence=conf_val,
                            bbox_xyxy=[float(v) for v in xyxy.tolist()],
                        )
                    )
        return dets

    @torch.no_grad()
    def detect_batch(self, frames: List[any]) -> List[List[Detection]]:
        if not frames:
            return []

        model = self._load_model()
        results = model.predict(
            source=frames,
            verbose=False,
            conf=self.confidence,
            iou=self.iou,
            device=0 if self._device == "cuda" else "cpu",
        )

        all_dets: List[List[Detection]] = []
        for r in results:
            frame_dets: List[Detection] = []
            if not hasattr(r, "boxes") or r.boxes is None or r.boxes.xyxy is None:
                all_dets.append(frame_dets)
                continue
            for xyxy, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                class_idx = int(cls.item()) if hasattr(cls, "item") else int(cls)
                conf_val = float(conf.item()) if hasattr(conf, "item") else float(conf)
                name = model.names.get(class_idx, str(class_idx)) if hasattr(model, "names") else str(class_idx)
                if name in self.sentient_classes:
                    frame_dets.append(
                        Detection(
                            class_name=name,
                            confidence=conf_val,
                            bbox_xyxy=[float(v) for v in xyxy.tolist()],
                        )
                    )
            all_dets.append(frame_dets)
        return all_dets

