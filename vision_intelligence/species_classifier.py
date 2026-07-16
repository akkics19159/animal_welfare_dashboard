from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class SpeciesPrediction:
    species: str
    confidence: float
    top5_predictions: List[Dict[str, Any]]
    feature_embedding: List[float]
    embedding_distance: float


class _BaseClassifierBackend:
    name = "unavailable"
    priority = 999

    def available(self) -> bool:
        return False

    def classify_batch(
        self,
        crops_bgr: Sequence[np.ndarray],
        taxonomy: Sequence[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        return [[] for _ in crops_bgr]


class _YoloLabelFallbackBackend(_BaseClassifierBackend):
    name = "yolo_fallback"
    priority = 100

    _MAP = {
        "person": "Human",
        "dog": "Dog",
        "cat": "Cat",
        "cow": "Cow",
        "horse": "Horse",
        "goat": "Goat",
        "sheep": "Sheep",
        "pig": "Pig",
        "bird": "Bird",
        "rabbit": "Rabbit",
        "monkey": "Monkey",
        "bear": "Bear",
        "elephant": "Elephant",
        "zebra": "Zebra",
        "giraffe": "Giraffe",
        "camel": "Camel",
        "buffalo": "Buffalo",
        "chicken": "Chicken",
        "duck": "Duck",
        "deer": "Deer",
    }

    def available(self) -> bool:
        return True

    def classify_from_detector(self, detector_label: str, taxonomy: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        mapped = self._MAP.get(str(detector_label or "").lower().strip(), "Other")
        scores = {row["name"]: 0.01 for row in taxonomy}
        scores[mapped] = 0.90
        total = sum(scores.values())
        out: List[Dict[str, Any]] = []
        for row in taxonomy:
            species = row["name"]
            conf = float(scores.get(species, 0.0) / total)
            out.append(
                {
                    "species": species,
                    "group": row.get("group", "unknown"),
                    "score": conf,
                    "distance": float(max(0.0, 1.0 - conf)),
                    "embedding": [conf],
                }
            )
        out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return out


class _OpenClipBackend(_BaseClassifierBackend):
    def __init__(self, *, name: str, model_name: str, pretrained: Optional[str], priority: int):
        self.name = name
        self.model_name = model_name
        self.pretrained = pretrained
        self.priority = priority

        self._ready = False
        self._load_error = False
        self._device = "cpu"
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_cache_key: Optional[Tuple[str, ...]] = None
        self._text_embeddings = None

    def _lazy_load(self) -> bool:
        if self._ready:
            return True
        if self._load_error:
            return False
        try:
            import torch
            import open_clip

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
            )
            model = model.to(self._device)
            model.eval()
            tokenizer = open_clip.get_tokenizer(self.model_name)

            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            self._ready = True
            return True
        except Exception:
            self._load_error = True
            self._ready = False
            return False

    def available(self) -> bool:
        return self._lazy_load()

    def _build_text_embeddings(self, prompts: Sequence[str]) -> bool:
        if not self._ready:
            return False
        key = tuple(prompts)
        if self._text_cache_key == key and self._text_embeddings is not None:
            return True
        try:
            import torch

            tokens = self._tokenizer(list(prompts)).to(self._device)
            with torch.no_grad():
                text_f = self._model.encode_text(tokens)
                text_f = text_f / text_f.norm(dim=-1, keepdim=True)
            self._text_embeddings = text_f
            self._text_cache_key = key
            return True
        except Exception:
            return False

    def classify_batch(
        self,
        crops_bgr: Sequence[np.ndarray],
        taxonomy: Sequence[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        if not self._lazy_load():
            return [[] for _ in crops_bgr]

        prompts = [str((row.get("prompts") or [f"a photo of a {row.get('name', 'animal').lower()}"])[0]) for row in taxonomy]
        if not self._build_text_embeddings(prompts):
            return [[] for _ in crops_bgr]

        try:
            import torch
            from PIL import Image

            valid_indices: List[int] = []
            image_tensors = []
            for idx, crop in enumerate(crops_bgr):
                if crop is None or crop.size == 0:
                    continue
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                image_tensors.append(self._preprocess(pil))
                valid_indices.append(idx)

            if not image_tensors:
                return [[] for _ in crops_bgr]

            images = torch.stack(image_tensors, dim=0).to(self._device)
            with torch.no_grad():
                image_f = self._model.encode_image(images)
                image_f = image_f / image_f.norm(dim=-1, keepdim=True)
                logits = image_f @ self._text_embeddings.T
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(float)
                sims = logits.detach().cpu().numpy().astype(float)
                emb = image_f.detach().cpu().numpy().astype(float)

            batch_rows: List[List[Dict[str, Any]]] = [[] for _ in crops_bgr]
            for local_i, original_i in enumerate(valid_indices):
                rows: List[Dict[str, Any]] = []
                for tax_i, tax in enumerate(taxonomy):
                    score = float(probs[local_i][tax_i])
                    similarity = float(sims[local_i][tax_i])
                    rows.append(
                        {
                            "species": tax.get("name", "Other"),
                            "group": tax.get("group", "unknown"),
                            "score": score,
                            "similarity": similarity,
                            "distance": float(max(0.0, 1.0 - similarity)),
                            "embedding": emb[local_i].tolist(),
                        }
                    )
                rows.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
                batch_rows[original_i] = rows
            return batch_rows
        except Exception:
            return [[] for _ in crops_bgr]


class SpeciesClassifier:
    def __init__(
        self,
        *,
        taxonomy_path: Optional[str] = None,
        backend_priority: Optional[List[str]] = None,
        temporal_window: int = 30,
        species_switch_margin: float = 0.08,
        low_confidence_threshold: float = 0.28,
        calibration_weights: Optional[Dict[str, float]] = None,
        backends: Optional[List[_BaseClassifierBackend]] = None,
    ):
        base_dir = Path(__file__).resolve().parent
        self.taxonomy_path = Path(taxonomy_path) if taxonomy_path else (base_dir / "species_taxonomy.json")
        self.temporal_window = int(max(5, temporal_window))
        self.species_switch_margin = float(max(0.0, species_switch_margin))
        self.low_confidence_threshold = float(max(0.0, min(1.0, low_confidence_threshold)))
        self.calibration_weights = calibration_weights or {
            "yolo": 0.15,
            "classifier": 0.35,
            "embedding": 0.20,
            "temporal": 0.20,
            "agreement": 0.10,
        }

        self._taxonomy = self._load_taxonomy()
        self._history: Dict[int, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.temporal_window))
        self._crop_cache: Dict[Tuple[int, Tuple[int, int, int, int]], SpeciesPrediction] = {}

        self._backend_priority = backend_priority or [
            "openclip_vith14",
            "openclip_vitl14",
            "siglip2",
            "yolo_fallback",
        ]
        self._backend_candidates = backends or self._default_backends()
        self._backend: Optional[_BaseClassifierBackend] = None
        self._backend_name = "none"
        self.model_loading_time_ms = 0.0

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def _load_taxonomy(self) -> List[Dict[str, Any]]:
        fallback = [
            {"name": "Human", "group": "human", "prompts": ["a photo of a human", "a photo of a person"]},
            {"name": "Dog", "group": "mammal", "prompts": ["a photo of a dog"]},
            {"name": "Cat", "group": "mammal", "prompts": ["a photo of a cat"]},
            {"name": "Cow", "group": "mammal", "prompts": ["a photo of a cow"]},
            {"name": "Horse", "group": "mammal", "prompts": ["a photo of a horse"]},
            {"name": "Goat", "group": "mammal", "prompts": ["a photo of a goat"]},
            {"name": "Sheep", "group": "mammal", "prompts": ["a photo of a sheep"]},
            {"name": "Pig", "group": "mammal", "prompts": ["a photo of a pig"]},
            {"name": "Bird", "group": "bird", "prompts": ["a photo of a bird"]},
            {"name": "Rabbit", "group": "mammal", "prompts": ["a photo of a rabbit"]},
            {"name": "Monkey", "group": "mammal", "prompts": ["a photo of a monkey"]},
            {"name": "Bear", "group": "mammal", "prompts": ["a photo of a bear"]},
            {"name": "Elephant", "group": "mammal", "prompts": ["a photo of an elephant"]},
            {"name": "Zebra", "group": "mammal", "prompts": ["a photo of a zebra"]},
            {"name": "Giraffe", "group": "mammal", "prompts": ["a photo of a giraffe"]},
            {"name": "Camel", "group": "mammal", "prompts": ["a photo of a camel"]},
            {"name": "Buffalo", "group": "mammal", "prompts": ["a photo of a buffalo"]},
            {"name": "Chicken", "group": "bird", "prompts": ["a photo of a chicken"]},
            {"name": "Duck", "group": "bird", "prompts": ["a photo of a duck"]},
            {"name": "Deer", "group": "mammal", "prompts": ["a photo of a deer"]},
            {"name": "Other", "group": "unknown", "prompts": ["a photo of another animal"]},
        ]
        try:
            payload = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))
            rows = payload.get("species") if isinstance(payload, dict) else None
            if isinstance(rows, list) and rows:
                return rows
        except Exception:
            pass
        return fallback

    def _default_backends(self) -> List[_BaseClassifierBackend]:
        return [
            _OpenClipBackend(name="openclip_vith14", model_name="ViT-H-14", pretrained="laion2b_s32b_b79k", priority=1),
            _OpenClipBackend(name="openclip_vitl14", model_name="ViT-L-14", pretrained="openai", priority=2),
            _OpenClipBackend(name="siglip2", model_name="ViT-SO400M-14-SigLIP2-384", pretrained="webli", priority=3),
            _YoloLabelFallbackBackend(),
        ]

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return
        t0 = time.time()

        by_name = {b.name: b for b in self._backend_candidates}
        ordered: List[_BaseClassifierBackend] = []
        for name in self._backend_priority:
            b = by_name.get(name)
            if b is not None:
                ordered.append(b)
        for b in sorted(self._backend_candidates, key=lambda x: x.priority):
            if b not in ordered:
                ordered.append(b)

        for backend in ordered:
            if backend.available():
                self._backend = backend
                self._backend_name = backend.name
                break

        if self._backend is None:
            self._backend = _YoloLabelFallbackBackend()
            self._backend_name = "yolo_fallback"

        self.model_loading_time_ms = float(max(0.0, (time.time() - t0) * 1000.0))

    def _map_detector_to_species(self, detector_label: str) -> str:
        return _YoloLabelFallbackBackend._MAP.get(str(detector_label or "").lower().strip(), "Other")

    def _safe_crop(self, frame_bgr: np.ndarray, bbox_xyxy: Sequence[float]) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        x1 = int(max(0, min(w - 1, round(float(bbox_xyxy[0])))))
        y1 = int(max(0, min(h - 1, round(float(bbox_xyxy[1])))))
        x2 = int(max(x1 + 1, min(w, round(float(bbox_xyxy[2])))))
        y2 = int(max(y1 + 1, min(h, round(float(bbox_xyxy[3])))))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return crop

    def _normalize_rows(self, rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in rows:
            conf = float(row.get("score", row.get("confidence", 0.0)) or 0.0)
            sim = float(row.get("similarity", conf) or conf)
            dist = float(row.get("distance", max(0.0, 1.0 - sim)) or max(0.0, 1.0 - sim))
            out.append(
                {
                    "species": str(row.get("species", "Other")),
                    "group": str(row.get("group", "unknown")),
                    "confidence": conf,
                    "similarity": sim,
                    "distance": dist,
                    "embedding": list(row.get("embedding", [])) if isinstance(row.get("embedding", []), list) else [],
                }
            )
        out.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        return out

    def _temporal_vote(self, track_id: int, species: str, confidence: float) -> Tuple[str, float, List[str], float]:
        hist = self._history[track_id]

        prev_species = hist[-1]["final_species"] if hist else species
        prev_conf = float(hist[-1]["species_confidence"]) if hist else 0.0
        if species != prev_species and confidence <= prev_conf + self.species_switch_margin:
            species = prev_species
            confidence = max(confidence, prev_conf * 0.97)

        hist.append({"final_species": species, "species_confidence": confidence})

        weights: Dict[str, float] = {}
        total = 0.0
        confidence_acc = 0.0
        n = len(hist)
        for idx, item in enumerate(hist):
            recency_w = (idx + 1) / max(1, n)
            weighted = recency_w * float(item["species_confidence"])
            sp = str(item["final_species"])
            weights[sp] = weights.get(sp, 0.0) + weighted
            confidence_acc += weighted
            total += recency_w

        temporal_species = max(weights.items(), key=lambda kv: kv[1])[0] if weights else species
        temporal_conf = float(confidence_acc / max(1e-6, total))
        temporal_consensus = float(weights.get(temporal_species, 0.0) / max(1e-6, sum(weights.values())))
        history = [str(item["final_species"]) for item in hist]
        return temporal_species, temporal_conf, history, temporal_consensus

    def _calibrate_confidence(
        self,
        *,
        detector_confidence: float,
        classifier_confidence: float,
        embedding_distance: float,
        temporal_confidence: float,
        agreement_score: float,
    ) -> float:
        weights = self.calibration_weights
        similarity = float(max(0.0, min(1.0, 1.0 - embedding_distance)))
        calibrated = (
            float(weights.get("yolo", 0.0)) * detector_confidence
            + float(weights.get("classifier", 0.0)) * classifier_confidence
            + float(weights.get("embedding", 0.0)) * similarity
            + float(weights.get("temporal", 0.0)) * temporal_confidence
            + float(weights.get("agreement", 0.0)) * agreement_score
        )
        return float(max(0.0, min(1.0, calibrated)))

    def _predict_single(
        self,
        *,
        rows: List[Dict[str, Any]],
        detector_label: str,
        detector_confidence: float,
        track_id: Optional[int],
    ) -> SpeciesPrediction:
        normalized = self._normalize_rows(rows)
        if not normalized:
            mapped = self._map_detector_to_species(detector_label)
            normalized = [{"species": mapped, "group": "unknown", "confidence": detector_confidence, "similarity": detector_confidence, "distance": 1.0 - detector_confidence, "embedding": []}]

        top5 = normalized[:5]
        top1 = top5[0]

        classifier_species = str(top1.get("species", "Other"))
        classifier_conf = float(top1.get("confidence", 0.0) or 0.0)
        embedding_distance = float(top1.get("distance", max(0.0, 1.0 - classifier_conf)) or max(0.0, 1.0 - classifier_conf))
        detector_species = self._map_detector_to_species(detector_label)

        if classifier_conf < self.low_confidence_threshold:
            candidate_species = detector_species
            classifier_conf = max(classifier_conf, detector_confidence * 0.75)
        else:
            candidate_species = classifier_species

        agreement = 1.0 if detector_species == classifier_species else 0.25

        temporal_species = candidate_species
        temporal_conf = classifier_conf
        temporal_consensus = 1.0
        history: List[str] = [candidate_species]
        if track_id is not None:
            temporal_species, temporal_conf, history, temporal_consensus = self._temporal_vote(
                track_id=track_id,
                species=candidate_species,
                confidence=classifier_conf,
            )

        final_conf = self._calibrate_confidence(
            detector_confidence=float(max(0.0, min(1.0, detector_confidence))),
            classifier_confidence=float(max(0.0, min(1.0, classifier_conf))),
            embedding_distance=float(max(0.0, min(1.0, embedding_distance))),
            temporal_confidence=float(max(0.0, min(1.0, temporal_conf))),
            agreement_score=float(max(0.0, min(1.0, agreement * temporal_consensus))),
        )

        payload_top5 = [
            {
                "species": row["species"],
                "confidence": float(row["confidence"]),
                "similarity": float(row["similarity"]),
                "distance": float(row["distance"]),
                "group": row["group"],
            }
            for row in top5
        ]

        embedding = top1.get("embedding") if isinstance(top1.get("embedding"), list) else []
        if not embedding:
            embedding = [float(row["confidence"]) for row in top5]

        return SpeciesPrediction(
            species=str(temporal_species),
            confidence=float(final_conf),
            top5_predictions=payload_top5,
            feature_embedding=[float(x) for x in embedding[:256]],
            embedding_distance=float(embedding_distance),
        )

    def predict(
        self,
        image: np.ndarray,
        *,
        detector_label: str = "unknown",
        detector_confidence: float = 0.0,
        track_id: Optional[int] = None,
    ) -> SpeciesPrediction:
        self._ensure_backend()

        backend = self._backend
        rows: List[Dict[str, Any]] = []
        if isinstance(backend, _YoloLabelFallbackBackend):
            rows = backend.classify_from_detector(detector_label, self._taxonomy)
        else:
            batch_rows = backend.classify_batch([image], self._taxonomy) if backend else [[]]
            rows = batch_rows[0] if batch_rows else []

        if not rows:
            fallback = _YoloLabelFallbackBackend()
            rows = fallback.classify_from_detector(detector_label, self._taxonomy)
        return self._predict_single(
            rows=rows,
            detector_label=detector_label,
            detector_confidence=detector_confidence,
            track_id=track_id,
        )

    def classify_tracks(self, *, frame_bgr: np.ndarray, tracks: Dict[int, Any]) -> float:
        self._ensure_backend()
        t0 = time.time()
        self._crop_cache.clear()

        tids: List[int] = []
        detector_labels: List[str] = []
        detector_confidences: List[float] = []
        crops: List[np.ndarray] = []
        cache_keys: List[Tuple[int, Tuple[int, int, int, int]]] = []

        for tid, tr in tracks.items():
            if int(getattr(tr, "lost_frames", 0) or 0) > 0:
                continue
            bbox = getattr(tr, "bbox_xyxy", None)
            if not bbox:
                continue

            rounded_bbox = (
                int(round(float(bbox[0]))),
                int(round(float(bbox[1]))),
                int(round(float(bbox[2]))),
                int(round(float(bbox[3]))),
            )
            cache_key = (int(tid), rounded_bbox)

            detector_label = str(getattr(tr, "detector_label", getattr(tr, "class_name", "unknown")) or "unknown")
            detector_conf = float(getattr(tr, "detection_confidence", 0.0) or 0.0)

            if cache_key in self._crop_cache:
                pred = self._crop_cache[cache_key]
                self._assign_prediction(
                    tr,
                    pred,
                    track_id=int(tid),
                    detector_label=detector_label,
                    detector_confidence=detector_conf,
                )
                continue

            crop = self._safe_crop(frame_bgr, bbox)
            tids.append(int(tid))
            detector_labels.append(detector_label)
            detector_confidences.append(detector_conf)
            crops.append(crop)
            cache_keys.append(cache_key)

        batch_rows: List[List[Dict[str, Any]]] = [[] for _ in crops]
        if crops:
            if isinstance(self._backend, _YoloLabelFallbackBackend):
                fallback = self._backend
                batch_rows = [fallback.classify_from_detector(det_label, self._taxonomy) for det_label in detector_labels]
            else:
                try:
                    batch_rows = self._backend.classify_batch(crops, self._taxonomy) if self._backend else [[] for _ in crops]
                except Exception:
                    batch_rows = [[] for _ in crops]

        for idx, tid in enumerate(tids):
            tr = tracks.get(tid)
            if tr is None:
                continue
            detector_label = detector_labels[idx]
            detector_conf = detector_confidences[idx]
            rows = batch_rows[idx] if idx < len(batch_rows) else []
            if not rows:
                rows = _YoloLabelFallbackBackend().classify_from_detector(detector_label, self._taxonomy)

            prediction = self._predict_single(
                rows=rows,
                detector_label=detector_label,
                detector_confidence=detector_conf,
                track_id=tid,
            )
            self._crop_cache[cache_keys[idx]] = prediction
            self._assign_prediction(
                tr,
                prediction,
                track_id=int(tid),
                detector_label=detector_label,
                detector_confidence=detector_conf,
            )

        return float(max(0.0, (time.time() - t0) * 1000.0))

    def _assign_prediction(
        self,
        tr: Any,
        prediction: SpeciesPrediction,
        *,
        track_id: int,
        detector_label: str,
        detector_confidence: float,
    ) -> None:
        classifier_label = prediction.top5_predictions[0]["species"] if prediction.top5_predictions else prediction.species

        setattr(tr, "detector_label", str(detector_label))
        setattr(tr, "classifier_label", str(classifier_label))
        setattr(tr, "final_species", str(prediction.species))
        setattr(tr, "class_name", str(prediction.species))
        setattr(tr, "species", str(prediction.species))

        setattr(tr, "species_confidence", float(prediction.confidence))
        setattr(tr, "classifier_confidence", float(prediction.top5_predictions[0].get("confidence", 0.0) if prediction.top5_predictions else 0.0))
        setattr(tr, "classification_confidence", float(prediction.confidence))

        similarity = float(max(0.0, min(1.0, 1.0 - prediction.embedding_distance)))
        agreement = 1.0 if self._map_detector_to_species(detector_label) == classifier_label else 0.25
        agreement = float(max(0.0, min(1.0, 0.5 * agreement + 0.5 * similarity)))

        setattr(tr, "agreement_score", agreement)
        setattr(tr, "embedding_distance", float(prediction.embedding_distance))
        setattr(tr, "top5_predictions", prediction.top5_predictions)
        setattr(tr, "top_5_predictions", prediction.top5_predictions)
        hist = self._history.get(int(track_id), deque())
        setattr(
            tr,
            "classification_history",
            [
                str(item["final_species"])
                for item in hist
            ],
        )
        setattr(tr, "species_embedding", prediction.feature_embedding)
        last_temporal = float(hist[-1]["species_confidence"]) if hist else float(prediction.confidence)
        setattr(tr, "temporal_confidence", last_temporal)
        setattr(tr, "species_model_name", self._backend_name)
        setattr(tr, "detection_confidence", float(detector_confidence))
