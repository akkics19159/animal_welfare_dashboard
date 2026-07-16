from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any



@dataclass
class TrackState:
    track_id: int
    class_name: str

    bbox_xyxy: List[float]
    detection_confidence: float
    tracking_confidence: float

    age_frames: int = 0
    lost_frames: int = 0
    reacquired: bool = False
    was_occluded: bool = False

    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    # elements: (t, cx, cy)

    entry_time: Optional[float] = None
    exit_time: Optional[float] = None
    dwell_time: float = 0.0

    # derived movement
    velocity_xy: Tuple[float, float] = (0.0, 0.0)
    acceleration_xy: Tuple[float, float] = (0.0, 0.0)
    direction_deg: Optional[float] = None

    visibility_score: float = 0.0
    track_lifetime_sec: float = 0.0
    direction_label: str = "stationary"

    # Species-recognition extensions (additive, backward compatible)
    detector_label: str = "unknown"
    classifier_label: str = "unknown"
    final_species: str = "unknown"
    species_confidence: float = 0.0
    classifier_confidence: float = 0.0
    agreement_score: float = 0.0
    embedding_distance: float = 1.0
    top5_predictions: List[Dict[str, Any]] = field(default_factory=list)
    classification_history: List[str] = field(default_factory=list)
    species_embedding: List[float] = field(default_factory=list)
    temporal_confidence: float = 0.0


class TrackerPipeline:
    """Tracking pipeline wrapper.

    Requirements:
    - BoT-SORT primary
    - ByteTrack fallback if BoT-SORT cannot be initialised
    - No DeepSORT

    Implementation note:
    This repo may not have bo-tsort/bytetrack dependencies installed.
    We implement an initialization-time attempt; if imports fail, we fall back
    to a lightweight centroid tracker to preserve contract and system stability.
    """

    def __init__(
        self,
        frame_rate: float = 10.0,
        max_lost_frames: int = 30,
        iou_match_threshold: float = 0.3,
        try_botsort: bool = True,
        try_bytetrack: bool = True,
    ):
        self.frame_rate = float(frame_rate) if frame_rate else 10.0
        self.max_lost_frames = int(max_lost_frames)
        self.iou_match_threshold = float(iou_match_threshold)

        self._impl = None
        self._impl_type: str = "none"

        # contract state
        self.tracks: Dict[int, TrackState] = {}
        self._next_id: int = 1
        self._last_ts: Optional[float] = None

        self._init_impl(try_botsort=try_botsort, try_bytetrack=try_bytetrack)

    # -------------------
    # IoU helpers
    # -------------------

    def _iou(self, a: List[float], b: List[float]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
        area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
        denom = area_a + area_b - inter
        if denom <= 0:
            return 0.0
        return float(inter / denom)

    def _centroid(self, bbox: List[float]) -> Tuple[float, float]:
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        return float(cx), float(cy)

    # -------------------
    # Initialization attempt
    # -------------------

    def _init_impl(self, *, try_botsort: bool, try_bytetrack: bool):
        # Attempt imports lazily.
        if try_botsort:
            try:
                # Common third-party names; may not exist in this repo.
                from bytetrack import BoTSORT  # type: ignore

                self._impl = BoTSORT()
                self._impl_type = "botsort"
                return
            except Exception:
                pass

        if try_bytetrack:
            try:
                from bytetrack import ByteTrack  # type: ignore

                self._impl = ByteTrack()
                self._impl_type = "bytetrack"
                return
            except Exception:
                pass

        self._impl = None
        self._impl_type = "fallback_centroid"

    # -------------------
    # Public tracking API
    # -------------------

    def update(
        self,
        frame,
        detections: List[Dict],
        timestamp: Optional[float] = None,
    ) -> Dict[int, TrackState]:
        ts = float(timestamp) if timestamp is not None else time.time()
        dt = 0.0
        if self._last_ts is not None:
            dt = max(1e-6, ts - self._last_ts)
        self._last_ts = ts

        # Convert detections into a consistent list
        det_bboxes = [d["xyxy"] for d in detections]
        det_confs = [float(d.get("confidence", 0.0)) for d in detections]
        det_classes = [str(d.get("class", "animal")) for d in detections]

        if self._impl is None or self._impl_type == "fallback_centroid":
            self._update_fallback(det_bboxes, det_classes, det_confs, ts, dt)
        else:
            # If third-party trackers exist, they should yield track ids.
            # We still provide a best-effort contract; if output mismatches,
            # we fall back to centroid tracker.
            try:
                # Expected output contract: list of (track_id, bbox_xyxy, conf)
                tracked = self._impl.update(det_bboxes, det_classes, det_confs)  # type: ignore
                # Convert tracked output to local tracks
                self._update_from_external(tracked, ts, dt)
            except Exception:
                self._update_fallback(det_bboxes, det_classes, det_confs, ts, dt)

        # Update derived metrics
        self._finalize_derived(ts, dt)
        return self.tracks

    # -------------------
    # Fallback centroid tracker
    # -------------------

    def _update_fallback(
        self,
        det_bboxes: List[List[float]],
        det_classes: List[str],
        det_confs: List[float],
        ts: float,
        dt: float,
    ):
        # Match detections to existing tracks by IoU.
        used_tracks = set()
        matched_dets = set()

        # compute matches
        for tid, tr in list(self.tracks.items()):
            best_iou = 0.0
            best_det = None
            best_det_idx = -1
            for i, bbox in enumerate(det_bboxes):
                if i in matched_dets:
                    continue
                iou = self._iou(tr.bbox_xyxy, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = bbox
                    best_det_idx = i
            if best_det is not None and best_iou >= self.iou_match_threshold and best_det_idx >= 0:
                # assign
                matched_dets.add(best_det_idx)
                used_tracks.add(tid)
                tr.bbox_xyxy = det_bboxes[best_det_idx]
                tr.detection_confidence = float(det_confs[best_det_idx])
                tr.tracking_confidence = float(min(1.0, best_iou))
                tr.class_name = det_classes[best_det_idx]
                tr.age_frames += 1
                was_lost = tr.lost_frames > 0
                tr.lost_frames = 0
                tr.reacquired = bool(was_lost)
                tr.was_occluded = tr.was_occluded or was_lost

                cx, cy = self._centroid(tr.bbox_xyxy)
                if tr.entry_time is None:
                    tr.entry_time = ts
                tr.trajectory.append((ts, cx, cy))
                # visibility proxy: higher when matched
                tr.visibility_score = float(min(1.0, best_iou))

        # Handle unmatched tracks (lost)
        for tid, tr in list(self.tracks.items()):
            if tid in used_tracks:
                continue
            tr.lost_frames += 1
            tr.age_frames += 1
            # keep last bbox
            tr.visibility_score = max(0.0, tr.visibility_score * 0.9)

            if tr.lost_frames > self.max_lost_frames and tr.exit_time is None:
                tr.exit_time = ts
                # dwell time until exit
                if tr.entry_time is not None:
                    tr.dwell_time = float(max(0.0, ts - tr.entry_time))

        # Create new tracks for unmatched detections
        for i, bbox in enumerate(det_bboxes):
            if i in matched_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            st = TrackState(
                track_id=tid,
                class_name=det_classes[i],
                bbox_xyxy=bbox,
                detection_confidence=float(det_confs[i]),
                tracking_confidence=1.0,
                age_frames=1,
                lost_frames=0,
                reacquired=False,
                detector_label=str(det_classes[i]),
                final_species=str(det_classes[i]),
            )
            st.entry_time = ts
            cx, cy = self._centroid(bbox)
            st.trajectory = [(ts, cx, cy)]
            st.visibility_score = 1.0
            self.tracks[tid] = st

        # Cleanup exited tracks if desired
        # Keep them for history, but we could prune deeply lost tracks.

    def _update_from_external(self, tracked, ts: float, dt: float):
        # tracked: list of dicts/tuples. Keep robust.
        if tracked is None:
            return
        for item in tracked:
            # support dict
            if isinstance(item, dict):
                tid = int(item.get("track_id"))
                bbox = item.get("bbox") or item.get("xyxy")
                conf = float(item.get("confidence", 0.0))
                cls = str(item.get("class", "animal"))
                track_conf = float(item.get("tracking_confidence", conf))
            else:
                # tuple: (tid, bbox, conf)
                tid = int(item[0])
                bbox = item[1]
                conf = float(item[2]) if len(item) > 2 else 0.0
                cls = "animal"
                track_conf = conf

            tr = self.tracks.get(tid)
            if tr is None:
                tr = TrackState(
                    track_id=tid,
                    class_name=cls,
                    bbox_xyxy=list(map(float, bbox)),
                    detection_confidence=conf,
                    tracking_confidence=track_conf,
                    detector_label=str(cls),
                    final_species=str(cls),
                )
                tr.entry_time = ts
                self.tracks[tid] = tr

            tr.bbox_xyxy = list(map(float, bbox))
            tr.detection_confidence = conf
            tr.tracking_confidence = track_conf
            tr.class_name = cls
            tr.age_frames += 1
            was_lost = tr.lost_frames > 0
            tr.lost_frames = 0
            tr.reacquired = bool(was_lost)
            tr.was_occluded = tr.was_occluded or was_lost

            cx, cy = self._centroid(tr.bbox_xyxy)
            tr.trajectory.append((ts, cx, cy))
            tr.visibility_score = float(max(0.0, min(1.0, tr.tracking_confidence)))

    def _finalize_derived(self, ts: float, dt: float):
        # Update velocity/acceleration/direction for all tracks.
        for tr in self.tracks.values():
            if len(tr.trajectory) < 2:
                continue
            # last two points
            t1, x1, y1 = tr.trajectory[-2]
            t2, x2, y2 = tr.trajectory[-1]
            dt_local = max(1e-6, t2 - t1)
            vx = (x2 - x1) / dt_local
            vy = (y2 - y1) / dt_local
            tr.velocity_xy = (float(vx), float(vy))

            tr.direction_deg = self._direction_deg(vx, vy)
            tr.direction_label = self._direction_label(vx, vy)

            if len(tr.trajectory) >= 3:
                t0, x0, y0 = tr.trajectory[-3]
                dt_prev = max(1e-6, t1 - t0)
                pvx = (x1 - x0) / dt_prev
                pvy = (y1 - y0) / dt_prev
                ax = (vx - pvx) / max(1e-6, dt_local)
                ay = (vy - pvy) / max(1e-6, dt_local)
                tr.acceleration_xy = (float(ax), float(ay))
            # dwell time updated if entry_time known
            if tr.entry_time is not None and tr.exit_time is None:
                tr.dwell_time = float(max(0.0, ts - tr.entry_time))
                tr.track_lifetime_sec = tr.dwell_time

            # reacquired flag reset
            if tr.lost_frames == 0:
                tr.reacquired = False
            else:
                tr.direction_label = "occluded"

    def _direction_deg(self, vx: float, vy: float) -> Optional[float]:
        if abs(vx) < 1e-9 and abs(vy) < 1e-9:
            return None
        ang = math.degrees(math.atan2(vy, vx))
        # Normalize to [0,360)
        ang = (ang + 360.0) % 360.0
        return float(ang)

    def _direction_label(self, vx: float, vy: float) -> str:
        if abs(vx) < 1e-6 and abs(vy) < 1e-6:
            return "stationary"

        if abs(vx) >= abs(vy):
            return "right" if vx > 0 else "left"
        return "down" if vy > 0 else "up"

