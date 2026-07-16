from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from .counting import IntelligentCounter
from .input_source_manager import InputSourceManager, NO_SOURCE_ERROR
from .tracking import TrackerPipeline
from .species_classifier import SpeciesClassifier
from .trajectory import TrajectoryAnalyticsComputer
from .yolo_detector import YoloDetector
from .vision_output import extend_vision_output


@dataclass
class VisionRunResult:
    video_result: Dict[str, Any]
    fps: Optional[float] = None


class VisionPipeline:
    """End-to-end vision intelligence pipeline.

        Preserves legacy output fields and adds modern tracking/counting analytics.
    """


    def __init__(
        self,
        *,
        yolo_model_path: str = "yolov8n.pt",
        yolo_confidence: float = 0.25,
        yolo_iou: float = 0.45,
        frame_process_max: int = 30,
        min_detected_for_motion: int = 1,
        roi_regions: Optional[List[Any]] = None,
        entry_lines: Optional[List[Any]] = None,
        exit_lines: Optional[List[Any]] = None,
    ):
        self.input_mgr = InputSourceManager()
        self.detector = YoloDetector(
            model_path=yolo_model_path,
            confidence=yolo_confidence,
            iou=yolo_iou,
        )
        self.tracker = TrackerPipeline(frame_rate=10.0)
        self.species_classifier = SpeciesClassifier()
        self.traj_computer = TrajectoryAnalyticsComputer()
        self.roi_regions = roi_regions or []
        self.entry_lines = entry_lines or []
        self.exit_lines = exit_lines or []
        self.counter = IntelligentCounter(
            regions=self.roi_regions,
            entry_lines=self.entry_lines,
            exit_lines=self.exit_lines,
        )

        self.frame_process_max = int(frame_process_max)
        self.min_detected_for_motion = int(min_detected_for_motion)

    def _estimate_motion_score_from_tracks(self, tracks: Dict[int, any]) -> float:
        # simple motion proxy using average instant speed
        speeds = []
        for tr in tracks.values():
            v = getattr(tr, "velocity_xy", (0.0, 0.0))
            speed_mag = float((v[0] ** 2 + v[1] ** 2) ** 0.5)
            speeds.append(speed_mag)
        if not speeds:
            return 0.0
        # normalize somewhat
        avg = sum(speeds) / len(speeds)
        return float(min(1.0, avg / 50.0))

    def _empty_video_result(self, error_message: str = NO_SOURCE_ERROR) -> Dict[str, Any]:
        return {
            "detections": [],
            "motion_score": 0.0,
            "agitated": False,
            "motion_agitation": False,
            "sentient_present": False,
            "frame_count": 0,
            "error": error_message,
            "fps": 0.0,
            "tracks": [],
            "occupancy": 0,
            "total_unique_individuals": 0,
            "species_wise_count": {},
            "region_wise_count": {},
            "entry_count": 0,
            "exit_count": 0,
            "maximum_occupancy": 0,
            "average_occupancy": 0.0,
        }

    def _detect_last_frame(self, frames: List[Any]) -> List[Dict[str, Any]]:
        if not frames:
            return []

        # Batch inference path for future throughput tuning.
        if len(frames) > 1:
            try:
                batch_dets = self.detector.detect_batch(frames)
                dets = batch_dets[-1] if batch_dets else []
            except Exception:
                dets = self.detector.detect(frames[-1])
        else:
            dets = self.detector.detect(frames[-1])

        return [
            {
                "class": d.class_name,
                "confidence": float(d.confidence),
                "xyxy": [float(v) for v in d.bbox_xyxy],
            }
            for d in dets
        ]

    def _attach_track_ids(self, detections: List[Dict[str, Any]], tracks: Dict[int, Any]) -> None:
        for det in detections:
            bbox = det.get("xyxy")
            if not bbox:
                continue
            best_tid = None
            best_iou = 0.0
            for tid, tr in tracks.items():
                tr_bbox = getattr(tr, "bbox_xyxy", None)
                if not tr_bbox:
                    continue
                x1 = max(bbox[0], tr_bbox[0])
                y1 = max(bbox[1], tr_bbox[1])
                x2 = min(bbox[2], tr_bbox[2])
                y2 = min(bbox[3], tr_bbox[3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                area_a = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
                area_b = max(0.0, tr_bbox[2] - tr_bbox[0]) * max(0.0, tr_bbox[3] - tr_bbox[1])
                denom = area_a + area_b - inter
                iou = float(inter / denom) if denom > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None and best_iou >= 0.1:
                det["track_id"] = int(best_tid)

    def _derive_behavioral_flags(self, tracks: Dict[int, Any], occupancy: int) -> Dict[str, bool]:
        long_inactivity = any(float(getattr(t, "stationary_duration", 0.0) or 0.0) >= 20.0 for t in tracks.values())
        repeated_circling = any(float(getattr(t, "smoothness", 0.0) or 0.0) <= 0.25 and len(getattr(t, "trajectory", []) or []) >= 8 for t in tracks.values())
        abnormal_movement = any(float(getattr(t, "avg_speed", 0.0) or 0.0) >= 120.0 for t in tracks.values())
        isolation = occupancy == 1
        overcrowding = occupancy >= 6
        escape_attempt = any(
            float(getattr(t, "velocity_xy", (0.0, 0.0))[0]) ** 2 + float(getattr(t, "velocity_xy", (0.0, 0.0))[1]) ** 2 >= 180.0 ** 2
            for t in tracks.values()
        )
        return {
            "overcrowding": overcrowding,
            "isolation": isolation,
            "long_inactivity": long_inactivity,
            "escape_attempt": escape_attempt,
            "repeated_circling": repeated_circling,
            "abnormal_movement": abnormal_movement,
        }

    def _build_result(self, *, frames: List[Any], source_meta: Any, frame_count: int, start_ts: float) -> VisionRunResult:
        if not frames or source_meta is None:
            return VisionRunResult(video_result=self._empty_video_result(), fps=0.0)

        ts = time.time()
        detections_legacy = self._detect_last_frame(frames)
        tracks = self.tracker.update(frames[-1], detections_legacy, timestamp=ts)

        visibility_scores: Dict[int, float] = {}
        for tid, tr in tracks.items():
            analytics = self.traj_computer.compute(
                getattr(tr, "trajectory", []) or [],
                dwell_time=getattr(tr, "dwell_time", 0.0) or 0.0,
            )
            tr.distance_travelled = analytics.distance_travelled
            tr.instant_speed = analytics.instant_speed
            tr.avg_speed = analytics.avg_speed
            tr.stationary_duration = analytics.stationary_duration
            tr.smoothness = analytics.smoothness
            tr.movement_density = analytics.movement_density
            tr.movement_direction_deg = analytics.movement_direction_deg
            visibility_scores[int(tid)] = float(getattr(tr, "visibility_score", 0.0) or 0.0)

        species_latency_ms = self.species_classifier.classify_tracks(
            frame_bgr=frames[-1],
            tracks=tracks,
        )

        motion_score = self._estimate_motion_score_from_tracks(tracks)
        agitated = motion_score >= 0.08
        counting_stats = self.counter.update(tracks)

        self._attach_track_ids(detections_legacy, tracks)

        occupancy = int(counting_stats.current_occupancy)
        behavioral_flags = self._derive_behavioral_flags(tracks, occupancy)

        legacy_video_result: Dict[str, Any] = {
            "detections": detections_legacy,
            "motion_score": float(motion_score),
            "agitated": bool(agitated),
            "motion_agitation": bool(agitated),
            "sentient_present": len(tracks) > 0,
            "frame_count": int(frame_count),
            "error": None,
            "source_type": getattr(source_meta, "source_type", None),
            "source": getattr(source_meta, "source", None),
            "observation_zones": [{"name": n, "polygon": p} for n, p in self.roi_regions],
            "entry_lines": [{"name": n, "line": l} for n, l in self.entry_lines],
            "exit_lines": [{"name": n, "line": l} for n, l in self.exit_lines],
            "species_classifier_model": getattr(self.species_classifier, "_backend_name", "none"),
            "species_classifier_loading_time_ms": float(getattr(self.species_classifier, "model_loading_time_ms", 0.0) or 0.0),
            "species_classification_latency_ms": float(species_latency_ms),
            **behavioral_flags,
        }

        dwell_vals = [float(getattr(t, "dwell_time", 0.0) or 0.0) for t in tracks.values()]
        avg_dwell = float(sum(dwell_vals) / max(1, len(dwell_vals))) if dwell_vals else 0.0
        longest_dwell = float(max(dwell_vals)) if dwell_vals else 0.0

        extended = extend_vision_output(
            legacy_video_result=legacy_video_result,
            tracks=tracks,
            occupancy=occupancy,
            counting={
                "current_occupancy": counting_stats.current_occupancy,
                "total_unique_individuals": counting_stats.total_unique_individuals,
                "species_wise_count": counting_stats.species_wise_count,
                "region_wise_count": counting_stats.region_wise_count,
                "entry_count": counting_stats.entry_count,
                "exit_count": counting_stats.exit_count,
                "maximum_occupancy": counting_stats.maximum_occupancy,
                "average_occupancy": counting_stats.average_occupancy,
                "average_dwell_time": avg_dwell,
                "longest_dwell_time": longest_dwell,
                "region_dwell_time": counting_stats.region_dwell_time,
                "region_movement_stats": counting_stats.region_movement_stats,
            },
            visibility_scores=visibility_scores,
        )

        elapsed = max(time.time() - start_ts, 1e-6)
        fps = float(frame_count / elapsed) if elapsed > 0 else 0.0
        extended["fps"] = fps
        return VisionRunResult(video_result=extended, fps=fps)

    def run_once(self, *, max_frames: int = 1) -> VisionRunResult:
        """Run one analysis cycle with webcam-first source selection."""

        frame_count = 0
        t0 = time.time()
        frames: List[Any] = []
        last_meta = None

        sample_frames = max(1, min(int(max_frames), self.frame_process_max))
        for frame, meta in self.input_mgr.get_frame_generator(max_frames=sample_frames):
            if frame is None:
                continue
            frames.append(frame)
            last_meta = meta
            frame_count += 1

        if not frames or last_meta is None:
            return VisionRunResult(video_result=self._empty_video_result(), fps=0.0)

        return self._build_result(
            frames=frames,
            source_meta=last_meta,
            frame_count=frame_count,
            start_ts=t0,
        )

    def run_stream(self, *, max_frames: Optional[int] = None) -> Generator[VisionRunResult, None, None]:
        """Yield continuously processed results for real-time monitoring."""
        start_ts = time.time()
        count = 0
        for frame, meta in self.input_mgr.get_frame_generator(max_frames=max_frames):
            if frame is None:
                yield VisionRunResult(video_result=self._empty_video_result(), fps=0.0)
                return
            count += 1
            yield self._build_result(
                frames=[frame],
                source_meta=meta,
                frame_count=count,
                start_ts=start_ts,
            )

