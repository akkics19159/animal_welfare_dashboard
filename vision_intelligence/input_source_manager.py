from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import cv2


SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".webm"}


def _default_video_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "video"


def _build_no_source_error(video_dir: Path) -> str:
    return (
        "No valid video source available.\n\n"
        "Please connect a webcam or place a supported video inside\n"
        f"{video_dir}"
    )


NO_SOURCE_ERROR = _build_no_source_error(_default_video_dir())


@dataclass
class SourceMetadata:
    source_type: str  # "webcam" | "video"
    source: str  # device index or filepath
    is_live: bool
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    selected_video_path: Optional[str] = None
    source_priority: int = 99


class InputSourceManager:
    """Webcam-first input manager.

    Rules:
      1) Try laptop webcam (index 0): open + frame health + responsiveness + fps acceptable.
      2) If webcam fails: discover local videos in data/video and pick newest valid file.
      3) If nothing works: yield no frames (pipeline must not crash).

    All cv2.VideoCapture usage for this subsystem is centralized here.
    """

    def __init__(
        self,
        webcam_index: int = 0,
        local_video_dir: Optional[str] = None,
        min_acceptable_fps: float = 5.0,
        webcam_health_timeout_sec: float = 2.5,
        source_health_check_every_n_frames: int = 30,
    ):
        self.webcam_index = webcam_index
        self.local_video_dir = Path(local_video_dir) if local_video_dir else _default_video_dir()
        self.min_acceptable_fps = float(min_acceptable_fps)
        self.webcam_health_timeout_sec = float(webcam_health_timeout_sec)
        self.source_health_check_every_n_frames = max(1, int(source_health_check_every_n_frames))

        self._last_probe: Dict[str, Optional[SourceMetadata]] = {"webcam": None, "video": None}

    # ----------------------
    # Webcam health checks
    # ----------------------

    def _try_open_capture(self, source) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(source)
        if cap is None or not cap.isOpened():
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            return None
        return cap

    def _read_one_frame(self, cap: cv2.VideoCapture):
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return frame

    def _measure_fps_quick(self, cap: cv2.VideoCapture, samples: int = 10) -> Optional[float]:
        # Try CAP_PROP_FPS first
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is not None and fps > 0:
                return float(fps)
        except Exception:
            pass

        # Fallback: quick timing loop
        frames = 0
        t0 = time.time()
        while frames < samples:
            ret, _ = cap.read()
            if not ret:
                break
            frames += 1
        dt = max(time.time() - t0, 1e-6)
        if frames <= 0:
            return None
        return frames / dt

    def _capture_metadata(self, cap: cv2.VideoCapture, *, source_type: str, source: str, is_live: bool, priority: int) -> SourceMetadata:
        fps_val: Optional[float] = None
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is not None and fps > 0:
                fps_val = float(fps)
        except Exception:
            fps_val = None

        width = None
        height = None
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        except Exception:
            pass

        return SourceMetadata(
            source_type=source_type,
            source=source,
            is_live=is_live,
            fps=fps_val,
            width=width,
            height=height,
            selected_video_path=source if source_type == "video" else None,
            source_priority=priority,
        )

    def _webcam_is_healthy(self) -> Tuple[bool, Optional[cv2.VideoCapture], Optional[SourceMetadata]]:
        cap = self._try_open_capture(self.webcam_index)
        if cap is None:
            return False, None, None

        t_deadline = time.time() + self.webcam_health_timeout_sec
        try:
            frame = self._read_one_frame(cap)
            if frame is None:
                cap.release()
                return False, None, None

            # responsiveness check: ensure we can read again
            while time.time() < t_deadline:
                frame2 = self._read_one_frame(cap)
                if frame2 is not None:
                    break
                time.sleep(0.05)
            else:
                cap.release()
                return False, None, None

            fps = self._measure_fps_quick(cap)
            if fps is None or fps < self.min_acceptable_fps:
                cap.release()
                return False, None, None

            meta = self._capture_metadata(
                cap,
                source_type="webcam",
                source=str(self.webcam_index),
                is_live=True,
                priority=1,
            )
            meta.fps = fps
            self._last_probe["webcam"] = meta
            return True, cap, meta
        except Exception:
            try:
                cap.release()
            except Exception:
                pass
            return False, None, None

    # ----------------------
    # Local video discovery
    # ----------------------

    def _is_supported_video(self, path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_VIDEO_EXTS

    def _list_candidate_videos(self):
        if not self.local_video_dir.exists():
            return []
        root = self.local_video_dir.resolve()
        videos = []
        for p in self.local_video_dir.rglob("*"):
            if not p.is_file() or not self._is_supported_video(p):
                continue
            try:
                resolved = p.resolve()
            except Exception:
                continue
            if root in resolved.parents or resolved == root:
                videos.append(p)
        # Newest by modification time
        videos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return videos

    def _is_video_readable(self, video_path: Path) -> bool:
        cap = None
        try:
            cap = self._try_open_capture(str(video_path))
            if cap is None:
                return False
            # Try reading a frame
            frame = self._read_one_frame(cap)
            return frame is not None
        except Exception:
            return False
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    def _open_newest_valid_video(self) -> Tuple[Optional[cv2.VideoCapture], Optional[SourceMetadata]]:
        candidates = self._list_candidate_videos()
        for p in candidates:
            if not self._is_video_readable(p):
                continue
            cap = self._try_open_capture(str(p))
            if cap is None:
                continue
            try:
                meta = self._capture_metadata(
                    cap,
                    source_type="video",
                    source=str(p),
                    is_live=False,
                    priority=2,
                )
                self._last_probe["video"] = meta
                return cap, meta
            except Exception:
                try:
                    cap.release()
                except Exception:
                    pass
                continue
        return None, None

    # ----------------------
    # Public API
    # ----------------------

    def probe_source(self) -> Tuple[Optional[cv2.VideoCapture], Optional[SourceMetadata], Optional[str]]:
        """Open the highest-priority healthy source and return capture + metadata."""

        ok, webcam_cap, webcam_meta = self._webcam_is_healthy()
        if ok and webcam_cap is not None and webcam_meta is not None:
            return webcam_cap, webcam_meta, None

        video_cap, video_meta = self._open_newest_valid_video()
        if video_cap is not None and video_meta is not None:
            return video_cap, video_meta, None

        return None, None, NO_SOURCE_ERROR

    def read_frame_once(self) -> Tuple[Optional[any], Optional[SourceMetadata], Optional[str]]:
        """Read one frame using the mandatory source priority logic."""

        cap, meta, err = self.probe_source()
        if cap is None or meta is None:
            return None, None, err or NO_SOURCE_ERROR
        try:
            frame = self._read_one_frame(cap)
            if frame is None:
                return None, meta, NO_SOURCE_ERROR
            return frame, meta, None
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def get_frame_generator(self, *, max_frames: Optional[int] = None) -> Generator[Tuple[Optional[any], Optional[SourceMetadata]], None, None]:
        """Yield (frame, metadata). If no source, yields (None, None) once and stops.

        The manager automatically attempts source switching when a source fails.
        """

        cap, meta, err = self.probe_source()
        if cap is None or meta is None:
            yield None, None
            return

        count = 0
        source_frame_count = 0
        try:
            while True:
                frame = self._read_one_frame(cap)
                if frame is None:
                    # Source exhausted or unhealthy. Attempt immediate failover once.
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap, meta, err = self.probe_source()
                    if cap is None or meta is None:
                        break
                    source_frame_count = 0
                    continue

                yield frame, meta
                count += 1
                source_frame_count += 1

                if max_frames is not None and count >= max_frames:
                    break

                # Periodic health check for live camera responsiveness.
                if (
                    meta.source_type == "webcam"
                    and source_frame_count % self.source_health_check_every_n_frames == 0
                ):
                    probe_frame = self._read_one_frame(cap)
                    if probe_frame is None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap, meta, err = self.probe_source()
                        if cap is None or meta is None:
                            break
                        source_frame_count = 0
                    else:
                        # continue with the health-probe frame to avoid dropping data
                        yield probe_frame, meta
                        count += 1
                        source_frame_count += 1
                        if max_frames is not None and count >= max_frames:
                            break
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def get_frame_generator_for_source(
        self,
        source: any,
        *,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[Optional[any], Optional[SourceMetadata]], None, None]:
        """Compatibility stream for explicit source requests.

        This method is used by legacy helpers that request a concrete source path.
        """
        cap = self._try_open_capture(source)
        if cap is None:
            yield None, None
            return

        src_type = "webcam" if isinstance(source, int) else "video"
        meta = self._capture_metadata(
            cap,
            source_type=src_type,
            source=str(source),
            is_live=src_type == "webcam",
            priority=1 if src_type == "webcam" else 2,
        )

        count = 0
        try:
            while True:
                frame = self._read_one_frame(cap)
                if frame is None:
                    break
                yield frame, meta
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
        finally:
            try:
                cap.release()
            except Exception:
                pass

