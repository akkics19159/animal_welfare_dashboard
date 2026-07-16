from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import base64
from io import BytesIO

import numpy as np
import PIL.Image as Image
import streamlit as st


def _decode_image_to_pil(img: Any) -> Optional[Image.Image]:
    """Accept a few possible image formats (best-effort)."""
    if img is None:
        return None

    if isinstance(img, Image.Image):
        return img

    if isinstance(img, str):
        # data URL or base64
        if img.startswith("data:image"):
            img = img.split(",", 1)[1]
        try:
            raw = base64.b64decode(img)
            return Image.open(BytesIO(raw)).convert("RGB")
        except Exception:
            return None

    if isinstance(img, np.ndarray):
        try:
            if img.dtype != np.uint8:
                arr = img.astype(np.uint8)
            else:
                arr = img
            return Image.fromarray(arr)
        except Exception:
            return None

    return None


def render_live_camera_view(data: Dict[str, Any]) -> None:
    st.subheader("Live camera feed")

    frame = data.get("frame") or data.get("image")
    pil = _decode_image_to_pil(frame)

    if pil is None:
        st.info("No frame data available from the current run.")
        return

    # Draw boxes if xyxy provided. This is best-effort and assumes xyxy is in pixel coords.
    video_result = data.get("video_result", {}) or {}
    detections = video_result.get("detections", []) or []
    tracks = video_result.get("tracks", []) or []

    if detections or tracks:
        import PIL.ImageDraw as ImageDraw
        import PIL.ImageFont as ImageFont

        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        w, h = pil.size
        for d in detections:
            xyxy = d.get("xyxy")
            if not xyxy or not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = xyxy
            # Clamp
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))

            track_id = d.get("track_id") or d.get("id")
            label = str(d.get("class") or "animal")
            conf = d.get("confidence")
            txt = f"{label}#{track_id}" if track_id is not None else f"{label}"
            if conf is not None:
                try:
                    txt += f" {float(conf):.2f}"
                except Exception:
                    pass

            draw.rectangle([x1, y1, x2, y2], outline="#60a5fa", width=3)
            if font is not None:
                text_w, text_h = draw.textbbox((0, 0), txt, font=font)[2:]
                draw.rectangle([x1, max(0, y1 - text_h), x1 + text_w + 4, y1], fill="#1d4ed8")
                draw.text((x1 + 2, max(0, y1 - text_h)), txt, fill="white", font=font)

        # Draw trajectories and direction arrows for tracked objects.
        for t in tracks:
            tr = t.get("trajectory") or []
            if len(tr) >= 2:
                points: List[Tuple[int, int]] = []
                for item in tr[-20:]:
                    if not isinstance(item, (list, tuple)) or len(item) < 3:
                        continue
                    points.append((int(item[1]), int(item[2])))
                if len(points) >= 2:
                    draw.line(points, fill="#f59e0b", width=2)

            bbox = t.get("bbox")
            vel = t.get("velocity") or [0.0, 0.0]
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                cx = int(0.5 * (bbox[0] + bbox[2]))
                cy = int(0.5 * (bbox[1] + bbox[3]))
                try:
                    vx = float(vel[0])
                    vy = float(vel[1])
                except Exception:
                    vx, vy = 0.0, 0.0
                draw.line([(cx, cy), (int(cx + 0.05 * vx), int(cy + 0.05 * vy))], fill="#ef4444", width=2)

        # Optional ROI overlays if provided by pipeline config.
        for region in video_result.get("observation_zones", []) or []:
            pts = region.get("polygon") or []
            if len(pts) >= 3:
                draw.polygon([(int(p[0]), int(p[1])) for p in pts], outline="#10b981", width=2)

        for region in video_result.get("restricted_zones", []) or []:
            pts = region.get("polygon") or []
            if len(pts) >= 3:
                draw.polygon([(int(p[0]), int(p[1])) for p in pts], outline="#dc2626", width=2)

        for line in video_result.get("entry_lines", []) or []:
            p1, p2 = line.get("line", [None, None])
            if p1 and p2:
                draw.line([(int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))], fill="#22c55e", width=2)

        for line in video_result.get("exit_lines", []) or []:
            p1, p2 = line.get("line", [None, None])
            if p1 and p2:
                draw.line([(int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))], fill="#ef4444", width=2)

        # Occupancy HUD
        occupancy = int(video_result.get("occupancy", len(tracks) if tracks else len(detections)) or 0)
        txt = f"Occupancy: {occupancy}"
        if font is not None:
            text_w, text_h = draw.textbbox((0, 0), txt, font=font)[2:]
            draw.rectangle([8, 8, 12 + text_w, 12 + text_h], fill="#0f172a")
            draw.text((10, 10), txt, fill="white", font=font)

    st.image(pil, width="stretch")

