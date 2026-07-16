from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


Point = Tuple[float, float]
Line = Tuple[Point, Point]
Polygon = List[Point]


def _point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray-casting point in polygon test."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-6) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _line_side(line: Line, point: Point) -> float:
    (x1, y1), (x2, y2) = line
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


@dataclass
class CountingStats:
    current_occupancy: int
    total_unique_individuals: int
    species_wise_count: Dict[str, int]
    region_wise_count: Dict[str, int]
    entry_count: int
    exit_count: int
    maximum_occupancy: int
    average_occupancy: float
    region_dwell_time: Dict[str, float]
    region_movement_stats: Dict[str, Dict[str, float]]


class IntelligentCounter:
    """Compute persistent counting stats from track states.

    This implementation is region-agnostic by default; it provides region-wise
    counts only if regions are configured.
    """

    def __init__(
        self,
        regions: Optional[List[Tuple[str, Polygon]]] = None,
        entry_lines: Optional[List[Tuple[str, Line]]] = None,
        exit_lines: Optional[List[Tuple[str, Line]]] = None,
        occupancy_history_window: int = 300,
    ):
        # regions: list of (name, polygon[(x,y), ...])
        self.regions = regions or []
        self.entry_lines = entry_lines or []
        self.exit_lines = exit_lines or []
        self.occupancy_history_window = int(occupancy_history_window)
        self._occupancy_history: List[int] = []

        self._max_occupancy = 0
        self._unique_ids = set()
        self._entry_count = 0
        self._exit_count = 0

        self._track_line_side: Dict[int, Dict[str, float]] = {}

        # track previous presence to detect entry/exit
        self._prev_visible_ids = set()

    def update(self, tracks: Dict[int, any]) -> CountingStats:
        visible_ids = set([tid for tid, tr in tracks.items() if getattr(tr, "lost_frames", 0) == 0])

        # detect entries/exits
        entered = visible_ids - self._prev_visible_ids
        exited = self._prev_visible_ids - visible_ids
        if not self.entry_lines and entered:
            self._entry_count += len(entered)
        if not self.exit_lines and exited:
            self._exit_count += len(exited)

        self._prev_visible_ids = set(visible_ids)

        # update unique ids
        self._unique_ids |= visible_ids

        # occupancy
        current_occ = len(visible_ids)
        self._occupancy_history.append(current_occ)
        if len(self._occupancy_history) > self.occupancy_history_window:
            self._occupancy_history.pop(0)

        self._max_occupancy = max(self._max_occupancy, current_occ)
        avg_occ = float(sum(self._occupancy_history) / max(1, len(self._occupancy_history)))

        species_wise: Dict[str, int] = {}
        region_wise: Dict[str, int] = {r[0]: 0 for r in self.regions}
        region_dwell_time: Dict[str, float] = {r[0]: 0.0 for r in self.regions}
        region_movement_stats: Dict[str, Dict[str, float]] = {
            r[0]: {"avg_velocity": 0.0, "avg_acceleration": 0.0, "movement_density": 0.0}
            for r in self.regions
        }
        region_counts_for_motion: Dict[str, int] = {r[0]: 0 for r in self.regions}

        for tid in visible_ids:
            tr = tracks[tid]
            cls = getattr(tr, "class_name", "animal")
            species_wise[str(cls)] = species_wise.get(str(cls), 0) + 1

            bbox = getattr(tr, "bbox_xyxy", None)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            center = (cx, cy)

            # Entry/exit via signed line-side crossing.
            side_state = self._track_line_side.setdefault(int(tid), {})
            for line_name, line in self.entry_lines:
                prev_side = side_state.get(f"entry::{line_name}")
                side = _line_side(line, center)
                if prev_side is not None and (prev_side < 0 <= side):
                    self._entry_count += 1
                side_state[f"entry::{line_name}"] = side

            for line_name, line in self.exit_lines:
                prev_side = side_state.get(f"exit::{line_name}")
                side = _line_side(line, center)
                if prev_side is not None and (prev_side > 0 >= side):
                    self._exit_count += 1
                side_state[f"exit::{line_name}"] = side

            for name, polygon in self.regions:
                if _point_in_polygon(center, polygon):
                    region_wise[name] = region_wise.get(name, 0) + 1
                    region_dwell_time[name] += float(getattr(tr, "dwell_time", 0.0) or 0.0)

                    vel = getattr(tr, "velocity_xy", (0.0, 0.0))
                    acc = getattr(tr, "acceleration_xy", (0.0, 0.0))
                    density = float(getattr(tr, "movement_density", 0.0) or 0.0)
                    region_movement_stats[name]["avg_velocity"] += float((vel[0] ** 2 + vel[1] ** 2) ** 0.5)
                    region_movement_stats[name]["avg_acceleration"] += float((acc[0] ** 2 + acc[1] ** 2) ** 0.5)
                    region_movement_stats[name]["movement_density"] += density
                    region_counts_for_motion[name] += 1

        for name in region_movement_stats:
            denom = max(1, region_counts_for_motion[name])
            region_movement_stats[name]["avg_velocity"] /= denom
            region_movement_stats[name]["avg_acceleration"] /= denom
            region_movement_stats[name]["movement_density"] /= denom

        return CountingStats(
            current_occupancy=int(current_occ),
            total_unique_individuals=int(len(self._unique_ids)),
            species_wise_count=species_wise,
            region_wise_count=region_wise,
            entry_count=int(self._entry_count),
            exit_count=int(self._exit_count),
            maximum_occupancy=int(self._max_occupancy),
            average_occupancy=float(avg_occ),
            region_dwell_time=region_dwell_time,
            region_movement_stats=region_movement_stats,
        )

