from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math


def _distance(p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])


@dataclass
class TrajectoryAnalytics:
    distance_travelled: float
    instant_speed: float
    avg_speed: float
    stationary_duration: float
    dwell_time: float
    smoothness: float
    movement_density: float
    movement_direction_deg: float


class TrajectoryAnalyticsComputer:
    """Compute trajectory analytics per track id."""

    def __init__(
        self,
        stationary_speed_threshold: float = 1.0,
        stationary_time_threshold_sec: float = 1.5,
        smoothness_window: int = 5,
    ):
        self.stationary_speed_threshold = float(stationary_speed_threshold)
        self.stationary_time_threshold_sec = float(stationary_time_threshold_sec)
        self.smoothness_window = int(smoothness_window)

    def compute(self, trajectory: List[Tuple[float, float, float]], dwell_time: float = 0.0) -> TrajectoryAnalytics:
        # trajectory elements: (t, cx, cy)
        if not trajectory or len(trajectory) < 2:
            return TrajectoryAnalytics(0.0, 0.0, 0.0, 0.0, dwell_time, 0.0, 0.0, 0.0)

        # distance
        distance = 0.0
        total_speed_samples = []
        stationary_duration = 0.0

        for i in range(1, len(trajectory)):
            t0, x0, y0 = trajectory[i - 1]
            t1, x1, y1 = trajectory[i]
            dt = max(1e-6, t1 - t0)
            d = _distance((x0, y0), (x1, y1))
            distance += d
            speed = d / dt
            total_speed_samples.append(speed)

            if speed <= self.stationary_speed_threshold:
                stationary_duration += dt

        instant_speed = float(total_speed_samples[-1]) if total_speed_samples else 0.0
        avg_speed = float(sum(total_speed_samples) / max(1, len(total_speed_samples)))

        # Smoothness: 1 - normalized jerkiness using angle changes
        smoothness = self._compute_smoothness(trajectory)

        # Movement density: travelled distance normalized by elapsed time.
        total_time = max(1e-6, trajectory[-1][0] - trajectory[0][0])
        movement_density = float(distance / total_time)

        # Aggregate movement direction from first to last point.
        _, x0, y0 = trajectory[0]
        _, x1, y1 = trajectory[-1]
        movement_direction_deg = float((math.degrees(math.atan2(y1 - y0, x1 - x0)) + 360.0) % 360.0)

        # Clamp dwell-related stationary metrics
        if stationary_duration < self.stationary_time_threshold_sec:
            stationary_duration = 0.0

        return TrajectoryAnalytics(
            distance_travelled=float(distance),
            instant_speed=float(instant_speed),
            avg_speed=float(avg_speed),
            stationary_duration=float(stationary_duration),
            dwell_time=float(dwell_time),
            smoothness=float(smoothness),
            movement_density=movement_density,
            movement_direction_deg=movement_direction_deg,
        )

    def _compute_smoothness(self, trajectory: List[Tuple[float, float, float]]) -> float:
        if len(trajectory) < 3:
            return 0.0
        # compute direction changes
        angles = []
        for i in range(1, len(trajectory)):
            t0, x0, y0 = trajectory[i - 1]
            t1, x1, y1 = trajectory[i]
            dx, dy = x1 - x0, y1 - y0
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                continue
            ang = math.degrees(math.atan2(dy, dx))
            angles.append(ang)
        if len(angles) < 2:
            return 0.0

        # use last window
        window = angles[-self.smoothness_window :]
        deltas = []
        for i in range(1, len(window)):
            d = abs(window[i] - window[i - 1])
            # wrap-around normalization
            d = min(d, 360.0 - d)
            deltas.append(d)

        if not deltas:
            return 0.0
        mean_delta = sum(deltas) / len(deltas)
        # smaller delta => smoother => closer to 1
        smoothness = max(0.0, min(1.0, 1.0 - (mean_delta / 180.0)))
        return smoothness

