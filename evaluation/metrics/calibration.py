from __future__ import annotations

from typing import Sequence


def compute_brier_score(scores: Sequence[float], targets: Sequence[int]) -> float:
    scores = [float(score) for score in scores]
    targets = [float(target) for target in targets]
    if len(scores) != len(targets):
        raise ValueError("scores and targets must have the same length")
    if not scores:
        return 0.0
    return sum((score - target) ** 2 for score, target in zip(scores, targets)) / len(scores)


def compute_calibration_metrics(scores: Sequence[float], targets: Sequence[int]) -> dict:
    brier_score = compute_brier_score(scores, targets)
    mean_score = sum(scores) / len(scores) if scores else 0.0
    mean_target = sum(targets) / len(targets) if targets else 0.0
    ece = abs(mean_score - mean_target)
    return {"brier_score": brier_score, "ece": ece}
