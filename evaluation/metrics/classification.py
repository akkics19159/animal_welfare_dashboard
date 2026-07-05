from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def _ensure_sequence(values: Sequence[float] | Iterable[float]) -> List[float]:
    return [float(value) for value in values]


def compute_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[int, int, int, int]:
    y_true = _ensure_sequence(y_true)
    y_pred = _ensure_sequence(y_pred)
    tp = sum(1 for t, p in zip(y_true, y_pred) if int(t) == 1 and int(p) == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if int(t) == 0 and int(p) == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if int(t) == 0 and int(p) == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if int(t) == 1 and int(p) == 0)
    return tp, tn, fp, fn


def compute_binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[float, float, float]:
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_detection_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    precision, recall, f1 = compute_binary_metrics(y_true, y_pred)
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
