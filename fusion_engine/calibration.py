from __future__ import annotations

import math
from typing import Dict

from .types import FusionOutput


class ConfidenceCalibrator:
    """Produces calibrated confidence and reliability scores."""

    def calibrate(self, output: FusionOutput, reliability_scores: Dict[str, float]) -> FusionOutput:
        if output.confidence <= 0:
            output.confidence = 0.0
        output.calibration_score = min(1.0, max(0.0, output.confidence))
        output.prediction_reliability = output.calibration_score
        output.confidence_interval = {
            "lower": max(0.0, output.confidence - 0.1),
            "upper": min(1.0, output.confidence + 0.1),
        }
        output.reliability_scores = reliability_scores
        return output


class UncertaintyEstimator:
    """Estimates epistemic, aleatoric and overall uncertainty."""

    def estimate(self, output: FusionOutput, reliability_scores: Dict[str, float]) -> FusionOutput:
        if not reliability_scores:
            output.uncertainty = 0.5
        else:
            avg_reliability = sum(reliability_scores.values()) / len(reliability_scores)
            output.uncertainty = max(0.0, min(1.0, 1.0 - avg_reliability))

        output.unified_representation = output.unified_representation or {}
        output.unified_representation["uncertainty_breakdown"] = {
            "epistemic": output.uncertainty * 0.5,
            "aleatoric": output.uncertainty * 0.3,
            "overall": output.uncertainty,
        }
        return output
