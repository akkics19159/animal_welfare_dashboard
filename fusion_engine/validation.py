from __future__ import annotations

import math
from typing import Optional

from .types import FeatureVector, ValidationReport


class FeatureValidator:
    """Validates feature vectors and reports issues."""

    def __init__(self, supported_versions: Optional[set[str]] = None):
        self.supported_versions = supported_versions or {"1.0", "1.1"}

    def validate(self, feature: Optional[FeatureVector]) -> ValidationReport:
        if feature is None:
            return ValidationReport(valid=False, errors=["feature vector missing"], modality="unknown")

        errors = []
        warnings = []

        if not feature.values:
            errors.append("no feature values")

        for key, value in feature.values.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                errors.append(f"invalid value for {key}")
                continue
            if not math.isfinite(float(value)):
                errors.append(f"non-finite value for {key}")
            if abs(float(value)) > 1e6:
                errors.append(f"out-of-range value for {key}")

        if feature.confidence < 0 or feature.confidence > 1:
            errors.append("confidence out of range")
        if feature.quality_score < 0 or feature.quality_score > 1:
            errors.append("quality score out of range")
        if feature.latency_ms < 0:
            errors.append("latency must be non-negative")
        if feature.version not in self.supported_versions:
            warnings.append(f"unsupported version {feature.version}")

        return ValidationReport(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            modality=feature.modality,
            version_check=feature.version in self.supported_versions,
        )
