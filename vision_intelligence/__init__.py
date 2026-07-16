"""Vision Intelligence subsystem.

This package modernizes input sourcing, tracking, counting, and trajectory analytics
while keeping backward compatibility with the legacy vision output contract.
"""

from .vision_pipeline import VisionPipeline

__all__ = ["VisionPipeline"]

