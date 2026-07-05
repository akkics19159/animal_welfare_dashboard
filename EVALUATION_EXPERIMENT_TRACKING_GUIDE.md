# Experiment Tracking Guide

Use ExperimentTracker to log benchmarks and configuration parameters.

Example:

```python
from evaluation.experiments.tracker import ExperimentTracker

tracker = ExperimentTracker("evaluation/experiments/history.json")
tracker.log_experiment("vision-baseline", {"accuracy": 0.91}, {"model": "yolo"})
```
