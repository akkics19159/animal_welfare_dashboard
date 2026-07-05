# Evaluation Architecture

## Package layout

- evaluation/metrics: reusable metrics for classification, calibration, detection, and reasoning quality.
- evaluation/benchmarks: subsystem-specific benchmark runners for vision, audio, sensor, fusion, welfare reasoning, and end-to-end evaluation.
- evaluation/datasets: lightweight dataset container used to describe benchmark samples.
- evaluation/reports: markdown, CSV, and JSON report generation.
- evaluation/visualization: helpers for exporting plot-ready data.
- evaluation/experiments: experiment tracking and history persistence.

## Usage

1. Instantiate the relevant benchmark class.
2. Pass predictions and targets or metadata into the benchmark evaluate method.
3. Save generated reports and experiment history.
