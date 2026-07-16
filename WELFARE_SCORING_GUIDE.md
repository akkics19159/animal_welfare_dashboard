# Welfare Scoring Guide

## Objective
Estimate probability of genuine welfare compromise using multimodal evidence, not single-modality heuristics.

## Inputs
- Vision indicators (agitation, occupancy, motion, behavior)
- Audio indicators (distress/non-distress probabilities, event type)
- Sensor anomalies (temperature, humidity, heart rate)
- Tracking and temporal descriptors

## Scoring Stages
1. Base modality scoring (legacy-compatible)
2. Confidence-weighted fusion
3. Bayesian-inspired modality accumulation
4. Rule/context reasoning adjustment
5. False-positive suppression for normal behaviours
6. Agreement/conflict adjustment

## Output Fields
- `welfare_score`
- `risk_level`
- `severity`
- `urgency`
- `confidence`
- `prediction_uncertainty`
- `agreement_score`
- `evidence_summary`
- `suppressed_evidence`
- `reasoning_trace`
- `temporal_welfare_state`
