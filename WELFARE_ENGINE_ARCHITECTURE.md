# Welfare Engine Architecture

## Scope
This document describes the upgraded in-place Welfare Assessment subsystem across fusion, welfare reasoning, and explainability layers.

## Subsystems
1. Fusion Engine (`fusion_engine/*`)
   - Confidence-weighted feature fusion
   - Bayesian-inspired evidence accumulation
   - Modality agreement scoring
   - Temporal context and timeline extraction

2. Welfare Reasoning (`welfare_reasoning/*`)
   - Rule-based explainable reasoning
   - Species-aware thresholds from configuration
   - Context and temporal state reasoning
   - Contradiction handling and confidence penalties

3. Decision Wrapper (`multimodal_engine.py`)
   - Integrates fusion + reasoning outputs
   - Applies configurable false-positive suppression
   - Emits backward-compatible + additive welfare contract fields

4. Explainability UI (`ui/components_live/*`)
   - Evidence summary, contradictions, suppressed evidence, reasoning trace, temporal state

## Compatibility
- Existing APIs preserved.
- Existing fields preserved.
- New outputs are additive only.
