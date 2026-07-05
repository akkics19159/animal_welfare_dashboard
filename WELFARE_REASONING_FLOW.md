# Welfare Reasoning Flow

1. The pipeline receives structured evidence from the multimodal system.
2. The reasoning engine loads a configuration-driven knowledge base.
3. Species profiles and behaviour definitions provide domain priors.
4. The context engine evaluates time of day, crowding, occupancy, environment, and quality signals.
5. The rule engine evaluates nested rules with thresholds and weighted evidence.
6. The constraint validator flags impossible or contradictory states.
7. Confidence-aware inference combines evidence strength, reliability, and uncertainty.
8. Evidence aggregation produces explainable support for the final assessment.
9. The explainability interface returns the final score, confidence, contradictions, and recommended review actions.
