# Welfare Reasoning Architecture

```mermaid
flowchart TD
    A[Unified Feature Representation] --> B[Knowledge Base]
    B --> C[Species Profiles]
    C --> D[Behaviour Library]
    D --> E[Context Engine]
    E --> F[Rule Engine]
    F --> G[Constraint Validation]
    G --> H[Confidence-aware Inference]
    H --> I[Contradiction Detection]
    I --> J[Evidence Aggregation]
    J --> K[Explainability Interface]
```

## Components

- KnowledgeBase: configuration-driven species, behaviour, environment, and rule definitions.
- ContextEngine: adjusts reasoning with time, crowding, environment, occupancy, and quality signals.
- RuleEngine: evaluates nested AND/OR/NOT/threshold/weighted rules from configuration.
- ConstraintValidator: flags conflicting states such as sleeping + running.
- ConfidenceAwareInferenceEngine: combines evidence, confidence, uncertainty, and contradiction status.
- WelfareReasoningEngine: orchestrates the pipeline and exposes a stable plugin interface for future backends.
