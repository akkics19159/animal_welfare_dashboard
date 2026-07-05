# Migration Report

## Previous state

The repository used a lightweight ontology module with a single flat rule list. It produced a score by checking a few boolean flags and regularizing the final probability with a simple weighted blend.

## New state

The ontology layer was replaced with a hybrid welfare reasoning engine built around:

- configuration-driven knowledge
- structured context evaluation
- rule-based reasoning with nested logic
- contradiction validation
- confidence-aware inference
- evidence aggregation and explanation output

## Compatibility

The legacy interface remains available through the ontology module:

- evaluate_rules()
- ontology_penalty()

These entry points now delegate to the new engine while keeping the surrounding pipeline intact.
