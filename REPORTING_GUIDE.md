# Reporting Guide

History and export/report payloads now include behaviour attributes where available.

## Added History Columns

- behaviour
- behaviour_probability
- behaviour_confidence
- behaviour_duration
- behaviour_transition
- behaviour_stability
- occupancy
- species
- risk_level

## Export Compatibility

Existing export endpoints remain unchanged and backward compatible.
New columns flow automatically into history CSV/JSON/Markdown exports when present.
