# TODO_LIVE_MONITORING (next implementation)

## Status: initial read

### Step 1 — Analyze explainability + alert data contracts
- [x] Reviewed `ui/pages/live_monitoring.py` and current panels.
- [x] Reviewed existing `welfare_reasoning/types.py` and `welfare_reasoning/inference.py` for explainability fields.
- [ ] Identify exact keys in current inference payload for: triggered rules, evidence, feature/modality contribution, attention weights, confidence, contradictions, alternative explanations, recommended human action.


### Step 2 — Implement ONLY Explainability Panel UI
- [x] Create `ui/components_live/explainability_panel.py`.

- [x] Add `render_explainability_panel(data)` and wire it into `ui/pages/live_monitoring.py`.

- [ ] Ensure panel renders all required sections (even if data missing) using safe fallbacks.

### Step 3 — Implement ONLY Alert Center UI + alert generation
- [x] Create `ui/components_live/alert_center_panel.py`.

- [x] Implement alert generation for:

  - High welfare risk
  - Crowding
  - Distress
  - Camera failure
  - Audio failure
  - Sensor failure
  - Missing modality
  - Low confidence
- [x] Store alert history in `alerts.json` (append-only with timestamp + severity).

- [x] Render history with severity filter (Info/Warning/Critical) and “Store alert history” persists across runs.
- [x] Wire into `ui/pages/live_monitoring.py`.


### Step 4 — Keep inference logic unchanged
- [x] Confirm no modifications to `_run_inference` and inference calls.
- [ ] Ensure only UI additions + history storage are implemented.

