# Species Recognition Architecture

## Scope
This architecture updates only species recognition while preserving all existing interfaces in acquisition, tracking, behaviour, fusion, reasoning, dashboard, analytics, and training.

## Data Flow
Detection -> Tracking -> Species Recognition -> Temporal Verification -> Confidence Calibration -> Final Species

## Stage Details
### Stage 1: Detection
YOLO is limited to:
- bounding boxes
- detection confidence
- object localization
- initial detector label

YOLO does not decide final species.

### Stage 2: Species Recognition
Input:
- tracked crop per active track

Classifier output:
- classifier top-5 probabilities
- feature embedding
- embedding distance

### Stage 3: Temporal Verification
For each track id, a fixed-size history of 30 predictions is maintained.
Recent predictions are weighted higher.
Species changes are gated by a configurable margin to prevent frame-to-frame oscillation.

### Stage 4: Confidence Calibration
Final confidence combines:
- detector confidence
- classifier confidence
- embedding similarity
- temporal confidence
- detector/classifier agreement score

### Stage 5: Contract Emission
Final species metadata is added to track and detection contracts while preserving all existing keys.

## Failure Mode
If model loading or backend inference fails:
- Automatically use YOLO-label fallback backend.
- Continue inference without crashing.

## Integration Rules
- Behaviour recognition consumes `final_species`.
- Fusion consumes `final_species`.
- Reasoning receives species derived from `final_species`.
- Dashboard presents detector vs classifier vs final species together.
