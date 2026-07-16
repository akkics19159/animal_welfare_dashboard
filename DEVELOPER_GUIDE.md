# Developer Guide

## Project layout
- Backend: `api_server.py`
- UI router + pages: `dashboard.py`, `ui/pages/*`
- UI shared state/components: `ui/state.py`, `ui/components.py`, `ui/components_live/*`

## Contract rules
- UI calls backend through `ui/api_client.py`.
- Page modules under `ui/pages/` must expose:
  - `render_page(api, backend_online, default_history_path, default_video_path)`
- Export endpoints must remain JSON-safe.

## How to add a new page
1. Create `ui/pages/<new_page>.py`.
2. Implement `render_page(...)`.
3. Add mapping in `dashboard.py` `_get_page_module()`.
4. Add role permissions in `ui/sidebar.py` `_role_permissions()`.

## How to add a new API endpoint
1. Implement in `api_server.py`.
2. Add an API client wrapper if needed (optional).
3. Add integration tests under `tests/test_*`.

## Testing strategy
- Unit tests already exist for fusion/evaluation/audio.
- Backend integration tests should use FastAPI `TestClient`.
- UI tests should focus on importability and render callable presence.

## AI logic freeze
Do not modify inference logic modules (e.g., `multimodal_engine.py`, `video_module.py`, `audio_module.py`, `sensors.py`, `ontology.py`) except for interface compatibility required by REST payload/response.


