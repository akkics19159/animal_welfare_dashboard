# TODO_API.md

## Completed
- [ ] Confirmed schemas for `/api/evaluation`, `/api/training`, `/api/dataset`, `/api/models`, `/api/alerts` from JSON sources.

## Remaining (in order)
1) Fix `/api/health` crash in `api_server.py` (move `import shutil` to top-level).
2) Implement structured logging in `api_server.py` (JSON structured events). Preserve current `/api/logs/{category}` contract.
3) Add export-only REST endpoints in `api_server.py`:
   - `/api/export/history`
   - `/api/export/analytics`
   - `/api/export/evaluation`
   - `/api/export/training`
   - `/api/export/dataset`
   - `/api/export/models`
   - `/api/export/alerts`
   - `/api/export/health`
   - `/api/export/charts/analytics`
   - Formats: csv|json|markdown (PDF graceful fallback if dependencies missing), charts return Plotly JSON.
4) Quick validation by running lightweight checks (import server / hit endpoints if a server is already used).

