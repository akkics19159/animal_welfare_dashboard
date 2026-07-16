# TODO (Animal Welfare Monitoring Platform)

## Completed
- (none yet)

## In Progress
- Implement missing Streamlit pages with REST API integration:
  - ✅ Alerts page (fetch `/api/alerts`, severity filters, acknowledge via `/api/alerts/acknowledge`)
  - ✅ History page (fetch `/api/history`, date filtering, charts + KPIs)
  - ✅ User Management page (RBAC UI matching roles in `ui/sidebar.py`)

## Next

- Run unit/smoke tests:
  - `python -m unittest tests.test_ui_pages_smoke`
  - `python -m unittest tests.test_api_contracts`
  - `python -m unittest tests.test_api_exports tests.test_export_contracts`
- Update docs (Dashboard Architecture + API usage notes)

