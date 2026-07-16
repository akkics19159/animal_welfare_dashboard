# UI Architecture

This folder contains the modular Streamlit dashboard implementation.

- `ui/theme.py`: theme + global CSS
- `ui/state.py`: session state initialization + shared state accessors
- `ui/api_client.py`: REST API wrapper used by UI
- `ui/components.py`: shared UI components (KPI cards, alerts, charts)
- `ui/sidebar.py`: navigation sidebar
- `ui/pages/*.py`: page implementations
- `dashboard.py`: thin router wiring sidebar -> page


