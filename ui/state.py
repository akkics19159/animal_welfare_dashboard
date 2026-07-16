from __future__ import annotations

import datetime as dt
import streamlit as st
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class DashboardConfig:
    video_weight: float = 0.4
    audio_weight: float = 0.4
    sensor_weight: float = 0.2
    ontology_strength: float = 0.6
    use_simulated_sensors: bool = True


DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_GLOBAL_DATE_DAYS = 30


def ensure_session_defaults() -> None:
    """Initialize session state keys used across pages."""
    if "last_analysis" not in st.session_state:
        st.session_state.last_analysis = None
    if "role" not in st.session_state:
        st.session_state.role = "Administrator"
    if "config" not in st.session_state:
        st.session_state.config = DashboardConfig()
    if "global_start_date" not in st.session_state:
        st.session_state.global_start_date = dt.date.today() - dt.timedelta(days=DEFAULT_GLOBAL_DATE_DAYS)
    if "global_end_date" not in st.session_state:
        st.session_state.global_end_date = dt.date.today()
    if "global_auto_refresh_enabled" not in st.session_state:
        st.session_state.global_auto_refresh_enabled = False
    if "global_auto_refresh_seconds" not in st.session_state:
        st.session_state.global_auto_refresh_seconds = 15
    if "run_counter" not in st.session_state:
        st.session_state.run_counter = 0
    if "current_run_token" not in st.session_state:
        st.session_state.current_run_token = ""


def get_config() -> DashboardConfig:
    ensure_session_defaults()
    return st.session_state.config


def set_config(cfg: DashboardConfig) -> None:
    st.session_state.config = cfg


def get_api_url() -> str:
    # Can be extended later for env var overrides.
    return DEFAULT_API_URL


def get_global_date_range() -> Tuple[dt.date, dt.date]:
    ensure_session_defaults()
    start = st.session_state.global_start_date
    end = st.session_state.global_end_date
    return start, end


def mark_new_run() -> str:
    ensure_session_defaults()
    st.session_state.run_counter = int(st.session_state.run_counter) + 1
    token = f"run-{st.session_state.run_counter}-{dt.datetime.now().timestamp()}"
    st.session_state.current_run_token = token
    return token


def get_current_run_token() -> str:
    ensure_session_defaults()
    return str(st.session_state.current_run_token or "")


def _coerce_datetime(value: Any) -> Optional[dt.datetime]:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time.min)
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = dt.datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            return parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return parsed
    except Exception:
        return None


def _read_field(record: Dict[str, Any], field_path: str) -> Any:
    parts = field_path.split(".")
    value: Any = record
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value.get(part)
    return value


def filter_records_by_date_range(
    records: List[Dict[str, Any]],
    start: dt.date,
    end: dt.date,
    *,
    date_fields: Optional[Iterable[str]] = None,
    include_if_missing: bool = True,
) -> List[Dict[str, Any]]:
    if not records:
        return records

    fields = list(date_fields or ("timestamp", "created_at", "updated_at", "metadata.created_at", "config.created_at"))
    start_dt = dt.datetime.combine(start, dt.time.min)
    end_dt = dt.datetime.combine(end, dt.time.max)
    filtered: List[Dict[str, Any]] = []

    for row in records:
        row_dt: Optional[dt.datetime] = None
        for field in fields:
            row_dt = _coerce_datetime(_read_field(row, field))
            if row_dt is not None:
                break

        if row_dt is None:
            if include_if_missing:
                filtered.append(row)
            continue

        if start_dt <= row_dt <= end_dt:
            filtered.append(row)

    return filtered
