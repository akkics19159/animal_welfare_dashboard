from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def _render_species_distribution(meta: Dict[str, Any]) -> None:
    dist = meta.get("species_distribution") or {}
    if not isinstance(dist, dict) or not dist:
        st.info("No species distribution found for this dataset version.")
        return

    # Stable order: by count desc
    items = sorted(dist.items(), key=lambda kv: float(kv[1]) if kv[1] is not None else 0, reverse=True)
    species, counts = zip(*items)

    st.subheader("Species distribution")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.bar_chart({"Samples": {s: c for s, c in zip(species, counts)}})
    with c2:
        st.markdown("**Top species**")
        topn = min(6, len(items))
        for i in range(topn):
            st.write(f"• {species[i]}: {counts[i]}")


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def render_page(api, backend_online: bool, default_history_path: str, default_video_path: str) -> None:
    st.markdown("<h1 class='main-title'>💽 Dataset Registry</h1>", unsafe_allow_html=True)

    if not backend_online:
        st.caption("Backend health check was inconclusive. Trying `/api/dataset` directly...")

    try:
        raw = api.get("/api/dataset")
    except Exception as e:
        st.warning("Dataset backend not reachable (API offline). Dataset registry requires `/api/dataset`.")
        st.error(f"Failed to load dataset registry: {e}")
        return

    # API returns list(db.values()) where values include dataset_id, version, metadata
    datasets: List[Dict[str, Any]] = raw if isinstance(raw, list) else []

    if not datasets:
        st.info("No datasets found in registry.")
        return

    # Build selector options
    # Prefer dataset_id; fallback to version
    options = []
    by_key: Dict[str, Dict[str, Any]] = {}
    for d in datasets:
        key = str(d.get("dataset_id") or d.get("version") or "unknown")
        options.append(key)
        by_key[key] = d

    selected_key = st.selectbox("Select dataset version", options)
    ds = by_key[selected_key]
    meta = ds.get("metadata") or {}

    version = ds.get("version") or selected_key
    dataset_id = ds.get("dataset_id") or selected_key

    # ---- Top KPIs ----
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    samples = meta.get("samples")
    validation_status = meta.get("validation_status")
    annotation_progress = meta.get("annotation_progress")

    with k1:
        st.metric("Dataset ID", str(dataset_id))
    with k2:
        st.metric("Validation", str(validation_status) if validation_status is not None else "N/A")
    with k3:
        st.metric("Sample count", str(samples) if samples is not None else "N/A")
    with k4:
        prog = _to_float(annotation_progress, default=0.0)
        st.metric("Annotation progress", f"{prog:.1f}%")

    st.divider()

    # ---- Validation + quality ----
    qc = meta.get("quality_score")
    created_at = meta.get("created_at")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Validation & quality")
        st.write(f"**Validation status:** {validation_status if validation_status is not None else 'N/A'}")
        st.write(f"**Quality score:** {qc if qc is not None else 'N/A'}")
        st.write(f"**Created at:** {created_at if created_at is not None else 'N/A'}")

    with c2:
        st.subheader("Annotation progress")
        prog = _to_float(annotation_progress, default=0.0)
        st.progress(min(max(prog / 100.0, 0.0), 1.0))
        st.caption("Progress is shown at the dataset-metadata level for the selected version.")

    # ---- Versions list (within dataset page) ----
    st.divider()
    st.subheader("All dataset versions")

    # Render as simple table (no pandas dependency here)
    rows = []
    for d in datasets:
        m = d.get("metadata") or {}
        rows.append(
            {
                "dataset_id": d.get("dataset_id"),
                "version": d.get("version"),
                "validation_status": m.get("validation_status"),
                "samples": m.get("samples"),
                "annotation_progress": m.get("annotation_progress"),
            }
        )

    # Streamlit can render list[dict] as dataframe-like table
    st.table(rows)

    # ---- Species distribution ----
    st.divider()
    _render_species_distribution(meta)

