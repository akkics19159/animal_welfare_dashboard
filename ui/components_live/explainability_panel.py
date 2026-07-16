from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _render_kv_table(rows: List[Dict[str, Any]], col_keys: List[str]) -> None:
    if not rows:
        st.info("No data available.")
        return

    table_rows = []
    for r in rows:
        table_rows.append({k: r.get(k) for k in col_keys})

    st.dataframe(table_rows, width="stretch", hide_index=True)


def _as_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    return {}


def render_explainability_panel(data: Dict[str, Any]) -> None:
    """Explainability panel.

    Expected (best-effort) fields in data['explanations'] or data['explainability']:
      - triggered_rules: List[str]
      - evidence: List[{name,value,weight,source,explanation}] (or any similar dict)
      - feature_contribution: Dict/ List
      - modality_contribution: Dict/ List
      - attention_weights: Dict/ List
      - confidence: float
      - contradictions: List[str]
      - alternative_explanations: List[str]
      - recommended_human_action: bool/str
    """

    st.subheader("Explainability")

    evidence_summary = data.get("evidence_summary") or {}
    suppressed_evidence = data.get("suppressed_evidence") or []
    reasoning_trace = data.get("reasoning_trace") or []
    temporal_state = data.get("temporal_welfare_state") or {}

    explainability = data.get("explainability") or {}

    explanations = data.get("explanations")
    if explanations is None:
        # Some backends may expose a single object.
        explanations = [explainability] if explainability else []

    explanations_list = _as_list(explanations)

    # Legacy/newer backends may return explanations as a list of strings.
    # Use explainability object as primary metadata in that case.
    dict_candidates = [e for e in explanations_list if isinstance(e, dict)]
    text_candidates = [str(e) for e in explanations_list if not isinstance(e, dict)]

    if not dict_candidates and not explainability and not text_candidates:
        st.info("No explainability metadata returned for this run.")
        return

    # Prefer dict explanation payloads; otherwise fall back to explainability object.
    primary = dict_candidates[0] if dict_candidates else _as_dict(explainability)

    triggered_rules = primary.get("triggered_rules") or []
    evidence = primary.get("evidence") or []
    feature_contribution = primary.get("feature_contribution") or primary.get("features")
    modality_contribution = primary.get("modality_contribution") or primary.get("modalities")
    attention_weights = primary.get("attention_weights") or primary.get("attention")
    confidence = primary.get("confidence")
    contradictions = primary.get("contradictions") or []
    alternative_explanations = primary.get("alternative_explanations") or primary.get(
        "alternatives"
    )
    recommended_human_action = primary.get("recommended_human_action")
    if recommended_human_action is None:
        recommended_human_action = primary.get("recommended_human_review")

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("**Confidence**")
        if confidence is None:
            st.write("n/a")
        else:
            st.write(f"{_safe_float(confidence):.3f}")

    with colB:
        st.markdown("**Recommended human action**")
        if recommended_human_action is None:
            st.write("n/a")
        elif isinstance(recommended_human_action, bool):
            st.write("✅ Review" if recommended_human_action else "❌ No action")
        else:
            st.write(str(recommended_human_action))

    st.divider()

    if evidence_summary:
        st.markdown("### Evidence summary")
        for bucket in ("primary_evidence", "supporting_evidence", "contradictory_evidence"):
            st.markdown(f"**{bucket.replace('_', ' ').title()}**")
            rows = evidence_summary.get(bucket) or []
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
            else:
                st.caption("None")
        conflicts = evidence_summary.get("conflicts") or []
        if conflicts:
            st.markdown("**Conflicts**")
            st.write([f"• {c}" for c in conflicts])

    if suppressed_evidence:
        st.divider()
        st.markdown("### Suppressed evidence")
        st.dataframe(suppressed_evidence, width="stretch", hide_index=True)

    if reasoning_trace:
        st.divider()
        st.markdown("### Reasoning trace")
        st.write([f"• {r}" for r in reasoning_trace])

    if text_candidates:
        st.divider()
        st.markdown("### Explanation notes")
        st.write([f"• {r}" for r in text_candidates])

    if temporal_state:
        st.divider()
        st.markdown("### Temporal welfare state")
        st.json(temporal_state)

    st.divider()

    # Triggered rules
    st.markdown("### Triggered rules")
    if triggered_rules:
        st.write(
            [f"• {r}" for r in triggered_rules] if isinstance(triggered_rules, list) else str(triggered_rules)
        )
    else:
        st.info("No triggered rules provided.")

    # Evidence
    st.markdown("### Evidence")
    evidence_list = _as_list(evidence)
    if evidence_list:
        # Normalize evidence items if possible.
        rows = []
        for e in evidence_list:
            if isinstance(e, dict):
                rows.append(
                    {
                        "name": e.get("name") or e.get("label") or "fact",
                        "value": e.get("value"),
                        "weight": e.get("weight"),
                        "source": e.get("source"),
                        "explanation": e.get("explanation"),
                    }
                )
            else:
                rows.append({"name": str(e), "value": None, "weight": None, "source": None, "explanation": None})

        # If many rows, keep it compact.
        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No evidence items provided.")

    st.divider()

    # Feature contribution
    st.markdown("### Feature contribution")
    if feature_contribution is None:
        st.info("n/a")
    else:
        st.json(feature_contribution)

    # Modality contribution
    st.markdown("### Modality contribution")
    if modality_contribution is None:
        st.info("n/a")
    else:
        st.json(modality_contribution)

    st.divider()

    # Attention weights
    st.markdown("### Attention weights")
    if attention_weights is None:
        st.info("n/a")
    else:
        st.json(attention_weights)

    st.divider()

    # Contradictions + alternatives
    st.markdown("### Contradictions")
    if contradictions:
        st.write([f"• {c}" for c in contradictions])
    else:
        st.write("None detected.")

    st.markdown("### Alternative explanations")
    alt_list = _as_list(alternative_explanations)
    if alt_list:
        st.write([f"• {a}" for a in alt_list])
    else:
        st.info("No alternative explanations provided.")

    # Other explanation items
    if len(dict_candidates) > 1:
        st.divider()
        st.markdown("#### Additional explanation candidates")
        for i, extra in enumerate(dict_candidates[1:], start=2):
            try:
                summary = extra.get("summary") or extra.get("assessment_label") or extra.get("label")
            except Exception:
                summary = None
            st.caption(f"Candidate {i}: {summary or 'n/a'}")



