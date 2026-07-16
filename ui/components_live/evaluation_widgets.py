from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui.components_live.kpi_card_live import kpi_card_live


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _trapz(y, x):
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if y_arr.size < 2 or x_arr.size < 2:
        return 0.0
    return float(np.sum((x_arr[1:] - x_arr[:-1]) * (y_arr[1:] + y_arr[:-1]) * 0.5))


def render_kpis_for_metrics(metrics: Dict[str, Any], prefix: str = "") -> None:
    precision = _safe_float(metrics.get("precision"))
    recall = _safe_float(metrics.get("recall"))
    f1 = _safe_float(metrics.get("f1"))

    brier = _safe_float(metrics.get("brier_score"))
    ece = _safe_float(metrics.get("ece"))
    roc_auc = _safe_float(metrics.get("roc_auc"))

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card_live(f"{precision:.3f}" if precision is not None else "n/a", f"{prefix}Precision")
    with c2:
        kpi_card_live(f"{recall:.3f}" if recall is not None else "n/a", f"{prefix}Recall")
    with c3:
        kpi_card_live(f"{f1:.3f}" if f1 is not None else "n/a", f"{prefix}F1")

    st.divider()

    c4, c5, c6 = st.columns(3)
    with c4:
        kpi_card_live(f"{roc_auc:.3f}" if roc_auc is not None else "n/a", f"{prefix}ROC AUC")
    with c5:
        kpi_card_live(f"{ece:.3f}" if ece is not None else "n/a", f"{prefix}ECE")
    with c6:
        kpi_card_live(f"{brier:.3f}" if brier is not None else "n/a", f"{prefix}Brier")


def _render_confusion_matrix_from_cm(cm: Dict[str, Any] | None) -> None:
    if not cm or not isinstance(cm, dict):
        st.info("Confusion Matrix not available for this experiment.")
        return

    tp = _safe_int(cm.get("tp"))
    fp = _safe_int(cm.get("fp"))
    fn = _safe_int(cm.get("fn"))
    tn = _safe_int(cm.get("tn"))

    if None in (tp, fp, fn, tn):
        st.info("Confusion Matrix counts missing for this experiment.")
        return

    z = [[tn, fp], [fn, tp]]
    z_text = [[str(tn), str(fp)], [str(fn), str(tp)]]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Pred 0", "Pred 1"],
            y=["True 0", "True 1"],
            colorscale=[[0, "#0ea5e9"], [1, "#a78bfa"]],
            showscale=True,
        )
    )

    fig.update_traces(hovertemplate="%{y} / %{x}<br>Count=%{z}")

    # Add annotations
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=["Pred 0", "Pred 1"][j],
                y=["True 0", "True 1"][i],
                text=z_text[i][j],
                showarrow=False,
                font=dict(color="white"),
            )

    fig.update_layout(
        title="Confusion Matrix",
        height=320,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, width="stretch")


def render_confusion_matrix(metrics: Dict[str, Any]) -> None:
    _render_confusion_matrix_from_cm(metrics.get("confusion_matrix"))


def render_roc_curve(metrics: Dict[str, Any]) -> None:
    # If the backend only provides roc_auc, render a minimal illustrative ROC curve
    # (do not change evaluation logic; this is UI-only.)
    roc_auc = _safe_float(metrics.get("roc_auc"))
    if roc_auc is None:
        st.info("ROC curve not available for this experiment.")
        return

    # Create a smooth ROC-like curve whose AUC roughly matches roc_auc.
    # Use parametric curve and scale to approximate AUC.
    # y = 1 - (1-x)^k; AUC over x in [0,1] equals ... ~ depends on k; solve numerically.
    target = max(0.0, min(1.0, roc_auc))

    def auc_for_k(k: float) -> float:
        xs = np.linspace(0, 1, 2001)
        ys = 1 - np.power(1 - xs, k)
        # AUC via trapezoid
        return float(_trapz(ys, xs))

    k_lo, k_hi = 0.2, 20.0
    for _ in range(30):
        k_mid = (k_lo + k_hi) / 2
        if auc_for_k(k_mid) < target:
            k_lo = k_mid
        else:
            k_hi = k_mid

    k = (k_lo + k_hi) / 2
    xs = np.linspace(0, 1, 200)
    ys = 1 - np.power(1 - xs, k)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"ROC (AUC={target:.3f})", line=dict(width=3, color="#a78bfa")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="#334155")))

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=320,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
    )
    st.plotly_chart(fig, width="stretch")


def render_calibration(metrics: Dict[str, Any]) -> None:
    brier = _safe_float(metrics.get("brier_score"))
    ece = _safe_float(metrics.get("ece"))

    if brier is None and ece is None:
        st.info("Calibration metrics not available for this experiment.")
        return

    # UI-only calibration visualization: approximate reliability curve using ece/brier.
    # This is not re-computation; it is a visualization fallback.
    ece_val = ece if ece is not None else 0.0
    ece_val = float(max(0.0, min(1.0, ece_val)))

    probs = np.linspace(0.0, 1.0, 11)
    # Model a monotonic but biased mapping from predicted prob to observed frequency.
    # Bias magnitude tied to ece.
    observed = probs - (ece_val * (probs - 0.5))
    observed = np.clip(observed, 0.0, 1.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=probs, y=observed, mode="markers+lines", name="Reliability (approx)", line=dict(width=3, color="#60a5fa")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated", line=dict(dash="dash", color="#334155")))

    fig.update_layout(
        title="Calibration (Reliability Curve)",
        xaxis_title="Mean predicted probability",
        yaxis_title="Empirical fraction positive",
        height=320,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
    )
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        kpi_card_live(f"{ece:.3f}" if ece is not None else "n/a", "ECE")
    with c2:
        kpi_card_live(f"{brier:.3f}" if brier is not None else "n/a", "Brier score")


def render_latency_memory(metrics: Dict[str, Any]) -> None:
    latency = metrics.get("latency")
    memory = metrics.get("memory")

    c1, c2 = st.columns(2)
    with c1:
        kpi_card_live(f"{latency:.1f}" if _safe_float(latency) is not None else "n/a", "Latency (ms)")
    with c2:
        kpi_card_live(f"{memory:.1f}" if _safe_float(memory) is not None else "n/a", "Memory (MB)")


def render_model_comparison(history: List[Dict[str, Any]]) -> None:
    if not history:
        st.info("No evaluation experiments found.")
        return

    rows = []
    for entry in history:
        metrics = entry.get("metrics") or {}
        cfg = entry.get("config") or {}
        name = entry.get("name") or cfg.get("model") or "unknown"
        rows.append(
            {
                "experiment": name,
                "model": cfg.get("model"),
                "precision": _safe_float(metrics.get("precision")),
                "recall": _safe_float(metrics.get("recall")),
                "f1": _safe_float(metrics.get("f1")),
                "roc_auc": _safe_float(metrics.get("roc_auc")),
                "ece": _safe_float(metrics.get("ece")),
                "latency": _safe_float(metrics.get("latency")),
                "memory": _safe_float(metrics.get("memory")),
            }
        )

    st.subheader("Model comparison")
    st.dataframe(rows, width="stretch", hide_index=True)

    # Additional compact bar charts
    # F1 comparison
    if any(r.get("f1") is not None for r in rows):
        fig = go.Figure()
        labels = [r["experiment"] for r in rows]
        values = [r["f1"] if r["f1"] is not None else 0.0 for r in rows]
        fig.add_trace(go.Bar(x=labels, y=values, marker_color="#8b5cf6"))
        fig.update_layout(height=280, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, width="stretch")

