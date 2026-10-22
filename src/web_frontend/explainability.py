"""
Explainability Page
-------------------
Restores the desktop command-center visual layout:
- EXPLAINABILITY header + SYSTEM ONLINE badge
- Live mode DevBadge
- Entity Selector combo
- 2 Metric Cards: ANOMALY SCORE and RISK SCORE
- FEATURE CONTRIBUTIONS: vibrant cyan/turquoise horizontal bar chart (#1de9b6)
- EVIDENCE: technical dark panel
- REASON: technical dark panel
- Threshold slider with cyber track and red handle
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.web_frontend.components import render_metric_card, render_dev_badge, render_header
from src.data.data_manager import get_explainability_data


def render_explainability() -> None:
    render_header("EXPLAINABILITY")
    render_dev_badge()
    from src.web_frontend.components import render_dataset_notice
    render_dataset_notice()

    explainability_data = get_explainability_data()
    explainability_ids = list(explainability_data.keys())

    if not explainability_ids:
        scol1, scol2 = st.columns(2)
        with scol1:
            render_metric_card("ANOMALY SCORE", "0.00")
        with scol2:
            render_metric_card("RISK SCORE", "0/100")
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.info("No explainability records loaded. Ingest an external Bitcoin transaction dataset to view feature contributions.")
        return

    # --- Entity Selector -----------------------------------------------------
    scol1, scol2 = st.columns([1, 4])
    with scol1:
        st.markdown(
            "<div style='color:#c0c0c0; font-family:\"JetBrains Mono\", Consolas, monospace; "
            "font-size:12px; font-weight:bold; padding-top:8px;'>Entity:</div>",
            unsafe_allow_html=True,
        )
    with scol2:
        entity_id = st.selectbox(
            "Choose an Entity ID to explain",
            options=explainability_ids,
            index=0,
            label_visibility="collapsed",
        )

    expl = explainability_data.get(entity_id)
    if expl is None:
        st.warning("Entity not found in live dataset.")
        return

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # --- Anomaly Score & Risk Score Cards ------------------------------------
    scol1, scol2 = st.columns(2)
    with scol1:
        render_metric_card("ANOMALY SCORE", f"{expl['anomaly_score']:.2f}")
    with scol2:
        render_metric_card("RISK SCORE", f"{expl['risk_score']}/100")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # --- Feature Contributions Bar Chart -------------------------------------
    st.markdown(
        "<div style='color:#b33939; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; font-weight:bold; letter-spacing:1px; margin-bottom:8px;'>"
        "FEATURE CONTRIBUTIONS"
        "</div>",
        unsafe_allow_html=True,
    )

    top_features = expl.get("top_features", [])
    if top_features:
        # Sort ascending so highest contribution is at the top
        sorted_features = sorted(top_features, key=lambda x: x.get("contribution", 0))
        y_labels = [f.get("feature", "") for f in sorted_features]
        x_values = [f.get("contribution", 0) for f in sorted_features]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=x_values,
            y=y_labels,
            orientation='h',
            marker=dict(
                color='#1de9b6',
                line=dict(color='#080404', width=1),
            ),
            text=[f"{v:.2f}" for v in x_values],
            textposition='auto',
            textfont=dict(family='JetBrains Mono, monospace', color='#080404', size=11),
            hovertemplate='%{y}: %{x:.4f}<extra></extra>',
        ))
        fig_bar.update_layout(
            plot_bgcolor='#080404',
            paper_bgcolor='#080404',
            font=dict(color='#c0c0c0', family='JetBrains Mono, monospace', size=11),
            xaxis=dict(
                showgrid=True,
                gridcolor='#333333',
                gridwidth=0.5,
                griddash='dash',
                showline=True,
                linecolor='#444444',
                range=[0, max(1.0, max(x_values) * 1.15 if x_values else 1.0)],
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                tickfont=dict(size=11, color='#e0e0e0'),
            ),
            margin=dict(l=20, r=20, t=10, b=30),
            height=max(180, len(y_labels) * 55),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # --- Evidence Panel ------------------------------------------------------
    st.markdown(
        "<div style='color:#6b1a1a; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; font-weight:bold; letter-spacing:1px; margin-bottom:6px;'>"
        "EVIDENCE"
        "</div>",
        unsafe_allow_html=True,
    )
    evidence_list = expl.get("evidence", [])
    if evidence_list:
        ev_lines = "".join(f"<div style='margin-bottom:4px;'>{i}. {ev}</div>" for i, ev in enumerate(evidence_list, 1))
        st.markdown(
            f"<div style='background-color:#111118; border:1px solid #2a0a0a; border-radius:4px; padding:12px; "
            f"font-family:\"JetBrains Mono\", Consolas, monospace; font-size:12px; color:#c0c0c0; line-height:1.6;'>"
            f"{ev_lines}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No evidence records available.")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # --- Reason Panel --------------------------------------------------------
    st.markdown(
        "<div style='color:#6b1a1a; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; font-weight:bold; letter-spacing:1px; margin-bottom:6px;'>"
        "REASON"
        "</div>",
        unsafe_allow_html=True,
    )
    human_reason = expl.get("human_reason", "Wallet flagged due to abnormal risk signals.")
    st.markdown(
        f"<div style='background-color:#111118; border:1px solid #2a0a0a; border-radius:4px; padding:12px; "
        f"font-family:\"JetBrains Mono\", Consolas, monospace; font-size:12px; color:#c0c0c0; line-height:1.6;'>"
        f"{human_reason}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # --- Threshold Sensitivity -----------------------------------------------
    tcol1, tcol2 = st.columns([1, 4])
    with tcol1:
        st.markdown(
            "<div style='color:#c0c0c0; font-family:\"JetBrains Mono\", Consolas, monospace; "
            "font-size:12px; padding-top:10px;'>Threshold:</div>",
            unsafe_allow_html=True,
        )
    with tcol2:
        threshold_val = st.slider(
            "Anomaly Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            label_visibility="collapsed",
        )
