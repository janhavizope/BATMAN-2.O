"""
Entity Investigation Page
-------------------------
Restores the desktop command-center visual layout:
- Header: ENTITY INVESTIGATION + SYSTEM ONLINE indicator
- Live mode DevBadge
- Entity Selector combo
- 2x3 Metric Card Grid: ENTITY ID, RISK SCORE, RISK LEVEL, ANOMALY SCORE, TRANSACTIONS, COUNTERPARTIES
- Tabs: Timeline, Network, Evidence, Features, Explanation, Notes
"""

import streamlit as st
import pandas as pd
from src.web_frontend.components import render_metric_card, render_dev_badge, render_header
from src.data.data_manager import get_entities, get_entity_ids


def render_entity_investigation() -> None:
    render_header("ENTITY INVESTIGATION")
    render_dev_badge()
    from src.web_frontend.components import render_dataset_notice
    render_dataset_notice()

    entity_ids = get_entity_ids()
    entities = get_entities()

    if not entity_ids:
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            render_metric_card("ENTITY ID", "N/A")
        with r1c2:
            render_metric_card("RISK SCORE", "0/100")
        with r1c3:
            render_metric_card("RISK LEVEL", "None")
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            render_metric_card("ANOMALY SCORE", "0.00")
        with r2c2:
            render_metric_card("TRANSACTIONS", "0")
        with r2c3:
            render_metric_card("COUNTERPARTIES", "0")
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.info("No entities available. Please ingest an external Bitcoin transaction dataset to begin investigation.")
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
            "Choose an Entity ID to investigate",
            options=entity_ids,
            index=0,
            label_visibility="collapsed",
        )

    entity = entities.get(entity_id)
    if entity is None:
        st.warning("Entity not found in live dataset.")
        return

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # --- Entity Profile Cards Grid (2 rows x 3 cols) ------------------------
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        render_metric_card("ENTITY ID", entity["entity_id"])
    with r1c2:
        render_metric_card("RISK SCORE", f"{entity['risk_score']}/100")
    with r1c3:
        render_metric_card("RISK LEVEL", str(entity["risk_level"]))

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        render_metric_card("ANOMALY SCORE", f"{entity['anomaly_score']:.2f}")
    with r2c2:
        render_metric_card("TRANSACTIONS", f"{entity['transaction_count']:,}")
    with r2c3:
        render_metric_card("COUNTERPARTIES", f"{entity['counterparty_count']:,}")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # --- Tabs ----------------------------------------------------------------
    tab_timeline, tab_network, tab_evidence, tab_features, tab_explanation, tab_notes = st.tabs(
        ["Timeline", "Network", "Evidence", "Features", "Explanation", "Notes"]
    )

    with tab_timeline:
        timeline_data = entity.get("timeline", [])
        if timeline_data:
            tdf = pd.DataFrame(timeline_data)
            st.dataframe(tdf, use_container_width=True, hide_index=True)
        else:
            st.info("No timeline events recorded.")

    with tab_network:
        net_data = entity.get("network_associations", [])
        if net_data:
            ndf = pd.DataFrame(net_data)
            st.dataframe(ndf, use_container_width=True, hide_index=True)
        else:
            st.info("No network associations recorded.")

    with tab_evidence:
        evidence_list = entity.get("evidence", [])
        if evidence_list:
            ev_html = "".join(f"<div style='margin-bottom:6px;'>{i}. {ev}</div>" for i, ev in enumerate(evidence_list, 1))
            st.markdown(
                f"<div style='background-color:#111118; border:1px solid #2a0a0a; border-radius:4px; padding:12px; "
                f"font-family:\"JetBrains Mono\", Consolas, monospace; font-size:12px; color:#c0c0c0; line-height:1.6;'>"
                f"{ev_html}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No evidence records available.")

    with tab_features:
        features_data = entity.get("top_features", [])
        if features_data:
            fdf = pd.DataFrame(features_data)
            st.dataframe(fdf, use_container_width=True, hide_index=True)
        else:
            st.info("No feature contributions available.")

    with tab_explanation:
        expl_text = entity.get("explanation", "No explanation available.")
        st.markdown(
            f"<div style='background-color:#111118; border:1px solid #2a0a0a; border-radius:4px; padding:12px; "
            f"font-family:\"JetBrains Mono\", Consolas, monospace; font-size:12px; color:#c0c0c0; line-height:1.6;'>"
            f"{expl_text}"
            f"</div>",
            unsafe_allow_html=True,
        )

    with tab_notes:
        st.text_area(
            "Investigation Notes",
            placeholder="Type investigation notes here...",
            height=120,
            key=f"notes_{entity_id}",
            label_visibility="collapsed",
        )
        st.button("Save Notes", disabled=True)
