"""
Entity Investigation Page
-------------------------
Restores the desktop command-center visual layout:
- Header: ENTITY INVESTIGATION
- Live mode DevBadge
- Entity Selector combo
- 2x3 Metric Card Grid: ENTITY ID, RISK SCORE, RISK LEVEL, ANOMALY SCORE, TRANSACTIONS, COUNTERPARTIES
- Tabs: Timeline, Network, Evidence, Features, Explanation, Notes
"""

import streamlit as st
import pandas as pd
from src.web_frontend.components import render_metric_card, render_dev_badge, render_header
from src.web_frontend.explainability import render_explanation_detail
from src.data.data_manager import get_entity, get_entity_ids


def render_entity_investigation() -> None:
    render_header("INVESTIGATE WALLET")
    render_dev_badge()
    from src.web_frontend.components import render_dataset_notice
    render_dataset_notice(    )

    entity_ids = get_entity_ids()

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
        st.info("No entities available. Run scripts/run_pipeline.py to create the wallet risk scores, "
                "or score your own data on the Upload & Score page.")
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

    entity = get_entity(entity_id)
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
    tab_why, tab_transactions, tab_timeline, tab_network, tab_notes = st.tabs(
        ["Why flagged", "Transactions", "Timeline", "Network", "Notes"],
        on_change="rerun",
    )

    if tab_why.open:
        # The rich explanation (verdict, feature contributions, licit/illicit
        # comparison, reason) — formerly the separate Explainability page.
        render_explanation_detail(entity_id, entity)

    if tab_transactions.open:
        tx_data = entity.get("transactions", [])
        if tx_data:
            st.caption(f"{len(tx_data):,} transactions this wallet sent or received (Elliptic++ AddrTx / TxAddr).")
            st.dataframe(pd.DataFrame(tx_data), width="stretch", hide_index=True)
        else:
            st.info("No wallet-transaction links available. They are written by scripts/score_wallets.py "
                    "when the Elliptic++ AddrTx / TxAddr edge lists are present.")

    if tab_timeline.open:
        timeline_data = entity.get("timeline", [])
        if timeline_data:
            st.dataframe(pd.DataFrame(timeline_data), width="stretch", hide_index=True)
        else:
            st.info("No timeline events recorded.")

    if tab_network.open:
        net_data = entity.get("network_associations", [])
        if net_data:
            st.dataframe(pd.DataFrame(net_data), width="stretch", hide_index=True)
        else:
            st.info("No network associations recorded.")

    if tab_notes.open:
        st.text_area(
            "Investigation Notes",
            placeholder="Type investigation notes here...",
            height=120,
            key=f"notes_{entity_id}",
            label_visibility="collapsed",
        )
        st.button("Save Notes", disabled=True)
