"""
Alerts Page
-----------
Restores the desktop command-center visual layout:
- ALERTS header + SYSTEM ONLINE badge
- Live mode DevBadge
- Single-line Filter Bar: Level combo (All, Critical, High, Medium, Low) + Entity filter input
- Technical Alert Feed table with red headers
- Technical ALERT DETAIL monospaced panel
"""

import streamlit as st
import pandas as pd
from src.web_frontend.components import render_dev_badge, render_header
from src.data.data_manager import get_alerts_df


def render_alerts() -> None:
    render_header("ALERTS")
    render_dev_badge()
    from src.web_frontend.components import render_dataset_notice
    render_dataset_notice()

    alerts_df = get_alerts_df()

    # --- Filters Row ---------------------------------------------------------
    fcol1, fcol2 = st.columns([1.5, 2.5])
    with fcol1:
        level_filter = st.selectbox(
            "Level:",
            options=["All", "Critical", "High", "Medium", "Low"],
            index=0,
        )
    with fcol2:
        entity_search = st.text_input(
            "Entity:",
            placeholder="Filter by Entity ID...",
        )

    # --- Apply Filters -------------------------------------------------------
    filtered = alerts_df.copy()
    if not filtered.empty:
        if level_filter != "All":
            filtered = filtered[filtered["Risk Level"] == level_filter]
        if entity_search:
            filtered = filtered[
                filtered["Entity ID"].str.contains(entity_search, case=False, na=False)
            ]

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # --- Alert Feed Table ----------------------------------------------------
    st.markdown(
        f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>"
        f"<div style='color:#b33939; font-family:\"Orbitron\", sans-serif; font-size:13px; font-weight:bold; letter-spacing:1px;'>ALERT FEED</div>"
        f"<div style='color:#888888; font-family:\"JetBrains Mono\", monospace; font-size:11px;'>Showing {len(filtered)} of {len(alerts_df)} live alerts</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.info("No alerts match the selected criteria.")
    else:
        display_cols = ["Rank", "Entity ID", "Risk Score", "Risk Level", "Main Reason"]
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # --- Alert Detail ----------------------------------------------------
        st.markdown(
            "<div style='color:#6b1a1a; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; font-weight:bold; letter-spacing:1px; margin-bottom:6px;'>"
            "ALERT DETAIL"
            "</div>",
            unsafe_allow_html=True,
        )

        selected_id = st.selectbox(
            "Select Entity to view alert details",
            options=filtered["Entity ID"].tolist(),
            index=0,
            label_visibility="collapsed",
        )

        matching_rows = filtered[filtered["Entity ID"] == selected_id]
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            full_id = row.get("Full Entity ID", row["Entity ID"])
            risk_score = row["Risk Score"]
            risk_level = row["Risk Level"]
            main_reason = row["Main Reason"]

            st.markdown(
                f"<div style='background-color:#111118; border:1px solid #2a0a0a; border-radius:4px; padding:14px; "
                f"font-family:\"JetBrains Mono\", Consolas, monospace; font-size:12px; color:#c0c0c0; line-height:1.7;'>"
                f"<strong>Entity:</strong>&nbsp;&nbsp;&nbsp;&nbsp;{full_id}<br>"
                f"<strong>Risk:</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{risk_score}/100<br>"
                f"<strong>Level:</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{risk_level}<br>"
                f"<strong>Reason:</strong>&nbsp;&nbsp;&nbsp;&nbsp;{main_reason}"
                f"</div>",
                unsafe_allow_html=True,
            )
