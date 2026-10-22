"""
Transaction Analysis Page
-------------------------
Restores the desktop command-center visual layout:
- TRANSACTION ANALYSIS header + SYSTEM ONLINE badge
- Live mode DevBadge
- Embedded Attack Graph canvas
- Cybersecurity DataTable with red headers
- Metric cards for selected transaction
- Monospaced Extended Details panel
- Bottom filter controls & Export actions (CSV, HTML)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from src.web_frontend.components import render_metric_card, render_dev_badge, render_header
from src.data.data_manager import get_transactions_df, get_tx_detail, get_graph_data


def render_transaction_analysis() -> None:
    render_header("TRANSACTION ANALYSIS")
    render_dev_badge()
    from src.web_frontend.components import render_dataset_notice
    render_dataset_notice()

    transactions_df = get_transactions_df()

    # --- Filters Row (top/bottom sync) --------------------------------------
    fcol1, fcol2, fcol3, fcol4 = st.columns([1.5, 1.5, 1, 1])
    with fcol1:
        flag_filter = st.selectbox("Flag", options=["All Flags", "Suspicious", "Normal"], index=0)
    with fcol2:
        min_amount = st.number_input("Min Amount (BTC)", min_value=0.0, value=0.0, step=0.1)

    # Filter data
    filtered_df = transactions_df.copy()
    if flag_filter == "Suspicious":
        filtered_df = filtered_df[filtered_df["Flag"].str.lower() != "normal"]
    elif flag_filter == "Normal":
        filtered_df = filtered_df[filtered_df["Flag"].str.lower() == "normal"]

    if min_amount > 0:
        filtered_df = filtered_df[filtered_df["Amount (BTC)"] >= min_amount]

    # --- Embedded High-Level Attack Graph ------------------------------------
    with st.expander("NETWORK ATTACK GRAPH PREVIEW", expanded=True):
        try:
            nodes, edges, _, _ = get_graph_data()
            G = nx.Graph()
            for n in nodes[:40]:
                G.add_node(n["id"], **n)
            for e in edges[:50]:
                if e["source"] in G.nodes and e["target"] in G.nodes:
                    G.add_edge(e["source"], e["target"])

            if len(G.nodes) > 0:
                pos = nx.spring_layout(G, seed=42, k=1.8)
                edge_x, edge_y = [], []
                for u, v in G.edges():
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color="rgba(231, 76, 60, 0.3)"),
                    hoverinfo="none",
                    mode="lines",
                )

                node_x, node_y, node_color, node_text = [], [], [], []
                for n in G.nodes():
                    x, y = pos[n]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(str(n)[:12])
                    node_color.append("#ffffff")

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode="markers",
                    marker=dict(size=6, color=node_color, line=dict(color="#8b0000", width=1)),
                    hovertext=node_text,
                    hoverinfo="text",
                )

                fig_graph = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=10, l=10, r=10, t=10),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor="#080404",
                        paper_bgcolor="#080404",
                        height=240,
                    ),
                )
                st.plotly_chart(fig_graph, use_container_width=True)
            else:
                st.markdown(
                    "<div style='text-align:center; padding:40px; color:#5e1313; font-family:\"Orbitron\", sans-serif; font-size:14px;'>"
                    "No graph data available"
                    "</div>",
                    unsafe_allow_html=True,
                )
        except Exception:
            st.markdown(
                "<div style='text-align:center; padding:40px; color:#5e1313; font-family:\"Orbitron\", sans-serif; font-size:14px;'>"
                "No graph data available"
                "</div>",
                unsafe_allow_html=True,
            )

    # --- Transaction Table ---------------------------------------------------
    st.markdown(
        "<div style='color:#b33939; font-family:\"Orbitron\", sans-serif; font-size:13px; font-weight:bold; letter-spacing:1px; margin-top:10px; margin-bottom:6px;'>"
        "TRANSACTION FEED"
        "</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(filtered_df.head(1000), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # --- Transaction Detail Cards --------------------------------------------
    st.markdown(
        "<div style='color:#ff4d4d; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; font-weight:bold; letter-spacing:1px; margin-bottom:8px;'>"
        "TRANSACTION DETAIL"
        "</div>",
        unsafe_allow_html=True,
    )

    tx_ids = filtered_df["Transaction ID"].tolist() if not filtered_df.empty else transactions_df["Transaction ID"].tolist()
    if tx_ids:
        selected_tx = st.selectbox(
            "Select Transaction to Inspect",
            options=tx_ids[:1000],
            index=0,
            label_visibility="collapsed",
        )
        matching_rows = transactions_df[transactions_df["Transaction ID"] == selected_tx]
    else:
        selected_tx = None
        matching_rows = pd.DataFrame()
        st.markdown(
            "<div style='background-color:#111118; border:1px solid #2a0a0a; border-radius:4px; padding:12px; "
            "font-family:\"JetBrains Mono\", monospace; font-size:11px; color:#666666;'>"
            "No transactions available to inspect."
            "</div>",
            unsafe_allow_html=True,
        )
    if not matching_rows.empty:
        tx_row = matching_rows.iloc[0]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            render_metric_card("TX ID", str(tx_row["Transaction ID"])[:8] + "...")
        with c2:
            render_metric_card("AMOUNT", f"{tx_row['Amount (BTC)']:.4f} BTC")
        with c3:
            render_metric_card("TIMESTAMP", str(tx_row["Timestamp"]).split(" ")[-1])
        with c4:
            render_metric_card("SOURCE", str(tx_row["Source"])[:8] + "...")
        with c5:
            render_metric_card("DESTINATION", str(tx_row["Destination"])[:8] + "...")
        with c6:
            render_metric_card("FLAG", str(tx_row["Flag"]))

        # Extended Details Panel
        edetail = get_tx_detail(selected_tx)
        conf = edetail.get('confirmations', 6) if edetail else 6
        fee = edetail.get('fee_btc', 0.0001) if edetail else 0.0001
        block = edetail.get('block_height', 834920) if edetail else 834920
        reason = edetail.get('reason', 'Standard transaction pattern') if edetail else 'Normal transaction flow'

        st.markdown(
            f"<div style='background-color:#111118; border:1px solid #2a0a0a; border-radius:4px; padding:12px; margin-top:10px; "
            f"font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; color:#c0c0c0; line-height:1.6;'>"
            f"<strong>Source:</strong> {tx_row['Source']}<br>"
            f"<strong>Destination:</strong> {tx_row['Destination']}<br>"
            f"<strong>Confirmations:</strong> {conf} &nbsp;|&nbsp; <strong>Fee:</strong> {fee:.8f} BTC &nbsp;|&nbsp; <strong>Block:</strong> {block}<br>"
            f"<strong>Reason:</strong> {reason}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # --- Bottom Export Buttons -----------------------------------------------
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    ecol1, ecol2, ecol3 = st.columns([4, 1, 1])
    with ecol2:
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download .csv",
            data=csv_data,
            file_name="batman_transactions.csv",
            mime="text/csv",
        )
    with ecol3:
        html_report = filtered_df.head(50).to_html()
        st.download_button(
            label="View HTML",
            data=html_report,
            file_name="batman_transactions.html",
            mime="text/html",
        )
