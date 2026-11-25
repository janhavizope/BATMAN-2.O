"""
Network Graph Page
------------------
The wallet/transaction network of the top alerted entities:
- Top anomalous entities ranking (with CSV / HTML export).
- An interactive node-link graph (wallets and the transactions between them).
- Graph statistics and a centrality table.

(The entity risk timeline that used to live here was removed — it duplicated the
Overview time charts and the Investigate Wallet timeline. The non-functional
Hop/Layout/Colour controls were removed too.)
"""

import streamlit as st
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from src.web_frontend.components import render_metric_card, render_dev_badge, render_header
from src.data.data_manager import get_graph_data, get_alerts_df
from src.core.backend.live.live_store import LiveStore
from src.core.backend.live.monitor import run_cycle
from src.core.backend.live.watchlist import load_watchlist
from src.config import LIVE_DB

NODE_COLOURS = {
    "Wallet":      "#cc2222",
    "Transaction": "#2266aa",
    "IP":          "#22aa55",
    "ASN":         "#cc8822",
    "Country":     "#8844aa",
}
LEVEL_COLOURS = {"CRITICAL": "#ff4d4d", "HIGH": "#ff8c42", "MEDIUM": "#e0c04d", "LOW": "#4d8b4d"}


def render_network_graph() -> None:
    render_header("Network Traffic Analysis")
    st.markdown(
        "<div style='text-align:center; font-family:\"JetBrains Mono\", Consolas, monospace; "
        "color:#888888; font-size:14px; font-weight:bold; margin-top:-8px; margin-bottom:12px;'>"
        "Top Anomalous Entities &amp; Their Transaction Network"
        "</div>",
        unsafe_allow_html=True,
    )
    render_dev_badge()
    from src.web_frontend.components import render_dataset_notice

    source = st.radio(
        "Data source", ["Dataset (historical)", "Live (current network)"], horizontal=True,
        index=1,
        help="Dataset: the Elliptic++ alerted wallets. Live: wallets and transactions seen on the live "
             "feed — run the Live Monitor to populate it.",
    )
    if source.startswith("Live"):
        _render_live_graph()
        return

    render_dataset_notice()

    # --- Export buttons (top 10) --------------------------------------------
    alerts_df = get_alerts_df()
    top10_df = alerts_df.head(10).copy() if not alerts_df.empty else pd.DataFrame()
    btn_col1, btn_col2, btn_col3 = st.columns([5, 1, 1])
    with btn_col2:
        if not top10_df.empty:
            csv_data = top10_df[["Rank", "Entity ID", "Risk Score"]].to_csv(index=False).encode("utf-8")
            st.download_button("Download .csv", data=csv_data, file_name="top10_anomalous_entities.csv", mime="text/csv")
        else:
            st.button("Download .csv", disabled=True)
    with btn_col3:
        if not top10_df.empty:
            html_data = top10_df[["Rank", "Entity ID", "Risk Score"]].to_html().encode("utf-8")
            st.download_button("View HTML", data=html_data, file_name="top10_anomalous_entities.html", mime="text/html")
        else:
            st.button("View HTML", disabled=True)

    # --- Top anomalous ranking ----------------------------------------------
    st.markdown(
        "<div style='color:#b33939; font-family:\"Orbitron\", sans-serif; font-size:13px; font-weight:bold; margin:6px 0;'>"
        "TOP ANOMALOUS RANKING</div>",
        unsafe_allow_html=True,
    )
    if not top10_df.empty:
        st.dataframe(top10_df[["Rank", "Entity ID", "Risk Score"]], width="stretch", hide_index=True)
    else:
        st.info("No ranking data available.")

    # --- Network graph -------------------------------------------------------
    st.markdown(
        "<div style='color:#b33939; font-family:\"Orbitron\", sans-serif; font-size:13px; font-weight:bold; margin:14px 0 6px 0;'>"
        "TRANSACTION NETWORK</div>",
        unsafe_allow_html=True,
    )
    nodes, edges, stats, cent_table = get_graph_data()
    G = nx.Graph()
    for node in nodes:
        G.add_node(node["id"], **node)
    for edge in edges:
        G.add_edge(edge["source"], edge["target"], relation=edge.get("relation", ""))

    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, seed=42, k=1.8)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="rgba(231, 76, 60, 0.25)"),
                                hoverinfo="none", mode="lines")

        node_x, node_y, node_text, node_color = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            ntype = G.nodes[node].get("type", "Wallet")
            risk = G.nodes[node].get("risk_score", 0)
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{G.nodes[node].get('label', node)}<br>Type: {ntype}<br>Risk: {risk}")
            node_color.append(NODE_COLOURS.get(ntype, "#ffffff"))
        node_trace = go.Scatter(x=node_x, y=node_y, mode="markers", hovertext=node_text, hoverinfo="text",
                                marker=dict(size=12, color=node_color, line=dict(width=1, color="#333333")))

        fig_network = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
            showlegend=False, hovermode="closest", margin=dict(b=20, l=20, r=20, t=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#080404", paper_bgcolor="#080404", height=450))
        st.plotly_chart(fig_network, width="stretch")
    else:
        st.info("No graph data available.")

    st.caption("Node colours: " + " | ".join(f"**{k}** ({v})" for k, v in NODE_COLOURS.items()))
    st.caption("ℹ️ To see **why** any of these wallets was flagged — the feature contributions and how it "
               "compares with typical licit/illicit wallets — open it on the **Investigate Wallet** page.")

    # --- Graph statistics ----------------------------------------------------
    st.markdown(
        "<div style='color:#ff4d4d; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; "
        "font-weight:bold; letter-spacing:1px; margin:14px 0 8px 0;'>GRAPH STATISTICS</div>",
        unsafe_allow_html=True,
    )
    gs1, gs2, gs3, gs4 = st.columns(4)
    with gs1:
        render_metric_card("TOTAL NODES", str(stats.get("total_nodes", 0)))
    with gs2:
        render_metric_card("TOTAL EDGES", str(stats.get("total_edges", 0)))
    with gs3:
        render_metric_card("CLUSTERS DETECTED", str(stats.get("clusters_detected", 0)))
    with gs4:
        avg = stats.get("avg_degree", 0)
        render_metric_card("AVG DEGREE", f"{avg:.2f}" if isinstance(avg, (int, float)) else str(avg))

    st.markdown(
        "<div style='color:#b33939; font-family:\"Orbitron\", sans-serif; font-size:13px; font-weight:bold; margin:14px 0 6px 0;'>"
        "CENTRALITY TABLE</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(cent_table, width="stretch", hide_index=True)


def _render_live_graph() -> None:
    """The wallets and transactions seen on the live feed, as a graph."""
    st.markdown(
        "<div style=\"background-color:#0a1a0a; border-left:3px solid #22aa55; border-radius:3px; "
        "padding:8px 14px; font-size:11px; font-family:'JetBrains Mono', monospace; color:#22aa55; "
        "margin-bottom:10px;\"><strong>[LIVE NETWORK]</strong> &mdash; wallets (red) and transactions "
        "(coloured by risk) seen on the live feed. Run the <strong>Live Monitor</strong> to populate it.</div>",
        unsafe_allow_html=True,
    )
    store = LiveStore(LIVE_DB)
    refresh_col, size_col = st.columns([2, 1])
    with refresh_col:
        refresh = st.button("Fetch latest live transactions", type="primary", key="graph_live_refresh")
    with size_col:
        max_tx = st.number_input("Transactions", min_value=5, max_value=100, value=25, step=5,
                                 key="graph_live_max_tx")
    if refresh:
        with st.spinner("Fetching and analysing the latest live transactions ..."):
            watchlist, _ = load_watchlist()
            result = run_cycle(store, max_tx=int(max_tx), watchlist=watchlist)
        if result.error:
            st.error(f"Live feed error: {result.error}")
        else:
            st.success(
                f"Live cycle complete — {result.valid} valid transactions, "
                f"{result.flagged_tx} flagged."
            )

    edges = store.graph_edges(limit=250)
    if edges.empty:
        st.info("No live network data yet. Open the **Live Monitor** page and let it fetch some "
                "transactions, then come back here.")
        return

    G = nx.Graph()
    for _, e in edges.iterrows():
        tx_node = f"tx:{e['txid'][:10]}"
        G.add_node(tx_node, kind="tx", level=e["risk_level"], label=e["txid"][:12])
        G.add_node(e["address"], kind="wallet", label=str(e["address"])[:12])
        G.add_edge(tx_node, e["address"])

    pos = nx.spring_layout(G, seed=42, k=0.9)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="rgba(136,136,136,0.25)"),
                            hoverinfo="none", mode="lines")

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for n in G.nodes():
        data = G.nodes[n]
        node_x.append(pos[n][0]); node_y.append(pos[n][1])
        if data.get("kind") == "tx":
            node_color.append(LEVEL_COLOURS.get(data.get("level", "LOW"), "#4d8b4d"))
            node_size.append(11)
            node_text.append(f"Transaction {data.get('label')}<br>Risk: {data.get('level')}")
        else:
            node_color.append("#cc2222"); node_size.append(8)
            node_text.append(f"Wallet {data.get('label')}")
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers", hovertext=node_text, hoverinfo="text",
                            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="#222")))
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        showlegend=False, hovermode="closest", margin=dict(b=20, l=20, r=20, t=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#080404", paper_bgcolor="#080404", height=460))
    st.plotly_chart(fig, width="stretch")
    st.caption("Transaction nodes are coloured by risk (green LOW → red CRITICAL); wallet nodes are red. "
               "Edges link a transaction to the wallets that sent or received in it.")

    tx_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "tx"]
    wallet_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "wallet"]
    flagged = sum(1 for n in tx_nodes if G.nodes[n].get("level") in ("HIGH", "CRITICAL"))
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        render_metric_card("LIVE TRANSACTIONS", f"{len(tx_nodes):,}")
    with g2:
        render_metric_card("LIVE WALLETS", f"{len(wallet_nodes):,}")
    with g3:
        render_metric_card("EDGES", f"{G.number_of_edges():,}")
    with g4:
        render_metric_card("FLAGGED TX", f"{flagged:,}")
