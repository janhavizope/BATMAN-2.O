"""
Overview Page
-------------
High-level dashboard matching original BATMAN desktop GUI aesthetic:
- DASHBOARD OVERVIEW title + SYSTEM ONLINE indicator
- [LIVE MODE] DevBadge
- 6 Dark KPI metric cards with thin red borders and Orbitron numbers
- Responsive charts: "Alerts over time (7d)" (neon green #21d093) + "Severity Distribution" donut
- RECENT ALERTS technical monospace summary panel
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.web_frontend.components import render_metric_card, render_dev_badge, render_header
from src.data.data_manager import get_overview_stats, get_alerts_df


def render_overview() -> None:
    # Top header with title and SYSTEM ONLINE badge
    render_header("DASHBOARD OVERVIEW")
    render_dev_badge()
    from src.web_frontend.components import render_dataset_notice
    render_dataset_notice()

    stats = get_overview_stats()
    alerts_df = get_alerts_df()

    # --- Row 1: Primary KPIs ------------------------------------------------
    row1 = st.columns(6)
    cards = [
        ("TOTAL TRANSACTIONS", f"{stats['total_transactions']:,}"),
        ("TOTAL WALLETS",      f"{stats['total_wallets']:,}"),
        ("TOTAL IPS",          f"{stats['total_ips']:,}") if "total_ips" in stats
        else ("KNOWN ILLICIT",  f"{stats['known_illicit']:,}"),
        ("ANOMALOUS",          f"{stats['anomalous_entities']:,}"),
        ("HIGH-RISK",          f"{stats['high_risk_entities']:,}"),
        ("CRITICAL ALERTS",    f"{stats['critical_alerts']:,}"),
    ]
    for col, (label, value) in zip(row1, cards):
        with col:
            render_metric_card(label, value)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # --- Row 2: Charts ------------------------------------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            "<div style='color:#c0c0c0; font-family:\"Orbitron\", sans-serif; font-size:13px; font-weight:bold; margin-bottom:8px;'>"
            "Flagged activity over time"
            "</div>",
            unsafe_allow_html=True,
        )

        # Activity timeline chart — hour buckets (transaction data) or Elliptic time steps
        from src.data.data_manager import get_activity_timeline
        times, counts = get_activity_timeline()

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=times,
            y=counts,
            mode='lines+markers',
            line=dict(color='#21d093', width=2.5, shape='spline'),
            marker=dict(size=5, color='#21d093', line=dict(color='#080404', width=1)),
            fill='tozeroy',
            fillcolor='rgba(33, 208, 147, 0.08)',
            hovertemplate='Time: %{x}<br>Count: %{y}<extra></extra>',
        ))
        fig_line.update_layout(
            plot_bgcolor='#0a0a0f',
            paper_bgcolor='#0a0a0f',
            font=dict(color='#888888', family='JetBrains Mono, monospace', size=10),
            xaxis=dict(
                showgrid=True,
                gridcolor='#2a3b45',
                gridwidth=0.5,
                griddash='dash',
                showline=True,
                linecolor='#2a3b45',
                tickmode='array',
                tickvals=times[::max(1, len(times) // 6)],
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#2a3b45',
                gridwidth=0.5,
                griddash='dash',
                showline=True,
                linecolor='#2a3b45',
            ),
            margin=dict(l=35, r=20, t=20, b=30),
            height=300,
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col2:
        st.markdown(
            "<div style='color:#c0c0c0; font-family:\"Orbitron\", sans-serif; font-size:13px; font-weight:bold; margin-bottom:8px;'>"
            "Severity Distribution"
            "</div>",
            unsafe_allow_html=True,
        )

        if alerts_df.empty:
            severity_counts = pd.DataFrame({'Severity': ['No Data'], 'Count': [1]})
            color_map = {'No Data': '#333333'}
        else:
            severity_counts = alerts_df['Risk Level'].value_counts().reset_index()
            severity_counts.columns = ['Severity', 'Count']
            color_map = {
                "Critical": "#f2353c",
                "High": "#f39c12",
                "Medium": "#3498db",
                "Low": "#2ecc71",
            }

        fig_donut = px.pie(
            severity_counts, values='Count', names='Severity',
            hole=0.45,
            color='Severity',
            color_discrete_map=color_map,
        )
        fig_donut.update_traces(
            marker=dict(line=dict(color='#0a0a0f', width=2)),
            textfont=dict(family='JetBrains Mono, monospace', color='#ffffff', size=11),
        )
        fig_donut.update_layout(
            plot_bgcolor='#0a0a0f',
            paper_bgcolor='#0a0a0f',
            font=dict(color='#c0c0c0', family='JetBrains Mono, monospace', size=10),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=0.95,
                font=dict(size=10, family='JetBrains Mono, monospace', color='#c0c0c0'),
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # --- Row 3: Recent Alerts Summary ---------------------------------------
    st.markdown(
        "<div style='color:#6b1a1a; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; font-weight:bold; letter-spacing:1px; margin-bottom:6px;'>"
        "RECENT ALERTS"
        "</div>",
        unsafe_allow_html=True,
    )

    if not alerts_df.empty:
        top_alerts = alerts_df.head(3)
        alert_lines = []
        for _, row in top_alerts.iterrows():
            level = row.get('Risk Level', 'High')
            eid = row.get('Full Entity ID', row.get('Entity ID', 'Unknown'))
            reason = row.get('Main Reason', 'anomaly detected')
            alert_lines.append(f"[{level}] {eid}  &mdash;  {reason}")
        alerts_html = "<br>".join(alert_lines)
    else:
        alerts_html = "[INFO] No alerts available &mdash; Ingest an external Bitcoin transaction dataset to begin analysis."

    st.markdown(
        f"<div style='background-color:#111118; border:1px solid #2a0a0a; border-radius:4px; padding:12px; "
        f"font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; color:#888888; line-height:1.6;'>"
        f"{alerts_html}"
        f"</div>",
        unsafe_allow_html=True,
    )
