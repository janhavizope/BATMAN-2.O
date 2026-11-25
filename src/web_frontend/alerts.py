"""
Alerts Page
-----------
Two views of the flagged wallets, in one place:
- ALERT FEED  — the curated alert list (level filter, entity search, alert detail).
- SCAN & LOOKUP — every scored wallet with an adjustable risk threshold, a ranked
  table filterable by group, and a per-wallet lookup comparing the wallet's key
  features with typical licit and illicit wallets (formerly the Wallet Scanner).
"""

from html import escape

import pandas as pd
import streamlit as st
from src.web_frontend.components import render_dev_badge, render_header, render_metric_card
from src.data.data_manager import get_alerts_df
from src.data.elliptic_source import WALLET_SCORES as SCORES_FILE, get_thresholds
from src.core.backend.live.live_store import LiveStore
from src.core.backend.live.monitor import run_cycle
from src.core.backend.live.watchlist import load_watchlist
from src.config import LIVE_DB

LIVE_TX_COLUMNS = ["txid", "risk_level", "risk_score", "supervised_risk", "n_inputs",
                   "n_outputs", "total_btc", "reasons", "fetched_at"]

GROUPS = {
    "Unlabelled wallets (new predictions)": "unknown",
    "Test-set wallets (verify against true label)": "test",
    "All wallets": None,
}


@st.cache_data
def _load_scores() -> pd.DataFrame:
    return pd.read_parquet(SCORES_FILE)


def _section_label(text: str) -> None:
    st.markdown(
        "<div style='color:#b33939; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; "
        f"font-weight:bold; letter-spacing:1px; margin:14px 0 8px 0;'>{text}</div>",
        unsafe_allow_html=True,
    )


def render_alerts() -> None:
    render_header("ALERTS")
    render_dev_badge()
    from src.web_frontend.components import render_dataset_notice
    render_dataset_notice()

    tab_live, tab_feed, tab_scan = st.tabs(
        ["Live Alerts", "Dataset Alert Feed", "Dataset Scan & Lookup"],
        on_change="rerun",
    )
    if tab_live.open:
        _render_live_alerts()
    if tab_feed.open:
        _render_alert_feed()
    if tab_scan.open:
        _render_scan_and_lookup()


def _render_live_alerts() -> None:
    """Alerts from the live feed: flagged live transactions and riskiest wallets."""
    st.markdown(
        "<div style=\"background-color:#0a1a0a; border-left:3px solid #22aa55; border-radius:3px; "
        "padding:8px 14px; font-size:11px; font-family:'JetBrains Mono', monospace; color:#22aa55; "
        "margin-bottom:10px;\"><strong>[LIVE ALERTS]</strong> &mdash; suspicious transactions and wallets "
        "seen on the live Bitcoin feed. Run the <strong>Live Monitor</strong> to populate it.</div>",
        unsafe_allow_html=True,
    )
    store = LiveStore(LIVE_DB)
    refresh_col, size_col = st.columns([2, 1])
    with refresh_col:
        refresh = st.button("Fetch latest live transactions", type="primary", key="alerts_live_refresh")
    with size_col:
        max_tx = st.number_input("Transactions", min_value=5, max_value=100, value=25, step=5,
                                 key="alerts_live_max_tx")
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

    summary = store.summary()
    if summary["total_transactions"] == 0:
        st.info("No live data yet. Open the **Live Monitor** page and let it fetch some transactions.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("FLAGGED (HIGH/CRIT)", f"{summary['flagged_transactions']:,}")
    with c2:
        render_metric_card("RISKY WALLETS", f"{summary['high_or_critical']:,}")
    with c3:
        render_metric_card("CYCLES RUN", f"{summary['cycles']:,}")

    _section_label("FLAGGED LIVE TRANSACTIONS (HIGH / CRITICAL)")
    flagged = store.transactions(limit=200, min_level="HIGH")
    if flagged.empty:
        st.caption("Nothing flagged on the live feed yet — most live traffic is ordinary. Keep the "
                   "Live Monitor running to catch mixers, sweeps, dust and watchlist hits.")
    else:
        show = [c for c in LIVE_TX_COLUMNS if c in flagged.columns]
        st.dataframe(flagged[show], width="stretch", hide_index=True)

    _section_label("RISKIEST LIVE WALLETS")
    wallets = store.wallets(limit=100)
    if wallets.empty:
        st.caption("No live wallets tracked yet.")
    else:
        cols = [c for c in ["address", "risk_score", "risk_level", "max_risk_score",
                            "total_tx_count", "top_reasons", "last_seen"] if c in wallets.columns]
        st.dataframe(wallets[cols], width="stretch", hide_index=True)


def _render_alert_feed() -> None:
    alerts_df = get_alerts_df()

    fcol1, fcol2 = st.columns([1.5, 2.5])
    with fcol1:
        level_filter = st.selectbox("Level:", options=["All", "Critical", "High", "Medium", "Low"], index=0)
    with fcol2:
        entity_search = st.text_input("Entity:", placeholder="Filter by Entity ID...")

    filtered = alerts_df.copy()
    if not filtered.empty:
        if level_filter != "All":
            filtered = filtered[filtered["Risk Level"] == level_filter]
        if entity_search:
            filtered = filtered[filtered["Entity ID"].str.contains(entity_search, case=False, na=False)]

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>"
        f"<div style='color:#b33939; font-family:\"Orbitron\", sans-serif; font-size:13px; font-weight:bold; letter-spacing:1px;'>ALERT FEED</div>"
        f"<div style='color:#888888; font-family:\"JetBrains Mono\", monospace; font-size:11px;'>Showing {len(filtered)} of {len(alerts_df)} alerts</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.info("No alerts match the selected criteria.")
        return

    display_cols = ["Rank", "Entity ID", "Risk Score", "Risk Level", "Main Reason"]
    st.dataframe(filtered[display_cols], width="stretch", hide_index=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#6b1a1a; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; "
        "font-weight:bold; letter-spacing:1px; margin-bottom:6px;'>ALERT DETAIL</div>",
        unsafe_allow_html=True,
    )
    selected_id = st.selectbox("Select Entity to view alert details",
                               options=filtered["Entity ID"].tolist(), index=0, label_visibility="collapsed")
    matching_rows = filtered[filtered["Entity ID"] == selected_id]
    if not matching_rows.empty:
        row = matching_rows.iloc[0]
        full_id = row.get("Full Entity ID", row["Entity ID"])
        st.markdown(
            f"<div style='background-color:#111118; border:1px solid #2a0a0a; border-radius:4px; padding:14px; "
            f"font-family:\"JetBrains Mono\", Consolas, monospace; font-size:12px; color:#c0c0c0; line-height:1.7;'>"
            f"<strong>Entity:</strong>&nbsp;&nbsp;&nbsp;&nbsp;{escape(str(full_id))}<br>"
            f"<strong>Risk:</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{row['Risk Score']}/100<br>"
            f"<strong>Level:</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{escape(str(row['Risk Level']))}<br>"
            f"<strong>Reason:</strong>&nbsp;&nbsp;&nbsp;&nbsp;{escape(str(row['Main Reason']))}"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_scan_and_lookup() -> None:
    if not SCORES_FILE.exists():
        st.info("No wallet scores found. Run scripts/train_elliptic_plus.py wallets, then scripts/score_wallets.py.")
        return

    scores = _load_scores()
    feature_cols = [c for c in scores.columns if c not in ("address", "time_step", "class", "risk", "split")]

    alert = get_thresholds()["wallet"]
    threshold = st.slider("Risk threshold (flag wallets at or above)", 0.05, 1.0,
                          round(alert["threshold"], 2), 0.01)
    st.caption(
        f"Default {alert['threshold']:.2f} is the alert threshold, chosen from the test-set precision-recall "
        f"curve to keep alerts near {alert['budget']:,} wallets. On test wallets it gives precision "
        f"{alert['precision']:.0%} and recall {alert['recall']:.0%}."
    )

    unknown = scores[scores["split"] == "unknown"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("WALLETS SCANNED", f"{len(scores):,}")
    with c2:
        render_metric_card("HIGH RISK (UNLABELLED)", f"{(unknown['risk'] >= threshold).sum():,}")
    with c3:
        render_metric_card("TEST-SET WALLETS", f"{(scores['split'] == 'test').sum():,}")
    with c4:
        render_metric_card("THRESHOLD", f"{threshold:.2f}")

    st.caption("Training-set wallets are excluded from the ranking: the model has already seen their labels, "
               "so their scores are not genuine predictions.")

    _section_label("HIGHEST-RISK WALLETS")
    gcol1, gcol2 = st.columns([3, 1])
    with gcol1:
        group = GROUPS[st.selectbox("Group", options=list(GROUPS), index=0)]
    with gcol2:
        top_n = st.number_input("Rows", min_value=10, max_value=1000, value=100, step=10)

    ranked = scores[scores["split"] != "train"]
    if group:
        ranked = ranked[ranked["split"] == group]
    ranked = ranked[ranked["risk"] >= threshold].sort_values("risk", ascending=False)
    st.caption(f"{len(ranked):,} wallets at or above {threshold:.2f}")
    table = ranked.head(int(top_n))[["address", "risk", "class", "split", "time_step"]].copy()
    table["risk"] = (table["risk"] * 100).round(1)
    st.dataframe(table.rename(columns={"risk": "risk %", "class": "true label"}),
                 width="stretch", hide_index=True)

    _section_label("WALLET LOOKUP")
    address = st.text_input("Wallet address", value=table["address"].iloc[0] if len(table) else "").strip()
    if not address:
        return
    match = scores[scores["address"] == address]
    if match.empty:
        st.warning("Address not found in the Elliptic++ wallets dataset.")
        return
    wallet = match.iloc[0]

    flagged = wallet["risk"] >= threshold
    l1, l2, l3 = st.columns(3)
    with l1:
        render_metric_card("RISK SCORE", f"{wallet['risk'] * 100:.1f}/100")
    with l2:
        render_metric_card("VERDICT", "SUSPICIOUS" if flagged else "NORMAL")
    with l3:
        render_metric_card("TRUE LABEL", f"{wallet['class'].upper()} ({wallet['split']})")

    labelled = scores[scores["class"] != "unknown"]
    medians = labelled.groupby("class")[feature_cols].median()
    comparison = pd.DataFrame({
        "feature": feature_cols,
        "this wallet": [wallet[c] for c in feature_cols],
        "typical licit": [medians.loc["licit", c] for c in feature_cols],
        "typical illicit": [medians.loc["illicit", c] for c in feature_cols],
    })
    _section_label("WALLET ACTIVITY VS TYPICAL WALLETS")
    st.dataframe(comparison, width="stretch", hide_index=True)
