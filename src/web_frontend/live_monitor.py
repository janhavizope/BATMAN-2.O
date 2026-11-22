"""
Live Monitor Page
-----------------
Monitor the live Bitcoin network. Each cycle pulls recent transactions from a
public mempool.space / Esplora API, maps them to the BATMAN schema, runs the
forensic analysis (live_risk) on each one, and stores the result. The page
shows the rolling analysis and can run a cycle on demand or auto-refresh.

The live score is the explainable forensic analysis — structural signals that
are meaningful in Bitcoin forensics (mixer/CoinJoin signatures, consolidation
sweeps, peeling chains, dust, watchlist hits) — not a trained model. Each flag
is an indicator of a pattern associated with illicit use, not proof of it.
"""

import time

import pandas as pd
import streamlit as st

from src.config import LIVE_DB
from src.core.backend.live.live_risk import SIGNALS
from src.core.backend.live.live_store import LiveStore
from src.core.backend.live.monitor import FEED_ERRORS, build_zmq_fetch, run_cycle
from src.core.backend.live.watchlist import load_watchlist
from src.web_frontend.components import render_header, render_metric_card

LEVEL_COLORS = {"CRITICAL": "#ff4d4d", "HIGH": "#ff8c42", "MEDIUM": "#e0c04d", "LOW": "#4d8b4d"}
WALLET_COLUMNS = ["address", "risk_score", "risk_level", "max_risk_score",
                  "times_scored", "total_tx_count", "top_reasons", "last_seen"]
TX_COLUMNS = ["txid", "risk_level", "risk_score", "n_inputs", "n_outputs",
              "total_btc", "fee_btc", "reasons", "fetched_at"]


@st.cache_resource
def _store() -> LiveStore:
    return LiveStore(LIVE_DB)


def _section(text: str) -> None:
    st.markdown(
        "<div style='color:#b33939; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; "
        f"font-weight:bold; letter-spacing:1px; margin:16px 0 8px 0;'>{text}</div>",
        unsafe_allow_html=True,
    )


def _live_badge(watchlist_size: int) -> None:
    extra = f" · watchlist: {watchlist_size} address(es)" if watchlist_size else " · watchlist: empty"
    st.markdown(
        "<div style=\"background-color:#0a1a0a; border:1px solid #153a15; "
        "border-left:3px solid #22aa55; border-radius:3px; padding:8px 14px; "
        "font-size:11px; font-family:'JetBrains Mono', Consolas, monospace; "
        f"color:#22aa55; margin-bottom:12px; letter-spacing:0.5px;\">"
        "<strong>[LIVE FEED]</strong> &mdash; real-time Bitcoin transactions from mempool.space, "
        f"analysed by the forensic engine (explainable signals){extra}."
        "</div>",
        unsafe_allow_html=True,
    )


def _level_bar(breakdown: dict) -> None:
    total = sum(breakdown.values())
    if total == 0:
        return
    segments = "".join(
        f"<div style='flex:{max(breakdown[lvl], 0)}; background:{LEVEL_COLORS[lvl]}; height:22px;'></div>"
        for lvl in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] if breakdown[lvl] > 0
    )
    legend = " &nbsp; ".join(
        f"<span style='color:{LEVEL_COLORS[lvl]};'>&#9632; {lvl} {breakdown[lvl]}</span>"
        for lvl in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    )
    st.markdown(
        f"<div style='display:flex; border-radius:3px; overflow:hidden; margin-bottom:6px;'>{segments}</div>"
        f"<div style='font-family:\"JetBrains Mono\", monospace; font-size:11px;'>{legend}</div>",
        unsafe_allow_html=True,
    )


def _signal_breakdown(txs: pd.DataFrame) -> pd.DataFrame:
    """How many transactions triggered each forensic signal, most common first."""
    reasons = txs["reasons"].fillna("")
    rows = [{"signal": s.name, "pattern": s.reason, "transactions": int(reasons.str.contains(
        s.reason, regex=False).sum())} for s in SIGNALS]
    frame = pd.DataFrame(rows)
    return frame[frame["transactions"] > 0].sort_values("transactions", ascending=False)


def render_live_monitor() -> None:
    render_header("LIVE MONITOR")
    watchlist, _ = load_watchlist()
    _live_badge(len(watchlist))
    st.caption(
        "Pulls recent transactions from the live Bitcoin mempool and analyses each one for "
        "structural risk patterns. Flags are explainable indicators (mixer signatures, "
        "consolidation sweeps, peeling chains, dust, watchlist hits), not a verdict of illicit activity. "
        "Nothing but the public API is contacted; analysis stays on this machine."
    )

    store = _store()

    c0, c1, c2, c3 = st.columns([1.3, 1.0, 1.0, 1.3])
    with c0:
        source = st.radio("Feed source", ["Public API", "My node (ZMQ)"], horizontal=False,
                          help="Public API: mempool.space. My node: your own Bitcoin Core over ZMQ.")
    with c1:
        max_tx = st.number_input("Transactions per cycle", min_value=5, max_value=100, value=25, step=5)
    with c2:
        auto = st.toggle("Auto-refresh", value=False, help="Run a cycle automatically on a timer.")
    with c3:
        interval = st.slider("Auto-refresh every (seconds)", 10, 120, 30, disabled=not auto)

    if st.button("Fetch & analyse now", type="primary") or auto:
        fetch_fn = None
        if source == "My node (ZMQ)":
            try:
                fetch_fn = build_zmq_fetch()
            except FEED_ERRORS as error:
                st.error(f"Could not connect to your node: {error}. Falling back to the public API "
                         "is not automatic — start bitcoind with ZMQ enabled (see docs/bitcoin.conf.sample).")
                st.stop()
        with st.spinner("Fetching live transactions and running forensic analysis ..."):
            try:
                result = run_cycle(store, max_tx=int(max_tx), watchlist=watchlist, fetch_fn=fetch_fn)
            except FEED_ERRORS as error:
                st.error(f"Could not reach the live feed: {error}")
                result = None
        if result is not None:
            if result.error:
                st.error(f"Feed error: {result.error}")
            else:
                st.success(
                    f"Cycle complete — fetched {result.fetched} transactions, {result.valid} valid, "
                    f"{result.wallets} wallets, {result.flagged_tx} flagged (HIGH/CRITICAL)."
                )

    summary = store.summary()
    if summary["total_transactions"] == 0:
        st.info("No data yet. Click **Fetch & analyse now** to pull the first batch of live transactions.")
        return

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("TRANSACTIONS SEEN", f"{summary['total_transactions']:,}")
    with m2:
        render_metric_card("FLAGGED (HIGH/CRIT)", f"{summary['flagged_transactions']:,}")
    with m3:
        render_metric_card("WALLETS TRACKED", f"{summary['total_wallets']:,}")
    with m4:
        render_metric_card("CYCLES RUN", f"{summary['cycles']:,}")

    _section("RISK DISTRIBUTION (TRANSACTIONS)")
    _level_bar(store.level_breakdown())

    txs = store.transactions(limit=2000)
    _section("PATTERNS DETECTED")
    signals = _signal_breakdown(txs)
    if signals.empty:
        st.caption("No risk patterns in the traffic seen so far — ordinary payments only.")
    else:
        st.dataframe(signals, width="stretch", hide_index=True)

    _section("FLAGGED TRANSACTIONS (HIGH / CRITICAL)")
    flagged = store.transactions(limit=200, min_level="HIGH")
    if flagged.empty:
        st.caption("Nothing flagged yet. Most live traffic is ordinary; keep the feed running to catch "
                   "mixers, large sweeps, and watchlist hits as they appear.")
    else:
        st.dataframe(_round_tx(flagged)[TX_COLUMNS], width="stretch", hide_index=True)

    _section("HIGHEST-RISK TRANSACTIONS (ALL LEVELS)")
    st.dataframe(_round_tx(txs.head(50))[TX_COLUMNS], width="stretch", hide_index=True)

    _section("RISKIEST WALLETS")
    wallets = store.wallets(limit=300)
    if not wallets.empty:
        table = wallets[WALLET_COLUMNS].copy()
        table["risk_score"] = table["risk_score"].round(1)
        table["max_risk_score"] = table["max_risk_score"].round(1)
        st.dataframe(table, width="stretch", hide_index=True)

    if auto:
        time.sleep(float(interval))
        st.rerun()


def _round_tx(txs: pd.DataFrame) -> pd.DataFrame:
    out = txs.copy()
    out["total_btc"] = out["total_btc"].round(6)
    out["fee_btc"] = out["fee_btc"].round(8)
    out["risk_score"] = out["risk_score"].round(1)
    return out
