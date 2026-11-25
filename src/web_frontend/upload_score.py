"""
Upload & Score Page
-------------------
Score your own transaction data with the unsupervised pipeline:
- Upload a CSV or JSON file of transactions
- Validate it (src/core/backend/validation/validator.py) and show the data
  quality report
- Engineer wallet features, run Isolation Forest + HDBSCAN, and rank wallets
  by risk with the reasons behind each score
- Download the results

No labels are needed and nothing leaves the machine.
"""

import json
import os
import tempfile

import pandas as pd
import streamlit as st

from src.config import EVALUATION_DIR
from src.core.backend.ingestion.ingestor import TransactionIngestor
from src.core.backend.pipeline import score_wallets
from src.web_frontend.components import render_header, render_metric_card

TEMPLATE = pd.DataFrame([
    {
        "txid": "a" * 64, "timestamp": "2026-01-05T10:00:00",
        "input_addresses": "walletA", "output_addresses": "walletB|walletC",
        "input_amounts": "1.0", "output_amounts": "0.6|0.3999", "fee": 0.0001,
    },
    {
        "txid": "b" * 64, "timestamp": "2026-01-05T10:20:00",
        "input_addresses": "walletB", "output_addresses": "walletD",
        "input_amounts": "0.6", "output_amounts": "0.5999", "fee": 0.0001,
    },
])

RESULT_COLUMNS = ["risk_score", "risk_level", "top_reasons", "total_tx_count",
                  "unique_counterparties", "total_sent", "total_received", "anomaly_score", "cluster_id"]


UNSUPERVISED_METRICS = EVALUATION_DIR / "elliptic_plus_unsupervised_metrics.json"


def _reliability_note() -> str:
    """How well this score did on real labelled wallets, from the committed evaluation."""
    note = ("This score is experimental. Use it to choose what to look at first, not as evidence: "
            "HIGH and CRITICAL mean 'unusual compared with the rest of this file', not 'illicit'.")
    try:
        metrics = json.loads(UNSUPERVISED_METRICS.read_text())
        unsupervised = metrics["test_wallets"]["unsupervised"]["roc_auc"]
        supervised = metrics["test_wallets"]["supervised_random_forest"]["roc_auc"]
    except (OSError, ValueError, KeyError):
        return note
    return (note + f" On real labelled Elliptic++ wallets it scored ROC-AUC {unsupervised:.2f} "
            f"(0.50 is random guessing), against {supervised:.2f} for the supervised wallet model.")


def _section_label(text: str) -> None:
    st.markdown(
        "<div style='color:#b33939; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; "
        f"font-weight:bold; letter-spacing:1px; margin:14px 0 8px 0;'>{text}</div>",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _ingest(content: bytes, suffix: str, require_hex_txid: bool):
    """Validate the uploaded bytes; the ingestor reads from a file path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(content)
        path = handle.name
    try:
        return TransactionIngestor(require_hex_txid=require_hex_txid).ingest(path)
    finally:
        os.remove(path)


@st.cache_data(show_spinner=False)
def _score(transactions: pd.DataFrame) -> pd.DataFrame:
    return score_wallets(transactions)


def _quality_report(report: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("ROWS", f"{report['total_records']:,}")
    with c2:
        render_metric_card("VALID", f"{report['valid_records']:,}")
    with c3:
        render_metric_card("INVALID", f"{report['invalid_records']:,}")
    with c4:
        render_metric_card("DUPLICATE TXIDS", f"{report['duplicate_txids']:,}")
    reasons = {k.replace("_", " "): v for k, v in report["invalid_reasons"].items() if v}
    if reasons:
        st.dataframe(pd.DataFrame({"issue": list(reasons), "rows": list(reasons.values())}),
                     width="stretch", hide_index=True)


def render_upload_score() -> None:
    render_header("UPLOAD & SCORE")
    st.caption(
        "Upload transactions to score their wallets with the unsupervised pipeline "
        "(Isolation Forest + HDBSCAN). Required columns: txid, timestamp, input_addresses, "
        "output_addresses, input_amounts, output_amounts. List fields use | or JSON arrays. "
        "fee and the network fields (src_ip, dst_ip, src_port, dst_port, asn, geo_country) are optional."
    )
    st.warning(_reliability_note())
    st.download_button("Download a template CSV", TEMPLATE.to_csv(index=False).encode("utf-8"),
                       file_name="batman_transactions_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Transaction file", type=["csv", "json"])
    lenient_txids = st.checkbox("Transaction ids are not 64-character hex (e.g. anonymised integer ids)")
    if uploaded is None:
        return

    try:
        transactions, report = _ingest(uploaded.getvalue(), os.path.splitext(uploaded.name)[1].lower(),
                                       not lenient_txids)
    except ValueError as error:
        st.error(f"The file could not be validated: {error}")
        return

    _section_label("DATA QUALITY")
    _quality_report(report)
    if transactions.empty:
        st.warning("No valid transactions to score.")
        return

    with st.spinner("Engineering features and scoring wallets ..."):
        results = _score(transactions)

    _section_label("WALLET RISK")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("WALLETS SCORED", f"{len(results):,}")
    with c2:
        render_metric_card("HIGH / CRITICAL", f"{results['risk_level'].isin(['HIGH', 'CRITICAL']).sum():,}")
    with c3:
        render_metric_card("HDBSCAN OUTLIERS", f"{(results['cluster_id'] == -1).sum():,}")
    st.caption("Risk is relative to the uploaded data: each signal is scaled between the lowest and "
               "highest wallet in this file. No labels are used.")

    table = results[RESULT_COLUMNS].reset_index()
    table["top_reasons"] = table["top_reasons"].map("; ".join)
    table["risk_score"] = table["risk_score"].round(1)
    st.dataframe(table.head(500), width="stretch", hide_index=True)
    st.download_button("Download all results (.csv)", table.to_csv(index=False).encode("utf-8"),
                       file_name="batman_wallet_scores.csv", mime="text/csv")

    _supervised_section(transactions)


def _supervised_section(transactions) -> None:
    """Score each uploaded transaction with the live-computable supervised model."""
    from src.core.backend.live.live_model import supervised_tx_risk, risk_level
    risk = supervised_tx_risk(transactions)
    if risk is None:
        return
    _section_label("SUPERVISED MODEL — TRANSACTION RISK")
    st.caption(
        "A supervised model (trained on real labelled Elliptic transactions, ROC-AUC 0.82 on held-out "
        "data) rates how likely each transaction is illicit, from its value, fee and input/output counts. "
        "It ranks transactions — higher is more suspicious. It is a learned second opinion alongside the "
        "unsupervised wallet score above, and reads best on real on-chain data (synthetic rows with no "
        "fees may read flat)."
    )
    out = pd.DataFrame({
        "txid": transactions["txid"].values,
        "supervised risk %": (risk.values * 100).round(1),
        "level": [risk_level(p) for p in risk.values],
        "n_inputs": transactions["input_addresses[]"].map(len).values,
        "n_outputs": transactions["output_addresses[]"].map(len).values,
    }).sort_values("supervised risk %", ascending=False)
    flagged = int((risk.values >= 0.5).sum())
    render_metric_card("TX RATED HIGH-RISK (>=50%)", f"{flagged:,}")
    st.dataframe(out.head(500), width="stretch", hide_index=True)
