"""
Wallet Scanner Page
-------------------
Risk scores for every wallet in the Elliptic++ dataset, produced by the
trained wallets Random Forest (scripts/score_wallets.py):
- 4 Metric Cards: WALLETS SCANNED, HIGH RISK (UNLABELLED), TEST-SET WALLETS, THRESHOLD
- Risk threshold slider
- Ranked table of the highest-risk wallets, filterable by group
- Wallet lookup: risk score, label and key feature values compared with
  typical licit / illicit wallets
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from src.web_frontend.components import render_header, render_metric_card

ROOT = Path(__file__).resolve().parent.parent.parent
SCORES_FILE = ROOT / "data" / "elliptic_plus" / "wallet_scores.parquet"

GROUPS = {
    "Unlabelled wallets (new predictions)": "unknown",
    "Test-set wallets (verify against true label)": "test",
    "All wallets": None,
}


@st.cache_data
def load_scores() -> pd.DataFrame:
    return pd.read_parquet(SCORES_FILE)


def _section_label(text: str) -> None:
    st.markdown(
        "<div style='color:#b33939; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; "
        f"font-weight:bold; letter-spacing:1px; margin:14px 0 8px 0;'>{text}</div>",
        unsafe_allow_html=True,
    )


def render_wallet_scanner() -> None:
    render_header("WALLET SCANNER")

    if not SCORES_FILE.exists():
        st.info("No wallet scores found. Run scripts/train_elliptic_plus.py wallets, "
                "then scripts/score_wallets.py.")
        return

    scores = load_scores()
    feature_cols = [c for c in scores.columns if c not in ("address", "time_step", "class", "risk", "split")]

    threshold = st.slider("Risk threshold (flag wallets at or above)", 0.05, 0.95, 0.50, 0.05)

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

    # --- Ranked table ----------------------------------------------------------
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
    st.dataframe(
        table.rename(columns={"risk": "risk %", "class": "true label"}),
        width="stretch",
        hide_index=True,
    )

    # --- Wallet lookup ---------------------------------------------------------
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
    _section_label("TOP MODEL FEATURES AND WALLET ACTIVITY VS TYPICAL WALLETS")
    st.dataframe(comparison, width="stretch", hide_index=True)
