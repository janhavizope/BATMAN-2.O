"""
derive_wallet_risk_from_tx.py
-----------------------------
Score each wallet from the risk of the transactions it took part in, using the
transaction model (illicit F1 ~0.79) instead of the much weaker wallet model
(~0.24), and compare the two rankings on the held-out test wallets.

A wallet is linked to a transaction when it is one of its inputs (it sent) or
one of its outputs (it received), from the Elliptic++ AddrTx / TxAddr edge
lists. Each wallet's derived risk is an aggregate of its transactions' model
risk: the maximum (a wallet touching one near-certain illicit transaction is
suspicious) and the mean.

This is a hindsight comparison: it aggregates all of a wallet's transactions,
the same whole-life basis the wallet model's features use, so the two rankings
are measured on equal footing. A causal, up-to-each-time-step version is a
separate follow-up.

Reads (from data/elliptic_plus/):
    tx_scores.parquet       txId, risk, split, class
    AddrTx_edgelist.csv     input_address, txId   (wallet sent)
    TxAddr_edgelist.csv     txId, output_address  (wallet received)
    wallet_scores.parquet   address, risk, class, split  (supervised wallet model)
Writes:
    data/elliptic_plus/wallet_tx_derived_scores.parquet
    outputs/evaluation/wallet_risk_from_tx_metrics.json

Run (from the project root, after score_transactions.py and score_wallets.py):
    python scripts/derive_wallet_risk_from_tx.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ELLIPTIC_PLUS_DIR as D, EVALUATION_DIR

DERIVED_FILE = D / "wallet_tx_derived_scores.parquet"
METRICS_FILE = EVALUATION_DIR / "wallet_risk_from_tx_metrics.json"
TOP_K = (100, 500, 1000, 5000)


def wallet_tx_risk() -> pd.DataFrame:
    """Per wallet: max and mean risk over the transactions it sent or received."""
    tx = pd.read_parquet(D / "tx_scores.parquet", columns=["txId", "risk"]).set_index("txId")["risk"]
    sent = pd.read_csv(D / "AddrTx_edgelist.csv").rename(columns={"input_address": "address"})
    received = pd.read_csv(D / "TxAddr_edgelist.csv").rename(columns={"output_address": "address"})
    links = pd.concat([sent[["address", "txId"]], received[["address", "txId"]]], ignore_index=True)
    links = links.drop_duplicates(["address", "txId"])
    links["risk"] = links["txId"].map(tx)
    links = links.dropna(subset=["risk"])
    grouped = links.groupby("address")["risk"]
    return pd.DataFrame({"tx_max": grouped.max(), "tx_mean": grouped.mean(), "n_txs": grouped.size()})


def ranking_metrics(score: pd.Series, labels: pd.Series) -> dict:
    """ROC-AUC and precision of the top-k wallets, on wallets that have a label."""
    joined = pd.concat([score.rename("s"), labels.rename("y")], axis=1, join="inner").dropna()
    ranked = joined.sort_values("s", ascending=False)["y"]
    return {
        "wallets": int(len(joined)),
        "roc_auc": float(roc_auc_score(joined["y"], joined["s"])) if joined["y"].nunique() == 2 else None,
        **{f"precision_at_{k}": float(ranked.head(k).mean()) for k in TOP_K if k <= len(ranked)},
    }


def main() -> None:
    for name in ("tx_scores.parquet", "wallet_scores.parquet", "AddrTx_edgelist.csv", "TxAddr_edgelist.csv"):
        if not (D / name).exists():
            sys.exit(f"Missing input: {D / name}. Run the scoring scripts / place the edge lists first.")

    print("Aggregating transaction risk per wallet ...", flush=True)
    derived = wallet_tx_risk()
    DERIVED_FILE.parent.mkdir(parents=True, exist_ok=True)
    derived.reset_index().to_parquet(DERIVED_FILE, index=False, compression="zstd")
    print(f"Derived risk for {len(derived):,} wallets -> {DERIVED_FILE}")

    wallets = pd.read_parquet(D / "wallet_scores.parquet", columns=["address", "risk", "class", "split"])
    test = wallets[wallets["split"] == "test"].set_index("address")
    labels = test["class"].map({"illicit": 1, "licit": 0})  # unknown -> NaN, dropped

    metrics = {
        "description": "Wallet risk derived from the transaction model, vs the supervised wallet model, on test wallets",
        "test_wallets_with_label": int(labels.notna().sum()),
        "derived_tx_max": ranking_metrics(derived["tx_max"], labels),
        "derived_tx_mean": ranking_metrics(derived["tx_mean"], labels),
        "supervised_wallet_model": ranking_metrics(test["risk"], labels),
        "coverage": {
            "test_wallets": int(len(test)),
            "with_a_transaction_link": int(test.index.isin(derived.index).sum()),
        },
    }
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Metrics -> {METRICS_FILE}")


if __name__ == "__main__":
    main()
