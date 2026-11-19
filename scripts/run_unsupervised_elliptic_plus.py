"""
run_unsupervised_elliptic_plus.py
---------------------------------
Runs the unsupervised pipeline (wallet features -> Isolation Forest ->
HDBSCAN -> risk score) on the real Elliptic++ wallet-transaction graph
instead of synthetic data, then checks the label-free ranking against the
Elliptic++ wallet labels.

How Elliptic++ is turned into the pipeline's transaction format:
    - inputs / outputs  from AddrTx_edgelist.csv / TxAddr_edgelist.csv
    - amounts           Elliptic++ has no per-address amounts, so each
                        transaction's in_BTC_total / out_BTC_total is split
                        evenly over its input / output addresses
    - fee               txs_features.csv "fees"
    - timestamp         only the time step is known (about two weeks each):
                        step k is mapped to 14 * (k - 1) days after an
                        arbitrary origin. Second-level timing features
                        (bursts, rolling windows, pass-through gaps) would be
                        artifacts of that, so they are dropped
    - network fields    none (real on-chain data has no IP / ASN / country),
                        so the network component drops out of the score

HDBSCAN is fitted on a random sample of 100,000 wallets and the others are
assigned with approximate_predict (src/core/backend/pipeline.py); a full fit
on 822k wallets takes hours.

Outputs:
    data/elliptic_plus_unsupervised/wallet_scores.parquet      (git-ignored)
    outputs/evaluation/elliptic_plus_unsupervised_metrics.json

Run (from the project root; takes a while on 822k wallets):
    python scripts/run_unsupervised_elliptic_plus.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, ELLIPTIC_PLUS_DIR, EVALUATION_DIR
from src.core.backend.pipeline import score_wallets
from src.core.backend.validation.validator import TransactionValidator

OUTPUT_DIR = DATA_DIR / "elliptic_plus_unsupervised"
SCORES_FILE = OUTPUT_DIR / "wallet_scores.parquet"
METRICS_FILE = EVALUATION_DIR / "elliptic_plus_unsupervised_metrics.json"
SUPERVISED_SCORES = ELLIPTIC_PLUS_DIR / "wallet_scores.parquet"

ORIGIN = pd.Timestamp("2000-01-01")
COARSE_TIME_FEATURES = [
    "tx_per_hour", "avg_inter_arrival_time", "burst_count", "max_tx_in_window",
    "max_fan_in_window", "max_fan_out_window", "min_pass_through_gap", "WINDOW_SECONDS",
]
TOP_K = (100, 500, 1000, 5000)


def build_transactions(data_dir: Path = ELLIPTIC_PLUS_DIR) -> pd.DataFrame:
    """Elliptic++ transactions in the pipeline's input format (see module docstring)."""
    txs = pd.read_csv(data_dir / "txs_features.csv",
                      usecols=["txId", "Time step", "fees", "in_BTC_total", "out_BTC_total"])
    inputs = pd.read_csv(data_dir / "AddrTx_edgelist.csv").groupby("txId")["input_address"].agg(list)
    outputs = pd.read_csv(data_dir / "TxAddr_edgelist.csv").groupby("txId")["output_address"].agg(list)
    txs = txs.join(inputs, on="txId").join(outputs, on="txId").dropna(subset=["input_address", "output_address"])

    def split_evenly(total, addresses):
        return [float(total) / len(addresses)] * len(addresses)

    return pd.DataFrame({
        "txid": txs["txId"].astype(str),
        "timestamp": ORIGIN + pd.to_timedelta((txs["Time step"] - 1) * 14, unit="D"),
        "input_addresses[]": txs["input_address"],
        "output_addresses[]": txs["output_address"],
        "input_amounts[]": [split_evenly(t, a) for t, a in zip(txs["in_BTC_total"], txs["input_address"])],
        "output_amounts[]": [split_evenly(t, a) for t, a in zip(txs["out_BTC_total"], txs["output_address"])],
        "fee": txs["fees"],
    })


def wallet_labels(data_dir: Path = ELLIPTIC_PLUS_DIR) -> pd.Series:
    """1 if any of a wallet's rows is illicit, 0 if licit, missing if only unknown."""
    wallets = pd.read_csv(data_dir / "wallets_features_classes_combined.csv", usecols=["address", "class"])
    labelled = wallets[wallets["class"] != 3]
    return (labelled["class"] == 1).groupby(labelled["address"]).any().astype(int)


def ranking_metrics(risk: pd.Series, labels: pd.Series) -> dict:
    """ROC-AUC and precision of the top-k wallets, on wallets that have a label."""
    joined = pd.concat([risk.rename("risk"), labels.rename("y")], axis=1, join="inner")
    ranked = joined.sort_values("risk", ascending=False)["y"]
    return {
        "wallets": len(joined),
        "illicit_rate": float(joined["y"].mean()),
        "roc_auc": float(roc_auc_score(joined["y"], joined["risk"])) if joined["y"].nunique() == 2 else None,
        **{f"precision_at_{k}": float(ranked.head(k).mean()) for k in TOP_K if k <= len(ranked)},
    }


def main() -> None:
    print("Building Elliptic++ transactions ...", flush=True)
    raw = build_transactions()
    clean, report = TransactionValidator(require_hex_txid=False).validate(raw)
    print(f"Valid transactions: {report['valid_records']:,} / {report['total_records']:,}", flush=True)

    print("Scoring wallets (features, Isolation Forest, HDBSCAN, risk) ...", flush=True)
    results = score_wallets(clean, drop_features=COARSE_TIME_FEATURES)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = results.drop(columns=["top_reasons"]).assign(top_reasons=results["top_reasons"].map("; ".join))
    out.to_parquet(SCORES_FILE, compression="zstd")
    print(f"Scored {len(results):,} wallets -> {SCORES_FILE}", flush=True)

    labels = wallet_labels()
    metrics = {
        "description": "Label-free unsupervised risk score on Elliptic++, checked against wallet labels",
        "valid_transactions": report["valid_records"],
        "wallets_scored": len(results),
        "risk_levels": results["risk_level"].value_counts().to_dict(),
        "all_labelled_wallets": ranking_metrics(results["risk_score"], labels),
    }
    # Same held-out wallets the supervised wallet model is evaluated on
    if SUPERVISED_SCORES.exists():
        supervised = pd.read_parquet(SUPERVISED_SCORES, columns=["address", "risk", "split"])
        test = supervised[supervised["split"] == "test"].set_index("address")
        test_labels = labels.reindex(test.index).dropna()
        metrics["test_wallets"] = {
            "unsupervised": ranking_metrics(results["risk_score"], test_labels),
            "supervised_random_forest": ranking_metrics(test["risk"], test_labels),
        }

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))
    print(json.dumps(metrics, indent=2, default=str))
    print(f"Metrics -> {METRICS_FILE}")


if __name__ == "__main__":
    main()
