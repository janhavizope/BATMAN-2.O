"""
score_transactions.py
---------------------
Uses the trained Elliptic++ transactions model to give every transaction a
risk score (predicted probability of being illicit), including those whose
class is unknown. Each transaction is tagged with the split it belonged to
during training (train / test / unknown).

Also stores the transaction flow graph (which transaction's outputs are spent
by which) from the original Elliptic dataset, which uses the same txIds.

Outputs (read by the Streamlit app):
    data/elliptic_plus/tx_scores.parquet
    data/elliptic_plus/tx_edges.parquet

Run (from the project root, after train_elliptic_plus.py txs):
    python scripts/score_transactions.py
"""

import joblib
import pandas as pd

from train_elliptic import MODEL_DIR, ROOT
from train_elliptic_plus import DATA_DIR, split

MODEL_FILE = MODEL_DIR / "elliptic_plus_txs_rf_model.joblib"
SCORES_FILE = DATA_DIR / "tx_scores.parquet"
EDGES_SOURCE = ROOT / "data" / "elliptic" / "elliptic_txs_edgelist.csv"
EDGES_FILE = DATA_DIR / "tx_edges.parquet"
INFO_COLUMNS = ["total_BTC", "fees", "size", "in_txs_degree", "out_txs_degree",
                "num_input_addresses", "num_output_addresses"]
CLASS_NAMES = {1: "illicit", 2: "licit", 3: "unknown"}


def main():
    saved = joblib.load(MODEL_FILE)
    model, features = saved["model"], saved["features"]

    # Same merge and split as train_elliptic_plus.load_txs, keeping txId
    df = pd.read_csv(DATA_DIR / "txs_features.csv").merge(
        pd.read_csv(DATA_DIR / "txs_classes.csv"), on="txId"
    )
    labelled = df[df["class"] != 3]
    X = labelled.drop(columns=["txId", "class"]).fillna(0)
    y = (labelled["class"] == 1).astype(int)
    X_train, X_test, _, _ = split(X, y, None)

    df["split"] = "unknown"
    df.loc[X_train.index, "split"] = "train"
    df.loc[X_test.index, "split"] = "test"

    print("Scoring all transactions ...")
    df["risk"] = model.predict_proba(df[features].fillna(0))[:, 1]

    out = df[["txId", "Time step", "class", "risk", "split"] + INFO_COLUMNS].rename(
        columns={"Time step": "time_step"}
    )
    out["class"] = out["class"].map(CLASS_NAMES)
    out.to_parquet(SCORES_FILE, index=False, compression="zstd")
    print(f"Scored {len(out)} transactions -> {SCORES_FILE}")
    print(out.groupby("split")["risk"].describe())

    edges = pd.read_csv(EDGES_SOURCE)
    edges.columns = ["source", "target"]
    edges.to_parquet(EDGES_FILE, index=False, compression="zstd")
    print(f"Saved {len(edges)} transaction flow edges -> {EDGES_FILE}")


if __name__ == "__main__":
    main()
