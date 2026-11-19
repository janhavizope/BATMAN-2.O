"""
prepare_elliptic.py
-------------------
Derives the original Elliptic dataset files from Elliptic++, so the Kaggle
download (which needs an account) is optional.

Elliptic++ (https://github.com/git-disl/EllipticPlusPlus) extends Elliptic:
its txs_features.csv holds Elliptic's columns (txId, time step, 93 local
and 72 aggregate features) followed by 17 extra features, and its
txs_edgelist.csv is Elliptic's transaction-flow edge list. This script
writes them in Elliptic's layout:

    data/elliptic/elliptic_txs_features.csv   no header: txId, time step, 165 features
    data/elliptic/elliptic_txs_classes.csv    txId, class ("1" illicit, "2" licit, "unknown")
    data/elliptic/elliptic_txs_edgelist.csv   txId1, txId2

Existing files in data/elliptic/ are left alone.

Run (from the project root):
    python scripts/prepare_elliptic.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ELLIPTIC_DIR, ELLIPTIC_PLUS_DIR

ELLIPTIC_COLUMNS = (
    ["txId", "Time step"]
    + [f"Local_feature_{i}" for i in range(1, 94)]
    + [f"Aggregate_feature_{i}" for i in range(1, 73)]
)
CLASS_NAMES = {1: "1", 2: "2", 3: "unknown"}


def main() -> None:
    ELLIPTIC_DIR.mkdir(parents=True, exist_ok=True)

    features_out = ELLIPTIC_DIR / "elliptic_txs_features.csv"
    if not features_out.exists():
        features = pd.read_csv(ELLIPTIC_PLUS_DIR / "txs_features.csv", usecols=ELLIPTIC_COLUMNS)
        features[ELLIPTIC_COLUMNS].to_csv(features_out, header=False, index=False)
        print(f"Wrote {len(features):,} transactions -> {features_out}")

    classes_out = ELLIPTIC_DIR / "elliptic_txs_classes.csv"
    if not classes_out.exists():
        classes = pd.read_csv(ELLIPTIC_PLUS_DIR / "txs_classes.csv")
        classes["class"] = classes["class"].map(CLASS_NAMES)
        classes.to_csv(classes_out, index=False)
        print(f"Wrote {len(classes):,} labels -> {classes_out}")

    edges_out = ELLIPTIC_DIR / "elliptic_txs_edgelist.csv"
    if not edges_out.exists():
        edges = pd.read_csv(ELLIPTIC_PLUS_DIR / "txs_edgelist.csv")
        edges.columns = ["txId1", "txId2"]
        edges.to_csv(edges_out, index=False)
        print(f"Wrote {len(edges):,} edges -> {edges_out}")


if __name__ == "__main__":
    main()
