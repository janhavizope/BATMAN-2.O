"""
train_elliptic_plus.py
----------------------
Trains Random Forest classifiers on the Elliptic++ dataset
(https://github.com/git-disl/EllipticPlusPlus):

    txs      -> classifies transactions (txs_features.csv + txs_classes.csv)
    wallets  -> classifies wallet addresses (wallets_features_classes_combined.csv)

Steps (same as train_elliptic.py):
    1. Load data from data/elliptic_plus/
    2. Keep only labelled rows (class 1 = illicit, 2 = licit; 3 = unknown is dropped)
    3. Temporal train/test split: train on time steps 1-34, test on 35-49.
       Wallets appear once per time step, so they are split by the time step
       they first appear in: a wallet first seen in steps 1-34 is a training
       wallet (only its rows up to step 34 are used), a wallet first seen in
       step 35 or later is a test wallet. The model is never tested on a
       wallet it trained on, and never trains on data from the test period.
    4. Drop time-revealing features (the time step, and the wallet
       block-number features) so the model learns behaviour, not timing
    5. Train RandomForestClassifier (class_weight="balanced")
    6. Evaluate on the test period, including illicit F1 per time step
    7. Save model, metrics and graphs

Run (from the project root):
    python scripts/train_elliptic_plus.py            # both datasets
    python scripts/train_elliptic_plus.py txs        # transactions only
    python scripts/train_elliptic_plus.py wallets    # wallets only
"""

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from train_elliptic import (
    MODEL_DIR, RANDOM_STATE, TRAIN_LAST_STEP, evaluate, print_metrics, save_graphs,
)
from src.config import ELLIPTIC_PLUS_DIR, EVALUATION_DIR, GRAPHS_DIR

DATA_DIR = ELLIPTIC_PLUS_DIR
EVAL_DIR = EVALUATION_DIR
GRAPHS_ROOT = GRAPHS_DIR / "elliptic_plus"

TIME_COLUMN = "Time step"
# Absolute block numbers say *when* a wallet was active, not how it behaves
WALLET_TIME_FEATURES = [
    "first_block_appeared_in", "last_block_appeared_in",
    "first_sent_block", "first_received_block",
]


def load_txs():
    features = pd.read_csv(DATA_DIR / "txs_features.csv")
    classes = pd.read_csv(DATA_DIR / "txs_classes.csv")
    df = features.merge(classes, on="txId")
    df = df[df["class"] != 3]

    X = df.drop(columns=["txId", "class", TIME_COLUMN])
    y = (df["class"] == 1).astype(int)  # 1 = illicit, 0 = licit
    return X, y, None, df[TIME_COLUMN]


def load_wallets():
    df = pd.read_csv(DATA_DIR / "wallets_features_classes_combined.csv")
    df = df[df["class"] != 3]

    groups = df["address"]
    X = df.drop(columns=["address", "class", TIME_COLUMN] + WALLET_TIME_FEATURES, errors="ignore")
    y = (df["class"] == 1).astype(int)
    return X, y, groups, df[TIME_COLUMN]


def split(X, y, groups, time_steps):
    """Temporal split. With groups (wallets), each group goes to one side only."""
    train = time_steps <= TRAIN_LAST_STEP
    if groups is None:
        test = ~train
    else:
        first_step = time_steps.groupby(groups).transform("min")
        test = first_step > TRAIN_LAST_STEP
    train, test = train.to_numpy(), test.to_numpy()
    return X[train], X[test], y[train], y[test]


def train(name, loader):
    print(f"\n{'=' * 60}\n  Elliptic++ {name}\n{'=' * 60}")
    X, y, groups, time_steps = loader()
    X = X.fillna(0)
    print(f"Labelled rows: {len(X)}  |  features: {X.shape[1]}")
    print(f"Illicit: {y.sum()}  |  Licit: {(y == 0).sum()}")
    if groups is not None:
        print(f"Unique wallets: {groups.nunique()}")

    X_train, X_test, y_train, y_test = split(X, y, groups, time_steps)
    print(f"Train: {len(X_train)} rows (steps 1-{TRAIN_LAST_STEP})  |  "
          f"Test: {len(X_test)} rows (steps {TRAIN_LAST_STEP + 1}+)\n")

    print("Training Random Forest ...")
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    metrics, y_pred, y_prob = evaluate(model, X_test, y_test, time_steps.loc[X_test.index])
    print_metrics(metrics)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top = importances.sort_values(ascending=False).head(10)
    print("\nTop 10 features:")
    print(top.to_string())

    MODEL_DIR.mkdir(exist_ok=True)
    model_file = MODEL_DIR / f"elliptic_plus_{name}_rf_model.joblib"
    joblib.dump({"model": model, "features": list(X.columns)}, model_file)
    print(f"\nModel saved to {model_file}")

    metrics_file = EVAL_DIR / f"elliptic_plus_{name}_metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(
        {"train_rows": len(X_train), **metrics, "top_features": top.to_dict()}, indent=2
    ))
    print(f"Metrics saved to {metrics_file}")

    save_graphs(y, y_test, y_pred, y_prob, np.asarray(metrics["confusion_matrix"]),
                metrics["classification_report"], importances, GRAPHS_ROOT / name)


DATASETS = {"txs": load_txs, "wallets": load_wallets}


def main():
    names = sys.argv[1:] or list(DATASETS)
    for name in names:
        train(name, DATASETS[name])


if __name__ == "__main__":
    main()
