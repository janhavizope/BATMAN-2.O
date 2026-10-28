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
    3. Stratified 80/20 train/test split. Wallets appear once per time step,
       so they are split by address: every row of a wallet lands on the same
       side, and the model is never tested on a wallet it trained on.
    4. Train RandomForestClassifier (class_weight="balanced")
    5. Evaluate on the held-out 20%
    6. Save model, metrics and graphs

Run (from the project root):
    python scripts/train_elliptic_plus.py            # both datasets
    python scripts/train_elliptic_plus.py txs        # transactions only
    python scripts/train_elliptic_plus.py wallets    # wallets only
"""

import json
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from train_elliptic import MODEL_DIR, RANDOM_STATE, ROOT, TEST_SIZE, save_graphs

DATA_DIR = ROOT / "data" / "elliptic_plus"
EVAL_DIR = ROOT / "outputs" / "evaluation"
GRAPHS_ROOT = ROOT / "outputs" / "graphs" / "elliptic_plus"


def load_txs():
    features = pd.read_csv(DATA_DIR / "txs_features.csv")
    classes = pd.read_csv(DATA_DIR / "txs_classes.csv")
    df = features.merge(classes, on="txId")
    df = df[df["class"] != 3]

    X = df.drop(columns=["txId", "class"])
    y = (df["class"] == 1).astype(int)  # 1 = illicit, 0 = licit
    return X, y, None


def load_wallets():
    df = pd.read_csv(DATA_DIR / "wallets_features_classes_combined.csv")
    df = df[df["class"] != 3]

    groups = df["address"]
    X = df.drop(columns=["address", "class"])
    y = (df["class"] == 1).astype(int)
    return X, y, groups


def split(X, y, groups):
    if groups is None:
        return train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
    # 5 folds -> one fold is 20% of the wallets; use it as the test set
    cv = StratifiedGroupKFold(n_splits=round(1 / TEST_SIZE), shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(cv.split(X, y, groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def train(name, loader):
    print(f"\n{'=' * 60}\n  Elliptic++ {name}\n{'=' * 60}")
    X, y, groups = loader()
    X = X.fillna(0)
    print(f"Labelled rows: {len(X)}  |  features: {X.shape[1]}")
    print(f"Illicit: {y.sum()}  |  Licit: {(y == 0).sum()}")
    if groups is not None:
        print(f"Unique wallets: {groups.nunique()}")

    X_train, X_test, y_train, y_test = split(X, y, groups)
    print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows "
          f"({len(X_test) / len(X):.0%})\n")

    print("Training Random Forest ...")
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["licit", "illicit"], output_dict=True
    )

    print("\n=== Results on the 20% test set ===")
    print(classification_report(y_test, y_pred, target_names=["licit", "illicit"]))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print("Confusion matrix (rows = actual, cols = predicted) [licit, illicit]:")
    print(cm)

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
    metrics_file.write_text(json.dumps({
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "top_features": top.to_dict(),
    }, indent=2))
    print(f"Metrics saved to {metrics_file}")

    save_graphs(y, y_test, y_pred, y_prob, cm, report, importances, GRAPHS_ROOT / name)


DATASETS = {"txs": load_txs, "wallets": load_wallets}


def main():
    names = sys.argv[1:] or list(DATASETS)
    for name in names:
        train(name, DATASETS[name])


if __name__ == "__main__":
    main()
