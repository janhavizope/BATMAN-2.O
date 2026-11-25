"""
train_live_tx_model.py
Train and evaluate the live-computable supervised transaction model.

It learns from only the features a raw live/uploaded transaction has (total
value, fee, number of input addresses, number of output addresses), so it can
run on the live feed and on uploads — unlike the full 166-feature Elliptic
model, which cannot. Trains on the temporal train split and reports held-out
test accuracy, then saves the model for the app to use.

Run:
    python scripts/train_live_tx_model.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ELLIPTIC_PLUS_DIR
from src.core.backend.live.live_model import FEATURES, MODEL_FILE, train_model


def main() -> int:
    frame = pd.read_parquet(ELLIPTIC_PLUS_DIR / "tx_scores.parquet")
    labelled = frame[frame["class"].isin(["illicit", "licit"])].copy()
    labelled["y"] = (labelled["class"] == "illicit").astype(int)
    test = labelled[labelled["split"] == "test"]

    print(f"Training live transaction model on features: {FEATURES}")
    model = train_model()
    proba = model.predict_proba(test[FEATURES].fillna(0.0))[:, 1]
    y = test["y"].values
    auc = roc_auc_score(y, proba)
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    print(f"\nHeld-out test ({len(test):,} transactions, {int(y.sum()):,} illicit):")
    print(f"  ROC-AUC        : {auc:.3f}   (full 166-feature model: 0.93)")
    print(f"  precision/recall at 0.50: {prec:.0%} / {rec:.0%}")
    print(f"\nSaved model -> {MODEL_FILE}")
    print("This model runs on live and uploaded transactions; the full model cannot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
