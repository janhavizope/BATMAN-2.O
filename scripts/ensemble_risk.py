"""
ensemble_risk.py
----------------
Combine the two engines into one verdict, weighting each by how reliable it is.

The supervised model is far more trustworthy (ROC-AUC ~0.92) than the label-free
unsupervised score (~0.76), so it must carry most of the weight. This script
measures a reliability-weighted blend

    combined = W_SUPERVISED * supervised + (1 - W_SUPERVISED) * unsupervised

on the held-out Elliptic++ test wallets, and reports the AUC at several weights
so the choice is grounded, not arbitrary.

Run:
    python scripts/ensemble_risk.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ELLIPTIC_PLUS_DIR
from scripts.score_wallets_domain_unsupervised import domain_risk

# Default production weight: the supervised model dominates the verdict.
# Measured finding: no blend beats supervised-alone (0.92); giving the
# unsupervised score real weight only lowers it. So where a supervised score
# exists it should essentially decide; the blend keeps a small, explainable
# nudge from the unsupervised second opinion.
W_SUPERVISED = 0.9


def combined_risk(supervised_0_1: pd.Series, unsupervised_0_1: pd.Series,
                  w_supervised: float = W_SUPERVISED) -> pd.Series:
    """Reliability-weighted blend of the two engines, each scaled to [0, 1]."""
    return w_supervised * supervised_0_1 + (1 - w_supervised) * unsupervised_0_1


def main() -> int:
    # Unsupervised domain score per address (from the raw feature CSV)
    raw = pd.read_csv(ELLIPTIC_PLUS_DIR / "wallets_features_classes_combined.csv")
    raw.columns = [c.strip().replace(" ", "_") for c in raw.columns]
    raw = raw.drop_duplicates("address").set_index("address")
    unsup = (domain_risk(raw) / 100.0)                      # 0..1

    # Supervised wallet risk per address
    wal = pd.read_parquet(ELLIPTIC_PLUS_DIR / "wallet_scores.parquet").drop_duplicates("address")
    wal = wal.set_index("address")
    df = wal[["class", "split", "risk"]].join(unsup.rename("unsup"), how="inner")
    df = df[df["class"].isin(["illicit", "licit"]) & (df["split"] == "test")].dropna()
    y = (df["class"] == "illicit").astype(int).values
    sup = df["risk"].clip(0, 1).values
    uns = df["unsup"].values
    print(f"held-out test wallets: {len(df):,}  (illicit: {int(y.sum()):,})")
    print(f"  supervised alone   ROC-AUC: {roc_auc_score(y, sup):.3f}")
    print(f"  unsupervised alone ROC-AUC: {roc_auc_score(y, uns):.3f}")
    print("  reliability-weighted blend:")
    best = (None, 0.0)
    for w in (0.5, 0.7, 0.85, 0.9, 0.95):
        c = w * sup + (1 - w) * uns
        a = roc_auc_score(y, c)
        flag = "  <- default" if abs(w - W_SUPERVISED) < 1e-9 else ""
        print(f"    W_supervised={w:.2f}  ROC-AUC {a:.3f}{flag}")
        if a > best[1]:
            best = (w, a)
    print(f"  best blend: W_supervised={best[0]:.2f} (ROC-AUC {best[1]:.3f})")
    print("\n  Takeaway: the supervised model must dominate the verdict. The")
    print("  unsupervised score is a lower-weight, explainable second opinion —")
    print("  most useful where no supervised label exists (uploads / live feed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
