"""
audit_wallet_feature_leakage.py
-------------------------------
Checks whether the Elliptic++ wallet features are *causal* — recomputed at each
time step from activity up to that step — or whole-lifetime aggregates that are
the same on every one of a wallet's time-step rows.

If a feature is constant across all of an address's rows, it is a single
lifetime aggregate: a training row from an early step then carries information
from the wallet's whole life, including the test period. The temporal split in
train_elliptic_plus.py still keeps train and test wallets disjoint, but such
features are not available in real time (when a wallet is first seen you cannot
know its future transactions), so metrics computed with them are optimistic for
a live deployment.

Run (from the project root; needs the raw wallets CSV in data/elliptic_plus/):
    python scripts/audit_wallet_feature_leakage.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ELLIPTIC_PLUS_DIR
from train_elliptic_plus import TIME_COLUMN, WALLET_TIME_FEATURES

# Columns the wallet model never uses as inputs
NON_FEATURES = ["address", "class", TIME_COLUMN] + WALLET_TIME_FEATURES


def main() -> None:
    csv = ELLIPTIC_PLUS_DIR / "wallets_features_classes_combined.csv"
    if not csv.exists():
        sys.exit(f"Raw wallets CSV not found: {csv}. See the README for how to download it.")

    print(f"Loading {csv.name} ...", flush=True)
    df = pd.read_csv(csv)
    df = df[df["class"] != 3]  # labelled wallets, as the model sees them
    print(f"Labelled rows: {len(df):,}   unique addresses: {df['address'].nunique():,}")

    steps = df.groupby("address")[TIME_COLUMN].nunique()
    multi = steps[steps >= 2].index
    multi_df = df[df["address"].isin(multi)]
    print(f"Addresses active in >=2 time steps: {len(multi):,}   (their rows: {len(multi_df):,})\n")
    if multi_df.empty:
        sys.exit("No address appears in more than one time step; cannot assess per-step variation.")

    feature_cols = [c for c in df.columns if c not in NON_FEATURES]
    # For each address, how many distinct values does each feature take across its steps?
    distinct = multi_df.groupby("address")[feature_cols].nunique()
    const_frac = (distinct == 1).mean().sort_values()  # 1.0 => always constant => lifetime aggregate

    varying = const_frac[const_frac < 1.0]
    print(f"Model-input features examined: {len(feature_cols)}")
    print(f"Features that EVER vary across a wallet's time steps: {len(varying)}")
    if varying.empty:
        print("=> Every feature is constant across an address's time steps: all wallet features\n"
              "   are whole-lifetime aggregates (non-causal). The per-time-step risk is therefore\n"
              "   identical across a wallet's steps, and the reported wallet metrics assume\n"
              "   information not available when a wallet is first seen.")
    else:
        print("\nLeast-constant features (fraction of multi-step addresses where constant):")
        print((varying * 100).round(1).to_string())


if __name__ == "__main__":
    main()
