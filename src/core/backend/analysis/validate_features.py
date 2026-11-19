"""Diagnose which behavioral features actually separate suspicious wallets.

This is a read-only diagnostic script used to compare wallet-level feature values
against transaction-derived ground truth before re-weighting a risk model. It does
not modify any scoring logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

FEATURE_COLUMNS = [
    "degree_centrality",
    "betweenness_centrality",
    "avg_inter_arrival_time",
    "burst_count",
    "avg_amount",
    "total_tx_count",
    "unique_ip_count",
    "unique_asn_count",
]


def get_data_dir() -> Path:
    """Return the dev-data directory containing parquet inputs."""
    from src.config import DEV_DATA_DIR
    return DEV_DATA_DIR


def normalize_wallet_column(df: pd.DataFrame, expected_name: str = "wallet_address") -> pd.DataFrame:
    """Ensure the wallet identifier is named consistently across parquet files."""
    result = df.copy()

    if expected_name in result.columns:
        return result

    if result.index.name == expected_name:
        return result.reset_index()

    for candidate in ["wallet", "wallet_id", "entity", "entity_id", "address"]:
        if candidate in result.columns:
            return result.rename(columns={candidate: expected_name})

    raise ValueError(
        f"Could not locate a '{expected_name}' column or index in the provided frame. "
        f"Columns found: {list(result.columns)}"
    )


def build_ground_truth(transactions_df: pd.DataFrame) -> pd.Series:
    """Create one wallet-level label from any suspicious raw transaction."""
    raw = normalize_wallet_column(transactions_df, "wallet_address").copy()

    if "is_suspicious" not in raw.columns:
        raise ValueError("transactions.parquet must contain an 'is_suspicious' column.")

    raw["is_suspicious_numeric"] = pd.to_numeric(raw["is_suspicious"], errors="coerce").fillna(0)

    wallet_truth = (
        raw.groupby("wallet_address", dropna=False)["is_suspicious_numeric"]
        .max()
        .astype(int)
        .rename("actually_suspicious")
    )
    return wallet_truth


def summarize_feature_signal(model_df: pd.DataFrame, feature: str) -> dict:
    """Compute the mean split and point-biserial association for one feature."""
    required = ["wallet_address", "actually_suspicious", feature]
    missing = [column for column in required if column not in model_df.columns]
    if missing:
        raise KeyError(f"Missing required columns for '{feature}': {missing}")

    analysis_df = model_df[["wallet_address", "actually_suspicious", feature]].dropna()
    if analysis_df.empty:
        return {
            "feature": feature,
            "mean_suspicious": np.nan,
            "mean_normal": np.nan,
            "r": np.nan,
            "p_value": np.nan,
            "meaningful": False,
        }

    y = analysis_df["actually_suspicious"].astype(int)
    x = pd.to_numeric(analysis_df[feature], errors="coerce")
    valid = pd.concat([x, y], axis=1).dropna()

    if valid.empty:
        return {
            "feature": feature,
            "mean_suspicious": np.nan,
            "mean_normal": np.nan,
            "r": np.nan,
            "p_value": np.nan,
            "meaningful": False,
        }

    x_valid = valid[feature].astype(float)
    y_valid = valid["actually_suspicious"].astype(int)

    suspicious_mask = y_valid == 1
    normal_mask = y_valid == 0

    mean_suspicious = float(x_valid[suspicious_mask].mean()) if suspicious_mask.any() else np.nan
    mean_normal = float(x_valid[normal_mask].mean()) if normal_mask.any() else np.nan

    r_value, p_value = pointbiserialr(x_valid, y_valid)

    return {
        "feature": feature,
        "mean_suspicious": mean_suspicious,
        "mean_normal": mean_normal,
        "r": float(r_value) if pd.notna(r_value) else np.nan,
        "p_value": float(p_value) if pd.notna(p_value) else np.nan,
        "meaningful": bool((p_value < 0.05) and (abs(r_value) > 0.1)),
    }


def print_feature_table(results_df: pd.DataFrame) -> None:
    """Print a readable summary of means, correlations, and significance."""
    display_df = results_df.copy()
    display_df["mean_suspicious"] = display_df["mean_suspicious"].map(
        lambda v: "nan" if pd.isna(v) else f"{v:.6f}"
    )
    display_df["mean_normal"] = display_df["mean_normal"].map(
        lambda v: "nan" if pd.isna(v) else f"{v:.6f}"
    )
    display_df["r"] = display_df["r"].map(lambda v: "nan" if pd.isna(v) else f"{v:.6f}")
    display_df["p_value"] = display_df["p_value"].map(
        lambda v: "nan" if pd.isna(v) else f"{v:.3e}"
    )
    display_df["meaningful"] = display_df["meaningful"].map({True: "YES", False: "NO"})

    print("\nFeature vs wallet-label summary")
    print("=" * 120)
    print(
        display_df[
            ["feature", "mean_suspicious", "mean_normal", "r", "p_value", "meaningful"]
        ].to_string(index=False)
    )


def print_ranked_features(results_df: pd.DataFrame) -> None:
    """Print the features sorted by the absolute value of their correlation."""
    ranked = results_df.copy()
    ranked["abs_r"] = ranked["r"].abs()
    ranked = ranked.sort_values(["abs_r", "p_value"], ascending=[False, True]).reset_index(drop=True)

    print("\nFeatures ranked by |r| (strongest signal first)")
    print("=" * 80)
    print(
        ranked[["feature", "r", "p_value", "abs_r"]]
        .rename(columns={"feature": "feature", "r": "r", "p_value": "p_value", "abs_r": "|r|"})
        .to_string(index=False)
    )


def print_pattern_breakdown(model_df: pd.DataFrame, transactions_df: pd.DataFrame) -> None:
    """If available, show the dominant pattern_type for suspicious wallets."""
    if "pattern_type" not in transactions_df.columns:
        print("\nNo 'pattern_type' column found in transactions.parquet; skipping pattern breakdown.")
        return

    suspicious_wallets = model_df.loc[model_df["actually_suspicious"], "wallet_address"].dropna().unique()
    if len(suspicious_wallets) == 0:
        print("\nNo suspicious wallets to summarize by pattern_type.")
        return

    tx_subset = normalize_wallet_column(transactions_df, "wallet_address").copy()
    tx_subset = tx_subset[tx_subset["wallet_address"].isin(suspicious_wallets)].copy()
    tx_subset = tx_subset.dropna(subset=["pattern_type"])

    if tx_subset.empty:
        print("\nNo non-null pattern_type values found among suspicious wallets.")
        return

    dominant_pattern = (
        tx_subset.groupby("wallet_address")["pattern_type"]
        .apply(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        .dropna()
        .rename("dominant_pattern_type")
    )

    if dominant_pattern.empty:
        print("\nNo dominant pattern_type could be inferred for suspicious wallets.")
        return

    summary = (
        dominant_pattern.value_counts()
        .rename_axis("dominant_pattern_type")
        .reset_index(name="wallet_count")
        .sort_values("wallet_count", ascending=False)
    )

    print("\nSuspicious-wallet pattern breakdown (dominant pattern per wallet)")
    print("=" * 80)
    print(summary.to_string(index=False))


def main() -> None:
    """Load the feature table, derive wallet truth from raw transactions, and print diagnostics."""
    data_dir = get_data_dir()
    features_path = data_dir / "features.parquet"
    transactions_path = data_dir / "transactions.parquet"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features parquet file: {features_path}")
    if not transactions_path.exists():
        raise FileNotFoundError(f"Missing transactions parquet file: {transactions_path}")

    print(f"Loading features from: {features_path}")
    print(f"Loading raw transactions from: {transactions_path}")

    features_df = normalize_wallet_column(pd.read_parquet(features_path), "wallet_address")
    transactions_df = normalize_wallet_column(pd.read_parquet(transactions_path), "wallet_address")

    wallet_truth = build_ground_truth(transactions_df)
    model_df = features_df.merge(wallet_truth.reset_index(), on="wallet_address", how="left")
    model_df["actually_suspicious"] = model_df["actually_suspicious"].fillna(0).astype(int)

    print(f"\nWallets in feature set: {len(model_df):,}")
    print(f"Wallets labeled suspicious by raw transactions: {int(model_df['actually_suspicious'].sum()):,}")
    print(f"Wallets labeled normal: {int((1 - model_df['actually_suspicious']).sum()):,}")

    feature_results: List[dict] = []
    for feature in FEATURE_COLUMNS:
        if feature not in model_df.columns:
            print(f"Warning: expected feature column '{feature}' not found in features.parquet; skipping.")
            continue

        feature_results.append(summarize_feature_signal(model_df, feature))

    results_df = pd.DataFrame(feature_results)
    if results_df.empty:
        print("No valid feature columns were available for comparison.")
        return

    results_df = results_df.sort_values("feature", ascending=True).reset_index(drop=True)
    print_feature_table(results_df)
    print_ranked_features(results_df)
    print_pattern_breakdown(model_df, transactions_df)

    print("\nInterpretation note")
    print("=" * 80)
    print(
        "These point-biserial r-values should be used to re-derive the feature weights in "
        "risk_engine.py's WEIGHTS dict. The strongest absolute correlations are the best "
        "candidates for larger weights, while weak or non-significant features should be "
        "down-weighted or revisited."
    )


if __name__ == "__main__":
    main()
