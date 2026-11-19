"""Diagnose which behavioral features actually separate suspicious wallets.

This script is intentionally read-only and diagnostic-only. It does not modify any
scoring logic or weights. Its purpose is to answer one question before re-tuning
risk_engine.py: which features correlate most strongly with the wallet-level
ground-truth label based on raw transaction data?
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr



def get_numeric_feature_columns(features_df: pd.DataFrame) -> List[str]:
    """Return all non-constant numeric feature columns, excluding wallet metadata."""
    excluded = {"wallet_address", "WINDOW_SECONDS"}
    candidates = [
        col for col in features_df.select_dtypes(include=[np.number]).columns.tolist()
        if col not in excluded and features_df[col].nunique(dropna=True) > 1
    ]
    return candidates


def get_data_dir() -> Path:
    """Return the dev-data directory used for feature and raw transaction files."""
    from src.config import DEV_DATA_DIR
    return DEV_DATA_DIR


def normalize_wallet_column(df: pd.DataFrame, expected_name: str = "wallet_address") -> pd.DataFrame:
    """Ensure the wallet identifier is named consistently across parquet files."""
    result = df.copy()

    if expected_name in result.columns:
        return result

    # Common case: wallet identifiers are stored as the DataFrame index.
    if result.index.name == expected_name:
        result = result.reset_index()
        return result

    # If the file uses a different identifier column name, prefer the obvious one.
    for candidate in ["wallet", "wallet_id", "entity", "entity_id", "address"]:
        if candidate in result.columns:
            result = result.rename(columns={candidate: expected_name})
            return result

    raise ValueError(
        f"Could not locate a '{expected_name}' column or index in the provided dataframe. "
        f"Columns found: {list(result.columns)}"
    )


def build_ground_truth(ground_truth_df: pd.DataFrame) -> pd.Series:
    """Return the wallet-level suspicious label already stored in the ground-truth parquet."""
    truth = ground_truth_df.copy()

    if "wallet_address" not in truth.columns:
        if truth.index.name == "wallet_address":
            truth = truth.reset_index()
        else:
            raise ValueError("ground_truth.parquet must contain wallet_address as a column or as the index name.")

    if "is_suspicious" not in truth.columns:
        raise ValueError("ground_truth.parquet must contain an 'is_suspicious' column.")

    wallet_truth = truth[["wallet_address", "is_suspicious"]].copy()
    wallet_truth["actually_suspicious"] = wallet_truth["is_suspicious"].astype(bool).astype(int)
    wallet_truth = wallet_truth.set_index("wallet_address")["actually_suspicious"]
    return wallet_truth


def summarize_feature_signal(model_df: pd.DataFrame, feature: str) -> dict:
    """Compute distribution and point-biserial statistics for one feature."""
    required = ["wallet_address", "actually_suspicious", feature]
    missing = [column for column in required if column not in model_df.columns]
    if missing:
        raise KeyError(f"Feature check is missing required columns for '{feature}': {missing}")

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
    """Print a clean summary table of feature means and point-biserial statistics."""
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
    """Print the features in descending order of absolute correlation magnitude."""
    ranked = results_df.copy()
    ranked["abs_r"] = ranked["r"].abs()
    ranked = ranked.sort_values(["abs_r", "p_value"], ascending=[False, True]).reset_index(drop=True)

    print("\nFeatures ranked by |r| (strongest signal first)")
    print("=" * 80)
    print(
        ranked[
            ["feature", "r", "p_value", "abs_r"]
        ].rename(columns={"feature": "feature", "r": "r", "p_value": "p_value", "abs_r": "|r|"})
        .to_string(index=False)
    )


def print_pattern_breakdown(model_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> None:
    """If available, summarize suspicious wallets by their pattern_type."""
    if "pattern_type" not in ground_truth_df.columns:
        print("\nNo 'pattern_type' column found in ground_truth.parquet; skipping pattern breakdown.")
        return

    suspicious_rows = ground_truth_df.copy()
    if "wallet_address" not in suspicious_rows.columns:
        if suspicious_rows.index.name == "wallet_address":
            suspicious_rows = suspicious_rows.reset_index()
        else:
            print("\nCould not find a wallet_address column or index in ground_truth.parquet.")
            return

    suspicious_rows = suspicious_rows[suspicious_rows["is_suspicious"].astype(bool)].copy()
    suspicious_rows = suspicious_rows[["wallet_address", "pattern_type"]].dropna(subset=["pattern_type"])

    if suspicious_rows.empty:
        print("\nNo suspicious wallets with a non-null pattern_type were found.")
        return

    summary = (
        suspicious_rows["pattern_type"]
        .value_counts()
        .rename_axis("pattern_type")
        .reset_index(name="wallet_count")
        .sort_values("wallet_count", ascending=False)
    )

    print("\nSuspicious-wallet pattern breakdown")
    print("=" * 80)
    print(summary.to_string(index=False))


def main() -> None:
    """Load feature and wallet-level ground-truth data, compute signal summaries, then print findings."""
    data_dir = get_data_dir()
    features_path = data_dir / "features.parquet"
    ground_truth_path = data_dir / "ground_truth.parquet"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features parquet file: {features_path}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Missing ground-truth parquet file: {ground_truth_path}")

    print(f"Loading features from: {features_path}")
    print(f"Loading wallet-level ground truth from: {ground_truth_path}")

    features_df = pd.read_parquet(features_path)
    ground_truth_df = pd.read_parquet(ground_truth_path)

    if ground_truth_df.index.name == "wallet_address":
        ground_truth_df = ground_truth_df.reset_index()

    features_df = normalize_wallet_column(features_df, "wallet_address")
    ground_truth_df = normalize_wallet_column(ground_truth_df, "wallet_address")

    wallet_truth = build_ground_truth(ground_truth_df)
    model_df = features_df.merge(wallet_truth.reset_index(), on="wallet_address", how="left")
    model_df["actually_suspicious"] = model_df["actually_suspicious"].fillna(0).astype(int)

    print(f"\nWallets in feature set: {len(model_df):,}")
    print(f"Wallets labeled suspicious by ground truth: {int(model_df['actually_suspicious'].sum()):,}")
    print(f"Wallets labeled normal: {int((1 - model_df['actually_suspicious']).sum()):,}")

    feature_columns = get_numeric_feature_columns(features_df)
    print(f"\nChecking {len(feature_columns)} numeric feature columns automatically: {feature_columns}")

    feature_results: List[dict] = []
    for feature in feature_columns:
        if feature not in model_df.columns:
            print(f"Warning: expected feature column '{feature}' not found in features.parquet; skipping.")
            continue

        summary = summarize_feature_signal(model_df, feature)
        feature_results.append(summary)

    results_df = pd.DataFrame(feature_results)
    if results_df.empty:
        print("No valid feature columns were available for comparison.")
        return

    results_df = results_df.sort_values(["feature"], ascending=True).reset_index(drop=True)
    print_feature_table(results_df)
    print_ranked_features(results_df)
    print_pattern_breakdown(model_df, ground_truth_df)

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
