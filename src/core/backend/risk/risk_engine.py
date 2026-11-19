"""
risk_engine.py
Combines anomaly, cluster, graph, temporal, transaction, and network
signals into a 0-100 wallet risk score.

Run from the project root:
    python -m src.core.backend.risk.risk_engine
"""

import os
import sys
from typing import Dict, List

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import (
    DEV_ANOMALY_SCORES, DEV_CLUSTER_LABELS, DEV_DATASET, DEV_FEATURES, DEV_RESULTS,
)
from src.core.backend.ingestion.ingestor import TransactionIngestor


# Each component is the mean of its signals, scaled 0-100 so that a higher
# value is riskier. The same table drives the alert reasons, so a reason is
# only reported for a signal that actually raised the score. No weight is
# derived from the ground-truth labels: the score stays unsupervised.
SIGNALS = {
    "graph": [
        ("degree", "unusually high transaction connectivity"),
        ("betweenness_centrality", "unusually high intermediary connectivity"),
    ],
    "temporal": [
        ("burst_count", "unusually frequent transaction bursts"),
        ("max_tx_in_window", "unusually many transactions in a short window"),
    ],
    "transaction": [
        ("avg_amount", "unusually high transaction amounts"),
        ("total_tx_count", "unusually high transaction volume"),
    ],
    "network": [
        ("unique_ip_count", "unusually high network diversity"),
        ("unique_asn_count", "unusually high provider diversity"),
    ],
}

COMPONENT_WEIGHTS = {
    "anomaly": 0.30,
    "cluster": 0.20,
    "graph": 0.125,
    "temporal": 0.125,
    "transaction": 0.125,
    "network": 0.125,
}


def min_max_scale(values: pd.Series, invert: bool = False) -> pd.Series:
    """Scale a numeric series to 0-100, optionally reversing its direction."""
    minimum = values.min()
    maximum = values.max()
    value_range = maximum - minimum

    if value_range == 0:
        scaled = pd.Series(0.0, index=values.index)
    else:
        scaled = (values - minimum) / value_range * 100

    return 100 - scaled if invert else scaled


def signal_columns(df: pd.DataFrame) -> List[str]:
    """Signal columns from SIGNALS that are present in df and not constant."""
    return [
        col for signals in SIGNALS.values() for col, _ in signals
        if col in df.columns and df[col].nunique(dropna=True) > 1
    ]


def load_ground_truth(csv_path: str) -> pd.DataFrame:
    """Return one suspicious/not-suspicious ground-truth row per wallet.

    Used only to report precision and recall after scoring, never to score.
    """
    ingestor = TransactionIngestor()
    raw_df, _ = ingestor.ingest(csv_path)
    wallet_truth: Dict[str, bool] = {}

    for _, row in raw_df.iterrows():
        input_wallets = row.get("input_addresses[]", [])
        output_wallets = row.get("output_addresses[]", [])

        if not isinstance(input_wallets, list):
            input_wallets = []
        if not isinstance(output_wallets, list):
            output_wallets = []

        is_suspicious = str(row.get("is_suspicious", "")).strip().lower() in ("true", "1")

        for wallet in input_wallets + output_wallets:
            wallet_truth[wallet] = wallet_truth.get(wallet, False) or is_suspicious

    return pd.DataFrame(
        {"actual_suspicious": wallet_truth.values()},
        index=pd.Index(wallet_truth.keys(), name="wallet_address"),
    )


def add_normalized_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add 0-100 risk components and the weighted risk_score."""
    result = df.copy()

    # Isolation Forest assigns lower scores to more anomalous observations.
    result["anomaly_component"] = min_max_scale(result["anomaly_score"], invert=True)
    result["cluster_component"] = (result["cluster_id"] == -1).astype(float) * 100

    for component, signals in SIGNALS.items():
        columns = [col for col in signal_columns(result) if col in dict(signals)]
        if columns:
            result[f"{component}_component"] = sum(min_max_scale(result[c]) for c in columns) / len(columns)

    # Components missing from the data, or constant in it (e.g. no IP fields),
    # drop out and the remaining weights are renormalised.
    weights = {c: w for c, w in COMPONENT_WEIGHTS.items() if f"{c}_component" in result.columns}
    total = sum(weights.values())
    result["risk_score"] = (
        sum(result[f"{c}_component"] * w for c, w in weights.items()) / total
    ).clip(0, 100)
    return result


def risk_level(score: float) -> str:
    """Map a numeric risk score to the requested label range."""
    if score <= 30:
        return "LOW"
    if score <= 60:
        return "MEDIUM"
    if score <= 80:
        return "HIGH"
    return "CRITICAL"


def build_reasons(row: pd.Series, averages: pd.Series) -> List[str]:
    """Return the two or three strongest plain-text deviations for a wallet.

    Every reason corresponds to a signal that raises risk_score, in the same
    direction the score uses (anomaly_score: lower is riskier; every other
    signal: higher is riskier).
    """
    candidates = []

    anomaly_spread = max(abs(averages["anomaly_score"]), 1e-12)
    anomaly_difference = (averages["anomaly_score"] - row["anomaly_score"]) / anomaly_spread
    if anomaly_difference > 0:
        candidates.append((anomaly_difference, "unusually anomalous behavior"))

    for signals in SIGNALS.values():
        for column, reason in signals:
            if column not in averages.index:
                continue
            spread = max(abs(averages[column]), 1e-12)
            relative_difference = (row[column] - averages[column]) / spread
            if relative_difference > 0:
                candidates.append((relative_difference, reason))

    if row["cluster_id"] == -1:
        candidates.append((1.0, "isolated as an HDBSCAN cluster outlier"))

    candidates.sort(reverse=True)
    reasons = [reason for _, reason in candidates[:3]]
    fallback_reasons = [
        "behavior is close to the dataset average",
        "no other dominant risk signal",
    ]
    while len(reasons) < 2:
        reasons.append(fallback_reasons[len(reasons)])
    return reasons


def main() -> None:
    """Build and save the final wallet risk results."""
    features_path = str(DEV_FEATURES)
    anomaly_path = str(DEV_ANOMALY_SCORES)
    cluster_path = str(DEV_CLUSTER_LABELS)
    raw_path = str(DEV_DATASET)
    output_path = str(DEV_RESULTS)

    required_paths = [features_path, anomaly_path, cluster_path, raw_path]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        print(f"Error: required input not found: {missing_paths}")
        sys.exit(1)

    print("Loading feature, anomaly, cluster, and ground-truth data...")
    features_df = pd.read_parquet(features_path)
    anomaly_df = pd.read_parquet(anomaly_path)[["anomaly_score"]]
    cluster_df = pd.read_parquet(cluster_path)[["cluster_id"]]

    # All model outputs use wallet_address as their index, so no entity mapping
    # is needed and every join remains aligned to the unique wallet key.
    result_df = features_df.join(anomaly_df, how="inner").join(cluster_df, how="inner")
    if result_df.empty:
        print("Error: no wallets were shared across the three parquet files.")
        sys.exit(1)

    result_df = add_normalized_components(result_df)
    averages = result_df[["anomaly_score"] + signal_columns(result_df)].mean()
    result_df["risk_level"] = result_df["risk_score"].map(risk_level)
    result_df["top_reasons"] = result_df.apply(
        lambda row: build_reasons(row, averages), axis=1
    )

    ground_truth_df = load_ground_truth(raw_path)
    result_df = result_df.join(ground_truth_df, how="left")
    result_df["actual_suspicious"] = (
        result_df["actual_suspicious"].astype("boolean").fillna(False).astype(bool)
    )

    output_columns = [
        "risk_score", "risk_level", "anomaly_score", "cluster_id", "top_reasons",
        "total_tx_count", "unique_counterparties",
    ]
    output_df = result_df[output_columns].reset_index()
    output_df["risk_score"] = output_df["risk_score"].round(2)
    output_df = output_df.sort_values("risk_score", ascending=False)
    output_df.to_json(output_path, orient="records", indent=2)

    print(f"Saved {len(output_df)} wallet results to {output_path}")
    print("\n--- TOP 10 RISKIEST WALLETS ---")
    print(output_df.head(10).to_string(index=False))

    flagged = result_df[ result_df["risk_level"].isin(["HIGH", "CRITICAL"]) ]
    true_positives = int(flagged["actual_suspicious"].sum())
    actual_suspicious = int(result_df["actual_suspicious"].sum())
    precision = true_positives / len(flagged) if len(flagged) else 0.0
    recall = true_positives / actual_suspicious if actual_suspicious else 0.0
    print("\n--- VALIDATION ---")
    print(f"High/Critical flagged wallets marked suspicious: {true_positives}/{len(flagged)}")
    print(f"Precision: {precision:.2%}")
    print(f"Suspicious wallets caught: {true_positives}/{actual_suspicious}")
    print(f"Recall: {recall:.2%}")

    # Ranking quality does not depend on the fixed level bands above
    ranked = result_df.sort_values("risk_score", ascending=False)["actual_suspicious"]
    if 0 < actual_suspicious < len(ranked):
        print(f"ROC-AUC: {roc_auc_score(ranked, result_df.loc[ranked.index, 'risk_score']):.3f}")
    for k in (100, 500, 1000):
        print(f"Precision of the top {k}: {ranked.head(k).mean():.2%}")


if __name__ == "__main__":
    main()
