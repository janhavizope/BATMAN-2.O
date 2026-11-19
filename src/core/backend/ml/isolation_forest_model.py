"""
isolation_forest_model.py
Trains an Isolation Forest anomaly detection model on engineered features,
calculates anomaly scores, and identifies the most anomalous entities.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

from src.config import DEV_ANOMALY_SCORES, DEV_DATASET, DEV_FEATURES

class TransactionAnomalyDetector:
    """
    Wrapper for scikit-learn's IsolationForest to detect anomalous Bitcoin entities.
    """

    def __init__(self, contamination: float = 0.03, random_state: int = 42,
                 n_estimators: int = 200, max_samples="sqrt"):
        # contamination represents the expected proportion of outliers/anomalies in the data.
        # fit() will construct the forest of isolation trees.
        # decision_function() returns the raw anomaly scores where lower values denote anomalies.
        self.max_samples = max_samples
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples=("auto" if max_samples == "sqrt" else max_samples),
        )

    def train(self, X: pd.DataFrame) -> np.ndarray:
        """
        Trains the Isolation Forest model on the numeric feature DataFrame X.

        Short Explanation:
        - fit(): Natively isolates observations by randomly selecting a feature and then
                 randomly selecting a split value between the maximum and minimum values of the selected feature.
                 Since recursive partitioning produces noticeably shorter paths for anomalies,
                 anomalous samples will require fewer splits to isolate.
        """
        if self.max_samples == "sqrt":
            self.model.set_params(max_samples=max(1, int(np.sqrt(len(X)))))
        self.model.fit(X)
        return self.get_anomaly_scores(X)

    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculates anomaly scores for each sample in X.

        Short Explanation:
        - decision_function(): Computes the average path length of a sample across all trees.
                               It returns negative scores for anomalies (closer to root) and
                               positive scores for normal instances.

        We return raw scores (lower/more negative = more anomalous) as standard.
        """
        raw_scores = self.model.decision_function(X)
        return raw_scores

from src.core.backend.ingestion.ingestor import TransactionIngestor

def load_wallet_entity_mapping(csv_path: str) -> dict:
    """Helper to map wallet_address to entity_id from the original CSV dataset."""
    if not os.path.exists(csv_path):
        return {}

    ingestor = TransactionIngestor()
    df, _ = ingestor.ingest(csv_path)
    wallet_to_entity = {}

    for _, row in df.iterrows():
        entity = row.get("entity_id", "UNKNOWN")
        inputs = row.get("input_addresses[]", [])
        outputs = row.get("output_addresses[]", [])

        if isinstance(inputs, list):
            for w in inputs:
                wallet_to_entity[w] = entity

        if isinstance(outputs, list):
            for w in outputs:
                wallet_to_entity[w] = entity

    return wallet_to_entity

def main():
    features_path = str(DEV_FEATURES)
    csv_path = str(DEV_DATASET)
    output_path = str(DEV_ANOMALY_SCORES)

    if not os.path.exists(features_path):
        print(f"Error: features.parquet not found at {features_path}. Run feature engineering first.")
        sys.exit(1)

    print(f"Loading features from {features_path}...")
    X = pd.read_parquet(features_path)

    # Ensure only numeric columns are fed to Isolation Forest
    # X index is wallet_address, all columns in features.parquet should be numeric
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_cols]

    print("Training Isolation Forest (n_estimators=200, contamination=0.03, max_samples=sqrt)...")
    detector = TransactionAnomalyDetector(contamination=0.03, n_estimators=200, max_samples="sqrt")
    anomaly_scores = detector.train(X_numeric)

    # Add anomaly_score column
    result_df = X.copy()
    result_df["anomaly_score"] = anomaly_scores

    # Save the output containing features + scores
    print(f"Saving anomaly scores to Parquet at {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_parquet(output_path)

    # Resolve entity_ids to print alongside wallets
    print("Loading entity mappings from raw dataset...")
    wallet_to_entity = load_wallet_entity_mapping(csv_path)

    # Prepare top 10 print df
    print_df = result_df[["anomaly_score"]].copy()
    print_df["entity_id"] = print_df.index.map(lambda w: wallet_to_entity.get(w, "UNKNOWN"))

    # Sort by anomaly score ascending (lower = more anomalous)
    top_10 = print_df.sort_values(by="anomaly_score", ascending=True).head(10)

    print("\n--- TOP 10 MOST ANOMALOUS ENTITIES ---")
    print(top_10.to_string(index=True))

if __name__ == "__main__":
    main()
