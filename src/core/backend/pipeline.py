"""
pipeline.py
Runs the whole unsupervised pipeline on a validated transaction DataFrame,
in memory: features -> Isolation Forest -> HDBSCAN -> risk score and reasons.

Used by the Streamlit Upload & Score page and by
scripts/run_unsupervised_elliptic_plus.py. The step-by-step CLI modules
(feature_engineering, isolation_forest_model, hdbscan_clustering,
risk_engine) do the same thing through files in data/dev/.
"""

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.core.backend.features.feature_engineering import WalletFeatureEngineer
from src.core.backend.ml.hdbscan_clustering import TransactionClusteredModel
from src.core.backend.ml.isolation_forest_model import TransactionAnomalyDetector
from src.core.backend.risk.risk_engine import (
    add_normalized_components, build_reasons, risk_level, signal_columns,
)

# HDBSCAN needs enough wallets to find dense groups; below this every wallet
# is treated as one cluster (no outlier boost).
MIN_WALLETS_TO_CLUSTER = 20
# Above this many wallets HDBSCAN is fitted on a sample and the rest are
# assigned with approximate_predict (a full fit on ~800k wallets takes hours).
MAX_CLUSTER_FIT_SAMPLES = 100_000


def score_wallets(transactions: pd.DataFrame, drop_features: Optional[Iterable[str]] = None,
                  max_cluster_fit_samples: int = MAX_CLUSTER_FIT_SAMPLES) -> pd.DataFrame:
    """Score every wallet in a validated transaction DataFrame.

    drop_features removes engineered features that the data cannot support
    (for example second-level timing features when timestamps are coarse).
    max_cluster_fit_samples caps how many wallets HDBSCAN is fitted on.
    Returns one row per wallet, highest risk first, with risk_score,
    risk_level, top_reasons, the score components and the features.
    """
    features = WalletFeatureEngineer().engineer_features(transactions)
    if drop_features:
        features = features.drop(columns=list(drop_features), errors="ignore")
    if features.empty:
        return features

    numeric = features.select_dtypes(include=[np.number])
    detector = TransactionAnomalyDetector(contamination=0.03, n_estimators=200, max_samples="sqrt")
    result = features.copy()
    result["anomaly_score"] = detector.train(numeric)

    if len(numeric) >= MIN_WALLETS_TO_CLUSTER:
        clusterer = TransactionClusteredModel(min_cluster_size=10, min_samples=10)
        result["cluster_id"] = clusterer.fit_predict_sampled(numeric, max_cluster_fit_samples)
    else:
        result["cluster_id"] = 0

    result = add_normalized_components(result)
    averages = result[["anomaly_score"] + signal_columns(result)].mean()
    result["risk_level"] = result["risk_score"].map(risk_level)
    result["top_reasons"] = [build_reasons(row, averages) for _, row in result.iterrows()]
    return result.sort_values("risk_score", ascending=False)
