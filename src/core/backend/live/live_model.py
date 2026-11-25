"""
live_model.py
A supervised model that CAN run on a raw live or uploaded transaction.

The strong Elliptic transaction model needs 166 engineered graph-aggregate
features that a single live transaction does not have. This model is trained on
only the features computable from any BATMAN-schema transaction — total value,
fee, number of input addresses, number of output addresses — so it runs live,
on uploads, and on the dataset alike. It is weaker than the full model
(ROC-AUC ~0.82 vs 0.93 on the held-out test) but far better than nothing, and
it is a learned model rather than hand-tuned rules.

It trains once from the committed scores (data/elliptic_plus/tx_scores.parquet,
Git LFS) and caches the trained model to models/live_tx_model.joblib. No raw
dataset download is needed. If the scores are unavailable (e.g. Git LFS not
pulled) the loader returns None and callers skip the supervised reading.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.config import ELLIPTIC_PLUS_DIR, MODELS_DIR

# The only features available from a raw BATMAN-schema transaction.
FEATURES = ["total_BTC", "fees", "num_input_addresses", "num_output_addresses"]
MODEL_FILE = MODELS_DIR / "live_tx_model.joblib"
_TX_SCORES = ELLIPTIC_PLUS_DIR / "tx_scores.parquet"

_model = None  # process-level cache


def _is_lfs_pointer(path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(64).startswith(b"version https://git-lfs")
    except OSError:
        return True


def train_model():
    """Train the live transaction model from the committed scores; cache to disk."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    frame = pd.read_parquet(_TX_SCORES)
    labelled = frame[frame["class"].isin(["illicit", "licit"])].copy()
    train = labelled[labelled["split"] == "train"]
    X = train[FEATURES].fillna(0.0)
    y = (train["class"] == "illicit").astype(int)
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    model.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    return model


def load_model():
    """Return the cached model, training it on first use. None if unavailable."""
    global _model
    if _model is not None:
        return _model
    import joblib
    if MODEL_FILE.exists():
        _model = joblib.load(MODEL_FILE)
        return _model
    if not _TX_SCORES.exists() or _is_lfs_pointer(_TX_SCORES):
        return None  # scores not present (Git LFS not pulled) — skip gracefully
    _model = train_model()
    return _model


def _features_from_clean(clean: pd.DataFrame) -> pd.DataFrame:
    """Compute the 4 live-computable features from a validated BATMAN frame."""
    def total(col):
        return clean[col].apply(lambda v: float(sum(v)) if isinstance(v, (list, tuple)) else 0.0)

    def count(col):
        return clean[col].apply(lambda v: len(v) if isinstance(v, (list, tuple)) else 0)

    return pd.DataFrame({
        "total_BTC": total("output_amounts[]"),
        "fees": pd.to_numeric(clean.get("fee", 0), errors="coerce").fillna(0.0),
        "num_input_addresses": count("input_addresses[]"),
        "num_output_addresses": count("output_addresses[]"),
    })


def supervised_tx_risk(clean: pd.DataFrame) -> Optional[pd.Series]:
    """Probability (0-1) that each validated transaction is illicit, or None.

    `clean` is a frame validated by TransactionValidator (list columns like
    `output_amounts[]`). Returns a Series aligned to `clean`'s index, or None
    when the model cannot be loaded.
    """
    model = load_model()
    if model is None or clean.empty:
        return None
    X = _features_from_clean(clean)[FEATURES].fillna(0.0)
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=clean.index, name="supervised_risk")


def risk_level(p: float) -> str:
    """Band a supervised probability into a level for display."""
    if p >= 0.80:
        return "CRITICAL"
    if p >= 0.50:
        return "HIGH"
    if p >= 0.30:
        return "MEDIUM"
    return "LOW"
