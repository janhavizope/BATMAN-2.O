"""
test_pipeline.py
Tests for the in-memory unsupervised pipeline used by the Upload & Score page.
"""

import numpy as np
import pandas as pd

from src.core.backend.pipeline import score_wallets
from src.core.backend.validation.validator import TransactionValidator


def transactions(n=60, with_network=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sender, receiver = f"w{rng.integers(0, 25)}", f"w{rng.integers(0, 25)}"
        amount = float(rng.integers(1, 100)) / 10
        row = {
            "txid": f"{i:064x}",
            "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=int(rng.integers(0, 5000))),
            "input_addresses[]": f"{sender}", "output_addresses[]": f"{receiver}",
            "input_amounts[]": f"{amount + 0.001}", "output_amounts[]": f"{amount}", "fee": 0.001,
        }
        if with_network:
            row.update(src_ip=f"10.0.0.{rng.integers(1, 9)}", dst_ip="10.0.1.1", src_port=8333,
                       dst_port=8333, asn=f"AS{rng.integers(1, 4)}", geo_country="IN")
        rows.append(row)
    return pd.DataFrame(rows)


def test_scores_every_wallet():
    clean, report = TransactionValidator().validate(transactions())
    assert report["valid_records"] == 60
    results = score_wallets(clean)
    assert results.index.is_unique
    assert results["risk_score"].between(0, 100).all()
    assert results["risk_score"].is_monotonic_decreasing
    assert results["top_reasons"].map(len).ge(2).all()
    assert {"anomaly_component", "cluster_component", "network_component"} <= set(results.columns)


def test_without_network_fields():
    clean, _ = TransactionValidator().validate(transactions(with_network=False))
    results = score_wallets(clean)
    # No IP / ASN variation: the network component drops out instead of diluting the score
    assert "network_component" not in results.columns
    assert results["risk_score"].max() > 0


def test_drop_features():
    clean, _ = TransactionValidator().validate(transactions())
    results = score_wallets(clean, drop_features=["burst_count", "max_tx_in_window"])
    assert "burst_count" not in results.columns
    assert "temporal_component" not in results.columns


def test_sampled_clustering_labels_every_wallet():
    clean, _ = TransactionValidator().validate(transactions(n=200))
    full = score_wallets(clean)
    sampled = score_wallets(clean, max_cluster_fit_samples=15)
    assert len(sampled) == len(full)
    assert sampled["cluster_id"].notna().all()
    assert sampled["risk_score"].between(0, 100).all()


def test_small_upload_skips_clustering():
    clean, _ = TransactionValidator().validate(transactions(n=3))
    results = score_wallets(clean)
    assert (results["cluster_id"] == 0).all()
