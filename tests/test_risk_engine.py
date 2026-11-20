"""
test_risk_engine.py
Unit tests for the unsupervised wallet risk engine.
"""

import unittest

import pandas as pd

from src.core.backend.risk import risk_engine
from src.core.backend.risk.risk_engine import add_normalized_components, build_reasons, signal_columns


def wallets(**overrides):
    base = {
        "anomaly_score": [0.2, 0.1, -0.3],
        "cluster_id": [0, 0, -1],
        "degree": [1, 2, 9],
        "betweenness_centrality": [0.0, 0.1, 0.5],
        "burst_count": [0, 0, 4],
        "max_tx_in_window": [1, 1, 5],
        "avg_amount": [0.1, 0.2, 3.0],
        "total_tx_count": [1, 2, 12],
        "unique_ip_count": [1, 1, 3],
        "unique_asn_count": [1, 1, 2],
    }
    base.update(overrides)
    return pd.DataFrame(base, index=["normal", "busier", "risky"])


class TestRiskEngine(unittest.TestCase):
    def test_no_label_derived_weights(self):
        self.assertFalse(hasattr(risk_engine, "WEIGHTS"))

    def test_every_component_contributes(self):
        result = add_normalized_components(wallets())
        for component in risk_engine.COMPONENT_WEIGHTS:
            self.assertIn(f"{component}_component", result.columns)
        # The wallet that is highest on every signal gets the maximum score, and the
        # all-low wallet ranks lowest. With rank scaling a wallet only reaches exactly
        # 0 when it is the unique minimum on every signal; "normal" ties "busier" at the
        # bottom of several signals, so it sits low but above 0.
        self.assertAlmostEqual(result.loc["risky", "risk_score"], 100.0)
        self.assertEqual(result["risk_score"].idxmin(), "normal")
        self.assertLess(result.loc["normal", "risk_score"], result.loc["busier", "risk_score"])
        self.assertTrue((result["risk_score"].between(0, 100)).all())

    def test_missing_network_fields_are_renormalised(self):
        df = wallets().drop(columns=["unique_ip_count", "unique_asn_count"])
        result = add_normalized_components(df)
        self.assertNotIn("network_component", result.columns)
        self.assertAlmostEqual(result.loc["risky", "risk_score"], 100.0)

    def test_reasons_follow_the_score_direction(self):
        df = add_normalized_components(wallets())
        averages = df[["anomaly_score"] + signal_columns(df)].mean()

        risky = build_reasons(df.loc["risky"], averages)
        self.assertEqual(len(risky), 3)
        self.assertNotIn("behavior is close to the dataset average", risky)

        # A wallet below average on every signal gets no "unusually high" reason
        normal = build_reasons(df.loc["normal"], averages)
        self.assertFalse(any("high" in reason for reason in normal))

    def test_reasons_only_use_scored_signals(self):
        scored = {reason for signals in risk_engine.SIGNALS.values() for _, reason in signals}
        scored |= {"unusually anomalous behavior", "isolated as an HDBSCAN cluster outlier",
                   "behavior is close to the dataset average", "no other dominant risk signal"}
        df = add_normalized_components(wallets())
        averages = df[["anomaly_score"] + signal_columns(df)].mean()
        for wallet in df.index:
            self.assertTrue(set(build_reasons(df.loc[wallet], averages)) <= scored)


if __name__ == "__main__":
    unittest.main()
