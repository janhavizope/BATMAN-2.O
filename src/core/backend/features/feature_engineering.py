"""
feature_engineering.py
Feature engineering module for the Bitcoin transaction monitoring backend.
Extracts transaction, wallet, temporal, graph, and network features for each unique wallet address.

Run from the project root:
    python -m src.core.backend.features.feature_engineering
"""

import os
import sys
from typing import Optional

import networkx as nx
import pandas as pd

from src.config import DEV_DATASET, DEV_FEATURES
from src.core.backend.graph.graph_builder import TransactionGraphBuilder

DATASET_PATH = str(DEV_DATASET)
WINDOW_SECONDS = 840

FEATURE_COLUMNS = [
    "WINDOW_SECONDS", "avg_amount", "max_transaction_amount", "avg_fee",
    "avg_input_count", "avg_output_count", "total_tx_count",
    "incoming_transaction_count", "outgoing_transaction_count",
    "total_sent", "total_received", "incoming_outgoing_ratio",
    "unique_counterparties", "fan_in_behavior", "fan_out_behavior",
    "tx_per_hour", "avg_inter_arrival_time", "burst_count",
    "max_tx_in_window", "max_fan_in_window", "max_fan_out_window",
    "min_pass_through_gap", "degree_centrality", "degree", "in_degree",
    "out_degree", "betweenness_centrality", "pagerank", "k_core",
    "unique_ip_count", "unique_asn_count", "unique_country_count",
]


def _explode_side(df: pd.DataFrame, address_col: str, amount_col: str) -> pd.DataFrame:
    """One row per (transaction row, address) on one side of each transaction.

    Addresses and amounts are matched by position; a missing amount is NaN.
    """
    addresses = df[[address_col]].reset_index(names="row").explode(address_col)
    addresses["pos"] = addresses.groupby("row").cumcount()
    amounts = df[[amount_col]].reset_index(names="row").explode(amount_col)
    amounts["pos"] = amounts.groupby("row").cumcount()
    side = addresses.merge(amounts, on=["row", "pos"], how="left").dropna(subset=[address_col])
    side = side.rename(columns={address_col: "wallet", amount_col: "amount"})
    side["amount"] = pd.to_numeric(side["amount"], errors="coerce")
    return side[["row", "wallet", "amount"]]


def _max_events_in_window(events: pd.DataFrame, window_seconds: int) -> pd.Series:
    """Per wallet, the most events in any window of window_seconds (both ends inclusive)."""
    if events.empty:
        return pd.Series(dtype=float)
    events = events[["wallet", "timestamp"]].sort_values(["wallet", "timestamp"]).assign(one=1.0)
    counts = (
        events.groupby("wallet")
        .rolling(f"{window_seconds}s", on="timestamp", closed="both")["one"]
        .sum()
    )
    return counts.groupby(level="wallet").max()


class WalletFeatureEngineer:
    """Engineer wallet-level behavioral and graph features from transaction data.

    A wallet listed in a transaction's inputs is *sending* in that
    transaction; a wallet listed in its outputs is *receiving*. A wallet's
    counterparties are the addresses on the other side of its transactions.
    Co-input addresses are not counterparties: under the common-input
    heuristic they usually belong to the same owner.
    """

    def __init__(self):
        self.graph_builder = TransactionGraphBuilder()

    def engineer_features(self, df: pd.DataFrame, G: Optional[nx.DiGraph] = None) -> pd.DataFrame:
        """Build wallet features with graph centrality and rolling-window temporal signals."""
        if G is None:
            G = self.graph_builder.build_graph(df)

        df = df.reset_index(drop=True)
        sent = _explode_side(df, "input_addresses[]", "input_amounts[]")
        received = _explode_side(df, "output_addresses[]", "output_amounts[]")
        if sent.empty and received.empty:
            return pd.DataFrame(columns=FEATURE_COLUMNS, index=pd.Index([], name="wallet_address"))

        tx = pd.DataFrame({
            "txid": df["txid"],
            "timestamp": pd.to_datetime(df["timestamp"]),
            "fee": df["fee"] if "fee" in df.columns else 0.0,
            "n_inputs": df["input_addresses[]"].str.len(),
            "n_outputs": df["output_addresses[]"].str.len(),
            "tx_amount": df["output_amounts[]"].apply(sum),
        })
        sent = sent.join(tx[["timestamp"]], on="row")
        received = received.join(tx[["timestamp"]], on="row")

        # Every transaction a wallet takes part in, counted once
        wallet_tx = (
            pd.concat([sent[["row", "wallet"]], received[["row", "wallet"]]])
            .join(tx, on="row")
            .drop_duplicates(["wallet", "txid"])
            .sort_values(["wallet", "timestamp"])
        )
        by_wallet = wallet_tx.groupby("wallet")
        features = pd.DataFrame(index=pd.Index(sorted(by_wallet.groups), name="wallet_address"))

        features["WINDOW_SECONDS"] = WINDOW_SECONDS
        features["avg_amount"] = by_wallet["tx_amount"].mean()
        features["max_transaction_amount"] = by_wallet["tx_amount"].max()
        features["avg_fee"] = by_wallet["fee"].mean()
        features["avg_input_count"] = by_wallet["n_inputs"].mean()
        features["avg_output_count"] = by_wallet["n_outputs"].mean()
        features["total_tx_count"] = by_wallet.size()

        sent_events = sent.drop_duplicates(["wallet", "row"])
        received_events = received.drop_duplicates(["wallet", "row"])
        totals = pd.DataFrame({
            "incoming_transaction_count": received_events.groupby("wallet").size(),
            "outgoing_transaction_count": sent_events.groupby("wallet").size(),
            "total_sent": sent.groupby("wallet")["amount"].sum(),
            "total_received": received.groupby("wallet")["amount"].sum(),
        })
        features = features.join(totals).fillna({c: 0 for c in totals.columns})
        features["incoming_outgoing_ratio"] = (
            features["total_received"] / features["total_sent"].clip(lower=1e-12)
        )

        # Counterparties: recipients of what the wallet sends, senders of what it receives
        recipients = sent[["row", "wallet"]].merge(
            received[["row", "wallet"]].rename(columns={"wallet": "counterparty"}), on="row")
        senders = received[["row", "wallet"]].merge(
            sent[["row", "wallet"]].rename(columns={"wallet": "counterparty"}), on="row")
        recipients = recipients[recipients["wallet"] != recipients["counterparty"]]
        senders = senders[senders["wallet"] != senders["counterparty"]]

        def distinct_counterparties(pairs):
            return pairs.groupby("wallet")["counterparty"].nunique().reindex(features.index, fill_value=0)

        features["unique_counterparties"] = distinct_counterparties(pd.concat([recipients, senders]))
        features["fan_in_behavior"] = (
            features["incoming_transaction_count"] / distinct_counterparties(senders).clip(lower=1)
        )
        features["fan_out_behavior"] = (
            features["outgoing_transaction_count"] / distinct_counterparties(recipients).clip(lower=1)
        )

        age_seconds = (by_wallet["timestamp"].max() - by_wallet["timestamp"].min()).dt.total_seconds()
        features["tx_per_hour"] = features["total_tx_count"] / (age_seconds / 3600.0).clip(lower=1.0)

        gaps = by_wallet["timestamp"].diff().dt.total_seconds()
        features["avg_inter_arrival_time"] = gaps.groupby(wallet_tx["wallet"]).mean()
        features["burst_count"] = (gaps < 10).groupby(wallet_tx["wallet"]).sum()

        features["max_tx_in_window"] = _max_events_in_window(wallet_tx, WINDOW_SECONDS)
        features["max_fan_in_window"] = _max_events_in_window(received_events, WINDOW_SECONDS)
        features["max_fan_out_window"] = _max_events_in_window(sent_events, WINDOW_SECONDS)

        # Shortest gap between receiving funds and next sending them
        if sent_events.empty or received_events.empty:
            features["min_pass_through_gap"] = 0.0
        else:
            outgoing = sent_events[["wallet", "timestamp"]].sort_values("timestamp")
            incoming = (
                received_events[["wallet", "timestamp"]]
                .rename(columns={"timestamp": "in_time"})
                .sort_values("in_time")
            )
            paired = pd.merge_asof(
                outgoing, incoming, left_on="timestamp", right_on="in_time",
                by="wallet", direction="backward",
            )
            paired["gap"] = (paired["timestamp"] - paired["in_time"]).dt.total_seconds()
            features["min_pass_through_gap"] = paired.groupby("wallet")["gap"].min()

        nodes = "wallet:" + features.index.to_series()
        betweenness_samples = min(len(G), min(50, max(10, len(G) // 20)))
        if betweenness_samples > 0:
            betweenness = nx.betweenness_centrality(G, k=betweenness_samples, seed=42)
        else:
            betweenness = {}
        features["degree_centrality"] = nodes.map(nx.degree_centrality(G))
        features["degree"] = nodes.map(dict(G.degree()))
        features["in_degree"] = nodes.map(dict(G.in_degree()))
        features["out_degree"] = nodes.map(dict(G.out_degree()))
        features["betweenness_centrality"] = nodes.map(betweenness)
        features["pagerank"] = nodes.map(nx.pagerank(G, max_iter=200) if len(G) else {})
        features["k_core"] = nodes.map(nx.core_number(G.to_undirected()) if len(G) else {})

        # The sender's IP when the wallet sends, the receiver's IP when it receives
        ips = pd.concat([
            sent_events[["wallet", "row"]].join(df["src_ip"].rename("ip"), on="row"),
            received_events[["wallet", "row"]].join(df["dst_ip"].rename("ip"), on="row"),
        ])
        features["unique_ip_count"] = ips.dropna(subset=["ip"]).groupby("wallet")["ip"].nunique()
        participation = wallet_tx[["wallet", "row"]]
        for column, name in [("asn", "unique_asn_count"), ("geo_country", "unique_country_count")]:
            values = participation.join(df[column], on="row").dropna(subset=[column])
            features[name] = values.groupby("wallet")[column].nunique()

        return features[FEATURE_COLUMNS].fillna(0)


if __name__ == "__main__":
    from src.core.backend.ingestion.ingestor import TransactionIngestor

    dataset_path = DATASET_PATH
    output_parquet_path = str(DEV_FEATURES)

    if not os.path.exists(dataset_path):
        print(f"Error: dataset not found at {dataset_path}")
        sys.exit(1)

    print(f"Loading transaction dataset from {dataset_path}...")
    ingestor = TransactionIngestor()
    df, _ = ingestor.ingest(dataset_path)

    print("Building transaction graph...")
    graph_builder = TransactionGraphBuilder()
    G = graph_builder.build_graph(df)

    print("Running feature engineering...")
    engineer = WalletFeatureEngineer()
    features_df = engineer.engineer_features(df, G)

    print(f"Saving features to Parquet at {output_parquet_path}...")
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    features_df.to_parquet(output_parquet_path)

    print("\n--- FIRST 5 ROWS ---")
    print(features_df.head(5).to_string())

    print("\n--- BASIC STATS (MIN, MAX, MEAN) ---")
    stats_df = pd.DataFrame({
        "min": features_df.min(),
        "max": features_df.max(),
        "mean": features_df.mean(),
    })
    print(stats_df.to_string())

    print("\nFeature engineering execution completed.")
