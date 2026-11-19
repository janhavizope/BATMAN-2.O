"""
graph_builder.py
Constructs a NetworkX wallet-transaction graph from Bitcoin transactions
and implements multi-hop wallet discovery.
"""

import networkx as nx
import pandas as pd
from typing import Dict


class TransactionGraphBuilder:
    """
    Builds and manages a directed graph of wallets and the transactions that
    move funds between them.
    """

    def __init__(self):
        pass

    def build_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Builds a directed graph from a transaction DataFrame.

        Node Types:
            - wallet: 'wallet:<address>'
            - transaction: 'tx:<txid>'

        Edges:
            - wallet -> transaction (inputs)
            - transaction -> wallet (outputs)

        IP, ASN and country values are not graph nodes. Real Bitcoin
        transactions don't carry them, and a few shared hub nodes would
        dominate PageRank and betweenness.
        """
        G = nx.DiGraph()

        if df.empty:
            return G

        tx_nodes = "tx:" + df["txid"].astype(str)
        G.add_nodes_from(
            (node, {"type": "transaction", "label": txid, "fee": fee, "timestamp": ts})
            for node, txid, fee, ts in zip(tx_nodes, df["txid"], df["fee"], df["timestamp"])
        )

        for column, direction in [("input_addresses[]", "wallet_to_tx"), ("output_addresses[]", "tx_to_wallet")]:
            links = pd.DataFrame({"tx": tx_nodes, "wallet": df[column]}).explode("wallet").dropna()
            wallet_nodes = "wallet:" + links["wallet"].astype(str)
            G.add_nodes_from(
                (node, {"type": "wallet", "label": wallet})
                for node, wallet in zip(wallet_nodes, links["wallet"])
            )
            if direction == "wallet_to_tx":
                pairs = zip(wallet_nodes, links["tx"])
            else:
                pairs = zip(links["tx"], wallet_nodes)
            G.add_edges_from(pairs, type=direction)

        return G

    def find_connected_wallets(self, G: nx.DiGraph, start_wallet: str, max_hops: int = 2) -> Dict[str, int]:
        """
        Finds all wallets connected to a given wallet within N hops.
        Traverses the graph as undirected to trace back and forth through
        transactions.

        Args:
            G: The wallet-transaction directed graph.
            start_wallet: Raw wallet address to start search from.
            max_hops: Maximum path length in the graph.

        Returns:
            Dict mapping wallet address string to shortest hop distance.
        """
        start_node = f"wallet:{start_wallet}" if not start_wallet.startswith("wallet:") else start_wallet
        if start_node not in G:
            return {}

        # BFS on undirected version to follow links in both directions
        undirected_G = G.to_undirected()

        try:
            distances = nx.single_source_shortest_path_length(undirected_G, start_node, cutoff=max_hops)
        except nx.NetworkXError:
            return {}

        connected_wallets = {}
        for node, dist in distances.items():
            if node != start_node and G.nodes[node].get("type") == "wallet":
                # Strip prefix for clean wallet outputs
                wallet_addr = node.split(":", 1)[1]
                connected_wallets[wallet_addr] = dist

        return connected_wallets
