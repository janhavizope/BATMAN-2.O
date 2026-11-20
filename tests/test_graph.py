"""
test_graph.py
Unit tests for data correlation, heterogeneous graph building,
and multi-hop graph traversal.
"""

import os
import sys
import unittest
import pandas as pd
import networkx as nx


from src.core.backend.graph.graph_builder import TransactionGraphBuilder

class TestGraphPipeline(unittest.TestCase):
    def setUp(self):
        self.graph_builder = TransactionGraphBuilder()
        
        # Simple dummy transaction data
        self.dummy_data = [
            {
                "txid": "tx1",
                "timestamp": "2026-08-25T12:00:00",
                "src_ip": "192.168.1.1",
                "dst_ip": "10.0.0.1",
                "src_port": 8333,
                "dst_port": 8333,
                "input_addresses[]": ["wallet_A"],
                "output_addresses[]": ["wallet_B"],
                "input_amounts[]": [1.0],
                "output_amounts[]": [0.99],
                "geo_country": "US",
                "asn": "AS100",
                "fee": 0.01,
                "script_type": "P2PKH",
                "block_height": 800000,
                "block_timestamp": "2026-08-25T12:05:00",
                "connection_duration": 1.0,
                "entity_id": "entity_1"
            },
            {
                "txid": "tx2",
                "timestamp": "2026-08-25T12:10:00",
                "src_ip": "192.168.1.1", # Same IP used by wallet_B (sender in this tx)
                "dst_ip": "10.0.0.2",
                "src_port": 8333,
                "dst_port": 8333,
                "input_addresses[]": ["wallet_B"],
                "output_addresses[]": ["wallet_C"],
                "input_amounts[]": [0.99],
                "output_amounts[]": [0.98],
                "geo_country": "US",
                "asn": "AS100",
                "fee": 0.01,
                "script_type": "P2PKH",
                "block_height": 800001,
                "block_timestamp": "2026-08-25T12:15:00",
                "connection_duration": 1.0,
                "entity_id": "entity_1"
            }
        ]
        self.df = pd.DataFrame(self.dummy_data)

    def test_graph_builder_nodes_and_edges(self):
        """Test that the NetworkX graph builds correct node and edge types."""
        G = self.graph_builder.build_graph(self.df)
        
        # Check node types
        self.assertEqual(G.nodes["wallet:wallet_A"]["type"], "wallet")
        self.assertEqual(G.nodes["tx:tx1"]["type"], "transaction")

        # IP / ASN / country values are not hub nodes
        self.assertEqual({d["type"] for _, d in G.nodes(data=True)}, {"wallet", "transaction"})
        self.assertEqual(G.number_of_nodes(), 5)  # 3 wallets + 2 transactions

        # Check specific edges
        # Wallet to Tx (inputs)
        self.assertTrue(G.has_edge("wallet:wallet_A", "tx:tx1"))
        # Tx to Wallet (outputs)
        self.assertTrue(G.has_edge("tx:tx1", "wallet:wallet_B"))
        self.assertTrue(G.has_edge("wallet:wallet_B", "tx:tx2"))
        self.assertEqual(G.number_of_edges(), 4)

    def test_multi_hop_traversal(self):
        """Test multi-hop wallet path traversal."""
        G = self.graph_builder.build_graph(self.df)
        
        # wallet_A (input to tx1) -> tx1 -> wallet_B (output of tx1) = 2 hops
        # wallet_B (input to tx2) -> tx2 -> wallet_C (output of tx2) = 2 hops
        # From wallet_A within 2 hops (should find wallet_B)
        connected_2 = self.graph_builder.find_connected_wallets(G, "wallet_A", max_hops=2)
        self.assertIn("wallet_B", connected_2)
        self.assertNotIn("wallet_C", connected_2) # C is 4 hops away (A -> tx1 -> B -> tx2 -> C)
        self.assertEqual(connected_2["wallet_B"], 2)
        
        # From wallet_A within 4 hops (should find wallet_B and wallet_C)
        connected_4 = self.graph_builder.find_connected_wallets(G, "wallet_A", max_hops=4)
        self.assertIn("wallet_B", connected_4)
        self.assertIn("wallet_C", connected_4)
        self.assertEqual(connected_4["wallet_C"], 4)

    def test_nonexistent_wallet(self):
        """Test that searching from a non-existent wallet returns empty."""
        G = self.graph_builder.build_graph(self.df)
        connected = self.graph_builder.find_connected_wallets(G, "wallet_nonexistent", max_hops=2)
        self.assertEqual(connected, {})

if __name__ == "__main__":
    unittest.main()
