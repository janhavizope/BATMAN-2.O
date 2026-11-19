"""
test_clustering.py
Unit tests for the Step 4 HDBSCAN Clustering pipeline.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np


from src.core.backend.ml.hdbscan_clustering import TransactionClusteredModel

class TestHDBSCANClustering(unittest.TestCase):
    def setUp(self):
        # Create a small dataset with 3 distinct clusters and 1 noise point
        # Each cluster will have 6 points (above min_cluster_size=5)
        np.random.seed(42)
        c1 = np.random.normal(loc=0.0, scale=0.1, size=(6, 4))
        c2 = np.random.normal(loc=10.0, scale=0.1, size=(6, 4))
        c3 = np.random.normal(loc=20.0, scale=0.1, size=(6, 4))
        noise = np.array([[50.0, 50.0, 50.0, 50.0]])
        
        all_data = np.vstack([c1, c2, c3, noise])
        self.df = pd.DataFrame(all_data, columns=[f"feature_{i}" for i in range(4)])
        self.df.index = [f"wallet_{i}" for i in range(19)]
        
    def test_clustering_output(self):
        """Test that HDBSCAN identifies clusters and noise correctly."""
        # Use min_cluster_size=5, min_samples=2
        clusterer = TransactionClusteredModel(min_cluster_size=5, min_samples=2)
        labels = clusterer.fit_predict(self.df)
        
        # Verify length matches input
        self.assertEqual(len(labels), 19)
        
        # Check that there are at least some valid clusters (nonnegative labels)
        unique_labels = set(labels)
        valid_labels = [l for l in unique_labels if l != -1]
        
        self.assertTrue(len(valid_labels) >= 2, "Should identify at least 2 clusters")
        
        # The noise point (last element, index 18) should be labeled as noise (-1)
        self.assertEqual(labels[18], -1, "Outlier point should be classified as noise (-1)")

if __name__ == "__main__":
    unittest.main()
