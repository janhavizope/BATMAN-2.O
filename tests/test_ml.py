"""
test_ml.py
Unit tests for the Step 4 ML Anomaly Detection pipeline (Isolation Forest).
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np


from src.core.backend.ml.isolation_forest_model import TransactionAnomalyDetector

class TestIsolationForestModel(unittest.TestCase):
    def setUp(self):
        # Create a small dummy features dataframe
        # 20 normal samples and 1 outlier sample
        np.random.seed(42)
        normal_data = np.random.normal(loc=0.0, scale=1.0, size=(20, 5))
        outlier_data = np.array([[10.0, 10.0, 10.0, 10.0, 10.0]])
        
        all_data = np.vstack([normal_data, outlier_data])
        self.df = pd.DataFrame(all_data, columns=[f"feature_{i}" for i in range(5)])
        self.df.index = [f"wallet_{i}" for i in range(21)]
        
    def test_training_and_score_inversion(self):
        """Test that the Isolation Forest model trains and produces inverted scores."""
        # 1 outlier in 21 samples is ~4.7%, contamination = 0.05 is appropriate
        detector = TransactionAnomalyDetector(contamination=0.05)
        scores = detector.train(self.df)
        
        # Verify length
        self.assertEqual(len(scores), 21)
        
        # Check scores: outlier should have the lowest (most negative) score
        outlier_index = 20 # index of the outlier (wallet_20)
        min_score_idx = np.argmin(scores)
        
        self.assertEqual(min_score_idx, outlier_index)
        self.assertTrue(scores[outlier_index] < 0, "Outlier should have negative anomaly score")
        
        # Normal samples should generally have higher scores
        self.assertTrue(scores[outlier_index] < np.mean(scores))

if __name__ == "__main__":
    unittest.main()
