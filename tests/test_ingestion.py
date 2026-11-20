"""
test_ingestion.py
Automated tests for validating schema validation, type casting,
missing value handling, duplicate checking, and data quality reporting.
"""

import os
import unittest
import tempfile
import json
import pandas as pd
import numpy as np
from src.core.backend.ingestion.ingestor import TransactionIngestor
from src.core.backend.validation.validator import TransactionValidator

class TestTransactionPipeline(unittest.TestCase):
    def setUp(self):
        self.validator = TransactionValidator()
        self.ingestor = TransactionIngestor()
        
        # Define a single valid row as a baseline for tests
        self.valid_row = {
            "timestamp": "2026-08-25T12:00:00",
            "src_ip": "192.168.1.1",
            "dst_ip": "10.0.0.1",
            "src_port": 8333,
            "dst_port": 50001,
            "txid": "a" * 64, # 64 hex characters
            "input_addresses[]": '["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"]',
            "output_addresses[]": '["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNb"]',
            "input_amounts[]": "[1.5]",
            "output_amounts[]": "[1.499]",
            "geo_country": "US",
            "asn": "AS1234",
            "fee": 0.001,
            "script_type": "P2PKH",
            "block_height": 800000,
            "block_timestamp": "2026-08-25T12:05:00",
            "connection_duration": 10.5,
            "entity_id": "entity_1"
        }

    def test_valid_transaction(self):
        """Test that a perfectly valid transaction passes validation."""
        df = pd.DataFrame([self.valid_row])
        clean_df, report = self.validator.validate(df)
        
        self.assertEqual(report["total_records"], 1)
        self.assertEqual(report["valid_records"], 1)
        self.assertEqual(report["invalid_records"], 0)
        self.assertEqual(len(clean_df), 1)
        
        # Verify lists were successfully parsed
        self.assertEqual(clean_df.iloc[0]["input_addresses[]"], ["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"])
        self.assertEqual(clean_df.iloc[0]["input_amounts[]"], [1.5])
        self.assertEqual(clean_df.iloc[0]["fee"], 0.001)

    def test_missing_required_columns(self):
        """Test that missing required columns raises a ValueError."""
        df = pd.DataFrame([self.valid_row])
        df = df.drop(columns=["input_addresses[]"]) # address lists are required

        with self.assertRaises(ValueError):
            self.validator.validate(df)

    def test_multi_element_list_fields(self):
        """Array fields given as Python lists with several elements are accepted."""
        row = dict(self.valid_row)
        row["input_addresses[]"] = ["walletA", "walletB"]
        row["input_amounts[]"] = [0.5, 0.501]
        clean_df, report = self.validator.validate(pd.DataFrame([row]))
        self.assertEqual(report["valid_records"], 1)
        self.assertEqual(clean_df.iloc[0]["input_addresses[]"], ["walletA", "walletB"])

    def test_network_fields_are_optional(self):
        """On-chain data without IP / port / ASN / country fields is valid."""
        df = pd.DataFrame([self.valid_row]).drop(
            columns=["src_ip", "dst_ip", "src_port", "dst_port", "geo_country", "asn"]
        )
        clean_df, report = self.validator.validate(df)
        self.assertEqual(report["valid_records"], 1)

    def test_integer_txids_allowed_when_not_strict(self):
        """Datasets with anonymised integer txids can opt out of the hex check."""
        row = dict(self.valid_row, txid="230425980")
        _, strict = self.validator.validate(pd.DataFrame([row]))
        _, lenient = TransactionValidator(require_hex_txid=False).validate(pd.DataFrame([row]))
        self.assertEqual(strict["valid_records"], 0)
        self.assertEqual(lenient["valid_records"], 1)

    def test_missing_critical_fields(self):
        """Test that missing critical fields (txid, timestamp) drops the records."""
        row_missing_txid = self.valid_row.copy()
        row_missing_txid["txid"] = np.nan
        
        row_missing_time = self.valid_row.copy()
        row_missing_time["timestamp"] = None
        
        df = pd.DataFrame([row_missing_txid, row_missing_time, self.valid_row])
        clean_df, report = self.validator.validate(df)
        
        self.assertEqual(report["total_records"], 3)
        self.assertEqual(report["valid_records"], 1)
        self.assertEqual(report["invalid_records"], 2)
        self.assertEqual(report["invalid_reasons"]["missing_critical_fields"], 2)
        self.assertEqual(len(clean_df), 1)

    def test_invalid_ip_format(self):
        """Test that invalid IPs are flagged and dropped."""
        row_invalid_ip = self.valid_row.copy()
        row_invalid_ip["src_ip"] = "999.999.999.999" # Invalid IPv4
        row_invalid_ip["txid"] = "b" * 64  # distinct txid (two txs never share one)

        df = pd.DataFrame([row_invalid_ip, self.valid_row])
        clean_df, report = self.validator.validate(df)
        
        self.assertEqual(report["valid_records"], 1)
        self.assertEqual(report["invalid_records"], 1)
        self.assertEqual(report["invalid_reasons"]["invalid_ip_format"], 1)

    def test_invalid_port_range(self):
        """Test that ports outside 1-65535 are flagged and dropped."""
        row_invalid_port = self.valid_row.copy()
        row_invalid_port["src_port"] = 999999
        row_invalid_port["txid"] = "b" * 64  # distinct txid (two txs never share one)

        df = pd.DataFrame([row_invalid_port, self.valid_row])
        clean_df, report = self.validator.validate(df)
        
        self.assertEqual(report["valid_records"], 1)
        self.assertEqual(report["invalid_records"], 1)
        self.assertEqual(report["invalid_reasons"]["invalid_port_range"], 1)

    def test_duplicate_txid(self):
        """Test that duplicate transaction IDs are counted and all but the first are dropped."""
        row1 = self.valid_row.copy()
        row2 = self.valid_row.copy()

        df = pd.DataFrame([row1, row2])
        clean_df, report = self.validator.validate(df)

        # One duplicate is reported and removed: only the first occurrence is kept
        self.assertEqual(report["duplicate_txids"], 1)
        self.assertEqual(report["valid_records"], 1)
        self.assertEqual(len(clean_df), 1)

    def test_unparseable_timestamp_is_dropped_not_crashed(self):
        """A row with a non-date timestamp is dropped as invalid, not raised as an error."""
        bad = self.valid_row.copy()
        bad["timestamp"] = "not-a-date"
        df = pd.DataFrame([bad, self.valid_row])
        clean_df, report = self.validator.validate(df)
        self.assertEqual(report["valid_records"], 1)
        self.assertEqual(report["invalid_reasons"]["missing_critical_fields"], 1)
        self.assertEqual(len(clean_df), 1)

    def test_negative_values(self):
        """Test that negative amounts or fees are dropped."""
        row_neg_fee = self.valid_row.copy()
        row_neg_fee["fee"] = -0.01
        row_neg_fee["txid"] = "b" * 64  # distinct txids (two txs never share one)

        row_neg_amount = self.valid_row.copy()
        row_neg_amount["input_amounts[]"] = "[-1.0]"
        row_neg_amount["txid"] = "c" * 64

        df = pd.DataFrame([row_neg_fee, row_neg_amount, self.valid_row])
        clean_df, report = self.validator.validate(df)
        
        self.assertEqual(report["valid_records"], 1)
        self.assertEqual(report["invalid_records"], 2)
        self.assertEqual(report["invalid_reasons"]["negative_amount_or_fee"], 2)

    def test_missing_values_imputed(self):
        """Test that missing non-critical values are imputed correctly."""
        row_missing_fields = self.valid_row.copy()
        row_missing_fields["geo_country"] = np.nan
        row_missing_fields["fee"] = np.nan
        
        df = pd.DataFrame([row_missing_fields])
        clean_df, report = self.validator.validate(df)
        
        self.assertEqual(report["valid_records"], 1)
        self.assertEqual(clean_df.iloc[0]["geo_country"], "UNKNOWN")
        self.assertEqual(clean_df.iloc[0]["fee"], 0.0)

    def test_inconsistent_amounts(self):
        """Test that inconsistent transaction amounts are flagged but not dropped."""
        row_inconsistent = self.valid_row.copy()
        row_inconsistent["input_amounts[]"] = "[10.0]"
        row_inconsistent["output_amounts[]"] = "[1.0]" # mismatch!
        row_inconsistent["fee"] = 0.0
        
        df = pd.DataFrame([row_inconsistent])
        clean_df, report = self.validator.validate(df)
        
        self.assertEqual(report["valid_records"], 1)
        self.assertEqual(report["invalid_reasons"]["inconsistent_amounts"], 1)

    def test_file_formats_csv_json_xml(self):
        """Test ingestion with CSV, JSON, and XML files."""
        df = pd.DataFrame([self.valid_row])
        
        # Test CSV Ingestion
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
            df.to_csv(tmp_csv.name, index=False)
        try:
            clean_csv, report_csv = self.ingestor.ingest(tmp_csv.name)
            self.assertEqual(report_csv["file_format"], "CSV")
            self.assertEqual(report_csv["valid_records"], 1)
        finally:
            os.remove(tmp_csv.name)
            
        # Test JSON Ingestion
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_json:
            df.to_json(tmp_json.name, orient="records")
        try:
            clean_json, report_json = self.ingestor.ingest(tmp_json.name)
            self.assertEqual(report_json["file_format"], "JSON")
            self.assertEqual(report_json["valid_records"], 1)
        finally:
            os.remove(tmp_json.name)

        # Test XML Ingestion
        # XML tags cannot contain bracket characters, so we rename them first.
        xml_df = df.rename(columns={
            "input_addresses[]": "input_addresses",
            "output_addresses[]": "output_addresses",
            "input_amounts[]": "input_amounts",
            "output_amounts[]": "output_amounts"
        })
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_xml:
            xml_df.to_xml(tmp_xml.name, index=False)
        try:
            clean_xml, report_xml = self.ingestor.ingest(tmp_xml.name)
            self.assertEqual(report_xml["file_format"], "XML")
            self.assertEqual(report_xml["valid_records"], 1)
            # Verify the ingestor mapped the columns back to the canonical bracketed names
            self.assertIn("input_addresses[]", clean_xml.columns)
            self.assertEqual(clean_xml.iloc[0]["input_addresses[]"], ["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"])
        finally:
            os.remove(tmp_xml.name)

if __name__ == "__main__":
    unittest.main()
