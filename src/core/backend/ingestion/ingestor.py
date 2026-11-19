"""
ingestor.py
Handles loading of transaction files (CSV, JSON, XML) offline and runs validation.
Provides a unified interface for data ingestion in the monitoring pipeline.
"""

import os
import pandas as pd
from typing import Tuple, Dict, Any
from src.core.backend.validation.validator import TransactionValidator

class TransactionIngestor:
    def __init__(self, require_hex_txid: bool = True):
        self.validator = TransactionValidator(require_hex_txid=require_hex_txid)

    def detect_format_and_load(self, file_path: str) -> pd.DataFrame:
        """
        Loads the data depending on the file extension (CSV, JSON, XML).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == ".csv":
            # Read CSV. Since array fields are stored as JSON strings or pipe-delimited,
            # we read them as strings so the validator can parse them.
            return pd.read_csv(file_path, comment='#', dtype={
                "input_addresses": str,
                "output_addresses": str,
                "input_amounts": str,
                "output_amounts": str,
                "txid": str,
                "geo_country": str,
                "asn": str,
                "script_type": str,
                "entity_id": str
            })
        elif ext == ".json":
            return pd.read_json(file_path)
        elif ext == ".xml":
            try:
                return pd.read_xml(file_path)
            except ImportError:
                # If lxml is missing, fall back to default parser
                return pd.read_xml(file_path, parser="etree")
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats are .csv, .json, .xml")

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes column names to ensure canonical representation (with brackets for array fields).
        This is particularly useful for formats like XML that do not support brackets in tag names.
        """
        mapping = {
            "input_addresses": "input_addresses[]",
            "output_addresses": "output_addresses[]",
            "input_amounts": "input_amounts[]",
            "output_amounts": "output_amounts[]"
        }
        rename_dict = {}
        for src, dst in mapping.items():
            if src in df.columns and dst not in df.columns:
                rename_dict[src] = dst
        if rename_dict:
            return df.rename(columns=rename_dict)
        return df

    def ingest(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads and validates a file, returning the clean DataFrame and data quality report.
        """
        df = self.detect_format_and_load(file_path)
        df = self.normalize_columns(df)
        cleaned_df, report = self.validator.validate(df)
        
        # Add file details to the quality report
        report["file_name"] = os.path.basename(file_path)
        report["file_size_bytes"] = os.path.getsize(file_path)
        report["file_format"] = os.path.splitext(file_path)[1].upper().replace(".", "")
        
        return cleaned_df, report

if __name__ == "__main__":
    import sys
    
    # Simple test CLI runner
    # Defaulting to the dev dataset generated in Step 1
    from src.config import DEV_DATASET
    default_dev_path = str(DEV_DATASET)
    path_to_load = sys.argv[1] if len(sys.argv) > 1 else default_dev_path
    
    print(f"Running Ingestion on: {path_to_load} ...")
    try:
        ingestor = TransactionIngestor()
        clean_df, report = ingestor.ingest(path_to_load)
        print("\n--- INGESTION SUCCESSFUL ---")
        print(f"File Name: {report['file_name']}")
        print(f"Format: {report['file_format']}")
        print(f"File Size: {report['file_size_bytes']} bytes")
        print(f"Total Rows Processed: {report['total_records']}")
        print(f"Valid Rows Retained: {report['valid_records']}")
        print(f"Invalid Rows Dropped: {report['invalid_records']}")
        print(f"Duplicate Txids Checked: {report['duplicate_txids']}")
        
        print("\nMissing / Imputed Values per Column:")
        for col, count in report['missing_values_imputed'].items():
            if count > 0:
                print(f"  - {col}: {count}")
                
        print("\nInvalid Row Reason Breakdown:")
        for reason, count in report['invalid_reasons'].items():
            if count > 0:
                print(f"  - {reason}: {count}")
                
        print(f"\nShape of Clean Dataframe: {clean_df.shape}")
        print("Data Ingestion Demo Done.")
    except Exception as e:
        print(f"Error during ingestion demo: {e}", file=sys.stderr)
