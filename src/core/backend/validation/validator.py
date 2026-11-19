"""
validator.py
Validates raw transaction dataframes, checks schemas, handles missing values,
detects duplicate txids, enforces types, and generates a data-quality report.
"""

import re
import json
import ipaddress
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

REQUIRED_COLUMNS = [
    "timestamp", "txid",
    "input_addresses[]", "output_addresses[]", "input_amounts[]", "output_amounts[]",
]

# Network metadata. Real on-chain data has none of these, so they are optional:
# a missing column is added as empty, and an empty value is not an error.
NETWORK_COLUMNS = ["src_ip", "dst_ip", "src_port", "dst_port", "geo_country", "asn"]

RECOMMENDED_COLUMNS = [
    "fee", "script_type", "block_height", "block_timestamp", "connection_duration", "entity_id"
]

# Regular expression for a valid Bitcoin txid (64 hex characters)
TXID_REGEX = re.compile(r"^[0-9a-fA-F]{64}$")

class TransactionValidator:
    def __init__(self, require_hex_txid: bool = True):
        # Real Bitcoin txids are 64 hex characters; datasets such as Elliptic
        # use anonymised integer ids instead.
        self.require_hex_txid = require_hex_txid
        
    def parse_array_field(self, val: Any) -> list:
        """Helper to parse list/array fields from JSON strings, lists, or nan."""
        # Lists first: pd.isna() on a multi-element list returns an array
        if isinstance(val, (list, np.ndarray)):
            return list(val)
        if pd.isna(val):
            return []
        if isinstance(val, str):
            val_str = val.strip()
            if not val_str:
                return []
            try:
                parsed = json.loads(val_str)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Fallback split if it's a comma-separated or pipe-separated string without brackets
            cleaned = val_str.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
            if cleaned:
                if "|" in cleaned:
                    return [item.strip() for item in cleaned.split("|")]
                return [item.strip() for item in cleaned.split(",")]
        return []

    def is_valid_ip(self, ip_str: Any) -> bool:
        """Verify if string is a valid IPv4 or IPv6 address or a synthetic ip."""
        if not isinstance(ip_str, str):
            return False
        ip_str = ip_str.strip()
        if ip_str.startswith("ip_"):
            return True
        try:
            ipaddress.ip_address(ip_str)
            return True
        except ValueError:
            return False

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates the input DataFrame and generates a data quality report.
        
        Args:
            df: Input pandas DataFrame to validate.
            
        Returns:
            A tuple containing:
                - The validated, cleaned and parsed DataFrame.
                - A dictionary containing the data quality report.
        """
        # Make a copy to avoid mutating the original
        working_df = df.copy()
        
        report = {
            "total_records": len(df),
            "valid_records": 0,
            "invalid_records": 0,
            "duplicate_txids": 0,
            "missing_values_imputed": {},
            "invalid_reasons": {
                "missing_required_columns": 0,
                "missing_critical_fields": 0,
                "invalid_ip_format": 0,
                "invalid_port_range": 0,
                "invalid_txid_format": 0,
                "invalid_array_format": 0,
                "negative_amount_or_fee": 0,
                "empty_addresses_or_amounts": 0,
                "inconsistent_amounts": 0
            }
        }
        
        # 1. Check for required columns
        missing_required = [col for col in REQUIRED_COLUMNS if col not in working_df.columns]
        if missing_required:
            report["invalid_reasons"]["missing_required_columns"] = len(working_df)
            report["invalid_records"] = len(working_df)
            raise ValueError(f"Input DataFrame is missing required columns: {missing_required}")
            
        # Ensure optional columns exist, initialize them as NaN if not present
        for col in NETWORK_COLUMNS + RECOMMENDED_COLUMNS:
            if col not in working_df.columns:
                working_df[col] = np.nan
                
        # Initialize counts for missing values
        for col in list(working_df.columns):
            report["missing_values_imputed"][col] = int(working_df[col].isna().sum())

        # 2. Flag and separate records with missing critical fields (timestamp, txid)
        critical_mask = working_df["timestamp"].isna() | working_df["txid"].isna()
        critical_missing_count = int(critical_mask.sum())
        report["invalid_reasons"]["missing_critical_fields"] += critical_missing_count
        
        # Filter out records missing critical fields early
        working_df = working_df[~critical_mask].copy()
        
        if len(working_df) == 0:
            report["invalid_records"] = report["total_records"]
            return working_df, report

        # 3. Handle duplicates
        # Find duplicates on txid (if txid is present)
        duplicate_mask = working_df.duplicated(subset=["txid"], keep="first")
        duplicate_count = int(duplicate_mask.sum())
        report["duplicate_txids"] = duplicate_count
        # We keep the first duplicate but log the warning.
        # Alternatively, we could drop them, but keeping the first is standard for ingestion.
        
        # 4. Cast and validate individual rows
        valid_indices = []
        
        # Prepare list fields
        array_cols = ["input_addresses[]", "output_addresses[]", "input_amounts[]", "output_amounts[]"]
        for col in array_cols:
            working_df[col] = working_df[col].apply(self.parse_array_field)
            
        # Process and validate row by row
        for idx, row in working_df.iterrows():
            is_valid = True
            reasons = []
            
            # Validate IP formats (when present)
            ips = [row[c] for c in ("src_ip", "dst_ip") if not pd.isna(row[c])]
            if not all(self.is_valid_ip(ip) for ip in ips):
                is_valid = False
                reasons.append("invalid_ip_format")

            # Validate Ports (when present)
            try:
                ports = [int(row[c]) for c in ("src_port", "dst_port") if not pd.isna(row[c])]
                if not all(1 <= port <= 65535 for port in ports):
                    is_valid = False
                    reasons.append("invalid_port_range")
            except (ValueError, TypeError):
                is_valid = False
                reasons.append("invalid_port_range")

            # Validate txid format
            if self.require_hex_txid and (not isinstance(row["txid"], str) or not TXID_REGEX.match(row["txid"])):
                is_valid = False
                reasons.append("invalid_txid_format")
                
            # Validate lists are not empty
            inputs = row["input_addresses[]"]
            outputs = row["output_addresses[]"]
            in_amounts = row["input_amounts[]"]
            out_amounts = row["output_amounts[]"]
            
            if not inputs or not outputs or not in_amounts or not out_amounts:
                is_valid = False
                reasons.append("empty_addresses_or_amounts")
            else:
                # Ensure input amounts and output amounts are numeric and positive
                try:
                    in_amounts = [float(x) for x in in_amounts]
                    out_amounts = [float(x) for x in out_amounts]
                    
                    # Update row values to clean lists of floats
                    working_df.at[idx, "input_amounts[]"] = in_amounts
                    working_df.at[idx, "output_amounts[]"] = out_amounts
                    
                    if any(x < 0 for x in in_amounts) or any(x < 0 for x in out_amounts):
                        is_valid = False
                        reasons.append("negative_amount_or_fee")
                except (ValueError, TypeError):
                    is_valid = False
                    reasons.append("invalid_array_format")
            
            # Validate fee
            fee = row["fee"]
            if not pd.isna(fee):
                try:
                    fee_val = float(fee)
                    working_df.at[idx, "fee"] = fee_val
                    if fee_val < 0:
                        is_valid = False
                        reasons.append("negative_amount_or_fee")
                except (ValueError, TypeError):
                    working_df.at[idx, "fee"] = 0.0 # Default / impute
            
            # Check transaction amount consistency: sum(inputs) approx sum(outputs) + fee
            if is_valid:
                # Check sums match up to float precision (e.g. 1e-5 tolerance)
                sum_in = sum(in_amounts)
                sum_out = sum(out_amounts)
                fee_val = float(row["fee"]) if not pd.isna(row["fee"]) else 0.0
                
                if abs(sum_in - (sum_out + fee_val)) > 1e-5:
                    # Inconsistent amounts, we log it but do not necessarily discard the record
                    # since some synthetic generators might have minor float issues, but let's log it
                    report["invalid_reasons"]["inconsistent_amounts"] += 1
            
            # Record status
            if is_valid:
                valid_indices.append(idx)
            else:
                # Log first reason found for statistical tracking
                if reasons:
                    report["invalid_reasons"][reasons[0]] += 1

        # Keep only valid records in the final dataframe
        cleaned_df = working_df.loc[valid_indices].copy()
        
        # 5. Type Casting and Imputations on the cleaned dataframe
        # Cast timestamp to datetime
        cleaned_df["timestamp"] = pd.to_datetime(cleaned_df["timestamp"])
        
        # Imputations for missing categorical / recommended values
        # geo_country
        cleaned_df["geo_country"] = cleaned_df["geo_country"].fillna("UNKNOWN").astype(str)
        # asn
        cleaned_df["asn"] = cleaned_df["asn"].fillna("UNKNOWN").astype(str)
        # script_type
        cleaned_df["script_type"] = cleaned_df["script_type"].fillna("UNKNOWN").astype(str)
        # entity_id
        cleaned_df["entity_id"] = cleaned_df["entity_id"].fillna("UNKNOWN").astype(str)
        
        # connection_duration
        if cleaned_df["connection_duration"].isna().any():
            median_conn = cleaned_df["connection_duration"].median()
            if pd.isna(median_conn):
                median_conn = 0.0
            cleaned_df["connection_duration"] = cleaned_df["connection_duration"].fillna(median_conn)
            
        # block_height
        cleaned_df["block_height"] = cleaned_df["block_height"].fillna(0).astype(int)
        
        # block_timestamp
        cleaned_df["block_timestamp"] = pd.to_datetime(cleaned_df["block_timestamp"])
        
        # fee
        cleaned_df["fee"] = cleaned_df["fee"].fillna(0.0).astype(float)
        
        # Update valid and invalid counts
        report["valid_records"] = len(cleaned_df)
        report["invalid_records"] = report["total_records"] - len(cleaned_df)
        
        return cleaned_df, report
