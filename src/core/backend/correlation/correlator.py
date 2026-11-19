"""
correlator.py
Correlates IP addresses, transactions, wallet addresses, ASNs, and countries.
Generates structured association tables to power investigations.
"""

import os
import pandas as pd
from typing import Dict, Any

from src.config import DEV_DATASET

# Dataset consumed by this pipeline
DATASET_PATH = str(DEV_DATASET)

class TransactionCorrelator:
    """
    Correlates blockchain entities (wallets, transactions) with network entities 
    (IPs, ASNs, Countries).
    
    IMPORTANT:
    These correlations are investigative associations only.
    They represent spatial/temporal co-occurrence in synthetic network traffic 
    and do NOT automatically constitute proof of wallet ownership or criminal activity.
    """
    
    def __init__(self):
        # Disclaimer check attribute for verification
        self.disclaimer = (
            "These associations represent investigative links for decision-support "
            "and do not constitute proof of wallet ownership or malicious activity."
        )

    def correlate(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Creates correlation and association tables between wallets, IPs, ASNs, and countries.
        
        Args:
            df: Validated transaction DataFrame.
            
        Returns:
            Dict containing:
                - 'wallet_network': DataFrame of wallet-IP-ASN-country associations.
                - 'ip_transaction': DataFrame of IP-transaction associations.
        """
        if df.empty:
            return {
                "wallet_network": pd.DataFrame(columns=[
                    "wallet_address", "ip_address", "asn", "geo_country", 
                    "association_count", "first_seen", "last_seen"
                ]),
                "ip_transaction": pd.DataFrame(columns=[
                    "ip_address", "txid", "direction"
                ])
            }

        # 1. Build Wallet Network Associations
        wallet_net_records = []
        
        # 2. Build IP Transaction Associations
        ip_tx_records = []

        for _, row in df.iterrows():
            txid = row["txid"]
            timestamp = row["timestamp"]
            src_ip = row["src_ip"]
            dst_ip = row["dst_ip"]
            asn = row["asn"]
            country = row["geo_country"]
            
            inputs = row["input_addresses[]"]
            outputs = row["output_addresses[]"]
            
            # Map input wallets to sending IP
            for wallet in inputs:
                wallet_net_records.append({
                    "wallet_address": wallet,
                    "ip_address": src_ip,
                    "asn": asn,
                    "geo_country": country,
                    "timestamp": timestamp
                })
                
            # Map output wallets to receiving IP
            for wallet in outputs:
                wallet_net_records.append({
                    "wallet_address": wallet,
                    "ip_address": dst_ip,
                    "asn": asn,
                    "geo_country": country,
                    "timestamp": timestamp
                })
                
            # Map IPs to the transaction propagation
            ip_tx_records.append({
                "ip_address": src_ip,
                "txid": txid,
                "direction": "source"
            })
            ip_tx_records.append({
                "ip_address": dst_ip,
                "txid": txid,
                "direction": "destination"
            })

        # Process wallet network associations DataFrame
        wallet_net_df = pd.DataFrame(wallet_net_records)
        if not wallet_net_df.empty:
            # Group by all identifiers and aggregate count + time range
            wallet_net_agg = wallet_net_df.groupby(
                ["wallet_address", "ip_address", "asn", "geo_country"]
            ).agg(
                association_count=("timestamp", "size"),
                first_seen=("timestamp", "min"),
                last_seen=("timestamp", "max")
            ).reset_index()
            # Sort by frequency
            wallet_net_agg = wallet_net_agg.sort_values(by="association_count", ascending=False).reset_index(drop=True)
        else:
            wallet_net_agg = pd.DataFrame(columns=[
                "wallet_address", "ip_address", "asn", "geo_country", 
                "association_count", "first_seen", "last_seen"
            ])

        # Process IP transaction associations DataFrame
        ip_tx_df = pd.DataFrame(ip_tx_records)
        if not ip_tx_df.empty:
            ip_tx_df = ip_tx_df.drop_duplicates().reset_index(drop=True)
        else:
            ip_tx_df = pd.DataFrame(columns=["ip_address", "txid", "direction"])

        return {
            "wallet_network": wallet_net_agg,
            "ip_transaction": ip_tx_df
        }
