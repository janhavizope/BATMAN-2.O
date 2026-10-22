"""
GUI Data Manager
----------------
Central facade for loading data into the GUI.

Supports two modes:
1. DEV mode  — loads from src.desktop_gui.dev_data (placeholder).
2. LIVE mode — loads from src.core.backend.* modules (future).

The DataManager does NOT perform analytical computation.
It only loads and converts data into schema objects.

DO NOT implement ML, risk scoring, or correlation here.
"""

from __future__ import annotations

from src.desktop_gui.data.schemas import (
    OverviewStats, AlertRecord, EntityProfile,
    TransactionRecord, GraphData, ExplanationRecord,
)


class DataManager:
    """Single point of truth for GUI data access."""

    def __init__(self):
        self._overview: OverviewStats | None = None
        self._alerts: list[AlertRecord] = []
        self._entities: dict[str, EntityProfile] = {}
        self._transactions: list[TransactionRecord] = []
        self._graph: GraphData | None = None
        self._explanations: dict[str, ExplanationRecord] = {}
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Load from backend (future)
    # ------------------------------------------------------------------
    def load_from_backend(self):
        """
        Load data produced by the backend modules.
        """
        if self._loaded:
            return

        import os
        import json
        import pandas as pd
        from pathlib import Path
        from src.core.backend.ingestion.ingestor import TransactionIngestor

        try:
            ROOT_DIR = Path(__file__).parent.parent.parent.parent
            DEV_DIR = ROOT_DIR / "data" / "dev"
            results_path = DEV_DIR / "results.json"
            dataset_path = DEV_DIR / "dev_dataset_50k.csv"

            if not results_path.exists() or not dataset_path.exists():
                print(f"[BATMAN INFO] No bundled dataset found at {DEV_DIR}. Initializing DataManager in empty state.")
                self._overview = OverviewStats(
                    total_transactions=0,
                    total_wallets=0,
                    total_ips=0,
                    anomalous_entities=0,
                    high_risk_entities=0,
                    critical_alerts=0
                )
                self._alerts = []
                self._transactions = []
                self._graph = GraphData(nodes=[], edges=[])
                self._scoreboard = {}
                self._raw_results = {}
                self._entity_ids = []
                self._loaded = True
                return

            # 1. Load results.json (Wallet-level analytics)
            with open(results_path, "r", encoding="utf-8") as f:
                results_data = json.load(f)

            # Store raw results indexed by entity_id for lazy loading
            self._raw_results = {}
            for r in results_data:
                eid = r.get("wallet_address", "")
                if eid:
                    self._raw_results[eid] = r
            
            # Sort IDs by risk score descending for general list retrieval
            results_sorted = sorted(results_data, key=lambda x: float(x.get("risk_score", 0)), reverse=True)
            self._entity_ids = [r.get("wallet_address", "") for r in results_sorted if r.get("wallet_address")]

            # 2. Load dataset (Transaction-level data)
            ingestor = TransactionIngestor()
            df = ingestor.detect_format_and_load(str(dataset_path))
            df = ingestor.normalize_columns(df)

            total_tx = len(df)
            total_wallets = len(self._raw_results)
            
            # IPs are src_ip and dst_ip
            total_ips = pd.concat([df["src_ip"], df["dst_ip"]]).nunique()

            critical_alerts = sum(1 for r in results_data if str(r.get("risk_level", "")).upper() == "CRITICAL")
            high_risk = sum(1 for r in results_data if str(r.get("risk_level", "")).upper() == "HIGH")
            anomalous = sum(1 for r in results_data if float(r.get("anomaly_score", 0)) < 0)

            self._overview = OverviewStats(
                total_transactions=total_tx,
                total_wallets=total_wallets,
                total_ips=total_ips,
                anomalous_entities=anomalous,
                high_risk_entities=high_risk,
                critical_alerts=critical_alerts
            )

            # --- Build Alerts ---
            self._alerts = []
            rank = 1
            for r in results_sorted:
                r_level = str(r.get("risk_level", "")).upper()
                if r_level in ("CRITICAL", "HIGH"):
                    reasons = r.get("top_reasons", [])
                    main_reason = reasons[0] if reasons else "Anomalous behavior detected"
                    self._alerts.append(AlertRecord(
                        rank=rank,
                        entity_id=r.get("wallet_address", ""),
                        risk_score=int(r.get("risk_score", 0)),
                        risk_level=r_level.title(),  # Convert to title case for GUI compatibility
                        main_reason=main_reason
                    ))
                    rank += 1

            # --- Build Transactions ---
            self._transactions = []
            # We load ALL 50k transactions into memory as requested.
            for _, row in df.iterrows():
                in_addrs = str(row.get("input_addresses[]", "")).split("|")
                out_addrs = str(row.get("output_addresses[]", "")).split("|")
                
                # Try to sum amounts if available, else 0
                in_amounts_str = str(row.get("input_amounts[]", "0"))
                try:
                    amt = sum(float(x) for x in in_amounts_str.split("|") if x.strip())
                except ValueError:
                    amt = 0.0

                # Convert ISO timestamp to 'YYYY-MM-DD HH:MM:SS' format for GUI
                ts_iso = str(row.get("timestamp", ""))
                ts_gui = ts_iso.replace("T", " ").replace("Z", "") if "T" in ts_iso else ts_iso

                flag = "Suspicious" if str(row.get("is_suspicious", "")).lower() == "true" else "Normal"
                
                self._transactions.append(TransactionRecord(
                    tx_id=str(row.get("txid", "")),
                    amount_btc=amt,
                    timestamp=ts_gui,
                    source=in_addrs[0] if in_addrs and in_addrs[0] else "Unknown",
                    destination=out_addrs[0] if out_addrs and out_addrs[0] else "Unknown",
                    flag=flag
                ))

            # --- Build Graph Data ---
            from src.desktop_gui.data.schemas import GraphNode, GraphEdge
            # Bound graph to top 15 entities to prevent GUI from freezing
            top_wallets = self._entity_ids[:15]
            nodes_dict = {}
            edges = []
            
            # Simple iteration to build small ego-network for top wallets
            for _, row in df.iterrows():
                in_addrs = str(row.get("input_addresses[]", "")).split("|")
                out_addrs = str(row.get("output_addresses[]", "")).split("|")
                
                involved = any(w in in_addrs or w in out_addrs for w in top_wallets)
                if involved:
                    tx_id = str(row.get("txid", ""))
                    src_ip = str(row.get("src_ip", ""))
                    
                    # Transactions and IPs have no inherent risk_score in this model; use 0
                    nodes_dict[tx_id] = GraphNode(id=tx_id, type="Transaction", label=tx_id[:6], risk_score=0)
                    if src_ip:
                        nodes_dict[src_ip] = GraphNode(id=src_ip, type="IP", label=src_ip, risk_score=0)
                    
                    for w in in_addrs:
                        if w:
                            w_risk = int(self._raw_results.get(w, {}).get("risk_score", 0))
                            nodes_dict[w] = GraphNode(id=w, type="Wallet", label=w[:6], risk_score=w_risk)
                            edges.append(GraphEdge(source=w, target=tx_id, relation="sent"))
                            if src_ip:
                                edges.append(GraphEdge(source=src_ip, target=w, relation="associated"))
                            
                    for w in out_addrs:
                        if w:
                            w_risk = int(self._raw_results.get(w, {}).get("risk_score", 0))
                            nodes_dict[w] = GraphNode(id=w, type="Wallet", label=w[:6], risk_score=w_risk)
                            edges.append(GraphEdge(source=tx_id, target=w, relation="received"))
            
            # Deduplicate edges using their source/target/relation signatures
            seen_edges = set()
            unique_edges = []
            for e in edges:
                sig = f"{e.source}-{e.target}-{e.relation}"
                if sig not in seen_edges:
                    seen_edges.add(sig)
                    unique_edges.append(e)

            self._graph = GraphData(nodes=list(nodes_dict.values()), edges=unique_edges)

            # --- Build Scoreboard ---
            self._scoreboard = {}
            df["month"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m")
            all_months = sorted(df["month"].dropna().unique())
            
            for w in top_wallets:
                raw = self._raw_results.get(w, {})
                w_risk = int(raw.get("risk_score", 0))
                
                # For scoreboard visualization, use exact matching via split
                w_df = df[
                    df["input_addresses[]"].fillna("").apply(lambda x: w in x.split("|")) | 
                    df["output_addresses[]"].fillna("").apply(lambda x: w in x.split("|"))
                ]
                monthly_counts = w_df.groupby("month").size().to_dict()
                
                cumulative = 0
                timeline = []
                for m in all_months:
                    cumulative += monthly_counts.get(m, 0) * w_risk
                    timeline.append({"time": m, "score": cumulative})
                    
                self._scoreboard[f"Wallet ({w[:6]})"] = timeline

            self._loaded = True
        except Exception as e:
            print("ERROR loading from backend:", e)
            import traceback
            traceback.print_exc()
            raise RuntimeError("Failed to load real data from ML backend") from e

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_overview(self) -> OverviewStats:
        if self._overview is None:
            self.load_from_backend()
        return self._overview  # type: ignore[return-value]

    def get_alerts(self) -> list[AlertRecord]:
        if not self._alerts:
            self.load_from_backend()
        return list(self._alerts)

    def get_entity(self, entity_id: str) -> EntityProfile | None:
        if not self._loaded:
            self.load_from_backend()
            
        r = self._raw_results.get(entity_id)
        if not r:
            return None
            
        r_level = str(r.get("risk_level", "")).upper().title()
        reasons = r.get("top_reasons", [])
        
        evidence = [f"Risk level: {r_level}", f"Total tx: {r.get('total_tx_count', 0)}"] + reasons
        
        return EntityProfile(
            entity_id=entity_id,
            risk_score=int(r.get("risk_score", 0)),
            risk_level=r_level,
            anomaly_score=float(r.get("anomaly_score", 0.0)),
            transaction_count=int(r.get("total_tx_count", 0)),
            counterparty_count=int(r.get("unique_counterparties", 0)),
            timeline=[], # Can be populated dynamically if needed
            network_associations=[],
            evidence=evidence,
            explanation=f"Wallet flagged based on anomaly and behavior scores."
        )

    def get_entity_ids(self) -> list[str]:
        if not self._loaded:
            self.load_from_backend()
        return self._entity_ids

    def get_transactions(self) -> list[TransactionRecord]:
        if not self._transactions:
            self.load_from_backend()
        return list(self._transactions)

    def get_graph(self) -> GraphData:
        if self._graph is None:
            self.load_from_backend()
        return self._graph  # type: ignore[return-value]

    def get_scoreboard(self) -> dict:
        if not hasattr(self, '_scoreboard') or not self._scoreboard:
            self.load_from_backend()
        return self._scoreboard

    def get_explanation(self, entity_id: str) -> ExplanationRecord | None:
        if not self._loaded:
            self.load_from_backend()
            
        r = self._raw_results.get(entity_id)
        if not r:
            return None
            
        r_level = str(r.get("risk_level", "")).upper().title()
        reasons = r.get("top_reasons", [])
        
        evidence = [f"Risk level: {r_level}", f"Total tx: {r.get('total_tx_count', 0)}"] + reasons
        
        return ExplanationRecord(
            entity_id=entity_id,
            anomaly_score=float(r.get("anomaly_score", 0.0)),
            risk_score=int(r.get("risk_score", 0)),
            top_features=[{"feature": reason, "contribution": 1.0} for reason in reasons],
            evidence=evidence,
            human_reason=f"Wallet flagged due to {len(reasons)} significant risk signals."
        )

    def get_explanation_ids(self) -> list[str]:
        if not self._loaded:
            self.load_from_backend()
        return self._entity_ids

    def is_data_available(self) -> bool:
        """Return True if transaction dataset is present in memory."""
        return bool(self._transactions or self._raw_results)

    # Legacy alias for test compatibility
    load_dev_data = load_from_backend
