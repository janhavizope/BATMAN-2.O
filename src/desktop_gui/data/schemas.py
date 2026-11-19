"""
GUI Data Schemas
----------------
Typed dictionary contracts for data flowing into the GUI.

These schemas define the interface between the GUI and any data source
(dev_data, backend modules, or Nutan's dataset loader).  The GUI only
reads these structures — it never produces analytical values.

DO NOT add ML, risk calculation, or correlation logic here.
"""

from __future__ import annotations
from typing import Any


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
class OverviewStats:
    """Aggregate dashboard metrics."""

    def __init__(
        self,
        total_transactions: int = 0,
        total_wallets: int = 0,
        total_ips: int = 0,
        anomalous_entities: int = 0,
        high_risk_entities: int = 0,
        critical_alerts: int = 0,
        known_illicit: int = 0,
    ):
        self.total_transactions = total_transactions
        self.total_wallets = total_wallets
        self.total_ips = total_ips
        self.known_illicit = known_illicit
        self.anomalous_entities = anomalous_entities
        self.high_risk_entities = high_risk_entities
        self.critical_alerts = critical_alerts

    def to_dict(self) -> dict[str, int]:
        return {
            "total_transactions": self.total_transactions,
            "total_wallets": self.total_wallets,
            "total_ips": self.total_ips,
            "anomalous_entities": self.anomalous_entities,
            "high_risk_entities": self.high_risk_entities,
            "critical_alerts": self.critical_alerts,
            "known_illicit": self.known_illicit,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OverviewStats":
        return cls(
            total_transactions=int(d.get("total_transactions", 0)),
            total_wallets=int(d.get("total_wallets", 0)),
            total_ips=int(d.get("total_ips", 0)),
            anomalous_entities=int(d.get("anomalous_entities", 0)),
            high_risk_entities=int(d.get("high_risk_entities", 0)),
            critical_alerts=int(d.get("critical_alerts", 0)),
            known_illicit=int(d.get("known_illicit", 0)),
        )


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------
class AlertRecord:
    """Single alert row for the Alerts page."""

    def __init__(
        self,
        rank: int = 0,
        entity_id: str = "",
        risk_score: int = 0,
        risk_level: str = "",
        main_reason: str = "",
    ):
        self.rank = rank
        self.entity_id = entity_id
        self.risk_score = risk_score
        self.risk_level = risk_level
        self.main_reason = main_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "Rank": self.rank,
            "Entity ID": self.entity_id,
            "Risk Score": self.risk_score,
            "Risk Level": self.risk_level,
            "Main Reason": self.main_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AlertRecord":
        return cls(
            rank=int(d.get("Rank", 0)),
            entity_id=str(d.get("Entity ID", "")),
            risk_score=int(d.get("Risk Score", 0)),
            risk_level=str(d.get("Risk Level", "")),
            main_reason=str(d.get("Main Reason", "")),
        )


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------
class EntityProfile:
    """Full entity investigation profile."""

    def __init__(
        self,
        entity_id: str = "",
        risk_score: int = 0,
        risk_level: str = "",
        anomaly_score: float = 0.0,
        transaction_count: int = 0,
        counterparty_count: int = 0,
        timeline: list[dict[str, str]] | None = None,
        network_associations: list[dict[str, str]] | None = None,
        evidence: list[str] | None = None,
        top_features: list[dict[str, Any]] | None = None,
        explanation: str = "",
    ):
        self.entity_id = entity_id
        self.risk_score = risk_score
        self.risk_level = risk_level
        self.anomaly_score = anomaly_score
        self.transaction_count = transaction_count
        self.counterparty_count = counterparty_count
        self.timeline = timeline or []
        self.network_associations = network_associations or []
        self.evidence = evidence or []
        self.top_features = top_features or []
        self.explanation = explanation

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EntityProfile":
        return cls(
            entity_id=str(d.get("entity_id", "")),
            risk_score=int(d.get("risk_score", 0)),
            risk_level=str(d.get("risk_level", "")),
            anomaly_score=float(d.get("anomaly_score", 0.0)),
            transaction_count=int(d.get("transaction_count", 0)),
            counterparty_count=int(d.get("counterparty_count", 0)),
            timeline=list(d.get("timeline", [])),
            network_associations=list(d.get("network_associations", [])),
            evidence=list(d.get("evidence", [])),
            top_features=list(d.get("top_features", [])),
            explanation=str(d.get("explanation", "")),
        )


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------
class TransactionRecord:
    """Single transaction record."""

    def __init__(
        self,
        tx_id: str = "",
        amount_btc: float = 0.0,
        timestamp: str = "",
        source: str = "",
        destination: str = "",
        flag: str = "",
    ):
        self.tx_id = tx_id
        self.amount_btc = amount_btc
        self.timestamp = timestamp
        self.source = source
        self.destination = destination
        self.flag = flag

    def to_dict(self) -> dict[str, Any]:
        return {
            "Transaction ID": self.tx_id,
            "Amount (BTC)": self.amount_btc,
            "Timestamp": self.timestamp,
            "Source": self.source,
            "Destination": self.destination,
            "Flag": self.flag,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TransactionRecord":
        return cls(
            tx_id=str(d.get("Transaction ID", d.get("tx_id", ""))),
            amount_btc=float(d.get("Amount (BTC)", d.get("amount_btc", 0.0))),
            timestamp=str(d.get("Timestamp", d.get("timestamp", ""))),
            source=str(d.get("Source", d.get("source", ""))),
            destination=str(d.get("Destination", d.get("destination", ""))),
            flag=str(d.get("Flag", d.get("flag", ""))),
        )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
class GraphNode:
    def __init__(self, id: str = "", type: str = "", label: str = "", risk_score: int = 0):
        self.id = id
        self.type = type
        self.label = label
        self.risk_score = risk_score

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "label": self.label, "risk_score": self.risk_score}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GraphNode":
        return cls(id=d["id"], type=d.get("type", ""), label=d.get("label", d["id"]), risk_score=int(d.get("risk_score", 0)))


class GraphEdge:
    def __init__(self, source: str = "", target: str = "", relation: str = ""):
        self.source = source
        self.target = target
        self.relation = relation

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target, "relation": self.relation}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "GraphEdge":
        return cls(source=d["source"], target=d["target"], relation=d.get("relation", ""))


class GraphData:
    """Complete graph payload for the Network Graph page."""

    def __init__(self, nodes: list[GraphNode] | None = None, edges: list[GraphEdge] | None = None):
        self.nodes = nodes or []
        self.edges = edges or []

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GraphData":
        return cls(
            nodes=[GraphNode.from_dict(n) for n in d.get("nodes", [])],
            edges=[GraphEdge.from_dict(e) for e in d.get("edges", [])],
        )


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------
class ExplanationRecord:
    """Explanation payload for the Explainability page."""

    def __init__(
        self,
        entity_id: str = "",
        anomaly_score: float = 0.0,
        risk_score: int = 0,
        top_features: list[dict[str, Any]] | None = None,
        evidence: list[str] | None = None,
        human_reason: str = "",
    ):
        self.entity_id = entity_id
        self.anomaly_score = anomaly_score
        self.risk_score = risk_score
        self.top_features = top_features or []
        self.evidence = evidence or []
        self.human_reason = human_reason

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExplanationRecord":
        return cls(
            entity_id=d.get("entity_id", ""),
            anomaly_score=float(d.get("anomaly_score", 0.0)),
            risk_score=int(d.get("risk_score", 0)),
            top_features=list(d.get("top_features", [])),
            evidence=list(d.get("evidence", [])),
            human_reason=str(d.get("human_reason", "")),
        )
