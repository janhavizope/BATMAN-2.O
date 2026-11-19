"""
GUI Data Manager
----------------
Central facade for loading data into the desktop GUI.

The desktop app and the Streamlit app share one data layer: this class
reads everything from src.data.data_manager (the Elliptic++ risk scores in
data/elliptic_plus/) and only converts it into the GUI schema objects.

DO NOT implement ML, risk scoring, or correlation here.
"""

from __future__ import annotations

from src.data import data_manager as data
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
        self._entity_ids: list[str] = []
        self._transactions: list[TransactionRecord] = []
        self._graph: GraphData | None = None
        self._scoreboard: dict = {}
        self._activity: tuple[list, list] = ([], [])
        self._explanations: dict[str, ExplanationRecord] = {}
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Load from the shared data layer
    # ------------------------------------------------------------------
    def load_from_backend(self):
        """Load the risk scores through src.data and convert them to schema objects."""
        if self._loaded:
            return

        if not data.is_data_available():
            print("[BATMAN INFO] No risk scores found in data/elliptic_plus/. "
                  "Initializing DataManager in empty state.")
            self._overview = OverviewStats()
            self._graph = GraphData(nodes=[], edges=[])
            self._loaded = True
            return

        try:
            self._overview = OverviewStats.from_dict(data.get_overview_stats())

            self._alerts = [
                AlertRecord(
                    rank=int(row["Rank"]),
                    entity_id=str(row["Full Entity ID"]),
                    risk_score=int(row["Risk Score"]),
                    risk_level=str(row["Risk Level"]),
                    main_reason=str(row["Main Reason"]),
                )
                for row in data.get_alerts()
            ]

            self._transactions = [TransactionRecord.from_dict(row) for row in data.get_transactions()]

            nodes, edges, _, _ = data.get_graph_data()
            self._graph = GraphData.from_dict({"nodes": nodes, "edges": edges})

            self._scoreboard = data.get_scoreboard()
            self._activity = data.get_activity_timeline()

            self._entities = {eid: EntityProfile.from_dict(e) for eid, e in data.get_entities().items()}
            self._entity_ids = list(self._entities)

            self._explanations = {
                eid: ExplanationRecord.from_dict({"entity_id": eid, **e})
                for eid, e in data.get_explainability_data().items()
            }
            self._loaded = True
        except Exception as e:
            raise RuntimeError("Failed to load risk scores from src.data") from e

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_overview(self) -> OverviewStats:
        if self._overview is None:
            self.load_from_backend()
        return self._overview  # type: ignore[return-value]

    def get_alerts(self) -> list[AlertRecord]:
        if not self._loaded:
            self.load_from_backend()
        return list(self._alerts)

    def get_entity(self, entity_id: str) -> EntityProfile | None:
        if not self._loaded:
            self.load_from_backend()
        return self._entities.get(entity_id)

    def get_entity_ids(self) -> list[str]:
        if not self._loaded:
            self.load_from_backend()
        return self._entity_ids

    def get_transactions(self) -> list[TransactionRecord]:
        if not self._loaded:
            self.load_from_backend()
        return list(self._transactions)

    def get_graph(self) -> GraphData:
        if self._graph is None:
            self.load_from_backend()
        return self._graph  # type: ignore[return-value]

    def get_scoreboard(self) -> dict:
        if not self._loaded:
            self.load_from_backend()
        return self._scoreboard

    def get_activity_timeline(self) -> tuple[list, list]:
        """(time step labels, flagged transaction counts) for the Overview chart."""
        if not self._loaded:
            self.load_from_backend()
        return self._activity

    def get_explanation(self, entity_id: str) -> ExplanationRecord | None:
        if not self._loaded:
            self.load_from_backend()
        return self._explanations.get(entity_id)

    def get_explanation_ids(self) -> list[str]:
        if not self._loaded:
            self.load_from_backend()
        return list(self._explanations)

    def is_data_available(self) -> bool:
        """Return True if risk scores are loaded."""
        return bool(self._transactions or self._entities)

    # Legacy alias for test compatibility
    load_dev_data = load_from_backend
