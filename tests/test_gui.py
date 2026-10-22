"""GUI-level tests for the data/state layer and page wiring."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PySide6.QtWidgets import QApplication

from src.desktop_gui.data.schemas import (
    OverviewStats, AlertRecord, EntityProfile, TransactionRecord,
    GraphData, GraphNode, GraphEdge, ExplanationRecord,
)
from src.desktop_gui.data.data_manager import DataManager
from src.desktop_gui.data.filters import filter_alerts, filter_transactions, filter_graph_nodes
from src.desktop_gui.state.app_state import AppState


@pytest.fixture(scope="session")
def app():
    """Create a single QApplication for all tests."""
    return QApplication.instance() or QApplication([])


# ── Schema Tests ──

class TestSchemas:
    def test_overview_stats(self):
        s = OverviewStats(total_transactions=100, total_wallets=10,
                          total_ips=5, anomalous_entities=3,
                          high_risk_entities=2, critical_alerts=1)
        assert s.total_transactions == 100
        assert s.to_dict()["total_wallets"] == 10

    def test_overview_from_dict(self):
        s = OverviewStats.from_dict({"total_transactions": 42, "total_wallets": 7})
        assert s.total_transactions == 42
        assert s.total_wallets == 7
        assert s.total_ips == 0

    def test_alert_record(self):
        a = AlertRecord(rank=1, entity_id="W001", risk_score=94,
                        risk_level="Critical", main_reason="high fan-out")
        assert a.rank == 1
        assert a.entity_id == "W001"
        d = a.to_dict()
        assert d["Risk Level"] == "Critical"

    def test_alert_from_dict(self):
        a = AlertRecord.from_dict({"Rank": 2, "Entity ID": "W017", "Risk Score": 80,
                                   "Risk Level": "High", "Main Reason": "test"})
        assert a.rank == 2
        assert a.entity_id == "W017"

    def test_graph_node(self):
        n = GraphNode(id="N1", type="Wallet", label="wallet1", risk_score=70)
        assert n.id == "N1"
        assert n.type == "Wallet"
        d = n.to_dict()
        assert d["label"] == "wallet1"

    def test_graph_edge(self):
        e = GraphEdge(source="N1", target="N2", relation="sent")
        assert e.source == "N1"
        assert e.relation == "sent"

    def test_transaction_record(self):
        t = TransactionRecord(tx_id="TX1", amount_btc=1.5, source="W001",
                              destination="W017", flag="high_risk")
        assert t.tx_id == "TX1"
        assert t.amount_btc == 1.5
        d = t.to_dict()
        assert d["Source"] == "W001"

    def test_entity_profile(self):
        ep = EntityProfile(entity_id="W001", risk_score=94, risk_level="Critical",
                           transaction_count=50, counterparty_count=12)
        assert ep.entity_id == "W001"
        assert ep.risk_score == 94

    def test_explanation_record(self):
        er = ExplanationRecord(entity_id="W001", anomaly_score=0.91,
                               risk_score=94, human_reason="high fan-out")
        assert er.entity_id == "W001"
        assert er.anomaly_score == 0.91

    def test_graph_data(self):
        gd = GraphData(
            nodes=[GraphNode(id="N1", type="Wallet")],
            edges=[GraphEdge(source="N1", target="N2")],
        )
        assert len(gd.nodes) == 1
        assert len(gd.edges) == 1


# ── DataManager Tests ──

class TestDataManager:
    def test_load_dev_data(self):
        dm = DataManager()
        dm.load_dev_data()
        assert dm.loaded is True

    def test_overview(self, app):
        dm = DataManager()
        dm.load_dev_data()
        ov = dm.get_overview()
        assert ov is not None
        assert ov.total_transactions >= 0
        assert ov.total_wallets >= 0

    def test_alerts(self, app):
        dm = DataManager()
        dm.load_dev_data()
        alerts = dm.get_alerts()
        assert isinstance(alerts, list)
        assert all(isinstance(a, AlertRecord) for a in alerts)

    def test_entity_ids(self, app):
        dm = DataManager()
        dm.load_dev_data()
        ids = dm.get_entity_ids()
        assert isinstance(ids, list)
        assert all(isinstance(i, str) for i in ids)

    def test_entity_profile(self, app):
        dm = DataManager()
        dm.load_dev_data()
        ids = dm.get_entity_ids()
        if ids:
            profile = dm.get_entity(ids[0])
            assert profile is not None
            assert profile.entity_id == ids[0]
        else:
            assert dm.get_entity("NONEXISTENT") is None

    def test_transactions(self, app):
        dm = DataManager()
        dm.load_dev_data()
        txns = dm.get_transactions()
        assert isinstance(txns, list)
        assert all(isinstance(t, TransactionRecord) for t in txns)

    def test_graph(self, app):
        dm = DataManager()
        dm.load_dev_data()
        g = dm.get_graph()
        assert g is not None
        assert isinstance(g.nodes, list)
        assert isinstance(g.edges, list)

    def test_explanations(self, app):
        dm = DataManager()
        dm.load_dev_data()
        ids = dm.get_explanation_ids()
        assert isinstance(ids, list)
        for eid in ids:
            ex = dm.get_explanation(eid)
            assert ex is not None
            assert isinstance(ex, ExplanationRecord)
            assert ex.entity_id == eid

    def test_loaded_flag(self):
        dm = DataManager()
        assert dm.loaded is False
        dm.load_dev_data()
        assert dm.loaded is True

    def test_get_entity_missing(self, app):
        dm = DataManager()
        dm.load_dev_data()
        assert dm.get_entity("NONEXISTENT") is None

    def test_get_explanation_missing(self, app):
        dm = DataManager()
        dm.load_dev_data()
        assert dm.get_explanation("NONEXISTENT") is None


# ── Filter Tests ──

class TestFilters:
    def _dm(self):
        dm = DataManager()
        dm._loaded = True
        dm._alerts = [
            AlertRecord(rank=1, entity_id="W001", risk_score=95, risk_level="Critical", main_reason="test critical"),
            AlertRecord(rank=2, entity_id="W017", risk_score=75, risk_level="High", main_reason="test high"),
            AlertRecord(rank=3, entity_id="W002", risk_score=50, risk_level="Medium", main_reason="test medium"),
        ]
        dm._transactions = [
            TransactionRecord(tx_id="TX001", timestamp="2024-01-01 12:00:00", source="W001", destination="W017", amount_btc=1.5, flag="Suspicious"),
            TransactionRecord(tx_id="TX002", timestamp="2024-01-01 12:05:00", source="W002", destination="W001", amount_btc=0.5, flag="Normal"),
        ]
        dm._graph = GraphData(
            nodes=[
                GraphNode(id="W001", type="Wallet", label="W001", risk_score=95),
                GraphNode(id="W017", type="Wallet", label="W017", risk_score=75),
                GraphNode(id="TX001", type="Transaction", label="TX001", risk_score=95),
            ],
            edges=[
                GraphEdge(source="W001", target="TX001", relation="sent"),
                GraphEdge(source="TX001", target="W017", relation="received"),
            ]
        )
        dm._entity_ids = ["W001", "W017", "W002"]
        return dm

    def test_filter_alerts_risk_level(self):
        dm = self._dm()
        alerts = dm.get_alerts()
        critical = filter_alerts(alerts, risk_level="Critical")
        assert all(a.risk_level == "Critical" for a in critical)
        assert len(critical) > 0

    def test_filter_alerts_entity_search(self):
        dm = self._dm()
        alerts = dm.get_alerts()
        ids = dm.get_entity_ids()
        filtered = filter_alerts(alerts, entity_search=ids[0])
        assert all(ids[0] in a.entity_id for a in filtered)

    def test_filter_alerts_score_range(self):
        dm = self._dm()
        alerts = dm.get_alerts()
        filtered = filter_alerts(alerts, min_score=90)
        assert all(a.risk_score >= 90 for a in filtered)

    def test_filter_transactions_entity(self):
        dm = self._dm()
        txns = dm.get_transactions()
        ids = dm.get_entity_ids()
        filtered = filter_transactions(txns, entity_id=ids[0])
        assert len(filtered) > 0
        assert all(ids[0].lower() in (t.source.lower() + t.destination.lower()) for t in filtered)

    def test_filter_transactions_flag(self):
        dm = self._dm()
        txns = dm.get_transactions()
        filtered = filter_transactions(txns, flag="high_risk")
        assert all(t.flag == "high_risk" for t in filtered)

    def test_filter_graph_nodes_type(self):
        dm = self._dm()
        graph = dm.get_graph()
        wallets = filter_graph_nodes(graph.nodes, types={"Wallet"})
        assert all(n.type == "Wallet" for n in wallets)
        assert len(wallets) > 0

    def test_filter_graph_nodes_entity(self):
        dm = self._dm()
        graph = dm.get_graph()
        ids = dm.get_entity_ids()
        if ids:
            filtered = filter_graph_nodes(graph.nodes, entity_id=ids[0])
            assert len(filtered) > 0


# ── AppState Tests ──

class TestAppState:
    def test_initial_state(self, app):
        st = AppState()
        assert st.selected_entity_id == ""
        assert st.selected_alert_entity_id == ""
        assert st.is_dataset_loaded is False

    def test_entity_signal(self, app):
        st = AppState()
        received = []
        st.entity_selected.connect(lambda eid: received.append(eid))
        st.selected_entity_id = "W001"
        assert received == ["W001"]

    def test_entity_same_no_signal(self, app):
        st = AppState()
        st.selected_entity_id = "W001"
        received = []
        st.entity_selected.connect(lambda eid: received.append(eid))
        st.selected_entity_id = "W001"
        assert received == []

    def test_alert_signal(self, app):
        st = AppState()
        received = []
        st.alert_selected.connect(lambda eid: received.append(eid))
        st.selected_alert_entity_id = "W001"
        assert received == ["W001"]

    def test_alert_sets_entity(self, app):
        st = AppState()
        entity_received = []
        alert_received = []
        st.entity_selected.connect(lambda eid: entity_received.append(eid))
        st.alert_selected.connect(lambda eid: alert_received.append(eid))
        st.selected_alert_entity_id = "W017"
        assert alert_received == ["W017"]
        assert entity_received == ["W017"]

    def test_filters_changed_signal(self, app):
        st = AppState()
        received = []
        st.filters_changed.connect(lambda: received.append(True))
        st.risk_level_filter = "High"
        assert len(received) == 1

    def test_risk_level_same_no_signal(self, app):
        st = AppState()
        received = []
        st.filters_changed.connect(lambda: received.append(True))
        st.risk_level_filter = "All"
        assert len(received) == 0

    def test_dataset_path_sets_loaded(self, app):
        st = AppState()
        st.dataset_path = "/some/path.csv"
        assert st.is_dataset_loaded is True


# ── Page Integration Tests ──

class TestPageIntegration:
    def _setup(self, app):
        from src.desktop_gui.main_window import MainWindow
        window = MainWindow()
        if "W001" not in window.data_manager.get_entity_ids():
            window.data_manager._entity_ids = ["W001", "W017"]
            window.pages["Entity Investigation"].entity_combo.addItems(["W001", "W017"])
            window.pages["Explainability"].entity_combo.addItems(["W001", "W017"])
        window.show()
        app.processEvents()
        return window

    def test_all_pages_render(self, app):
        window = self._setup(app)
        count = window.stacked.count()
        for i in range(count):
            window.stacked.setCurrentIndex(i)
            app.processEvents()
        assert count == 6

    def test_entity_propagation(self, app):
        window = self._setup(app)
        state = window.app_state
        state.selected_entity_id = "W001"
        app.processEvents()
        ei = window.pages["Entity Investigation"].entity_combo.currentText()
        xp = window.pages["Explainability"].entity_combo.currentText()
        assert ei == "W001"
        assert xp == "W001"

    def test_entity_change_propagation(self, app):
        window = self._setup(app)
        state = window.app_state
        state.selected_entity_id = "W001"
        app.processEvents()
        state.selected_entity_id = "W017"
        app.processEvents()
        ei = window.pages["Entity Investigation"].entity_combo.currentText()
        xp = window.pages["Explainability"].entity_combo.currentText()
        assert ei == "W017"
        assert xp == "W017"

    def test_sidebar_navigation(self, app):
        window = self._setup(app)
        for i in range(window.stacked.count()):
            window.sidebar.list_widget.setCurrentRow(i)
            app.processEvents()
            assert window.stacked.currentIndex() == i
