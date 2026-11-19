"""
Overview Page
-------------
KPI grid + professional charts — cybersecurity dashboard aesthetic.
No summary table. No health boxes. Graph-heavy layout.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGridLayout, QHBoxLayout,
)
from PySide6.QtGui import QFont

from src.desktop_gui.widgets.metric_card import MetricCard
from src.desktop_gui.widgets.dev_badge import DevBadge
from src.desktop_gui.data.data_manager import DataManager
from src.desktop_gui.state.app_state import AppState
from src.desktop_gui.charts.overview_charts import SeverityDistributionDonut, ActivityTimelineChart

class OverviewPage(QWidget):
    def __init__(self, app_state: AppState, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self._dm = data_manager
        self._state = app_state

        self.setStyleSheet("background-color: #0a0a0f;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # Header
        header_row = QHBoxLayout()
        title = QLabel("DASHBOARD OVERVIEW")
        title.setFont(QFont("Consolas", 22, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0; letter-spacing: 4px;")
        header_row.addWidget(title)
        header_row.addStretch()
        layout.addLayout(header_row)

        layout.addWidget(DevBadge())

        # KPI cards
        self._kpi_grid = QGridLayout()
        self._kpi_grid.setSpacing(10)
        layout.addLayout(self._kpi_grid)

        # Charts row
        charts_row = QHBoxLayout()
        charts_row.setSpacing(16)

        right_chart_container = QVBoxLayout()
        right_label = QLabel("Flagged transactions per time step")
        right_label.setStyleSheet("color: #c0c0c0; font-family: 'Orbitron', sans-serif; font-size: 13px; font-weight: bold;")
        right_chart_container.addWidget(right_label)
        self.activity_chart = ActivityTimelineChart()
        right_chart_container.addWidget(self.activity_chart)
        charts_row.addLayout(right_chart_container, stretch=2)

        left_chart_container = QVBoxLayout()
        left_label = QLabel("Severity Distribution")
        left_label.setStyleSheet("color: #c0c0c0; font-family: 'Orbitron', sans-serif; font-size: 13px; font-weight: bold;")
        left_chart_container.addWidget(left_label)
        self.risk_chart = SeverityDistributionDonut()
        left_chart_container.addWidget(self.risk_chart)
        charts_row.addLayout(left_chart_container, stretch=1)

        layout.addLayout(charts_row, stretch=1)

        # Recent alerts summary
        alerts_label = QLabel("RECENT ALERTS")
        alerts_label.setStyleSheet("color: #6b1a1a; font-family: Consolas; font-size: 11px; font-weight: bold;")
        layout.addWidget(alerts_label)

        self.alerts_summary = QLabel()
        self.alerts_summary.setWordWrap(True)
        self.alerts_summary.setStyleSheet(
            "color: #888; font-family: Consolas; font-size: 11px; "
            "background-color: #111118; border: 1px solid #2a0a0a; "
            "border-radius: 4px; padding: 10px;"
        )
        layout.addWidget(self.alerts_summary)

        layout.addStretch()

        self.refresh()

    def refresh(self):
        stats = self._dm.get_overview()
        d = stats.to_dict()

        # KPI cards
        self._clear_layout(self._kpi_grid)
        card_specs = [
            ("TOTAL TRANSACTIONS", f"{d['total_transactions']:,}"),
            ("TOTAL WALLETS",      f"{d['total_wallets']:,}"),
            ("KNOWN ILLICIT",      f"{d['known_illicit']:,}"),
            ("ANOMALOUS",          f"{d['anomalous_entities']:,}"),
            ("HIGH-RISK",          f"{d['high_risk_entities']:,}"),
            ("CRITICAL ALERTS",    f"{d['critical_alerts']:,}"),
        ]
        for idx, (label, value) in enumerate(card_specs):
            self._kpi_grid.addWidget(MetricCard(label, value), 0, idx)

        # Risk distribution chart
        alerts = self._dm.get_alerts()
        risk_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for a in alerts:
            if a.risk_level in risk_counts:
                risk_counts[a.risk_level] += 1
        self.risk_chart.render_chart(risk_counts)

        # Activity timeline chart — flagged transactions per Elliptic time step
        labels, counts = self._dm.get_activity_timeline()
        self.activity_chart.render_chart(labels, counts)

        # Alerts summary
        top_alerts = alerts[:3]
        lines = []
        for a in top_alerts:
            lines.append(f"[{a.risk_level}] {a.entity_id}  —  {a.main_reason}")
        self.alerts_summary.setText("\n".join(lines) if lines else "No alerts.")

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
