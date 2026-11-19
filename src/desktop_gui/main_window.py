"""
Main Window
-----------
QMainWindow with sidebar, stacked pages, shared AppState, and DataManager.
All pages share the same AppState and DataManager instances.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter,
    QStackedWidget, QStatusBar, QFileDialog,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

from src.desktop_gui.widgets.dev_badge import data_status
from src.desktop_gui.sidebar import Sidebar
from src.desktop_gui.state.app_state import AppState
from src.desktop_gui.data.data_manager import DataManager
from src.desktop_gui.pages.overview_page import OverviewPage
from src.desktop_gui.pages.alerts_page import AlertsPage
from src.desktop_gui.pages.entity_investigation_page import EntityInvestigationPage
from src.desktop_gui.pages.transaction_analysis_page import TransactionAnalysisPage
from src.desktop_gui.pages.network_graph_page import NetworkGraphPage
from src.desktop_gui.pages.explainability_page import ExplainabilityPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BATMAN — Bitcoin Anomaly Traffic & Monitoring Analysis Network")
        self.resize(1400, 850)

        # --- Shared state + data --------------------------------------------
        self.app_state = AppState(self)
        self.data_manager = DataManager()
        self.data_manager.load_from_backend()

        # --- Menu bar --------------------------------------------------------
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open Dataset...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_dataset)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # --- Central widget --------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.sidebar = Sidebar()
        self.sidebar.setFixedWidth(220)
        splitter.addWidget(self.sidebar)

        # Pages — pass shared state and data manager to each (sidebar order)
        self.pages: dict[str, QWidget] = {
            "Overview":                 OverviewPage(self.app_state, self.data_manager),
            "Transaction Analysis":     TransactionAnalysisPage(self.app_state, self.data_manager),
            "Network Graph":            NetworkGraphPage(self.app_state, self.data_manager),
            "Entity Investigation":     EntityInvestigationPage(self.app_state, self.data_manager),
            "Alerts":                   AlertsPage(self.app_state, self.data_manager),
            "Explainability":           ExplainabilityPage(self.app_state, self.data_manager),
        }

        self.stacked = QStackedWidget()
        for page in self.pages.values():
            self.stacked.addWidget(page)
        splitter.addWidget(self.stacked)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        # --- Sidebar signal → page switch ------------------------------------
        self.sidebar.page_changed.connect(self.stacked.setCurrentIndex)

        # --- Status bar ------------------------------------------------------
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(data_status())

        # --- Wire cross-page interaction -------------------------------------
        self.app_state.entity_selected.connect(self._on_entity_selected)

    def _on_entity_selected(self, entity_id: str):
        """Update status bar when an entity is selected across pages."""
        self.status_bar.showMessage(f"Selected entity: {entity_id}")

    def _open_dataset(self):
        """Open a file dialog to load a dataset (placeholder — backend not yet connected)."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Dataset",
            "",
            "CSV Files (*.csv);;JSON Files (*.json);;All Files (*)",
        )
        if path:
            self.app_state.dataset_path = path
            self.status_bar.showMessage(f"Dataset selected: {path}  (backend not yet connected)")
