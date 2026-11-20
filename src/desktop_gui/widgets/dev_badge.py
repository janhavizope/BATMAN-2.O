"""
Dev Badge Widget
----------------
Banner stating which data the desktop pages are showing.
"""

from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt

from datetime import datetime

from src.data import elliptic_source


def data_status() -> str:
    """One-line description of the loaded data, based on what is on disk."""
    if elliptic_source.is_available():
        scored_on = datetime.fromtimestamp(elliptic_source.WALLET_SCORES.stat().st_mtime).strftime("%Y-%m-%d")
        return f"[PRECOMPUTED SCORES] Elliptic++ Random Forest risk scores, written {scored_on}. Not a live feed."
    if elliptic_source.scores_status() == "lfs_pointer":
        return ("[NO SCORES] data/elliptic_plus/ holds Git LFS pointers, not the score files: "
                "run git lfs install && git lfs pull")
    return "[NO SCORES] No risk scores found in data/elliptic_plus/: run scripts/run_pipeline.py"


class DevBadge(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText(f"  {data_status()}  ")
        self.setStyleSheet(
            "QLabel {"
            "  background-color: #0a1a0a;"
            "  border: 1px solid #153a15;"
            "  border-left: 3px solid #1a8b1a;"
            "  border-radius: 3px;"
            "  padding: 8px 14px;"
            "  font-size: 11px;"
            "  font-family: Consolas;"
            "  color: #1a8b1a;"
            "}"
        )
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
