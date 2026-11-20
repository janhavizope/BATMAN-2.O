"""
Data layer facade shared by the Streamlit pages and the desktop app.

Every value comes from the precomputed Elliptic++ risk scores read by
src.data.elliptic_source; pages and the desktop DataManager call these
functions and never read score files themselves.
"""

from src.data import elliptic_source
from src.data.elliptic_source import (  # noqa: F401  (re-exported)
    get_activity_timeline,
    get_alerts_df,
    get_entities,
    get_entity_ids,
    get_explainability_data,
    get_graph_data,
    get_model_drift,
    get_overview_stats,
    get_scoreboard,
    get_thresholds,
    get_transactions_df,
    get_tx_detail,
)


def is_data_available() -> bool:
    """True when the Elliptic++ score files are present."""
    return elliptic_source.is_available()


def get_transactions() -> list:
    """Transactions as a list of dicts (desktop app)."""
    return get_transactions_df().to_dict("records")


def get_alerts() -> list:
    """Alerts as a list of dicts (desktop app)."""
    return get_alerts_df().to_dict("records")
