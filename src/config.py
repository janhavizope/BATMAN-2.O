"""
Project paths, in one place.

Every module and script resolves its input and output locations from here
instead of counting parent directories or editing sys.path.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
ELLIPTIC_DIR = DATA_DIR / "elliptic"
ELLIPTIC_PLUS_DIR = DATA_DIR / "elliptic_plus"
DEV_DATA_DIR = DATA_DIR / "dev"

MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"

FONTS_DIR = ROOT_DIR / "src" / "resources" / "fonts"

# Live monitoring (online pipeline) files
LIVE_DATA_DIR = DATA_DIR / "live"
LIVE_DB = LIVE_DATA_DIR / "live_monitor.sqlite"


def _find_cookie() -> str:
    """Locate bitcoind's .cookie file.

    BATMAN_RPC_COOKIE wins if set. Otherwise try the common data directories,
    including D:\\BitcoinData (the live-monitor node setup) and Bitcoin Core's
    per-OS default, and return the first that exists; if none do, return the
    default path so the error message names a real location.
    """
    import os
    import sys

    env = os.environ.get("BATMAN_RPC_COOKIE")
    if env:
        return env

    if sys.platform == "win32":
        default = Path.home() / "AppData" / "Local" / "Bitcoin" / ".cookie"
        candidates = [Path("D:/BitcoinData/.cookie"), default,
                      Path.home() / "AppData" / "Roaming" / "Bitcoin" / ".cookie"]
    else:
        default = Path.home() / ".bitcoin" / ".cookie"
        candidates = [default]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(default)


def get_node_settings() -> dict:
    """Bitcoin Core connection settings for the ZMQ feed, from the environment.

    Defaults are localhost and the conventional ZMQ port. Authentication uses
    bitcoind's .cookie file (located by _find_cookie) when no rpcuser/rpcpassword
    is set.
    """
    import os

    return {
        "rpc_url": os.environ.get("BATMAN_RPC_URL", "http://127.0.0.1:8332"),
        "rpc_user": os.environ.get("BATMAN_RPC_USER"),
        "rpc_password": os.environ.get("BATMAN_RPC_PASSWORD"),
        "rpc_cookie": _find_cookie(),
        "zmq_endpoint": os.environ.get("BATMAN_ZMQ_ENDPOINT", "tcp://127.0.0.1:28333"),
    }

# Unsupervised pipeline (synthetic data) files
DEV_DATASET = DEV_DATA_DIR / "dev_dataset_50k.csv"
DEV_FEATURES = DEV_DATA_DIR / "features.parquet"
DEV_ANOMALY_SCORES = DEV_DATA_DIR / "anomaly_scores.parquet"
DEV_CLUSTER_LABELS = DEV_DATA_DIR / "cluster_labels.parquet"
DEV_RESULTS = DEV_DATA_DIR / "results.json"
