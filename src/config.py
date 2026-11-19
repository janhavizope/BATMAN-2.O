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

# Unsupervised pipeline (synthetic data) files
DEV_DATASET = DEV_DATA_DIR / "dev_dataset_50k.csv"
DEV_FEATURES = DEV_DATA_DIR / "features.parquet"
DEV_ANOMALY_SCORES = DEV_DATA_DIR / "anomaly_scores.parquet"
DEV_CLUSTER_LABELS = DEV_DATA_DIR / "cluster_labels.parquet"
DEV_RESULTS = DEV_DATA_DIR / "results.json"
