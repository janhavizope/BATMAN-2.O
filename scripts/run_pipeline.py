"""
run_pipeline.py
---------------
One command for the whole supervised pipeline: check the datasets, train,
score, and evaluate. The Streamlit app reads the score files it writes.

    1. Check that the Elliptic++ CSVs are in data/elliptic_plus/ (see README
       for how to download them)
    2. Derive the original Elliptic files from Elliptic++ if they are missing
    3. Train the Elliptic and Elliptic++ models (temporal split; metrics and
       charts go to outputs/)
    4. Score every transaction and wallet -> data/elliptic_plus/*.parquet

Run (from the project root):
    python scripts/run_pipeline.py                 # supervised pipeline
    python scripts/run_pipeline.py --unsupervised  # synthetic-data pipeline instead
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import ELLIPTIC_PLUS_DIR

# The original Elliptic files (data/elliptic/) are derived from Elliptic++ by
# prepare_elliptic.py when they are missing, so only Elliptic++ is required.
REQUIRED = {
    "Elliptic++ (https://github.com/git-disl/EllipticPlusPlus)": [
        ELLIPTIC_PLUS_DIR / "txs_features.csv",
        ELLIPTIC_PLUS_DIR / "txs_classes.csv",
        ELLIPTIC_PLUS_DIR / "txs_edgelist.csv",
        ELLIPTIC_PLUS_DIR / "wallets_features_classes_combined.csv",
        ELLIPTIC_PLUS_DIR / "AddrTx_edgelist.csv",
        ELLIPTIC_PLUS_DIR / "TxAddr_edgelist.csv",
    ],
}

SUPERVISED_STEPS = [
    ["scripts/prepare_elliptic.py"],
    ["scripts/train_elliptic.py"],
    ["scripts/train_elliptic_plus.py"],
    ["scripts/score_transactions.py"],
    ["scripts/score_wallets.py"],
    ["scripts/derive_wallet_risk_from_tx.py"],
]

UNSUPERVISED_STEPS = [
    ["scripts/generate_dev_dataset.py"],
    ["-m", "src.core.backend.features.feature_engineering"],
    ["-m", "src.core.backend.ml.isolation_forest_model"],
    ["-m", "src.core.backend.ml.hdbscan_clustering"],
    ["-m", "src.core.backend.risk.risk_engine"],
]


def missing_datasets() -> list:
    lines = []
    for source, files in REQUIRED.items():
        absent = [f for f in files if not f.exists()]
        if absent:
            lines.append(f"  {source}:")
            lines += [f"    {f.relative_to(ROOT)}" for f in absent]
    return lines


def run(steps) -> None:
    for step in steps:
        print(f"\n>>> python {' '.join(step)}", flush=True)
        result = subprocess.run([sys.executable, *step], cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"Step failed (exit {result.returncode}): {' '.join(step)}")


def main() -> None:
    if "--unsupervised" in sys.argv[1:]:
        run(UNSUPERVISED_STEPS)
        print("\nDone. Wrote data/dev/results.json: the unsupervised scores plus a "
              "precision/recall/ROC-AUC report against the synthetic labels. This is a "
              "research artifact — the web and desktop apps show the committed Elliptic++ "
              "scores, not this file. To score your own data with this pipeline, use the "
              "Upload & Score page in the web app.")
        return

    missing = missing_datasets()
    if missing:
        sys.exit("Missing dataset files. Download them and place them here:\n" + "\n".join(missing))
    run(SUPERVISED_STEPS)
    print("\nDone. Start the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
