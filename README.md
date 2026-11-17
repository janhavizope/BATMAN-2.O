# BATMAN 2.0 — Bitcoin Anomaly Traffic & Monitoring Analysis Network

BATMAN scores Bitcoin wallets and transactions for illicit activity and explains each score, so an analyst can decide what to investigate first. It runs locally with no cloud services.

Built for **Smart India Hackathon 2026**, problem **SIH26146** (AI/Blockchain & Cyber Security).

---

## Status at a glance

| Component | State | Notes |
|---|---|---|
| **Streamlit web app** (`app.py`) | ✅ Works after cloning | Reads the precomputed Elliptic++ risk scores committed in `data/elliptic_plus/`. |
| **Supervised models** (`scripts/train_*.py`) | ✅ Reproducible | Random Forests on Elliptic and Elliptic++. You need to download the datasets to retrain. See [Evaluation caveats](#evaluation-caveats). |
| **Unsupervised pipeline** (`src/core/backend/`) | ⚠️ Experimental | Isolation Forest + HDBSCAN on a **synthetic** dataset. It isn't connected to the web app. |
| **Desktop app** (`main.py`, PySide6) | ⚠️ Needs pipeline output | Reads `data/dev/results.json` from the unsupervised pipeline. Every screen is empty until you run that pipeline. |

---

## Quick start (web app)

Requires Python 3.10 or newer.

```bash
git clone https://github.com/janhavizope/BATMAN-2.O.git
cd BATMAN-2.O

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Open http://127.0.0.1:8501. No dataset download is needed. The app uses the scores in `data/elliptic_plus/*.parquet`.

### Pages

| Page | What it shows |
|---|---|
| **Overview** | Wallet and transaction totals, flagged activity per Elliptic time step, severity breakdown, top alerts |
| **Transaction Analysis** | Held-out and unlabelled transactions with risk %, true label, and linked transactions |
| **Network Graph** | Transaction-flow subgraph around the highest-risk transactions, centrality table, wallet risk over time |
| **Entity Investigation** | Profile of a flagged wallet: activity timeline, key attributes, evidence |
| **Alerts** | Flagged wallets ranked by risk, with the top contributing feature |
| **Explainability** | Per-wallet feature contributions (Saabas decomposition of the Random Forest), compared with typical licit and illicit wallets |
| **Model Results** | Accuracy, precision, recall, ROC-AUC, and evaluation charts for each model |
| **Wallet Scanner** | Adjustable risk threshold, ranked list of risky wallets, lookup by address |

Only wallets and transactions the model **did not train on** (the test split and unlabelled data) are shown as predictions. Training-set scores are hidden because the model has already seen those labels.

---

## Data

| Dataset | Used for | Source |
|---|---|---|
| **Elliptic** | Transaction model, transaction-flow edges | [Kaggle: ellipticco/elliptic-data-set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) |
| **Elliptic++** | Transaction model and wallet model | [github.com/git-disl/EllipticPlusPlus](https://github.com/git-disl/EllipticPlusPlus) |

The raw CSVs are larger than GitHub's 100 MB limit and are not committed. To retrain, place them here:

```
data/elliptic/elliptic_txs_features.csv
data/elliptic/elliptic_txs_classes.csv
data/elliptic/elliptic_txs_edgelist.csv
data/elliptic_plus/txs_features.csv
data/elliptic_plus/txs_classes.csv
data/elliptic_plus/wallets_features_classes_combined.csv
```

Elliptic covers 49 time steps (about two weeks each) of real Bitcoin transactions. Each one is labelled illicit, licit, or unknown. Elliptic++ adds about 823k wallet addresses with behavioural features.

---

## Rebuilding the models and scores

Run these from the project root:

```bash
python scripts/train_elliptic.py              # Elliptic transaction model
python scripts/train_elliptic_plus.py         # Elliptic++ transaction and wallet models
python scripts/score_transactions.py          # -> data/elliptic_plus/tx_scores.parquet, tx_edges.parquet
python scripts/score_wallets.py               # -> wallet_scores / wallet_timeline / wallet_explanations.parquet
```

Outputs:

- Models go to `models/` (git-ignored).
- Metrics go to `outputs/evaluation/elliptic*_metrics.json`.
- Charts go to `outputs/graphs/elliptic*/`.

### Current results (random 80/20 hold-out)

| Model | Accuracy | Precision (illicit) | Recall (illicit) | ROC-AUC |
|---|---|---|---|---|
| Elliptic transactions | 98.97% | 98.8% | 90.5% | 0.997 |
| Elliptic++ transactions | 98.96% | 99.2% | 90.1% | 0.996 |
| Elliptic++ wallets (grouped by address) | 98.51% | 93.9% | 86.4% | 0.994 |

### Evaluation caveats

Read these numbers as an **optimistic upper bound**:

- **The split is random, but the data is a time series.** The standard Elliptic benchmark trains on time steps 1–34 and tests on 35–49. A random split lets the model learn from later time periods. On the committed scores, illicit-transaction recall falls from **92%** (steps 1–34) to **87%** (35–42) to **64%** (43–49). Steps 43 onward follow a dark-market shutdown, which is the known hard case for this dataset.
- **Time leaks into the features.** `time_step` and the block-number wallet features (`first_block_appeared_in`, `last_block_appeared_in`, …) are model inputs, so the model partly learns *when* a wallet was active rather than *how* it behaves.
- **A fixed 0.5 threshold flags about 23k unlabelled or test wallets.** That is too many to review by hand, so the UI shows the top 500.

Fixes for these are tracked in the [TODO](#todo) list.

---

## Unsupervised pipeline (experimental)

This is the original design: score wallets without labels, using Isolation Forest and HDBSCAN. It currently runs only on a synthetic dataset, and its output is read only by the desktop app.

```bash
python scripts/generate_dev_dataset.py                       # -> data/dev/dev_dataset_50k.csv (synthetic)
python src/core/backend/features/feature_engineering.py      # -> data/dev/features.parquet
python src/core/backend/ml/isolation_forest_model.py         # -> data/dev/anomaly_scores.parquet
python src/core/backend/ml/hdbscan_clustering.py             # -> data/dev/cluster_labels.parquet
python src/core/backend/risk/risk_engine.py                  # -> data/dev/results.json
python main.py                                               # desktop app reads results.json
```

On the synthetic data, the last report scored **precision 62% and recall 22.5%** on 141 wallets (`outputs/evaluation/model_evaluation_report.txt`).

The synthetic data includes IP, ASN, and country fields. Real Bitcoin transactions don't carry these, so those features have no counterpart in Elliptic or on-chain data.

---

## Project structure

```
BATMAN-2.O/
├── app.py                      # Streamlit entry point (main interface)
├── main.py                     # PySide6 desktop entry point
├── requirements.txt
├── .streamlit/config.toml      # Theme and server settings
├── data/elliptic_plus/         # Precomputed scores the web app reads (committed)
├── scripts/
│   ├── train_elliptic.py       # Supervised RF, Elliptic
│   ├── train_elliptic_plus.py  # Supervised RF, Elliptic++ (txs + wallets)
│   ├── score_transactions.py   # Score every transaction, export edges
│   ├── score_wallets.py        # Score every wallet, per-feature explanations
│   └── generate_dev_*.py       # Synthetic data generators (unsupervised pipeline)
├── src/
│   ├── data/                   # Data layer for the web app
│   │   ├── data_manager.py     #   Facade used by every page
│   │   └── elliptic_source.py  #   Reads the Elliptic++ score files
│   ├── web_frontend/           # Streamlit pages and shared components
│   ├── desktop_gui/            # PySide6 app (pages, widgets, charts, state)
│   ├── core/backend/           # Unsupervised pipeline: ingestion, validation,
│   │                           #   features, graph, ml, risk, explainability, alerts
│   └── resources/fonts/        # Bundled Orbitron and JetBrains Mono
├── outputs/
│   ├── evaluation/             # Metrics JSON and text reports
│   └── graphs/                 # Evaluation charts
├── tests/                      # pytest suite
└── docs/                       # Research and architecture references
```

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

The tests cover the unsupervised backend (ingestion, validation, features, graph, clustering, risk, alerts, evaluation) and the desktop GUI's data layer and state. The web app's data layer (`src/data/`) and the training and scoring scripts **don't have tests yet**.

---

## TODO

Priorities: **P0** = the results or claims are wrong until fixed · **P1** = a core feature is missing or broken · **P2** = cleanup and quality.

### P0 — Evaluation and correctness

- [ ] **Use a temporal train/test split.** Train on time steps 1–34, test on 35–49, and report illicit-class F1 per time step. Both scripts currently use a random 80/20 split: [train_elliptic.py:166](scripts/train_elliptic.py#L166), [train_elliptic_plus.py:70](scripts/train_elliptic_plus.py#L70).
- [ ] **Drop time-revealing features.** `time_step` is still a model input ([train_elliptic.py:67](scripts/train_elliptic.py#L67), [train_elliptic_plus.py:53](scripts/train_elliptic_plus.py#L53)). For wallets ([train_elliptic_plus.py:63](scripts/train_elliptic_plus.py#L63)), also drop the block-number features: `first_block_appeared_in`, `last_block_appeared_in`, `first_sent_block`, `first_received_block`.
- [ ] **Re-run training and scoring, then update the metrics** in this README and on the Model Results page.
- [ ] **Remove the label-derived weights** in [risk_engine.py:24](src/core/backend/risk/risk_engine.py#L24). They are correlations with the ground-truth labels, which makes the "unsupervised" score partly supervised and evaluated on its own labels.
- [ ] **Make the alert reasons agree with the score.** The weights treat *low* degree, fees, and counterparties as risky, but `build_reasons` reports "unusually *high*" values. Four computed components (graph, temporal, transaction, network) are unused in `risk_score` ([risk_engine.py:131](src/core/backend/risk/risk_engine.py#L131)).
- [ ] **Fix sent-vs-received in feature engineering.** Rows where the wallet is an *input* (sending) are counted as `incoming_transaction_count` ([feature_engineering.py:140](src/core/backend/features/feature_engineering.py#L140)), which also inverts `min_pass_through_gap`. Co-input addresses are counted as counterparties ([lines 150 and 156](src/core/backend/features/feature_engineering.py#L150)), but they usually belong to the same owner.
- [ ] **Desktop transaction flag shows the label, not a prediction.** [desktop data_manager.py:158](src/desktop_gui/data/data_manager.py#L158) reads the ground-truth `is_suspicious` column. It also compares against `"true"`, while `scripts/generate_dev_dataset.py` writes `1`/`0`, so every row shows "Normal".

### P1 — Product

- [ ] **Choose the alert threshold from the precision-recall curve.** The fixed 0.5 ([elliptic_source.py:25](src/data/elliptic_source.py#L25)) flags about 23k wallets. Target a reviewable alert volume instead, and expose the threshold in one place.
- [ ] **Load the Elliptic++ wallet↔transaction edge lists** (`AddrTx`, `TxAddr`). That lets Entity Investigation show a wallet's transactions, and lets the Network Graph page show the graph around the listed entities instead of an unrelated transaction subgraph.
- [ ] **Add a way to ingest your own data.** Upload a CSV, validate it with `validation/validator.py`, score it, and show the results. Remove the "Ingest external transaction records" text ([components.py:74](src/web_frontend/components.py#L74)) until this exists.
- [ ] **Use one data layer for both apps.** The desktop app should call `src/data/` instead of its own `desktop_gui/data/data_manager.py`, or the desktop app should be retired.
- [ ] **Run the unsupervised pipeline on Elliptic++** instead of synthetic data.
- [ ] **Remove IP/ASN/country hub nodes** from the graph ([graph_builder.py:91](src/core/backend/graph/graph_builder.py#L91)). Real Bitcoin transactions don't carry these fields, and hub nodes distort PageRank and betweenness.
- [ ] **Replace hardcoded status badges** ("SYSTEM ONLINE", "[LIVE MODE]") with real status, or remove them.

### P2 — Cleanup and engineering

- [ ] **Delete the dead alert → risk-score → explain → evaluate chain.** It reads `clustered_entities.parquet`, which nothing writes, from `src/core/data/dev/` ([generate_alerts.py:42](src/core/backend/alerts/generate_alerts.py#L42)).
- [ ] **Remove duplicate modules:** `feature_engineering_OLD.py`, `ml/clustering_model.py` (duplicate HDBSCAN), `bitcoin-monitor/`, extra dataset generators, and `tune_unsupervised_models.py` (imports `backend.*` paths that no longer exist).
- [ ] **Centralise paths** in one config module, replacing `parents[4]`, `../..` and `sys.path.append` in each module.
- [ ] **Add a single pipeline command** for download → train → score → evaluate.
- [ ] **Pin dependency versions** in `requirements.txt`.
- [ ] **Add tests** for `src/data/` and the training and scoring scripts, using small fixtures. Current GUI tests pass on empty data, and `tests/test_alerts.py` re-implements the logic it is meant to test.
- [ ] **Add CI** (GitHub Actions) to run the tests on every push.
- [ ] **Load desktop fonts only from `src/resources/fonts`.** [theme_setup.py:24](src/desktop_gui/utils/theme_setup.py#L24) downloads missing fonts from GitHub, which breaks offline use.
- [ ] **Replace `iterrows()` loops** in feature engineering and the graph builder with vectorised pandas.
- [ ] **Escape dataset values** before inserting them into `unsafe_allow_html` markup.
- [ ] **Move large parquet files to Git LFS** or a release asset.
- [ ] **Update or remove `PROJECT_STRUCTURE.md`.** It lists a `config/` folder that doesn't exist.

---

## Prebuilt desktop build (Linux)

A Linux AppImage of the desktop interface is available on [Google Drive](https://drive.google.com/file/d/1XKrpaMlFW2FChI7zCK9j118-lNAXL5-P/view?usp=sharing).

```bash
chmod +x BATMAN-x86_64.AppImage
./BATMAN-x86_64.AppImage
```

---

## References

- Weber et al. (2019), *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics* (Elliptic dataset)
- Elmougy & Liu (2023), *Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics* (Elliptic++)
- Meiklejohn et al. (2013), *A Fistful of Bitcoins: Characterizing Payments Among Men with No Names*
- Liu, Ting & Zhou (2008), *Isolation Forest*; Campello, Moulavi & Sander (2013), *HDBSCAN*

More background: [`docs/RESEARCH_ARCHITECTURE_REFERENCES.md`](docs/RESEARCH_ARCHITECTURE_REFERENCES.md)

---

## Team

**Team Lead:** Janhavi Zope

**Members:** Ankita Khokhar · Nutanapushpa Vasarla · Neha More · Lakshmi Sai Lekhana Bavikatta · Keziah Ann Mathew

Symbiosis Skills And Professional University — Smart India Hackathon 2026, Problem ID SIH26146
