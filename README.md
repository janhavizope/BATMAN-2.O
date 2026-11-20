# BATMAN 2.0 — Bitcoin Anomaly Traffic & Monitoring Analysis Network

BATMAN scores Bitcoin wallets and transactions for illicit activity and explains each score, so an analyst can decide what to investigate first. It runs locally with no cloud services.

Built for **Smart India Hackathon 2026**, problem **SIH26146** (AI/Blockchain & Cyber Security).

---

## Status at a glance

| Component | State | Notes |
|---|---|---|
| **Streamlit web app** (`app.py`) | ✅ Works after cloning | Reads the precomputed Elliptic++ risk scores committed in `data/elliptic_plus/` (Git LFS). |
| **Supervised models** (`scripts/train_*.py`) | ✅ Reproducible | Random Forests on Elliptic and Elliptic++, trained with a temporal split and no time-revealing features. See [Results](#results-temporal-split). |
| **Unsupervised pipeline** (`src/core/backend/`) | ⚠️ Experimental | Isolation Forest + HDBSCAN with no label-derived weights. Runs on synthetic data, on Elliptic++, and on files uploaded in the web app. |
| **Desktop app** (`main.py`, PySide6) | ✅ Works after cloning | Uses the same data layer (`src/data/`) as the web app. |

---

## Quick start (web app)

Requires Python 3.10 or newer (dependencies are pinned and tested on 3.13) and [Git LFS](https://git-lfs.com) for the score files.

```bash
git lfs install
git clone https://github.com/janhavizope/BATMAN-2.O.git
cd BATMAN-2.O

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py               # web app
python main.py                     # desktop app (same data)
```

Open http://127.0.0.1:8501. No dataset download is needed. The app uses the scores in `data/elliptic_plus/*.parquet`. If you cloned before installing Git LFS, the score files are small pointer files. Both apps detect this and tell you to run `git lfs install && git lfs pull`, which downloads the real files.

### Pages

| Page | What it shows |
|---|---|
| **Overview** | Wallet and transaction totals, flagged transactions per Elliptic time step, severity breakdown, top alerts |
| **Transaction Analysis** | Held-out and unlabelled transactions with risk %, true label, and linked transactions |
| **Network Graph** | The top 10 alerted wallets, the transactions they sent or received, and the transaction flows between those; centrality table; wallet risk over time |
| **Entity Investigation** | Profile of a flagged wallet: activity timeline, its transactions, key attributes, evidence |
| **Alerts** | Flagged wallets ranked by risk, with the top contributing feature |
| **Explainability** | Per-wallet feature contributions (Saabas decomposition of the Random Forest), compared with typical licit and illicit wallets |
| **Wallet Scanner** | Adjustable risk threshold (defaults to the alert threshold), ranked list of risky wallets, lookup by address |
| **Upload & Score** | Upload your own transactions (CSV or JSON), see the validation report, and get every wallet ranked by the unsupervised risk score with its reasons |

Only wallets and transactions the model **did not train on** (the test split and unlabelled data) are shown as predictions. Training-set scores are hidden because the model has already seen those labels.

### Alert threshold

The alert threshold is not a fixed 0.5. `choose_threshold` in [`src/data/elliptic_source.py`](src/data/elliptic_source.py) starts from the threshold with the best illicit F1 on the test split's precision-recall curve. It then raises the threshold until the number of flagged wallets fits a review budget (`WALLET_ALERT_BUDGET = 500`, `TX_ALERT_BUDGET = 1000`). Those budgets are the one place to change alert volume.

On the current scores the wallet threshold is 0.997. It flags 570 wallets (ties at the threshold push it past 500), down from 26,006 at 0.5. On test wallets that threshold gives precision 64% and recall 0.2%. Wallet-level prediction under a temporal split is weak (see [Results](#results-temporal-split)), so treat wallet alerts as leads, not findings. The transaction threshold is also 0.997. It flags 1,187 transactions with test precision 100% and recall 16%.

**Severity (Critical / High / Medium) is a wallet's position in the risk-ranked alert list:** Critical is the top 10%, High the next 30%, and Medium the rest (`CRITICAL_SHARE` and `HIGH_SHARE` in `elliptic_source.py`). Alerts are ranked by `(risk, total_txs)`, so wallets tied on model risk are ordered by transaction count — a weak secondary signal, but enough to keep the bands from collapsing. On the current scores the 570 alerts have **31 distinct risk values**, but they are tightly bunched against 1.0: 229 wallets sit at exactly 1.0 and the other 341 span the range 0.9967–0.99998. Banding by position gives **57 Critical, 171 High, 342 Medium**. The bands order the alerts for triage; they do not claim 57 wallets are meaningfully riskier than the next 171, because probabilities this close to 1.0 are not calibrated. Making severity carry real information needs a better-calibrated score (see [TODO](#open)), which the positional banding is a stopgap for — an earlier version banded by thirds of the probability range and a later one by strict rank, where a single 229-wide tie at 1.0 took all of Critical+High and left High empty.

### Upload & Score

Required columns: `txid`, `timestamp`, `input_addresses`, `output_addresses`, `input_amounts`, `output_amounts`. List fields use `|` or JSON arrays. `fee` and the network fields (`src_ip`, `dst_ip`, `src_port`, `dst_port`, `asn`, `geo_country`) are optional, because real on-chain data has no network fields. The page offers a template CSV. Uploads are validated with [`validator.py`](src/core/backend/validation/validator.py) and scored in memory with [`pipeline.py`](src/core/backend/pipeline.py). Scores are relative to the uploaded file, and nothing leaves the machine.

The page shows a warning at the top. On real labelled Elliptic++ wallets this unsupervised score reached ROC-AUC 0.48, close to random guessing (see [Unsupervised pipeline](#unsupervised-pipeline-experimental)). Its HIGH and CRITICAL levels mean "unusual compared with the rest of the file", not "illicit".

---

## Data

| Dataset | Used for | Source |
|---|---|---|
| **Elliptic++** | Transaction model, wallet model, wallet↔transaction links, transaction-flow edges | [github.com/git-disl/EllipticPlusPlus](https://github.com/git-disl/EllipticPlusPlus) |
| **Elliptic** | Transaction model | Derived from Elliptic++ by `scripts/prepare_elliptic.py`; the original is on [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) |

The raw CSVs are larger than GitHub's 100 MB limit and are not committed. To retrain, place these files from the Elliptic++ repository (they are Git LFS files there, or use the Google Drive folder linked from its README) in `data/elliptic_plus/`:

```
data/elliptic_plus/txs_features.csv                     (Transactions Dataset/)
data/elliptic_plus/txs_classes.csv                      (Transactions Dataset/)
data/elliptic_plus/txs_edgelist.csv                     (Transactions Dataset/)
data/elliptic_plus/wallets_features_classes_combined.csv (Actors Dataset/)
data/elliptic_plus/AddrTx_edgelist.csv                  (Actors Dataset/)
data/elliptic_plus/TxAddr_edgelist.csv                  (Actors Dataset/)
```

Elliptic++'s `txs_features.csv` contains Elliptic's columns (txId, time step, 93 local and 72 aggregate features) followed by 17 extra ones, and `txs_edgelist.csv` is Elliptic's edge list (234,355 edges). So `prepare_elliptic.py` writes `data/elliptic/` in Elliptic's layout when it is missing, and you don't need a Kaggle account. If you have the Kaggle files, put them in `data/elliptic/` and they are used as they are.

Elliptic covers 49 time steps (about two weeks each) of real Bitcoin transactions. Each one is labelled illicit, licit, or unknown. Elliptic++ adds about 823k wallet addresses with behavioural features, and the links between wallets and transactions.

---

## Rebuilding the models and scores

Run this from the project root after placing the datasets:

```bash
python scripts/run_pipeline.py
```

It checks that every dataset file is present (and lists any that are missing), then runs these steps in order. You can also run them one by one:

```bash
python scripts/prepare_elliptic.py            # data/elliptic/ from Elliptic++ (skipped if present)
python scripts/train_elliptic.py              # Elliptic transaction model
python scripts/train_elliptic_plus.py         # Elliptic++ transaction and wallet models
python scripts/score_transactions.py          # -> data/elliptic_plus/tx_scores.parquet, tx_edges.parquet
python scripts/score_wallets.py               # -> wallet_scores / wallet_timeline / wallet_explanations /
                                              #    wallet_tx_edges.parquet
```

How the models are trained and evaluated:

- **Temporal split.** Train on time steps 1–34 and test on 35–49, which is the standard Elliptic benchmark. Each wallet is assigned by the time step it first appears in. A wallet first seen in steps 1–34 trains, using only its rows up to step 34. A wallet first seen in step 35 or later is tested. No wallet is in both sets.
- **No time-revealing features.** `time_step` / `Time step`, and the wallet block-number features (`first_block_appeared_in`, `last_block_appeared_in`, `first_sent_block`, `first_received_block`), are used only for the split, never as model inputs.
- **Metrics per time step.** Each metrics file includes `f1_illicit_by_time_step` for the test period.

Outputs:

- Models go to `models/` (git-ignored).
- Metrics go to `outputs/evaluation/elliptic*_metrics.json`.
- Charts go to `outputs/graphs/elliptic*/`.

### Results (temporal split)

Test set: time steps 35–49. Illicit is the positive class.

| Model | Train / test rows | Accuracy | Precision (illicit) | Recall (illicit) | F1 (illicit) | ROC-AUC |
|---|---|---|---|---|---|---|
| Elliptic transactions | 29,894 / 16,670 | 97.76% | 91.7% | 72.0% | 0.807 | 0.939 |
| Elliptic++ transactions | 29,894 / 16,670 | 97.97% | 95.4% | 72.3% | 0.822 | 0.932 |
| Elliptic++ wallets | 245,667 / 114,345 | 92.47% | 48.7% | 20.2% | 0.286 | 0.862 |

Mean illicit F1 per time step:

| Model | Steps 35–42 | Steps 43–49 |
|---|---|---|
| Elliptic transactions | 0.88 | 0.04 |
| Elliptic++ transactions | 0.90 | 0.07 |
| Elliptic++ wallets | 0.26 | 0.14 |

### Evaluation notes

- **The dark-market shutdown breaks the transaction models.** F1 is high up to step 42 and falls to almost zero from step 43, after the dark-market shutdown. This matches the original Elliptic paper. The model does not detect the new illicit patterns that follow.
- **Wallet prediction is much harder than the old numbers suggested.** With the earlier random split and block-number features, the wallet model reported F1 0.90 (precision 93.9%, recall 86.4%). Under a temporal split, without those features and without testing on wallets it trained on, F1 is 0.29. Most of the old score came from leakage: the model recognised when a wallet was active, or recognised wallets it had already seen.
- **Choosing a threshold on the test set is slightly optimistic.** The alert threshold's precision and recall are measured on the same test rows used to pick it.

---

## Unsupervised pipeline (experimental)

This is the original design: score wallets without labels, using Isolation Forest and HDBSCAN. It runs on synthetic data, on Elliptic++, and on files uploaded in the web app (the **Upload & Score** page).

```bash
python scripts/run_pipeline.py --unsupervised      # synthetic data, step by step (below)
python scripts/run_unsupervised_elliptic_plus.py   # real Elliptic++ wallet-transaction graph
```

```bash
python scripts/generate_dev_dataset.py                       # -> data/dev/dev_dataset_50k.csv (synthetic)
python -m src.core.backend.features.feature_engineering      # -> data/dev/features.parquet
python -m src.core.backend.ml.isolation_forest_model         # -> data/dev/anomaly_scores.parquet
python -m src.core.backend.ml.hdbscan_clustering             # -> data/dev/cluster_labels.parquet
python -m src.core.backend.risk.risk_engine                  # -> data/dev/results.json
```

How the score works:

- **Label-free.** The risk score combines anomaly, HDBSCAN-outlier, graph, temporal, transaction and network components, with fixed weights (`COMPONENT_WEIGHTS` in [`risk_engine.py`](src/core/backend/risk/risk_engine.py)). No weight comes from the ground-truth labels. The labels are used only to report precision and recall after scoring.
- **Missing signals drop out.** A component whose signals are missing or constant (for example, no IP fields in real data) drops out, and the remaining weights are renormalised.
- **Reasons match the score.** One `SIGNALS` table defines both the score components and the alert reasons. A reason is reported only for a signal that raised the score, in the same direction.

**Synthetic data** (298,686 wallets, 7.2% suspicious):

- **ROC-AUC:** 0.864.
- **Precision of the top 100 / 500 / 1,000 wallets:** 100% / 99.6% / 80%.
- **Risk bands:** no wallet scored above 60, so nothing reaches HIGH or CRITICAL. The fixed 30 / 60 / 80 bands don't fit this score.
- **Comparison with the old report:** precision 62% on 141 flagged wallets, but those scores used weights fitted to the labels.

In this dataset almost every address appears in a single transaction, so many wallet features hardly vary.

**Elliptic++** (`run_unsupervised_elliptic_plus.py`, 822,942 wallets scored, 265,354 with a label, 5.4% illicit). The label-free score is checked against the wallet labels it never saw:

| Ranking | ROC-AUC | Precision@100 | Precision@500 | Precision@1000 |
|---|---|---|---|---|
| Unsupervised score (all labelled wallets) | 0.556 | 2% | 11% | 12% |
| Unsupervised score (held-out test wallets) | 0.482 | 13% | 10% | 8% |
| Supervised Random Forest (same test wallets) | 0.839 | 78% | 71% | 52% |

The honest result: **on real Bitcoin behavioural features the hand-weighted unsupervised score is close to random** (ROC-AUC 0.48–0.56), while the supervised model reaches 0.84 on the same wallets. The fixed component weights that work on the synthetic data (which was built with injected fan-in / fan-out / burst patterns) do not capture what separates illicit wallets in Elliptic++. The unsupervised pipeline is useful as a label-free triage tool and for the Upload & Score page, but it is not a substitute for the supervised model where labels exist.

---

## Project structure

```
BATMAN-2.O/
├── app.py                      # Streamlit entry point (main interface)
├── main.py                     # PySide6 desktop entry point
├── requirements.txt            # Pinned dependencies
├── pytest.ini
├── .gitattributes              # data/elliptic_plus/*.parquet are Git LFS files
├── .github/workflows/tests.yml # CI: runs pytest on every push and pull request
├── .streamlit/config.toml      # Theme and server settings
├── data/elliptic_plus/         # Precomputed scores both apps read (committed, Git LFS)
├── scripts/
│   ├── run_pipeline.py         # One command: check data -> prepare -> train -> score
│   ├── prepare_elliptic.py     # Elliptic files derived from Elliptic++
│   ├── train_elliptic.py       # Supervised RF, Elliptic (temporal split)
│   ├── train_elliptic_plus.py  # Supervised RF, Elliptic++ (txs + wallets)
│   ├── score_transactions.py   # Score every transaction, export edges
│   ├── score_wallets.py        # Score every wallet, explanations, wallet-transaction links
│   ├── run_unsupervised_elliptic_plus.py  # Unsupervised pipeline on Elliptic++
│   └── generate_dev_dataset.py # Synthetic data generator (unsupervised pipeline)
├── src/
│   ├── config.py               # Every input and output path
│   ├── data/                   # Data layer shared by the web and desktop apps
│   │   ├── data_manager.py     #   Facade used by every page and the desktop DataManager
│   │   └── elliptic_source.py  #   Reads the Elliptic++ score files, picks alert thresholds
│   ├── web_frontend/           # Streamlit pages and shared components
│   ├── desktop_gui/            # PySide6 app (pages, widgets, charts, state)
│   ├── core/backend/           # Unsupervised pipeline: ingestion, validation,
│   │                           #   features, graph, ml, risk, pipeline.py (in-memory run)
│   └── resources/fonts/        # Bundled Orbitron and JetBrains Mono
├── outputs/
│   ├── evaluation/             # Metrics JSON
│   └── graphs/                 # Evaluation charts
├── tests/                      # pytest suite
└── docs/                       # Research and architecture references
```

---

## Tests

```bash
pytest
```

The suite covers:

- **Unsupervised backend:** ingestion and validation (including data without network fields), features (including send/receive direction and co-input counterparties), graph, clustering, the risk engine, and the in-memory pipeline used by Upload & Score.
- **Shared data layer** (`tests/test_data_source.py`): threshold choice, position-based severity (that every band is filled despite a large tie at the top), Git LFS pointer detection, hidden training rows, alerts, entities, wallet-transaction links, the network graph, and the desktop DataManager on top of it.
- **Training and scoring scripts** (`tests/test_training_scripts.py`): the temporal and wallet splits, dropped time features, and a full train → score run on small synthetic Elliptic++ fixtures.
- **Desktop GUI:** pages and state.

CI runs the same suite on GitHub Actions. It caches the Git LFS score files between runs, so each run doesn't re-download about 55 MB from the repository's LFS quota.

---

## TODO

Priorities: **P0** = the results or claims are wrong until fixed · **P1** = a core feature is missing or broken · **P2** = cleanup and quality.

### P0 — Evaluation and correctness

- [x] **Use a temporal train/test split.** Train on time steps 1–34, test on 35–49, and report illicit-class F1 per time step.
- [x] **Drop time-revealing features.** `time_step` and the wallet block-number features are no longer model inputs.
- [x] **Re-run training and scoring, then update the metrics.** See [Results](#results-temporal-split). The committed scores come from the retrained models.
- [x] **Remove the label-derived weights** from `risk_engine.py`.
- [x] **Make the alert reasons agree with the score.** One `SIGNALS` table drives both, and all components count toward `risk_score`.
- [x] **Fix sent-vs-received in feature engineering.** Inputs are sends and outputs are receives (`min_pass_through_gap` is fixed too). Co-input addresses are no longer counted as counterparties.
- [x] **Desktop transaction flag shows a prediction.** The desktop app now shows the model's flags from the shared data layer and never reads `is_suspicious`.

### P1 — Product

- [x] **Choose the alert threshold from the precision-recall curve.** See [Alert threshold](#alert-threshold).
- [x] **Load the Elliptic++ wallet↔transaction edge lists** (`AddrTx`, `TxAddr`). Entity Investigation lists each wallet's transactions. The Network Graph shows the alerted wallets and their transactions.
- [x] **Add a way to ingest your own data.** The **Upload & Score** page validates and scores an uploaded CSV or JSON.
- [x] **Use one data layer for both apps.** The desktop `DataManager` is now an adapter over `src/data/`.
- [x] **Run the unsupervised pipeline on Elliptic++** with `scripts/run_unsupervised_elliptic_plus.py`.
- [x] **Remove IP/ASN/country hub nodes** from the graph.
- [x] **Replace hardcoded status badges.** Both apps report which data is loaded and when it was scored.

### P2 — Cleanup and engineering

- [x] **Delete the dead alert → risk-score → explain → evaluate chain** and its tests.
- [x] **Remove duplicate modules:** `feature_engineering_OLD.py`, `ml/clustering_model.py`, `bitcoin-monitor/`, `scripts/generate_dev_data.py`, `ml/generate_synthetic_data.py` + `ml/train_model.py` (and its model file), and `tune_unsupervised_models.py`.
- [x] **Centralise paths** in [`src/config.py`](src/config.py). Backend modules run with `python -m`, and the tests rely on `pytest.ini`.
- [x] **Add a single pipeline command:** `python scripts/run_pipeline.py`.
- [x] **Pin dependency versions** in `requirements.txt` (and add the missing `scipy`).
- [x] **Add tests** for `src/data/` and the training and scoring scripts, using small fixtures.
- [x] **Add CI** (GitHub Actions) to run the tests on every push.
- [x] **Load desktop fonts only from `src/resources/fonts`.** Nothing is downloaded any more.
- [x] **Replace `iterrows()` loops** in feature engineering and the graph builder with vectorised pandas.
- [x] **Escape dataset values** before inserting them into `unsafe_allow_html` markup.
- [x] **Move large parquet files to Git LFS.** `data/elliptic_plus/*.parquet` are tracked by Git LFS from now on. Earlier commits still hold the old copies; removing those would need a history rewrite (`git lfs migrate`) and a force-push.
- [x] **Update or remove `PROJECT_STRUCTURE.md`.** It now points to this README.

### Review follow-ups (November 2026)

These come from a review of the changes above.

#### Done

- [x] **Position-based severity.** Critical, High and Medium are a wallet's position in the risk-ranked alert list (top 10% / next 30% / rest), with ties ordered by transaction count. An earlier rank rule let the 229-wide tie at risk 1.0 fill Critical+High and leave High empty; positional banding keeps the 10/30/60 split (57 / 171 / 342). See [Alert threshold](#alert-threshold).
- [x] **Detect Git LFS pointer files.** If the repository is cloned without Git LFS, both apps say to run `git lfs pull` instead of failing when they read the scores (`scores_status()` in `elliptic_source.py`).
- [x] **Warn on the Upload & Score page** that the unsupervised score is close to random on real labelled wallets.
- [x] **Cache Git LFS files in CI** so each run doesn't use up the LFS bandwidth quota.

#### Open

**P0 — evaluation**

- [ ] **Check the remaining wallet features for future information.** `lifetime_in_blocks` is still the 10th most important wallet feature. If it, `num_timesteps_appeared_in`, `total_txs` or `btc_transacted_total` are computed over a wallet's whole history, a training row from step 10 reveals activity after step 34. Check whether each one stays constant for an address across its time steps. If it does, drop it or recompute it up to each time step, then retrain.
- [ ] **Choose the alert threshold on a validation period, not the test set.** For example, train on steps 1–29, pick the threshold on steps 30–34, then retrain on 1–34. Report test precision and recall at that threshold without tuning on the test set.

**P1 — models and product**

- [ ] **Derive wallet risk from transaction risk.** The transaction model is much stronger than the wallet model (F1 0.82 vs 0.29). Score each wallet from the transactions it sent or received (for example, the highest transaction risk via `AddrTx` / `TxAddr`), and compare that ranking with the wallet model on test wallets. `score_wallets.py` currently writes these links only for flagged wallets, so the export must cover every wallet.
- [ ] **Break ties among alerts.** The 570 wallet alerts have 31 distinct risk values, but 229 tie at exactly 1.0 and the rest are packed into 0.9967–0.99998 — RF probabilities this close to 1.0 are not calibrated, so the ordering is nearly arbitrary. Severity currently breaks ties by transaction count. Calibrate the probabilities on the validation period (`CalibratedClassifierCV`), or add a principled secondary ranking signal, so alerts can be ordered and severity carries real information.
- [ ] **Revisit "keep each wallet's highest-risk time step".** Taking the maximum over time steps pushes wallets that appear in many steps toward the top. Compare this with the latest or mean risk.
- [ ] **Rank-scale the unsupervised signals.** `min_max_scale` in `risk_engine.py` lets one extreme wallet compress every other wallet toward 0 on heavy-tailed signals (amount, degree). Try percentile ranks, then re-run `run_unsupervised_elliptic_plus.py` and update the numbers above.

**P2 — operations**

- [ ] **Show performance drift in the app.** Add the per-time-step F1 chart to the dashboard, and document that the models need retraining when illicit behaviour changes (as after step 43).
- [ ] **Remove the old parquet copies from git history** (`git lfs migrate import`). This rewrites history, so everyone must re-clone afterwards. Agree on it as a team first.
- [ ] **Rebuild the Linux AppImage** from the current code.

---

## Prebuilt desktop build (Linux)

A Linux AppImage of the desktop interface is available on [Google Drive](https://drive.google.com/file/d/1XKrpaMlFW2FChI7zCK9j118-lNAXL5-P/view?usp=sharing). It was built before the changes above.

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
