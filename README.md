# BATMAN 2.0 — Bitcoin Anomaly Traffic & Monitoring Analysis Network

BATMAN scores Bitcoin wallets and transactions for illicit activity and explains each score, so an analyst can decide what to investigate first. It runs locally with no cloud services.

Built for **Smart India Hackathon 2026**, problem **SIH26146** (AI/Blockchain & Cyber Security).

---

## Status at a glance

| Component | State | Notes |
|---|---|---|
| **Streamlit web app** (`app.py`) | ✅ Works after cloning | Reads the precomputed Elliptic++ risk scores committed in `data/elliptic_plus/` (Git LFS). |
| **Live monitor** (`src/core/backend/live/`) | ✅ Live, explainable | Pulls current Bitcoin transactions from the mempool.space API and runs a forensic analysis (explainable structural signals) on each one. Indicators, not a verdict — see [Live monitoring](#live-monitoring-online). |
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

The alert threshold is not a fixed 0.5. `choose_threshold` in [`src/data/elliptic_source.py`](src/data/elliptic_source.py) takes the threshold with the best illicit F1 on the **validation period** (steps 30–34, which the model did not train on), then raises it until the number of flagged wallets fits a review budget (`WALLET_ALERT_BUDGET = 500`, `TX_ALERT_BUDGET = 1000`). Precision and recall are reported on the **test period** (steps 35–49), which is never used to pick the threshold, so they are not optimistic. Those budgets are the one place to change alert volume.

The wallet risk the app shows is **derived from the transaction model** — a wallet's risk is the highest risk of any transaction it sent or received — because that ranks wallets far better than the wallet model (ROC-AUC 0.92 vs 0.73; see [the TODO](#open) and `scripts/derive_wallet_risk_from_tx.py`). On the current scores the wallet threshold is 1.0: it flags **936 wallets** — every wallet exposed to a near-certain illicit transaction (ties at 1.0 push the count past the 500 budget) — with test precision 100% and recall 0.5%. The transaction threshold is 0.993. It flags 1,022 transactions with test precision 100% and recall 2.9%.

**Severity (Critical / High / Medium) is a wallet's position in the risk-ranked alert list:** Critical is the top 10%, High the next 30%, and Medium the rest (`CRITICAL_SHARE` and `HIGH_SHARE` in `elliptic_source.py`). Peak transaction risk saturates at 1.0 for every one of the 936 alerts, so they are ordered by a second signal — the wallet's **mean** transaction risk, i.e. how pervasive the illicit exposure is. Banding by position then gives **94 Critical, 281 High, 561 Medium**, where Critical wallets have essentially all of their transactions near-certain illicit (mean risk 1.00), High around 0.94, and Medium touched a bad transaction amid more mixed activity (mean ~0.56). The bands now carry real information — unlike the earlier wallet-model scores, which tied 166 wallets at exactly 1.0 with nothing to separate them.

### Upload & Score

Required columns: `txid`, `timestamp`, `input_addresses`, `output_addresses`, `input_amounts`, `output_amounts`. List fields use `|` or JSON arrays. `fee` and the network fields (`src_ip`, `dst_ip`, `src_port`, `dst_port`, `asn`, `geo_country`) are optional, because real on-chain data has no network fields. The page offers a template CSV. Uploads are validated with [`validator.py`](src/core/backend/validation/validator.py) and scored in memory with [`pipeline.py`](src/core/backend/pipeline.py). Scores are relative to the uploaded file, and nothing leaves the machine.

The page shows a warning at the top. On real labelled Elliptic++ wallets this unsupervised score reached ROC-AUC 0.54, close to random guessing (see [Unsupervised pipeline](#unsupervised-pipeline-experimental)). Its HIGH and CRITICAL levels mean "unusual compared with the rest of the file", not "illicit".

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

- **Temporal split.** Train on time steps 1–29, hold out 30–34 as a validation period (used only to pick the alert threshold), and test on 35–49. Each wallet is assigned by the time step it first appears in: first seen in 1–29 trains (using only its rows up to step 29), 30–34 is a validation wallet, 35 or later is a test wallet. No wallet is in more than one set, and none trains on validation or test data.
- **No time-revealing features.** `time_step` / `Time step`, and the wallet block-number features (`first_block_appeared_in`, `last_block_appeared_in`, `first_sent_block`, `first_received_block`), are used only for the split, never as model inputs.
- **Metrics per time step.** Each metrics file includes `f1_illicit_by_time_step` for the test period.

Outputs:

- Models go to `models/` (git-ignored).
- Metrics go to `outputs/evaluation/elliptic*_metrics.json`.
- Charts go to `outputs/graphs/elliptic*/`.

### Results (temporal split)

Training on steps 1–29, validation on 30–34 (threshold only), test on steps 35–49. Illicit is the positive class.

| Model | Train / test rows | Accuracy | Precision (illicit) | Recall (illicit) | F1 (illicit) | ROC-AUC |
|---|---|---|---|---|---|---|
| Elliptic transactions | 26,381 / 16,670 | 97.37% | 86.6% | 70.5% | 0.777 | 0.933 |
| Elliptic++ transactions | 26,381 / 16,670 | 97.53% | 88.8% | 71.0% | 0.789 | 0.930 |
| Elliptic++ wallets | 215,725 / 114,345 | 84.73% | 19.2% | 32.7% | 0.242 | 0.770 |

Mean illicit F1 per time step:

| Model | Steps 35–42 | Steps 43–49 |
|---|---|---|
| Elliptic transactions | 0.86 | 0.04 |
| Elliptic++ transactions | 0.87 | 0.04 |
| Elliptic++ wallets | 0.28 | 0.16 |

### Evaluation notes

- **The dark-market shutdown breaks the transaction models.** F1 is high up to step 42 and falls to almost zero from step 43, after the dark-market shutdown. This matches the original Elliptic paper. The model does not detect the new illicit patterns that follow.
- **Wallet prediction is much harder than the old numbers suggested.** With the earlier random split and block-number features, the wallet model reported F1 0.90 (precision 93.9%, recall 86.4%). Under a temporal split, without those features and without testing on wallets it trained on, illicit F1 is 0.24. Most of the old score came from leakage: the model recognised when a wallet was active, or recognised wallets it had already seen.
- **The wallet features are whole-lifetime aggregates, not causal.** `scripts/audit_wallet_feature_leakage.py` shows that every wallet model-input feature is identical across all of an address's time-step rows: they summarise the wallet's entire life, not activity up to each step. The temporal split still keeps training and test wallets disjoint (so the test numbers are not inflated by train/test overlap), but the features are not available in real time — when a wallet is first seen you cannot yet know its later transactions — so the reported wallet metrics are optimistic for a live deployment. (The app no longer uses the wallet model for the displayed risk — it derives wallet risk from the transaction model instead; see the [TODO](#open).)
- **The threshold is chosen on a validation period, not the test set.** The models train on steps 1–29, the alert threshold is picked on the held-out validation period (steps 30–34), and precision/recall are reported on the test period (steps 35–49), which is never used for tuning. Holding 30–34 out of training costs some headline performance (for example wallet illicit F1 0.29 → 0.24 and ROC-AUC 0.86 → 0.77 versus training on 1–34), but the reported numbers are now honest for a model that has not seen the data it is measured on.

---

## Live monitoring (online)

Everything above scores a **static, historical** dataset (Elliptic/Elliptic++, real
Bitcoin from 2016). The live monitor scores the **current** network instead.

```bash
# web: open the app and use the "Live Monitor" page (Fetch & analyse / Auto-refresh)
streamlit run app.py

# headless, public API feed (no setup):
py -m src.core.backend.live.monitor --interval 30 --max-tx 40

# headless, your own full node (firsthand, no rate limit — see node setup below):
# each cycle processes everything the node announced since the last cycle
py -m src.core.backend.live.monitor --source zmq --interval 30
```

### Two feeds

| Feed | Setup | Throughput | Use |
|---|---|---|---|
| **Public API** (default) | none | ~10 tx/cycle, rate limited (a small sample of traffic) | demos, trying it out |
| **Your node (ZMQ)** | run Bitcoin Core (tested on 32.1) | every transaction, firsthand, instant | real monitoring |

**Node setup (ZMQ feed).** The node is a **pruned** Bitcoin Core node — it joins the
real network and relays every transaction (so the live feed is complete), but keeps only
~20 GB of recent blocks instead of the ~600 GB a full archival node needs, so it fits on
a normal drive. [`docs/bitcoin.conf.sample`](docs/bitcoin.conf.sample) is a ready config
that stores everything on `D:\BitcoinData` and enables RPC + ZMQ. Steps:

1. Install [Bitcoin Core](https://bitcoincore.org/en/download/) (tested on 32.1) and
   `pip install pyzmq`.
2. Copy the sample to `D:\BitcoinData\bitcoin.conf`, then start the node:
   `bitcoind -datadir=D:\BitcoinData -conf=D:\BitcoinData\bitcoin.conf`. The first sync
   takes hours to a few days. That is an estimate, not a fixed requirement: even a pruned
   node downloads and verifies the whole chain once, then deletes old blocks. Pruning
   saves disk, not sync time. Sync time depends mostly on the disk (SSD vs hard drive),
   then RAM, CPU and bandwidth. Keep the datadir on an SSD and raise `dbcache` (the
   sample sets 4000 MB) to shorten it. Check progress with `py scripts/node_status.py`.
3. Point BATMAN at the node's cookie (written in the datadir):
   `set BATMAN_RPC_COOKIE=D:\BitcoinData\.cookie` (and `BATMAN_ZMQ_ENDPOINT` /
   `BATMAN_RPC_URL` if you changed the ports). Defaults match the sample config.
4. Run `--source zmq`, or pick **My node (ZMQ)** on the Live Monitor page.

[`zmq_feed.py`](src/core/backend/live/zmq_feed.py) subscribes to the node's `hashtx`
topic and resolves each transaction's full detail through the local RPC
([`node_rpc.py`](src/core/backend/live/node_rpc.py)). It talks only to your node, and
nothing leaves the machine. pyzmq is the only added dependency, and only this feed
needs it.

**Resolving the senders.** Bitcoin Core's `getrawtransaction` verbosity 2 adds each
input's previous output (`prevout`) and the `fee` only from block data. A transaction
still in the mempool, which is every transaction the live feed sees, comes back without
them. Without `prevout` there are no sending addresses. So `resolve_transactions` in
[`zmq_feed.py`](src/core/backend/live/zmq_feed.py) looks them up itself, in JSON-RPC
batches (a few HTTP requests per cycle, not one per input):

- **`gettxout`** for each input. The coin being spent is still unspent on-chain while the
  spending transaction is only in the mempool, so this works on a pruned node without
  `txindex`.
- **`getrawtransaction` on the parent** when the parent is itself unconfirmed.
- **`getmempoolentry`** for the fee.
- **The latest 6 blocks** (`getrawtransaction` with the block hash) for a transaction
  mined before its cycle ran. A pruned node keeps recent blocks, and inside a block Core
  does include prevouts and the fee.

bitcoind also announces every transaction a second time when it is mined. The listener
remembers recent transaction ids and skips those repeats.

**The subscription stays open between cycles.** ZMQ is publish/subscribe: bitcoind does
not keep messages for a subscriber that is not connected. So `ZMQListener` holds one
connection open on a background thread and queues every announced transaction id, and
each cycle processes the whole queue (up to `ZMQ_MAX_PER_CYCLE` = 5,000; the rest wait a
cycle). Transactions announced while the monitor sleeps are kept. The CLI and the Live
Monitor page share the same listener, so page reruns don't reconnect. Each cycle reports
the feed's health:

- **announced:** transactions the node sent us.
- **queued:** waiting for the next cycle.
- **missed:** gaps in bitcoind's per-message sequence numbers, i.e. messages that never
  arrived. It should stay at 0.
- **unresolved:** announced transactions the node could no longer return. They were
  evicted, or mined more than 6 blocks before their cycle ran.
- **repeats skipped:** re-announcements of transactions already queued (bitcoind sends
  each one again when it is mined).

(Before this fix, each cycle opened a new connection, read up to 40 transactions, and
closed it. Everything announced during the 30-second pause was lost, so the "full
firehose" was really a sample of about 40 per cycle.)

One cycle, in [`src/core/backend/live/`](src/core/backend/live/):

1. **Fetch** — [`mempool_feed.py`](src/core/backend/live/mempool_feed.py) pulls recent
   transactions from the public mempool.space / Esplora API (standard library only,
   no new dependency) and maps each one to the BATMAN schema: inputs' previous
   outputs give the sending addresses and amounts, satoshis become BTC, and
   coinbase / OP\_RETURN entries without an address are skipped.
2. **Validate** — the same [`validator.py`](src/core/backend/validation/validator.py)
   the Upload & Score page uses.
3. **Analyse** — [`live_risk.py`](src/core/backend/live/live_risk.py) scores each
   transaction on structural forensic signals and explains every point. Wallet risk
   is the highest risk of any transaction the address took part in.
4. **Store** — [`live_store.py`](src/core/backend/live/live_store.py) keeps a rolling
   SQLite store (`data/live/live_monitor.sqlite`, git-ignored) of the scored
   transactions and per-wallet latest/peak risk, and prunes itself to stay bounded.

### The forensic signals

Each is computable from one transaction, needs no training, and is reported only when it
raised the score (one `SIGNALS` table drives both the score and the reasons, like the
batch `risk_engine`):

| Signal | Pattern it catches |
|---|---|
| `watchlist_hit` | an address on your known-bad list (`data/live/watchlist.json`) — decisive, forces CRITICAL |
| `equal_outputs` | many outputs of identical value — CoinJoin / mixer signature |
| `high_fan_out` | funds split across many outputs — distribution / peeling / payout batching |
| `high_fan_in` | many inputs consolidated — sweep / collection |
| `peeling_chain` | two outputs, one a tiny remainder — a layering step |
| `dust_outputs` | outputs below the dust limit — spam / address tainting |
| `high_value` / `round_value` | large or suspiciously round totals |
| `self_churn` | an input address reused as an output amid several outputs |

**What it is, and isn't.** These are **indicators of patterns associated with illicit use,
not proof**. A mixer, an exchange batch payout, and a consolidation sweep are all
legitimate most of the time; the engine surfaces them for a human to triage and explains
why. It deliberately leaves ordinary payments LOW — most live traffic is ordinary.

This replaces the near-random *unsupervised* score (ROC-AUC ~0.54 on real labels) as the
live signal. The strong supervised transaction model (ROC-AUC 0.93) still **cannot** run
live — it needs Elliptic's 166 engineered graph-aggregate features, which are not
reproducible from a raw live transaction — so the forensic engine is the honest,
explainable live analysis until a supervised model is retrained on live-computable
features. The public API is rate limited, so `/mempool/recent` returns about 10
transactions per cycle regardless of `--max-tx`; run a full node's ZMQ feed for higher
throughput. The watchlist is empty by default — populate `data/live/watchlist.json`
(a JSON array of addresses, or `{"address", "reason"}` objects) from a source you trust.

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
| Unsupervised score (all labelled wallets) | 0.607 | 1% | 0.8% | 1.6% |
| Unsupervised score (held-out test wallets) | 0.536 | 0% | 1.6% | 2.7% |
| Supervised Random Forest (same test wallets) | 0.732 | 18% | 13% | 12.6% |

The honest result: **on real Bitcoin behavioural features the rank-scaled unsupervised score is close to random** (ROC-AUC 0.54–0.61), while the supervised model reaches 0.73 on the same wallets. The fixed component weights that work on the synthetic data (which was built with injected fan-in / fan-out / burst patterns) do not capture what separates illicit wallets in Elliptic++. The unsupervised pipeline is useful as a label-free triage tool and for the Upload & Score page, but it is not a substitute for the supervised model where labels exist. (The supervised numbers here are lower than in [Results](#results-temporal-split) because the model now trains on steps 1–29, and this ranks one row per wallet rather than per time step.)

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
- **Live monitoring** (`tests/test_live.py`, `tests/test_live_zmq.py`): the API feed, forensic scoring, the store, and the ZMQ feed. The ZMQ tests use a fake node that answers the way Bitcoin Core does (no `prevout` for mempool transactions), plus a real local ZMQ publisher.
- **Real Bitcoin Core node, optional** (`tests/test_live_bitcoind.py`): starts a throwaway **regtest** bitcoind (a private chain, ready in seconds, no sync), sends real transactions, and checks the ZMQ feed end to end. It covers transactions sent between cycles, transactions mined before their cycle, repeat announcements, and a child of an unconfirmed parent. It is skipped unless you point it at a `bitcoind` binary:
  ```bash
  set BATMAN_TEST_BITCOIND=C:\path\to\bitcoin-32.1\bin\bitcoind.exe
  pytest tests/test_live_bitcoind.py -v
  ```

CI runs the same suite on GitHub Actions (without the optional real-node test). It caches the Git LFS score files between runs, so each run doesn't re-download about 55 MB from the repository's LFS quota.

### Manual check — Upload & Score

The **Upload & Score** page was exercised end to end on a sample of 33 transactions: 24 ordinary one-input/one-output payments, plus a fan-in "hub" (8 wallets paying one address within a two-minute burst, then a single sweep-out). The file was run through the exact path the page uses — `TransactionIngestor().ingest(...)` then `pipeline.score_wallets(...)` — and the Streamlit app was started to confirm it serves (`/_stcore/health` and the main page both returned HTTP 200, with no errors in the server log).

Result: all 33 rows validated (0 invalid, 0 duplicate txids) and 35 wallets were scored. The pipeline ranked the **hub Critical** (84/100 — *"many transactions in a short window; high transaction volume; high connectivity"*), its **8 feeders High** (*"unusually anomalous; HDBSCAN outlier; high transaction amounts"*), and the ordinary payers **Medium** — the reasons matching the direction each signal pushed the score. Scores are relative to the uploaded file and unsupervised (ROC-AUC ~0.54 on real labels), so they mean "unusual within this file", not "illicit"; on this crafted file the flagged wallets were the injected anomalies.

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

- [x] **Position-based severity.** Critical, High and Medium are a wallet's position in the risk-ranked alert list (top 10% / next 30% / rest). An earlier rank rule let a large tie at risk 1.0 fill Critical+High and leave High empty; positional banding keeps the 10/30/60 split, and ties are now ordered by mean transaction risk so the bands carry information (94 / 281 / 561 on the current scores). See [Alert threshold](#alert-threshold).
- [x] **Detect Git LFS pointer files.** If the repository is cloned without Git LFS, both apps say to run `git lfs pull` instead of failing when they read the scores (`scores_status()` in `elliptic_source.py`).
- [x] **Warn on the Upload & Score page** that the unsupervised score is close to random on real labelled wallets.
- [x] **Cache Git LFS files in CI** so each run doesn't use up the LFS bandwidth quota.
- [x] **Drop duplicate transactions on upload.** `validator.py` kept every row sharing a `txid` (it only counted them); it now keeps the first and drops the rest, so a repeated transaction is not double-counted in wallet features.
- [x] **Stop a bad timestamp from failing a whole upload.** `validator.py` parses the timestamp during row validation, so one row with an unparseable date is dropped as invalid instead of raising `DateParseError` for the entire file.
- [x] **Rank-scale the unsupervised signals.** `rank_scale` (was `min_max_scale`) in `risk_engine.py` scales each signal by percentile rank, so one extreme wallet no longer compresses the others toward 0 on heavy-tailed signals (amount, degree).
- [x] **Remove dead/duplicate modules.** Deleted `feature_extractor.py` (superseded by `feature_engineering.py`), `analysis/validate_features.py` and `validation/feature_correlation_check.py` (two copies of the same unused diagnostic), and `correlation/correlator.py` (used only by a test) — none were imported by the apps or pipeline.
- [x] **Make the pipeline's output messages honest.** `run_pipeline.py --unsupervised` no longer claims the desktop app reads `data/dev/results.json` (nothing reads it).
- [x] **Choose the alert threshold on a validation period, not the test set.** The models now train on steps 1–29, `choose_threshold` picks the threshold on the held-out validation period (30–34), and precision/recall are reported on the test period (35–49), never used for tuning. See [Alert threshold](#alert-threshold) and the [evaluation notes](#evaluation-notes).
- [x] **Causal risk timeline.** `score_wallets.py` writes each flagged wallet's risk as it accrues over time — by each step, the highest risk among its transactions up to that step — so the network-graph "entity risk timeline" rises to the wallet's headline risk instead of being a flat line.

#### Open

**P0 — evaluation**

- [x] **Checked the wallet features for future information** (`scripts/audit_wallet_feature_leakage.py`). Result: *every* wallet model-input feature is constant across an address's time steps, so all of them are whole-lifetime aggregates — not only `lifetime_in_blocks`, `num_timesteps_appeared_in`, `total_txs` and `btc_transacted_total`. The temporal split still keeps training and test wallets disjoint, so the test numbers are not inflated by overlap, but the features use hindsight (they summarise a wallet's whole life, including after it is first seen), so they are not valid for real-time scoring. **Still open:** recompute the features up to each time step — this needs the transaction-level data, not the aggregate CSV — then retrain, so the wallet scores become causal.

**P1 — models and product**

- [x] **Derive wallet risk from transaction risk, and adopt it as the app's wallet risk** (`scripts/derive_wallet_risk_from_tx.py`, then `score_wallets.py`). Scoring each wallet by the maximum risk of the transactions it sent or received (via `AddrTx` / `TxAddr`) ranks the held-out test wallets far better than the wallet model: **ROC-AUC 0.92 vs 0.73**, precision@1000 99% vs 13%. Part of the gap reflects that wallet and transaction labels share the same entity source, but the test wallets' transactions are all in the test period (never in the transaction model's training set), so it is a genuine held-out result. `score_wallets.py` now writes this as the wallet `risk`, keeps the wallet model's probability as `rf_risk` (behavioural context for the per-feature explanations), and keeps the mean transaction risk (`tx_mean`) to order severity. **Still open:** build a causal, up-to-each-time-step version for real-time use.
- [x] **Break ties among alerts.** Resolved by deriving wallet risk from transactions (above). Calibration could not have helped: it is a monotonic transform, so the wallets tied at exactly 1.0 would stay tied. The adopted wallet risk ties them at peak risk 1.0 too, but severity now orders them by **mean transaction risk** — a principled continuous signal — so Critical/High/Medium reflect how pervasive each wallet's illicit exposure is (see [Alert threshold](#alert-threshold)).
- [x] **Revisited "keep each wallet's highest-risk time step".** Because every wallet feature is constant across its time steps (see the audit above), a wallet's risk is identical at every step, so the maximum, latest and mean are the same — the choice has no effect on the current data. It will matter once the features are made causal; `score_wallets.py` keeps the maximum until then.

**P2 — operations**

- [x] **Show performance drift in the app.** The Overview page now shows the transaction model's illicit F1 per test time step (green where it works, red where F1 < 0.5), with a caption that detection collapses after the step-43 dark-market shutdown and the models need retraining when illicit behaviour changes (`get_model_drift()` in `elliptic_source.py`).
- [ ] **Remove the old parquet copies from git history** (`git lfs migrate import`). **Not done — deliberately deferred.** This rewrites every commit SHA and needs a force-push, which breaks every teammate's clone (everyone must re-clone) and cannot be undone, so it must be agreed as a team first and run as a separate announced step after this work is merged.
- [ ] **Rebuild the Linux AppImage** from the current code.

---

## Recent progress — live monitoring & the ZMQ node (November 2026)

A running log of the live-monitoring work added in this phase. Everything above
still stands; this section only records what is new and what is in progress.

### Done

- **Live monitoring is built and working on the public API feed.** The app now has a
  **Live Monitor** page (and a headless CLI) that pulls *current* Bitcoin transactions
  from the mempool.space / Esplora public API, validates them, scores each one with the
  forensic engine, and stores them in a rolling SQLite feed — real live traffic, no
  setup, no cost. This is fully demo-ready on its own.
- **Explainable forensic scoring instead of the near-random unsupervised score.**
  [`live_risk.py`](src/core/backend/live/live_risk.py) scores each transaction on
  structural signals (mixer-style equal outputs, fan-in sweeps, fan-out payouts,
  peeling chains, dust, round/high value, watchlist hits) and explains every point.
  Verified on live data: it flagged a real 1→54 equal-output mixer (HIGH) and a
  62-input consolidation (MEDIUM) while leaving ordinary payments LOW.
- **The full live backend** under [`src/core/backend/live/`](src/core/backend/live/):
  feed (`mempool_feed.py`), scoring (`live_risk.py`), rolling store (`live_store.py`),
  watchlist (`watchlist.py`), node RPC (`node_rpc.py`), ZMQ feed (`zmq_feed.py`), and the
  cycle/loop runner (`monitor.py`). Covered by tests in
  [`tests/test_live.py`](tests/test_live.py) and
  [`tests/test_live_zmq.py`](tests/test_live_zmq.py).
- **ZMQ feed fixed so it really is complete.** It used to open a new subscription on every
  cycle, read up to 40 transactions, then close it. Everything the node announced during
  the pause between cycles was lost. A persistent background listener now keeps the
  subscription open and queues every transaction; each cycle processes the whole queue.
  It also counts gaps in bitcoind's sequence numbers, so lost messages are reported
  instead of disappearing silently. See [Two feeds](#two-feeds). It is tested end to end
  against a real local ZMQ publisher.
- **ZMQ feed checked against a real Bitcoin Core node (32.1, regtest).** That check found
  that the feed produced **no transactions at all** on a real node. Core gives no
  `prevout` for mempool transactions, so no sender could be found, and every transaction
  was dropped. The earlier tests had used hand-made data that included `prevout`. The
  feed now resolves inputs itself (`gettxout`, mempool parents, the fee from
  `getmempoolentry`), finds transactions mined before their cycle ran, and skips
  bitcoind's repeat announcements. On the real node: 50 transactions sent between cycles
  gave 50 processed and 0 missed; 5 mined before their cycle were all found; 56 repeats
  were skipped. This now runs as `tests/test_live_bitcoind.py`.

### In progress — adding the ZMQ node feed

- **We are adding the second feed: a local Bitcoin Core node over ZMQ**, for the complete
  firehose instead of the rate-limited API sample (see [Two feeds](#two-feeds)). The code
  is already in place (`zmq_feed.py` + `node_rpc.py`); what remains is the one-time node
  sync.
- **The node is a pruned Bitcoin Core node** (keeps ~20 GB, not the full ~600 GB), stored
  on `D:\BitcoinData`, with RPC + ZMQ enabled. BATMAN reads its `.cookie` automatically
  (`src/config.py` auto-detects `D:\BitcoinData\.cookie`).
- **Initial sync is running** on the teammate machine that hosts the node (no node is
  installed on every machine). Check progress any time with `py scripts/node_status.py`.
  The sync downloads and verifies the chain once (hours to a couple of days on a laptop);
  it does not block anything — the Live Monitor works on the public API feed meanwhile.
- **When sync finishes** (`node_status.py` shows `READY`), switch to the node feed with
  `py -m src.core.backend.live.monitor --source zmq`, or pick **My node (ZMQ)** on the
  Live Monitor page. One flag — no other change.

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
