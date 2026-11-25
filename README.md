# BATMAN 2.0 — Bitcoin Anomaly Traffic & Monitoring Analysis Network

**BATMAN scores Bitcoin wallets and transactions for likely illicit activity, and explains every score — so an analyst can see *what* to investigate first and *why*.** It runs entirely on your own machine: no cloud services, no account, nothing leaves the computer.

Built for **Smart India Hackathon 2026**, problem **SIH26146** (AI / Blockchain & Cyber Security).

---

## What it does, in plain terms

Money laundering on Bitcoin hides in plain sight: every transaction is public, but there are hundreds of millions of them, and almost all are ordinary. BATMAN does two complementary things:

1. **Historical analysis (offline).** It learns from a large, real, *labelled* dataset of Bitcoin activity (the Elliptic / Elliptic++ datasets — see [Data](#data--where-everything-comes-from)) where some transactions and wallets are known to be illicit. A machine-learning model trained on this data ranks wallets and transactions by how likely they are to be illicit, and the app explains each ranking.

2. **Live monitoring (online).** It watches the *current* Bitcoin network in real time through a free public API, and flags transactions whose **shape** matches patterns associated with laundering — mixers, consolidation sweeps, peeling chains, dust spam, and addresses on a watchlist. Each flag is explained in words.

Everywhere, a score is a **starting point for a human investigation, not a verdict**. BATMAN surfaces the suspicious few out of the ordinary many, and always says why.

---

## How BATMAN tells "suspicious" from "ordinary"

This is the heart of the project, so here it is in plain terms.

**First, an honest truth: no single clue proves anything.** Every Bitcoin transaction is cryptographically real — there is no "fake" transaction. And a perfectly ordinary person *can* occasionally do something that looks suspicious. So BATMAN never judges a wallet on one transaction or one signal. It weighs **several dimensions together**, and the real signal comes from the **combination**:

| Dimension | What ordinary activity looks like | What can look suspicious |
|---|---|---|
| **Structure** (how many inputs/outputs) | 1 input → 1–2 outputs (a normal payment + change) | **many inputs → one output** (a *sweep*, collecting funds); **one → many outputs** (a *payout / peeling* distribution); **many equal-value outputs** (a *mixer / CoinJoin* hiding the trail) |
| **Amount** | varied, purposeful amounts | **very large** totals, **suspiciously round** numbers (1.00000000 BTC), or lots of **tiny "dust"** outputs used to spam/taint addresses |
| **Speed / timing** | occasional transactions over time | **bursts** — many transactions in a very short window, or funds hopping through several wallets rapidly (fast layering) |
| **Repetition** | a one-off pattern now and then | the **same laundering shape repeated systematically** — split, mix, peel, consolidate, again and again |
| **Company it keeps** (connectivity) | ordinary counterparties | money that **traces back, within a few hops, to an address already known to be criminal** (a watchlist / known dark-market address) — the single strongest real-world signal |

So the difference between a normal and a criminal wallet is rarely *one* transaction — it's **doing suspicious things repeatedly, at scale, while the money keeps connecting to other known-bad actors.** The historical ML model learns this combination from ~166 features at once; the live engine checks the structural signals above on each transaction.

**And we only ever give suggestions.** BATMAN hands a human a **ranked, explained shortlist** — *"these are the most suspicious, and here's exactly why each one"* — then a person (with more evidence, or a watchlist confirmation) makes the real decision. It does not, and is not meant to, declare anyone guilty.

---

## Status at a glance

| Component | State | What it is |
|---|---|---|
| **Web app** (`app.py`, Streamlit) | ✅ Works after cloning | The main interface: 7 pages built on precomputed Elliptic++ risk scores committed to the repo (Git LFS). |
| **Live monitor** (`src/core/backend/live/`) | ✅ Live & explainable | Pulls current Bitcoin transactions from the public mempool.space API and runs an explainable forensic analysis on each. |
| **Supervised models** (`scripts/train_*.py`) | ✅ Reproducible | Random Forests on Elliptic and Elliptic++, trained with an honest temporal split and no time-revealing features. |
| **Unsupervised pipeline** (`src/core/backend/`) | ⚠️ Experimental | Isolation Forest + HDBSCAN with no label-derived weights. Powers the **Upload & Score** page. |
| **Desktop app** (`main.py`, PySide6) | ✅ Works after cloning | Same data layer (`src/data/`) as the web app, as a native desktop window. |

---

## Quick start (web app)

Requires **Python 3.10+** (pinned and tested on 3.13) and [**Git LFS**](https://git-lfs.com) (the committed score files are stored with LFS).

```bash
git lfs install
git clone https://github.com/janhavizope/BATMAN-2.O.git
cd BATMAN-2.O

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py               # web app  ->  http://127.0.0.1:8501
python main.py                     # desktop app (same data)
```

No dataset download is needed. The app reads the scores in `data/elliptic_plus/*.parquet`. If you cloned **before** installing Git LFS, those files are tiny pointer files — both apps detect this and tell you to run `git lfs install && git lfs pull` to fetch the real data.

---

## The web app, page by page

| Page | What it shows |
|---|---|
| **Overview** | Wallet and transaction totals, flagged transactions per time step, severity breakdown, top alerts, and model-performance drift over time. |
| **Live Monitor** | Real-time Bitcoin transactions from the public API, each scored by the forensic engine, with a normal-vs-suspicious breakdown (auto-refreshes). |
| **Transaction Analysis** | Held-out and unlabelled transactions with risk %, true label, and linked transactions (highest-risk first). |
| **Network Graph** | The top alerted wallets, the transactions they sent/received, the flows between them, graph statistics, and a centrality table. |
| **Investigate Wallet** | A flagged wallet's full profile in tabs — **Why flagged** (verdict, feature contributions coloured by direction, and a comparison with typical licit/illicit wallets), Transactions, Timeline, Network. |
| **Alerts** | Two tabs: the **Alert Feed** (flagged wallets ranked by risk with the top reason) and **Scan & Lookup** (adjustable risk threshold, ranked list, and per-address lookup). |
| **Upload & Score** | Upload your own transactions (CSV or JSON), see the validation report, and get every wallet ranked by the unsupervised score with reasons. |

Only wallets and transactions the model **did not train on** (the test split and unlabelled data) are shown as predictions — training-set scores are hidden, because the model has already seen those labels.

### How the alert threshold and severity are chosen

The alert threshold is **not** a fixed 0.5. `choose_threshold` in [`src/data/elliptic_source.py`](src/data/elliptic_source.py) takes the threshold with the best illicit F1 on a **held-out validation period** (time steps 30–34, never trained on), then raises it until the number of flagged items fits a review budget (`WALLET_ALERT_BUDGET = 500`, `TX_ALERT_BUDGET = 1000`). Precision and recall are reported on the **test period** (steps 35–49), which is never used to pick the threshold, so the numbers are not optimistic.

The wallet risk shown in the app is **derived from the transaction model** — a wallet's risk is the highest risk of any transaction it sent or received — because that ranks wallets far better than the stand-alone wallet model (ROC-AUC 0.92 vs 0.73). On the current scores this flags **936 wallets** at test precision 100%, recall 0.5%; the transaction threshold (0.993) flags 1,022 transactions at test precision 100%, recall 2.9%.

**Severity (Critical / High / Medium) is a wallet's position in the risk-ranked list:** top 10% Critical, next 30% High, the rest Medium. Because peak risk saturates at 1.0 for every alert, ties are ordered by the wallet's **mean** transaction risk (how pervasive the illicit exposure is), giving **94 Critical, 281 High, 561 Medium**.

### Upload & Score — bring your own data

Required columns: `txid`, `timestamp`, `input_addresses`, `output_addresses`, `input_amounts`, `output_amounts` (list fields use `|` or JSON arrays). `fee` and network fields are optional — real on-chain data has no network fields. The page offers a template CSV. Uploads are validated by [`validator.py`](src/core/backend/validation/validator.py) and scored in memory by [`pipeline.py`](src/core/backend/pipeline.py); scores are relative to the uploaded file and nothing leaves the machine.

> ⚠️ This page uses the **unsupervised** score, which on real labelled wallets is close to random (ROC-AUC ~0.54). Its HIGH/CRITICAL levels mean *"unusual compared with the rest of this file"*, not *"illicit"*. See [Unsupervised pipeline](#unsupervised-pipeline-experimental).

---

## Data — where everything comes from

All analysis is built on two public, academic datasets of **real** Bitcoin activity. We did not collect or label any transactions ourselves; the ground-truth labels come from the datasets' authors.

> **Key point — the models are trained on real data, not synthetic data.** Every supervised model BATMAN ships is trained on *real, labelled* Bitcoin transactions and wallets from the **Elliptic / Elliptic++** datasets. Synthetic data appears **only** as an optional demo for the experimental [unsupervised pipeline](#unsupervised-pipeline-experimental) and as tiny fixtures in the test suite — it is **never** used to train the models whose results we report.

| Dataset | What we use it for | Source |
|---|---|---|
| **Elliptic++** | Transaction model, wallet model, wallet↔transaction links, transaction-flow edges | [github.com/git-disl/EllipticPlusPlus](https://github.com/git-disl/EllipticPlusPlus) |
| **Elliptic** | Transaction model | Derived from Elliptic++ by `scripts/prepare_elliptic.py`; the original is on [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) |

**What's in them.** Elliptic covers **49 time steps** (about two weeks each) of real Bitcoin transactions from 2016, each labelled **illicit, licit, or unknown**. Elliptic++ adds about **823,000 wallet addresses** with behavioural features, and the links between wallets and transactions. Features are pre-engineered graph aggregates (local and neighbourhood statistics) published by the datasets' authors.

**The raw CSVs are not committed** (they exceed GitHub's 100 MB limit). To retrain from scratch, place these files from the Elliptic++ repository into `data/elliptic_plus/`:

```
data/elliptic_plus/txs_features.csv                      (Transactions Dataset/)
data/elliptic_plus/txs_classes.csv                       (Transactions Dataset/)
data/elliptic_plus/txs_edgelist.csv                      (Transactions Dataset/)
data/elliptic_plus/wallets_features_classes_combined.csv (Actors Dataset/)
data/elliptic_plus/AddrTx_edgelist.csv                   (Actors Dataset/)
data/elliptic_plus/TxAddr_edgelist.csv                   (Actors Dataset/)
```

`prepare_elliptic.py` reconstructs `data/elliptic/` in Elliptic's original layout from these files, so a Kaggle account is not required. **For the live monitor, no dataset is needed at all** — it reads transactions straight from the public API.

---

## How the models are built and evaluated

Run everything from the project root after placing the datasets:

```bash
python scripts/run_pipeline.py          # check data -> prepare -> train -> score, in order
```

or step by step:

```bash
python scripts/prepare_elliptic.py       # data/elliptic/ from Elliptic++ (skipped if present)
python scripts/train_elliptic.py         # Elliptic transaction model
python scripts/train_elliptic_plus.py    # Elliptic++ transaction and wallet models
python scripts/score_transactions.py     # -> tx_scores.parquet, tx_edges.parquet
python scripts/score_wallets.py          # -> wallet scores / timeline / explanations / edges
```

**The method, and why it's honest:**

- **Temporal split.** Train on steps 1–29, hold out 30–34 to pick the alert threshold, test on 35–49. Each wallet belongs to exactly one set by when it first appears — no wallet trains and is then tested on.
- **No time-revealing features.** `time_step` and the block-number features are used only to make the split, never as model inputs — otherwise the model would "cheat" by recognising *when* a wallet was active.
- **Metrics reported per time step**, so performance over time is visible.

### Results (temporal split)

Illicit is the positive class. Test = steps 35–49.

| Model | Train / test rows | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Elliptic transactions | 26,381 / 16,670 | 97.37% | 86.6% | 70.5% | 0.777 | 0.933 |
| Elliptic++ transactions | 26,381 / 16,670 | 97.53% | 88.8% | 71.0% | 0.789 | 0.930 |
| Elliptic++ wallets | 215,725 / 114,345 | 84.73% | 19.2% | 32.7% | 0.242 | 0.770 |

### What the numbers honestly mean

- **The dark-market shutdown breaks the transaction models.** F1 is high through step 42 and collapses to near zero from step 43, after a real dark-market shutdown changed what illicit activity looks like. The model can't detect patterns it never saw — exactly as the original Elliptic paper reports. (The **Overview** page visualises this drift.)
- **Wallet prediction is genuinely hard.** Old write-ups claimed wallet F1 ~0.90 — but that came from a random split and block-number features, i.e. leakage. Under an honest temporal split it is 0.24. We report the honest number.
- **So the app derives wallet risk from the transaction model** (max risk of a wallet's transactions), which ranks far better (ROC-AUC 0.92 vs 0.73).
- **The threshold is tuned on validation, scored on test** — never on the data it's measured on.

---

## How accurate is it? — plain summary

Every number below is measured on data the models **did not train on** (or on transactions we invented by hand). None of it is made up.

| What we tested | Score | In plain words |
|---|---|---|
| **Supervised transaction model** (real held-out transactions) | **ROC-AUC 0.93**, precision 89% | Separates real illicit from licit transactions very well; when it flags, it's usually right. Runs on the dataset (needs its 166 graph features). |
| **Live-computable supervised model** (runs on live & uploaded transactions) | **ROC-AUC 0.82** | A second model trained on only the features a raw transaction has (value, fee, input/output counts), so it scores the **live feed and uploads** — where the full model can't. A ranking signal, not a high-confidence verdict. |
| **Supervised wallet risk** (balanced test: 50 real criminal + 50 real legit wallets, held-out) | **83% accuracy, 95% precision, 0.89 AUC** | Catches 7 of 10 criminals with almost no false alarms (2 of 50 legit). When it says "criminal," it's right 95% of the time. |
| **Unsupervised score** (label-free, real wallets) | **ROC-AUC 0.76** (was 0.54) | A fallback for when there are no labels; clearly better than random, but weaker than the supervised model. |
| **Live forensic engine** (10 hand-made transactions) | **10/10 correct** | Cleared all 4 ordinary payments, flagged all 5 laundering patterns (mixer, sweep, peeling, dust, watchlist), and correctly did **not** over-flag a large-but-ordinary round transfer. |
| **Combined (ensemble)** | supervised dominates (weight 0.9) | Mixing in the weaker unsupervised score can't beat the supervised model alone, so the strong model decides. |

**The honest headline:** the supervised model is the real detector — **right ~95% of the time when it raises a flag, catching ~70% of criminals** on a fair held-out test. The unsupervised and live-forensic engines fill in where no labels exist (uploads and the live feed). Fairness check: training used time steps 1–29 and the test wallets' transactions are all from steps 35–49, with **zero wallet overlap** between the two — the model had never seen them.

The two tests are reproducible: `python scripts/handmade_detection_test.py` (invented transactions) and the balanced wallet test in the evaluation notes.

---

## Live monitoring (online)

Everything above scores a **static, historical** dataset. The live monitor scores the **current** Bitcoin network instead — through a free public API, with no setup, no account, and no cost.

```bash
# In the app: open the "Live Monitor" page and click "Fetch & analyse now" (or enable Auto-refresh)
streamlit run app.py

# Headless, from the command line:
python -m src.core.backend.live.monitor --interval 30 --max-tx 40
```

**The feed — and yes, it's real.** Transactions come from the public [mempool.space / Esplora](https://mempool.space) API, a free Bitcoin blockchain explorer that serves the **actual live Bitcoin network** — real, unconfirmed transactions that real people and services are broadcasting *right now*. Nothing here is simulated or generated. It uses the standard library only (no extra dependency). The API is rate limited, so a cycle returns roughly 10 recent transactions regardless of `--max-tx` — a live sample of genuine current traffic. Only the public API is contacted, and all analysis stays on your machine.

**One cycle**, in [`src/core/backend/live/`](src/core/backend/live/):

1. **Fetch** — [`mempool_feed.py`](src/core/backend/live/mempool_feed.py) pulls recent transactions and maps each to the BATMAN schema (sending addresses/amounts from each input's previous output; satoshis → BTC; coinbase / OP\_RETURN entries without an address are skipped).
2. **Validate** — the same [`validator.py`](src/core/backend/validation/validator.py) the Upload & Score page uses.
3. **Analyse** — [`live_risk.py`](src/core/backend/live/live_risk.py) scores each transaction on structural forensic signals and explains every point. A wallet's risk is the highest risk of any transaction it took part in.
4. **Store** — [`live_store.py`](src/core/backend/live/live_store.py) keeps a rolling SQLite store (`data/live/live_monitor.sqlite`, git-ignored) that prunes itself to stay bounded.

### The forensic signals

Each is computable from a **single** transaction, needs **no training**, and is reported only when it raised the score — so every flag comes with its reason:

| Signal | Pattern it catches |
|---|---|
| `watchlist_hit` | an address on your known-bad list (`data/live/watchlist.json`) — forces CRITICAL |
| `equal_outputs` | many outputs of identical value — CoinJoin / mixer signature |
| `high_fan_out` | funds split across many outputs — distribution / peeling / payout batching |
| `high_fan_in` | many inputs consolidated — a sweep / collection |
| `peeling_chain` | two outputs, one a tiny remainder — a layering step |
| `dust_outputs` | outputs below the dust limit — spam / address tainting |
| `high_value` / `round_value` | large or suspiciously round totals |
| `self_churn` | an input address reused as an output amid several outputs |

**What it is, and isn't.** These are **indicators of patterns associated with illicit use, not proof.** Mixers, exchange batch payouts, and consolidation sweeps are usually legitimate; the engine surfaces them for a human to triage and explains why, and deliberately leaves ordinary payments LOW.

*The two live signals.* The forensic engine above is the explainable, rule-based signal. Alongside it, a **live-computable supervised model** ([`live_model.py`](src/core/backend/live/live_model.py)) adds a learned probability: it is trained on only the features a raw transaction has (value, fee, input/output counts), reaches **ROC-AUC 0.82** on held-out data, and runs on the live feed and uploads — the full 0.93 model **cannot**, because it needs Elliptic's 166 engineered graph-aggregate features. The live feed shows this as a `supervised_risk` column next to the forensic flags, and Upload & Score shows it per transaction. It trains itself once from the committed scores (`python scripts/train_live_tx_model.py` to rebuild it) — no dataset download needed. The **watchlist** is empty by default — populate `data/live/watchlist.json` (a JSON array of addresses, or `{"address", "reason"}` objects) from a source you trust.

---

## Unsupervised pipeline (experimental)

This was the project's original design, and it powers the **Upload & Score** page. It's worth understanding the difference from the main model.

**Supervised (our main model) vs unsupervised (this pipeline):**

- **Supervised** learns from an **answer key** — the Elliptic++ labels tell it which wallets were illicit, and it learns the pattern. It *needs* labels.
- **Unsupervised** has **no answer key at all.** It learns what "normal" looks like from the data itself and flags whatever **stands out**. It never sees a label.

**Why build it?** In the real world you usually *don't* have labels — someone hands you a pile of transactions nobody has tagged. That's exactly the Upload & Score page: drop in your own CSV with no labels, and we can still rank the oddest wallets. It's **label-free triage**.

**How it works — two algorithms:**

- **Isolation Forest** (anomaly detector): an outlier is *easy to isolate* with a few random cuts; a normal point is buried in the crowd and takes many. Each wallet gets an "unusualness" score.
- **HDBSCAN** (clustering): groups wallets that behave alike; any wallet that fits **no group** is an outlier.
- These, plus graph / temporal / amount signals, combine with **fixed weights** (`COMPONENT_WEIGHTS` in [`risk_engine.py`](src/core/backend/risk/risk_engine.py)) — **no weight comes from the labels** — into a 0–100 risk score, each with reasons. Missing signals drop out and the rest renormalise (e.g. no IP fields in real on-chain data).

```bash
python scripts/run_pipeline.py --unsupervised      # synthetic data, step by step
python scripts/run_unsupervised_elliptic_plus.py   # real Elliptic++ wallet-transaction graph
```

**The honest result:** on **real** Elliptic++ labelled wallets the *generic* anomaly score is close to random (ROC-AUC 0.54–0.61), while the supervised model reaches 0.92 on the same wallets. Generic "unusualness" is not the same as "illicit" on real Bitcoin — the most *unusual* wallets are big legitimate services (exchanges, pools), not criminals.

**Improving it with domain rules (0.54 → 0.76).** Blind anomaly detection fails, but a **label-free score built from AML-motivated red flags** does much better. [`scripts/score_wallets_domain_unsupervised.py`](scripts/score_wallets_domain_unsupervised.py) ranks wallets by two classic signals — **fee-as-share of value** (willing to overpay to move funds fast) and **transaction count / churn** — combined by rank with **no label-fitted weights**. Measured on the held-out test wallets it reaches **ROC-AUC 0.76** (vs ~0.54 for the generic score). Still well below the supervised 0.92 — unsupervised cannot learn "illicit" without examples — but now clearly better than random and fully explainable.

**Weighting the two engines.** When both a supervised and an unsupervised score exist, the verdict should trust the reliable one. [`scripts/ensemble_risk.py`](scripts/ensemble_risk.py) measures a reliability-weighted blend `W·supervised + (1−W)·unsupervised`: the result is that **no blend beats supervised-alone** (giving the unsupervised real weight only lowers the AUC), so the supervised model dominates the verdict (default `W = 0.9`). The unsupervised score's real job is where supervised **can't** run — uploaded data with no labels (the **Upload & Score** page) and the **live feed**.

---

## Project structure

```
BATMAN-2.O/
├── app.py                      # Streamlit entry point (main interface)
├── main.py                     # PySide6 desktop entry point
├── requirements.txt            # Pinned dependencies
├── data/elliptic_plus/         # Precomputed scores both apps read (committed, Git LFS)
├── scripts/                    # Data prep, training, scoring, and pipeline commands
├── src/
│   ├── config.py               # Every input and output path, in one place
│   ├── data/                   # Data layer shared by the web and desktop apps
│   │   ├── data_manager.py     #   Facade used by every page
│   │   └── elliptic_source.py  #   Reads the score files, picks alert thresholds
│   ├── web_frontend/           # Streamlit pages and shared components
│   ├── desktop_gui/            # PySide6 app (pages, widgets, charts, state)
│   ├── core/backend/
│   │   ├── live/               #   Live monitor: feed, forensic scoring, store, watchlist
│   │   ├── ingestion/ validation/ features/ graph/ ml/ risk/   # Unsupervised pipeline
│   │   └── pipeline.py         #   In-memory run used by Upload & Score
│   └── resources/fonts/        # Bundled Orbitron and JetBrains Mono
├── outputs/                    # Evaluation metrics (JSON) and charts
├── tests/                      # pytest suite
└── docs/                       # Research and architecture references
```

---

## Tests

```bash
pytest
```

The suite (114 tests) covers:

- **Unsupervised backend:** ingestion and validation (including data without network fields), features, graph, clustering, the risk engine, and the in-memory pipeline used by Upload & Score.
- **Shared data layer** (`tests/test_data_source.py`): threshold choice, position-based severity, Git LFS pointer detection, hidden training rows, alerts, entities, wallet↔transaction links, and the network graph.
- **Training and scoring scripts** (`tests/test_training_scripts.py`): temporal and wallet splits, dropped time features, and a full train → score run on small synthetic fixtures.
- **Live monitoring** (`tests/test_live.py`): the public-API feed, the schema mapping, forensic scoring, and the rolling store.
- **Desktop GUI:** pages and state.

CI runs the same suite on GitHub Actions on every push and pull request (`.github/workflows/tests.yml`), caching the Git LFS score files between runs.

### Manual check — Upload & Score

The **Upload & Score** page was exercised end to end on 33 transactions (24 ordinary payments plus a fan-in "hub": 8 wallets paying one address in a two-minute burst, then a single sweep-out), through the exact path the page uses, with the Streamlit app serving (`/_stcore/health` and the main page both HTTP 200). All 33 rows validated and 35 wallets were scored: the pipeline ranked the **hub Critical** (*"many transactions in a short window; high connectivity"*), its **8 feeders High**, and ordinary payers **Medium** — reasons matching the direction each signal pushed the score.

---

## Known limitations & future work

We report these openly — honesty about what a model *can't* do is part of the work.

- **Wallet features are whole-lifetime aggregates, not causal.** Every wallet feature is constant across an address's time steps, so they summarise a wallet's entire life (including the future). The temporal split keeps train/test wallets disjoint, but these features aren't available in real time — so the wallet metrics are optimistic for live use. **Next:** recompute features up to each time step from transaction-level data, then retrain for causal scoring.
- **A live-computable supervised model.** The strong transaction model needs engineered graph-aggregate features that a single live transaction can't provide. **Next:** retrain on features computable from raw live data, to replace the forensic engine's heuristics with a learned model live.
- **Detection decays when behaviour changes** (the step-43 dark-market break). Any deployment needs periodic retraining; the Overview page shows when drift sets in.

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

**Members:** Ankita Khokhar  · Neha More ·  Keziah Ann Mathew

Symbiosis Skills And Professional University 
