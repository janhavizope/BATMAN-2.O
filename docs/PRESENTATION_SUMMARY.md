# BATMAN 2.0 — Presentation Summary

**Bitcoin Anomaly Traffic & Monitoring Analysis Network**
Smart India Hackathon 2026 · Problem **SIH26146** (AI / Blockchain & Cyber Security)
Symbiosis Skills and Professional University

This is a plain-language summary of what the project does, how it works, and —
importantly — what its limits are. It reflects the system as it is actually
built today. For commands and file layout, see the [README](../README.md).

---

## 1. What the project does

BATMAN scores Bitcoin **wallets and transactions** for illicit activity, and
**explains every score**, so an analyst can decide what to investigate first.

It runs **fully offline**. There are no live blockchain APIs, cloud services,
or online AI calls during execution. Everything — the models, the scores, the
dashboard — runs on the local machine.

The goal is not to declare a wallet "guilty." It is to **rank and explain**:
turn hundreds of thousands of wallets into a short, ordered list of leads, each
with a reason attached.

---

## 2. The data

Two public research datasets of real Bitcoin activity:

| Dataset | What it adds |
|---|---|
| **Elliptic** | ~204k transactions across 49 two-week time steps, each labelled illicit / licit / unknown |
| **Elliptic++** | ~823k wallet addresses with behavioural features, plus the links between wallets and transactions |

The raw files are large (hundreds of MB) and are not stored in the repository.
The small precomputed **risk scores** that the app reads are committed (via Git
LFS), so the app works immediately after cloning.

---

## 3. How it works

### a. Supervised models (the main engine)

Two **Random Forest** classifiers — one for transactions, one for wallets —
learn to tell illicit from licit from the labelled data.

The honest part of the design is **how they are evaluated**:

- **Time-based split.** The models train on time steps 1–34 and are tested on
  35–49 — the future they have never seen. This is the standard, hard Elliptic
  benchmark. Wallets are split so none appears in both training and testing.
- **No "cheating" features.** The time step and block-number features (which
  reveal *when* a wallet was active) are removed before training, so the model
  must learn *behaviour*, not timing.

### b. Unsupervised pipeline (works without labels)

For data that has **no labels at all**, a second pipeline scores wallets using
**Isolation Forest** (how anomalous a wallet is) and **HDBSCAN** (whether it
falls outside every normal behaviour cluster), combined with graph, timing,
transaction and network signals into a single 0–100 risk score.

Crucially, **none of the weights are fitted to the labels** — the score is
genuinely unsupervised. Labels are used only afterwards, to measure how good
the ranking turned out to be.

### c. Explanations

Every flagged wallet comes with a reason. For the supervised model, a Saabas
decomposition breaks the Random Forest's score into per-feature contributions.
For the unsupervised score, each reason corresponds to a signal that actually
pushed the score up — so the explanation always matches the number.

---

## 4. What the user sees

A **Streamlit web app** and a **PySide6 desktop app**, both reading the same
data layer. Pages include:

- **Overview** — totals, flagged activity per time step, severity breakdown
- **Transaction Analysis** — held-out and unlabelled transactions with risk %
- **Network Graph** — the top alerted wallets, the transactions they touched,
  and the money flows between them
- **Entity Investigation** — a flagged wallet's profile, timeline and transactions
- **Alerts** — wallets ranked by risk, with the top contributing factor
- **Explainability** — per-feature contributions vs. typical licit / illicit wallets
- **Wallet Scanner** — adjustable threshold and address lookup
- **Upload & Score** — bring your own transaction file (CSV/JSON); it is
  validated and scored in memory, and nothing leaves the machine

Only wallets and transactions the model **did not train on** are shown as
predictions. The alert threshold is chosen automatically from the
precision-recall curve to keep the number of alerts reviewable.

---

## 5. Results — stated honestly

Test set = time steps 35–49 (the future). Illicit is the positive class.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Elliptic transactions | 91.7% | 72.0% | 0.81 | 0.94 |
| Elliptic++ transactions | 95.4% | 72.3% | 0.82 | 0.93 |
| Elliptic++ wallets | 48.7% | 20.2% | 0.29 | 0.86 |

What these numbers honestly tell us:

- **Transactions are detected well — until the market shutdown.** F1 stays high
  through step 42, then drops sharply from step 43, which follows a real
  dark-market shutdown. The model does not recognise the new illicit patterns
  that appear afterwards. This is a known hard case for this dataset, not a bug.
- **Wallet-level prediction is genuinely hard.** Under a fair time-based split,
  wallet F1 is 0.29 — far below the ~0.90 that earlier, leaky evaluations
  suggested. Most of that old number came from the model recognising *when* a
  wallet was active, or recognising wallets it had already trained on.
- **The unsupervised score is near-random on real data.** On Elliptic++ its
  ROC-AUC is about 0.5, while the supervised model reaches 0.84. The
  hand-weighted score works on the synthetic demo data (which has injected
  patterns) but does not capture what separates illicit wallets in real data.
  It is useful as label-free triage, not as a replacement for the supervised model.

We present these limits on purpose. A system that **knows where it is weak** is
more trustworthy than one that reports impressive numbers that fall apart under
scrutiny.

---

## 6. Engineering

- **Tech stack:** Python, scikit-learn, pandas, NetworkX, Streamlit, PySide6.
- **One command** rebuilds everything: `python scripts/run_pipeline.py`
  (check datasets → prepare → train → score).
- **Reproducible and tested:** a pytest suite (90+ tests) covers the data
  layer, feature engineering, the models, the risk engine and the scoring
  scripts; GitHub Actions runs it on every push.
- **Offline by design:** fonts are bundled, scores are precomputed, no network
  calls at runtime.

---

## 7. Team

**Team Lead:** Janhavi Zope
**Members:** Ankita Khokhar · Nutanapushpa Vasarla · Neha More · Lakshmi Sai Lekhana Bavikatta · Keziah Ann Mathew
