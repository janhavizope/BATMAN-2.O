"""
score_wallets.py
----------------
Gives every wallet a risk score. The score the app shows is derived from the
transaction model: a wallet's risk is the highest risk of any transaction it
sent or received (via AddrTx / TxAddr). The transaction model is much stronger
than the wallet model (illicit F1 ~0.79 vs ~0.24), so this ranks wallets far
better (ROC-AUC 0.92 vs 0.73 on test wallets; see derive_wallet_risk_from_tx.py).
The wallet model's own probability is kept as `rf_risk` for reference and drives
the per-feature behavioural explanations. If the transaction scores / edge lists
are absent, risk falls back to the wallet model.

Each wallet is tagged with the split it belonged to (train / val / test /
unknown) so that only validation, test and unknown wallets are presented as
genuine predictions.

Outputs (read by the Streamlit app):
    data/elliptic_plus/wallet_scores.parquet        one row per wallet
    data/elliptic_plus/wallet_timeline.parquet      risk per time step for flagged wallets
    data/elliptic_plus/wallet_explanations.parquet  per-feature contributions for every
                                                    flagged wallet
    data/elliptic_plus/wallet_tx_edges.parquet      transactions each flagged wallet sent
                                                    or received (from AddrTx / TxAddr)

Run (from the project root, after train_elliptic_plus.py wallets):
    python scripts/score_wallets.py
"""

import joblib
import numpy as np
import pandas as pd

from train_elliptic import MODEL_DIR, TRAIN_LAST_STEP, VAL_LAST_STEP
from train_elliptic_plus import DATA_DIR, TIME_COLUMN

MODEL_FILE = MODEL_DIR / "elliptic_plus_wallets_rf_model.joblib"
SCORES_FILE = DATA_DIR / "wallet_scores.parquet"
TIMELINE_FILE = DATA_DIR / "wallet_timeline.parquet"
EXPLANATIONS_FILE = DATA_DIR / "wallet_explanations.parquet"
WALLET_TX_FILE = DATA_DIR / "wallet_tx_edges.parquet"
TX_SCORES_FILE = DATA_DIR / "tx_scores.parquet"
ADDRTX_FILE = DATA_DIR / "AddrTx_edgelist.csv"
TXADDR_FILE = DATA_DIR / "TxAddr_edgelist.csv"
N_TOP_FEATURES = 8
FLAG_THRESHOLD = 0.5
# Descriptive columns kept for display even if they are not top features
INFO_COLUMNS = [
    "total_txs", "transacted_w_address_total", "btc_transacted_total",
    "num_timesteps_appeared_in", "first_block_appeared_in", "last_block_appeared_in",
    "lifetime_in_blocks",
]
CLASS_NAMES = {1: "illicit", 2: "licit", 3: "unknown"}


def tree_contributions(model, X):
    """Per-sample feature contributions of a random forest (Saabas method).

    Walks each sample's decision path in every tree and credits the change in
    predicted illicit probability at each split to the feature split on.
    The prediction equals the forest's mean root probability (bias) plus the
    sum of a row's contributions.
    """
    X = np.asarray(X, dtype=np.float32)
    contrib = np.zeros(X.shape)
    for est in model.estimators_:
        tree = est.tree_
        value = tree.value[:, 0, :]
        prob = value[:, 1] / value.sum(axis=1)
        parent = np.full(tree.node_count, -1)
        internal = tree.children_left != -1
        parent[tree.children_left[internal]] = np.flatnonzero(internal)
        parent[tree.children_right[internal]] = np.flatnonzero(internal)

        # Every (sample, node) pair on a decision path, excluding the root
        path = est.decision_path(X)
        rows = np.repeat(np.arange(X.shape[0]), np.diff(path.indptr))
        nodes = path.indices
        keep = nodes != 0
        rows, nodes = rows[keep], nodes[keep]
        np.add.at(contrib, (rows, tree.feature[parent[nodes]]), prob[nodes] - prob[parent[nodes]])
    return contrib / len(model.estimators_)


def tx_derived_risk():
    """Max risk of the transactions each wallet sent (AddrTx) or received (TxAddr),
    from the transaction model. Returns None if the transaction scores or edge
    lists are missing. The transaction model is much stronger than the wallet
    model (illicit F1 ~0.79 vs ~0.24), so this ranks wallets far better; see
    scripts/derive_wallet_risk_from_tx.py for the comparison on test wallets."""
    if not (TX_SCORES_FILE.exists() and ADDRTX_FILE.exists() and TXADDR_FILE.exists()):
        return None
    tx = pd.read_parquet(TX_SCORES_FILE, columns=["txId", "risk"]).set_index("txId")["risk"]
    sent = pd.read_csv(ADDRTX_FILE).rename(columns={"input_address": "address"})
    received = pd.read_csv(TXADDR_FILE).rename(columns={"output_address": "address"})
    links = pd.concat([sent[["address", "txId"]], received[["address", "txId"]]], ignore_index=True)
    links["risk"] = links["txId"].map(tx)
    grouped = links.dropna(subset=["risk"]).groupby("address")["risk"]
    # max = flags a wallet exposed to any near-certain illicit transaction;
    # mean = how pervasive that exposure is, used to order severity (max alone
    # saturates at 1.0 for many wallets and cannot separate them).
    return pd.DataFrame({"tx_max": grouped.max(), "tx_mean": grouped.mean()})


def main():
    saved = joblib.load(MODEL_FILE)
    model, features = saved["model"], saved["features"]

    print("Scoring all wallets ...")
    df = pd.read_csv(DATA_DIR / "wallets_features_classes_combined.csv")
    # The wallet model's own probability, kept for reference and for the
    # per-feature explanations below.
    df["rf_risk"] = model.predict_proba(df[features].fillna(0))[:, 1]
    # The risk the app shows is derived from the (much stronger) transaction
    # model: a wallet's risk is the highest risk of any transaction it took part
    # in. It falls back to the wallet model only if the transaction data is absent.
    derived = tx_derived_risk()
    if derived is not None:
        df["tx_max"] = df["address"].map(derived["tx_max"]).fillna(0.0)
        df["tx_mean"] = df["address"].map(derived["tx_mean"]).fillna(0.0)
        df["risk"] = df["tx_max"]  # flag on peak transaction risk
        print("Wallet risk = highest risk of the wallet's transactions; tx_mean kept to order severity")
    else:
        df["risk"] = df["tx_max"] = df["tx_mean"] = df["rf_risk"]
        print("Transaction scores / edge lists not found: wallet risk falls back to the wallet model")

    # Tag each wallet by the period it first appears in (labelled wallets only),
    # matching the temporal split in train_elliptic_plus: train 1..TRAIN_LAST_STEP,
    # validation (used to pick the alert threshold) up to VAL_LAST_STEP, test after.
    # Unlabelled wallets stay 'unknown'. A wallet belongs wholly to one split.
    labelled = df[df["class"] != 3]
    first_step = labelled.groupby("address")[TIME_COLUMN].min()
    bands = pd.cut(first_step, bins=[-1, TRAIN_LAST_STEP, VAL_LAST_STEP, float("inf")],
                   labels=["train", "val", "test"])
    split_of = pd.Series(bands.astype(str), index=first_step.index)
    df["split"] = df["address"].map(split_of).fillna("unknown")

    top_features = (
        pd.Series(model.feature_importances_, index=features)
        .sort_values(ascending=False).head(N_TOP_FEATURES).index.tolist()
    )
    display_cols = top_features + [c for c in INFO_COLUMNS if c not in top_features]

    # One row per wallet (its features are constant across its time steps)
    best = df.sort_values("risk", ascending=False).drop_duplicates("address")
    out = best[["address", "Time step", "class", "risk", "rf_risk", "tx_mean", "split"] + display_cols].rename(
        columns={"Time step": "time_step"}
    )
    out["class"] = out["class"].map(CLASS_NAMES)
    out.to_parquet(SCORES_FILE, index=False, compression="zstd")
    print(f"Scored {len(out)} wallets -> {SCORES_FILE}")
    print(out.groupby("split")["risk"].describe())

    # Risk per time step for flagged wallets that the model did not train on
    flagged = out[(out["risk"] >= FLAG_THRESHOLD) & (out["split"] != "train")]["address"]
    timeline = df[df["address"].isin(flagged)][["address", "Time step", "risk"]]
    timeline.rename(columns={"Time step": "time_step"}).to_parquet(TIMELINE_FILE, index=False, compression="zstd")
    print(f"Timeline for {flagged.size} flagged wallets -> {TIMELINE_FILE}")

    # The wallet model's per-feature view of each flagged wallet (behavioural
    # context; the displayed risk itself comes from the wallet's transactions).
    explained = best[best["address"].isin(flagged)]
    contrib = tree_contributions(model, explained[features].fillna(0))
    explanations = pd.DataFrame(contrib, columns=features, index=explained["address"])
    explanations = explanations.reset_index().melt(
        id_vars="address", var_name="feature", value_name="contribution"
    )
    explanations.to_parquet(EXPLANATIONS_FILE, index=False, compression="zstd")
    bias = np.mean([est.tree_.value[0, 0, 1] / est.tree_.value[0, 0].sum() for est in model.estimators_])
    gap = np.abs(bias + contrib.sum(axis=1) - explained["rf_risk"].to_numpy()).max() if len(explained) else 0.0
    print(f"Explanations for {len(explained)} wallets -> {EXPLANATIONS_FILE} "
          f"(max reconstruction error {gap:.2e})")

    # Which transactions each flagged wallet took part in
    edges = wallet_tx_edges(set(flagged))
    if edges is not None:
        edges.to_parquet(WALLET_TX_FILE, index=False, compression="zstd")
        print(f"{len(edges)} wallet-transaction links for flagged wallets -> {WALLET_TX_FILE}")


def wallet_tx_edges(addresses):
    """Sent / received links between the given wallets and transactions.

    Returns None when the Elliptic++ AddrTx / TxAddr edge lists are missing.
    """
    sources = {
        "sent": (DATA_DIR / "AddrTx_edgelist.csv", "input_address"),
        "received": (DATA_DIR / "TxAddr_edgelist.csv", "output_address"),
    }
    if not all(path.exists() for path, _ in sources.values()):
        print("AddrTx / TxAddr edge lists not found: skipping wallet-transaction links")
        return None
    parts = []
    for direction, (path, column) in sources.items():
        links = pd.read_csv(path).rename(columns={column: "address"})
        parts.append(links[links["address"].isin(addresses)][["address", "txId"]].assign(direction=direction))
    return pd.concat(parts, ignore_index=True).drop_duplicates()


if __name__ == "__main__":
    main()
