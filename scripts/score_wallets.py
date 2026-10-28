"""
score_wallets.py
----------------
Uses the trained Elliptic++ wallets model to give every wallet in the
dataset a risk score (predicted probability of being illicit), including
the wallets whose class is unknown.

Each wallet appears once per time step; it is scored on every row and keeps
its highest score. Each wallet is tagged with the split it belonged to
during training (train / test / unknown) so that only test and unknown
scores are presented as genuine predictions.

Outputs (read by the Streamlit app):
    data/elliptic_plus/wallet_scores.parquet        one row per wallet
    data/elliptic_plus/wallet_timeline.parquet      risk per time step for flagged wallets
    data/elliptic_plus/wallet_explanations.parquet  per-feature contributions for every
                                                    flagged wallet

Run (from the project root, after train_elliptic_plus.py wallets):
    python scripts/score_wallets.py
"""

import joblib
import numpy as np
import pandas as pd

from train_elliptic import MODEL_DIR
from train_elliptic_plus import DATA_DIR, load_wallets, split

MODEL_FILE = MODEL_DIR / "elliptic_plus_wallets_rf_model.joblib"
SCORES_FILE = DATA_DIR / "wallet_scores.parquet"
TIMELINE_FILE = DATA_DIR / "wallet_timeline.parquet"
EXPLANATIONS_FILE = DATA_DIR / "wallet_explanations.parquet"
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


def main():
    saved = joblib.load(MODEL_FILE)
    model, features = saved["model"], saved["features"]

    # Recreate the training split (same random_state) to tag train/test wallets
    X, y, groups = load_wallets()
    X_train, X_test, _, _ = split(X.fillna(0), y, groups)
    split_of = pd.concat([
        pd.Series("train", index=groups.loc[X_train.index].unique()),
        pd.Series("test", index=groups.loc[X_test.index].unique()),
    ])

    print("Scoring all wallets ...")
    df = pd.read_csv(DATA_DIR / "wallets_features_classes_combined.csv")
    df["risk"] = model.predict_proba(df[features].fillna(0))[:, 1]
    df["split"] = df["address"].map(split_of).fillna("unknown")

    top_features = (
        pd.Series(model.feature_importances_, index=features)
        .sort_values(ascending=False).head(N_TOP_FEATURES).index.tolist()
    )
    display_cols = top_features + [c for c in INFO_COLUMNS if c not in top_features]

    # Keep each wallet's highest-risk time step
    best = df.sort_values("risk", ascending=False).drop_duplicates("address")
    out = best[["address", "Time step", "class", "risk", "split"] + display_cols].rename(
        columns={"Time step": "time_step"}
    )
    out["class"] = out["class"].map(CLASS_NAMES)
    out.to_parquet(SCORES_FILE, index=False)
    print(f"Scored {len(out)} wallets -> {SCORES_FILE}")
    print(out.groupby("split")["risk"].describe())

    # Risk per time step for flagged wallets that the model did not train on
    flagged = out[(out["risk"] >= FLAG_THRESHOLD) & (out["split"] != "train")]["address"]
    timeline = df[df["address"].isin(flagged)][["address", "Time step", "risk"]]
    timeline.rename(columns={"Time step": "time_step"}).to_parquet(TIMELINE_FILE, index=False)
    print(f"Timeline for {flagged.size} flagged wallets -> {TIMELINE_FILE}")

    # Why each flagged wallet was flagged
    explained = best[best["address"].isin(flagged)]
    contrib = tree_contributions(model, explained[features].fillna(0))
    explanations = pd.DataFrame(contrib, columns=features, index=explained["address"])
    explanations = explanations.reset_index().melt(
        id_vars="address", var_name="feature", value_name="contribution"
    )
    explanations.to_parquet(EXPLANATIONS_FILE, index=False)
    bias = np.mean([est.tree_.value[0, 0, 1] / est.tree_.value[0, 0].sum() for est in model.estimators_])
    gap = np.abs(bias + contrib.sum(axis=1) - explained["risk"].to_numpy()).max()
    print(f"Explanations for {len(explained)} wallets -> {EXPLANATIONS_FILE} "
          f"(max reconstruction error {gap:.2e})")


if __name__ == "__main__":
    main()
