"""
test_training_scripts.py
Tests for scripts/train_elliptic*.py and scripts/score_*.py on small
synthetic Elliptic / Elliptic++ fixtures (the real datasets are not needed).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import score_transactions  # noqa: E402
import score_wallets  # noqa: E402
import train_elliptic  # noqa: E402
import train_elliptic_plus  # noqa: E402
from train_elliptic import TRAIN_LAST_STEP, VAL_LAST_STEP  # noqa: E402

RNG = np.random.default_rng(0)


def make_wallets(n=120):
    """Wallets active in 1-3 consecutive time steps; illicit ones look different."""
    rows = []
    for i in range(n):
        illicit = i % 4 == 0
        start = int(RNG.integers(1, 48))
        for step in range(start, min(start + int(RNG.integers(1, 4)), 50)):
            signal = 5.0 if illicit else 1.0
            rows.append({
                "address": f"addr{i}",
                "Time step": step,
                "class": 1 if illicit else (3 if i % 7 == 0 else 2),
                "total_txs": signal * 3 + RNG.random(),
                "transacted_w_address_total": signal + RNG.random(),
                "btc_transacted_total": signal * 10 + RNG.random(),
                "num_timesteps_appeared_in": 2,
                "lifetime_in_blocks": signal * 100 + RNG.random(),
                "fees_total": signal * 0.01 + RNG.random() * 0.001,
                "first_block_appeared_in": step * 1000,
                "last_block_appeared_in": step * 1000 + 50,
                "first_sent_block": step * 1000 + 10,
                "first_received_block": step * 1000,
            })
    return pd.DataFrame(rows)


def make_txs(n=400):
    illicit = np.arange(n) % 5 == 0
    signal = np.where(illicit, 5.0, 1.0)
    features = pd.DataFrame({
        "txId": np.arange(n) + 1000,
        "Time step": RNG.integers(1, 50, n),
        "Local_feature_1": signal + RNG.random(n),
        "total_BTC": signal * 2 + RNG.random(n),
        "fees": signal * 0.001 + RNG.random(n) * 0.0001,
        "size": signal * 100 + RNG.random(n),
        "in_txs_degree": RNG.integers(1, 4, n),
        "out_txs_degree": RNG.integers(1, 4, n),
        "num_input_addresses": RNG.integers(1, 4, n),
        "num_output_addresses": RNG.integers(1, 4, n),
    })
    classes = pd.DataFrame({
        "txId": features["txId"],
        "class": np.where(illicit, 1, np.where(np.arange(n) % 9 == 0, 3, 2)),
    })
    edges = pd.DataFrame({"txId1": features["txId"][:-1].to_numpy(), "txId2": features["txId"][1:].to_numpy()})
    return features, classes, edges


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point every script at a temporary data/models/outputs tree."""
    data = tmp_path / "elliptic_plus"
    data.mkdir()
    elliptic = tmp_path / "elliptic"
    elliptic.mkdir()
    models = tmp_path / "models"

    make_wallets().to_csv(data / "wallets_features_classes_combined.csv", index=False)
    features, classes, edges = make_txs()
    features.to_csv(data / "txs_features.csv", index=False)
    classes.to_csv(data / "txs_classes.csv", index=False)
    edges.to_csv(elliptic / "elliptic_txs_edgelist.csv", index=False)

    monkeypatch.setattr(train_elliptic_plus, "DATA_DIR", data)
    monkeypatch.setattr(train_elliptic_plus, "MODEL_DIR", models)
    monkeypatch.setattr(train_elliptic_plus, "EVAL_DIR", tmp_path / "evaluation")
    monkeypatch.setattr(train_elliptic_plus, "GRAPHS_ROOT", tmp_path / "graphs")
    monkeypatch.setattr(score_wallets, "DATA_DIR", data)
    monkeypatch.setattr(score_wallets, "MODEL_FILE", models / "elliptic_plus_wallets_rf_model.joblib")
    monkeypatch.setattr(score_wallets, "SCORES_FILE", data / "wallet_scores.parquet")
    monkeypatch.setattr(score_wallets, "TIMELINE_FILE", data / "wallet_timeline.parquet")
    monkeypatch.setattr(score_wallets, "EXPLANATIONS_FILE", data / "wallet_explanations.parquet")
    monkeypatch.setattr(score_transactions, "DATA_DIR", data)
    monkeypatch.setattr(score_transactions, "MODEL_FILE", models / "elliptic_plus_txs_rf_model.joblib")
    monkeypatch.setattr(score_transactions, "SCORES_FILE", data / "tx_scores.parquet")
    monkeypatch.setattr(score_transactions, "EDGES_SOURCE", elliptic / "elliptic_txs_edgelist.csv")
    monkeypatch.setattr(score_transactions, "EDGES_FILE", data / "tx_edges.parquet")
    return tmp_path


def test_temporal_split_for_transactions():
    X = pd.DataFrame({"f": range(6)})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    steps = pd.Series([1, TRAIN_LAST_STEP, TRAIN_LAST_STEP + 1, 40, 2, 49])
    X_train, X_test, _, _ = train_elliptic_plus.split(X, y, None, steps)
    assert list(X_train["f"]) == [0, 1, 4]   # steps 1, 29, 2 (<= TRAIN_LAST_STEP)
    # step TRAIN_LAST_STEP+1 (f=2) is in the validation period -> neither set
    assert list(X_test["f"]) == [3, 5]       # steps 40, 49 (> VAL_LAST_STEP)


def test_wallet_split_uses_first_appearance():
    X = pd.DataFrame({"f": range(6)})
    y = pd.Series([0, 0, 1, 1, 0, 0])
    groups = pd.Series(["val", "val", "new", "new", "early", "early"])
    steps = pd.Series([30, 36, 35, 40, 5, 20])
    X_train, X_test, _, _ = train_elliptic_plus.split(X, y, groups, steps)
    # "early" (first seen step 5) trains on both its rows; "val" (first seen 30)
    # falls in the validation period, so it is in neither set; "new" (first seen 35) is test
    assert list(X_train["f"]) == [4, 5]
    assert list(X_test["f"]) == [2, 3]
    assert set(groups[X_train.index]).isdisjoint(groups[X_test.index])


def test_time_features_are_not_inputs(workspace):
    X, _, _, _ = train_elliptic_plus.load_wallets()
    assert "Time step" not in X.columns
    for column in train_elliptic_plus.WALLET_TIME_FEATURES:
        assert column not in X.columns
    X, _, _, _ = train_elliptic_plus.load_txs()
    assert "Time step" not in X.columns


def test_f1_by_time_step():
    f1 = train_elliptic.f1_by_time_step(
        pd.Series([1, 1, 0, 1]), np.array([1, 0, 0, 1]), pd.Series([35, 35, 36, 36])
    )
    assert f1 == {35: pytest.approx(2 / 3), 36: 1.0}


def test_train_and_score_wallets(workspace):
    train_elliptic_plus.train("wallets", train_elliptic_plus.load_wallets)
    metrics = json.loads((workspace / "evaluation" / "elliptic_plus_wallets_metrics.json").read_text())
    assert metrics["split"].startswith("temporal")
    assert all(int(step) > TRAIN_LAST_STEP for step in metrics["f1_illicit_by_time_step"])
    assert (workspace / "graphs" / "wallets" / "confusion_matrix.png").exists()

    score_wallets.main()
    scores = pd.read_parquet(workspace / "elliptic_plus" / "wallet_scores.parquet")
    assert scores["address"].is_unique
    assert set(scores["split"]) <= {"train", "val", "test", "unknown"}
    # Wallets first seen after the validation period are tagged as test wallets
    wallets = pd.read_csv(workspace / "elliptic_plus" / "wallets_features_classes_combined.csv")
    labelled = wallets[wallets["class"] != 3]
    first_step = labelled.groupby("address")["Time step"].min()
    late = first_step[first_step > VAL_LAST_STEP].index
    assert (scores.set_index("address").loc[late, "split"] == "test").all()

    explanations = pd.read_parquet(workspace / "elliptic_plus" / "wallet_explanations.parquet")
    assert set(explanations["address"]) <= set(scores.loc[scores["split"] != "train", "address"])


def test_train_and_score_transactions(workspace):
    train_elliptic_plus.train("txs", train_elliptic_plus.load_txs)
    score_transactions.main()
    scores = pd.read_parquet(workspace / "elliptic_plus" / "tx_scores.parquet")
    labelled = scores[scores["class"] != "unknown"]
    assert (labelled.loc[labelled["split"] == "train", "time_step"] <= TRAIN_LAST_STEP).all()
    assert labelled.loc[labelled["split"] == "val", "time_step"].between(TRAIN_LAST_STEP + 1, VAL_LAST_STEP).all()
    assert (labelled.loc[labelled["split"] == "test", "time_step"] > VAL_LAST_STEP).all()
    assert (scores.loc[scores["class"] == "unknown", "split"] == "unknown").all()
    assert scores["risk"].between(0, 1).all()


def test_tree_contributions_reconstruct_the_prediction():
    from sklearn.ensemble import RandomForestClassifier
    X = RNG.random((80, 4))
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    contrib = score_wallets.tree_contributions(model, X)
    bias = np.mean([est.tree_.value[0, 0, 1] / est.tree_.value[0, 0].sum() for est in model.estimators_])
    np.testing.assert_allclose(bias + contrib.sum(axis=1), model.predict_proba(X)[:, 1], atol=1e-6)
