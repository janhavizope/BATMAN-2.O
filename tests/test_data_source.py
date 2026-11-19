"""
test_data_source.py
Tests for src/data/elliptic_source.py (the Streamlit data layer) on small
score-file fixtures.
"""

import pandas as pd
import pytest

from src.data import data_manager, elliptic_source


def wallet_scores():
    rows = []
    # (split, class, risk) for 12 wallets
    spec = [
        ("train", "illicit", 0.99), ("train", "licit", 0.10),
        ("test", "illicit", 0.97), ("test", "illicit", 0.90), ("test", "illicit", 0.40),
        ("test", "licit", 0.60), ("test", "licit", 0.05), ("test", "licit", 0.02),
        ("unknown", "unknown", 0.95), ("unknown", "unknown", 0.80),
        ("unknown", "unknown", 0.55), ("unknown", "unknown", 0.01),
    ]
    for i, (split, cls, risk) in enumerate(spec):
        rows.append({
            "address": f"wallet{i:02d}", "time_step": 10 + i, "class": cls, "risk": risk, "split": split,
            "total_txs": 5 + i, "transacted_w_address_total": 3, "btc_transacted_total": 1.5,
            "num_timesteps_appeared_in": 2, "first_block_appeared_in": 1000, "last_block_appeared_in": 1100,
            "lifetime_in_blocks": 100,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def scores(tmp_path, monkeypatch):
    wallets = wallet_scores()
    wallets.to_parquet(tmp_path / "wallet_scores.parquet")
    flagged = wallets[wallets["risk"] >= 0.5]
    pd.DataFrame({
        "address": flagged["address"].repeat(3).to_numpy(),
        "time_step": [1, 2, 3] * len(flagged),
        "risk": flagged["risk"].repeat(3).to_numpy(),
    }).to_parquet(tmp_path / "wallet_timeline.parquet")
    pd.DataFrame({
        "address": flagged["address"].repeat(2).to_numpy(),
        "feature": ["total_txs", "btc_transacted_total"] * len(flagged),
        "contribution": [0.3, 0.1] * len(flagged),
    }).to_parquet(tmp_path / "wallet_explanations.parquet")
    txs = pd.DataFrame({
        "txId": [1, 2, 3, 4, 5, 6], "time_step": [1, 2, 40, 41, 42, 43],
        "class": ["licit", "illicit", "illicit", "licit", "unknown", "unknown"],
        "risk": [0.1, 0.9, 0.95, 0.2, 0.85, 0.3],
        "split": ["train", "train", "test", "test", "unknown", "unknown"],
        "total_BTC": 1.0, "fees": 0.001, "size": 200, "in_txs_degree": 1, "out_txs_degree": 1,
        "num_input_addresses": 1, "num_output_addresses": 2,
    })
    txs.to_parquet(tmp_path / "tx_scores.parquet")
    pd.DataFrame({"source": [3, 3, 5], "target": [4, 6, 6]}).to_parquet(tmp_path / "tx_edges.parquet")

    # WALLET_TX_EDGES is optional and not written here: tests that need it add it
    for name in ("WALLET_SCORES", "WALLET_TIMELINE", "WALLET_EXPLANATIONS", "TX_SCORES", "TX_EDGES", "WALLET_TX_EDGES"):
        path = getattr(elliptic_source, name)
        monkeypatch.setattr(elliptic_source, name, tmp_path / path.name)
    monkeypatch.setattr(elliptic_source, "_CACHE", {})
    monkeypatch.setattr(elliptic_source, "WALLET_ALERT_BUDGET", 3)
    monkeypatch.setattr(elliptic_source, "TX_ALERT_BUDGET", 10)
    return wallets


def test_is_available(scores, tmp_path, monkeypatch):
    assert elliptic_source.is_available()
    monkeypatch.setattr(elliptic_source, "TX_EDGES", tmp_path / "missing.parquet")
    assert not elliptic_source.is_available()


def test_choose_threshold_respects_budget(scores):
    result = elliptic_source.choose_threshold(scores, budget=3)
    pool = scores[scores["split"] != "train"]
    assert result["flagged"] == (pool["risk"] >= result["threshold"]).sum() == 3
    # top 3 non-training wallets are 0.97, 0.95 and 0.90
    assert result["threshold"] == pytest.approx(0.90)
    assert result["precision"] == 1.0
    assert result["recall"] == pytest.approx(2 / 3)


def test_choose_threshold_uses_best_f1_when_under_budget(scores):
    result = elliptic_source.choose_threshold(scores, budget=1_000)
    # Test rows: at 0.40 precision 3/4, recall 1 (F1 0.86) beats 0.90's 1 and 2/3 (F1 0.80)
    assert result["threshold"] == pytest.approx(0.40)
    assert result["recall"] == 1.0


def test_choose_threshold_floor(scores):
    assert elliptic_source.choose_threshold(scores, budget=1_000, floor=0.96)["threshold"] == 0.96


def test_training_wallets_are_never_alerts(scores):
    alerts = elliptic_source.get_alerts_df()
    assert "wallet00" not in set(alerts["Entity ID"])  # train split, risk 0.99
    assert list(alerts["Entity ID"]) == ["wallet02", "wallet08", "wallet03"]
    assert list(alerts["Rank"]) == [1, 2, 3]
    assert alerts["Main Reason"].iloc[0] == "Top factor: total txs"


def test_overview_counts(scores):
    stats = elliptic_source.get_overview_stats()
    assert stats["total_wallets"] == 12
    assert stats["total_transactions"] == 6
    assert stats["anomalous_entities"] == 3


def test_transactions_hide_training_rows(scores):
    txs = elliptic_source.get_transactions_df()
    assert set(txs["Transaction ID"]) == {"3", "4", "5", "6"}
    assert txs.set_index("Transaction ID").loc["3", "Destination"] == "4, 6"
    assert elliptic_source.get_tx_detail("999") is None
    assert elliptic_source.get_tx_detail("3")["flag"] == "Suspicious"


def test_entities_and_explanations(scores):
    entities = elliptic_source.get_entities()
    assert set(entities) == {"wallet02", "wallet08", "wallet03"}
    entity = entities["wallet08"]
    assert entity["risk_score"] == 95
    assert "no ground-truth label" in entity["explanation"]
    assert len(entity["timeline"]) == 3
    explained = elliptic_source.get_explainability_data()
    assert explained["wallet02"]["top_features"][0]["feature"] == "total_txs"


def test_graph_data(scores):
    nodes, edges, stats, centrality = elliptic_source.get_graph_data()
    assert stats["total_edges"] == len(edges) == 3
    assert {n["id"] for n in nodes} >= {"3", "4", "5", "6"}
    assert not centrality.empty


def test_data_manager_delegates_to_scores(scores):
    assert data_manager.is_data_available()
    assert data_manager.get_overview_stats() == elliptic_source.get_overview_stats()
    assert data_manager.get_alerts()[0]["Entity ID"] == "wallet02"


def test_wallet_transaction_links(scores, tmp_path, monkeypatch):
    """With AddrTx / TxAddr links, entities list their transactions and the
    network graph is built around the alerted wallets."""
    pd.DataFrame({
        "address": ["wallet02", "wallet02", "wallet08"],
        "txId": [3, 4, 4],
        "direction": ["sent", "received", "sent"],
    }).to_parquet(tmp_path / "wallet_tx_edges.parquet")
    monkeypatch.setattr(elliptic_source, "WALLET_TX_EDGES", tmp_path / "wallet_tx_edges.parquet")
    monkeypatch.setattr(elliptic_source, "_CACHE", {})

    transactions = elliptic_source.get_entities()["wallet02"]["transactions"]
    assert {t["Transaction ID"] for t in transactions} == {"3", "4"}

    nodes, edges, stats, _ = elliptic_source.get_graph_data()
    wallets = {n["id"] for n in nodes if n["type"] == "Wallet"}
    assert wallets == {"wallet02", "wallet08", "wallet03"}
    relations = {(e["source"], e["target"], e["relation"]) for e in edges}
    assert ("wallet02", "3", "sent") in relations
    assert ("4", "wallet02", "received") in relations
    assert ("3", "4", "funds") in relations  # transaction flow between linked transactions


def test_desktop_uses_the_shared_data_layer(scores):
    from src.desktop_gui.data.data_manager import DataManager
    dm = DataManager()
    dm.load_from_backend()
    assert dm.get_overview().anomalous_entities == elliptic_source.get_overview_stats()["anomalous_entities"]
    assert [a.entity_id for a in dm.get_alerts()] == ["wallet02", "wallet08", "wallet03"]
    assert dm.get_entity("wallet08").risk_score == 95
    assert {t.tx_id for t in dm.get_transactions()} == {"3", "4", "5", "6"}
