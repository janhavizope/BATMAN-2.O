"""
test_live.py
Tests for the live monitoring modules on fixtures (no network):
- mempool_feed._tx_to_row maps an Esplora transaction to the BATMAN schema.
- live_risk scores the forensic signals and explains each score.
- analyze_frame aggregates transaction risk to wallets.
- LiveStore records cycles, tracks peak risk, and prunes.
- the watchlist loader is tolerant of a missing/bad file.
"""

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.core.backend.ingestion.ingestor import TransactionIngestor
from src.core.backend.live import live_risk, live_store, mempool_feed, watchlist
from src.core.backend.validation.validator import TransactionValidator

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


# --- feed mapping ------------------------------------------------------

def esplora_tx(txid="a" * 64):
    return {
        "txid": txid, "fee": 2_000, "status": {"confirmed": False},
        "vin": [
            {"prevout": {"scriptpubkey_address": "bc1qsender1", "value": 100_000}},
            {"prevout": {"scriptpubkey_address": "bc1qsender2", "value": 50_000}},
        ],
        "vout": [
            {"scriptpubkey_address": "bc1qreceiver", "value": 148_000},
            {"scriptpubkey_address": None, "value": 0},  # OP_RETURN, no address
        ],
    }


def test_tx_to_row_maps_schema_and_skips_addressless():
    row = mempool_feed._tx_to_row(esplora_tx(), NOW)
    assert row["input_addresses"] == "bc1qsender1|bc1qsender2"
    assert row["output_addresses"] == "bc1qreceiver"  # OP_RETURN skipped
    assert row["fee"] == pytest.approx(2_000 / mempool_feed.SATS_PER_BTC)


def test_tx_to_row_drops_coinbase():
    coinbase = {"txid": "b" * 64, "status": {"confirmed": False}, "vin": [{"is_coinbase": True}],
                "vout": [{"scriptpubkey_address": "bc1qminer", "value": 625_000_000}]}
    assert mempool_feed._tx_to_row(coinbase, NOW) is None


def test_mapped_row_passes_validator():
    frame = TransactionIngestor().normalize_columns(pd.DataFrame([mempool_feed._tx_to_row(esplora_tx(), NOW)]))
    _, report = TransactionValidator().validate(frame)
    assert report["valid_records"] == 1 and report["invalid_records"] == 0


# --- forensic signals --------------------------------------------------

def view(ins, outs, in_amt=None, out_amt=None, txid="a" * 64):
    return live_risk.TxView(
        txid=txid, input_addresses=ins, output_addresses=outs,
        input_amounts=in_amt if in_amt is not None else [1.0] * len(ins),
        output_amounts=out_amt if out_amt is not None else [1.0] * len(outs),
    )


def test_ordinary_payment_is_low():
    risk = live_risk.analyze_transaction(view(["a"], ["b", "a"], [1.0], [0.6, 0.3999]))
    assert risk.risk_level == "LOW"


def test_equal_outputs_flag_mixer():
    outs = [f"o{i}" for i in range(12)]
    risk = live_risk.analyze_transaction(view(["a"], outs, [12.0], [1.0] * 12))
    assert any("CoinJoin" in r or "mixer" in r for r in risk.reasons)
    assert risk.risk_level in ("MEDIUM", "HIGH")


def test_high_fan_in_consolidation():
    ins = [f"i{i}" for i in range(60)]
    risk = live_risk.analyze_transaction(view(ins, ["sweep"], [1.0] * 60, [59.9]))
    assert any("fan-in" in r for r in risk.reasons)


def test_peeling_chain_detected():
    risk = live_risk.analyze_transaction(view(["a"], ["pay", "change"], [10.0], [0.05, 9.94]))
    assert any("peeling" in r for r in risk.reasons)


def test_watchlist_hit_is_critical():
    risk = live_risk.analyze_transaction(view(["bad"], ["b"], [1.0], [0.99]), watchlist={"bad"})
    assert risk.risk_level == "CRITICAL" and risk.risk_score == 100.0


def test_no_reason_without_contribution():
    # Every reported reason must correspond to a positive component.
    outs = [f"o{i}" for i in range(12)]
    risk = live_risk.analyze_transaction(view(["a"], outs, [12.0], [1.0] * 12))
    triggered = {name for name, pts in risk.components.items() if pts > 0}
    for signal in live_risk.SIGNALS:
        if signal.reason in risk.reasons:
            assert signal.name in triggered


# --- frame aggregation + store ----------------------------------------

def clean_frame():
    """Two validated transactions: one mixer-like, one ordinary."""
    mixer = {"txid": "a" * 64, "timestamp": NOW.isoformat(),
             "input_addresses[]": ["src"], "output_addresses[]": [f"o{i}" for i in range(12)],
             "input_amounts[]": [12.0], "output_amounts[]": [1.0] * 12, "fee": 0.0, "block_height": 0}
    ordinary = {"txid": "c" * 64, "timestamp": NOW.isoformat(),
                "input_addresses[]": ["p"], "output_addresses[]": ["q", "p"],
                "input_amounts[]": [1.0], "output_amounts[]": [0.6, 0.3999], "fee": 0.0001, "block_height": 0}
    return pd.DataFrame([mixer, ordinary])


def test_analyze_frame_aggregates_to_wallets():
    tx_risk, wallet_risk = live_risk.analyze_frame(clean_frame())
    assert len(tx_risk) == 2
    assert wallet_risk.index.name == "wallet_address"
    # The mixer source wallet carries the mixer transaction's risk.
    assert wallet_risk.loc["src", "risk_score"] >= 40


def test_store_records_and_breaks_down(tmp_path):
    store = live_store.LiveStore(tmp_path / "live.sqlite")
    tx_risk, wallet_risk = live_risk.analyze_frame(clean_frame())
    store.record_cycle(tx_risk, wallet_risk, tx_fetched=2, tx_valid=2)
    summary = store.summary()
    assert summary["total_transactions"] == 2
    assert summary["cycles"] == 1
    breakdown = store.level_breakdown()
    assert sum(breakdown.values()) == 2


def test_store_tracks_peak_and_prunes(tmp_path):
    store = live_store.LiveStore(tmp_path / "live.sqlite")
    high = pd.DataFrame({"risk_score": 80.0, "risk_level": "HIGH", "top_reasons": [["r"]],
                         "total_tx_count": 1}, index=pd.Index(["w"], name="wallet_address"))
    low = pd.DataFrame({"risk_score": 10.0, "risk_level": "LOW", "top_reasons": [["r"]],
                        "total_tx_count": 1}, index=pd.Index(["w"], name="wallet_address"))
    tx = lambda t: pd.DataFrame([{"txid": t, "timestamp": "", "n_inputs": 1, "n_outputs": 1,
                                  "total_btc": 1.0, "fee_btc": 0.0, "block_height": 0,
                                  "risk_score": 80.0, "risk_level": "HIGH", "reasons": ["r"]}])
    store.record_cycle(tx("a" * 64), high, 1, 1)
    store.record_cycle(tx("b" * 64), low, 1, 1)
    row = store.wallets().iloc[0]
    assert row["times_scored"] == 2
    assert row["risk_score"] == pytest.approx(10.0)       # latest
    assert row["max_risk_score"] == pytest.approx(80.0)   # peak kept
    store.prune(keep_transactions=1, keep_wallets=1)
    assert len(store.transactions()) == 1


# --- watchlist loader --------------------------------------------------

def test_watchlist_missing_file_is_empty(tmp_path):
    addresses, reasons = watchlist.load_watchlist(tmp_path / "none.json")
    assert addresses == set() and reasons == {}


def test_watchlist_parses_strings_and_objects(tmp_path):
    path = tmp_path / "wl.json"
    path.write_text(json.dumps(["bc1qaaa", {"address": "bc1qbbb", "reason": "OFAC"}]))
    addresses, reasons = watchlist.load_watchlist(path)
    assert addresses == {"bc1qaaa", "bc1qbbb"}
    assert reasons["bc1qbbb"] == "OFAC"


def test_watchlist_bad_json_is_empty(tmp_path):
    path = tmp_path / "wl.json"
    path.write_text("{not json")
    assert watchlist.load_watchlist(path) == (set(), {})


# --- live-computable supervised model ----------------------------------

def test_supervised_live_model_scores_in_range():
    """The live supervised model returns a probability per transaction, or None
    (when the committed scores are not available, e.g. Git LFS not pulled)."""
    from src.core.backend.live import live_model

    frame = TransactionIngestor().normalize_columns(pd.DataFrame([
        {"txid": "a" * 64, "timestamp": "2024-01-01T00:00:00", "input_addresses": "x",
         "output_addresses": "y|x", "input_amounts": "1.0", "output_amounts": "0.6|0.3999",
         "fee": 0.0002}]))
    clean, _ = TransactionValidator().validate(frame)
    risk = live_model.supervised_tx_risk(clean)
    if risk is None:
        pytest.skip("live supervised model unavailable (tx_scores not pulled)")
    assert len(risk) == 1
    assert 0.0 <= float(risk.iloc[0]) <= 1.0
    assert live_model.risk_level(0.9) == "CRITICAL"
    assert live_model.risk_level(0.0) == "LOW"
