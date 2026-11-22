"""
test_live_zmq.py
Tests for the full-node ZMQ feed path, without pyzmq or a live node:
- core_tx_to_row maps a Bitcoin Core getrawtransaction(verbosity=2) object to
  the BATMAN schema, across address-field spellings, and the mapped row passes
  the validator and the forensic analysis.
- the JSON-RPC client builds cookie/basic auth and parses results and errors.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.core.backend.ingestion.ingestor import TransactionIngestor
from src.core.backend.live import live_risk, node_rpc, zmq_feed
from src.core.backend.validation.validator import TransactionValidator

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def core_tx(address_key="address"):
    """A Core verbosity-2 transaction: two inputs, one real + one OP_RETURN output."""
    def spk(addr):
        return {address_key: addr} if addr else {"type": "nulldata"}
    return {
        "txid": "a" * 64, "fee": 0.00002,
        "vin": [
            {"prevout": {"value": 0.001, "scriptPubKey": spk("bc1qsender1")}},
            {"prevout": {"value": 0.0005, "scriptPubKey": spk("bc1qsender2")}},
        ],
        "vout": [
            {"value": 0.00148, "scriptPubKey": spk("bc1qreceiver")},
            {"value": 0.0, "scriptPubKey": spk(None)},  # OP_RETURN
        ],
    }


def test_core_mapping_v22_address():
    row = zmq_feed.core_tx_to_row(core_tx("address"), NOW)
    assert row["input_addresses"] == "bc1qsender1|bc1qsender2"
    assert row["output_addresses"] == "bc1qreceiver"  # OP_RETURN skipped
    assert row["fee"] == pytest.approx(0.00002)


def test_core_mapping_legacy_addresses_list():
    """Older Core reports scriptPubKey.addresses as a list."""
    tx = core_tx("addresses")
    # The helper above put a bare string; fix it to a list for the legacy case.
    for vin in tx["vin"]:
        spk = vin["prevout"]["scriptPubKey"]
        spk["addresses"] = [spk.pop("addresses")]
    for vout in tx["vout"]:
        spk = vout["scriptPubKey"]
        if "addresses" in spk:
            spk["addresses"] = [spk.pop("addresses")]
    row = zmq_feed.core_tx_to_row(tx, NOW)
    assert row["input_addresses"] == "bc1qsender1|bc1qsender2"


def test_core_mapping_drops_coinbase():
    coinbase = {"txid": "b" * 64, "vin": [{"coinbase": "03..."}],
                "vout": [{"value": 6.25, "scriptPubKey": {"address": "bc1qminer"}}]}
    assert zmq_feed.core_tx_to_row(coinbase, NOW) is None


def test_core_mapping_derives_fee_when_absent():
    tx = core_tx()
    del tx["fee"]
    row = zmq_feed.core_tx_to_row(tx, NOW)
    # inputs 0.0015 - outputs 0.00148 = 0.00002
    assert row["fee"] == pytest.approx(0.00002, abs=1e-9)


def test_core_mapped_row_validates_and_analyses():
    frame = TransactionIngestor().normalize_columns(pd.DataFrame([zmq_feed.core_tx_to_row(core_tx(), NOW)]))
    clean, report = TransactionValidator().validate(frame)
    assert report["valid_records"] == 1
    tx_risk, wallet_risk = live_risk.analyze_frame(clean)
    assert len(tx_risk) == 1 and not wallet_risk.empty


# --- RPC client (mocked transport) ------------------------------------

def test_rpc_cookie_auth(tmp_path):
    cookie = tmp_path / ".cookie"
    cookie.write_text("__cookie__:secretpass")
    rpc = node_rpc.BitcoinRPC("http://127.0.0.1:8332", cookie_path=str(cookie))
    assert rpc._auth.startswith("Basic ")


def test_rpc_missing_credentials_raises():
    with pytest.raises(node_rpc.NodeRPCError):
        node_rpc.BitcoinRPC("http://127.0.0.1:8332")


def test_rpc_parses_result(monkeypatch):
    rpc = node_rpc.BitcoinRPC("http://127.0.0.1:8332", user="u", password="p")

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"result": {"txid": "abc"}, "error": null}'

    monkeypatch.setattr(node_rpc.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    assert rpc.get_transaction_verbose("abc") == {"txid": "abc"}


def test_rpc_surfaces_node_error(monkeypatch):
    rpc = node_rpc.BitcoinRPC("http://127.0.0.1:8332", user="u", password="p")

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"result": null, "error": {"code": -5, "message": "No such tx"}}'

    monkeypatch.setattr(node_rpc.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    with pytest.raises(node_rpc.NodeRPCError):
        rpc.call("getrawtransaction", ["abc", 2])
