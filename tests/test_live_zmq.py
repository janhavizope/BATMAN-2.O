"""
test_live_zmq.py
Tests for the full-node ZMQ feed path, without a live node:
- core_tx_to_row maps a Bitcoin Core getrawtransaction(verbosity=2) object to
  the BATMAN schema, across address-field spellings, and the mapped row passes
  the validator and the forensic analysis.
- the JSON-RPC client builds cookie/basic auth and parses results and errors.
- the persistent listener: sequence-gap accounting, the bounded queue,
  draining into rows, monitor-cycle stats, and (with a real local ZMQ
  publisher standing in for bitcoind) that transactions announced between
  cycles are kept rather than lost.
"""

import struct
import time
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.core.backend.ingestion.ingestor import TransactionIngestor
from src.core.backend.live import live_risk, monitor, node_rpc, zmq_feed
from src.core.backend.live.live_store import LiveStore
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


# --- Persistent listener ---------------------------------------------

def message(txid_byte: int, sequence: int, topic: bytes = b"hashtx"):
    """A bitcoind ZMQ message: [topic, 32-byte txid, uint32 LE sequence]."""
    return [topic, bytes([txid_byte]) * 32, struct.pack("<I", sequence)]


def test_sequence_tracker_counts_gaps_only():
    tracker = zmq_feed.SequenceTracker()
    assert [tracker.record(s) for s in (5, 6, 7)] == [0, 0, 0]  # first has no baseline
    assert tracker.record(10) == 2                               # 8 and 9 missed
    assert tracker.missed == 2


def test_sequence_tracker_wraparound_and_restart():
    tracker = zmq_feed.SequenceTracker()
    tracker.record(2**32 - 1)
    assert tracker.record(0) == 0          # uint32 wraps: 0 follows 4294967295
    tracker.record(500_000)                # (a real gap of 499,999 is counted)
    assert tracker.record(0) == 0          # big backwards jump = node restarted
    assert tracker.restarts == 1
    assert tracker.missed == 499_999


def test_listener_queue_order_overflow_and_topic_filter():
    listener = zmq_feed.ZMQListener("tcp://127.0.0.1:1", queue_size=3)
    for i in range(5):
        listener._handle(message(i, i))
    listener._handle(message(9, 5, topic=b"hashblock"))  # other topics ignored
    stats = listener.stats()
    assert stats["received"] == 5 and stats["queued"] == 3 and stats["overflow"] == 2
    assert stats["missed"] == 0
    # oldest two were dropped; the rest come out oldest first
    assert listener.drain() == [bytes([i]).hex() * 32 for i in (2, 3, 4)]
    assert listener.drain() == []


class FakeRPC:
    """Stands in for BitcoinRPC: knows some txids, raises for the rest."""

    def __init__(self, known):
        self.known = known

    def get_transaction_verbose(self, txid):
        if txid not in self.known:
            raise node_rpc.NodeRPCError("No such mempool or blockchain transaction")
        tx = core_tx()
        tx["txid"] = txid
        return tx


def test_fetch_from_listener_resolves_counts_and_keeps_remainder():
    listener = zmq_feed.ZMQListener("tcp://127.0.0.1:1")
    for i in range(4):
        listener._handle(message(i, i))
    txids = [bytes([i]).hex() * 32 for i in range(4)]
    rpc = FakeRPC(known={txids[0], txids[2]})

    frame = zmq_feed.fetch_from_listener(rpc, listener, max_tx=3)
    assert list(frame["txid"]) == [txids[0], txids[2]]  # txids[1] unresolved, skipped
    assert listener.stats()["unresolved"] == 1
    assert listener.drain() == [txids[3]]               # beyond max_tx: still queued


def test_fetch_from_listener_raises_when_listener_failed():
    listener = zmq_feed.ZMQListener("tcp://127.0.0.1:1")
    listener.error = "ZMQ listener stopped: boom"
    with pytest.raises(zmq_feed.ZMQFeedError):
        zmq_feed.fetch_from_listener(FakeRPC(set()), listener)


def test_run_cycle_processes_whole_queue_and_reports_stats(tmp_path):
    listener = zmq_feed.ZMQListener("tcp://127.0.0.1:1")
    for i in range(60):  # more than the API's per-cycle default of 40
        listener._handle(message(i, i))
    txids = {bytes([i]).hex() * 32 for i in range(60)}
    fetch = monitor.ZMQFetch(FakeRPC(known=txids), listener)

    result = monitor.run_cycle(LiveStore(tmp_path / "live.sqlite"), max_tx=None,
                               watchlist=set(), fetch_fn=fetch)
    assert result.error is None
    assert result.fetched == 60
    assert result.feed_stats["queued"] == 0 and result.feed_stats["received"] == 60
    assert "60 announced" in monitor.format_feed_stats(result.feed_stats)
    assert monitor.format_feed_stats(None) == ""


def test_listener_keeps_transactions_announced_between_cycles():
    """End to end over a real ZMQ socket: nothing published while the monitor
    is "asleep" between cycles is lost, and a skipped sequence number is counted."""
    zmq = pytest.importorskip("zmq")
    publisher = zmq.Context.instance().socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    listener = zmq_feed.ZMQListener(f"tcp://127.0.0.1:{port}").start()
    try:
        def wait_until(condition, timeout=10.0):
            deadline = time.time() + timeout
            while time.time() < deadline:
                if condition():
                    return True
                time.sleep(0.02)
            return False

        # PUB/SUB drops messages until the subscription is in place: warm up first.
        sequence = 0
        while listener.stats()["received"] == 0:
            publisher.send_multipart(message(0xEE, sequence))
            sequence += 1
            time.sleep(0.02)
            assert sequence < 500, "listener never received the warm-up messages"
        assert wait_until(lambda: listener.stats()["received"] >= 1)
        time.sleep(0.1)
        listener.drain()
        received_before = listener.stats()["received"]
        missed_before = listener.stats()["missed"]

        # Cycle 1 drains; then 30 more are announced while "the monitor sleeps",
        # with one sequence number skipped (a lost message).
        for i in range(30):
            if i == 10:
                sequence += 1  # bitcoind sent one we never got
            publisher.send_multipart(message(i, sequence))
            sequence += 1
        assert wait_until(lambda: listener.stats()["received"] >= received_before + 30)

        # Cycle 2: every one of those 30 is still queued.
        assert listener.drain() == [bytes([i]).hex() * 32 for i in range(30)]
        assert listener.stats()["missed"] - missed_before == 1
        assert listener.running
    finally:
        listener.stop()
        publisher.close(linger=0)
    assert not listener.running


def test_get_listener_is_shared(monkeypatch):
    started = []
    monkeypatch.setattr(zmq_feed, "_LISTENERS", {})
    monkeypatch.setattr(zmq_feed.ZMQListener, "start",
                        lambda self, timeout=5.0: started.append(self) or self)
    monkeypatch.setattr(zmq_feed.ZMQListener, "running", property(lambda self: self in started))
    first = zmq_feed.get_listener("tcp://127.0.0.1:28333")
    second = zmq_feed.get_listener("tcp://127.0.0.1:28333")
    assert first is second and len(started) == 1
