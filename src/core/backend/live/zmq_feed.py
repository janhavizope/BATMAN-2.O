"""
zmq_feed.py
A live transaction feed backed by your own Bitcoin Core full node, over ZMQ.

bitcoind publishes every transaction it accepts on a ZMQ socket. We subscribe
to the `hashtx` topic (each message is a new transaction's id), then ask the
local node over RPC for the full transaction and each input's previous output,
which gives the sending addresses and amounts (resolve_transactions). Each
transaction is mapped to the same BATMAN schema the mempool.space feed and
the Upload & Score page use, so the rest of the pipeline — validation,
forensic analysis, the store, the web page — is unchanged.

This is the firsthand, real-time feed: no public API, no rate limit, every
transaction the moment the node sees it. It needs a running node with ZMQ
enabled (see the README's node setup); it is tested against Bitcoin Core 32.1.
A pruned node without txindex is fine: the coins a mempool transaction spends
are still in the node's chainstate, which gettxout reads.

ZMQ is publish/subscribe: the node does not queue messages for a subscriber
that is not connected. So the subscriber must stay connected the whole time.
ZMQListener keeps one SUB socket open on a background thread and puts every
announced txid into an in-memory queue; each monitor cycle drains that queue
(fetch_from_listener). Nothing announced between cycles is lost. Bitcoin
Core numbers each message per topic, so the listener counts any gap in that
sequence and reports it in stats() instead of losing messages silently.

pyzmq is imported lazily inside connect_subscriber, so the mapping functions
here can be imported and tested without pyzmq or a node present.
"""

from __future__ import annotations

import struct
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.core.backend.live.node_rpc import BitcoinRPC, NodeRPCError


class ZMQFeedError(RuntimeError):
    """Raised when the ZMQ feed cannot be set up or read."""


def _address_of(script_pubkey: Dict[str, Any]) -> Optional[str]:
    """Pull the address from a scriptPubKey, across Core versions.

    Bitcoin Core v22+ uses the singular `address`; older versions use a list
    `addresses`. Returns None for scripts with no address (e.g. OP_RETURN).
    """
    if not isinstance(script_pubkey, dict):
        return None
    if script_pubkey.get("address"):
        return script_pubkey["address"]
    addresses = script_pubkey.get("addresses")
    if isinstance(addresses, list) and addresses:
        return addresses[0]
    return None


def core_tx_to_row(tx: Dict[str, Any], fetched_at: datetime) -> Optional[Dict[str, Any]]:
    """Map a Core transaction with prevouts filled in (see resolve_transactions) to a row.

    Amounts are already in BTC. Inputs/outputs without an address (coinbase
    inputs, OP_RETURN outputs, or a prevout the node could not resolve) are
    skipped so addresses and amounts stay aligned. Returns None when no usable
    input or output address remains.
    """
    in_addresses: List[str] = []
    in_amounts: List[float] = []
    for vin in tx.get("vin", []):
        prevout = vin.get("prevout") or {}
        address = _address_of(prevout.get("scriptPubKey") or {})
        if address is None:
            continue
        in_addresses.append(address)
        in_amounts.append(float(prevout.get("value", 0.0)))

    out_addresses: List[str] = []
    out_amounts: List[float] = []
    for vout in tx.get("vout", []):
        address = _address_of(vout.get("scriptPubKey") or {})
        if address is None:
            continue
        out_addresses.append(address)
        out_amounts.append(float(vout.get("value", 0.0)))

    if not in_addresses or not out_addresses:
        return None

    # resolve_transactions sets `fee` from getmempoolentry. Otherwise derive it
    # from all inputs and outputs, but only when every input's prevout is known:
    # with an input missing, the difference would understate the fee.
    fee = tx.get("fee")
    if fee is None:
        spending = [vin for vin in tx.get("vin", []) if "coinbase" not in vin]
        if spending and all(vin.get("prevout") for vin in spending):
            fee = max(sum(float(vin["prevout"].get("value", 0.0)) for vin in spending)
                      - sum(float(v.get("value", 0.0)) for v in tx.get("vout", [])), 0.0)
        else:
            fee = 0.0

    block_time = tx.get("blocktime")
    timestamp = (datetime.fromtimestamp(block_time, tz=timezone.utc)
                 if block_time else fetched_at)

    return {
        "txid": tx.get("txid"),
        "timestamp": timestamp.isoformat(),
        "input_addresses": "|".join(in_addresses),
        "output_addresses": "|".join(out_addresses),
        "input_amounts": "|".join(f"{a:.8f}" for a in in_amounts),
        "output_amounts": "|".join(f"{a:.8f}" for a in out_amounts),
        "fee": float(fee),
        "block_height": 0,
    }


def connect_subscriber(endpoint: str):
    """Open a ZMQ SUB socket on the node's `hashtx` topic. Lazy-imports pyzmq."""
    try:
        import zmq
    except ImportError as error:
        raise ZMQFeedError(
            "pyzmq is not installed. Run `pip install pyzmq` (it is in requirements.txt)."
        ) from error
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 0)
    socket.setsockopt_string(zmq.SUBSCRIBE, "hashtx")
    try:
        socket.connect(endpoint)
    except Exception as error:  # zmq.ZMQError
        raise ZMQFeedError(f"could not connect to ZMQ endpoint {endpoint}: {error}") from error
    return socket


# Most txids the listener holds between cycles. At a few transactions per
# second this is hours of traffic; past it the oldest are dropped and counted.
DEFAULT_QUEUE_SIZE = 200_000
# How many recent txids are remembered to drop duplicate announcements
# (several blocks' worth of mainnet transactions).
DEFAULT_SEEN_SIZE = 500_000
# A jump in the sequence number larger than this is a node restart (the
# counter starts again from 0), not that many lost messages.
SEQUENCE_RESET_GAP = 1_000_000
POLL_MS = 500


class SequenceTracker:
    """Counts messages missed according to Bitcoin Core's per-topic sequence numbers.

    Every ZMQ message from bitcoind carries a little-endian uint32 sequence
    number that increases by one per message on that topic. A jump means the
    subscriber missed messages in between (for example the socket's buffer
    overflowed). The first message has nothing to compare with, and a large
    backwards jump is a node restart, so neither counts as missed.
    """

    def __init__(self):
        self.last: Optional[int] = None
        self.missed = 0
        self.restarts = 0

    def record(self, sequence: int) -> int:
        """Record one sequence number; return how many messages were missed before it."""
        gap = 0
        if self.last is not None:
            expected = (self.last + 1) % 2**32
            gap = (sequence - expected) % 2**32
            if gap > SEQUENCE_RESET_GAP:
                self.restarts += 1
                gap = 0
            self.missed += gap
        self.last = sequence
        return gap


class ZMQListener:
    """A persistent subscriber to bitcoind's `hashtx` topic on a background thread.

    start() opens one SUB socket and keeps it open; the thread puts every
    announced txid into a bounded in-memory queue. drain() hands queued txids
    to a monitor cycle. Because the socket never closes between cycles,
    transactions announced while the monitor is sleeping are kept, not lost.

    The socket is created, used and closed only on the listener thread, as
    pyzmq sockets are not thread-safe.
    """

    def __init__(self, endpoint: str, queue_size: int = DEFAULT_QUEUE_SIZE,
                 seen_size: int = DEFAULT_SEEN_SIZE):
        self.endpoint = endpoint
        self._queue: deque = deque()
        self._queue_size = queue_size
        # Recently queued txids, to skip the re-announcement when they are mined
        self._seen: set = set()
        self._seen_order: deque = deque()
        self._seen_size = seen_size
        self.duplicates = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sequence = SequenceTracker()
        self.received = 0
        self.overflow = 0
        self.unresolved = 0  # txids the node could not return details for (updated by fetch)
        self.error: Optional[str] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, timeout: float = 5.0) -> "ZMQListener":
        """Start the listener thread (idempotent). Raises ZMQFeedError if it cannot connect."""
        if self.running:
            return self
        self._stop.clear()
        self._ready.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run, name="batman-zmq-listener", daemon=True)
        self._thread.start()
        self._ready.wait(timeout)
        if self.error:
            raise ZMQFeedError(self.error)
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        try:
            socket = connect_subscriber(self.endpoint)
        except ZMQFeedError as error:
            self.error = str(error)
            self._ready.set()
            return
        import zmq

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        self._ready.set()
        try:
            while not self._stop.is_set():
                if not dict(poller.poll(POLL_MS)):
                    continue
                parts = socket.recv_multipart()
                self._handle(parts)
        except Exception as error:  # zmq.ZMQError or a malformed message
            self.error = f"ZMQ listener stopped: {error}"
        finally:
            poller.unregister(socket)
            socket.close(linger=0)

    def _handle(self, parts: List[bytes]) -> None:
        """Queue one [topic, txid, sequence] message from bitcoind."""
        if len(parts) < 2 or parts[0] != b"hashtx":
            return
        txid = parts[1].hex()
        with self._lock:
            self.received += 1
            if len(parts) >= 3 and len(parts[2]) == 4:
                self._sequence.record(struct.unpack("<I", parts[2])[0])
            # bitcoind announces a transaction again when it is mined into a
            # block; one we already queued is not analysed twice.
            if txid in self._seen:
                self.duplicates += 1
                return
            self._seen.add(txid)
            self._seen_order.append(txid)
            if len(self._seen_order) > self._seen_size:
                self._seen.discard(self._seen_order.popleft())
            if len(self._queue) >= self._queue_size:
                self._queue.popleft()
                self.overflow += 1
            self._queue.append(txid)

    def drain(self, max_items: Optional[int] = None) -> List[str]:
        """Remove and return up to `max_items` queued txids, oldest first (all if None)."""
        with self._lock:
            count = len(self._queue) if max_items is None else min(max_items, len(self._queue))
            return [self._queue.popleft() for _ in range(count)]

    def stats(self) -> Dict[str, Any]:
        """Counters for the feed's health: received, queued, missed, overflow, unresolved, restarts."""
        with self._lock:
            return {
                "running": self.running,
                "received": self.received,
                "queued": len(self._queue),
                "missed": self._sequence.missed,
                "overflow": self.overflow,
                "unresolved": self.unresolved,
                "duplicates": self.duplicates,
                "node_restarts": self._sequence.restarts,
                "error": self.error,
            }


_LISTENERS: Dict[str, ZMQListener] = {}
_LISTENERS_LOCK = threading.Lock()


def get_listener(endpoint: str) -> ZMQListener:
    """The one running listener for `endpoint`, started on first use.

    Shared by the CLI loop and the Streamlit page, so a page rerun reuses the
    same connection instead of opening (and missing messages on) a new one.
    """
    with _LISTENERS_LOCK:
        listener = _LISTENERS.get(endpoint)
        if listener is None:
            listener = ZMQListener(endpoint)
            _LISTENERS[endpoint] = listener
    if listener.error or not listener.running:
        listener.start()
    return listener


# Transactions resolved per JSON-RPC batch (each batch is a few HTTP requests).
RESOLVE_BATCH = 500
# How many of the latest blocks to search for a transaction that was mined
# before its cycle ran (6 blocks is about an hour on mainnet).
RECENT_BLOCK_DEPTH = 6


def _prevout_of(output: Dict[str, Any]) -> Dict[str, Any]:
    return {"value": output.get("value", 0.0), "scriptPubKey": output.get("scriptPubKey") or {}}


def _from_recent_blocks(rpc: BitcoinRPC, txids: List[str],
                        depth: int = RECENT_BLOCK_DEPTH) -> List[Dict[str, Any]]:
    """Find transactions mined into one of the last `depth` blocks.

    Without txindex, getrawtransaction finds a confirmed transaction only when
    told its block. A pruned node keeps recent blocks, and for a transaction
    in a block verbosity 2 does include each input's prevout and the fee (from
    the block's undo data). So a transaction mined between being announced
    and being processed is still analysed.
    """
    remaining = list(txids)
    found: List[Dict[str, Any]] = []
    if not remaining:
        return found
    try:
        block = rpc.call("getbestblockhash")
    except NodeRPCError:
        return found
    for _ in range(depth):
        if not remaining or not block:
            break
        results = rpc.batch([("getrawtransaction", [txid, 2, block]) for txid in remaining])
        found += [tx for tx in results if isinstance(tx, dict)]
        remaining = [txid for txid, tx in zip(remaining, results) if not isinstance(tx, dict)]
        try:
            block = rpc.call("getblockheader", [block]).get("previousblockhash")
        except NodeRPCError:
            break
    return found


def resolve_transactions(rpc: BitcoinRPC, txids: List[str]) -> tuple[List[Dict[str, Any]], int]:
    """Fetch transactions with every input's previous output and the fee filled in.

    Bitcoin Core's getrawtransaction verbosity 2 adds `prevout` and `fee` only
    from block undo data, so for a transaction still in the mempool (every
    transaction the live feed sees) it returns neither. Without the prevouts
    there are no sending addresses, and the transaction cannot be analysed.
    So this resolves them itself, in JSON-RPC batches:

    1. getrawtransaction(txid, 1) for each announced transaction.
    2. gettxout(parent, n, include_mempool=False) for each input: the coin it
       spends is still unspent in the node's chainstate (the spend is only in
       the mempool), so this works on a pruned node without txindex.
    3. For inputs whose parent is itself unconfirmed, getrawtransaction on
       the parent (in the mempool, so available without txindex).
    4. getmempoolentry for the fee.

    A transaction no longer in the mempool is looked up in the latest blocks
    (_from_recent_blocks), where verbosity 2 does carry prevouts and the fee.

    Returns (transactions, missing): `missing` counts txids the node no longer
    had (evicted, or mined more than RECENT_BLOCK_DEPTH blocks ago). An input that still cannot be resolved
    is left without a prevout and skipped by core_tx_to_row.
    """
    if not txids:
        return [], 0
    raws = rpc.batch([("getrawtransaction", [txid, 1]) for txid in txids])
    txs = [tx for tx in raws if isinstance(tx, dict)]
    not_in_mempool = [txid for txid, tx in zip(txids, raws) if not isinstance(tx, dict)]
    # Mined since it was announced: look in the latest blocks (kept even when pruned)
    txs += _from_recent_blocks(rpc, not_in_mempool)
    missing = len(txids) - len(txs)

    needs_prevout = [vin for tx in txs for vin in tx.get("vin", [])
                     if "coinbase" not in vin and not vin.get("prevout") and "txid" in vin]
    outputs = rpc.batch([("gettxout", [vin["txid"], vin["vout"], False]) for vin in needs_prevout])
    unresolved_inputs = []
    for vin, output in zip(needs_prevout, outputs):
        if isinstance(output, dict):
            vin["prevout"] = _prevout_of(output)
        else:
            unresolved_inputs.append(vin)

    if unresolved_inputs:
        parent_ids = sorted({vin["txid"] for vin in unresolved_inputs})
        parents = dict(zip(parent_ids, rpc.batch([("getrawtransaction", [p, 1]) for p in parent_ids])))
        for vin in unresolved_inputs:
            parent = parents.get(vin["txid"])
            if isinstance(parent, dict) and vin["vout"] < len(parent.get("vout", [])):
                vin["prevout"] = _prevout_of(parent["vout"][vin["vout"]])

    no_fee = [tx for tx in txs if tx.get("fee") is None]
    entries = rpc.batch([("getmempoolentry", [tx["txid"]]) for tx in no_fee])
    for tx, entry in zip(no_fee, entries):
        if isinstance(entry, dict) and isinstance(entry.get("fees"), dict):
            tx["fee"] = entry["fees"].get("base")
    return txs, missing


def fetch_from_listener(rpc: BitcoinRPC, listener: ZMQListener,
                        max_tx: Optional[int] = None) -> pd.DataFrame:
    """Resolve the txids queued since the last cycle into BATMAN schema rows.

    Takes up to `max_tx` queued txids (all of them if None); the rest stay
    queued for the next cycle. A transaction the node no longer has (already
    mined and pruned, or evicted) is skipped and counted as unresolved.
    Raises ZMQFeedError if the listener thread has stopped.
    """
    if listener.error:
        raise ZMQFeedError(listener.error)
    fetched_at = datetime.now(tz=timezone.utc)
    txids = listener.drain(max_tx)
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(txids), RESOLVE_BATCH):
        try:
            txs, missing = resolve_transactions(rpc, txids[start:start + RESOLVE_BATCH])
        except NodeRPCError as error:
            raise ZMQFeedError(f"could not read transactions from the node: {error}") from error
        listener.unresolved += missing
        for tx in txs:
            row = core_tx_to_row(tx, fetched_at)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    import time

    from src.config import get_node_settings

    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 15
    settings = get_node_settings()
    rpc = BitcoinRPC(url=settings["rpc_url"], user=settings["rpc_user"],
                     password=settings["rpc_password"], cookie_path=settings["rpc_cookie"])
    print(f"Pinging node at {settings['rpc_url']} ...")
    rpc.ping()
    print(f"Connected. Listening on {settings['zmq_endpoint']} for {seconds:.0f}s ...")
    listener = get_listener(settings["zmq_endpoint"])
    time.sleep(seconds)
    frame = fetch_from_listener(rpc, listener)
    print(f"Captured {len(frame)} transactions. Feed stats: {listener.stats()}")
    if not frame.empty:
        with pd.option_context("display.max_colwidth", 20, "display.width", 160):
            print(frame[["txid", "timestamp", "input_addresses", "output_addresses", "fee"]].head(10))
