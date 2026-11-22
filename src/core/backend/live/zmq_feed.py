"""
zmq_feed.py
A live transaction feed backed by your own Bitcoin Core full node, over ZMQ.

bitcoind publishes every transaction it accepts on a ZMQ socket. We subscribe
to the `hashtx` topic (each message is a new transaction's id), then ask the
local node over RPC for the full transaction (getrawtransaction verbosity 2,
which includes each input's previous output, so we get the sending addresses
and amounts). Each transaction is mapped to the same BATMAN schema the
mempool.space feed and the Upload & Score page use, so the rest of the
pipeline — validation, forensic analysis, the store, the web page — is
unchanged.

This is the firsthand, real-time feed: no public API, no rate limit, every
transaction the moment the node sees it. It needs a running node with ZMQ
enabled (see the README's node setup). A pruned node is fine; its chainstate
still resolves the prevouts of current mempool transactions.

pyzmq is imported lazily inside connect_subscriber, so the mapping functions
here can be imported and tested without pyzmq or a node present.
"""

from __future__ import annotations

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
    """Map a Core getrawtransaction(verbosity=2) object to a BATMAN schema row.

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

    # getrawtransaction reports `fee` for mempool transactions; derive it
    # otherwise from the resolved inputs and outputs when both are complete.
    fee = tx.get("fee")
    if fee is None:
        fee = max(sum(in_amounts) - sum(out_amounts), 0.0)

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


def fetch_zmq_transactions(rpc: BitcoinRPC, endpoint: str, max_tx: int = 40,
                           timeout_ms: int = 10_000) -> pd.DataFrame:
    """Collect up to `max_tx` live transactions from the node's ZMQ feed.

    Waits up to `timeout_ms` for each message; returns what it gathered when
    the batch is full or the feed goes quiet. A transaction the node can no
    longer resolve is skipped, not fatal.
    """
    import zmq  # already importable if connect_subscriber succeeded

    socket = connect_subscriber(endpoint)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    fetched_at = datetime.now(tz=timezone.utc)
    rows: List[Dict[str, Any]] = []
    try:
        while len(rows) < max_tx:
            if not dict(poller.poll(timeout_ms)):
                break  # no new transaction within the window
            _topic, body, _seq = socket.recv_multipart()
            txid = body.hex()
            try:
                tx = rpc.get_transaction_verbose(txid)
            except NodeRPCError:
                continue
            row = core_tx_to_row(tx, fetched_at)
            if row is not None:
                rows.append(row)
    finally:
        poller.unregister(socket)
        socket.close(linger=0)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    from src.config import get_node_settings

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    settings = get_node_settings()
    rpc = BitcoinRPC(url=settings["rpc_url"], user=settings["rpc_user"],
                     password=settings["rpc_password"], cookie_path=settings["rpc_cookie"])
    print(f"Pinging node at {settings['rpc_url']} ...")
    rpc.ping()
    print(f"Connected. Waiting for {count} live transactions on {settings['zmq_endpoint']} ...")
    frame = fetch_zmq_transactions(rpc, settings["zmq_endpoint"], max_tx=count)
    print(f"Captured {len(frame)} transactions.")
    if not frame.empty:
        with pd.option_context("display.max_colwidth", 20, "display.width", 160):
            print(frame[["txid", "timestamp", "input_addresses", "output_addresses", "fee"]].head(10))
