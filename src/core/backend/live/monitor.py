"""
monitor.py
The live monitoring loop. One cycle:

    fetch recent Bitcoin transactions  (mempool API, or your own node over ZMQ)
      -> validate against the BATMAN schema  (validator)
      -> forensic analysis of each one       (live_risk)
      -> persist the cycle                    (LiveStore)

Run one cycle from code with `run_cycle(store)`, or run continuously from the
command line:

    py -m src.core.backend.live.monitor --interval 30 --max-tx 40          # public API
    py -m src.core.backend.live.monitor --source zmq --interval 30         # your full node

The live score is the explainable forensic analysis (live_risk), not a trained
model — see the README. Each flag is an indicator of a pattern associated with
illicit use, not proof of it.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from src.config import LIVE_DB, get_node_settings
from src.core.backend.ingestion.ingestor import TransactionIngestor
from src.core.backend.live.live_risk import analyze_frame
from src.core.backend.live.live_store import LiveStore
from src.core.backend.live.mempool_feed import MempoolFeedError, fetch_live_transactions
from src.core.backend.live.watchlist import load_watchlist
from src.core.backend.validation.validator import TransactionValidator

# Any feed's "cannot reach the source" error, caught so the loop survives outages.
FEED_ERRORS: tuple = (MempoolFeedError,)
try:  # the ZMQ feed is optional (needs pyzmq + a node); import defensively
    from src.core.backend.live.node_rpc import NodeRPCError
    from src.core.backend.live.zmq_feed import ZMQFeedError
    FEED_ERRORS = (MempoolFeedError, ZMQFeedError, NodeRPCError)
except ImportError:  # pragma: no cover
    pass


API_DEFAULT_MAX_TX = 40
# With the ZMQ feed a cycle processes everything queued since the last one, up
# to this many transactions (one local RPC call each); the rest wait a cycle.
ZMQ_MAX_PER_CYCLE = 5_000


@dataclass
class CycleResult:
    fetched: int
    valid: int
    wallets: int
    flagged_tx: int = 0
    error: Optional[str] = None
    # ZMQ feed health after the cycle (received / queued / missed / ...), else None
    feed_stats: Optional[dict] = None


class ZMQFetch:
    """A cycle fetch function backed by the shared persistent ZMQ listener."""

    def __init__(self, rpc, listener):
        self.rpc = rpc
        self.listener = listener

    def __call__(self, max_tx: Optional[int]) -> pd.DataFrame:
        from src.core.backend.live.zmq_feed import fetch_from_listener
        return fetch_from_listener(self.rpc, self.listener, max_tx or ZMQ_MAX_PER_CYCLE)

    def stats(self) -> dict:
        return self.listener.stats()


def _validate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate an in-memory feed frame the way the ingestor validates a file."""
    frame = TransactionIngestor().normalize_columns(frame)
    return TransactionValidator().validate(frame)


def run_cycle(store: LiveStore, max_tx: Optional[int] = API_DEFAULT_MAX_TX, watchlist=None,
              fetch_fn: Optional[Callable[[Optional[int]], pd.DataFrame]] = None) -> CycleResult:
    """Run one fetch -> validate -> forensic analysis -> store cycle.

    `fetch_fn(max_tx)` returns the transactions to score; it defaults to the
    public mempool API. Pass the node ZMQ feed (see build_zmq_fetch) to monitor
    firsthand; with it, max_tx=None processes everything queued since the last
    cycle. Feed errors are caught and returned on the result rather than
    raised, so a long-running loop survives a transient outage.
    """
    if watchlist is None:
        watchlist, _ = load_watchlist()
    if fetch_fn is None:
        fetch_fn = lambda n: fetch_live_transactions(max_tx=n or API_DEFAULT_MAX_TX)
    stats_fn = getattr(fetch_fn, "stats", None)

    def result(**fields) -> CycleResult:
        return CycleResult(feed_stats=stats_fn() if stats_fn else None, **fields)

    try:
        transactions = fetch_fn(max_tx)
    except FEED_ERRORS as error:
        return result(fetched=0, valid=0, wallets=0, error=str(error))

    fetched = len(transactions)
    if transactions.empty:
        return result(fetched=0, valid=0, wallets=0)

    clean, report = _validate_frame(transactions)
    if clean.empty:
        return result(fetched=fetched, valid=0, wallets=0)

    tx_risk, wallet_risk = analyze_frame(clean, watchlist=watchlist)
    store.record_cycle(tx_risk, wallet_risk, tx_fetched=fetched, tx_valid=report["valid_records"])
    store.prune()
    flagged = int(tx_risk["risk_level"].isin(["HIGH", "CRITICAL"]).sum()) if not tx_risk.empty else 0
    return result(fetched=fetched, valid=report["valid_records"],
                  wallets=len(wallet_risk), flagged_tx=flagged)


def build_zmq_fetch() -> ZMQFetch:
    """Build a fetch function backed by your Bitcoin Core node's ZMQ feed.

    Reads node settings from the environment (see src.config.get_node_settings),
    verifies the node answers, and attaches to the shared persistent listener
    for the configured endpoint (started on first use, then reused, so the
    subscription stays open between cycles and page reruns). Raises a feed
    error if pyzmq or the node is unavailable.
    """
    from src.core.backend.live.node_rpc import BitcoinRPC
    from src.core.backend.live.zmq_feed import get_listener

    settings = get_node_settings()
    rpc = BitcoinRPC(url=settings["rpc_url"], user=settings["rpc_user"],
                     password=settings["rpc_password"], cookie_path=settings["rpc_cookie"])
    rpc.ping()  # fail fast with a clear error if the node is not reachable
    return ZMQFetch(rpc, get_listener(settings["zmq_endpoint"]))


def format_feed_stats(stats: Optional[dict]) -> str:
    """One line describing ZMQ feed health, or '' for the API feed."""
    if not stats:
        return ""
    text = (f"node feed: {stats['received']:,} announced, {stats['queued']:,} queued, "
            f"{stats['missed']:,} missed, {stats['unresolved']:,} unresolved")
    if stats.get("duplicates"):
        text += f", {stats['duplicates']:,} repeats skipped (re-announced when mined)"
    if stats["overflow"]:
        text += f", {stats['overflow']:,} dropped (queue full)"
    return text


def run_loop(interval: float = 30.0, max_tx: Optional[int] = None, db_path=LIVE_DB,
             max_cycles: Optional[int] = None, source: str = "api") -> None:
    """Run cycles forever (or `max_cycles` times), pausing `interval` seconds.

    max_tx=None means the feed's default: 40 for the API, and everything
    queued (up to ZMQ_MAX_PER_CYCLE) for the node feed.
    """
    store = LiveStore(db_path)
    fetch_fn = None
    if source == "zmq":
        print("Connecting to your Bitcoin node (ZMQ) ...")
        fetch_fn = build_zmq_fetch()
        print("Node reachable; listening for transactions continuously.")
        batch = f"all queued (up to {max_tx or ZMQ_MAX_PER_CYCLE:,})"
    else:
        max_tx = max_tx or API_DEFAULT_MAX_TX
        batch = f"up to {max_tx}"
    print(f"BATMAN live monitor ({source}) -> {db_path}")
    print(f"Processing {batch} transactions every {interval:.0f}s. Ctrl-C to stop.\n")
    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        result = run_cycle(store, max_tx=max_tx, fetch_fn=fetch_fn)
        stamp = time.strftime("%H:%M:%S")
        health = format_feed_stats(result.feed_stats)
        health = f" | {health}" if health else ""
        if result.error:
            print(f"[{stamp}] cycle {cycle}: feed error — {result.error}{health}")
        else:
            print(f"[{stamp}] cycle {cycle}: fetched {result.fetched}, valid {result.valid}, "
                  f"wallets {result.wallets}, flagged tx {result.flagged_tx}{health}")
        if max_cycles is not None and cycle >= max_cycles:
            break
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            break
    print("\nStopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="BATMAN live Bitcoin monitor")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between cycles")
    parser.add_argument("--max-tx", type=int, default=None,
                        help=f"transactions per cycle (default: {API_DEFAULT_MAX_TX} for api; "
                             f"everything queued, up to {ZMQ_MAX_PER_CYCLE:,}, for zmq)")
    parser.add_argument("--cycles", type=int, default=None, help="stop after this many cycles")
    parser.add_argument("--db", default=str(LIVE_DB), help="path to the SQLite store")
    parser.add_argument("--source", choices=["api", "zmq"], default="api",
                        help="api: public mempool.space feed; zmq: your own Bitcoin node")
    args = parser.parse_args()
    run_loop(interval=args.interval, max_tx=args.max_tx, db_path=args.db,
             max_cycles=args.cycles, source=args.source)


if __name__ == "__main__":
    main()
