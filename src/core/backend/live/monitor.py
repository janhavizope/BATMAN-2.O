"""
monitor.py
The live monitoring loop. One cycle:

    fetch recent Bitcoin transactions  (public mempool.space / Esplora API)
      -> validate against the BATMAN schema  (validator)
      -> forensic analysis of each one       (live_risk)
      -> persist the cycle                    (LiveStore)

Run one cycle from code with `run_cycle(store)`, or run continuously from the
command line:

    py -m src.core.backend.live.monitor --interval 30 --max-tx 40

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

from src.config import LIVE_DB
from src.core.backend.ingestion.ingestor import TransactionIngestor
from src.core.backend.live.live_risk import analyze_frame
from src.core.backend.live.live_store import LiveStore
from src.core.backend.live.mempool_feed import MempoolFeedError, fetch_live_transactions
from src.core.backend.live.watchlist import load_watchlist
from src.core.backend.validation.validator import TransactionValidator

# The feed's "cannot reach the source" error, caught so the loop survives outages.
FEED_ERRORS: tuple = (MempoolFeedError,)

DEFAULT_MAX_TX = 40


@dataclass
class CycleResult:
    fetched: int
    valid: int
    wallets: int
    flagged_tx: int = 0
    error: Optional[str] = None


def _validate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate an in-memory feed frame the way the ingestor validates a file."""
    frame = TransactionIngestor().normalize_columns(frame)
    return TransactionValidator().validate(frame)


def run_cycle(store: LiveStore, max_tx: int = DEFAULT_MAX_TX, watchlist=None,
              fetch_fn: Optional[Callable[[int], pd.DataFrame]] = None) -> CycleResult:
    """Run one fetch -> validate -> forensic analysis -> store cycle.

    `fetch_fn(max_tx)` returns the transactions to score; it defaults to the
    public mempool API. Feed errors are caught and returned on the result rather
    than raised, so a long-running loop survives a transient outage.
    """
    if watchlist is None:
        watchlist, _ = load_watchlist()
    if fetch_fn is None:
        fetch_fn = lambda n: fetch_live_transactions(max_tx=n)

    try:
        transactions = fetch_fn(max_tx)
    except FEED_ERRORS as error:
        return CycleResult(fetched=0, valid=0, wallets=0, error=str(error))

    fetched = len(transactions)
    if transactions.empty:
        return CycleResult(fetched=0, valid=0, wallets=0)

    clean, report = _validate_frame(transactions)
    if clean.empty:
        return CycleResult(fetched=fetched, valid=0, wallets=0)

    tx_risk, wallet_risk = analyze_frame(clean, watchlist=watchlist)
    # Attach the live-computable supervised model's probability (0-100) per tx,
    # as a learned second opinion alongside the forensic score. Best-effort.
    try:
        from src.core.backend.live.live_model import supervised_tx_risk
        sup = supervised_tx_risk(clean)
        if sup is not None and not tx_risk.empty:
            sup_by_txid = dict(zip(clean["txid"], (sup.values * 100.0)))
            tx_risk["supervised_risk"] = tx_risk["txid"].map(sup_by_txid).round(1)
    except Exception:  # pragma: no cover - supervised reading is optional
        pass

    # Wallet<->transaction edges for the live network graph (capped per tx so a
    # big mixer/sweep does not flood the graph).
    risk_by_txid = dict(zip(tx_risk["txid"], tx_risk["risk_level"])) if not tx_risk.empty else {}
    edge_rows = []
    for _, row in clean.iterrows():
        txid = row["txid"]
        level = risk_by_txid.get(txid, "LOW")
        for addr in list(row["input_addresses[]"])[:15]:
            edge_rows.append({"txid": txid, "address": addr, "role": "input", "risk_level": level})
        for addr in list(row["output_addresses[]"])[:15]:
            edge_rows.append({"txid": txid, "address": addr, "role": "output", "risk_level": level})
    tx_edges = pd.DataFrame(edge_rows)

    store.record_cycle(tx_risk, wallet_risk, tx_fetched=fetched, tx_valid=report["valid_records"],
                       tx_edges=tx_edges)
    store.prune()
    flagged = int(tx_risk["risk_level"].isin(["HIGH", "CRITICAL"]).sum()) if not tx_risk.empty else 0
    return CycleResult(fetched=fetched, valid=report["valid_records"],
                       wallets=len(wallet_risk), flagged_tx=flagged)


def run_loop(interval: float = 30.0, max_tx: int = DEFAULT_MAX_TX, db_path=LIVE_DB,
             max_cycles: Optional[int] = None) -> None:
    """Run cycles forever (or `max_cycles` times), pausing `interval` seconds."""
    store = LiveStore(db_path)
    print(f"BATMAN live monitor (public API) -> {db_path}")
    print(f"Fetching up to {max_tx} transactions every {interval:.0f}s. Ctrl-C to stop.\n")
    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        result = run_cycle(store, max_tx=max_tx)
        stamp = time.strftime("%H:%M:%S")
        if result.error:
            print(f"[{stamp}] cycle {cycle}: feed error — {result.error}")
        else:
            print(f"[{stamp}] cycle {cycle}: fetched {result.fetched}, valid {result.valid}, "
                  f"wallets {result.wallets}, flagged tx {result.flagged_tx}")
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
    parser.add_argument("--max-tx", type=int, default=DEFAULT_MAX_TX,
                        help="transactions to fetch per cycle")
    parser.add_argument("--cycles", type=int, default=None, help="stop after this many cycles")
    parser.add_argument("--db", default=str(LIVE_DB), help="path to the SQLite store")
    args = parser.parse_args()
    run_loop(interval=args.interval, max_tx=args.max_tx, db_path=args.db,
             max_cycles=args.cycles)


if __name__ == "__main__":
    main()
