"""
mempool_feed.py
Pulls live Bitcoin transactions from a public mempool.space / Esplora API and
maps them into the BATMAN transaction schema the validator expects:

    txid, timestamp, input_addresses, output_addresses,
    input_amounts, output_amounts, fee

Amounts are in BTC (the API reports satoshis; we divide by 1e8). Unconfirmed
mempool transactions have no block time, so their timestamp is the moment we
fetched them. Only the standard library is used, so no new dependency is added.

The public API is rate limited. Keep `max_tx` modest (tens, not thousands) and
leave a short pause between cycles; that is what `monitor.py` does.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

# mempool.space hosts an Esplora-compatible API. Blockstream's
# https://blockstream.info/api is a drop-in alternative base URL.
DEFAULT_BASE_URL = "https://mempool.space/api"
FALLBACK_BASE_URL = "https://blockstream.info/api"
SATS_PER_BTC = 100_000_000
USER_AGENT = "BATMAN-2.0 live monitor (research)"


class MempoolFeedError(RuntimeError):
    """Raised when the live feed cannot be reached or returns unexpected data."""


def _get(url: str, timeout: float = 15.0) -> Any:
    """GET a URL and parse JSON, with a descriptive error on failure."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        raise MempoolFeedError(f"{url} returned HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise MempoolFeedError(f"could not reach {url}: {error}") from error
    try:
        return json.loads(payload)
    except json.JSONDecodeError as error:
        raise MempoolFeedError(f"{url} returned invalid JSON") from error


def _recent_txids(base_url: str, limit: int) -> List[str]:
    """Most recently seen mempool transaction ids (newest first)."""
    recent = _get(f"{base_url}/mempool/recent")
    if not isinstance(recent, list):
        raise MempoolFeedError("unexpected response from /mempool/recent")
    txids = [item["txid"] for item in recent if isinstance(item, dict) and "txid" in item]
    return txids[:limit]


def _tx_to_row(tx: Dict[str, Any], fetched_at: datetime) -> Optional[Dict[str, Any]]:
    """Map one Esplora transaction object to a BATMAN schema row.

    Inputs and outputs without an address (coinbase inputs, OP_RETURN outputs)
    are skipped so address and amount lists stay aligned. Returns None when the
    transaction has no usable input or output address (e.g. a coinbase tx).
    """
    in_addresses: List[str] = []
    in_amounts: List[float] = []
    for vin in tx.get("vin", []):
        prevout = vin.get("prevout") or {}
        address = prevout.get("scriptpubkey_address")
        if address is None:
            continue
        in_addresses.append(address)
        in_amounts.append(prevout.get("value", 0) / SATS_PER_BTC)

    out_addresses: List[str] = []
    out_amounts: List[float] = []
    for vout in tx.get("vout", []):
        address = vout.get("scriptpubkey_address")
        if address is None:
            continue
        out_addresses.append(address)
        out_amounts.append(vout.get("value", 0) / SATS_PER_BTC)

    if not in_addresses or not out_addresses:
        return None

    status = tx.get("status") or {}
    block_time = status.get("block_time")
    if status.get("confirmed") and block_time:
        timestamp = datetime.fromtimestamp(block_time, tz=timezone.utc)
    else:
        timestamp = fetched_at

    return {
        "txid": tx.get("txid"),
        "timestamp": timestamp.isoformat(),
        # Pipe-delimited lists are one of the formats the validator accepts.
        "input_addresses": "|".join(in_addresses),
        "output_addresses": "|".join(out_addresses),
        "input_amounts": "|".join(f"{a:.8f}" for a in in_amounts),
        "output_amounts": "|".join(f"{a:.8f}" for a in out_amounts),
        "fee": tx.get("fee", 0) / SATS_PER_BTC,
        "block_height": status.get("block_height", 0) or 0,
    }


def fetch_live_transactions(max_tx: int = 40, base_url: str = DEFAULT_BASE_URL,
                            pause: float = 0.05) -> pd.DataFrame:
    """Fetch up to `max_tx` current transactions as a BATMAN-schema DataFrame.

    It reads the recent mempool, then fetches each transaction's full detail
    (inputs with their previous outputs give the sending addresses). `pause`
    is a short sleep between detail requests to be gentle on the public API.

    Raises MempoolFeedError if the feed is unreachable. An individual
    transaction that fails to fetch is skipped, not fatal.
    """
    try:
        txids = _recent_txids(base_url, max_tx)
    except MempoolFeedError as error:
        if base_url == DEFAULT_BASE_URL:
            base_url = FALLBACK_BASE_URL
            txids = _recent_txids(base_url, max_tx)
        else:
            raise
    fetched_at = datetime.now(tz=timezone.utc)
    rows: List[Dict[str, Any]] = []
    for txid in txids:
        try:
            tx = _get(f"{base_url}/tx/{txid}")
        except MempoolFeedError:
            continue
        row = _tx_to_row(tx, fetched_at)
        if row is not None:
            rows.append(row)
        if pause:
            time.sleep(pause)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"Fetching {count} live Bitcoin transactions from {DEFAULT_BASE_URL} ...")
    frame = fetch_live_transactions(max_tx=count)
    print(f"Mapped {len(frame)} transactions to the BATMAN schema.")
    if not frame.empty:
        with pd.option_context("display.max_colwidth", 24, "display.width", 160):
            print(frame[["txid", "timestamp", "input_addresses", "output_addresses", "fee"]].head(10))
