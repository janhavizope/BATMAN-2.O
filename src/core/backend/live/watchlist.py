"""
watchlist.py
Loads a user-maintained list of known-bad Bitcoin addresses. A live
transaction touching any of them is decisive (CRITICAL) in live_risk.

The list is deliberately empty by default: labelling a real address as illicit
is a claim that needs a source (an OFAC SDN entry, an abuse report, a court
record). Populate data/live/watchlist.json yourself, or point the loader at a
feed you trust. Format — a JSON array of addresses, or objects with a reason:

    ["bc1q...addr1", {"address": "1A1z...", "reason": "OFAC SDN 2023-01-01"}]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set, Tuple

from src.config import LIVE_DATA_DIR

WATCHLIST_PATH = LIVE_DATA_DIR / "watchlist.json"


def load_watchlist(path: Path | str = WATCHLIST_PATH) -> Tuple[Set[str], Dict[str, str]]:
    """Return (addresses, reasons). Missing or malformed file -> empty, never raises."""
    path = Path(path)
    if not path.exists():
        return set(), {}
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set(), {}
    if not isinstance(entries, list):
        return set(), {}

    addresses: Set[str] = set()
    reasons: Dict[str, str] = {}
    for entry in entries:
        if isinstance(entry, str):
            addresses.add(entry)
        elif isinstance(entry, dict) and entry.get("address"):
            address = str(entry["address"])
            addresses.add(address)
            if entry.get("reason"):
                reasons[address] = str(entry["reason"])
    return addresses, reasons
