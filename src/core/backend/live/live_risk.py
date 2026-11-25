"""
live_risk.py
Forensic risk analysis for a single live Bitcoin transaction.

The unsupervised wallet pipeline is near-random on real labels, so it is only
triage. This module scores each transaction on structural patterns that are
genuinely meaningful in Bitcoin forensics and computable from the transaction
alone — no model, no training, every point explained. It is still an
*indicator* of patterns associated with illicit use, not proof of it.

One SIGNALS table drives both the score and the reasons, so a reason is only
ever reported for a signal that actually raised the score (the same contract
as the batch risk_engine).

Signals (each contributes up to `weight` points; total is capped at 100):
  watchlist_hit    an input/output address is on the user watchlist (forces CRITICAL)
  high_fan_out     many outputs — fan-out distribution / peeling / payout batching
  high_fan_in      many inputs — consolidation / sweep of many sources
  equal_outputs    several outputs of identical value — CoinJoin / mixer signature
  peeling_chain    two outputs, one a tiny remainder — classic layering step
  dust_outputs     outputs below the dust threshold — spam / address tainting
  round_value      large, suspiciously round total — unusual for organic payments
  high_value       large total output value
  self_churn       an input address reappears as an output — self-funding churn
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

# Thresholds (BTC unless noted). Deliberately conservative; tune per deployment.
DUST_BTC = 0.000006           # ~546 sat, Bitcoin Core's default dust limit for P2PKH
HIGH_VALUE_BTC = 50.0         # total output value considered large
ROUND_VALUE_BTC = 1.0         # round-number check only applies above this
FAN_OUT_SOFT = 10             # outputs beyond this start adding points
FAN_OUT_HARD = 50             # outputs at/above this get the full weight
FAN_IN_SOFT = 6               # consolidation sweeps are suspicious from fewer inputs
FAN_IN_HARD = 30              # ~30 inputs is already a strong sweep -> full weight

CRITICAL = "CRITICAL"
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"


@dataclass
class TxView:
    """The parts of a transaction the signals need."""
    txid: str
    input_addresses: Sequence[str]
    output_addresses: Sequence[str]
    input_amounts: Sequence[float]
    output_amounts: Sequence[float]
    fee: float = 0.0


@dataclass
class Signal:
    name: str
    weight: float
    reason: str
    # score(tx, watchlist) -> fraction of weight earned in [0, 1]
    score: Callable[[TxView, Set[str]], float]


def _ramp(value: float, soft: float, hard: float) -> float:
    """0 below soft, 1 at/above hard, linear between."""
    if value <= soft:
        return 0.0
    if value >= hard:
        return 1.0
    return (value - soft) / (hard - soft)


def _watchlist_hit(tx: TxView, watchlist: Set[str]) -> float:
    if not watchlist:
        return 0.0
    addresses = set(tx.input_addresses) | set(tx.output_addresses)
    return 1.0 if addresses & watchlist else 0.0


def _high_fan_out(tx: TxView, _watchlist: Set[str]) -> float:
    return _ramp(len(tx.output_addresses), FAN_OUT_SOFT, FAN_OUT_HARD)


def _high_fan_in(tx: TxView, _watchlist: Set[str]) -> float:
    return _ramp(len(tx.input_addresses), FAN_IN_SOFT, FAN_IN_HARD)


def _equal_outputs(tx: TxView, _watchlist: Set[str]) -> float:
    """Many outputs sharing one value is the signature of a CoinJoin / mixer."""
    amounts = [round(a, 8) for a in tx.output_amounts]
    if len(amounts) < 3:
        return 0.0
    most_common = max((amounts.count(a) for a in set(amounts)), default=0)
    if most_common < 3:
        return 0.0
    # 3 equal -> partial, 10+ equal -> full.
    return _ramp(most_common, 3, 10)


def _peeling_chain(tx: TxView, _watchlist: Set[str]) -> float:
    """Two outputs where one is a tiny remainder: a layering 'peel'."""
    if len(tx.output_amounts) != 2:
        return 0.0
    small, large = sorted(tx.output_amounts)
    total = small + large
    if total <= 0:
        return 0.0
    return 1.0 if small / total < 0.02 else 0.0


def _dust_outputs(tx: TxView, _watchlist: Set[str]) -> float:
    if not tx.output_amounts:
        return 0.0
    dust = sum(1 for a in tx.output_amounts if 0 < a < DUST_BTC)
    return _ramp(dust, 1, 5)


def _round_value(tx: TxView, _watchlist: Set[str]) -> float:
    total = sum(tx.output_amounts)
    if total < ROUND_VALUE_BTC:
        return 0.0
    # Exactly round to a whole BTC (within a hair) is unusual for organic spends.
    return 1.0 if abs(total - round(total)) < 1e-6 else 0.0


def _high_value(tx: TxView, _watchlist: Set[str]) -> float:
    return _ramp(sum(tx.output_amounts), HIGH_VALUE_BTC, HIGH_VALUE_BTC * 10)


def _self_churn(tx: TxView, _watchlist: Set[str]) -> float:
    # A two-output change-back to self is the normal payment pattern, so it is
    # not flagged; reuse amid more outputs is churn worth noting.
    reused = set(tx.input_addresses) & set(tx.output_addresses)
    return 1.0 if reused and len(tx.output_addresses) > 2 else 0.0


SIGNALS: List[Signal] = [
    Signal("watchlist_hit", 100, "address is on the watchlist of known-bad addresses", _watchlist_hit),
    Signal("high_fan_out", 35, "high fan-out: funds split across many outputs (distribution / peeling)", _high_fan_out),
    Signal("high_fan_in", 45, "high fan-in: many inputs consolidated (sweep / collection)", _high_fan_in),
    Signal("equal_outputs", 40, "many equal-value outputs (CoinJoin / mixer signature)", _equal_outputs),
    Signal("peeling_chain", 42, "peeling chain: a tiny remainder output (layering step)", _peeling_chain),
    Signal("dust_outputs", 20, "dust outputs (spam / address tainting)", _dust_outputs),
    Signal("round_value", 10, "suspiciously round total value", _round_value),
    Signal("high_value", 20, "large total value moved", _high_value),
    Signal("self_churn", 10, "input address reused as an output (self-churn)", _self_churn),
]


@dataclass
class TxRisk:
    txid: str
    risk_score: float              # 0-100
    risk_level: str
    reasons: List[str] = field(default_factory=list)
    components: Dict[str, float] = field(default_factory=dict)


def _level(score: float, watchlisted: bool) -> str:
    if watchlisted:
        return CRITICAL
    if score >= 70:
        return HIGH          # heuristics alone never auto-escalate to CRITICAL
    if score >= 40:
        return MEDIUM
    if score >= 15:
        return LOW
    return LOW


def analyze_transaction(tx: TxView, watchlist: Optional[Set[str]] = None) -> TxRisk:
    """Score one transaction on the forensic signals and explain the score."""
    watchlist = watchlist or set()
    components: Dict[str, float] = {}
    earned = 0.0
    reasons: List[str] = []
    watchlisted = False

    for signal in SIGNALS:
        fraction = signal.score(tx, watchlist)
        points = signal.weight * fraction
        components[signal.name] = points
        if points > 0:
            earned += points
            reasons.append(signal.reason)
            if signal.name == "watchlist_hit":
                watchlisted = True

    score = min(earned, 100.0)
    # A watchlist hit is decisive on its own.
    if watchlisted:
        score = 100.0
    return TxRisk(txid=tx.txid, risk_score=round(score, 1),
                  risk_level=_level(score, watchlisted),
                  reasons=reasons, components=components)


# --- frame-level analysis, used by the monitor cycle --------------------

def _tx_view_from_row(row: pd.Series) -> TxView:
    return TxView(
        txid=row["txid"],
        input_addresses=list(row["input_addresses[]"]),
        output_addresses=list(row["output_addresses[]"]),
        input_amounts=list(row["input_amounts[]"]),
        output_amounts=list(row["output_amounts[]"]),
        fee=float(row.get("fee", 0) or 0),
    )


def analyze_frame(clean: pd.DataFrame,
                  watchlist: Optional[Set[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score every transaction in a validated frame and aggregate to wallets.

    Returns (tx_risk, wallet_risk):
      tx_risk      one row per transaction, with risk_score/level/reasons and
                   the fields the store keeps (counts, value, fee).
      wallet_risk  one row per address, indexed by address, risk = the highest
                   risk of any transaction the address took part in, with that
                   transaction's reasons. Matches what LiveStore expects.
    """
    watchlist = watchlist or set()
    tx_rows: List[dict] = []
    best: Dict[str, Tuple[float, str, List[str]]] = {}
    tx_count: Dict[str, int] = defaultdict(int)

    for _, row in clean.iterrows():
        view = _tx_view_from_row(row)
        risk = analyze_transaction(view, watchlist)
        out_total = float(sum(view.output_amounts)) if len(view.output_amounts) else 0.0
        tx_rows.append({
            "txid": view.txid,
            "timestamp": str(row.get("timestamp", "")),
            "n_inputs": len(view.input_addresses),
            "n_outputs": len(view.output_addresses),
            "total_btc": out_total,
            "fee_btc": view.fee,
            "block_height": int(row.get("block_height", 0) or 0),
            "risk_score": risk.risk_score,
            "risk_level": risk.risk_level,
            "reasons": risk.reasons,
        })
        for address in set(view.input_addresses) | set(view.output_addresses):
            tx_count[address] += 1
            if address not in best or risk.risk_score > best[address][0]:
                best[address] = (risk.risk_score, risk.risk_level, risk.reasons)

    tx_risk = pd.DataFrame(tx_rows)
    wallet_risk = pd.DataFrame(
        [{"risk_score": score, "risk_level": level, "top_reasons": reasons,
          "total_tx_count": tx_count[addr]}
         for addr, (score, level, reasons) in best.items()],
        index=pd.Index(list(best.keys()), name="wallet_address"),
    )
    if not wallet_risk.empty:
        wallet_risk = wallet_risk.sort_values("risk_score", ascending=False)
    return tx_risk, wallet_risk
