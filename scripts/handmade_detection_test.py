"""
handmade_detection_test.py
--------------------------
A self-contained detection test on transactions we INVENT here — none come from
the Elliptic dataset, the stored database, or the live feed. We hand-build
ordinary (normal) transactions and criminal-style transactions, run them through
the live forensic engine, and check it flags the criminal ones, clears the
normal ones, and does not over-flag an ambiguous large payment.

Run:
    python scripts/handmade_detection_test.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.backend.ingestion.ingestor import TransactionIngestor
from src.core.backend.validation.validator import TransactionValidator
from src.core.backend.live.live_risk import analyze_frame

T = "2024-06-01T12:00:00"


def tx(idx, ins, outs, in_amts, out_amts):
    return dict(txid=format(idx, "064x"), timestamp=T,
                input_addresses="|".join(ins), output_addresses="|".join(outs),
                input_amounts="|".join(map(str, in_amts)),
                output_amounts="|".join(map(str, out_amts)), fee=0.0001)


# (description, expected: normal | criminal | borderline, builder)
SPECS = [
    ("Ordinary payment (buy + change)", "normal",
     lambda i: tx(i, ["alice"], ["shop", "alice"], [0.5], [0.3, 0.1999])),
    ("Split bill to two friends", "normal",
     lambda i: tx(i, ["bob"], ["carol", "dave", "bob"], [0.9], [0.3, 0.3, 0.2999])),
    ("Small coffee payment", "normal",
     lambda i: tx(i, ["eve"], ["cafe", "eve"], [0.02], [0.004, 0.0159])),
    ("Salary payment", "normal",
     lambda i: tx(i, ["employer"], ["worker"], [1.2], [1.1999])),
    ("MIXER: 1 -> 20 identical outputs", "criminal",
     lambda i: tx(i, ["mix_in"], [f"m{j}" for j in range(20)], [20.0], [0.99] * 20)),
    ("SWEEP: 40 wallets -> 1 collector", "criminal",
     lambda i: tx(i, [f"src{j}" for j in range(40)], ["collector"], [0.1] * 40, [3.99])),
    ("PEELING CHAIN: pay + tiny remainder", "criminal",
     lambda i: tx(i, ["peel"], ["recipient", "peel"], [8.0], [7.999, 0.0009])),
    ("DUST SPAM: 30 dust outputs", "criminal",
     lambda i: tx(i, ["duster"], [f"v{j}" for j in range(30)], [0.01], [0.0000054] * 30)),
    ("WATCHLIST: pays a known-bad address", "criminal",
     lambda i: tx(i, ["sender"], ["1KnownBadAddr"], [0.5], [0.4999])),
    # A single large round transfer is common in LEGITIMATE exchange activity,
    # so the engine should NOT flag it (it needs a laundering structure).
    ("LARGE ROUND VALUE: exactly 100 BTC (borderline)", "borderline",
     lambda i: tx(i, ["whale"], ["dest"], [100.0], [100.0])),
]


def main() -> int:
    rows, meta = [], []
    for i, (desc, expected, build) in enumerate(SPECS):
        row = build(i)
        rows.append(row)
        meta.append((row["txid"], desc, expected))

    frame = TransactionIngestor().normalize_columns(pd.DataFrame(rows))
    clean, report = TransactionValidator().validate(frame)
    tx_risk, _ = analyze_frame(clean, watchlist={"1KnownBadAddr"})
    by_id = {r["txid"]: r for _, r in tx_risk.iterrows()}

    print("HAND-MADE TRANSACTIONS (invented here - not from any dataset or feed)")
    print(f"validated: {report['valid_records']}/{report['total_records']}\n")
    print(f"{'description':<48} {'expected':<11} {'detected':<9} {'score':>5}  ok")
    print("-" * 86)
    correct = 0
    for txid, desc, expected in meta:
        r = by_id[txid]
        flagged = r["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL")
        ok = flagged if expected == "criminal" else not flagged
        correct += ok
        print(f"{desc:<48} {expected:<11} {r['risk_level']:<9} {r['risk_score']:>5.0f}  {'YES' if ok else 'NO'}")
    print("-" * 86)
    print(f"\nCorrect: {correct}/{len(meta)}  ({correct / len(meta):.0%})")
    print("Normal/borderline transactions should read LOW; criminal ones MEDIUM/HIGH/CRITICAL.")
    return 0 if correct == len(meta) else 1


if __name__ == "__main__":
    sys.exit(main())
