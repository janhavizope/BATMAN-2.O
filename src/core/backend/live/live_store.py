"""
live_store.py
A small SQLite rolling store for the live monitor. It keeps the transactions
seen in each cycle (with their forensic risk) and a running risk record per
wallet, so the web page can show the current state without re-fetching.

Two main tables:
  live_transactions  one row per transaction, with its forensic risk + reasons
  live_wallets       one row per address, with its latest and peak risk

The risk is the forensic transaction score from live_risk (explainable
structural signals), aggregated to a wallet as the highest risk of any
transaction it took part in. The store keeps both the latest cycle's score and
the highest score a wallet has ever reached.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_transactions (
    txid         TEXT PRIMARY KEY,
    fetched_at   TEXT NOT NULL,
    timestamp    TEXT,
    n_inputs     INTEGER,
    n_outputs    INTEGER,
    total_btc    REAL,
    fee_btc      REAL,
    block_height INTEGER,
    risk_score   REAL,
    risk_level   TEXT,
    reasons      TEXT
);
CREATE TABLE IF NOT EXISTS live_wallets (
    address        TEXT PRIMARY KEY,
    first_seen     TEXT NOT NULL,
    last_seen      TEXT NOT NULL,
    times_scored   INTEGER NOT NULL,
    risk_score     REAL,
    risk_level     TEXT,
    max_risk_score REAL,
    top_reasons    TEXT,
    total_tx_count INTEGER
);
CREATE TABLE IF NOT EXISTS live_cycles (
    cycle_at     TEXT PRIMARY KEY,
    tx_fetched   INTEGER,
    tx_valid     INTEGER,
    wallets      INTEGER,
    flagged_tx   INTEGER
);
"""

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


class LiveStore:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def record_cycle(self, tx_risk: pd.DataFrame, wallet_risk: pd.DataFrame,
                     tx_fetched: int, tx_valid: int) -> None:
        """Persist one monitoring cycle: scored transactions and wallet risk."""
        now = datetime.now(tz=timezone.utc).isoformat()
        flagged = int((tx_risk["risk_level"].isin(["HIGH", "CRITICAL"])).sum()) if not tx_risk.empty else 0
        with closing(self._connect()) as conn:
            self._upsert_transactions(conn, tx_risk, now)
            self._upsert_wallets(conn, wallet_risk, now)
            conn.execute(
                "INSERT OR REPLACE INTO live_cycles "
                "(cycle_at, tx_fetched, tx_valid, wallets, flagged_tx) VALUES (?, ?, ?, ?, ?)",
                (now, int(tx_fetched), int(tx_valid), int(len(wallet_risk)), flagged),
            )
            conn.commit()

    def _upsert_transactions(self, conn: sqlite3.Connection, txs: pd.DataFrame, now: str) -> None:
        rows: List[tuple] = []
        for _, tx in txs.iterrows():
            rows.append((
                tx["txid"], now, str(tx.get("timestamp", "")),
                int(tx.get("n_inputs", 0) or 0), int(tx.get("n_outputs", 0) or 0),
                float(tx.get("total_btc", 0) or 0), float(tx.get("fee_btc", 0) or 0),
                int(tx.get("block_height", 0) or 0),
                float(tx.get("risk_score", 0) or 0), str(tx.get("risk_level", "LOW")),
                "; ".join(tx.get("reasons", []) or []),
            ))
        conn.executemany(
            "INSERT OR IGNORE INTO live_transactions "
            "(txid, fetched_at, timestamp, n_inputs, n_outputs, total_btc, fee_btc, "
            "block_height, risk_score, risk_level, reasons) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    def _upsert_wallets(self, conn: sqlite3.Connection, scores: pd.DataFrame, now: str) -> None:
        for address, row in scores.iterrows():
            score = float(row["risk_score"])
            level = str(row["risk_level"])
            reasons = "; ".join(row.get("top_reasons", []) or [])
            tx_count = int(row.get("total_tx_count", 0) or 0)
            existing = conn.execute(
                "SELECT times_scored, max_risk_score, risk_level "
                "FROM live_wallets WHERE address = ?", (address,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO live_wallets (address, first_seen, last_seen, times_scored, "
                    "risk_score, risk_level, max_risk_score, top_reasons, total_tx_count) "
                    "VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)",
                    (address, now, now, score, level, score, reasons, tx_count),
                )
            else:
                max_score = max(existing["max_risk_score"], score)
                # Keep the more severe band ever seen as the peak level.
                peak_level = level if RISK_ORDER.get(level, 0) >= RISK_ORDER.get(
                    existing["risk_level"], 0) else existing["risk_level"]
                conn.execute(
                    "UPDATE live_wallets SET last_seen = ?, times_scored = times_scored + 1, "
                    "risk_score = ?, risk_level = ?, max_risk_score = ?, top_reasons = ?, "
                    "total_tx_count = total_tx_count + ? WHERE address = ?",
                    (now, score, peak_level, max_score, reasons, tx_count, address),
                )

    def prune(self, keep_transactions: int = 5000, keep_wallets: int = 5000) -> None:
        """Keep the store bounded: drop the oldest/least-risky rows beyond the limits."""
        with closing(self._connect()) as conn:
            conn.execute(
                "DELETE FROM live_transactions WHERE txid NOT IN "
                "(SELECT txid FROM live_transactions ORDER BY fetched_at DESC LIMIT ?)",
                (keep_transactions,),
            )
            conn.execute(
                "DELETE FROM live_wallets WHERE address NOT IN "
                "(SELECT address FROM live_wallets ORDER BY max_risk_score DESC LIMIT ?)",
                (keep_wallets,),
            )
            conn.commit()

    # --- read side, used by the web page ---------------------------------

    def wallets(self, limit: int = 1000) -> pd.DataFrame:
        with closing(self._connect()) as conn:
            return pd.read_sql_query(
                "SELECT * FROM live_wallets ORDER BY max_risk_score DESC LIMIT ?",
                conn, params=(limit,),
            )

    def transactions(self, limit: int = 1000, min_level: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM live_transactions"
        params: list = []
        if min_level:
            allowed = [lvl for lvl, rank in RISK_ORDER.items() if rank >= RISK_ORDER.get(min_level, 0)]
            query += " WHERE risk_level IN (%s)" % ",".join("?" * len(allowed))
            params.extend(allowed)
        query += " ORDER BY risk_score DESC, fetched_at DESC LIMIT ?"
        params.append(limit)
        with closing(self._connect()) as conn:
            return pd.read_sql_query(query, conn, params=tuple(params))

    def summary(self) -> Dict[str, Any]:
        with closing(self._connect()) as conn:
            def scalar(sql: str) -> Any:
                return conn.execute(sql).fetchone()[0]
            last_cycle = conn.execute(
                "SELECT cycle_at, tx_fetched, tx_valid, wallets, flagged_tx FROM live_cycles "
                "ORDER BY cycle_at DESC LIMIT 1").fetchone()
            return {
                "total_transactions": scalar("SELECT COUNT(*) FROM live_transactions"),
                "flagged_transactions": scalar(
                    "SELECT COUNT(*) FROM live_transactions WHERE risk_level IN ('HIGH', 'CRITICAL')"),
                "total_wallets": scalar("SELECT COUNT(*) FROM live_wallets"),
                "high_or_critical": scalar(
                    "SELECT COUNT(*) FROM live_wallets WHERE risk_level IN ('HIGH', 'CRITICAL')"),
                "cycles": scalar("SELECT COUNT(*) FROM live_cycles"),
                "last_cycle": dict(last_cycle) if last_cycle else None,
            }

    def level_breakdown(self) -> Dict[str, int]:
        """Transaction count per risk level, for the analysis panel."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT risk_level, COUNT(*) AS n FROM live_transactions GROUP BY risk_level"
            ).fetchall()
        counts = {level: 0 for level in RISK_ORDER}
        for row in rows:
            counts[row["risk_level"]] = row["n"]
        return counts
