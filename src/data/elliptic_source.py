"""
Elliptic++ data source for the Streamlit pages.

Serves the same functions as data_manager (overview stats, transactions,
alerts, entities, graph, scoreboard, explainability) from the risk scores
written by scripts/score_wallets.py and scripts/score_transactions.py.

Only wallets / transactions the models did not train on (test and unknown
splits) are presented as predictions. Entities are wallets; the transaction
graph comes from the Elliptic transaction flow edge list.

Score and supporting parquet files are loaded on demand so pages only pay the
I/O and processing cost for the datasets they actually use.
"""

import numpy as np
import networkx as nx
import pandas as pd

import json

from src.config import ELLIPTIC_PLUS_DIR as DATA_DIR, EVALUATION_DIR

WALLET_SCORES = DATA_DIR / "wallet_scores.parquet"
WALLET_TIMELINE = DATA_DIR / "wallet_timeline.parquet"
WALLET_EXPLANATIONS = DATA_DIR / "wallet_explanations.parquet"
TX_SCORES = DATA_DIR / "tx_scores.parquet"
TX_EDGES = DATA_DIR / "tx_edges.parquet"
WALLET_TX_EDGES = DATA_DIR / "wallet_tx_edges.parquet"

# Alert thresholds are chosen from the test-split precision-recall curve so
# that at most this many wallets / transactions are flagged (see choose_threshold)
WALLET_ALERT_BUDGET = 500
TX_ALERT_BUDGET = 1_000
# score_wallets.py stores timelines and explanations for wallets at or above 0.5
MIN_WALLET_THRESHOLD = 0.5

MAX_TRANSACTIONS = 10_000
MAX_ENTITIES = 500
MAX_EXPLAINED = 50
GRAPH_SEEDS = 40
GRAPH_ENTITIES = 10
MAX_TXS_PER_ENTITY = 15
OVERVIEW_WALLET_COLUMNS = (
    "address",
    "time_step",
    "class",
    "risk",
    "split",
    "total_txs",
    "transacted_w_address_total",
    "btc_transacted_total",
    "num_timesteps_appeared_in",
    "first_block_appeared_in",
    "last_block_appeared_in",
    "lifetime_in_blocks",
)

_CACHE = {}


REQUIRED_FILES = ("WALLET_SCORES", "WALLET_TIMELINE", "WALLET_EXPLANATIONS", "TX_SCORES", "TX_EDGES")
# A Git LFS pointer file (the repository was cloned without Git LFS) starts with this
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"

# Severity is an alert's position in the risk-ranked list: the top 10% are
# Critical, the next 30% High, the rest Medium. Positions come from the
# (risk, total_txs) order, so wallets tied on model risk are separated by their
# transaction count instead of all collapsing into one band.
CRITICAL_SHARE = 0.10
HIGH_SHARE = 0.40


def _is_lfs_pointer(path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:
        return False


def scores_status() -> str:
    """'ok', 'missing' (a score file does not exist) or 'lfs_pointer' (Git LFS not pulled)."""
    paths = [globals()[name] for name in REQUIRED_FILES]
    if not all(p.exists() for p in paths):
        return "missing"
    if any(_is_lfs_pointer(p) for p in paths):
        return "lfs_pointer"
    return "ok"


def is_available() -> bool:
    return scores_status() == "ok"


def choose_threshold(scores: pd.DataFrame, budget: int, floor: float = 0.0) -> dict:
    """Pick an alert threshold on the validation period, report it on the test period.

    Takes the threshold with the best illicit F1 on the validation split (time
    steps the model did not train on and that are not the test set), then raises
    it until at most `budget` non-training rows are flagged. Precision and recall
    are measured on the held-out test split, which is never used to choose the
    threshold, so they are not optimistic. Rows tied at the threshold are all
    flagged, so the flagged count can exceed the budget. If there is no
    validation split (older score files), only the budget and floor apply.
    """
    val = scores[scores["split"] == "val"]
    test = scores[scores["split"] == "test"]
    pool = scores.loc[scores["split"] != "train", "risk"]
    threshold = floor
    if val["class"].eq("illicit").any() and val["class"].ne("illicit").any():
        risks = val["risk"].to_numpy()
        labels = (val["class"] == "illicit").to_numpy()
        order = np.argsort(risks)
        sorted_risks = risks[order]
        positive_suffix = np.cumsum(labels[order][::-1], dtype=np.int64)[::-1]
        thresholds = np.unique(risks)
        starts = np.searchsorted(sorted_risks, thresholds, side="left")
        predicted = len(risks) - starts
        true_positive = positive_suffix[starts]
        f1 = 2 * true_positive / (predicted + labels.sum())
        threshold = max(threshold, float(thresholds[np.argmax(f1)]))
    if len(pool) > budget:
        threshold = max(threshold, float(pool.nlargest(budget).min()))

    flagged_test = test["risk"] >= threshold
    illicit = test["class"] == "illicit"
    return {
        "threshold": threshold,
        "precision": float((flagged_test & illicit).sum() / max(flagged_test.sum(), 1)),
        "recall": float((flagged_test & illicit).sum() / max(illicit.sum(), 1)),
        "flagged": int((pool >= threshold).sum()),
        "budget": budget,
    }


def get_thresholds() -> dict:
    """Wallet and transaction alert thresholds, with their test precision/recall."""
    data = _load()
    return {"wallet": data["wallet_threshold"], "transaction": data["tx_threshold"]}


def get_model_drift() -> dict:
    """Illicit-F1 of the Elliptic++ transaction model on each test time step.

    Shows how detection degrades over time: it is strong through step 42 and
    collapses after the step-43 dark-market shutdown, so the models need
    retraining when illicit behaviour shifts. Returns {'steps': [...], 'f1':
    [...]} (empty if the metrics file is missing).
    """
    try:
        by_step = json.loads((EVALUATION_DIR / "elliptic_plus_txs_metrics.json").read_text())
        by_step = by_step["f1_illicit_by_time_step"]
    except (OSError, ValueError, KeyError):
        return {"steps": [], "f1": []}
    steps = sorted(int(s) for s in by_step)
    return {"steps": steps, "f1": [float(by_step[str(s)]) for s in steps]}


def _wallet_threshold() -> float:
    return _load()["wallet_threshold"]["threshold"]


def _tx_threshold() -> float:
    return _load()["tx_threshold"]["threshold"]


def _severity_bands(n: int) -> list:
    """Severity label for each of `n` alerts, ordered highest-risk first.

    The alert threshold sits near 1.0 (~0.997 on the current scores), and
    Random Forest probabilities that close to 1.0 are not calibrated, so
    cutting the probability range into bands says little. Position does:
    Critical = the top CRITICAL_SHARE of alerts, High = up to HIGH_SHARE,
    Medium = the rest. The caller ranks alerts by (risk, total_txs), so
    wallets tied on model risk are separated by transaction count rather than
    all landing in one band. This keeps the 10/30/60 split even when a large
    tie sits at the top (229 wallets score exactly 1.0 on the current data);
    calibrating the probabilities (see the README TODO) would make the order
    within a tie reflect the model instead of this secondary signal.
    """
    if n <= 0:
        return []
    critical = max(1, int(np.ceil(n * CRITICAL_SHARE)))
    high = max(critical, int(np.ceil(n * HIGH_SHARE)))
    return ["Critical" if i < critical else "High" if i < high else "Medium" for i in range(n)]


def _step(time_step) -> str:
    return f"Step-{int(time_step):02d}"


def _fmt(value) -> str:
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.4g}"


def _pretty(feature: str) -> str:
    return feature.replace("_", " ")


def _read_wallet_columns() -> list:
    """Read only the fields needed for summaries and alert ranking."""
    import pyarrow.parquet as parquet

    available = set(parquet.read_schema(WALLET_SCORES).names)
    columns = [column for column in OVERVIEW_WALLET_COLUMNS if column in available]
    secondary = "tx_mean" if "tx_mean" in available else "total_txs"
    if secondary not in columns:
        columns.append(secondary)
    return columns


def _load_parquet(key: str, path) -> pd.DataFrame:
    if key not in _CACHE:
        _CACHE[key] = pd.read_parquet(path)
    return _CACHE[key]


def _load_wallet_features() -> pd.DataFrame:
    return _load_parquet("wallet_features", WALLET_SCORES).set_index("address", drop=False)


def _wallet_feature_row(address: str) -> pd.Series:
    wallets = _load_wallet_features()
    matches = wallets.loc[address]
    return matches.iloc[0] if isinstance(matches, pd.DataFrame) else matches


def _load_medians() -> pd.DataFrame:
    if "medians" not in _CACHE:
        wallets = _load_wallet_features()
        feature_cols = [
            c for c in wallets.columns
            if c not in ("address", "time_step", "class", "risk", "split")
        ]
        _CACHE["medians"] = wallets[wallets["class"] != "unknown"].groupby("class")[feature_cols].median()
    return _CACHE["medians"]


def _load():
    if "wallets" not in _CACHE:
        wallets = pd.read_parquet(WALLET_SCORES, columns=_read_wallet_columns())
        txs = pd.read_parquet(TX_SCORES)
        _CACHE["wallets"] = wallets
        _CACHE["wallet_threshold"] = choose_threshold(wallets, WALLET_ALERT_BUDGET, MIN_WALLET_THRESHOLD)
        _CACHE["tx_threshold"] = choose_threshold(txs, TX_ALERT_BUDGET)
        # Alerts are ranked by risk, then by a secondary signal that separates
        # wallets tied on risk: tx_mean (how pervasive the illicit exposure is)
        # when the scores carry it, otherwise transaction count. Peak risk
        # (tx_max) saturates at 1.0 for many alerts, so this secondary key is
        # what gives the severity bands information.
        secondary = "tx_mean" if "tx_mean" in wallets.columns else "total_txs"
        _CACHE["flagged"] = (
            wallets[(wallets["split"] != "train") & (wallets["risk"] >= _CACHE["wallet_threshold"]["threshold"])]
            .sort_values(["risk", secondary], ascending=False)
            .reset_index(drop=True)
        )
        # Severity by position in that ranked list (highest risk first)
        _CACHE["flagged"]["severity"] = _severity_bands(len(_CACHE["flagged"]))
        _CACHE["txs"] = txs
        _CACHE["txs_eval"] = txs[txs["split"] != "train"].sort_values("risk", ascending=False)
    return _CACHE


def _wallet_transactions(address: str) -> pd.DataFrame:
    """Transactions a wallet sent or received, with their model risk, newest first."""
    data = _load()
    wallet_tx = _load_parquet("wallet_tx", WALLET_TX_EDGES) if WALLET_TX_EDGES.exists() else pd.DataFrame(
        columns=["address", "txId", "direction"]
    )
    links = wallet_tx[wallet_tx["address"] == address]
    return (
        links.merge(data["txs"][["txId", "time_step", "risk", "class", "total_BTC"]], on="txId", how="left")
        .sort_values(["time_step", "risk"], ascending=False)
    )


def _top_contributions(address: str, n: int = 5) -> pd.DataFrame:
    expl = _load_parquet("explanations", WALLET_EXPLANATIONS)
    rows = expl[(expl["address"] == address) & (expl["contribution"] > 0)]
    return rows.sort_values("contribution", ascending=False).head(n)


def _feature_evidence(wallet: pd.Series, contributions: pd.DataFrame) -> list:
    medians = _load_medians()
    evidence = []
    for feature in contributions["feature"]:
        if feature in wallet.index and feature in medians.columns:
            evidence.append(
                f"{_pretty(feature)} = {_fmt(wallet[feature])} "
                f"(typical licit {_fmt(medians.loc['licit', feature])}, "
                f"typical illicit {_fmt(medians.loc['illicit', feature])})"
            )
    return evidence


def _reason(wallet: pd.Series, contributions: pd.DataFrame) -> str:
    risk_score = int(wallet["risk"] * 100)
    reason = (f"Wallet {wallet['address'][:10]}... has a risk score of {risk_score}/100 — "
              f"the highest risk among the transactions it took part in, from the Elliptic++ "
              f"transaction model (much stronger than the wallet model). ")
    if not contributions.empty:
        top = ", ".join(_pretty(f) for f in contributions["feature"].head(3))
        reason += f"The wallet model separately highlights these behavioural features: {top}. "
    if wallet["class"] == "unknown":
        reason += "This wallet has no ground-truth label: the score is a new prediction."
    else:
        reason += f"Ground-truth label: {wallet['class']} (held-out test wallet)."
    return reason


def get_overview_stats():
    data = _load()
    flagged = data["flagged"]
    # The KPI cards use the same per-alert severity as the alerts table
    levels = flagged["severity"]
    return {
        "total_transactions": len(data["txs"]),
        "total_wallets": len(data["wallets"]),
        "known_illicit": int((data["wallets"]["class"] == "illicit").sum()),
        "anomalous_entities": len(flagged),
        "high_risk_entities": int(levels.isin(["High", "Critical"]).sum()),
        "critical_alerts": int((levels == "Critical").sum()),
    }


def get_activity_timeline():
    """Flagged (held-out or unlabelled) transactions per Elliptic time step."""
    txs = _load()["txs_eval"]
    counts = txs[txs["risk"] >= _tx_threshold()].groupby("time_step").size()
    counts = counts.reindex(range(int(txs["time_step"].min()), int(txs["time_step"].max()) + 1), fill_value=0)
    return [_step(t) for t in counts.index], counts.tolist()


def _neighbours(column: str, other: str, transaction_ids: set | None = None) -> pd.Series:
    edges = _load_parquet("edges", TX_EDGES)
    if transaction_ids is not None:
        edges = edges[edges[column].isin(transaction_ids)]
    return edges.groupby(column, sort=False)[other].agg(list)


def _describe_links(tx_ids: list) -> str:
    if not tx_ids:
        return "outside dataset"
    shown = ", ".join(str(t) for t in tx_ids[:3])
    return shown + (f" (+{len(tx_ids) - 3} more)" if len(tx_ids) > 3 else "")


def get_transactions_df():
    data = _load()
    if "transactions_df" not in data:
        txs = data["txs_eval"].head(MAX_TRANSACTIONS)
        threshold = _tx_threshold()
        transaction_ids = set(txs["txId"])
        sources = _neighbours("target", "source", transaction_ids)
        destinations = _neighbours("source", "target", transaction_ids)
        data["transactions_df"] = pd.DataFrame({
            "Transaction ID": txs["txId"].astype(str),
            "Amount (BTC)": txs["total_BTC"],
            "Timestamp": txs["time_step"].map(_step),
            "Source": [_describe_links(sources.get(t, [])) for t in txs["txId"]],
            "Destination": [_describe_links(destinations.get(t, [])) for t in txs["txId"]],
            "Flag": ["Suspicious" if r >= threshold else "Normal" for r in txs["risk"]],
            "Risk (%)": (txs["risk"] * 100).round(1),
            "True Label": txs["class"],
        })
    return data["transactions_df"]


def get_tx_detail(tx_id: str):
    txs = _load()["txs"]
    row = txs[txs["txId"].astype(str) == str(tx_id)]
    if row.empty:
        return None
    row = row.iloc[0]
    tx_row = get_transactions_df()
    tx_row = tx_row[tx_row["Transaction ID"] == str(tx_id)]
    return {
        "transaction_id": str(row["txId"]),
        "amount_btc": row["total_BTC"],
        "timestamp": _step(row["time_step"]),
        "source": tx_row["Source"].iloc[0] if not tx_row.empty else "",
        "destination": tx_row["Destination"].iloc[0] if not tx_row.empty else "",
        "confirmations": "N/A",
        "fee_btc": row["fees"],
        "block_height": f"N/A (Elliptic time step {int(row['time_step'])})",
        "flag": "Suspicious" if row["risk"] >= _tx_threshold() else "Normal",
        "reason": (f"Predicted probability of illicit: {row['risk']:.2f} "
                   f"(true label: {row['class']}, {row['split']} set)"),
    }


def get_alerts_df(include_reasons: bool = True):
    data = _load()
    cache_key = "alerts_df" if include_reasons else "alerts_df_without_reasons"
    if cache_key not in data:
        flagged = data["flagged"]
        if include_reasons:
            expl = _load_parquet("explanations", WALLET_EXPLANATIONS)
            top_feature = (
                expl[expl["contribution"] > 0]
                .sort_values("contribution", ascending=False)
                .drop_duplicates("address")
                .set_index("address")["feature"]
            )
        else:
            top_feature = pd.Series(dtype=object)
        data[cache_key] = pd.DataFrame({
            "Rank": range(1, len(flagged) + 1),
            "Entity ID": flagged["address"],
            "Risk Score": (flagged["risk"] * 100).astype(int),
            "Risk Level": flagged["severity"].to_numpy(),
            "Main Reason": flagged["address"].map(top_feature).map(
                lambda f: f"Top factor: {_pretty(f)}" if isinstance(f, str) else "Risk score above threshold"
            ),
            "Full Entity ID": flagged["address"],
            "Timestamp": flagged["time_step"].map(_step),
        })
    return data[cache_key]


def _build_entity(
    flagged_wallet: pd.Series,
    timeline: pd.DataFrame | None = None,
    contributions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
) -> dict:
    addr = flagged_wallet["address"]
    wallet = _wallet_feature_row(addr).copy()
    wallet["severity"] = flagged_wallet["severity"]
    if contributions is None:
        contributions = _top_contributions(addr)
    if timeline is None:
        timeline = _load_parquet("timeline", WALLET_TIMELINE)
        timeline = timeline[timeline["address"] == addr]
    steps = timeline[timeline["address"] == addr].sort_values("time_step")
    if transactions is None:
        transactions = _wallet_transactions(addr)
    return {
        "entity_id": addr,
        "risk_score": int(wallet["risk"] * 100),
        "risk_level": wallet["severity"],
        "anomaly_score": wallet["risk"],
        "transaction_count": int(wallet["total_txs"]),
        "counterparty_count": int(wallet["transacted_w_address_total"]),
        "timeline": [
            {"timestamp": _step(r["time_step"]), "event": f"Active in time step {int(r['time_step'])}: model risk {r['risk']:.0%}"}
            for _, r in steps.iterrows()
        ],
        "network_associations": [
            {"type": "First block", "value": _fmt(wallet["first_block_appeared_in"])},
            {"type": "Last block", "value": _fmt(wallet["last_block_appeared_in"])},
            {"type": "Lifetime (blocks)", "value": _fmt(wallet["lifetime_in_blocks"])},
            {"type": "Time steps active", "value": _fmt(wallet["num_timesteps_appeared_in"])},
            {"type": "BTC transacted", "value": _fmt(wallet["btc_transacted_total"])},
            {"type": "Ground truth", "value": f"{wallet['class']} ({wallet['split']})"},
        ],
        "evidence": _feature_evidence(wallet, contributions),
        "top_features": [
            {"feature": _pretty(r["feature"]), "contribution": round(r["contribution"], 3)}
            for _, r in contributions.iterrows()
        ],
        "explanation": _reason(wallet, contributions),
        "transactions": [
            {
                "Transaction ID": str(r["txId"]),
                "Direction": r["direction"],
                "Time step": "" if pd.isna(r["time_step"]) else _step(r["time_step"]),
                "Amount (BTC)": r["total_BTC"],
                "Risk (%)": None if pd.isna(r["risk"]) else round(r["risk"] * 100, 1),
                "True Label": r["class"],
            }
            for _, r in transactions.iterrows()
        ],
    }


def get_entity(address: str):
    data = _load()
    if "entities" in data:
        return data["entities"].get(address)

    flagged_wallets = data["flagged"]
    match = flagged_wallets[flagged_wallets["address"] == address]
    if match.empty:
        return None

    profiles = _CACHE.setdefault("entity_profiles", {})
    if address not in profiles:
        profiles[address] = _build_entity(match.iloc[0])
    return profiles[address]


def get_entities():
    data = _load()
    if "entities" not in data:
        flagged = data["flagged"].head(MAX_ENTITIES)
        timeline = _load_parquet("timeline", WALLET_TIMELINE)
        timeline_groups = {address: rows for address, rows in timeline.groupby("address", sort=False)}
        explanations = _load_parquet("explanations", WALLET_EXPLANATIONS)
        positive = explanations[explanations["contribution"] > 0]
        contribution_groups = {address: rows for address, rows in positive.groupby("address", sort=False)}
        wallet_tx = (
            _load_parquet("wallet_tx", WALLET_TX_EDGES)
            if WALLET_TX_EDGES.exists()
            else pd.DataFrame(columns=["address", "txId", "direction"])
        )
        transactions = wallet_tx.merge(
            data["txs"][["txId", "time_step", "risk", "class", "total_BTC"]],
            on="txId",
            how="left",
        ).sort_values(["time_step", "risk"], ascending=False)
        transaction_groups = {address: rows for address, rows in transactions.groupby("address", sort=False)}
        data["entities"] = {
            wallet["address"]: _build_entity(
                wallet,
                timeline_groups.get(wallet["address"], pd.DataFrame(columns=timeline.columns)),
                contribution_groups.get(wallet["address"], pd.DataFrame(columns=explanations.columns)),
                transaction_groups.get(wallet["address"], pd.DataFrame(columns=transactions.columns)),
            )
            for _, wallet in flagged.iterrows()
        }
    return data["entities"]


def get_entity_ids():
    return _load()["flagged"].head(MAX_ENTITIES)["address"].tolist()


def _entity_graph(data):
    """Top alerted wallets, the transactions they took part in, and the flows between those."""
    wallets = data["flagged"].head(GRAPH_ENTITIES)
    wallet_tx = (
        _load_parquet("wallet_tx", WALLET_TX_EDGES)
        if WALLET_TX_EDGES.exists()
        else pd.DataFrame(columns=["address", "txId", "direction"])
    )
    links = wallet_tx[wallet_tx["address"].isin(wallets["address"])]
    links = links.groupby("address").head(MAX_TXS_PER_ENTITY)
    tx_ids = set(links["txId"])
    risk = data["txs"].set_index("txId")["risk"]

    nodes = [
        {"id": a, "type": "Wallet", "label": a[:6], "risk_score": int(r * 100)}
        for a, r in zip(wallets["address"], wallets["risk"])
    ]
    nodes += [
        {"id": str(t), "type": "Transaction", "label": str(t)[:6], "risk_score": int(risk.get(t, 0) * 100)}
        for t in tx_ids
    ]
    edge_list = [
        {"source": a, "target": str(t), "relation": "sent"} if d == "sent"
        else {"source": str(t), "target": a, "relation": "received"}
        for a, t, d in zip(links["address"], links["txId"], links["direction"])
    ]
    edges = _load_parquet("edges", TX_EDGES)
    flows = edges[edges["source"].isin(tx_ids) & edges["target"].isin(tx_ids)]
    edge_list += [{"source": str(s), "target": str(t), "relation": "funds"} for s, t in zip(flows["source"], flows["target"])]
    return nodes, edge_list


def _transaction_graph(data):
    """Transaction-flow subgraph around the highest-risk transactions."""
    txs, edges = data["txs"], _load_parquet("edges", TX_EDGES)
    seeds = set(data["txs_eval"].head(GRAPH_SEEDS)["txId"])
    sub = edges[edges["source"].isin(seeds) | edges["target"].isin(seeds)]
    risk = txs.set_index("txId")["risk"]
    node_ids = set(sub["source"]) | set(sub["target"]) | seeds
    nodes = [
        {"id": str(tx), "type": "Transaction", "label": str(tx)[:6], "risk_score": int(risk.get(tx, 0) * 100)}
        for tx in node_ids
    ]
    edge_list = [{"source": str(s), "target": str(t), "relation": "funds"} for s, t in zip(sub["source"], sub["target"])]
    return nodes, edge_list


def get_graph_data():
    """Graph around the top alerted wallets when wallet-transaction links exist,
    otherwise the transaction-flow subgraph around the highest-risk transactions."""
    data = _load()
    wallet_tx = (
        _load_parquet("wallet_tx", WALLET_TX_EDGES)
        if WALLET_TX_EDGES.exists()
        else pd.DataFrame(columns=["address", "txId", "direction"])
    )
    if not wallet_tx.empty:
        nodes, edge_list = _entity_graph(data)
    else:
        nodes, edge_list = _transaction_graph(data)

    G = nx.Graph()
    G.add_nodes_from(n["id"] for n in nodes)
    G.add_edges_from((e["source"], e["target"]) for e in edge_list)

    components = {n: i for i, comp in enumerate(nx.connected_components(G)) for n in comp}
    deg = dict(G.degree())
    stats = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "clusters_detected": nx.number_connected_components(G),
        "avg_degree": round(sum(deg.values()) / len(deg), 2) if deg else 0.0,
    }
    cent = nx.degree_centrality(G)
    cent_table = pd.DataFrame([
        {"Rank": i + 1, "Node": n, "Centrality Score": round(c, 2), "Cluster": components[n]}
        for i, (n, c) in enumerate(sorted(cent.items(), key=lambda x: x[1], reverse=True)[:10])
    ])
    return nodes, edge_list, stats, cent_table


def get_scoreboard():
    """Risk per time step for the highest-risk flagged wallets active in several time steps."""
    data = _load()
    timeline = _load_parquet("timeline", WALLET_TIMELINE)
    active = timeline.groupby("address")["time_step"].nunique()
    flagged = data["flagged"]
    candidates = flagged[flagged["address"].map(active).fillna(0) >= 3].head(6)
    scoreboard = {}
    for addr in candidates["address"]:
        steps = timeline[timeline["address"] == addr].sort_values("time_step")
        scoreboard[f"Wallet ({addr[:6]})"] = [
            {"time": int(r["time_step"]), "score": int(r["risk"] * 100)} for _, r in steps.iterrows()
        ]
    return scoreboard


def get_explainability_data():
    data = _load()
    explainability = {}
    for _, flagged_wallet in data["flagged"].head(MAX_EXPLAINED).iterrows():
        wallet = _wallet_feature_row(flagged_wallet["address"]).copy()
        wallet["severity"] = flagged_wallet["severity"]
        contributions = _top_contributions(wallet["address"])
        explainability[wallet["address"]] = {
            "anomaly_score": wallet["risk"],
            "risk_score": int(wallet["risk"] * 100),
            "top_features": [
                {"feature": _pretty(r["feature"]), "contribution": round(r["contribution"], 3), "direction": "increases"}
                for _, r in contributions.iterrows()
            ],
            "evidence": _feature_evidence(wallet, contributions)
                        + [f"Ground-truth label: {wallet['class']} ({wallet['split']} set)"],
            "human_reason": _reason(wallet, contributions),
        }
    return explainability
