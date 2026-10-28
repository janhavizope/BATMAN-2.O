"""
Elliptic++ data source for the Streamlit pages.

Serves the same functions as data_manager (overview stats, transactions,
alerts, entities, graph, scoreboard, explainability) from the risk scores
written by scripts/score_wallets.py and scripts/score_transactions.py.

Only wallets / transactions the models did not train on (test and unknown
splits) are presented as predictions. Entities are wallets; the transaction
graph comes from the Elliptic transaction flow edge list.
"""

import networkx as nx
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "elliptic_plus"
WALLET_SCORES = DATA_DIR / "wallet_scores.parquet"
WALLET_TIMELINE = DATA_DIR / "wallet_timeline.parquet"
WALLET_EXPLANATIONS = DATA_DIR / "wallet_explanations.parquet"
TX_SCORES = DATA_DIR / "tx_scores.parquet"
TX_EDGES = DATA_DIR / "tx_edges.parquet"

FLAG_THRESHOLD = 0.5
MAX_TRANSACTIONS = 10_000
MAX_ENTITIES = 500
MAX_EXPLAINED = 50
GRAPH_SEEDS = 40

_CACHE = {}


def is_available() -> bool:
    return all(p.exists() for p in (WALLET_SCORES, WALLET_TIMELINE, WALLET_EXPLANATIONS, TX_SCORES, TX_EDGES))


def _risk_level(risk: float) -> str:
    return "Critical" if risk > 0.8 else "High" if risk > 0.6 else "Medium" if risk >= FLAG_THRESHOLD else "Low"


def _step(time_step) -> str:
    return f"Step-{int(time_step):02d}"


def _fmt(value) -> str:
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.4g}"


def _pretty(feature: str) -> str:
    return feature.replace("_", " ")


def _load():
    if "wallets" not in _CACHE:
        wallets = pd.read_parquet(WALLET_SCORES)
        txs = pd.read_parquet(TX_SCORES)
        _CACHE["wallets"] = wallets
        _CACHE["flagged"] = (
            wallets[(wallets["split"] != "train") & (wallets["risk"] >= FLAG_THRESHOLD)]
            .sort_values(["risk", "total_txs"], ascending=False)
            .reset_index(drop=True)
        )
        feature_cols = [c for c in wallets.columns if c not in ("address", "time_step", "class", "risk", "split")]
        _CACHE["medians"] = wallets[wallets["class"] != "unknown"].groupby("class")[feature_cols].median()
        _CACHE["txs"] = txs
        _CACHE["txs_eval"] = txs[txs["split"] != "train"].sort_values("risk", ascending=False)
        _CACHE["edges"] = pd.read_parquet(TX_EDGES)
        _CACHE["timeline"] = pd.read_parquet(WALLET_TIMELINE)
        _CACHE["explanations"] = pd.read_parquet(WALLET_EXPLANATIONS)
    return _CACHE


def _top_contributions(address: str, n: int = 5) -> pd.DataFrame:
    expl = _load()["explanations"]
    rows = expl[(expl["address"] == address) & (expl["contribution"] > 0)]
    return rows.sort_values("contribution", ascending=False).head(n)


def _feature_evidence(wallet: pd.Series, contributions: pd.DataFrame) -> list:
    medians = _load()["medians"]
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
    reason = (f"Wallet {wallet['address'][:10]}... has a risk score of {risk_score}/100 "
              f"from the Elliptic++ Random Forest wallet model. ")
    if not contributions.empty:
        top = ", ".join(_pretty(f) for f in contributions["feature"].head(3))
        reason += f"The features that pushed its score up the most are: {top}. "
    if wallet["class"] == "unknown":
        reason += "This wallet has no ground-truth label: the score is a new prediction."
    else:
        reason += f"Ground-truth label: {wallet['class']} (held-out test wallet)."
    return reason


def get_overview_stats():
    data = _load()
    flagged = data["flagged"]
    return {
        "total_transactions": len(data["txs"]),
        "total_wallets": len(data["wallets"]),
        "known_illicit": int((data["wallets"]["class"] == "illicit").sum()),
        "anomalous_entities": len(flagged),
        "high_risk_entities": int((flagged["risk"] > 0.6).sum()),
        "critical_alerts": int((flagged["risk"] > 0.8).sum()),
    }


def get_activity_timeline():
    """Flagged (held-out or unlabelled) transactions per Elliptic time step."""
    txs = _load()["txs_eval"]
    counts = txs[txs["risk"] >= FLAG_THRESHOLD].groupby("time_step").size()
    counts = counts.reindex(range(int(txs["time_step"].min()), int(txs["time_step"].max()) + 1), fill_value=0)
    return [_step(t) for t in counts.index], counts.tolist()


def _neighbours(column: str, other: str) -> pd.Series:
    edges = _load()["edges"]
    return edges.groupby(column)[other].apply(list)


def _describe_links(tx_ids: list) -> str:
    if not tx_ids:
        return "outside dataset"
    shown = ", ".join(str(t) for t in tx_ids[:3])
    return shown + (f" (+{len(tx_ids) - 3} more)" if len(tx_ids) > 3 else "")


def get_transactions_df():
    data = _load()
    if "transactions_df" not in data:
        txs = data["txs_eval"].head(MAX_TRANSACTIONS)
        sources = _neighbours("target", "source")
        destinations = _neighbours("source", "target")
        data["transactions_df"] = pd.DataFrame({
            "Transaction ID": txs["txId"].astype(str),
            "Amount (BTC)": txs["total_BTC"],
            "Timestamp": txs["time_step"].map(_step),
            "Source": [_describe_links(sources.get(t, [])) for t in txs["txId"]],
            "Destination": [_describe_links(destinations.get(t, [])) for t in txs["txId"]],
            "Flag": ["Suspicious" if r >= FLAG_THRESHOLD else "Normal" for r in txs["risk"]],
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
        "flag": "Suspicious" if row["risk"] >= FLAG_THRESHOLD else "Normal",
        "reason": (f"Predicted probability of illicit: {row['risk']:.2f} "
                   f"(true label: {row['class']}, {row['split']} set)"),
    }


def get_alerts_df():
    data = _load()
    if "alerts_df" not in data:
        flagged = data["flagged"]
        expl = data["explanations"]
        top_feature = (
            expl[expl["contribution"] > 0]
            .sort_values("contribution", ascending=False)
            .drop_duplicates("address")
            .set_index("address")["feature"]
        )
        data["alerts_df"] = pd.DataFrame({
            "Rank": range(1, len(flagged) + 1),
            "Entity ID": flagged["address"],
            "Risk Score": (flagged["risk"] * 100).astype(int),
            "Risk Level": flagged["risk"].map(_risk_level),
            "Main Reason": flagged["address"].map(top_feature).map(
                lambda f: f"Top factor: {_pretty(f)}" if isinstance(f, str) else "Risk score above threshold"
            ),
            "Full Entity ID": flagged["address"],
            "Timestamp": flagged["time_step"].map(_step),
        })
    return data["alerts_df"]


def get_entities():
    data = _load()
    if "entities" not in data:
        timeline = data["timeline"]
        entities = {}
        for _, wallet in data["flagged"].head(MAX_ENTITIES).iterrows():
            addr = wallet["address"]
            contributions = _top_contributions(addr)
            steps = timeline[timeline["address"] == addr].sort_values("time_step")
            entities[addr] = {
                "entity_id": addr,
                "risk_score": int(wallet["risk"] * 100),
                "risk_level": _risk_level(wallet["risk"]),
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
                    {"feature": r["feature"], "contribution": round(r["contribution"], 3)}
                    for _, r in contributions.iterrows()
                ],
                "explanation": _reason(wallet, contributions),
            }
        data["entities"] = entities
    return data["entities"]


def get_entity_ids():
    return list(get_entities().keys())


def get_graph_data():
    data = _load()
    txs, edges = data["txs"], data["edges"]
    seeds = set(data["txs_eval"].head(GRAPH_SEEDS)["txId"])
    sub = edges[edges["source"].isin(seeds) | edges["target"].isin(seeds)]
    risk = txs.set_index("txId")["risk"]

    G = nx.Graph()
    node_ids = set(sub["source"]) | set(sub["target"]) | seeds
    nodes = []
    for tx in node_ids:
        score = int(risk.get(tx, 0) * 100)
        G.add_node(str(tx))
        nodes.append({"id": str(tx), "type": "Transaction", "label": str(tx)[:6], "risk_score": score})
    edge_list = []
    for s, t in zip(sub["source"], sub["target"]):
        G.add_edge(str(s), str(t))
        edge_list.append({"source": str(s), "target": str(t), "relation": "funds"})

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
    timeline = data["timeline"]
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
    for _, wallet in data["flagged"].head(MAX_EXPLAINED).iterrows():
        contributions = _top_contributions(wallet["address"])
        explainability[wallet["address"]] = {
            "anomaly_score": wallet["risk"],
            "risk_score": int(wallet["risk"] * 100),
            "top_features": [
                {"feature": r["feature"], "contribution": round(r["contribution"], 3), "direction": "increases"}
                for _, r in contributions.iterrows()
            ],
            "evidence": _feature_evidence(wallet, contributions)
                        + [f"Ground-truth label: {wallet['class']} ({wallet['split']} set)"],
            "human_reason": _reason(wallet, contributions),
        }
    return explainability
