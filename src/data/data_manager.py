import pandas as pd
import numpy as np
import joblib
import networkx as nx
from pathlib import Path

from src.data import elliptic_source

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
ML_DIR = ROOT_DIR / "src" / "core" / "backend" / "ml"
DATA_FILE = ML_DIR / "synthetic_bitcoin_transactions.csv"
MODEL_FILE = ML_DIR / "illicit_tx_rf_model.joblib"

_CACHE = {}

def load_model():
    if "model" not in _CACHE:
        _CACHE["model"] = joblib.load(MODEL_FILE)
    return _CACHE["model"]

def _legacy_available() -> bool:
    """Check if the transaction database or synthetic CSV is available."""
    db_path = ROOT_DIR / "bitcoin_transactions.db"
    return db_path.exists() or DATA_FILE.exists()

def _use_elliptic() -> bool:
    """Fall back to the Elliptic++ risk scores when no transaction dataset is present."""
    return not _legacy_available() and elliptic_source.is_available()

def is_data_available() -> bool:
    """Check if any transaction dataset or database is available."""
    return _legacy_available() or elliptic_source.is_available()

def load_and_predict_data():
    if "data" not in _CACHE:
        db_path = ROOT_DIR / "bitcoin_transactions.db"
        if db_path.exists():
            import sqlite3
            conn = sqlite3.connect(db_path)
            df = pd.read_sql("SELECT * FROM transactions", conn)
            conn.close()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        elif DATA_FILE.exists():
            df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
        else:
            # Graceful empty state when no dataset is bundled
            _CACHE["data"] = (pd.DataFrame(), None)
            return _CACHE["data"]
            
        # Feature engineering as per train_model.py
        df["fee_to_amount_ratio"] = df["fee_btc"] / df["amount_btc"].replace(0, np.nan)
        df["fee_to_amount_ratio"] = df["fee_to_amount_ratio"].fillna(0)
    
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
    
        df["fan_ratio"] = df["num_outputs"] / df["num_inputs"].replace(0, np.nan)
        df["fan_ratio"] = df["fan_ratio"].fillna(0)
    
        df["ip_first_octet"] = df["ip_address"].str.extract(r'(\d+)')[0].fillna(0).astype(int)
        
        # Load model
        model_dict = load_model()
        model = model_dict["model"]
        feature_cols = model_dict["feature_cols"]
        
        X = df[feature_cols]
        
        # Predictions
        df["pred_illicit"] = model.predict(X)
        df["pred_proba"] = model.predict_proba(X)[:, 1]
        _CACHE["data"] = (df, model_dict)
    
    return _CACHE["data"]

def get_overview_stats():
    if _use_elliptic():
        return elliptic_source.get_overview_stats()
    df, _ = load_and_predict_data()
    if df.empty:
        return {
            "total_transactions": 0,
            "total_wallets": 0,
            "total_ips": 0,
            "anomalous_entities": 0,
            "high_risk_entities": 0,
            "critical_alerts": 0,
        }
    total_tx = len(df)
    total_wallets = len(set(df["input_address"]).union(set(df["output_address"])))
    total_ips = df["ip_address"].nunique()
    
    critical_alerts = len(df[df["pred_proba"] > 0.8])
    high_risk_entities = len(df[df["pred_proba"] > 0.6]["input_address"].unique())
    anomalous_entities = len(df[df["pred_illicit"] == 1]["input_address"].unique())
    
    return {
        "total_transactions": total_tx,
        "total_wallets": total_wallets,
        "total_ips": total_ips,
        "anomalous_entities": anomalous_entities,
        "high_risk_entities": high_risk_entities,
        "critical_alerts": critical_alerts,
    }

def get_transactions_df():
    if _use_elliptic():
        return elliptic_source.get_transactions_df()
    df, _ = load_and_predict_data()
    if df.empty:
        return pd.DataFrame(columns=["Transaction ID", "Amount (BTC)", "Timestamp", "Source", "Destination", "Flag"])
    tx_df = pd.DataFrame({
        "Transaction ID": df["txid"].astype(str),
        "Amount (BTC)": df["amount_btc"],
        "Timestamp": df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Source": df["input_address"],
        "Destination": df["output_address"],
        "Flag": np.where(df["pred_illicit"] == 1, "Suspicious", "Normal")
    })
    return tx_df

def get_activity_timeline():
    """Return (labels, counts) for the Overview activity chart."""
    if _use_elliptic():
        return elliptic_source.get_activity_timeline()
    txns_df = get_transactions_df()
    if txns_df.empty:
        return ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"], [0, 0, 0, 0, 0, 0]
    ts_series = pd.to_datetime(txns_df["Timestamp"], errors="coerce").dropna()
    hour_counts = ts_series.dt.strftime("%H:00").value_counts().sort_index()
    return hour_counts.index.tolist(), hour_counts.values.tolist()

def get_transactions():
    """Returns a list of dicts for the PySide6 app"""
    df = get_transactions_df()
    return df.to_dict('records')

def get_tx_detail(tx_id: str):
    if _use_elliptic():
        return elliptic_source.get_tx_detail(tx_id)
    df, _ = load_and_predict_data()
    if df.empty:
        return None
    row = df[df["txid"] == tx_id]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "transaction_id": row["txid"],
        "amount_btc": row["amount_btc"],
        "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        "source": row["input_address"],
        "destination": row["output_address"],
        "confirmations": 0,
        "fee_btc": row["fee_btc"],
        "block_height": row["block_height"],
        "flag": "Suspicious" if row["pred_illicit"] == 1 else "Normal",
        "reason": f"Predicted probability of illicit: {row['pred_proba']:.2f}"
    }

def get_alerts_df():
    if _use_elliptic():
        return elliptic_source.get_alerts_df()
    df, _ = load_and_predict_data()
    if df.empty:
        return pd.DataFrame(columns=["Rank", "Entity ID", "Risk Score", "Risk Level", "Main Reason", "Full Entity ID", "Timestamp"])
    suspicious = df[df["pred_illicit"] == 1].copy()
    suspicious = suspicious.sort_values(by="pred_proba", ascending=False)
    
    alerts = []
    for rank, (idx, row) in enumerate(suspicious.iterrows(), 1):
        risk_level = "Critical" if row["pred_proba"] > 0.8 else "High" if row["pred_proba"] > 0.6 else "Medium"
        alerts.append({
            "Rank": rank,
            "Entity ID": row["txid"][:8] + "...",
            "Risk Score": int(row["pred_proba"] * 100),
            "Risk Level": risk_level,
            "Main Reason": f"High fan ratio ({row['fan_ratio']:.2f})" if row['fan_ratio'] > 5 else "Unusual transaction pattern",
            "Full Entity ID": row["txid"],
            "Timestamp": row["timestamp"]
        })
    return pd.DataFrame(alerts) if alerts else pd.DataFrame(columns=["Rank", "Entity ID", "Risk Score", "Risk Level", "Main Reason", "Full Entity ID", "Timestamp"])

def get_alerts():
    """Returns a list of dicts for the PySide6 app"""
    df = get_alerts_df()
    return df.to_dict('records')

def get_entities():
    if _use_elliptic():
        return elliptic_source.get_entities()
    df, _ = load_and_predict_data()
    if df.empty:
        return {}
    
    entities = {}
    grouped = df.groupby("input_address")
    
    count = 0
    for addr, group in grouped:
        if count >= 500:
            break
        count += 1
        avg_proba = group["pred_proba"].mean()
        risk_score = int(avg_proba * 100)
        risk_level = "Critical" if avg_proba > 0.8 else "High" if avg_proba > 0.6 else "Medium" if avg_proba > 0.4 else "Low"
        
        entities[addr] = {
            "entity_id": addr,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "anomaly_score": avg_proba,
            "transaction_count": len(group),
            "counterparty_count": group["output_address"].nunique(),
            "timeline": [{"timestamp": r["timestamp"].strftime("%Y-%m-%d %H:%M:%S"), "event": f"Sent {r['amount_btc']:.4f} BTC to {r['output_address'][:8]}..."} for _, r in group.head(5).iterrows()],
            "network_associations": [
                {"type": "IP", "value": group["ip_address"].iloc[0]},
                {"type": "ASN", "value": "Unknown"}
            ],
            "evidence": [f"Average predicted probability: {avg_proba:.2f}", f"Sent {len(group)} transactions"],
            "top_features": [
                {"feature": "fan_ratio", "contribution": round(group["fan_ratio"].mean(), 2)},
                {"feature": "amount_btc", "contribution": round(group["amount_btc"].mean(), 4)}
            ],
            "explanation": f"Wallet {addr[:8]}... has a risk score of {risk_score}."
        }
    return entities

def get_entity_ids():
    return list(get_entities().keys())

def get_graph_data():
    if _use_elliptic():
        return elliptic_source.get_graph_data()
    df, _ = load_and_predict_data()
    if df.empty:
        empty_stats = {"total_nodes": 0, "total_edges": 0, "clusters_detected": 0, "avg_degree": 0.0}
        empty_cent = pd.DataFrame(columns=["Rank", "Node", "Centrality Score", "Cluster"])
        return [], [], empty_stats, empty_cent
    
    subset = df.sort_values(by="pred_proba", ascending=False).head(200)
    
    nodes = []
    edges = []
    G = nx.Graph()
    
    for _, row in subset.iterrows():
        tx_id = row["txid"]
        src = row["input_address"]
        dst = row["output_address"]
        ip = row["ip_address"]
        
        G.add_node(tx_id, type="Transaction", label=tx_id[:6], risk_score=int(row["pred_proba"]*100))
        G.add_node(src, type="Wallet", label=src[:6], risk_score=int(row["pred_proba"]*100))
        G.add_node(dst, type="Wallet", label=dst[:6], risk_score=0)
        G.add_node(ip, type="IP", label=ip, risk_score=int(row["pred_proba"]*100))
        
        G.add_edge(src, tx_id)
        G.add_edge(tx_id, dst)
        G.add_edge(ip, src)
        
        nodes.append({"id": tx_id, "type": "Transaction", "label": tx_id[:6], "risk_score": int(row["pred_proba"]*100)})
        nodes.append({"id": src, "type": "Wallet", "label": src[:6], "risk_score": int(row["pred_proba"]*100)})
        nodes.append({"id": dst, "type": "Wallet", "label": dst[:6], "risk_score": 0})
        nodes.append({"id": ip, "type": "IP", "label": ip, "risk_score": int(row["pred_proba"]*100)})
        
        edges.append({"source": src, "target": tx_id, "relation": "sent"})
        edges.append({"source": tx_id, "target": dst, "relation": "received"})
        edges.append({"source": ip, "target": src, "relation": "associated"})
        
    unique_nodes = list({n["id"]: n for n in nodes}.values())
    unique_edges = list({f"{e['source']}-{e['target']}": e for e in edges}.values())
    
    deg = dict(G.degree())
    avg_deg = sum(deg.values()) / len(deg) if deg else 0
    
    stats = {
        "total_nodes": len(unique_nodes),
        "total_edges": len(unique_edges),
        "clusters_detected": nx.number_connected_components(G),
        "avg_degree": round(avg_deg, 2)
    }
    
    cent = nx.degree_centrality(G)
    cent_table = pd.DataFrame([
        {"Rank": i+1, "Node": n, "Centrality Score": round(c, 2), "Cluster": 0}
        for i, (n, c) in enumerate(sorted(cent.items(), key=lambda x: x[1], reverse=True)[:10])
    ])
    
    return unique_nodes, unique_edges, stats, cent_table

def get_scoreboard():
    if _use_elliptic():
        return elliptic_source.get_scoreboard()
    df, _ = load_and_predict_data()
    if df.empty:
        return {}
    
    # 1. Find top 10 most anomalous entities (by average pred_proba)
    grouped = df.groupby("input_address")["pred_proba"].mean().sort_values(ascending=False)
    top_10 = grouped.head(10).index.tolist()
    
    # 2. Filter dataset to just those 10
    subset = df[df["input_address"].isin(top_10)].copy()
    
    # 3. Create a Year-Month column for the timeline
    subset["month"] = subset["timestamp"].dt.strftime("%Y-%m")
    
    # 4. Get all unique months sorted to ensure alignment across lines
    all_months = sorted(df["timestamp"].dt.strftime("%Y-%m").unique())
    if not all_months:
        return {}
        
    scoreboard = {}
    for addr in top_10:
        addr_df = subset[subset["input_address"] == addr]
        monthly_counts = addr_df.groupby("month").size().to_dict()
        
        cumulative_score = 0
        timeline = []
        for m in all_months:
            # We add risk-weighted volume over time
            cumulative_score += monthly_counts.get(m, 0) * int(grouped[addr] * 100)
            timeline.append({"time": m, "score": cumulative_score})
            
        scoreboard[f"Wallet ({addr[:6]})"] = timeline
        
    return scoreboard

def get_explainability_data():
    if _use_elliptic():
        return elliptic_source.get_explainability_data()
    df, model_dict = load_and_predict_data()
    if df.empty:
        return {}
    
    # Find top anomalous entities (by average pred_proba)
    grouped = df.groupby("input_address")
    
    explainability = {}
    
    # Just take the top 50 suspicious wallets
    top_wallets = grouped["pred_proba"].mean().sort_values(ascending=False).head(50).index.tolist()
    
    for addr in top_wallets:
        group = df[df["input_address"] == addr]
        avg_proba = group["pred_proba"].mean()
        risk_score = int(avg_proba * 100)
        
        # Calculate some simple feature contributions based on the group's mean vs overall mean
        fan_ratio_mean = group["fan_ratio"].mean()
        fee_ratio_mean = group["fee_to_amount_ratio"].mean()
        amount_mean = group["amount_btc"].mean()
        
        top_features = []
        if fan_ratio_mean > df["fan_ratio"].mean():
            top_features.append({"feature": "fan_ratio", "contribution": round(fan_ratio_mean, 2), "direction": "increases"})
        if fee_ratio_mean > df["fee_to_amount_ratio"].mean():
            top_features.append({"feature": "fee_to_amount_ratio", "contribution": round(fee_ratio_mean, 2), "direction": "increases"})
        if amount_mean > df["amount_btc"].mean():
            top_features.append({"feature": "amount_btc", "contribution": round(amount_mean, 4), "direction": "increases"})
            
        if not top_features:
             top_features.append({"feature": "unusual_pattern", "contribution": 0.5, "direction": "increases"})
             
        evidence = [
            f"Average predicted probability of illicit activity: {avg_proba:.2f}",
            f"Executed {len(group)} transactions",
            f"Interacted with {group['output_address'].nunique()} unique counterparties",
        ]
        
        human_reason = f"Wallet {addr[:8]}... is flagged with a risk score of {risk_score} due to anomalous transaction patterns identified by the Isolation Forest model. "
        if fan_ratio_mean > 2:
             human_reason += "It exhibits a high fan-out ratio, suggesting possible money laundering or peeling chain behaviour. "
             
        explainability[addr] = {
            "anomaly_score": avg_proba,
            "risk_score": risk_score,
            "top_features": top_features,
            "evidence": evidence,
            "human_reason": human_reason
        }
        
    return explainability
