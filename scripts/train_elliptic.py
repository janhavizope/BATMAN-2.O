"""
train_elliptic.py
-----------------
Trains a Random Forest classifier on the Elliptic Bitcoin dataset
(https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) to detect
illicit Bitcoin transactions.

Steps:
    1. Load features + labels from data/elliptic/
    2. Keep only labelled transactions (class 1 = illicit, 2 = licit;
       "unknown" rows have no ground truth and are dropped)
    3. Stratified 80/20 train/test split
    4. Train RandomForestClassifier (class_weight="balanced" because only
       ~10% of labelled transactions are illicit)
    5. Evaluate on the held-out 20%: precision/recall/F1, confusion matrix,
       ROC-AUC
    6. Save model + metrics to models/ and outputs/evaluation/
    7. Save graphs to outputs/graphs/elliptic/

Run (from the project root):
    python scripts/train_elliptic.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "elliptic"
FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = DATA_DIR / "elliptic_txs_classes.csv"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "elliptic_rf_model.joblib"
METRICS_FILE = ROOT / "outputs" / "evaluation" / "elliptic_metrics.json"
GRAPHS_DIR = ROOT / "outputs" / "graphs" / "elliptic"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_data():
    # Features file has no header: txId, time_step, then 165 anonymised features
    features = pd.read_csv(FEATURES_FILE, header=None)
    n_extra = features.shape[1] - 2
    features.columns = ["txId", "time_step"] + [f"f{i}" for i in range(1, n_extra + 1)]

    classes = pd.read_csv(CLASSES_FILE)
    df = features.merge(classes, on="txId")
    df = df[df["class"] != "unknown"]

    X = df.drop(columns=["txId", "class"])
    y = (df["class"] == "1").astype(int)  # 1 = illicit, 0 = licit
    return X, y


def save_graphs(y, y_test, y_pred, y_prob, cm, report, importances, graphs_dir=GRAPHS_DIR):
    graphs_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    labels = ["licit", "illicit"]

    # 1. Class distribution of the labelled data
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts().sort_index()
    ax.bar(labels, counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_title("Labelled transactions by class")
    ax.set_ylabel("Transactions")
    fig.tight_layout()
    fig.savefig(graphs_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (20% test set)")
    fig.tight_layout()
    fig.savefig(graphs_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 3. ROC curve
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, name="Random Forest", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(graphs_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    # 4. Precision-recall curve (more informative than ROC for imbalanced data)
    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, name="Random Forest", ax=ax)
    ax.set_title("Precision-recall curve")
    fig.tight_layout()
    fig.savefig(graphs_dir / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    # 5. Precision / recall / F1 per class
    metrics = pd.DataFrame(
        {c: [report[c]["precision"], report[c]["recall"], report[c]["f1-score"]] for c in labels},
        index=["precision", "recall", "f1-score"],
    )
    ax = metrics.plot(kind="bar", figsize=(6, 4), color=["#4C72B0", "#C44E52"], rot=0)
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper center", ncol=2)
    ax.set_title("Metrics per class (20% test set)")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    ax.figure.tight_layout()
    ax.figure.savefig(graphs_dir / "metrics_per_class.png", dpi=150)
    plt.close(ax.figure)

    # 6. Top 15 most important features
    fig, ax = plt.subplots(figsize=(6, 5))
    importances.sort_values(ascending=False).head(15)[::-1].plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Top 15 feature importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(graphs_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    # 7. Predicted illicit probability, split by actual class
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = [i / 50 for i in range(51)]
    ax.hist(y_prob[y_test.values == 0], bins=bins, alpha=0.7, color="#4C72B0", label="actual licit")
    ax.hist(y_prob[y_test.values == 1], bins=bins, alpha=0.7, color="#C44E52", label="actual illicit")
    ax.set_yscale("log")
    ax.axvline(0.5, color="k", linestyle="--", label="decision threshold")
    ax.legend()
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Predicted illicit probability (test set)")
    ax.set_xlabel("Predicted probability of illicit")
    fig.tight_layout()
    fig.savefig(graphs_dir / "probability_distribution.png", dpi=150)
    plt.close(fig)

    print(f"Graphs saved to {graphs_dir}")


def main():
    print(f"Loading data from {DATA_DIR} ...")
    X, y = load_data()
    print(f"Labelled transactions: {len(X)}  |  features: {X.shape[1]}")
    print(f"Illicit: {y.sum()}  |  Licit: {(y == 0).sum()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows (80%)  |  Test: {len(X_test)} rows (20%)\n")

    print("Training Random Forest ...")
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["licit", "illicit"], output_dict=True
    )

    print("\n=== Results on the 20% test set ===")
    print(classification_report(y_test, y_pred, target_names=["licit", "illicit"]))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print("Confusion matrix (rows = actual, cols = predicted) [licit, illicit]:")
    print(cm)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top = importances.sort_values(ascending=False).head(10)
    print("\nTop 10 features:")
    print(top.to_string())

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns)}, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps({
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "top_features": top.to_dict(),
    }, indent=2))
    print(f"Metrics saved to {METRICS_FILE}")

    save_graphs(y, y_test, y_pred, y_prob, cm, report, importances)


if __name__ == "__main__":
    main()
