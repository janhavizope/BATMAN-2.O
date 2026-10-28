"""
Model Results Page
------------------
Shows how the supervised Random Forest models trained on the Elliptic and
Elliptic++ datasets performed on their held-out 20% test sets:
- Model selector
- 4 Metric Cards: ACCURACY, PRECISION, RECALL (illicit class) and ROC-AUC
- Dataset / split summary
- Evaluation graphs produced by scripts/train_elliptic*.py
"""

import json
from pathlib import Path

import streamlit as st
from src.web_frontend.components import render_header, render_metric_card

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = ROOT / "outputs" / "evaluation"
GRAPHS_DIR = ROOT / "outputs" / "graphs"

MODELS = {
    "Elliptic++ Wallets": ("elliptic_plus_wallets_metrics.json", GRAPHS_DIR / "elliptic_plus" / "wallets",
                           "Classifies wallet addresses. Split by wallet: no wallet is in both train and test."),
    "Elliptic++ Transactions": ("elliptic_plus_txs_metrics.json", GRAPHS_DIR / "elliptic_plus" / "txs",
                                "Classifies transactions using 183 features (local, aggregate and augmented)."),
    "Elliptic Transactions": ("elliptic_metrics.json", GRAPHS_DIR / "elliptic",
                              "Classifies transactions using the original 166 Elliptic features."),
}

GRAPHS = [
    ("confusion_matrix.png", "CONFUSION MATRIX"),
    ("roc_curve.png", "ROC CURVE"),
    ("metrics_per_class.png", "METRICS PER CLASS"),
    ("feature_importance.png", "FEATURE IMPORTANCE"),
    ("precision_recall_curve.png", "PRECISION-RECALL CURVE"),
    ("probability_distribution.png", "PREDICTED PROBABILITY"),
    ("class_distribution.png", "CLASS DISTRIBUTION"),
]


def _section_label(text: str) -> None:
    st.markdown(
        "<div style='color:#b33939; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; "
        f"font-weight:bold; letter-spacing:1px; margin:14px 0 8px 0;'>{text}</div>",
        unsafe_allow_html=True,
    )


def render_model_results() -> None:
    render_header("MODEL RESULTS")

    model_name = st.selectbox("Model", options=list(MODELS), index=0)
    metrics_file, graphs_dir, description = MODELS[model_name]

    metrics_path = EVAL_DIR / metrics_file
    if not metrics_path.exists():
        st.info("This model has not been trained yet. Run scripts/train_elliptic_plus.py "
                "(or scripts/train_elliptic.py) to generate its results.")
        return
    metrics = json.loads(metrics_path.read_text())
    illicit = metrics["classification_report"]["illicit"]

    st.caption(description)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("ACCURACY", f"{metrics['accuracy']:.2%}")
    with c2:
        render_metric_card("PRECISION (ILLICIT)", f"{illicit['precision']:.2%}")
    with c3:
        render_metric_card("RECALL (ILLICIT)", f"{illicit['recall']:.2%}")
    with c4:
        render_metric_card("ROC-AUC", f"{metrics['roc_auc']:.4f}")

    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    st.markdown(
        "<div style='background-color:#0d0808; border:1px solid #4a1111; border-left:3px solid #b33939; "
        "border-radius:4px; padding:10px 14px; margin:10px 0; font-family:\"JetBrains Mono\", Consolas, monospace; "
        "font-size:11px; color:#d4d4d4; line-height:1.6;'>"
        f"<strong style='color:#ff4d4d;'>[80/20 SPLIT]</strong> "
        f"Trained on {metrics['train_rows']:,} rows, tested on {metrics['test_rows']:,} unseen rows.<br>"
        f"Caught <strong>{tp:,}</strong> of {tp + fn:,} illicit &nbsp;|&nbsp; missed {fn:,} &nbsp;|&nbsp; "
        f"{fp:,} false alarms out of {tn + fp:,} licit."
        "</div>",
        unsafe_allow_html=True,
    )

    available = [(f, label) for f, label in GRAPHS if (graphs_dir / f).exists()]
    for i in range(0, len(available), 2):
        cols = st.columns(2)
        for col, (filename, label) in zip(cols, available[i:i + 2]):
            with col:
                _section_label(label)
                st.image(str(graphs_dir / filename), width="stretch")
