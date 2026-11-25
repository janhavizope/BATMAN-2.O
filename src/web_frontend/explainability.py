"""
Explainability detail
----------------------
Explains *why* a wallet got its score, as a reusable block rendered inside the
Investigate Wallet page (it is no longer a standalone page):
- a plain-English verdict box (level, main drivers, ground truth if known);
- feature contributions coloured by whether each factor pushed risk UP (red) or
  DOWN (teal), from a Saabas decomposition of the Random Forest;
- a comparison of the wallet's values against typical licit and illicit wallets,
  with which side each feature leans toward.

`render_explanation_detail(entity_id, data)` takes a dict with risk_score,
anomaly_score, top_features, evidence, and explanation/human_reason.
"""

import re
from html import escape

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.web_frontend.components import render_metric_card

UP_COLOR = "#ff6b6b"      # a feature that pushes risk up
DOWN_COLOR = "#1de9b6"    # a feature that pushes risk down

# Plain-English names for the model's features, so the explanation reads simply.
_PLAIN = {
    "transacted w address total": "number of different wallets it dealt with",
    "total txs": "total number of transactions",
    "num txs as sender": "number of times it sent money",
    "num txs as receiver": "number of times it received money",
    "fees as share total": "total fees as a share of value moved",
    "fees as share mean": "typical fee as a share of value",
    "fees as share median": "typical (median) fee as a share of value",
    "fees as share min": "smallest fee as a share of value",
    "fees as share max": "largest fee as a share of value",
    "fees total": "total fees paid",
    "fees mean": "typical fee paid",
    "fees median": "typical (median) fee paid",
    "btc transacted total": "total Bitcoin moved",
    "btc transacted mean": "typical Bitcoin per transaction",
    "btc transacted median": "typical (median) Bitcoin per transaction",
    "btc transacted max": "largest single amount moved",
    "btc transacted min": "smallest single amount moved",
    "btc sent total": "total Bitcoin sent",
    "btc sent max": "largest amount sent",
    "btc received total": "total Bitcoin received",
    "btc received max": "largest amount received",
    "btc received min": "smallest amount received",
    "lifetime in blocks": "how long the wallet stayed active",
    "num timesteps appeared in": "number of active periods",
    "blocks btwn txs min": "shortest gap between its transactions",
    "blocks btwn txs mean": "average gap between its transactions",
    "blocks btwn txs max": "longest gap between its transactions",
    "blocks btwn txs median": "typical gap between its transactions",
}


def _plain(name) -> str:
    """Translate a model feature name into plain English (falls back to the name)."""
    key = str(name).strip().lower().replace("_", " ")
    return _PLAIN.get(key, str(name).replace("_", " "))


def _direction(feature: dict) -> str:
    """A feature's risk direction; inferred from the contribution sign if absent."""
    d = feature.get("direction")
    if d in ("increases", "decreases"):
        return d
    return "increases" if feature.get("contribution", 0) >= 0 else "decreases"

# value = X (typical licit L, typical illicit I)
_COMPARE_RE = re.compile(r"^(.*?)\s*=\s*([-\d.]+)\s*\(typical licit\s*([-\d.]+),\s*typical illicit\s*([-\d.]+)\)")


def _level(score: float) -> tuple[str, str]:
    if score >= 80:
        return "CRITICAL", "#ff4d4d"
    if score >= 60:
        return "HIGH", "#ff8c42"
    if score >= 35:
        return "MEDIUM", "#e0c04d"
    return "LOW", "#4d8b4d"


def _parse_comparisons(evidence: list) -> list:
    rows = []
    for ev in evidence or []:
        m = _COMPARE_RE.match(str(ev))
        if not m:
            continue
        feat, val, licit, illicit = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
        d_lic, d_ill = abs(val - licit), abs(val - illicit)
        leans = "neutral" if abs(d_lic - d_ill) < 1e-9 else ("illicit" if d_ill < d_lic else "licit")
        rows.append({"Feature": _plain(feat), "This wallet": val,
                     "Typical licit": licit, "Typical illicit": illicit, "Leans toward": leans})
    return rows


def _ground_truth(evidence: list) -> str | None:
    for ev in evidence or []:
        if str(ev).lower().startswith("ground-truth label"):
            return str(ev)
    return None


def render_explanation_detail(entity_id: str, data: dict) -> None:
    """Render the full 'why was this wallet scored this way' block for one wallet."""
    score = float(data.get("risk_score", 0) or 0)
    level, level_color = _level(score)
    top_features = data.get("top_features", []) or []
    evidence_list = data.get("evidence", []) or []
    human_reason = data.get("human_reason") or data.get("explanation") or \
        "Wallet flagged due to abnormal risk signals."

    # --- Plain-English verdict box ------------------------------------------
    drivers = [f for f in top_features if _direction(f) == "increases"][:3]
    driver_text = ", ".join(escape(_plain(f.get("feature", ""))) for f in drivers) or "several behavioral factors"
    gt = _ground_truth(evidence_list)
    gt_line = f"<br><span style='color:#888;'>{escape(gt)}</span>" if gt else ""
    st.markdown(
        f"<div style='background-color:#120c0c; border:1px solid {level_color}; "
        f"border-left:4px solid {level_color}; border-radius:5px; padding:14px 16px; margin:6px 0 14px 0; "
        f"font-family:\"JetBrains Mono\", Consolas, monospace; font-size:13px; color:#d4d4d4; line-height:1.6;'>"
        f"<span style='color:{level_color}; font-weight:bold; font-size:15px;'>{level} &middot; {score:.0f}/100</span><br>"
        f"Wallet <strong>{escape(str(entity_id)[:16])}&hellip;</strong> is flagged mainly because of "
        f"<strong>{driver_text}</strong>. Each factor below shows how much it pushed the score "
        f"up (red) or down (teal). A score is an indicator for review, not a verdict.{gt_line}"
        f"</div>",
        unsafe_allow_html=True,
    )

    scol1, scol2 = st.columns(2)
    with scol1:
        render_metric_card("RISK SCORE", f"{data.get('risk_score', 0)}/100")
    with scol2:
        render_metric_card("ANOMALY SCORE", f"{float(data.get('anomaly_score', 0) or 0):.2f}")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # --- Feature contributions (coloured by direction) -----------------------
    st.markdown(
        "<div style='color:#b33939; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; "
        "font-weight:bold; letter-spacing:1px; margin-bottom:4px;'>WHAT DROVE THE SCORE</div>",
        unsafe_allow_html=True,
    )
    st.caption("Each bar is how much a behavioral feature moved this wallet's risk, from a Saabas "
               "decomposition of the Random Forest. Red pushes risk up; teal pulls it down.")
    if top_features:
        sorted_features = sorted(top_features, key=lambda x: x.get("contribution", 0))
        y_labels = [_plain(f.get("feature", "")) for f in sorted_features]
        x_values = [f.get("contribution", 0) for f in sorted_features]
        colors = [UP_COLOR if _direction(f) == "increases" else DOWN_COLOR for f in sorted_features]
        fig_bar = go.Figure(go.Bar(
            x=x_values, y=y_labels, orientation="h",
            marker=dict(color=colors, line=dict(color="#080404", width=1)),
            text=[f"{v:.2f}" for v in x_values], textposition="auto",
            textfont=dict(family="JetBrains Mono, monospace", color="#080404", size=11),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_bar.update_layout(
            plot_bgcolor="#080404", paper_bgcolor="#080404",
            font=dict(color="#c0c0c0", family="JetBrains Mono, monospace", size=11),
            xaxis=dict(showgrid=True, gridcolor="#333333", gridwidth=0.5, griddash="dash",
                       showline=True, linecolor="#444444",
                       range=[0, max(1.0, max(x_values) * 1.15 if x_values else 1.0)]),
            yaxis=dict(showgrid=False, showline=False, tickfont=dict(size=11, color="#e0e0e0")),
            margin=dict(l=20, r=20, t=10, b=30), height=max(180, len(y_labels) * 55),
        )
        st.plotly_chart(fig_bar, width="stretch")
    else:
        st.info("No feature contributions available for this wallet.")

    # --- Comparison to typical licit / illicit -------------------------------
    comparisons = _parse_comparisons(evidence_list)
    if comparisons:
        st.markdown(
            "<div style='color:#b33939; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; "
            "font-weight:bold; letter-spacing:1px; margin:12px 0 4px 0;'>"
            "HOW IT COMPARES WITH ORDINARY vs ILLICIT WALLETS</div>",
            unsafe_allow_html=True,
        )
        st.caption("For each feature: this wallet's value next to the typical licit and typical illicit "
                   "wallet, and which side it sits closer to.")
        cmp_df = pd.DataFrame(comparisons)

        def _style_leans(val):
            if val == "illicit":
                return "color:#ff6b6b; font-weight:bold;"
            if val == "licit":
                return "color:#1de9b6; font-weight:bold;"
            return "color:#e0c04d;"
        styled = cmp_df.style.map(_style_leans, subset=["Leans toward"]).format(
            {"This wallet": "{:.3g}", "Typical licit": "{:.3g}", "Typical illicit": "{:.3g}"})
        st.dataframe(styled, width="stretch", hide_index=True)
        leans_ill = sum(1 for c in comparisons if c["Leans toward"] == "illicit")
        st.caption(f"On {leans_ill} of {len(comparisons)} compared feature(s), this wallet sits closer to "
                   "the typical **illicit** profile." if leans_ill else
                   "On the compared feature(s), this wallet sits closer to the typical **licit** profile.")

    # --- Plain-language reason ----------------------------------------------
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='background-color:#111118; border:1px solid #2a0a0a; border-radius:4px; padding:12px; "
        f"font-family:\"JetBrains Mono\", Consolas, monospace; font-size:12px; color:#c0c0c0; line-height:1.6;'>"
        f"{escape(str(human_reason))}</div>",
        unsafe_allow_html=True,
    )
