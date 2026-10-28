"""
BATMAN — Bitcoin Anomaly Traffic & Monitoring Analysis Network
==============================================================

Main Streamlit entry point.
Restores the original visual design across all pages and components.
"""

import streamlit as st
from src.web_frontend.components import get_font_css, render_footer

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BATMAN — Bitcoin Anomaly Traffic & Monitoring Analysis Network",
    page_icon="🦇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 1. Offline Font Registration (Orbitron & JetBrains Mono)
# ---------------------------------------------------------------------------
font_css = get_font_css()
st.markdown(f"<style>\n{font_css}\n</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. Global CSS — Exact restoration of original desktop command-center aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── App Shell & Base Typography ─────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    background-color: #080404 !important;
    color: #d4d4d4 !important;
    font-family: 'JetBrains Mono', Consolas, monospace !important;
}

/* Responsive padding matching desktop command center layout */
[data-testid="stMainBlockContainer"], .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* ── Top-bar / header strip ──────────────────────────────────────────────── */
header[data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: #080404 !important;
    border-bottom: 1px solid #3b0b0b !important;
    height: 2.5rem !important;
}
.stAppDeployButton,
.stDeployButton,
[data-testid="stToolbarActions"],
[data-testid="stStatusWidget"],
#MainMenu,
header[data-testid="stHeader"] button {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}

/* ── Sidebar Shell ───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0d0808 !important;
    border-right: 1px solid #3b0b0b !important;
    min-width: 270px !important;
    max-width: 285px !important;
    width: 275px !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #0d0808 !important;
    width: 275px !important;
}
[data-testid="stSidebarContent"] {
    background-color: #0d0808 !important;
    padding: 14px 10px !important;
}
[data-testid="stSidebarCollapseButton"] button {
    color: #8b0000 !important;
}

/* ── Sidebar Navigation Items ────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stRadio"] {
    margin-top: 6px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    background-color: transparent !important;
    border: 1px solid transparent !important;
    border-left: 3px solid transparent !important;
    border-radius: 4px !important;
    padding: 9px 12px !important;
    margin-bottom: 3px !important;
    color: #a0a0a0 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    display: flex !important;
    align-items: center !important;
    white-space: nowrap !important;
    cursor: pointer !important;
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background-color: rgba(40, 10, 10, 0.8) !important;
    color: #c06c6c !important;
    border-color: #5e1313 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background-color: rgba(94, 19, 19, 0.4) !important;
    color: #d4d4d4 !important;
    border: 1px solid #4a1111 !important;
    border-left: 3px solid #b33939 !important;
    font-weight: bold !important;
}
/* Completely hide the default radio circle in Streamlit sidebar */
[data-testid="stSidebar"] [data-testid="stRadio"] div:has(> [data-testid="stMarkdownContainer"]) > *:not([data-testid="stMarkdownContainer"]),
[data-testid="stSidebar"] [data-testid="stRadio"] div[class*="etak9234"],
[data-testid="stSidebar"] [data-testid="stRadio"] div[class*="etak9235"],
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    position: absolute !important;
    pointer-events: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"] * {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* ── Metric Cards ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: rgba(13, 5, 5, 0.95) !important;
    border: 1px solid #8b0000 !important;
    border-radius: 8px !important;
    padding: 8px 10px !important;
    min-height: 80px !important;
    box-shadow: 0 0 10px rgba(139, 0, 0, 0.15) !important;
    transition: background-color 0.2s ease, border-color 0.2s ease !important;
}
[data-testid="stMetric"]:hover {
    background-color: rgba(40, 10, 10, 0.95) !important;
    border: 1px solid #ff3333 !important;
}
[data-testid="stMetricLabel"] {
    margin-bottom: 2px !important;
    overflow: visible !important;
}
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {
    color: #ff4d4d !important;
    font-family: 'JetBrains Mono', Consolas, monospace !important;
    font-size: 10px !important;
    font-weight: bold !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    margin: 0 !important;
    padding: 0 !important;
    white-space: nowrap !important;
    overflow: visible !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div {
    color: #e0e0e0 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 24px !important;
    font-weight: bold !important;
    letter-spacing: 1px !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Headings ───────────────────────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: #e0e0e0 !important;
    letter-spacing: 2px !important;
}
h4, h5, h6 {
    font-family: 'Orbitron', sans-serif !important;
    color: #c0c0c0 !important;
    letter-spacing: 1px !important;
}
.section-title {
    color: #b33939 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: bold !important;
    font-size: 14px !important;
    letter-spacing: 1.5px !important;
    margin-top: 14px !important;
    margin-bottom: 8px !important;
    text-transform: uppercase !important;
}
.section-label {
    color: #ff4d4d !important;
    font-family: 'JetBrains Mono', Consolas, monospace !important;
    font-size: 11px !important;
    font-weight: bold !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
}

/* ── DataFrames & Tables ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    background-color: #0d0808 !important;
    border: 1px solid #4a1111 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] iframe {
    background-color: #0d0808 !important;
}

/* ── Plotly Chart Containers ─────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    background-color: #0a0a0f !important;
    border: 1px solid #8b0000 !important;
    border-radius: 8px !important;
    padding: 8px !important;
}

/* ── Form Inputs (Text, Area, Number, Date) ──────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea textarea,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input {
    background-color: #111118 !important;
    border: 1px solid #2a0a0a !important;
    color: #c0c0c0 !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', Consolas, monospace !important;
    font-size: 12px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stDateInput"] input:focus {
    border-color: #b33939 !important;
    box-shadow: 0 0 5px rgba(179, 57, 57, 0.4) !important;
    outline: none !important;
}

/* ── Selectbox & Multiselect ─────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background-color: #111118 !important;
    border: 1px solid #2a0a0a !important;
    color: #c0c0c0 !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', Consolas, monospace !important;
    font-size: 12px !important;
}
[data-baseweb="popover"], [data-baseweb="menu"], [data-baseweb="select"] ul {
    background-color: #0d0808 !important;
    border: 1px solid #4a1111 !important;
    color: #d4d4d4 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-baseweb="menu"] li:hover {
    background-color: #3b0b0b !important;
    color: #ffffff !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button,
button[kind="primary"],
button[kind="secondary"] {
    background-color: #170909 !important;
    border: 1px solid #5e1313 !important;
    color: #e0e0e0 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 12px !important;
    font-weight: bold !important;
    letter-spacing: 1px !important;
    border-radius: 4px !important;
    padding: 6px 16px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover:not(:disabled) {
    background-color: #3b0b0b !important;
    border-color: #b33939 !important;
    color: #ffffff !important;
    box-shadow: 0 0 8px rgba(179, 57, 57, 0.4) !important;
}
.stButton > button:disabled {
    background-color: #100606 !important;
    border-color: #2a0a0a !important;
    color: #555555 !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] {
    border: none !important;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #2a0a0a !important;
    background-color: transparent !important;
    gap: 4px !important;
}
[data-testid="stTabs"] [role="tab"] {
    background-color: #111118 !important;
    color: #888888 !important;
    border: 1px solid #2a0a0a !important;
    border-bottom: none !important;
    font-family: 'JetBrains Mono', Consolas, monospace !important;
    font-size: 11px !important;
    padding: 8px 16px !important;
    border-radius: 4px 4px 0 0 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background-color: #1a0a0a !important;
    color: #e0e0e0 !important;
    border-bottom: 2px solid #8b1a1a !important;
    font-weight: bold !important;
}
[data-testid="stTabs"] [data-testid="stTabContent"] {
    background-color: #080404 !important;
    border: 1px solid #2a0a0a !important;
    border-top: none !important;
    padding: 16px !important;
    border-radius: 0 0 4px 4px !important;
}

/* ── Sliders ─────────────────────────────────────────────────────────────── */
.stSlider [data-testid="stSliderTrack"] {
    background-color: #1a0a0a !important;
    border: 1px solid #2a0a0a !important;
    height: 4px !important;
}
.stSlider [data-testid="stSliderThumb"] {
    background-color: #8b1a1a !important;
    border: 2px solid #ff3333 !important;
    width: 14px !important;
    height: 14px !important;
}
.stSlider [data-testid="stSliderThumbValue"] {
    color: #c0c0c0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #8b0000 !important;
    opacity: 0.6 !important;
    margin: 16px 0 !important;
}

/* ── Info / Success / Warning / Error Callouts ────────────────────────────── */
[data-testid="stInfo"] {
    background-color: #111118 !important;
    border: 1px solid #2a0a0a !important;
    border-left: 3px solid #b33939 !important;
    color: #c0c0c0 !important;
    font-family: 'JetBrains Mono', Consolas, monospace !important;
    font-size: 12px !important;
    border-radius: 4px !important;
}
[data-testid="stSuccess"] {
    background-color: #0a1a0a !important;
    border: 1px solid #153a15 !important;
    border-left: 3px solid #1a8b1a !important;
    color: #1a8b1a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 4px !important;
}
[data-testid="stWarning"] {
    background-color: #1a140a !important;
    border: 1px solid #3a2a10 !important;
    border-left: 3px solid #f39c12 !important;
    color: #f39c12 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 4px !important;
}
[data-testid="stError"] {
    background-color: #1a0808 !important;
    border: 1px solid #4a1111 !important;
    border-left: 3px solid #f2353c !important;
    color: #f2353c !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 4px !important;
}

/* ── Scrollbars ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #080404;
}
::-webkit-scrollbar-thumb {
    background: #4a1111;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #8b0000;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Brand, Navigation & Footer
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding-top:10px;'>"
        "<div style='font-family:\"Orbitron\", sans-serif; font-size:32px; font-weight:bold; color:#d4d4d4; letter-spacing:3px;'>BATMAN</div>"
        "<div style='font-family:\"Orbitron\", sans-serif; font-size:10px; font-weight:bold; color:#c06c6c; margin-top:4px; line-height:1.3;'>BITCOIN ANOMALY TRAFFIC AND<br>MONITORING ANALYSIS NETWORK</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-top:1px solid #8b0000; margin:14px 0;'>", unsafe_allow_html=True)

    nav_options = [
        "> Overview",
        "> Transaction Analysis",
        "> Network Graph",
        "> Entity Investigation",
        "> Alerts",
        "> Explainability",
        "> Model Results",
        "> Wallet Scanner",
    ]

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-top:1px solid #8b0000; margin:20px 0 14px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center; font-family:\"Orbitron\", sans-serif; font-size:12px; font-weight:bold; color:#c06c6c; line-height:1.4;'>"
        "BITCOIN<br>ANALYSIS ENGINE"
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
clean_page = selected_nav.replace("> ", "").strip()

if clean_page == "Overview":
    from src.web_frontend.overview import render_overview
    render_overview()

elif clean_page == "Transaction Analysis":
    from src.web_frontend.transaction_analysis import render_transaction_analysis
    render_transaction_analysis()

elif clean_page == "Network Graph":
    from src.web_frontend.network_graph import render_network_graph
    render_network_graph()

elif clean_page == "Entity Investigation":
    from src.web_frontend.entity_investigation import render_entity_investigation
    render_entity_investigation()

elif clean_page == "Alerts":
    from src.web_frontend.alerts import render_alerts
    render_alerts()

elif clean_page == "Explainability":
    from src.web_frontend.explainability import render_explainability
    render_explainability()

elif clean_page == "Model Results":
    from src.web_frontend.model_results import render_model_results
    render_model_results()

elif clean_page == "Wallet Scanner":
    from src.web_frontend.wallet_scanner import render_wallet_scanner
    render_wallet_scanner()

# ---------------------------------------------------------------------------
# Footer — Status bar style
# ---------------------------------------------------------------------------
render_footer()
