"""
Shared UI Components
--------------------
Reusable display elements used across all BATMAN pages.
Restores the original visual design tokens:
- JetBrains Mono and Orbitron typography (offline base64 fonts)
- Dark near-black (#080404) & maroon palette
- Thin bright-red borders (#8b0000)
- Neon-green model indicators (#1a8b1a, #22aa55)
"""

import os
import base64
from pathlib import Path
import streamlit as st


@st.cache_data
def get_font_css() -> str:
    """Read local font files and return @font-face CSS rules for offline usage."""
    fonts_dir = Path(__file__).parent.parent / "resources" / "fonts"
    orbitron_path = fonts_dir / "Orbitron.ttf"
    jb_path = fonts_dir / "JetBrainsMono.ttf"

    css_parts = []
    if orbitron_path.exists():
        with open(orbitron_path, "rb") as f:
            orb_b64 = base64.b64encode(f.read()).decode("utf-8")
        css_parts.append(f"""
        @font-face {{
            font-family: 'Orbitron';
            src: url(data:font/truetype;charset=utf-8;base64,{orb_b64}) format('truetype');
            font-weight: normal;
            font-style: normal;
        }}
        """)

    if jb_path.exists():
        with open(jb_path, "rb") as f:
            jb_b64 = base64.b64encode(f.read()).decode("utf-8")
        css_parts.append(f"""
        @font-face {{
            font-family: 'JetBrains Mono';
            src: url(data:font/truetype;charset=utf-8;base64,{jb_b64}) format('truetype');
            font-weight: normal;
            font-style: normal;
        }}
        """)

    return "\n".join(css_parts)


def render_dev_badge() -> None:
    """Display cybersecurity-style badge indicating live ML connection and data status."""
    from src.data.data_manager import is_data_available
    if is_data_available():
        st.markdown(
            "<div style=\"background-color:#0a1a0a; border:1px solid #153a15; "
            "border-left:3px solid #1a8b1a; border-radius:3px; padding:8px 14px; "
            "font-size:11px; font-family:'JetBrains Mono', Consolas, monospace; "
            "color:#1a8b1a; margin-bottom:14px; letter-spacing:0.5px;\">"
            "<strong>[LIVE MODE] ML MODEL CONNECTED</strong> &mdash; "
            "Data is being populated by the trained Random Forest inferences."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style=\"background-color:#140808; border:1px solid #8b0000; "
            "border-left:3px solid #ff4d4d; border-radius:3px; padding:8px 14px; "
            "font-size:11px; font-family:'JetBrains Mono', Consolas, monospace; "
            "color:#ff8888; margin-bottom:14px; letter-spacing:0.5px;\">"
            "<strong>[STANDBY MODE] ML MODEL READY</strong> &mdash; "
            "Random Forest model is loaded. No bundled dataset present. Ingest external transaction records to analyze."
            "</div>",
            unsafe_allow_html=True,
        )


def render_dataset_notice() -> None:
    """Display notice if no dataset is currently loaded."""
    from src.data.data_manager import is_data_available
    if not is_data_available():
        st.markdown(
            "<div style=\"background-color:#0d0808; border:1px solid #4a1111; "
            "border-left:3px solid #b33939; border-radius:4px; padding:10px 14px; "
            "margin-bottom:16px; font-family:'JetBrains Mono', Consolas, monospace; "
            "font-size:11px; color:#d4d4d4; line-height:1.5;\">"
            "<strong style='color:#ff4d4d;'>[DATASET REQUIRED]</strong> "
            "All bundled sample and development datasets have been removed. "
            "Provide an external Bitcoin transaction file (CSV/JSON) or connect a database to run anomaly detection."
            "</div>",
            unsafe_allow_html=True,
        )


def render_metric_card(label: str, value: str, delta: str | None = None) -> None:
    """Render a single KPI metric card inside the current Streamlit column."""
    st.metric(label=label, value=value, delta=delta)


def render_header(title: str | None = None) -> None:
    """Render the page header with cyberpunk style."""
    if title:
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:10px;'>"
            f"<div style='font-family:\"Orbitron\", sans-serif; font-size:24px; font-weight:bold; color:#e0e0e0; letter-spacing:4px;'>{title}</div>"
            f"<div style='color:#22aa55; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; "
            f"border:1px solid #22aa55; border-radius:3px; padding:4px 10px; font-weight:bold;'>SYSTEM ONLINE</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    """Render the status bar footer matching the desktop status bar."""
    from src.data.data_manager import is_data_available
    data_status = "EXTERNAL_DATASET_LOADED" if is_data_available() else "NO_DATASET_LOADED"
    st.markdown(
        f"<div style='margin-top:40px; border-top:1px solid #3b0b0b; padding-top:8px; "
        f"color:#a52a2a; font-family:\"JetBrains Mono\", Consolas, monospace; font-size:11px; letter-spacing:1px;'>"
        f"SYS_STATUS: ONLINE &nbsp;|&nbsp; {data_status}"
        f"</div>",
        unsafe_allow_html=True,
    )
