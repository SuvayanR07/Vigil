"""
VIGIL — Streamlit UI for the adverse event report classifier.

Two modes:
  - Live   : calls pipeline/classify.py against local Ollama
  - Demo   : loads pre-computed ClassifiedReports from data/demo_results.json
             (for Streamlit Cloud, where Ollama is unavailable)
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CONFIDENCE_THRESHOLD,
    DEMO_RESULTS_PATH,
    OLLAMA_URL,
    TEST_NARRATIVES_PATH,
)
from pipeline.customer import (
    Customer,
    create_customer,
    list_customers,
    load_customer,
)

VALIDATION_RESULTS_PATH = Path(__file__).parent / "data" / "validation_results.json"
GITHUB_URL = "https://github.com/SuvayanR07/vigil-adverse-event-classifier"


# --------------------------------------------------------------------------- #
# Design system — ported from VIGIL v2 design handoff                          #
# --------------------------------------------------------------------------- #

VIGIL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --primary: #D91E63; --primary-light: #FDE8F1; --primary-hover: #B91855;
  --primary-glow: rgba(217,30,99,.13);
  --success: #059669; --success-light: #D1FAE5; --success-border: #6EE7B7;
  --danger: #DC2626; --danger-light: #FEE2E2; --danger-border: #FCA5A5;
  --warning: #D97706; --warning-light: #FEF3C7; --warning-border: #FCD34D;
  --bg: #F8F9FB; --surface: #FFFFFF; --surface-2: #F3F4F6;
  --border: #E5E7EB; --border-strong: #D1D5DB;
  --text: #111827; --text-2: #6B7280; --text-3: #9CA3AF;
  --sidebar-bg: #0C0C0C; --sidebar-surface: #181818; --sidebar-border: #262626;
  --sidebar-text: #F5F5F5; --sidebar-text-2: #9CA3AF;
  --r: 8px; --rl: 12px;
  --shadow: 0 1px 3px rgba(0,0,0,.06), 0 4px 18px rgba(0,0,0,.05);
  --shadow-md: 0 4px 8px rgba(0,0,0,.06), 0 14px 36px rgba(0,0,0,.09);
}

html, body, .stApp, [class*="css"] {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
.stApp { background: var(--bg); color: var(--text); }
code, pre, .mono { font-family: 'IBM Plex Mono', monospace; }

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; height: 0; }

/* Widen main content on cloud viewport */
.block-container {
  padding-top: 1.1rem !important;
  padding-bottom: 2.5rem !important;
  max-width: 1280px !important;
}

/* ── Sidebar (dark, near-black, radial magenta glow) ─────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--sidebar-border);
}
section[data-testid="stSidebar"] > div:first-child {
  background: var(--sidebar-bg) !important;
}
section[data-testid="stSidebar"]::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 220px;
  background: radial-gradient(ellipse at 50% -20%, rgba(217,30,99,.14) 0%, transparent 65%);
  pointer-events: none; z-index: 0;
}
section[data-testid="stSidebar"] > div {
  position: relative; z-index: 1;
}
section[data-testid="stSidebar"] * { color: var(--sidebar-text); }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
  color: var(--sidebar-text-2) !important;
  font-size: 12px;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  color: var(--sidebar-text-2) !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  margin-top: 1rem !important;
  margin-bottom: 0.35rem !important;
}
section[data-testid="stSidebar"] hr {
  border-color: var(--sidebar-border) !important;
  margin: 0.75rem 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stMetric"] {
  background: var(--sidebar-surface) !important;
  border: 1px solid var(--sidebar-border) !important;
  padding: 8px 10px !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
  color: var(--sidebar-text) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-weight: 300 !important;
  font-size: 20px !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
  color: var(--sidebar-text-2) !important;
}
section[data-testid="stSidebar"] .stButton > button {
  background: var(--sidebar-surface) !important;
  color: var(--sidebar-text) !important;
  border: 1px solid var(--sidebar-border) !important;
  border-radius: 8px !important;
  font-size: 12px !important;
  font-weight: 500 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  border-color: var(--primary) !important;
  color: var(--primary) !important;
  background: var(--sidebar-surface) !important;
}
section[data-testid="stSidebar"] [data-testid="stAlert"] {
  background: var(--sidebar-surface) !important;
  border: 1px solid var(--sidebar-border) !important;
  color: var(--sidebar-text-2) !important;
  border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stAlert"] * {
  color: var(--sidebar-text-2) !important;
}
section[data-testid="stSidebar"] a { color: var(--primary) !important; }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 0 !important; background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0 4px;
}
.stTabs [data-baseweb="tab"] {
  padding: 12px 18px !important;
  font-size: 13px !important; font-weight: 500 !important;
  color: var(--text-2) !important;
  background: transparent !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -1px !important;
  border-radius: 0 !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text) !important; }
.stTabs [aria-selected="true"] {
  color: var(--primary) !important;
  border-bottom-color: var(--primary) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: 8px !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  transition: all 0.18s ease !important;
}
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
  background: var(--primary) !important;
  color: white !important;
  border: none !important;
  padding: 0.55rem 1.3rem !important;
}
.stButton > button[kind="primary"]:hover:not(:disabled),
.stFormSubmitButton > button[kind="primary"]:hover:not(:disabled) {
  background: var(--primary-hover) !important;
  box-shadow: 0 4px 18px rgba(217,30,99,.38) !important;
  transform: scale(1.02);
}
.stButton > button[kind="secondary"],
.stDownloadButton > button {
  border: 1.5px solid var(--border-strong) !important;
  color: var(--text-2) !important;
  background: var(--surface) !important;
}
.stButton > button[kind="secondary"]:hover:not(:disabled),
.stDownloadButton > button:hover:not(:disabled) {
  border-color: var(--primary) !important;
  color: var(--primary) !important;
  background: var(--primary-light) !important;
}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea,
.stSelectbox [data-baseweb="select"] > div,
.stFileUploader > section, .stNumberInput input {
  border: 1.5px solid var(--border) !important;
  border-radius: 12px !important;
  background: var(--surface) !important;
  font-size: 13.5px !important;
  color: var(--text) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px var(--primary-glow) !important;
}
.stFileUploader > section {
  border-style: dashed !important;
  background: var(--surface) !important;
  padding: 1.2rem !important;
}
.stFileUploader > section:hover {
  border-color: var(--primary) !important;
  background: var(--primary-light) !important;
}

/* ── Metrics (main area) ─────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
  box-shadow: var(--shadow);
}
[data-testid="stMetricLabel"] {
  font-size: 10px !important;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--text-3) !important;
  font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-weight: 300 !important;
  font-size: 28px !important;
  color: var(--text) !important;
}

/* ── Expanders ──────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  box-shadow: var(--shadow) !important;
  margin-bottom: 10px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] [role="button"] {
  padding: 12px 16px !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-2) !important;
}

/* ── Dataframes ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  overflow: hidden !important;
  background: var(--surface) !important;
  box-shadow: var(--shadow) !important;
}

/* ── Alerts ──────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 12px !important; }

/* ── Progress bar ───────────────────────────────────────────────────────── */
.stProgress > div > div > div > div { background: var(--primary) !important; }

/* ── Radio buttons ──────────────────────────────────────────────────────── */
.stRadio [role="radiogroup"] label {
  font-size: 12px !important;
}

/* ── Typography ─────────────────────────────────────────────────────────── */
h1, h2, h3, h4 { font-family: 'IBM Plex Sans', sans-serif; }
h1 { font-weight: 600; letter-spacing: -0.01em; }

/* ── Custom VIGIL components (used via st.markdown) ─────────────────────── */

.vigil-hero {
  display: flex; align-items: center; gap: 14px;
  padding: 2px 0 14px; margin-bottom: 12px;
  border-bottom: 1px solid var(--border);
}
.vigil-hero-icon {
  width: 40px; height: 40px; background: var(--primary); border-radius: 10px;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  box-shadow: 0 0 14px rgba(217,30,99,.55), 0 0 32px rgba(217,30,99,.22);
}
.vigil-hero-icon svg { width: 22px; height: 22px; stroke: white; }
.vigil-hero-text { flex: 1; }
.vigil-hero-title {
  font-size: 20px; font-weight: 600; letter-spacing: 0.16em;
  color: var(--text); line-height: 1;
}
.vigil-hero-sub {
  font-size: 10.5px; color: var(--text-3); letter-spacing: 0.09em;
  text-transform: uppercase; margin-top: 4px;
}
.vigil-mode-pill {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 11px; font-weight: 500; padding: 4px 10px; border-radius: 100px;
  font-family: 'IBM Plex Sans', sans-serif;
}
.vigil-mode-pill.live { color: var(--success); background: var(--success-light); border: 1px solid var(--success-border); }
.vigil-mode-pill.demo { color: var(--warning); background: var(--warning-light); border: 1px solid var(--warning-border); }
.vigil-mode-pill .dot { width: 6px; height: 6px; border-radius: 50%; }
.vigil-mode-pill.live .dot { background: var(--success); animation: pdot 2s infinite; }
.vigil-mode-pill.demo .dot { background: var(--warning); }
@keyframes pdot { 0%,100%{opacity:1}50%{opacity:.4} }

.vigil-sb-logo {
  display: flex; align-items: center; gap: 10px;
  padding: 4px 0 6px;
}
.vigil-sb-logo-icon {
  width: 30px; height: 30px; background: var(--primary); border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 14px rgba(217,30,99,.55), 0 0 32px rgba(217,30,99,.22);
}
.vigil-sb-logo-icon svg { width: 16px; height: 16px; stroke: white; }
.vigil-sb-wordmark {
  font-size: 15px; font-weight: 600; letter-spacing: 0.14em; color: white;
}
.vigil-sb-tagline {
  font-size: 9px; color: var(--sidebar-text-2); letter-spacing: 0.08em;
  text-transform: uppercase; margin: -4px 0 10px 40px;
}

.sev-banner {
  border-radius: 12px; padding: 16px 18px; margin: 4px 0 14px;
  display: flex; align-items: center; justify-content: space-between; gap: 14px;
}
.sev-banner.serious    { background: var(--danger-light);  border: 1px solid var(--danger-border); }
.sev-banner.non-serious{ background: var(--success-light); border: 1px solid var(--success-border); }
.sev-banner .label { font-size: 17px; font-weight: 600; line-height: 1.1; }
.sev-banner.serious    .label { color: var(--danger); }
.sev-banner.non-serious .label { color: var(--success); }
.sev-banner .reason { font-size: 12px; color: var(--text-2); margin-top: 4px; }
.sev-banner .conf-num {
  font-size: 26px; font-weight: 300;
  font-family: 'IBM Plex Mono', monospace; line-height: 1;
}
.sev-banner .conf-num.high { color: var(--success); }
.sev-banner .conf-num.med  { color: var(--warning); }
.sev-banner .conf-num.low  { color: var(--danger); }
.sev-banner .conf-lbl {
  font-size: 9px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--text-3); margin-top: 2px;
}
.sev-banner .conf-block { text-align: right; }

.reaction-row {
  padding: 11px 0; border-bottom: 1px solid var(--border);
}
.reaction-row:last-child { border-bottom: none; padding-bottom: 2px; }
.reaction-top {
  display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
}
.reaction-term { font-size: 14px; font-weight: 500; color: var(--text); }
.reaction-pt { font-size: 12px; color: var(--text-2); margin-top: 2px; }
.meddra-pill {
  font-family: 'IBM Plex Mono', monospace; font-size: 11px;
  color: var(--primary); background: var(--primary-light);
  padding: 3px 8px; border-radius: 4px;
  border: 1px solid rgba(217,30,99,.22); white-space: nowrap; flex-shrink: 0;
}
.soc-chip {
  display: inline-block; margin-top: 6px;
  font-size: 10.5px; color: var(--text-3);
  background: var(--surface-2); border: 1px solid var(--border);
  padding: 2px 8px; border-radius: 4px;
}
.conf-chip {
  display: inline-flex; margin-top: 6px; margin-left: 6px;
  padding: 2px 7px; border-radius: 5px;
  font-size: 10.5px; font-family: 'IBM Plex Mono', monospace; font-weight: 500;
}
.conf-chip.high { color: var(--success); background: var(--success-light); }
.conf-chip.med  { color: var(--warning); background: var(--warning-light); }
.conf-chip.low  { color: var(--danger);  background: var(--danger-light); }

.drug-row {
  padding: 11px 0; border-bottom: 1px solid var(--border);
}
.drug-row:last-child { border-bottom: none; }
.drug-name { font-size: 14px; font-weight: 600; color: var(--text); }
.drug-meta { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 5px; }
.drug-tag {
  font-size: 11px; color: var(--text-2); background: var(--surface-2);
  border: 1px solid var(--border); padding: 2px 8px; border-radius: 5px;
}

.stat-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px 18px; box-shadow: var(--shadow);
  transition: all 0.18s ease; height: 100%;
}
.stat-card:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }
.stat-card .label {
  font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--text-3); margin-bottom: 8px;
}
.stat-card .value {
  font-size: 28px; font-weight: 300;
  font-family: 'IBM Plex Mono', monospace; line-height: 1; color: var(--text);
}
.stat-card .sub { font-size: 11px; color: var(--text-3); margin-top: 5px; }

.section-cap {
  font-size: 10.5px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--text-3);
  margin: 16px 0 6px;
}

.fda-flag {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 7px 12px; margin: 4px 6px 0 0;
  border-radius: 8px; background: var(--danger-light);
  border: 1px solid var(--danger-border);
  font-size: 12px; font-weight: 500; color: var(--danger);
}

/* ── Responsive for narrow Streamlit Cloud viewport ─────────────────────── */
@media (max-width: 900px) {
  .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
  .vigil-hero { gap: 10px; }
  .vigil-hero-title { font-size: 17px; letter-spacing: 0.12em; }
  .sev-banner { flex-direction: column; align-items: flex-start; }
  .sev-banner .conf-block { text-align: left; }
  .stTabs [data-baseweb="tab"] { padding: 10px 10px !important; font-size: 12px !important; }
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
</style>
"""

# Inline SVG shields used in the logos
_SHIELD_SVG_WHITE = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 2L4 6v6c0 5.5 3.5 10.7 8 12 4.5-1.3 8-6.5 8-12V6L12 2z"/>'
    '<path d="M9 12l2 2 4-4"/></svg>'
)


_PLOTLY_TEMPLATE_REGISTERED = False


def _register_plotly_theme() -> None:
    """Register a VIGIL plotly template once per process."""
    global _PLOTLY_TEMPLATE_REGISTERED
    if _PLOTLY_TEMPLATE_REGISTERED:
        return
    import plotly.graph_objects as go  # local import

    pio.templates["vigil"] = go.layout.Template(
        layout=dict(
            font=dict(family="IBM Plex Sans, sans-serif", color="#111827", size=12),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            colorway=["#D91E63", "#059669", "#D97706", "#6366F1", "#0EA5E9", "#DC2626"],
            xaxis=dict(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB"),
            yaxis=dict(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB"),
            legend=dict(font=dict(size=11, color="#6B7280")),
            margin=dict(l=40, r=20, t=40, b=40),
        )
    )
    pio.templates.default = "plotly_white+vigil"
    _PLOTLY_TEMPLATE_REGISTERED = True


def inject_design_system() -> None:
    """Inject the VIGIL v2 design-system CSS. Call once per page render."""
    st.markdown(VIGIL_CSS, unsafe_allow_html=True)
    _register_plotly_theme()


# --------------------------------------------------------------------------- #
# Intro animation — ported from VIGIL v2 design handoff                       #
#                                                                              #
# Streamlit strips <script> from markdown, so we render the intro via         #
# components.v1.html (a tiny iframe) and have its JS inject a fullscreen       #
# overlay + canvas into the *parent* document. Plays once per session         #
# (sessionStorage gate in the JS) and is also gated by st.session_state so    #
# we don't re-mount the iframe on every Streamlit rerun.                      #
# --------------------------------------------------------------------------- #

_INTRO_COMPONENT_HTML = """
<script>
(function() {
  const SEEN = 'vigil-intro-seen';
  const parentDoc = window.parent.document;
  const parentWin = window.parent;

  // Already mounted or already seen this tab -> do nothing.
  if (parentDoc.getElementById('vigil-intro-overlay')) return;
  if (parentWin.sessionStorage.getItem(SEEN)) return;
  parentWin.sessionStorage.setItem(SEEN, '1');

  // --- Inject style into parent doc ---
  const style = parentDoc.createElement('style');
  style.id = 'vigil-intro-style';
  style.textContent = `
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
    #vigil-intro-overlay {
      position: fixed; inset: 0; background: #050505; z-index: 2147483647;
      display: flex; align-items: center; justify-content: center;
      transition: opacity 0.55s ease;
    }
    #vigil-intro-overlay.fading { opacity: 0; pointer-events: none; }
    #vigil-intro-canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
    .vi-content {
      position: relative; z-index: 1; text-align: center;
      display: flex; flex-direction: column; align-items: center;
      font-family: 'IBM Plex Sans', sans-serif;
    }
    .vi-logo {
      width: 76px; height: 76px; background: #D91E63; border-radius: 50%;
      display: flex; align-items: center; justify-content: center; margin-bottom: 26px;
      opacity: 0; animation: vi-ifu 0.8s ease forwards 0.7s;
      box-shadow: 0 0 24px rgba(217,30,99,.75), 0 0 70px rgba(217,30,99,.3), 0 0 120px rgba(217,30,99,.12);
    }
    .vi-logo svg { width: 34px; height: 34px; }
    .vi-wordmark {
      font-size: 62px; font-weight: 600; letter-spacing: 0.3em;
      color: #fff; line-height: 1; margin-bottom: 10px;
      opacity: 0; animation: vi-ifu 0.65s ease forwards 1.55s;
    }
    .vi-sub {
      font-size: 10.5px; font-weight: 400; letter-spacing: 0.38em;
      color: rgba(255,255,255,.38); text-transform: uppercase;
      opacity: 0; animation: vi-ifu 0.5s ease forwards 2.15s;
    }
    .vi-skip {
      position: fixed; bottom: 28px; right: 28px;
      background: none; border: none; color: rgba(255,255,255,.35);
      font-family: 'IBM Plex Sans', sans-serif; font-size: 11.5px;
      cursor: pointer; letter-spacing: 0.04em;
      opacity: 0; animation: vi-ifa 0.4s ease forwards 2.7s;
      transition: color 0.2s ease;
    }
    .vi-skip:hover { color: rgba(255,255,255,.8); }
    @keyframes vi-ifu { from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)} }
    @keyframes vi-ifa { from{opacity:0}to{opacity:1} }
    /* Smooth reveal of the app underneath once the overlay fades */
    body.vi-intro-active > :not(#vigil-intro-overlay):not(#vigil-intro-style) { transition: opacity 0.35s ease; }
  `;
  parentDoc.head.appendChild(style);

  // --- Build overlay in parent doc ---
  const overlay = parentDoc.createElement('div');
  overlay.id = 'vigil-intro-overlay';
  overlay.innerHTML = `
    <canvas id="vigil-intro-canvas"></canvas>
    <div class="vi-content">
      <div class="vi-logo">
        <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 2L4 6v6c0 5.5 3.5 10.7 8 12 4.5-1.3 8-6.5 8-12V6L12 2z"/>
          <path d="M9 12l2 2 4-4"/>
        </svg>
      </div>
      <div class="vi-wordmark">VIGIL</div>
      <div class="vi-sub">Adverse Event Classifier</div>
    </div>
    <button class="vi-skip" id="vigil-intro-skip" type="button">Skip intro</button>
  `;
  parentDoc.body.appendChild(overlay);

  const canvas = parentDoc.getElementById('vigil-intro-canvas');
  const ctx = canvas.getContext('2d');
  let animFrame, dismissTimer;

  function resize() {
    canvas.width = parentWin.innerWidth;
    canvas.height = parentWin.innerHeight;
  }
  resize();
  parentWin.addEventListener('resize', resize);

  // Particle network
  const N = 65;
  const particles = Array.from({length: N}, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    vx: (Math.random() - 0.5) * 0.45,
    vy: (Math.random() - 0.5) * 0.45,
    r: Math.random() * 2 + 0.8,
    op: Math.random() * 0.55 + 0.15
  }));
  const hexes = [
    [0.12,0.18,85],[0.88,0.12,110],[0.75,0.82,78],[0.08,0.78,55],
    [0.5,0.06,65],[0.93,0.55,92],[0.35,0.9,70],[0.6,0.45,48]
  ];

  function drawHex(x, y, size, op) {
    ctx.save();
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = (i * Math.PI) / 3 - Math.PI / 6;
      const px = x + size * Math.cos(a), py = y + size * Math.sin(a);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.strokeStyle = `rgba(217,30,99,${op})`;
    ctx.lineWidth = 1; ctx.stroke(); ctx.restore();
  }

  function animate() {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    hexes.forEach(([rx, ry, size]) => drawHex(rx * W, ry * H, size, 0.035));
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(217,30,99,${p.op})`;
      ctx.fill();
    });
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < 140) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(217,30,99,${(1 - d / 140) * 0.12})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }
    animFrame = parentWin.requestAnimationFrame(animate);
  }
  animate();

  function dismiss() {
    parentWin.cancelAnimationFrame(animFrame);
    clearTimeout(dismissTimer);
    overlay.classList.add('fading');
    setTimeout(() => { if (overlay.parentNode) overlay.remove(); }, 580);
  }
  parentDoc.getElementById('vigil-intro-skip').addEventListener('click', dismiss);
  dismissTimer = setTimeout(dismiss, 3800);
})();
</script>
"""


def play_intro_once() -> None:
    """Render the intro animation exactly once per Streamlit session."""
    if st.session_state.get("_intro_played"):
        return
    # Guard before the component call so reruns triggered by the component
    # mount itself don't try to render a second time.
    st.session_state["_intro_played"] = True
    try:
        import streamlit.components.v1 as components
        # height=0 — the iframe contributes no layout; the JS pops an overlay
        # into the parent document instead.
        components.html(_INTRO_COMPONENT_HTML, height=0, scrolling=False)
    except Exception:
        # Never let the intro break the app
        pass


def render_main_header(mode_short: str, customer_name: str | None = None) -> None:
    """Magenta-glow logo + wordmark + mode pill. Mirrors the design header."""
    pill_class = "live" if mode_short == "Live" else "demo"
    pill_label = "Live · Ollama" if mode_short == "Live" else "Demo · Cached"
    tag = f"Adverse Event Classifier · {customer_name}" if customer_name else "Adverse Event Classifier"
    st.markdown(
        f"""
        <div class="vigil-hero">
          <div class="vigil-hero-icon">{_SHIELD_SVG_WHITE}</div>
          <div class="vigil-hero-text">
            <div class="vigil-hero-title">VIGIL</div>
            <div class="vigil-hero-sub">{tag}</div>
          </div>
          <span class="vigil-mode-pill {pill_class}"><span class="dot"></span>{pill_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_logo() -> None:
    """Dark-sidebar magenta-glow logo block."""
    st.markdown(
        f"""
        <div class="vigil-sb-logo">
          <div class="vigil-sb-logo-icon">{_SHIELD_SVG_WHITE}</div>
          <div class="vigil-sb-wordmark">VIGIL</div>
        </div>
        <div class="vigil-sb-tagline">Pharmacovigilance</div>
        """,
        unsafe_allow_html=True,
    )


def _conf_class(conf: float) -> str:
    if conf >= 0.8:
        return "high"
    if conf >= 0.5:
        return "med"
    return "low"


def stat_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="stat-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>'
    )


# --------------------------------------------------------------------------- #
# Caching / data loaders                                                       #
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_test_narratives() -> list[str]:
    if not TEST_NARRATIVES_PATH.exists():
        return []
    with open(TEST_NARRATIVES_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_demo_results() -> list[dict]:
    if not DEMO_RESULTS_PATH.exists():
        return []
    with open(DEMO_RESULTS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_validation_metrics() -> dict | None:
    if not VALIDATION_RESULTS_PATH.exists():
        return None
    with open(VALIDATION_RESULTS_PATH) as f:
        return json.load(f).get("metrics")


@st.cache_data(show_spinner=False, ttl=30)
def ollama_available() -> bool:
    """Ping Ollama's /api/tags endpoint with a short timeout."""
    try:
        resp = requests.get(
            OLLAMA_URL.replace("/api/generate", "/api/tags"),
            timeout=2,
        )
        return resp.status_code == 200
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Classification helpers                                                       #
# --------------------------------------------------------------------------- #

def classify_live(narrative: str, customer_id: str | None = None) -> tuple[dict, str | None]:
    """
    Lazy import so Streamlit Cloud (no chromadb/ollama) can still load the UI.
    Returns (report_dict, report_id). report_id is set when customer_id is supplied.
    """
    from pipeline.classify import classify_report
    report = classify_report(narrative, customer_id=customer_id)
    rid = report.__dict__.get("_report_id")
    return report.model_dump(), rid


def classify_demo(narrative: str) -> dict | None:
    """Match narrative against the pre-cached demo results."""
    demo = load_demo_results()
    target = narrative.strip()
    # Exact match first
    for entry in demo:
        if entry.get("narrative", "").strip() == target and "report" in entry:
            return entry["report"]
    # Fallback: first 60-char prefix match
    prefix = target[:60]
    for entry in demo:
        if entry.get("narrative", "").startswith(prefix) and "report" in entry:
            return entry["report"]
    return None


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #

def _reaction_df(coded_reactions: list[dict]) -> pd.DataFrame:
    if not coded_reactions:
        return pd.DataFrame(columns=["Verbatim", "MedDRA PT", "PT Code", "SOC", "Confidence"])
    return pd.DataFrame([
        {
            "Verbatim": m.get("verbatim_term", ""),
            "MedDRA PT": m.get("pt_name", ""),
            "PT Code": m.get("pt_code", ""),
            "SOC": m.get("soc_name", ""),
            "Confidence": round(m.get("confidence", 0.0), 3),
        }
        for m in coded_reactions
    ])


def _drug_df(drugs: list[dict]) -> pd.DataFrame:
    if not drugs:
        return pd.DataFrame(columns=["Name", "Dose", "Route", "Indication"])
    return pd.DataFrame([
        {
            "Name": d.get("name", ""),
            "Dose": d.get("dose") or "—",
            "Route": d.get("route") or "—",
            "Indication": d.get("indication") or "—",
        }
        for d in drugs
    ])


def _render_severity_banner(report: dict) -> None:
    is_serious = report.get("is_serious", False)
    criteria_hit = [k.replace("_", " ").title()
                    for k, v in report.get("seriousness_criteria", {}).items() if v]
    label = "🚨 SERIOUS EVENT" if is_serious else "✅ Non-serious event"
    reason = (
        f"FDA criteria met: {', '.join(criteria_hit)}" if (is_serious and criteria_hit)
        else ("No FDA seriousness criteria were met." if not is_serious else "Criteria met.")
    )
    conf = report.get("severity_confidence", 0.0) or 0.0
    conf_cls = _conf_class(conf)
    cls = "serious" if is_serious else "non-serious"
    st.markdown(
        f"""
        <div class="sev-banner {cls}">
          <div>
            <div class="label">{label}</div>
            <div class="reason">{reason}</div>
          </div>
          <div class="conf-block">
            <div class="conf-num {conf_cls}">{conf:.2f}</div>
            <div class="conf-lbl">Severity confidence</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_reactions_html(coded_reactions: list[dict]) -> None:
    if not coded_reactions:
        st.info("No reactions coded.")
        return
    rows = []
    for m in coded_reactions:
        verbatim = (m.get("verbatim_term") or "").strip()
        pt_name = (m.get("pt_name") or "Unknown").strip()
        pt_code = (m.get("pt_code") or "").strip()
        soc = (m.get("soc_name") or "").strip()
        conf = float(m.get("confidence", 0.0) or 0.0)
        conf_cls = _conf_class(conf)
        soc_html = f'<span class="soc-chip">{soc}</span>' if soc else ""
        conf_chip = (
            f'<span class="conf-chip {conf_cls}">{conf:.2f}</span>'
        )
        pill = f'<span class="meddra-pill">{pt_code or "—"}</span>' if pt_code else ""
        rows.append(
            f'<div class="reaction-row">'
            f'  <div class="reaction-top">'
            f'    <div><div class="reaction-term">{verbatim}</div>'
            f'      <div class="reaction-pt">→ {pt_name}</div>'
            f'      <div>{soc_html}{conf_chip}</div></div>'
            f'    {pill}'
            f'  </div>'
            f'</div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def _render_drugs_html(drugs: list[dict], empty_msg: str) -> None:
    if not drugs:
        st.markdown(f'<div style="color:var(--text-3);font-size:12px;padding:8px 0;">{empty_msg}</div>',
                    unsafe_allow_html=True)
        return
    rows = []
    for d in drugs:
        name = (d.get("name") or "—").strip()
        tags = []
        for key in ("dose", "route", "indication"):
            v = d.get(key)
            if v:
                tags.append(f'<span class="drug-tag">{key.title()}: {v}</span>')
        tag_html = f'<div class="drug-meta">{"".join(tags)}</div>' if tags else ""
        rows.append(f'<div class="drug-row"><div class="drug-name">{name}</div>{tag_html}</div>')
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_report(report: dict) -> None:
    """Render a ClassifiedReport dict using the VIGIL v2 design system."""
    # --- Severity banner + top stats ---
    _render_severity_banner(report)

    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        st.metric("Reactions coded", len(report.get("coded_reactions", [])))
    with scol2:
        st.metric("Suspect drugs", len(report.get("suspect_drugs", [])))
    with scol3:
        flags = len(report.get("flags_for_review", []) or [])
        st.metric("Flags for review", flags)

    # --- Patient ---
    with st.expander("👤 Patient Demographics", expanded=True):
        patient = report.get("patient") or {}
        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric("Age", patient.get("age") or "—")
        pcol2.metric("Sex", patient.get("sex") or "—")
        pcol3.metric("Weight", patient.get("weight") or "—")

    # --- Drugs ---
    with st.expander("💊 Suspect Drugs", expanded=True):
        _render_drugs_html(report.get("suspect_drugs", []), "No suspect drugs extracted.")

    with st.expander("💊 Concomitant Drugs"):
        _render_drugs_html(report.get("concomitant_drugs", []), "No concomitant drugs extracted.")

    # --- Reactions ---
    with st.expander("🩺 Adverse Reactions & MedDRA Codes", expanded=True):
        _render_reactions_html(report.get("coded_reactions", []))

    # --- Timeline / outcome ---
    with st.expander("📅 Timeline & Outcome"):
        for label, key in [
            ("Onset", "onset_timeline"),
            ("Dechallenge", "dechallenge"),
            ("Outcome", "outcome"),
            ("Reporter", "reporter_type"),
        ]:
            val = report.get(key)
            st.write(f"**{label}:** {val if val else '—'}")

    # --- Severity detail ---
    with st.expander("⚖️ Severity Assessment"):
        criteria = report.get("seriousness_criteria", {}) or {}
        if not criteria:
            st.info("No severity criteria evaluated.")
        else:
            cdf = pd.DataFrame([
                {"Criterion": k.replace("_", " ").title(), "Met": "✅" if v else "—"}
                for k, v in criteria.items()
            ])
            st.dataframe(cdf, use_container_width=True, hide_index=True)

    # --- Flags ---
    flags = report.get("flags_for_review", [])
    if flags:
        with st.expander(f"⚠️ Flags for Review ({len(flags)})", expanded=True):
            chips = "".join(f'<span class="fda-flag">⚠ {f}</span>' for f in flags)
            st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)

    # --- Download ---
    st.download_button(
        "⬇️ Download JSON",
        data=json.dumps(report, indent=2),
        file_name=f"vigil_report_{int(time.time())}.json",
        mime="application/json",
    )


# --------------------------------------------------------------------------- #
# Tabs                                                                         #
# --------------------------------------------------------------------------- #

def render_corrections_ui(report: dict, report_id: str) -> None:
    """
    Corrections panel rendered below a classified report. The user can:
      - Reassign any MedDRA code by picking from its top-5 candidates (or
        typing a replacement PT name / code).
      - Toggle any seriousness criterion on or off.
    """
    st.divider()
    st.subheader("✏️ Corrections — help VIGIL learn")
    st.caption(
        "Your corrections are saved to this organization's private history. "
        "After the same correction is made twice, VIGIL will apply it "
        "automatically to future reports."
    )

    # --- MedDRA corrections ---
    meddra_corrections: list[dict] = []
    coded = report.get("coded_reactions", []) or []
    if coded:
        st.markdown("**Reaction → MedDRA PT**")
        for i, match in enumerate(coded):
            verbatim = match.get("verbatim_term", "")
            current_pt = match.get("pt_name", "")
            candidates = match.get("candidates", []) or []

            # Build options: current choice + all RAG candidates + "keep as-is"
            option_keys = []
            option_labels = ["— keep as-is —"]
            # first the existing candidates, then the current pick
            seen = set()
            for c in candidates:
                code = c.get("pt_code", "")
                if code and code not in seen:
                    option_keys.append(c)
                    option_labels.append(
                        f"{c.get('pt_name','')} ({code})"
                        + (f"  sim={c.get('similarity',0):.2f}" if c.get("similarity") else "")
                    )
                    seen.add(code)

            cols = st.columns([3, 4])
            cols[0].markdown(f"`{verbatim}` → **{current_pt}**")
            choice = cols[1].selectbox(
                "Replace with",
                options=range(len(option_labels)),
                format_func=lambda i, lbl=option_labels: lbl[i],
                key=f"corr_{report_id}_{i}",
                label_visibility="collapsed",
            )
            if choice > 0:
                new_c = option_keys[choice - 1]
                if new_c.get("pt_code") != match.get("pt_code"):
                    meddra_corrections.append({
                        "field_type": "meddra",
                        "verbatim_term": verbatim,
                        "original": {
                            "pt_name": current_pt,
                            "pt_code": match.get("pt_code", ""),
                            "soc_name": match.get("soc_name", ""),
                        },
                        "corrected": {
                            "pt_name": new_c.get("pt_name", ""),
                            "pt_code": new_c.get("pt_code", ""),
                            "soc_name": new_c.get("soc_name", ""),
                            "hlt_name": new_c.get("hlt_name", ""),
                        },
                    })

    # --- Severity corrections ---
    st.markdown("**Seriousness criteria**")
    criteria = report.get("seriousness_criteria", {}) or {}
    severity_corrections: list[dict] = []
    sev_cols = st.columns(3)
    for i, (k, v) in enumerate(criteria.items()):
        col = sev_cols[i % 3]
        new_v = col.checkbox(
            k.replace("_", " ").title(),
            value=bool(v),
            key=f"sev_{report_id}_{k}",
        )
        if new_v != bool(v):
            severity_corrections.append({
                "field_type": "severity",
                "criterion": k,
                "original": bool(v),
                "corrected": new_v,
            })

    all_corrections = meddra_corrections + severity_corrections
    if st.button(
        "💾 Save Corrections",
        type="primary",
        disabled=not all_corrections,
        key=f"save_{report_id}",
    ):
        customer_id = st.session_state.get("customer_id")
        if not customer_id:
            st.error("No customer context — cannot save feedback.")
            return
        from pipeline.feedback import save_feedback
        from pipeline.adaptive import record_correction

        save_feedback(customer_id, report_id, all_corrections)
        for corr in meddra_corrections:
            record_correction(customer_id, corr["verbatim_term"], corr["corrected"])

        learned = sum(
            1 for corr in meddra_corrections
            if corr["corrected"].get("pt_code")
        )
        st.success(
            f"Saved {len(all_corrections)} correction(s). "
            f"{learned} MedDRA mapping(s) recorded — VIGIL will prioritize these "
            "after the same correction is seen twice."
        )
        # Clear the corrections so the UI resets cleanly on rerun
        st.session_state["last_report_id"] = None


def _run_classification(narrative: str, mode: str, button_key: str) -> None:
    """
    Shared classify-button + render block used by all three input sub-tabs.
    Each sub-tab gives its narrative source and a unique button_key.
    """
    if st.button(
        "🔬 Classify Report",
        type="primary",
        disabled=not narrative.strip(),
        key=button_key,
    ):
        customer_id = st.session_state.get("customer_id")
        with st.spinner(f"Classifying in {mode} mode..."):
            t0 = time.time()
            if mode == "Live":
                try:
                    report, rid = classify_live(narrative, customer_id=customer_id)
                except Exception as e:
                    st.error(f"Pipeline failed: {type(e).__name__}: {e}")
                    return
            else:
                report = classify_demo(narrative)
                rid = None
                if report is None:
                    st.warning(
                        "No matching pre-cached result for this narrative. "
                        "Switch to Live mode (requires local Ollama) or pick an example."
                    )
                    return
            elapsed = time.time() - t0
            st.caption(f"Processed in {elapsed:.1f}s")
            st.session_state.setdefault("session_reports", []).append(report)
            # Stash the last report + its persistence id so the corrections UI
            # (rendered below) can reach them across reruns.
            st.session_state["last_report"] = report
            st.session_state["last_report_id"] = rid
            st.session_state["last_narrative"] = narrative

    # Render the last classification (persists across reruns triggered by
    # the corrections UI below)
    last_report = st.session_state.get("last_report")
    if last_report is not None:
        render_report(last_report)
        last_rid = st.session_state.get("last_report_id")
        if last_rid and mode == "Live":
            render_corrections_ui(last_report, last_rid)


def _subtab_paste(mode: str) -> None:
    narratives = load_test_narratives()
    example_labels = ["— Select an example —"] + [
        f"{i + 1}. {n[:90].strip()}..." for i, n in enumerate(narratives)
    ]

    def _on_example_change():
        idx = st.session_state.get("example_selector", 0)
        if idx > 0:
            st.session_state["narrative_input_widget"] = narratives[idx - 1]

    st.selectbox(
        "Example reports",
        options=range(len(example_labels)),
        format_func=lambda i: example_labels[i],
        index=0,
        key="example_selector",
        on_change=_on_example_change,
    )

    narrative = st.text_area(
        "Narrative",
        height=180,
        key="narrative_input_widget",
        placeholder="Paste or type an adverse event narrative here...",
    )

    _run_classification(narrative, mode, button_key="classify_paste")


def _subtab_document(mode: str) -> None:
    st.caption(
        "Upload a photo, scan, or PDF of an adverse event report. "
        "Text is extracted via Tesseract OCR, then you can review and edit "
        "it before classification."
    )

    if mode == "Demo":
        st.info(
            "📄 Document/audio upload requires local installation. "
            "See the README for setup instructions (`brew install tesseract poppler`). "
            "Demo Mode supports pasted text with pre-cached results."
        )
        return

    # Probe dependencies before showing the uploader
    from pipeline.ocr import (
        OCRDependencyError,
        extract_text_from_image,
        extract_text_from_pdf,
        is_available,
    )
    ok, reason = is_available()
    if not ok:
        st.error(f"OCR unavailable.\n\n```\n{reason}\n```")
        return

    uploaded = st.file_uploader(
        "Upload document",
        type=["png", "jpg", "jpeg", "pdf"],
        key="ocr_uploader",
    )

    if uploaded is not None:
        # Re-run OCR only when the file changes (compare name+size)
        fingerprint = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("ocr_fingerprint") != fingerprint:
            file_bytes = uploaded.read()
            try:
                with st.spinner("Extracting text from document..."):
                    if uploaded.name.lower().endswith(".pdf"):
                        extracted = extract_text_from_pdf(file_bytes)
                    else:
                        extracted = extract_text_from_image(file_bytes, uploaded.name)
            except OCRDependencyError as e:
                st.error(f"OCR dependency missing:\n\n```\n{e}\n```")
                return
            except ValueError as e:
                st.error(f"Could not read file: {e}")
                return

            if not extracted:
                st.warning(
                    "OCR returned no text. Try a higher-resolution image or check "
                    "that the document contains readable text (not pure handwriting)."
                )

            st.session_state["ocr_fingerprint"] = fingerprint
            st.session_state["ocr_text_widget"] = extracted

    narrative = st.text_area(
        "Extracted text (edit to fix OCR errors)",
        height=220,
        key="ocr_text_widget",
        placeholder="OCR output will appear here after upload…",
    )

    _run_classification(narrative, mode, button_key="classify_document")


def _subtab_audio(mode: str) -> None:
    st.caption(
        "Upload a voice recording (doctor dictation or patient call). "
        "Audio is transcribed locally with Whisper, then you can review and "
        "edit the transcript before classification."
    )

    if mode == "Demo":
        st.info(
            "🎙️ Document/audio upload requires local installation. "
            "See the README for setup instructions (`pip install openai-whisper` "
            "+ `brew install ffmpeg`). Demo Mode supports pasted text with "
            "pre-cached results."
        )
        return

    from pipeline.transcriber import (
        TranscriberDependencyError,
        is_available,
        transcribe_audio,
    )
    ok, reason = is_available()
    if not ok:
        st.error(f"Audio transcription unavailable.\n\n```\n{reason}\n```")
        return

    uploaded = st.file_uploader(
        "Upload audio",
        type=["mp3", "wav", "m4a", "ogg"],
        key="audio_uploader",
    )

    if uploaded is not None:
        fingerprint = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("audio_fingerprint") != fingerprint:
            file_bytes = uploaded.read()
            try:
                with st.spinner(
                    "Transcribing audio (first run downloads Whisper ~140MB)..."
                ):
                    transcript = transcribe_audio(file_bytes, uploaded.name)
            except TranscriberDependencyError as e:
                st.error(f"Transcriber dependency missing:\n\n```\n{e}\n```")
                return
            except ValueError as e:
                st.error(f"Could not transcribe file: {e}")
                return

            if not transcript:
                st.warning("Whisper returned no text. The audio may be empty or silent.")

            st.session_state["audio_fingerprint"] = fingerprint
            st.session_state["audio_text_widget"] = transcript

        # Let the user listen back
        st.audio(uploaded)

    narrative = st.text_area(
        "Transcribed text (edit to fix transcription errors)",
        height=220,
        key="audio_text_widget",
        placeholder="Transcript will appear here after upload…",
    )

    _run_classification(narrative, mode, button_key="classify_audio")


def tab_classify(mode: str) -> None:
    st.subheader("Classify a single report")

    paste_tab, doc_tab, audio_tab = st.tabs([
        "📝 Paste Text",
        "📄 Upload Document",
        "🎙️ Upload Audio",
    ])
    with paste_tab:
        _subtab_paste(mode)
    with doc_tab:
        _subtab_document(mode)
    with audio_tab:
        _subtab_audio(mode)


def tab_batch(mode: str) -> None:
    st.subheader("Batch process reports from CSV")
    st.caption("CSV must include columns: `id`, `narrative`.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    if "narrative" not in df.columns:
        st.error("CSV must have a 'narrative' column.")
        return
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    st.write(f"Loaded **{len(df)}** rows.")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("▶ Process batch", type="primary"):
        rows: list[dict] = []
        progress = st.progress(0.0, text="Starting...")

        for i, row in enumerate(df.itertuples(index=False), start=1):
            narrative = str(getattr(row, "narrative", "") or "").strip()
            rid = getattr(row, "id", i)
            progress.progress(i / len(df), text=f"Processing {i}/{len(df)}...")
            if not narrative:
                rows.append({"id": rid, "error": "empty narrative"})
                continue
            try:
                if mode == "Live":
                    report, _ = classify_live(
                        narrative,
                        customer_id=st.session_state.get("customer_id"),
                    )
                else:
                    report = classify_demo(narrative) or {}
                if not report:
                    rows.append({"id": rid, "error": "no demo match (use Live mode)"})
                    continue
                st.session_state.setdefault("session_reports", []).append(report)
                rows.append({
                    "id": rid,
                    "is_serious": report.get("is_serious"),
                    "severity_confidence": report.get("severity_confidence"),
                    "n_reactions": len(report.get("coded_reactions", [])),
                    "top_reactions": "; ".join(
                        m.get("pt_name", "") for m in report.get("coded_reactions", [])[:3]
                    ),
                    "suspect_drugs": "; ".join(
                        d.get("name", "") for d in report.get("suspect_drugs", [])
                    ),
                    "flags": len(report.get("flags_for_review", [])),
                })
            except Exception as e:
                rows.append({"id": rid, "error": f"{type(e).__name__}: {e}"})

        progress.empty()
        out = pd.DataFrame(rows)
        st.success(f"Processed {len(out)} rows.")
        st.dataframe(out, use_container_width=True)

        csv_buf = io.StringIO()
        out.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download results CSV",
            data=csv_buf.getvalue(),
            file_name="vigil_batch_results.csv",
            mime="text/csv",
        )


def tab_learning(customer_id: str, customer_name: str) -> None:
    """Learning Analytics tab — shows how adaptive learning is progressing."""
    from pipeline.adaptive import get_custom_terms
    from pipeline.analytics import get_learning_metrics

    st.subheader("📈 Learning Analytics")
    st.caption(
        f"Private learning progress for **{customer_name}**. "
        "Nothing here is shared across organizations."
    )

    metrics = get_learning_metrics(customer_id)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card(
        "Reports processed", str(metrics["total_reports_processed"])
    ), unsafe_allow_html=True)
    c2.markdown(stat_card(
        "Corrections made", str(metrics["total_corrections_made"]),
        "User-supplied MedDRA or severity fixes",
    ), unsafe_allow_html=True)
    c3.markdown(stat_card(
        "Correction rate", f"{metrics['correction_rate']:.0%}",
        "Fraction of reports the user corrected at least once",
    ), unsafe_allow_html=True)
    c4.markdown(stat_card(
        "Authoritative terms", str(metrics["authoritative_terms_count"]),
        "Seen ≥2 times · applied automatically",
    ), unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    improvement = metrics["estimated_accuracy_improvement"]
    if metrics["total_reports_processed"] >= 4:
        if improvement > 0.05:
            st.success(
                f"📉 Correction rate dropped **{improvement:.0%}** between the first "
                "and second halves of reports. Learning is working."
            )
        elif improvement < -0.05:
            st.warning(
                f"⚠️ Correction rate rose **{abs(improvement):.0%}** over time. "
                "Review your corrections — VIGIL may be misapplying a learned term."
            )
        else:
            st.info("Correction rate is roughly flat — too early to measure improvement.")

    # Badge
    n_auth = metrics["authoritative_terms_count"]
    if n_auth > 0:
        st.markdown(
            f"### 🎓 VIGIL has learned **{n_auth}** custom term"
            f"{'s' if n_auth != 1 else ''} for {customer_name}"
        )

    # --- Trend chart ---
    trend = metrics.get("trend_points", [])
    if len(trend) >= 2:
        trend_df = pd.DataFrame(trend)
        fig = px.line(
            trend_df,
            x="report_index",
            y="correction_rate",
            title="Cumulative correction rate (lower = better)",
            markers=True,
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        fig.update_xaxes(title="Report #")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Classify at least 2 reports to see the correction-rate trend.")

    # --- Top corrected terms ---
    st.markdown("#### Top corrected verbatim terms")
    top = metrics.get("top_corrected_terms", [])
    if top:
        st.dataframe(
            pd.DataFrame(top).rename(columns={"term": "Verbatim", "count": "Corrections"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No corrections recorded yet.")

    # --- Custom terms table ---
    st.markdown("#### Learned MedDRA mappings")
    custom = get_custom_terms(customer_id)
    if custom:
        rows = []
        for term, entry in custom.items():
            freq = entry.get("frequency", 0)
            rows.append({
                "Verbatim": term,
                "MedDRA PT": entry.get("pt_name", ""),
                "PT Code": entry.get("pt_code", ""),
                "SOC": entry.get("soc_name", ""),
                "Corrections": freq,
                "Status": "✅ Authoritative" if freq >= 2 else "⏳ Pending",
            })
        rows.sort(key=lambda r: r["Corrections"], reverse=True)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "Terms become **authoritative** after 2 matching corrections — "
            "at that point VIGIL applies them automatically."
        )
    else:
        st.caption("No custom mappings recorded yet. Correct a MedDRA code in the Classify tab to start teaching VIGIL.")


def tab_dashboard() -> None:
    st.subheader("Session dashboard")
    reports = st.session_state.get("session_reports", [])

    if not reports:
        st.info("Classify at least one report to populate the dashboard.")
        return

    st.caption(f"Showing stats across {len(reports)} session report(s).")

    col1, col2 = st.columns(2)

    # --- Top 10 reactions ---
    with col1:
        rxn_counts: dict[str, int] = {}
        for r in reports:
            for m in r.get("coded_reactions", []):
                name = m.get("pt_name") or "Unknown"
                rxn_counts[name] = rxn_counts.get(name, 0) + 1
        if rxn_counts:
            rxn_df = (
                pd.DataFrame(
                    [{"Reaction": k, "Count": v} for k, v in rxn_counts.items()]
                )
                .sort_values("Count", ascending=False)
                .head(10)
            )
            fig = px.bar(
                rxn_df, x="Count", y="Reaction", orientation="h",
                title="Top 10 adverse reactions",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No reactions yet.")

    # --- Serious vs non-serious ---
    with col2:
        serious = sum(1 for r in reports if r.get("is_serious"))
        non_serious = len(reports) - serious
        pie_df = pd.DataFrame([
            {"Severity": "Serious", "Count": serious},
            {"Severity": "Non-serious", "Count": non_serious},
        ])
        fig = px.pie(
            pie_df, values="Count", names="Severity",
            title="Serious vs non-serious",
            color="Severity",
            color_discrete_map={"Serious": "#D64545", "Non-serious": "#49A078"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Confidence histogram ---
    confidences = [
        m.get("confidence", 0.0)
        for r in reports
        for m in r.get("coded_reactions", [])
    ]
    if confidences:
        conf_df = pd.DataFrame({"Confidence": confidences})
        fig = px.histogram(
            conf_df, x="Confidence", nbins=20,
            title="MedDRA match confidence distribution",
        )
        fig.add_vline(
            x=CONFIDENCE_THRESHOLD, line_dash="dash", line_color="red",
            annotation_text=f"Threshold ({CONFIDENCE_THRESHOLD})",
        )
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_update_check() -> dict:
    from pipeline.updater import check_for_updates
    return check_for_updates()


def render_onboarding() -> None:
    """First-launch gate. Asks for an org name or lets the user pick an existing one."""
    st.markdown(
        f"""
        <div class="vigil-hero">
          <div class="vigil-hero-icon">{_SHIELD_SVG_WHITE}</div>
          <div class="vigil-hero-text">
            <div class="vigil-hero-title">VIGIL</div>
            <div class="vigil-hero-sub">Welcome · Choose your organization</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="color:var(--text-2);font-size:13px;line-height:1.6;max-width:680px;">'
        "Each organization (clinic, pharmacy, or med-shop) gets its own <b>private "
        "learning profile</b>. All reports, corrections, and learned MedDRA mappings "
        "stay isolated to your organization — nothing is shared."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    existing = list_customers()

    if existing:
        st.markdown("#### Continue as an existing organization")
        labels = ["— select —"] + [
            f"{c.name}  ·  {c.reports_processed} reports" for c in existing
        ]
        idx = st.selectbox(
            "Organization",
            options=range(len(labels)),
            format_func=lambda i: labels[i],
            key="onboard_existing",
        )
        if idx > 0 and st.button("Continue", type="primary", key="onboard_continue"):
            c = existing[idx - 1]
            st.session_state["customer_id"] = c.customer_id
            st.session_state["customer_name"] = c.name
            st.rerun()
        st.divider()

    st.markdown("#### Create a new organization")
    with st.form("onboard_create"):
        name = st.text_input(
            "Organization name",
            placeholder="e.g. Riverside Family Pharmacy",
        )
        submitted = st.form_submit_button("Create profile", type="primary")
        if submitted:
            if not name.strip():
                st.error("Please enter a name.")
            else:
                c = create_customer(name.strip())
                st.session_state["customer_id"] = c.customer_id
                st.session_state["customer_name"] = c.name
                st.success(f"Created profile for **{c.name}**.")
                st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="VIGIL - Adverse Event Classifier",
        layout="wide",
        page_icon="🩺",
        initial_sidebar_state="expanded",
    )
    inject_design_system()
    play_intro_once()

    # --- Onboarding gate ---
    if not st.session_state.get("customer_id"):
        render_onboarding()
        return

    customer_id = st.session_state["customer_id"]
    customer = load_customer(customer_id)
    if customer is None:
        # Profile was deleted underneath us — restart onboarding
        st.session_state.pop("customer_id", None)
        st.session_state.pop("customer_name", None)
        st.rerun()
    customer_name = customer.name

    # --- Sidebar ---
    ollama_ok = ollama_available()

    with st.sidebar:
        render_sidebar_logo()
        # --- Organization ---
        st.header("Organization")
        st.markdown(
            f"<div style='color:var(--sidebar-text);font-weight:600;font-size:13px;'>"
            f"{customer_name}</div>"
            f"<div style='color:var(--sidebar-text-2);font-size:10px;font-family:\"IBM Plex Mono\",monospace;"
            f"margin-top:2px;'>{customer_id}</div>",
            unsafe_allow_html=True,
        )
        ocol1, ocol2 = st.columns(2)
        ocol1.metric("Reports", customer.reports_processed)
        ocol2.metric("Learned", customer.custom_mappings_count)
        if st.button("Switch organization", use_container_width=True):
            st.session_state.pop("customer_id", None)
            st.session_state.pop("customer_name", None)
            st.session_state.pop("last_report", None)
            st.session_state.pop("last_report_id", None)
            st.rerun()

        # --- Update banner ---
        try:
            upd = _cached_update_check()
        except Exception:
            upd = None
        if upd and upd.get("update_available"):
            msg = (
                f"🔔 **Update available**: "
                f"v{upd['current_version']} → v{upd['latest_version']}"
            )
            if upd.get("meddra_update_available"):
                msg += "\n\nThis release ships **new MedDRA terms**."
            st.info(msg)

        st.divider()
        st.header("Settings")

        if ollama_ok:
            mode = st.radio(
                "Mode",
                options=["Live (Ollama)", "Demo (Pre-cached)"],
                index=0,
                help="Live runs the full pipeline. Demo uses pre-computed results.",
            )
        else:
            st.info("Ollama not detected on localhost:11434 — running in **Demo** mode.")
            mode = "Demo (Pre-cached)"

        mode_short = "Live" if mode.startswith("Live") else "Demo"

        st.divider()
        st.header("About")
        st.markdown(
            "**VIGIL** is a pharmacovigilance tool that takes raw adverse event "
            "text and outputs structured, MedDRA-coded safety reports. Runs 100% "
            "locally via Ollama + Gemma 2B."
        )
        st.markdown(f"[📖 GitHub]({GITHUB_URL})")

        st.divider()
        st.header("Validation")
        metrics = load_validation_metrics()
        if metrics:
            st.metric("Severity accuracy", f"{metrics['severity_accuracy']:.0%}")
            st.metric("SOC accuracy", f"{metrics['soc_accuracy']:.0%}")
            st.metric("MedDRA PT F1", f"{metrics['pt_f1']:.2f}")
            st.caption(f"Against {metrics['n_reports']} FAERS reports")
        else:
            st.caption("No validation results found.")

    # --- Hero header ---
    render_main_header(mode_short, customer_name)

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Classify", "📊 Batch", "📈 Dashboard", "🎓 Learning",
    ])
    with tab1:
        tab_classify(mode_short)
    with tab2:
        tab_batch(mode_short)
    with tab3:
        tab_dashboard()
    with tab4:
        tab_learning(customer_id, customer_name)


if __name__ == "__main__":
    main()
