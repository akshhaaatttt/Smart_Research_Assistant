"""Visual theme for the Streamlit app: injected CSS plus small HTML components
(hero banner, KPI cards, source cards). Pure Streamlit — no external assets."""

from __future__ import annotations

import html
from typing import List, Tuple

import streamlit as st

PRIMARY = "#0d9488"       # teal-600
PRIMARY_DARK = "#0f766e"  # teal-700
PRIMARY_LIGHT = "#14b8a6" # teal-500

_CSS = """
<style>
/* ---- base ---- */
.stApp { background: linear-gradient(180deg,#f7fbfb 0%,#eef5f4 100%); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1320px; }
h1,h2,h3 { color:#0f172a; }

/* ---- hero banner ---- */
.hero {
  background: linear-gradient(120deg,#0d9488 0%,#14b8a6 55%,#2dd4bf 100%);
  border-radius: 18px; padding: 24px 30px; margin-bottom: 18px;
  box-shadow: 0 12px 34px -12px rgba(13,148,136,.55);
  display:flex; align-items:center; gap:18px;
}
.hero .logo { font-size: 2.6rem; line-height:1; }
.hero h1 { color:#fff; font-size:1.8rem; font-weight:800; margin:0; letter-spacing:-.01em; }
.hero p  { color:rgba(255,255,255,.92); margin:5px 0 0; font-size:.95rem; max-width:760px; }
.hero .pill {
  margin-left:auto; background:rgba(255,255,255,.18); color:#fff;
  border:1px solid rgba(255,255,255,.35); border-radius:999px;
  padding:6px 14px; font-size:.8rem; font-weight:700; white-space:nowrap;
}

/* ---- KPI cards ---- */
.kpi-row { display:flex; gap:14px; margin:2px 0 20px; flex-wrap:wrap; }
.kpi {
  flex:1; min-width:150px; background:#fff; border:1px solid #e2e8f0;
  border-radius:14px; padding:14px 18px; box-shadow:0 1px 3px rgba(15,23,42,.05);
}
.kpi .k-label { font-size:.7rem; text-transform:uppercase; letter-spacing:.07em;
  color:#64748b; font-weight:700; }
.kpi .k-value { font-size:1.55rem; font-weight:800; color:#0f172a; line-height:1.25; }
.kpi .k-sub { font-size:.74rem; color:#0d9488; font-weight:700; }

/* ---- buttons ---- */
.stButton > button {
  border-radius:10px; font-weight:600; border:1px solid #cbd5e1; transition:all .15s;
}
.stButton > button:hover { border-color:#0d9488; color:#0d9488; }
.stButton > button[kind="primary"] { background:#0d9488; border-color:#0d9488; color:#fff; }
.stButton > button[kind="primary"]:hover { background:#0f766e; border-color:#0f766e; color:#fff; }

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap:4px; }
.stTabs [data-baseweb="tab"] {
  border-radius:9px 9px 0 0; padding:8px 18px; font-weight:600; color:#64748b;
}
.stTabs [aria-selected="true"] { color:#0d9488 !important; background:#f0fdfa; }
.stTabs [data-baseweb="tab-highlight"] { background:#0d9488; }

/* ---- metric cards ---- */
[data-testid="stMetric"] {
  background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:14px 18px;
  box-shadow:0 1px 3px rgba(15,23,42,.05);
}
[data-testid="stMetricValue"] { color:#0f172a; font-weight:800; }

/* ---- expander & dataframe ---- */
[data-testid="stExpander"] { border:1px solid #e2e8f0; border-radius:12px; }
[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; border:1px solid #e2e8f0; }

/* ---- sidebar ---- */
[data-testid="stSidebar"] { border-right:1px solid #e2e8f0; }
.side-card {
  background:#fff; border:1px solid #e2e8f0; border-radius:12px;
  padding:12px 14px; font-size:.8rem; line-height:1.75; margin-top:6px;
}
.side-card b { color:#0f172a; }
.badge { display:inline-block; background:#0d9488; color:#fff; border-radius:999px;
  padding:2px 9px; font-size:.7rem; font-weight:700; }

/* ---- source cards (chat) ---- */
.src-card {
  background:#fff; border:1px solid #e2e8f0; border-left:3px solid #0d9488;
  border-radius:8px; padding:9px 12px; margin-bottom:8px;
}
.src-head { font-weight:700; color:#0f172a; font-size:.84rem; }
.src-sim  { float:right; color:#0d9488; font-weight:700; font-size:.76rem; }
.src-body { color:#475569; font-size:.82rem; margin-top:4px; }
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, pill: str = "") -> None:
    pill_html = f'<span class="pill">{html.escape(pill)}</span>' if pill else ""
    st.markdown(
        f'<div class="hero"><span class="logo">📚</span>'
        f'<div><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>'
        f"{pill_html}</div>",
        unsafe_allow_html=True,
    )


def kpi_row(items: List[Tuple[str, str, str]]) -> None:
    """Render a row of KPI cards from (label, value, sub) tuples."""
    cards = "".join(
        f'<div class="kpi"><div class="k-label">{html.escape(label)}</div>'
        f'<div class="k-value">{html.escape(str(value))}</div>'
        f'<div class="k-sub">{html.escape(sub)}</div></div>'
        for label, value, sub in items
    )
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)


def source_card(head: str, similarity: float, body: str) -> None:
    st.markdown(
        f'<div class="src-card"><span class="src-sim">sim {similarity:.3f}</span>'
        f'<div class="src-head">{html.escape(head)}</div>'
        f'<div class="src-body">{html.escape(body)}</div></div>',
        unsafe_allow_html=True,
    )
