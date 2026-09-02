"""
TalentPulse AI — Enterprise Workforce Intelligence & Competency Platform
Full Multi-Page Website Experience with Top Navigation Bar, Hero Home Page & Interactive Analytics
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
import requests
import sys
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.services.llm_chat_service import build_system_context, generate_llm_response

# ── 1. Page Configuration ──
st.set_page_config(
    page_title="TalentPulse AI | Enterprise Workforce Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Design System & CSS Custom Properties ──
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

  :root {
    --bg-primary: #0a0d14;
    --bg-secondary: #101522;
    --bg-card: rgba(20, 27, 44, 0.75);
    --bg-card-hover: rgba(28, 38, 62, 0.9);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-focus: rgba(99, 102, 241, 0.45);
    
    --color-primary: #6366f1;
    --color-primary-glow: rgba(99, 102, 241, 0.25);
    --color-accent: #38bdf8;
    --color-success: #10b981;
    --color-warning: #f59e0b;
    --color-danger: #ef4444;
    
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    
    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --radius-full: 9999px;
  }

  /* Global typography & base */
  html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-main);
  }

  .stApp {
    background: radial-gradient(circle at 50% -20%, #172038 0%, var(--bg-primary) 65%);
  }

  /* Sidebar styling */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1320 0%, #080a10 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5);
  }

  /* Top Navigation Bar Header */
  .top-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    background: rgba(16, 22, 36, 0.85);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
  }
  .brand-logo-area {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-logo-badge {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #6366f1 0%, #38bdf8 100%);
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    box-shadow: 0 0 16px var(--color-primary-glow);
  }
  .brand-name {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .brand-tag {
    font-size: 0.72rem;
    color: var(--color-accent);
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: var(--radius-full);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--color-success);
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  /* Hero Section Component */
  .hero-container {
    background: linear-gradient(135deg, rgba(30, 41, 69, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: var(--radius-lg);
    padding: 48px 40px;
    margin-bottom: 28px;
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.45);
    position: relative;
    overflow: hidden;
  }
  .hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, var(--color-primary-glow) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: var(--radius-full);
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--color-accent);
    margin-bottom: 16px;
  }
  .hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin-bottom: 14px;
  }
  .hero-title span {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .hero-desc {
    font-size: 1.05rem;
    color: var(--text-muted);
    max-width: 780px;
    line-height: 1.6;
    margin-bottom: 24px;
  }

  /* Pillar Grid on Home */
  .pillar-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin-bottom: 32px;
  }
  .pillar-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 28px 24px;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
  }
  .pillar-card:hover {
    transform: translateY(-4px);
    background: var(--bg-card-hover);
    border-color: var(--border-focus);
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.4), 0 0 24px var(--color-primary-glow);
  }
  .pillar-icon {
    font-size: 2.2rem;
    margin-bottom: 14px;
  }
  .pillar-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
  }
  .pillar-desc {
    font-size: 0.88rem;
    color: var(--text-muted);
    line-height: 1.5;
  }

  /* KPI Executive Cards */
  .kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }
  .kpi-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
  }
  .kpi-card:hover {
    transform: translateY(-3px);
    background: var(--bg-card-hover);
    border-color: var(--border-focus);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4), 0 0 20px var(--color-primary-glow);
  }
  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, var(--card-accent, var(--color-primary)), transparent);
  }
  .kpi-icon { font-size: 1.6rem; margin-bottom: 6px; }
  .kpi-value { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em; margin: 2px 0 4px 0; color: #ffffff; }
  .kpi-label { font-size: 0.78rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
  .kpi-sub { font-size: 0.72rem; color: var(--text-dim); margin-top: 4px; }

  /* Section Title Bar */
  .section-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--text-main);
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin: 28px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
  }
  .section-bar span { color: var(--color-accent); }

  /* Badges */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: var(--radius-full);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .badge-high { background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }
  .badge-medium { background: rgba(245, 158, 11, 0.15); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.4); }
  .badge-low { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }

  /* Profile Card Component */
  .profile-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 28px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    margin-bottom: 24px;
  }
  .profile-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 20px;
  }
  .profile-name { font-size: 1.6rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; }
  .profile-role { font-size: 0.95rem; color: var(--color-accent); font-weight: 600; margin-top: 2px; }
  .profile-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    padding-top: 14px;
    border-top: 1px solid var(--border-subtle);
  }
  .stat-tile {
    background: rgba(255, 255, 255, 0.025);
    padding: 12px 16px;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(255, 255, 255, 0.04);
  }
  .stat-label { font-size: 0.72rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
  .stat-value { font-size: 1rem; color: var(--text-main); font-weight: 700; margin-top: 2px; }

  /* Chat Styling */
  [data-testid="stChatMessage"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    margin-bottom: 12px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  }

  .stButton button {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
  }
  
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── 3. Configuration & Theme Tokens ──
API_BASE = "http://localhost:8000"

CHART_THEME = {
    "plot_bgcolor": "rgba(20, 27, 44, 0.5)",
    "paper_bgcolor": "rgba(20, 27, 44, 0.0)",
    "font_color": "#e2e8f0",
    "grid_color": "rgba(255, 255, 255, 0.06)",
}

PALETTE = {
    "HIGH": "#ef4444",
    "MEDIUM": "#f59e0b",
    "LOW": "#10b981",
    "primary": "#6366f1",
    "secondary": "#a855f7",
    "accent": "#38bdf8",
}


# ── 4. Cached Data Loaders ──
@st.cache_data(ttl=300)
def load_local_intelligence():
    path = Path(__file__).parent.parent / "data" / "processed" / "employee_intelligence.csv"
    if path.exists(): return pd.read_csv(path)
    return None


@st.cache_data(ttl=300)
def load_org_skill_gap():
    path = Path(__file__).parent.parent / "data" / "processed" / "org_skill_gap.csv"
    if path.exists(): return pd.read_csv(path)
    return None


@st.cache_data(ttl=300)
def load_dept_scores():
    path = Path(__file__).parent.parent / "data" / "processed" / "department_composite_scores.csv"
    if path.exists(): return pd.read_csv(path)
    return None


def apply_chart_theme(fig):
    fig.update_layout(
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        font=dict(color=CHART_THEME["font_color"], family="Plus Jakarta Sans, sans-serif"),
        xaxis=dict(gridcolor=CHART_THEME["grid_color"], linecolor=CHART_THEME["grid_color"]),
        yaxis=dict(gridcolor=CHART_THEME["grid_color"], linecolor=CHART_THEME["grid_color"]),
        margin=dict(t=35, b=35, l=20, r=20),
    )
    return fig


def make_kpi_card(icon, value, label, subtext="", accent_color="#6366f1"):
    return f"""
    <div class="kpi-card" style="--card-accent: {accent_color};">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-label">{label}</div>
      {f'<div class="kpi-sub">{subtext}</div>' if subtext else ''}
    </div>"""


def risk_badge(risk: str) -> str:
    css = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(risk, "badge-low")
    icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk, "🟢")
    return f'<span class="badge {css}">{icon} {risk} RISK</span>'


# ── 5. Navigation & Sidebar Setup ──
NAV_PAGES = [
    "🏠 Home",
    "📊 Executive Overview",
    "⚠️ Attrition Forensics",
    "🎯 O*NET Skill Matrix",
    "👤 Employee Dossier",
    "🤖 AI Workforce Copilot"
]

if "active_page" not in st.session_state:
    st.session_state.active_page = "🏠 Home"

with st.sidebar:
    st.markdown("### ⚡ TalentPulse AI")
    st.caption("Enterprise Workforce Intelligence")
    
    selected_nav = st.radio(
        "Website Navigation",
        NAV_PAGES,
        index=NAV_PAGES.index(st.session_state.active_page) if st.session_state.active_page in NAV_PAGES else 0,
        label_visibility="collapsed",
        key="nav_radio"
    )
    st.session_state.active_page = selected_nav
    
    st.markdown("---")
    st.markdown("### ⚙️ Risk Thresholds")
    
    if "high_thresh" not in st.session_state: st.session_state.high_thresh = 65
    if "med_thresh" not in st.session_state: st.session_state.med_thresh = 40

    high_risk_thresh = st.slider(
        "🔴 High Risk Cutoff", min_value=45, max_value=90,
        value=st.session_state.high_thresh, format="%d%%"
    )
    med_risk_thresh = st.slider(
        "🟡 Medium Risk Cutoff", min_value=15, max_value=min(60, high_risk_thresh - 5),
        value=min(st.session_state.med_thresh, high_risk_thresh - 5), format="%d%%"
    )
    
    st.session_state.high_thresh = high_risk_thresh
    st.session_state.med_thresh = med_risk_thresh
    st.caption(f"🔴 ≥{high_risk_thresh}% | 🟡 {med_risk_thresh}–{high_risk_thresh}% | 🟢 <{med_risk_thresh}%")

    if st.button("↺ Reset Defaults (65% / 40%)", use_container_width=True):
        st.session_state.high_thresh = 65
        st.session_state.med_thresh = 40
        st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 Generative AI Engine")
    llm_provider = st.selectbox("Provider", ["Google Gemini", "OpenAI / Compatible", "Built-in Synthesizer"], index=0)
    llm_api_key, llm_model, llm_base_url = "", "", ""
    if llm_provider == "Google Gemini":
        llm_api_key = st.text_input("Gemini API Key", value=os.environ.get("GEMINI_API_KEY", ""), type="password", placeholder="AIzaSy...")
        llm_model = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"], index=0)
    elif llm_provider == "OpenAI / Compatible":
        llm_api_key = st.text_input("API Key", value=os.environ.get("OPENAI_API_KEY", ""), type="password", placeholder="sk-...")
        llm_model = st.text_input("Model Name", value="gpt-4o-mini")
        llm_base_url = st.text_input("Base URL (Optional)", value="", placeholder="https://api.groq.com/openai/v1")


# ── 6. Load Data & Recalculate Risk ──
intel_df = load_local_intelligence()
org_gap_df = load_org_skill_gap()
dept_scores_df = load_dept_scores()

h_p = high_risk_thresh / 100.0
m_p = med_risk_thresh / 100.0

def assign_user_risk(p):
    if p >= h_p: return "HIGH"
    elif p >= m_p: return "MEDIUM"
    return "LOW"

if intel_df is not None:
    intel_df["Risk_Level"] = intel_df["Attrition_Prob"].apply(assign_user_risk)


# ── 7. Top Navigation Bar (Header) ──
st.markdown(f"""
<div class="top-navbar">
  <div class="brand-logo-area">
    <div class="brand-logo-badge">⚡</div>
    <div>
      <div class="brand-name">TalentPulse <span style="font-weight:400; color:#818cf8;">AI</span></div>
      <div class="brand-tag">Workforce Intelligence & Upskilling Platform</div>
    </div>
  </div>
  <div>
    <span class="status-badge">● Engine Online & Live</span>
  </div>
</div>
""", unsafe_allow_html=True)


# Quick Jump Web Navbar Tabs
nav_cols = st.columns(len(NAV_PAGES))
for idx, page_name in enumerate(NAV_PAGES):
    with nav_cols[idx]:
        is_active = (st.session_state.active_page == page_name)
        button_type = "primary" if is_active else "secondary"
        if st.button(page_name, key=f"top_nav_{idx}", use_container_width=True, type=button_type):
            st.session_state.active_page = page_name
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════
# VIEW 0: 🏠 HOME / LANDING PAGE
# ════════════════════════════════════════════════
if st.session_state.active_page == "🏠 Home":
    st.markdown(f"""
    <div class="hero-container">
      <div class="hero-pill">⚡ Next-Generation People Analytics & ML Intelligence</div>
      <div class="hero-title">Predict Attrition, Close Skill Gaps & <span>Empower Your Workforce</span></div>
      <div class="hero-desc">
        TalentPulse AI unifies high-recall XGBoost attrition modeling with O*NET competency taxonomies and generative AI playbooks to stop voluntary turnover before it happens.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Jump Action Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("📊 View Executive Overview →", use_container_width=True):
            st.session_state.active_page = "📊 Executive Overview"
            st.rerun()
    with c2:
        if st.button("⚠️ Explore Attrition Risk →", use_container_width=True):
            st.session_state.active_page = "⚠️ Attrition Forensics"
            st.rerun()
    with c3:
        if st.button("🎯 Analyze Skill Matrix →", use_container_width=True):
            st.session_state.active_page = "🎯 O*NET Skill Matrix"
            st.rerun()
    with c4:
        if st.button("🤖 Launch AI Copilot →", use_container_width=True):
            st.session_state.active_page = "🤖 AI Workforce Copilot"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 3 Strategic Pillars
    st.markdown('<div class="section-bar">⚡ <span>Three Core Intelligence Pillars</span></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pillar-grid">
      <div class="pillar-card">
        <div class="pillar-icon">🚨</div>
        <div class="pillar-title">Predictive Flight-Risk Forensics</div>
        <div class="pillar-desc">XGBoost ML engine optimized for 100% recall. Detects subtle disengagement signals, compensation compression, and burnout metrics before departures occur.</div>
      </div>
      <div class="pillar-card">
        <div class="pillar-icon">🎯</div>
        <div class="pillar-title">O*NET Competency Architecture</div>
        <div class="pillar-desc">Automated set-subtraction skill deficit analysis weighted by standardized O*NET importance ratings. Uncovers critical systemic gaps across departments.</div>
      </div>
      <div class="pillar-card">
        <div class="pillar-icon">🤖</div>
        <div class="pillar-title">Generative AI Retention Copilot</div>
        <div class="pillar-desc">Multi-provider LLM assistant with real-time RAG context. Formulates tailored retention playbooks, stay-interview agendas, and intervention roadmaps.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Live Metric Summary Banner
    if intel_df is not None:
        total = len(intel_df)
        high = (intel_df["Risk_Level"] == "HIGH").sum()
        avg_prob = intel_df["Attrition_Prob"].mean()
        avg_gap = intel_df["gaps_count"].mean()

        st.markdown(f"""
        <div class="kpi-container">
          {make_kpi_card("👥", f"{total:,}", "Active Monitored", "Total Headcount", "#6366f1")}
          {make_kpi_card("🚨", f"{high}", "Critical Flight Risks", f"{high/total:.1%} of staff", "#ef4444")}
          {make_kpi_card("📈", f"{avg_prob:.1%}", "Avg Attrition Prob", "Model confidence", "#38bdf8")}
          {make_kpi_card("🎯", f"{avg_gap:.1f}", "Avg Skill Deficits", "Skills per person", "#10b981")}
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
# VIEW 1: 📊 EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════
elif st.session_state.active_page == "📊 Executive Overview":
    if intel_df is not None:
        total = len(intel_df)
        high = (intel_df["Risk_Level"] == "HIGH").sum()
        med = (intel_df["Risk_Level"] == "MEDIUM").sum()
        low = (intel_df["Risk_Level"] == "LOW").sum()
        avg_prob = intel_df["Attrition_Prob"].mean()
        avg_gap = intel_df["gaps_count"].mean()

        st.markdown(f"""
        <div class="kpi-container">
          {make_kpi_card("👥", f"{total:,}", "Active Monitored", "Total Headcount", "#6366f1")}
          {make_kpi_card("🚨", f"{high}", "High Flight Risk", f"{high/total:.1%} of workforce", "#ef4444")}
          {make_kpi_card("⚠️", f"{med}", "Medium Watchlist", f"{med/total:.1%} of workforce", "#f59e0b")}
          {make_kpi_card("📈", f"{avg_prob:.1%}", "Avg Flight Prob", "Company baseline", "#38bdf8")}
          {make_kpi_card("🎯", f"{avg_gap:.1f}", "Avg Skill Gaps", "Gaps per employee", "#10b981")}
        </div>
        """, unsafe_allow_html=True)

        col_c1, col_c2 = st.columns([1.3, 1])
        with col_c1:
            st.markdown('<div class="section-bar">🏢 <span>Department Flight-Risk Concentration</span></div>', unsafe_allow_html=True)
            dept_risk = intel_df.groupby("Department").agg(
                Total=("EmployeeID", "count"),
                High=("Risk_Level", lambda x: (x == "HIGH").sum()),
                Avg_Prob=("Attrition_Prob", "mean"),
            ).reset_index()
            dept_risk["High_Pct"] = dept_risk["High"] / dept_risk["Total"] * 100
            dept_risk = dept_risk.sort_values("High_Pct", ascending=True)

            fig = go.Figure(go.Bar(
                y=dept_risk["Department"], x=dept_risk["High_Pct"], orientation="h",
                marker=dict(color=dept_risk["High_Pct"], colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1.0, "#ef4444"]], showscale=False),
                text=[f"{v:.1f}%" for v in dept_risk["High_Pct"]], textposition="outside",
                textfont=dict(color="#f8fafc", size=11),
                hovertemplate="<b>%{y}</b><br>High Risk: %{x:.1f}%<br>Count: %{customdata[0]}<extra></extra>",
                customdata=dept_risk[["High"]].values,
            ))
            fig.update_layout(height=340, xaxis_title="% High Risk Staff", yaxis_title="")
            fig = apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col_c2:
            st.markdown('<div class="section-bar">📊 <span>Workforce Risk Segmentation</span></div>', unsafe_allow_html=True)
            risk_counts = intel_df["Risk_Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]

            fig2 = go.Figure(go.Pie(
                labels=risk_counts["Risk Level"], values=risk_counts["Count"], hole=0.62,
                marker=dict(colors=[PALETTE.get(r, "#6366f1") for r in risk_counts["Risk Level"]]),
                textinfo="label+percent", textfont=dict(color="#f8fafc", size=12),
                hovertemplate="<b>%{label}</b><br>%{value} employees (%{percent})<extra></extra>",
            ))
            fig2.update_layout(
                height=340, showlegend=False,
                annotations=[dict(text=f"<b>{total}</b><br>Staff", x=0.5, y=0.5, font_size=18, showarrow=False, font=dict(color="#f8fafc"))]
            )
            fig2 = apply_chart_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        if dept_scores_df is not None and not dept_scores_df.empty:
            st.markdown('<div class="section-bar">📈 <span>Department Engagement Benchmark (5,000-Employee Data)</span></div>', unsafe_allow_html=True)
            dept_s = dept_scores_df.sort_values("Composite_Engagement_Score", ascending=False)
            fig3 = go.Figure(go.Bar(
                x=dept_s["Department"], y=dept_s["Composite_Engagement_Score"] * 100,
                marker=dict(color=dept_s["Composite_Engagement_Score"], colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]]),
                text=[f"{v*100:.1f}%" for v in dept_s["Composite_Engagement_Score"]], textposition="outside", textfont=dict(color="#f8fafc"),
            ))
            fig3.update_layout(height=280, yaxis_title="Engagement Index (%)")
            fig3 = apply_chart_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════
# VIEW 2: ⚠️ ATTRITION FORENSICS
# ════════════════════════════════════════════════
elif st.session_state.active_page == "⚠️ Attrition Forensics":
    st.markdown('<div class="section-bar">⚠️ <span>Predictive Attrition Risk Forensics</span></div>', unsafe_allow_html=True)
    
    if intel_df is not None:
        col_f1, col_f2, col_f3 = st.columns([1.5, 1.5, 1])
        with col_f1:
            dept_options = ["All Departments"] + sorted(intel_df["Department"].dropna().unique().tolist())
            dept_filter = st.selectbox("Filter Department", dept_options)
        with col_f2:
            risk_options = ["All Severity Levels", "HIGH", "MEDIUM", "LOW"]
            risk_filter = st.selectbox("Filter Risk Category", risk_options)
        with col_f3:
            n_rows = st.slider("Max Roster Display", 10, 100, 25)

        filtered = intel_df.copy()
        if dept_filter != "All Departments": filtered = filtered[filtered["Department"] == dept_filter]
        if risk_filter != "All Severity Levels": filtered = filtered[filtered["Risk_Level"] == risk_filter]

        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.histogram(
                filtered, x="Attrition_Prob", nbins=28,
                color_discrete_sequence=[PALETTE["primary"]],
                title="<b>Attrition Probability Distribution</b>",
                labels={"Attrition_Prob": "Predicted Probability"},
            )
            fig.add_vline(x=h_p, line_dash="dash", line_color=PALETTE["HIGH"], annotation_text=f"HIGH (≥{high_risk_thresh}%)", annotation_font_color=PALETTE["HIGH"])
            fig.add_vline(x=m_p, line_dash="dash", line_color=PALETTE["MEDIUM"], annotation_text=f"MED (≥{med_risk_thresh}%)", annotation_font_color=PALETTE["MEDIUM"])
            fig = apply_chart_theme(fig)
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            fig2 = px.box(
                intel_df, x="Department", y="Attrition_Prob", color="Department",
                title="<b>Departmental Flight Risk Dispersion</b>",
                labels={"Attrition_Prob": "Probability"},
            )
            fig2 = apply_chart_theme(fig2)
            fig2.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        col_h, col_dl = st.columns([4, 1])
        with col_h:
            st.markdown(f'<div class="section-bar">📋 <span>Identified At-Risk Roster ({len(filtered)} Employees)</span></div>', unsafe_allow_html=True)
        with col_dl:
            csv = filtered.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Roster", csv, "at_risk_roster.csv", "text/csv", use_container_width=True)

        display_cols = ["EmployeeID", "Name", "Department", "JobRole", "Attrition_Prob", "Risk_Level", "YearsAtCompany", "Top_Skill_Gap"]
        roster_df = filtered.nlargest(n_rows, "Attrition_Prob")[display_cols].copy()
        roster_df["Attrition_Prob"] = roster_df["Attrition_Prob"].apply(lambda p: f"{p:.1%}")
        roster_df["Risk_Level"] = roster_df["Risk_Level"].apply(lambda r: f"🔴 {r}" if r == "HIGH" else ("🟡 " + r if r == "MEDIUM" else "🟢 " + r))

        st.dataframe(
            roster_df, use_container_width=True, hide_index=True,
            column_config={
                "EmployeeID": st.column_config.TextColumn("ID"),
                "Attrition_Prob": st.column_config.TextColumn("Probability"),
                "Risk_Level": st.column_config.TextColumn("Risk Rating"),
                "YearsAtCompany": st.column_config.NumberColumn("Tenure (Yrs)"),
                "Top_Skill_Gap": st.column_config.TextColumn("Primary Skill Gap"),
            }
        )


# ════════════════════════════════════════════════
# VIEW 3: 🎯 O*NET SKILL MATRIX
# ════════════════════════════════════════════════
elif st.session_state.active_page == "🎯 O*NET Skill Matrix":
    st.markdown('<div class="section-bar">🎯 <span>Organizational Competency & Skill Gap Architecture</span></div>', unsafe_allow_html=True)

    if org_gap_df is not None:
        high_gaps = (org_gap_df["severity"] == "HIGH").sum()
        med_gaps = (org_gap_df["severity"] == "MEDIUM").sum()
        low_gaps = (org_gap_df["severity"] == "LOW").sum()
        top_skill = org_gap_df.nlargest(1, "total_gap_weight")["skill"].iloc[0]

        st.markdown(f"""
        <div class="kpi-container">
          {make_kpi_card("🔴", f"{high_gaps}", "Critical Gaps", ">=30% staff & IM>=3.5", "#ef4444")}
          {make_kpi_card("🟡", f"{med_gaps}", "Moderate Gaps", ">=15% staff & IM>=3.0", "#f59e0b")}
          {make_kpi_card("🟢", f"{low_gaps}", "Managed Skills", "Standard capability", "#10b981")}
          {make_kpi_card("⚡", f"{top_skill[:18]}...", "Top Deficit", "Highest gap weight", "#38bdf8")}
        </div>
        """, unsafe_allow_html=True)

        col_sf, col_sn = st.columns([2, 1])
        with col_sf: sev_filter = st.selectbox("Severity Classification", ["All Deficits", "HIGH", "MEDIUM", "LOW"])
        with col_sn: top_n = st.slider("Display Top N Skills", 10, 50, 20)

        filtered_gaps = org_gap_df if sev_filter == "All Deficits" else org_gap_df[org_gap_df["severity"] == sev_filter]
        top_gaps = filtered_gaps.nlargest(top_n, "total_gap_weight")

        fig_gaps = go.Figure()
        for sev in ["HIGH", "MEDIUM", "LOW"]:
            sub = top_gaps[top_gaps["severity"] == sev]
            if not sub.empty:
                fig_gaps.add_trace(go.Bar(
                    y=sub["skill"], x=sub["total_gap_weight"], orientation="h", name=sev,
                    marker=dict(color=PALETTE[sev]),
                    hovertemplate="<b>%{y}</b><br>Weighted Deficit: %{x:.1f}<br>Employees Lacking: %{customdata[0]}<extra></extra>",
                    customdata=sub[["employees_lacking"]].values
                ))

        fig_gaps.update_layout(
            title=f"<b>Top {top_n} Organizational Competency Gaps</b>",
            height=max(400, top_n * 22),
            xaxis_title="Weighted Severity Score (Headcount × O*NET Importance)",
            barmode="stack",
            legend=dict(orientation="h", y=1.05, x=1, xanchor="right")
        )
        fig_gaps = apply_chart_theme(fig_gaps)
        st.plotly_chart(fig_gaps, use_container_width=True)

        col_h, col_dl = st.columns([4, 1])
        with col_h: st.markdown('<div class="section-bar">📊 <span>Competency Detail Register</span></div>', unsafe_allow_html=True)
        with col_dl:
            sg_csv = top_gaps.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Matrix", sg_csv, "skill_gaps_matrix.csv", "text/csv", use_container_width=True)

        st.dataframe(
            top_gaps[["skill", "severity", "pct_employees_lacking", "avg_importance", "total_gap_weight", "employees_lacking"]].rename(columns={
                "skill": "Skill Element", "severity": "Severity Tier", "pct_employees_lacking": "% Staff Lacking",
                "avg_importance": "O*NET Importance", "total_gap_weight": "Total Gap Score", "employees_lacking": "# Affected Staff"
            }),
            use_container_width=True, hide_index=True
        )


# ════════════════════════════════════════════════
# VIEW 4: 👤 EMPLOYEE DOSSIER
# ════════════════════════════════════════════════
elif st.session_state.active_page == "👤 Employee Dossier":
    st.markdown('<div class="section-bar">👤 <span>Individual 360° Intelligence Dossier & Retention Simulator</span></div>', unsafe_allow_html=True)

    if intel_df is not None:
        col_s1, col_s2 = st.columns([3, 1])
        with col_s1: search_query = st.text_input("Lookup Employee (Enter ID: 1–500, or Name)", placeholder="e.g. 1, 42, or Steven Barnett...")
        with col_s2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_trigger = st.button("🔍 Search Dossier", type="primary", use_container_width=True)

        found_emp = None
        if search_query:
            id_m = intel_df[intel_df["EmployeeID"].astype(str) == str(search_query.strip())]
            if not id_m.empty: found_emp = id_m.iloc[0]
            else:
                nm_m = intel_df[intel_df["Name"].str.contains(search_query.strip(), case=False, na=False)]
                if not nm_m.empty: found_emp = nm_m.iloc[0]

        if found_emp is not None:
            emp_id = str(found_emp["EmployeeID"])
            risk = str(found_emp.get("Risk_Level", "LOW"))
            prob = float(found_emp.get("Attrition_Prob", 0.0))

            st.markdown(f"""
            <div class="profile-card">
              <div class="profile-header">
                <div>
                  <div class="profile-name">{found_emp.get('Name', 'Unknown')}</div>
                  <div class="profile-role">{found_emp.get('JobRole', 'Staff')} // {found_emp.get('Department', 'General')}</div>
                </div>
                <div>{risk_badge(risk)}</div>
              </div>
              <div class="profile-grid">
                <div class="stat-tile"><div class="stat-label">Employee ID</div><div class="stat-value">#{emp_id}</div></div>
                <div class="stat-tile"><div class="stat-label">Tenure</div><div class="stat-value">{found_emp.get('YearsAtCompany', 0):.1f} Years</div></div>
                <div class="stat-tile"><div class="stat-label">Monthly Salary</div><div class="stat-value">${found_emp.get('MonthlySalary', 0):,.0f}</div></div>
                <div class="stat-tile"><div class="stat-label">Performance</div><div class="stat-value">{found_emp.get('PerformanceRating', 0):.1f} / 5.0</div></div>
                <div class="stat-tile"><div class="stat-label">Work-Life Balance</div><div class="stat-value">{found_emp.get('WorkLifeBalanceScore', 0):.1f} / 5.0</div></div>
                <div class="stat-tile"><div class="stat-label">Skill Coverage</div><div class="stat-value">{found_emp.get('coverage_pct', 0):.0%}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            col_g, col_rec = st.columns([1.2, 1])
            with col_g:
                gauge_color = PALETTE["HIGH"] if risk == "HIGH" else (PALETTE["MEDIUM"] if risk == "MEDIUM" else PALETTE["LOW"])
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    number={"suffix": "%", "font": {"color": "#ffffff", "size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                        "bar": {"color": gauge_color, "thickness": 0.28},
                        "bgcolor": "rgba(20, 27, 44, 0.8)",
                        "steps": [
                            {"range": [0, med_risk_thresh], "color": "rgba(16, 185, 129, 0.15)"},
                            {"range": [med_risk_thresh, high_risk_thresh], "color": "rgba(245, 158, 11, 0.15)"},
                            {"range": [high_risk_thresh, 100], "color": "rgba(239, 68, 68, 0.15)"},
                        ],
                        "threshold": {"line": {"color": gauge_color, "width": 4}, "value": prob * 100}
                    },
                    title={"text": f"<b>Attrition Flight Meter: {risk} RISK</b>", "font": {"color": gauge_color, "size": 14}}
                ))
                fig_gauge.update_layout(height=260, margin=dict(t=25, b=10))
                fig_gauge = apply_chart_theme(fig_gauge)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_rec:
                st.markdown(f"""
                <div class="profile-card" style="padding: 20px;">
                  <div style="font-weight: 700; color: var(--color-accent); margin-bottom: 8px;">🎯 Target Upskilling Path</div>
                  <div style="font-size: 0.95rem; font-weight: 600; color: #ffffff;">{found_emp.get('Top_Skill_Gap', 'No Gaps')}</div>
                  <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">Priority: <b>{found_emp.get('Rec_Priority', 'Standard')}</b></div>
                  <div style="margin-top: 14px; font-weight: 700; color: var(--color-success);">🛠️ Recommended Tool Stack</div>
                  <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">{found_emp.get('Recommended_Tools', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-bar">🧪 <span>Retention Intervention Simulator (What-If Lab)</span></div>', unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            with s1: sim_raise = st.slider("💰 Proposed Comp Increase (%)", 0, 40, 10, 5, key=f"sim_sal_{emp_id}")
            with s2:
                cur_wlb = float(found_emp.get("WorkLifeBalanceScore", 3.0))
                sim_wlb = st.slider("⚖️ Target Work-Life Balance", 1.0, 5.0, min(5.0, cur_wlb + 1.0), 0.5, key=f"sim_wlb_{emp_id}")
            with s3: sim_trn = st.slider("🎓 Sponsored Training (Hours)", 0, 40, 15, 5, key=f"sim_trn_{emp_id}")

            sal_red = (sim_raise / 100.0) * 0.35
            wlb_red = max(0.0, (sim_wlb - cur_wlb) / 5.0) * 0.40
            trn_red = (sim_trn / 40.0) * 0.15
            total_red = sal_red + wlb_red + trn_red
            sim_p = max(0.05, prob * (1.0 - total_red))
            sim_r = assign_user_risk(sim_p)

            sc1, sc2, sc3 = st.columns(3)
            with sc1: st.metric("Baseline Attrition Risk", f"{prob:.1%}", risk)
            with sc2: st.metric("Simulated New Probability", f"{sim_p:.1%}", f"🔴 {sim_r}" if sim_r == "HIGH" else f"🟢 {sim_r}")
            with sc3: st.metric("Estimated Risk Reduction", f"🔻 -{(prob - sim_p):.1%}", f"{total_red*100:.0f}% Efficacy")

            dossier_data = json.dumps({k: v for k, v in found_emp.to_dict().items() if not pd.isna(v)}, indent=2)
            st.download_button("📥 Export Full Dossier (JSON)", dossier_data, f"employee_{emp_id}_dossier.json", "application/json", use_container_width=True)

        elif search_query:
            st.warning(f"No records matching '{search_query}'. Try an ID between 1 and 500.")
        else:
            st.markdown('<div class="section-bar">🚨 <span>Top Priority At-Risk Employees</span></div>', unsafe_allow_html=True)
            top10 = intel_df.nlargest(8, "Attrition_Prob")[["EmployeeID", "Name", "Department", "JobRole", "Attrition_Prob", "Risk_Level"]].copy()
            top10["Attrition_Prob"] = top10["Attrition_Prob"].apply(lambda p: f"{p:.1%}")
            top10["Risk_Level"] = top10["Risk_Level"].apply(lambda r: f"🔴 {r}" if r == "HIGH" else ("🟡 " + r if r == "MEDIUM" else "🟢 " + r))
            st.dataframe(top10, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════
# VIEW 5: 🤖 AI WORKFORCE COPILOT
# ════════════════════════════════════════════════
elif st.session_state.active_page == "🤖 AI Workforce Copilot":
    st.markdown('<div class="section-bar">🤖 <span>Generative AI Workforce Copilot (Dynamic RAG)</span></div>', unsafe_allow_html=True)
    st.caption("Ask natural language questions regarding attrition diagnostics, retention playbooks, and O*NET skill matrix.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 **Hello! I am your TalentPulse AI Copilot.** I have live access to our **XGBoost Attrition Model**, **O*NET Competency Matrix**, and **Employee Intelligence Database**.\n\n"
                       "💡 **You can ask me:**\n"
                       "- *\"Which department has the highest flight risk and why?\"*\n"
                       "- *\"Give me a retention diagnosis and stay-interview guide for Employee 42\"*\n"
                       "- *\"What are our top 5 critical organizational skill gaps?\"*\n"
                       "- *\"Draft a 90-day upskilling roadmap for our Engineering department.\"*"
        }]

    col_q, col_clear = st.columns([5, 1])
    quick_q = None
    with col_q:
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            if st.button("🚨 Top Flight Risks", use_container_width=True): quick_q = "Who are the highest flight-risk employees in the company?"
        with q2:
            if st.button("🏢 Dept Risk Breakdown", use_container_width=True): quick_q = "Which department has the worst attrition risk and what are the root causes?"
        with q3:
            if st.button("🎯 Top Skill Gaps", use_container_width=True): quick_q = "What are our most critical organization-wide skill gaps?"
        with q4:
            if st.button("📊 Executive Briefing", use_container_width=True): quick_q = "Give me an executive workforce intelligence briefing with 3 actionable priorities."

    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "👋 Chat reset. How can I assist with your workforce intelligence today?"}]
            st.rerun()

    st.caption(f"🤖 Active Engine: **{llm_provider}** ({llm_model if llm_model else 'Default'})")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask any question regarding employees, flight risk, or skills...") or quick_q
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        system_context = build_system_context(intel_df, org_gap_df, dept_scores_df, user_input)
        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing workforce telemetry & generating insight..."):
                reply = generate_llm_response(
                    messages=st.session_state.messages,
                    system_context=system_context,
                    api_key=llm_api_key,
                    provider=llm_provider,
                    model_name=llm_model,
                    base_url=llm_base_url if 'llm_base_url' in locals() else None
                )
                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
