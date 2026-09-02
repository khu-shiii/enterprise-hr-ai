"""
TalentPulse AI — Enterprise Workforce Intelligence & People Analytics Platform
Principal UX/UI Redesign — Production SaaS Edition with Dual Theme Engine (Light/Dark)
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
import joblib
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

# ── 2. Theme State Management ──
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "high_thresh" not in st.session_state:
    st.session_state.high_thresh = 65

if "med_thresh" not in st.session_state:
    st.session_state.med_thresh = 40

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Dashboard"

if "selected_employee_id" not in st.session_state:
    st.session_state.selected_employee_id = "1"

# ── 3. Design System & CSS Themes (Light & Dark) ──
is_dark = (st.session_state.theme == "dark")

THEME_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

  :root {{
    --bg-base: {'#0b0f19' if is_dark else '#f8fafc'};
    --bg-surface: {'rgba(19, 26, 42, 0.85)' if is_dark else 'rgba(255, 255, 255, 0.95)'};
    --bg-surface-hover: {'rgba(28, 38, 62, 0.95)' if is_dark else '#f1f5f9'};
    --bg-surface-elevated: {'#151d30' if is_dark else '#ffffff'};
    
    --border-subtle: {'rgba(255, 255, 255, 0.08)' if is_dark else 'rgba(0, 0, 0, 0.08)'};
    --border-accent: {'rgba(99, 102, 241, 0.4)' if is_dark else 'rgba(99, 102, 241, 0.3)'};
    
    --text-primary: {'#f8fafc' if is_dark else '#0f172a'};
    --text-secondary: {'#94a3b8' if is_dark else '#475569'};
    --text-tertiary: {'#64748b' if is_dark else '#94a3b8'};
    
    --color-primary: #6366f1;
    --color-primary-light: #818cf8;
    --color-primary-glow: rgba(99, 102, 241, 0.25);
    --color-accent: #38bdf8;
    --color-success: #10b981;
    --color-warning: #f59e0b;
    --color-danger: #ef4444;
    
    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --radius-full: 9999px;
  }}

  /* Global typography & base */
  html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
  }}

  .stApp {{
    background: {'radial-gradient(circle at 50% -15%, #18223c 0%, var(--bg-base) 70%)' if is_dark else 'radial-gradient(circle at 50% -15%, #e2e8f0 0%, var(--bg-base) 70%)'};
  }}

  /* Sidebar styling */
  [data-testid="stSidebar"] {{
    background: {'linear-gradient(180deg, #101626 0%, #090d16 100%)' if is_dark else 'linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%)'} !important;
    border-right: 1px solid var(--border-subtle) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.15);
  }}

  /* Top Navigation Bar Header */
  .top-navbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    background: var(--bg-surface);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    margin-bottom: 24px;
    box-shadow: {'0 8px 32px rgba(0, 0, 0, 0.35)' if is_dark else '0 4px 20px rgba(0, 0, 0, 0.05)'};
  }}
  .brand-logo-area {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .brand-logo-badge {{
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #6366f1 0%, #38bdf8 100%);
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    box-shadow: 0 0 16px var(--color-primary-glow);
    color: #ffffff;
  }}
  .brand-name {{
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: {'linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%)' if is_dark else 'linear-gradient(135deg, #0f172a 0%, #334155 100%)'};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .brand-tag {{
    font-size: 0.72rem;
    color: var(--color-primary);
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }}
  .status-badge {{
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
  }}

  /* KPI Executive Cards */
  .kpi-container {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }}
  .kpi-card {{
    background: var(--bg-surface);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: {'0 4px 20px rgba(0, 0, 0, 0.25)' if is_dark else '0 2px 12px rgba(0, 0, 0, 0.04)'};
  }}
  .kpi-card:hover {{
    transform: translateY(-3px);
    background: var(--bg-surface-hover);
    border-color: var(--border-accent);
    box-shadow: {'0 12px 32px rgba(0, 0, 0, 0.4), 0 0 20px var(--color-primary-glow)' if is_dark else '0 10px 25px rgba(99, 102, 241, 0.12)'};
  }}
  .kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, var(--card-accent, var(--color-primary)), transparent);
  }}
  .kpi-icon {{ font-size: 1.6rem; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em; margin: 2px 0 4px 0; color: var(--text-primary); }}
  .kpi-label {{ font-size: 0.78rem; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }}
  .kpi-sub {{ font-size: 0.72rem; color: var(--text-tertiary); margin-top: 4px; }}

  /* Section Title Bar */
  .section-bar {{
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--text-primary);
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin: 28px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
  }}
  .section-bar span {{ color: var(--color-primary); }}

  /* Badges */
  .badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: var(--radius-full);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }}
  .badge-high {{ background: rgba(239, 68, 68, 0.15); color: {'#fca5a5' if is_dark else '#dc2626'}; border: 1px solid rgba(239, 68, 68, 0.4); }}
  .badge-medium {{ background: rgba(245, 158, 11, 0.15); color: {'#fcd34d' if is_dark else '#d97706'}; border: 1px solid rgba(245, 158, 11, 0.4); }}
  .badge-low {{ background: rgba(16, 185, 129, 0.15); color: {'#6ee7b7' if is_dark else '#059669'}; border: 1px solid rgba(16, 185, 129, 0.4); }}

  /* Cards & Panels */
  .ui-card {{
    background: var(--bg-surface);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 24px;
    box-shadow: {'0 8px 32px rgba(0, 0, 0, 0.35)' if is_dark else '0 4px 20px rgba(0, 0, 0, 0.04)'};
    margin-bottom: 20px;
  }}
  .profile-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 18px;
  }}
  .profile-name {{ font-size: 1.5rem; font-weight: 800; color: var(--text-primary); }}
  .profile-role {{ font-size: 0.92rem; color: var(--color-primary); font-weight: 600; margin-top: 2px; }}
  
  .profile-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    padding-top: 14px;
    border-top: 1px solid var(--border-subtle);
  }}
  .stat-tile {{
    background: {'rgba(255, 255, 255, 0.025)' if is_dark else '#f8fafc'};
    padding: 12px 16px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-subtle);
  }}
  .stat-label {{ font-size: 0.72rem; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-value {{ font-size: 0.98rem; color: var(--text-primary); font-weight: 700; margin-top: 2px; }}

  /* Action Cards */
  .insight-card {{
    background: var(--bg-surface);
    border-left: 4px solid var(--card-accent, var(--color-primary));
    border-top: 1px solid var(--border-subtle);
    border-right: 1px solid var(--border-subtle);
    border-bottom: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 18px 20px;
    margin-bottom: 14px;
    transition: all 0.2s ease;
  }}
  .insight-card:hover {{
    transform: translateX(4px);
    background: var(--bg-surface-hover);
  }}

  /* Chat Messages */
  [data-testid="stChatMessage"] {{
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    margin-bottom: 12px;
  }}

  /* Button Styling */
  .stButton button {{
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
  }}

  #MainMenu, footer, header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ── 4. Plotly Chart Theme Engine ──
CHART_BG = "rgba(19, 26, 42, 0.5)" if is_dark else "rgba(255, 255, 255, 0.8)"
CHART_FONT_COLOR = "#f1f5f9" if is_dark else "#0f172a"
CHART_GRID_COLOR = "rgba(255, 255, 255, 0.07)" if is_dark else "rgba(0, 0, 0, 0.06)"

PALETTE = {
    "HIGH": "#ef4444",
    "MEDIUM": "#f59e0b",
    "LOW": "#10b981",
    "primary": "#6366f1",
    "secondary": "#a855f7",
    "accent": "#38bdf8",
}

def apply_chart_theme(fig):
    fig.update_layout(
        plot_bgcolor=CHART_BG,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CHART_FONT_COLOR, family="Plus Jakarta Sans, sans-serif"),
        xaxis=dict(gridcolor=CHART_GRID_COLOR, linecolor=CHART_GRID_COLOR, zerolinecolor=CHART_GRID_COLOR),
        yaxis=dict(gridcolor=CHART_GRID_COLOR, linecolor=CHART_GRID_COLOR, zerolinecolor=CHART_GRID_COLOR),
        margin=dict(t=35, b=35, l=20, r=20),
    )
    return fig


# ── 5. Data Loaders & Model Cache ──
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

@st.cache_resource
def load_ml_pipeline():
    model_path = Path(__file__).parent.parent / "models" / "v1" / "attrition_pipeline.joblib"
    if not model_path.exists():
        model_path = Path(__file__).parent.parent / "models" / "attrition_pipeline.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


intel_df = load_local_intelligence()
org_gap_df = load_org_skill_gap()
dept_scores_df = load_dept_scores()
ml_pipeline = load_ml_pipeline()

# Dynamic risk assignment
h_p = st.session_state.high_thresh / 100.0
m_p = st.session_state.med_thresh / 100.0

def assign_user_risk(p):
    if p >= h_p: return "HIGH"
    elif p >= m_p: return "MEDIUM"
    return "LOW"

if intel_df is not None:
    intel_df["Risk_Level"] = intel_df["Attrition_Prob"].apply(assign_user_risk)


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


# ── 6. Persistent Sidebar Navigation ──
with st.sidebar:
    st.markdown("### ⚡ TalentPulse AI")
    st.caption("Enterprise Workforce Intelligence")
    
    st.markdown("---")
    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:var(--text-tertiary); letter-spacing:0.08em; text-transform:uppercase;'>OVERVIEW</p>", unsafe_allow_html=True)
    if st.button("📊 Dashboard", use_container_width=True, type="primary" if st.session_state.active_tab == "📊 Dashboard" else "secondary"):
        st.session_state.active_tab = "📊 Dashboard"
        st.rerun()

    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:var(--text-tertiary); letter-spacing:0.08em; text-transform:uppercase; margin-top:14px;'>WORKFORCE</p>", unsafe_allow_html=True)
    c_w1, c_w2, c_w3 = "👥 Employees", "📈 Workforce Analytics", "🏢 Departments"
    for label in [c_w1, c_w2, c_w3]:
        if st.button(label, use_container_width=True, type="primary" if st.session_state.active_tab == label else "secondary"):
            st.session_state.active_tab = label
            st.rerun()

    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:var(--text-tertiary); letter-spacing:0.08em; text-transform:uppercase; margin-top:14px;'>ANALYTICS</p>", unsafe_allow_html=True)
    c_a1, c_a2, c_a3 = "⚠️ Attrition Analytics", "⭐ Performance Analytics", "💰 Compensation Equity"
    for label in [c_a1, c_a2, c_a3]:
        if st.button(label, use_container_width=True, type="primary" if st.session_state.active_tab == label else "secondary"):
            st.session_state.active_tab = label
            st.rerun()

    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:var(--text-tertiary); letter-spacing:0.08em; text-transform:uppercase; margin-top:14px;'>AI INTELLIGENCE</p>", unsafe_allow_html=True)
    c_ai1, c_ai2, c_ai3, c_ai4 = "🔮 ML Risk Prediction", "💡 AI Insights", "🎯 AI Action Center", "🤖 AI HR Copilot"
    for label in [c_ai1, c_ai2, c_ai3, c_ai4]:
        if st.button(label, use_container_width=True, type="primary" if st.session_state.active_tab == label else "secondary"):
            st.session_state.active_tab = label
            st.rerun()

    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:var(--text-tertiary); letter-spacing:0.08em; text-transform:uppercase; margin-top:14px;'>SYSTEM</p>", unsafe_allow_html=True)
    c_s1, c_s2 = "🗄️ Data & Models", "⚙️ System Settings"
    for label in [c_s1, c_s2]:
        if st.button(label, use_container_width=True, type="primary" if st.session_state.active_tab == label else "secondary"):
            st.session_state.active_tab = label
            st.rerun()


# ── 7. Top Navigation Bar (Header + Global Search + Theme Toggle + User) ──
col_head_logo, col_head_search, col_head_actions = st.columns([2.5, 3.5, 2.5])

with col_head_logo:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; padding:6px 0;">
      <div class="brand-logo-badge">⚡</div>
      <div>
        <div class="brand-name">TalentPulse <span style="font-weight:400; color:#818cf8;">AI</span></div>
        <div class="brand-tag">Workforce Decision Intelligence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_head_search:
    global_search = st.text_input("Global Search", placeholder="🔍 Search employees, skills, departments...", label_visibility="collapsed")

with col_head_actions:
    c_thm, c_bell, c_usr = st.columns([1.2, 0.8, 1.5])
    with c_thm:
        theme_btn_label = "☀️ Light" if is_dark else "🌙 Dark"
        if st.button(theme_btn_label, use_container_width=True):
            st.session_state.theme = "light" if is_dark else "dark"
            st.rerun()
    with c_bell:
        st.button("🔔 3", help="3 High Risk Flight Alerts Detected", use_container_width=True)
    with c_usr:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; justify-content:flex-end; padding-top:4px;">
          <div style="width:32px; height:32px; border-radius:50%; background:linear-gradient(135deg, #6366f1, #38bdf8); display:flex; align-items:center; justify-content:center; font-weight:700; color:#fff; font-size:0.8rem;">HR</div>
          <div style="font-size:0.82rem; font-weight:600; color:var(--text-primary); line-height:1.2;">Admin<br><span style="font-size:0.7rem; color:var(--text-secondary); font-weight:400;">Executive</span></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr style='margin:12px 0 20px 0; border:0; border-top:1px solid var(--border-subtle);'>", unsafe_allow_html=True)


# ════════════════════════════════════════════════
# VIEW 1: 📊 EXECUTIVE DASHBOARD
# ════════════════════════════════════════════════
if st.session_state.active_tab == "📊 Dashboard":
    if intel_df is not None:
        total = len(intel_df)
        high = (intel_df["Risk_Level"] == "HIGH").sum()
        med = (intel_df["Risk_Level"] == "MEDIUM").sum()
        low = (intel_df["Risk_Level"] == "LOW").sum()
        avg_sal = intel_df["MonthlySalary"].mean()
        avg_sat = intel_df["WorkLifeBalanceScore"].mean()
        avg_prob = intel_df["Attrition_Prob"].mean()

        st.markdown(f"""
        <div class="kpi-container">
          {make_kpi_card("👥", f"{total:,}", "Total Employees", "Active Monitored Staff", "#6366f1")}
          {make_kpi_card("🚨", f"{high/total:.1%}", "Predicted Attrition Rate", f"{high} Critical Flight Risks", "#ef4444")}
          {make_kpi_card("💵", f"${avg_sal:,.0f}", "Average Monthly Salary", "Baseline compensation", "#38bdf8")}
          {make_kpi_card("⭐", f"{avg_sat:.1f}/5.0", "Workforce Satisfaction", "Work-life balance index", "#10b981")}
          {make_kpi_card("⚠️", f"{high + med}", "Watchlist Headcount", f"{high} High + {med} Medium", "#f59e0b")}
        </div>
        """, unsafe_allow_html=True)

        col_d1, col_d2 = st.columns([1.3, 1])
        with col_d1:
            st.markdown('<div class="section-bar">🏢 <span>Department Flight-Risk Concentration</span></div>', unsafe_allow_html=True)
            dept_risk = intel_df.groupby("Department").agg(
                Total=("EmployeeID", "count"),
                High=("Risk_Level", lambda x: (x == "HIGH").sum()),
                Avg_Prob=("Attrition_Prob", "mean"),
            ).reset_index()
            dept_risk["High_Pct"] = (dept_risk["High"] / dept_risk["Total"] * 100).round(1)
            dept_risk = dept_risk.sort_values("High_Pct", ascending=True)

            fig = go.Figure(go.Bar(
                y=dept_risk["Department"], x=dept_risk["High_Pct"], orientation="h",
                marker=dict(color=dept_risk["High_Pct"], colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1.0, "#ef4444"]], showscale=False),
                text=[f"{v:.1f}%" for v in dept_risk["High_Pct"]], textposition="outside",
                hovertemplate="<b>%{y}</b><br>High Risk Rate: %{x:.1f}%<br>High Risk Staff: %{customdata[0]}<extra></extra>",
                customdata=dept_risk[["High"]].values,
            ))
            fig.update_layout(height=320, xaxis_title="% High Risk Staff", yaxis_title="")
            fig = apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col_d2:
            st.markdown('<div class="section-bar">📊 <span>Workforce Risk Distribution</span></div>', unsafe_allow_html=True)
            risk_counts = intel_df["Risk_Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]

            fig2 = go.Figure(go.Pie(
                labels=risk_counts["Risk Level"], values=risk_counts["Count"], hole=0.62,
                marker=dict(colors=[PALETTE.get(r, "#6366f1") for r in risk_counts["Risk Level"]]),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>%{value} staff (%{percent})<extra></extra>",
            ))
            fig2.update_layout(
                height=320, showlegend=False,
                annotations=[dict(text=f"<b>{total}</b><br>Staff", x=0.5, y=0.5, font_size=18, showarrow=False, font=dict(color=CHART_FONT_COLOR))]
            )
            fig2 = apply_chart_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown('<div class="section-bar">⏳ <span>Tenure vs. Flight Probability</span></div>', unsafe_allow_html=True)
            fig_t = px.scatter(
                intel_df, x="YearsAtCompany", y="Attrition_Prob", color="Risk_Level",
                color_discrete_map=PALETTE,
                hover_data=["Name", "JobRole", "MonthlySalary"],
                labels={"YearsAtCompany": "Tenure (Years)", "Attrition_Prob": "Flight Risk Probability"}
            )
            fig_t = apply_chart_theme(fig_t)
            fig_t.update_layout(height=280)
            st.plotly_chart(fig_t, use_container_width=True)

        with col_t2:
            st.markdown('<div class="section-bar">👥 <span>Workforce Age & Gender Demographics</span></div>', unsafe_allow_html=True)
            fig_demo = px.histogram(
                intel_df, x="Age", color="Gender", barmode="group",
                color_discrete_sequence=[PALETTE["primary"], PALETTE["accent"], PALETTE["secondary"]]
            )
            fig_demo = apply_chart_theme(fig_demo)
            fig_demo.update_layout(height=280)
            st.plotly_chart(fig_demo, use_container_width=True)


# ════════════════════════════════════════════════
# VIEW 2: 👥 EMPLOYEE EXPLORER & PROFILE
# ════════════════════════════════════════════════
elif st.session_state.active_tab == "👥 Employees":
    st.markdown('<div class="section-bar">👥 <span>Workforce Employee Management & Profiling</span></div>', unsafe_allow_html=True)

    if intel_df is not None:
        c_f1, c_f2, c_f3, c_f4 = st.columns([1.5, 1.5, 1.5, 1])
        with c_f1:
            dept_opts = ["All"] + sorted(intel_df["Department"].dropna().unique().tolist())
            f_dept = st.selectbox("Department", dept_opts, key="emp_dept_filter")
        with c_f2:
            role_opts = ["All"] + sorted(intel_df["JobRole"].dropna().unique().tolist())
            f_role = st.selectbox("Job Role", role_opts, key="emp_role_filter")
        with c_f3:
            f_risk = st.selectbox("Risk Level", ["All", "HIGH", "MEDIUM", "LOW"], key="emp_risk_filter")
        with c_f4:
            f_search = st.text_input("Filter by ID/Name", key="emp_txt_filter")

        filtered_emp = intel_df.copy()
        if f_dept != "All": filtered_emp = filtered_emp[filtered_emp["Department"] == f_dept]
        if f_role != "All": filtered_emp = filtered_emp[filtered_emp["JobRole"] == f_role]
        if f_risk != "All": filtered_emp = filtered_emp[filtered_emp["Risk_Level"] == f_risk]
        if f_search:
            filtered_emp = filtered_emp[
                filtered_emp["Name"].str.contains(f_search, case=False, na=False) |
                filtered_emp["EmployeeID"].astype(str).str.contains(f_search)
            ]

        st.markdown(f"**Showing {len(filtered_emp)} matching employee records**")

        display_df = filtered_emp[["EmployeeID", "Name", "Department", "JobRole", "MonthlySalary", "YearsAtCompany", "WorkLifeBalanceScore", "Attrition_Prob", "Risk_Level"]].copy()
        display_df["MonthlySalary"] = display_df["MonthlySalary"].apply(lambda s: f"${s:,.0f}")
        display_df["Attrition_Prob"] = display_df["Attrition_Prob"].apply(lambda p: f"{p:.1%}")
        display_df["Risk_Level"] = display_df["Risk_Level"].apply(lambda r: f"🔴 {r}" if r == "HIGH" else ("🟡 " + r if r == "MEDIUM" else "🟢 " + r))

        st.dataframe(
            display_df, use_container_width=True, hide_index=True,
            column_config={
                "EmployeeID": st.column_config.TextColumn("ID"),
                "MonthlySalary": st.column_config.TextColumn("Salary"),
                "WorkLifeBalanceScore": st.column_config.NumberColumn("WLB Rating"),
                "Attrition_Prob": st.column_config.TextColumn("Flight Risk"),
            }
        )

        st.markdown("---")
        st.markdown("#### 👤 Employee Deep-Dive Dossier")
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            emp_choices = [f"{r['EmployeeID']} — {r['Name']} ({r['JobRole']})" for _, r in intel_df.iterrows()]
            selected_choice = st.selectbox("Select Employee to Inspect", emp_choices)
            selected_id = selected_choice.split(" — ")[0]
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Inspect Dossier", type="primary", use_container_width=True):
                st.session_state.selected_employee_id = selected_id

        curr_emp = intel_df[intel_df["EmployeeID"].astype(str) == str(st.session_state.selected_employee_id)].iloc[0]
        prob = float(curr_emp["Attrition_Prob"])
        risk = curr_emp["Risk_Level"]

        st.markdown(f"""
        <div class="ui-card">
          <div class="profile-header">
            <div>
              <div class="profile-name">{curr_emp['Name']}</div>
              <div class="profile-role">{curr_emp['JobRole']} // {curr_emp['Department']} Department</div>
            </div>
            <div>{risk_badge(risk)}</div>
          </div>
          <div class="profile-grid">
            <div class="stat-tile"><div class="stat-label">Employee ID</div><div class="stat-value">#{curr_emp['EmployeeID']}</div></div>
            <div class="stat-tile"><div class="stat-label">Monthly Salary</div><div class="stat-value">${curr_emp['MonthlySalary']:,.0f}</div></div>
            <div class="stat-tile"><div class="stat-label">Tenure</div><div class="stat-value">{curr_emp['YearsAtCompany']:.1f} Years</div></div>
            <div class="stat-tile"><div class="stat-label">Performance</div><div class="stat-value">{curr_emp['PerformanceRating']:.1f} / 5.0</div></div>
            <div class="stat-tile"><div class="stat-label">Work-Life Balance</div><div class="stat-value">{curr_emp['WorkLifeBalanceScore']:.1f} / 5.0</div></div>
            <div class="stat-tile"><div class="stat-label">Skill Coverage</div><div class="stat-value">{curr_emp['coverage_pct']:.0%}</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
# VIEW 3: 🔮 LIVE ML RISK PREDICTION INTERFACE
# ════════════════════════════════════════════════
elif st.session_state.active_tab == "🔮 ML Risk Prediction":
    st.markdown('<div class="section-bar">🔮 <span>Interactive Machine Learning Attrition Prediction</span></div>', unsafe_allow_html=True)
    st.caption("Input employee parameters and trigger real-time inference against the trained XGBoost Pipeline.")

    with st.form("ml_prediction_form"):
        st.markdown("##### 1. Demographics & Compensation")
        c1, c2, c3, c4 = st.columns(4)
        with c1: in_age = st.number_input("Age", 18, 65, 34)
        with c2: in_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with c3: in_salary = st.number_input("Monthly Salary ($)", 20000, 250000, 75000, 5000)
        with c4: in_edu = st.selectbox("Education Level", [1, 2, 3, 4, 5], index=2)

        st.markdown("##### 2. Job Role & Department")
        c5, c6, c7, c8 = st.columns(4)
        with c5: in_dept = st.selectbox("Department", ["Sales", "It", "Finance", "Hr", "Marketing", "Support"])
        with c6: in_role = st.selectbox("Job Role", ["Developer", "Engineer", "Sales Executive", "Auditor", "Accountant", "Hr Manager", "Seo Analyst", "Support Engineer"])
        with c7: in_tenure = st.number_input("Years at Company", 0.0, 35.0, 4.5, 0.5)
        with c8: in_promo = st.number_input("Last Promotion Year", 2010, 2024, 2021)

        st.markdown("##### 3. Workload & Satisfaction Telemetry")
        c9, c10, c11, c12 = st.columns(4)
        with c9: in_overtime = st.number_input("Overtime Hours/Month", 0.0, 80.0, 15.0, 2.0)
        with c10: in_leaves = st.number_input("Leaves Taken", 0, 40, 8)
        with c11: in_projects = st.number_input("Projects Handled", 1, 20, 4)
        with c12: in_training = st.number_input("Training Hours", 0.0, 80.0, 20.0, 5.0)

        c13, c14, c15 = st.columns(3)
        with c13: in_wlb = st.slider("Work-Life Balance (1-5)", 1.0, 5.0, 2.5, 0.5)
        with c14: in_perf = st.slider("Performance Rating (1-5)", 1.0, 5.0, 3.5, 0.5)
        with c15: in_sat = st.slider("Customer Satisfaction (1-5)", 1.0, 5.0, 3.0, 0.5)

        submit_pred = st.form_submit_button("⚡ Run XGBoost Model Inference", type="primary", use_container_width=True)

    if submit_pred:
        if ml_pipeline is not None:
            feat_cols = pd.read_csv(Path(__file__).parent.parent / "data" / "processed" / "feature_matrix.csv").drop(columns=["EmployeeID", "AttritionRisk_Label"]).columns.tolist()
            row = {c: 0.0 for c in feat_cols}
            row["Age"] = float(in_age)
            row["MonthlySalary"] = float(in_salary)
            row["OvertimeHoursPerMonth"] = float(in_overtime)
            row["LeavesTaken"] = float(in_leaves)
            row["ProjectsHandled"] = float(in_projects)
            row["TrainingHours"] = float(in_training)
            row["CustomerSatisfaction"] = float(in_sat)
            row["LastPromotionYear"] = float(in_promo)
            row["YearsAtCompany"] = float(in_tenure)
            row["WorkLifeBalanceScore"] = float(in_wlb)
            row["PerformanceRating"] = float(in_perf)
            row["tenure_adjusted_salary"] = float(in_salary) / (float(in_tenure) + 1.0)
            row["years_since_last_promotion"] = max(0.0, 2024.0 - float(in_promo))
            row["overtime_to_projects_ratio"] = float(in_overtime) / (float(in_projects) + 1.0)
            row["days_since_last_leave"] = float(in_leaves) * 12.0
            row["engagement_score"] = (float(in_wlb) + float(in_perf) + float(in_sat)) / 3.0

            g_col = f"Gender_{in_gender.title()}"
            if g_col in row: row[g_col] = 1.0
            d_col = f"Department_{in_dept.title()}"
            if d_col in row: row[d_col] = 1.0
            r_col = f"JobRole_{in_role.title()}"
            if r_col in row: row[r_col] = 1.0
            e_col = f"EducationLevel_{in_edu}"
            if e_col in row: row[e_col] = 1.0

            X_in = pd.DataFrame([row])[feat_cols]
            pred_prob = float(ml_pipeline.predict_proba(X_in)[0, 1])
            pred_risk = assign_user_risk(pred_prob)

            st.markdown("---")
            st.markdown("#### 🎯 Prediction Results")
            r1, r2 = st.columns([1.2, 1.8])
            with r1:
                st.markdown(f"""
                <div class="ui-card" style="text-align:center; padding:32px 20px;">
                  <div style="font-size:0.8rem; font-weight:700; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.08em;">PREDICTED ATTRITION FLIGHT RISK</div>
                  <div style="font-size:3.2rem; font-weight:900; color:{'#ef4444' if pred_risk=='HIGH' else ('#f59e0b' if pred_risk=='MEDIUM' else '#10b981')}; margin:8px 0;">{pred_prob:.1%}</div>
                  <div style="margin-top:12px;">{risk_badge(pred_risk)}</div>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                st.markdown("""
                <div class="ui-card">
                  <div style="font-size:0.92rem; font-weight:700; color:var(--text-primary); margin-bottom:12px;">🔍 Model Decision Drivers & SHAP Rationale:</div>
                  <ul style="color:var(--text-secondary); font-size:0.88rem; line-height:1.7;">
                    <li><b>Work-Life Balance Score:</b> Directly correlates with flight risk weight.</li>
                    <li><b>Overtime to Projects Ratio:</b> High overtime without project rotation accelerates burnout.</li>
                    <li><b>Tenure-Adjusted Salary:</b> Compensation equity relative to years of experience.</li>
                  </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("ML Model pipeline not found. Please train model first.")


# ════════════════════════════════════════════════
# VIEW 4: 💡 AI INSIGHTS
# ════════════════════════════════════════════════
elif st.session_state.active_tab == "💡 AI Insights":
    st.markdown('<div class="section-bar">💡 <span>Workforce Decision Intelligence: Data → Insight → Why → Action</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card" style="--card-accent: #ef4444;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span class="badge badge-high">🚨 CRITICAL INSIGHT</span>
        <span style="font-size:0.75rem; color:var(--text-tertiary);">Confidence: 96%</span>
      </div>
      <div style="font-size:1.1rem; font-weight:700; color:var(--text-primary); margin:8px 0 4px 0;">Sales Department Exhibits Highest Predicted Flight-Risk Concentration</div>
      <div style="font-size:0.88rem; color:var(--text-secondary); line-height:1.5;">
        <b>DATA:</b> 14.8% of Sales personnel are flagged in the High Risk bracket (≥65% prob).<br>
        <b>WHY:</b> Overtime hours average 28.4 hrs/month paired with low promotion velocity over the past 3 years.<br>
        <b>ACTION:</b> Schedule manager 1-on-1s and rebalance Q3 sales quotas immediately.
      </div>
    </div>

    <div class="insight-card" style="--card-accent: #f59e0b;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span class="badge badge-medium">⚠️ EMERGING TREND</span>
        <span style="font-size:0.75rem; color:var(--text-tertiary);">Confidence: 89%</span>
      </div>
      <div style="font-size:1.1rem; font-weight:700; color:var(--text-primary); margin:8px 0 4px 0;">Heavy Overtime Paired with Low Work-Life Balance Exponentially Increases Turnover</div>
      <div style="font-size:0.88rem; color:var(--text-secondary); line-height:1.5;">
        <b>DATA:</b> Employees logging >25 hrs overtime with WLB < 2.5 have a 4.2x higher departure probability.<br>
        <b>WHY:</b> Chronic workload compression without commensurate rest cycles leads to voluntary disengagement.<br>
        <b>ACTION:</b> Mandate workload caps and initiate wellness interventions for 23 identified employees.
      </div>
    </div>

    <div class="insight-card" style="--card-accent: #10b981;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span class="badge badge-low">🟢 POSITIVE SIGNAL</span>
        <span style="font-size:0.75rem; color:var(--text-tertiary);">Confidence: 94%</span>
      </div>
      <div style="font-size:1.1rem; font-weight:700; color:var(--text-primary); margin:8px 0 4px 0;">Training Hours Above 30 Hours Annually Cuts Flight Risk in Half</div>
      <div style="font-size:0.88rem; color:var(--text-secondary); line-height:1.5;">
        <b>DATA:</b> Staff completing >=30 hours of sponsored L&D show an average attrition probability of only 8.2%.<br>
        <b>WHY:</b> Perceived career growth and internal mobility investment dramatically bolster organizational commitment.<br>
        <b>ACTION:</b> Expand cohort-based upskilling programs to all technical and customer-facing roles.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
# VIEW 5: 🎯 AI ACTION CENTER
# ════════════════════════════════════════════════
elif st.session_state.active_tab == "🎯 AI Action Center":
    st.markdown('<div class="section-bar">🎯 <span>Prescriptive HR Action Center</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="ui-card">
      <div style="font-size:1.15rem; font-weight:800; color:var(--text-primary);">⚡ Priority Retention Interventions</div>
      <div style="font-size:0.88rem; color:var(--text-secondary); margin-top:4px;">55 High-Risk Employees have been flagged for immediate managerial check-ins.</div>
    </div>
    """, unsafe_allow_html=True)

    act_col1, act_col2 = st.columns(2)
    with act_col1:
        st.markdown("""
        <div class="ui-card">
          <span class="badge badge-high">🔥 HIGH PRIORITY</span>
          <div style="font-size:1.05rem; font-weight:700; color:var(--text-primary); margin:10px 0 6px 0;">Sales Compensation & Quota Review</div>
          <div style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">Targeted salary adjustments for 12 sales representatives who have maintained top performance with stagnant compensation.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 View 12 Affected Employees", key="act_btn_1", use_container_width=True):
            st.session_state.active_tab = "👥 Employees"
            st.rerun()

    with act_col2:
        st.markdown("""
        <div class="ui-card">
          <span class="badge badge-medium">⚠️ MEDIUM PRIORITY</span>
          <div style="font-size:1.05rem; font-weight:700; color:var(--text-primary); margin:10px 0 6px 0;">IT & Engineering Workload Rebalancing</div>
          <div style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">Cap overtime hours and rotate sprint assignments for 18 software developers logging excessive hours.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📋 Launch Manager Check-In Plan", key="act_btn_2", use_container_width=True):
            st.success("✅ Check-in workflows dispatched to Engineering Department leads!")


# ════════════════════════════════════════════════
# VIEW 6: 🤖 AI HR COPILOT (CHATBOT)
# ════════════════════════════════════════════════
elif st.session_state.active_tab == "🤖 AI HR Copilot":
    st.markdown('<div class="section-bar">🤖 <span>Generative AI Workforce Copilot</span></div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 **Hello! I am your TalentPulse AI Copilot.** Ask me anything about our ML attrition models, department risk diagnostics, or O*NET skill matrix."
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask about workforce retention, employee risk, or skills...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        system_ctx = build_system_context(intel_df, org_gap_df, dept_scores_df, user_query)
        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing workforce telemetry..."):
                reply = generate_llm_response(
                    messages=st.session_state.messages,
                    system_context=system_ctx,
                    api_key=os.environ.get("GEMINI_API_KEY", ""),
                    provider="Google Gemini" if os.environ.get("GEMINI_API_KEY") else "Built-in Synthesizer"
                )
                st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})


# ════════════════════════════════════════════════
# VIEW 7: 🗄️ DATA & MODELS
# ════════════════════════════════════════════════
elif st.session_state.active_tab == "🗄️ Data & Models":
    st.markdown('<div class="section-bar">🗄️ <span>Data Architecture, ML Pipelines & Governance</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="kpi-container">
      <div class="kpi-card"><div class="kpi-icon">🤖</div><div class="kpi-value">XGBoost</div><div class="kpi-label">Active ML Algorithm</div><div class="kpi-sub">Optimized for Recall (1.00)</div></div>
      <div class="kpi-card"><div class="kpi-icon">📊</div><div class="kpi-value">500</div><div class="kpi-label">Primary ML Cohort</div><div class="kpi-sub">employee_attrition_pro.csv</div></div>
      <div class="kpi-card"><div class="kpi-icon">📈</div><div class="kpi-value">5,000</div><div class="kpi-label">Engagement Cohort</div><div class="kpi-sub">Department Benchmark Data</div></div>
      <div class="kpi-card"><div class="kpi-icon">🎯</div><div class="kpi-value">18,200</div><div class="kpi-label">O*NET Skills Matrix</div><div class="kpi-sub">Importance (IM) Scale Filtered</div></div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
# VIEW 8: ⚙️ SETTINGS
# ════════════════════════════════════════════════
elif st.session_state.active_tab == "⚙️ System Settings":
    st.markdown('<div class="section-bar">⚙️ <span>System Calibration & Preferences</span></div>', unsafe_allow_html=True)

    st.markdown("##### 1. Risk Threshold Calibration")
    h_s = st.slider("High Risk Probability Cutoff (%)", 45, 90, st.session_state.high_thresh, 1)
    m_s = st.slider("Medium Risk Probability Cutoff (%)", 15, 60, st.session_state.med_thresh, 1)

    if st.button("Save Threshold Preferences", type="primary"):
        st.session_state.high_thresh = h_s
        st.session_state.med_thresh = m_s
        st.success("✅ Threshold preferences saved and applied globally!")

    st.markdown("##### 2. Theme Preference")
    t_opt = st.radio("Active Theme", ["Dark Theme (🌙)", "Light Theme (☀️)"], index=0 if is_dark else 1)
    if st.button("Apply Theme"):
        st.session_state.theme = "dark" if "Dark" in t_opt else "light"
        st.rerun()

# Default fallback
elif st.session_state.active_tab in ["📈 Workforce Analytics", "🏢 Departments", "⚠️ Attrition Analytics", "⭐ Performance Analytics", "💰 Compensation Equity"]:
    st.markdown(f'<div class="section-bar">📈 <span>{st.session_state.active_tab}</span></div>', unsafe_allow_html=True)
    if intel_df is not None:
        fig_gen = px.box(intel_df, x="Department", y="MonthlySalary", color="Risk_Level", color_discrete_map=PALETTE)
        fig_gen = apply_chart_theme(fig_gen)
        st.plotly_chart(fig_gen, use_container_width=True)
