"""
Enterprise HR AI — Workforce Intelligence & Retention Platform
Enterprise Dark Slate & Indigo Edition — Clean, Minimalist & Emoji-Free
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
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.services.llm_chat_service import build_system_context, generate_llm_response

# ── 1. Page Configuration ──
st.set_page_config(
    page_title="Workforce Intelligence Platform",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Enterprise Slate & Indigo Design System ──
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg-main: #0B0F17;
    --bg-card: #151C28;
    --bg-card-hover: #1A2332;
    --bg-subtle: #1C2638;
    
    --border-color: #20293A;
    --border-accent: rgba(99, 102, 241, 0.4);
    
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-dim: #64748B;
    
    --color-primary: #6366F1;
    --color-primary-light: #818CF8;
    --color-primary-glow: rgba(99, 102, 241, 0.15);
    
    --risk-high: #EF4444;
    --risk-med: #F59E0B;
    --risk-low: #10B981;
    
    --radius-card: 8px;
    --radius-sm: 6px;
    --radius-full: 9999px;
  }

  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: var(--text-primary);
  }

  .stApp {
    background-color: var(--bg-main);
  }

  /* Sidebar styling */
  [data-testid="stSidebar"] {
    background-color: #0E1420 !important;
    border-right: 1px solid var(--border-color) !important;
  }
  [data-testid="stSidebar"] > div:first-child {
    padding-bottom: 60px !important;
  }

  /* Top Navigation Bar */
  .header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-card);
    margin-bottom: 24px;
  }
  .header-title-group {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .header-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
  }
  .header-subtitle {
    font-size: 0.78rem;
    color: var(--text-secondary);
    font-weight: 500;
  }
  .status-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: var(--radius-full);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--risk-low);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: var(--risk-low);
  }

  /* KPI Executive Metrics Row */
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }
  .metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-card);
    padding: 18px 20px;
    transition: border-color 0.2s ease, transform 0.2s ease;
  }
  .metric-card:hover {
    border-color: var(--border-accent);
    transform: translateY(-2px);
  }
  .metric-title {
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .metric-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 4px 0 2px 0;
    letter-spacing: -0.02em;
  }
  .metric-subtitle {
    font-size: 0.74rem;
    color: var(--text-dim);
  }

  /* Section Title Bar */
  .section-heading {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin: 24px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-color);
  }

  /* Clean Risk Badges (No Emojis) */
  .badge-container {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  .badge-high {
    background: rgba(239, 68, 68, 0.12);
    color: #FCA5A5;
    border: 1px solid rgba(239, 68, 68, 0.3);
  }
  .badge-medium {
    background: rgba(245, 158, 11, 0.12);
    color: #FCD34D;
    border: 1px solid rgba(245, 158, 11, 0.3);
  }
  .badge-low {
    background: rgba(16, 185, 129, 0.12);
    color: #6EE7B7;
    border: 1px solid rgba(16, 185, 129, 0.3);
  }

  /* Cards & Surface Components */
  .surface-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-card);
    padding: 20px;
    margin-bottom: 20px;
  }

  .stat-grid-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-top: 16px;
    padding-top: 14px;
    border-top: 1px solid var(--border-color);
  }
  .stat-tile-box {
    background: var(--bg-subtle);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    padding: 12px 14px;
  }
  .stat-tile-lbl {
    font-size: 0.7rem;
    color: var(--text-secondary);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .stat-tile-val {
    font-size: 0.96rem;
    color: var(--text-primary);
    font-weight: 700;
    margin-top: 2px;
  }

  /* Form & Native Widgets Override */
  [data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-card) !important;
    margin-bottom: 12px !important;
  }

  [data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    padding: 14px !important;
  }

  .stButton button {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
  }
  .stButton button[kind="primary"] {
    background-color: var(--color-primary) !important;
    border: 1px solid var(--color-primary) !important;
    color: #ffffff !important;
  }

  /* Sidebar Collapsible Chevron */
  [data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    color: var(--color-primary) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    padding: 6px !important;
  }

  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── 3. Data Loaders & Model Loading ──
DEPT_ACRONYM_MAP = {"It": "IT", "Hr": "HR", "it": "IT", "hr": "HR"}

def clean_dept_name(name):
    return DEPT_ACRONYM_MAP.get(str(name), str(name))

@st.cache_data(ttl=300)
def load_datasets():
    base = Path(__file__).parent.parent / "data" / "processed"
    intel = pd.read_csv(base / "employee_intelligence.csv") if (base / "employee_intelligence.csv").exists() else None
    org_gap = pd.read_csv(base / "org_skill_gap.csv") if (base / "org_skill_gap.csv").exists() else None
    dept_s = pd.read_csv(base / "department_composite_scores.csv") if (base / "department_composite_scores.csv").exists() else None

    if intel is not None and "Department" in intel.columns:
        intel["Department"] = intel["Department"].apply(clean_dept_name)
    if dept_s is not None and "Department" in dept_s.columns:
        dept_s["Department"] = dept_s["Department"].apply(clean_dept_name)

    return intel, org_gap, dept_s

@st.cache_resource
def load_model():
    p = Path(__file__).parent.parent / "models" / "v1" / "attrition_pipeline.joblib"
    if not p.exists(): p = Path(__file__).parent.parent / "models" / "attrition_pipeline.joblib"
    return joblib.load(p) if p.exists() else None

intel_df, org_gap_df, dept_scores_df = load_datasets()
ml_model = load_model()

# ── 4. Sidebar Controls & Threshold Calibration ──
if "high_thresh" not in st.session_state: st.session_state.high_thresh = 65
if "med_thresh" not in st.session_state: st.session_state.med_thresh = 40

with st.sidebar:
    st.markdown("### TalentPulse AI")
    st.caption("Workforce Intelligence Platform")
    st.markdown("---")
    
    NAV_ITEMS = [
        "Executive Overview",
        "Attrition Risk Analytics",
        "Competency & Skill Gaps",
        "Employee Directory",
        "ML Risk Prediction",
        "Workforce AI Assistant"
    ]
    nav = st.radio("Navigation", NAV_ITEMS, label_visibility="collapsed")
    
    st.markdown("---")
    
    with st.expander("Risk Threshold Calibration", expanded=False):
        st.session_state.high_thresh = st.slider("High Risk Cutoff (%)", 45, 90, st.session_state.high_thresh, 1)
        st.session_state.med_thresh = st.slider("Medium Risk Cutoff (%)", 15, 60, st.session_state.med_thresh, 1)
        st.caption(f"High: ≥{st.session_state.high_thresh}% | Medium: {st.session_state.med_thresh}%–{st.session_state.high_thresh}% | Low: <{st.session_state.med_thresh}%")
        if st.button("Reset Defaults (65/40)", use_container_width=True):
            st.session_state.high_thresh = 65
            st.session_state.med_thresh = 40
            st.rerun()

    with st.expander("AI Engine Configuration", expanded=False):
        ai_provider = st.selectbox("Provider", ["Google Gemini", "OpenAI / Compatible", "Built-in Synthesizer"], index=0)
        ai_key = st.text_input("API Key", value=os.environ.get("GEMINI_API_KEY", ""), type="password", placeholder="AIzaSy...")

    with st.expander("Model Telemetry", expanded=False):
        st.caption("Algorithm: XGBoost Classifier\nRecall: 1.00 (100%)\nF1-Score: 1.00\nCohort: 500 Monitored Staff")

# Apply dynamic threshold segmentation
h_p = st.session_state.high_thresh / 100.0
m_p = st.session_state.med_thresh / 100.0

def assign_risk_category(prob):
    if prob >= h_p: return "HIGH"
    elif prob >= m_p: return "MEDIUM"
    return "LOW"

if intel_df is not None:
    intel_df["Risk_Level"] = intel_df["Attrition_Prob"].apply(assign_risk_category)

# Unified Color Palette Mapping
PALETTE = {
    "HIGH": "#EF4444",      # Coral Red
    "MEDIUM": "#F59E0B",    # Amber
    "LOW": "#10B981",       # Emerald Green
    "primary": "#6366F1",   # Indigo Blue
    "slate": "#94A3B8"      # Slate Gray
}

def style_chart(fig):
    fig.update_layout(
        plot_bgcolor="#151C28",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC", family="Inter, sans-serif", size=11),
        xaxis=dict(gridcolor="#20293A", linecolor="#20293A", zerolinecolor="#20293A"),
        yaxis=dict(gridcolor="#20293A", linecolor="#20293A", zerolinecolor="#20293A"),
        margin=dict(t=25, b=25, l=15, r=15),
    )
    return fig

def render_risk_badge(risk: str) -> str:
    css = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(risk, "badge-low")
    return f'<span class="badge-container {css}">{risk} RISK</span>'


# ── 5. Clean Top Navigation Header (No Emojis) ──
now_timestamp = datetime.now().strftime("%H:%M:%S UTC")
st.markdown(f"""
<div class="header-bar">
  <div class="header-title-group">
    <div>
      <div class="header-title">{nav}</div>
      <div class="header-subtitle">Workforce analytics and predictive turnover modeling</div>
    </div>
  </div>
  <div>
    <span class="status-tag"><span class="status-dot"></span> System Active · {now_timestamp}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
# VIEW 1: EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════
if nav == "Executive Overview":
    if intel_df is not None:
        total_headcount = len(intel_df)
        high_risk_count = (intel_df["Risk_Level"] == "HIGH").sum()
        med_risk_count = (intel_df["Risk_Level"] == "MEDIUM").sum()
        avg_flight_risk = intel_df["Attrition_Prob"].mean()
        avg_base_salary = intel_df["MonthlySalary"].mean()

        st.markdown(f"""
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-title">Total Headcount</div>
            <div class="metric-value">{total_headcount:,}</div>
            <div class="metric-subtitle">Active monitored staff</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">High Flight Risk</div>
            <div class="metric-value" style="color:#EF4444;">{high_risk_count}</div>
            <div class="metric-subtitle">{high_risk_count/total_headcount:.1%} of workforce (≥{st.session_state.high_thresh}%)</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Watchlist (Medium Risk)</div>
            <div class="metric-value" style="color:#F59E0B;">{med_risk_count}</div>
            <div class="metric-subtitle">{f'{med_risk_count} staff in {st.session_state.med_thresh}%–{st.session_state.high_thresh}% band' if med_risk_count > 0 else '0 in configured band'}</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Average Flight Risk</div>
            <div class="metric-value">{avg_flight_risk:.1%}</div>
            <div class="metric-subtitle">Org-wide average (15% benchmark)</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Average Base Salary</div>
            <div class="metric-value">${avg_base_salary:,.0f}</div>
            <div class="metric-subtitle">Annual base compensation</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([1.4, 1])

        with col_left:
            st.markdown('<div class="section-heading">High-Risk Concentration by Department</div>', unsafe_allow_html=True)
            dept_stats = intel_df.groupby("Department").agg(
                Total=("EmployeeID", "count"),
                High=("Risk_Level", lambda x: (x == "HIGH").sum()),
            ).reset_index()
            dept_stats["High_Pct"] = (dept_stats["High"] / dept_stats["Total"] * 100).round(1)
            dept_stats = dept_stats.sort_values("High_Pct", ascending=True)

            fig_bar = go.Figure(go.Bar(
                y=dept_stats["Department"],
                x=dept_stats["High_Pct"],
                orientation="h",
                marker=dict(
                    color=dept_stats["High_Pct"],
                    colorscale=[[0, PALETTE["LOW"]], [0.5, PALETTE["MEDIUM"]], [1.0, PALETTE["HIGH"]]],
                    showscale=False
                ),
                text=[f"{v:.1f}%" for v in dept_stats["High_Pct"]],
                textposition="outside",
                textfont=dict(color="#F8FAFC", size=11),
                hovertemplate="<b>%{y}</b><br>High-Risk Proportion: %{x:.1f}%<br>High-Risk Headcount: %{customdata[0]}<br>Department Total: %{customdata[1]}<extra></extra>",
                customdata=dept_stats[["High", "Total"]].values,
            ))
            fig_bar.update_layout(height=280, xaxis_title="% of Department Flagged High Risk", yaxis_title="")
            fig_bar = style_chart(fig_bar)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-heading">Workforce Risk Distribution</div>', unsafe_allow_html=True)
            risk_keys = ["HIGH", "MEDIUM", "LOW"]
            counts_map = intel_df["Risk_Level"].value_counts().to_dict()
            pie_values = [counts_map.get(k, 0) for k in risk_keys]
            pie_colors = [PALETTE[k] for k in risk_keys]

            fig_pie = go.Figure(go.Pie(
                labels=[f"{k} RISK" for k in risk_keys],
                values=pie_values,
                hole=0.62,
                marker=dict(colors=pie_colors),
                textinfo="label+percent",
                textfont=dict(color="#F8FAFC", size=11),
                hovertemplate="<b>%{label}</b><br>Headcount: %{value} employees (%{percent})<extra></extra>",
                sort=False
            ))
            fig_pie.update_layout(
                height=280,
                showlegend=True,
                legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center", font=dict(size=11, color="#94A3B8")),
                annotations=[dict(text=f"<b>{total_headcount}</b><br>Staff", x=0.5, y=0.5, font_size=15, showarrow=False, font=dict(color="#F8FAFC"))]
            )
            fig_pie = style_chart(fig_pie)
            st.plotly_chart(fig_pie, use_container_width=True)


# ════════════════════════════════════════════════
# VIEW 2: ATTRITION RISK ANALYTICS
# ════════════════════════════════════════════════
elif nav == "Attrition Risk Analytics":
    st.markdown('<div class="section-heading">Attrition Risk Dispersion & Identified Roster</div>', unsafe_allow_html=True)

    if intel_df is not None:
        f_c1, f_c2, f_c3 = st.columns([1.5, 1.5, 1])
        with f_c1: filter_dept = st.selectbox("Department", ["All Departments"] + sorted(intel_df["Department"].dropna().unique().tolist()))
        with f_c2: filter_risk = st.selectbox("Risk Level", ["All Severity Levels", "HIGH", "MEDIUM", "LOW"])
        with f_c3: display_limit = st.slider("Display Limit", 10, 100, 25)

        filtered_data = intel_df.copy()
        if filter_dept != "All Departments": filtered_data = filtered_data[filtered_data["Department"] == filter_dept]
        if filter_risk != "All Severity Levels": filtered_data = filtered_data[filtered_data["Risk_Level"] == filter_risk]

        chart_c1, chart_c2 = st.columns(2)
        with chart_c1:
            fig_hist = px.histogram(
                filtered_data, x="Attrition_Prob", nbins=24, title="<b>Model Flight Probability Distribution</b>",
                color_discrete_sequence=[PALETTE["primary"]],
                labels={"Attrition_Prob": "Predicted Probability"}
            )
            fig_hist.add_vline(x=h_p, line_dash="dash", line_color=PALETTE["HIGH"], annotation_text=f"HIGH (≥{st.session_state.high_thresh}%)", annotation_font_color=PALETTE["HIGH"])
            fig_hist.add_vline(x=m_p, line_dash="dash", line_color=PALETTE["MEDIUM"], annotation_text=f"MED (≥{st.session_state.med_thresh}%)", annotation_font_color=PALETTE["MEDIUM"])
            fig_hist = style_chart(fig_hist)
            fig_hist.update_layout(height=280)
            st.plotly_chart(fig_hist, use_container_width=True)

        with chart_c2:
            fig_box = px.box(
                intel_df, x="Department", y="Attrition_Prob", color="Department",
                title="<b>Departmental Risk Dispersion Box Plot</b>",
                labels={"Attrition_Prob": "Flight Risk Probability"}
            )
            fig_box = style_chart(fig_box)
            fig_box.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown(f"**Identified At-Risk Employees ({len(filtered_data)} matching records)**")
        cols_to_show = ["EmployeeID", "Name", "Department", "JobRole", "Attrition_Prob", "Risk_Level", "YearsAtCompany", "Top_Skill_Gap"]
        roster_table = filtered_data.nlargest(display_limit, "Attrition_Prob")[cols_to_show].copy()
        roster_table["Attrition_Prob"] = roster_table["Attrition_Prob"].apply(lambda p: f"{p:.1%}")
        roster_table["Risk_Level"] = roster_table["Risk_Level"].apply(lambda r: f"{r} RISK")

        st.dataframe(
            roster_table, use_container_width=True, hide_index=True,
            column_config={
                "EmployeeID": st.column_config.TextColumn("ID"),
                "Attrition_Prob": st.column_config.TextColumn("Flight Risk"),
                "Risk_Level": st.column_config.TextColumn("Classification"),
                "YearsAtCompany": st.column_config.NumberColumn("Tenure (Years)"),
                "Top_Skill_Gap": st.column_config.TextColumn("Primary Skill Deficit"),
            }
        )


# ════════════════════════════════════════════════
# VIEW 3: COMPETENCY & SKILL GAPS
# ════════════════════════════════════════════════
elif nav == "Competency & Skill Gaps":
    st.markdown('<div class="section-heading">Organizational Competency Architecture & Skill Deficits</div>', unsafe_allow_html=True)

    if org_gap_df is not None:
        s_c1, s_c2 = st.columns([1.5, 1])
        with s_c1: severity_choice = st.selectbox("Severity Classification", ["All Deficits", "HIGH", "MEDIUM", "LOW"])
        with s_c2: top_count = st.slider("Number of Skills", 10, 40, 15)

        gap_subset = org_gap_df if severity_choice == "All Deficits" else org_gap_df[org_gap_df["severity"] == severity_choice]
        top_skills = gap_subset.nlargest(top_count, "total_gap_weight")

        fig_skills = px.bar(
            top_skills, x="total_gap_weight", y="skill", orientation="h", color="severity",
            color_discrete_map=PALETTE, title="<b>Top Deficits by O*NET Weighted Importance</b>",
            labels={"total_gap_weight": "Total Gap Score", "skill": "Skill Name"}
        )
        fig_skills = style_chart(fig_skills)
        fig_skills.update_layout(height=max(360, top_count * 20), yaxis_title="", xaxis_title="Total Gap Score (Headcount × Importance)")
        st.plotly_chart(fig_skills, use_container_width=True)

        st.dataframe(
            top_skills[["skill", "severity", "pct_employees_lacking", "avg_importance", "total_gap_weight", "employees_lacking"]].rename(columns={
                "skill": "Skill Element", "severity": "Severity Tier", "pct_employees_lacking": "% Employees Lacking",
                "avg_importance": "O*NET Importance", "total_gap_weight": "Weighted Gap Score", "employees_lacking": "Affected Headcount"
            }),
            use_container_width=True, hide_index=True
        )


# ════════════════════════════════════════════════
# VIEW 4: EMPLOYEE DIRECTORY
# ════════════════════════════════════════════════
elif nav == "Employee Directory":
    st.markdown('<div class="section-heading">Employee Profile & Record Lookup</div>', unsafe_allow_html=True)

    if intel_df is not None:
        search_query = st.text_input("Search Employee ID (1–500) or Full Name", value="1")
        matched_emp = None
        if search_query:
            match_by_id = intel_df[intel_df["EmployeeID"].astype(str) == str(search_query.strip())]
            if not match_by_id.empty: matched_emp = match_by_id.iloc[0]
            else:
                match_by_name = intel_df[intel_df["Name"].str.contains(search_query.strip(), case=False, na=False)]
                if not match_by_name.empty: matched_emp = match_by_name.iloc[0]

        if matched_emp is not None:
            prob_val = float(matched_emp["Attrition_Prob"])
            risk_val = matched_emp["Risk_Level"]

            st.markdown(f"""
            <div class="surface-card">
              <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                  <div style="font-size:1.4rem; font-weight:700; color:#F8FAFC;">{matched_emp['Name']}</div>
                  <div style="font-size:0.88rem; color:#94A3B8; margin-top:2px;">{matched_emp['JobRole']} · {matched_emp['Department']} Department</div>
                </div>
                <div>{render_risk_badge(risk_val)}</div>
              </div>
              <div class="stat-grid-row">
                <div class="stat-tile-box"><div class="stat-tile-lbl">Employee ID</div><div class="stat-tile-val">#{matched_emp['EmployeeID']}</div></div>
                <div class="stat-tile-box"><div class="stat-tile-lbl">Annual Salary</div><div class="stat-tile-val">${matched_emp['MonthlySalary']:,.0f}</div></div>
                <div class="stat-tile-box"><div class="stat-tile-lbl">Tenure</div><div class="stat-tile-val">{matched_emp['YearsAtCompany']:.1f} Years</div></div>
                <div class="stat-tile-box"><div class="stat-tile-lbl">Work-Life Balance</div><div class="stat-tile-val">{matched_emp['WorkLifeBalanceScore']:.1f} / 5.0</div></div>
                <div class="stat-tile-box"><div class="stat-tile-lbl">Primary Skill Deficit</div><div class="stat-tile-val">{matched_emp.get('Top_Skill_Gap', 'None')}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            col_gauge, col_sim = st.columns([1, 1.4])
            with col_gauge:
                g_color = PALETTE.get(risk_val, "#6366F1")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob_val * 100, number={"suffix": "%", "font": {"color": "#F8FAFC", "size": 34}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#94A3B8"},
                        "bar": {"color": g_color},
                        "steps": [
                            {"range": [0, st.session_state.med_thresh], "color": "rgba(16, 185, 129, 0.15)"},
                            {"range": [st.session_state.med_thresh, st.session_state.high_thresh], "color": "rgba(245, 158, 11, 0.15)"},
                            {"range": [st.session_state.high_thresh, 100], "color": "rgba(239, 68, 68, 0.15)"},
                        ],
                    },
                    title={"text": f"<b>Flight Risk Meter: {risk_val} RISK</b>", "font": {"color": g_color, "size": 13}}
                ))
                fig_gauge.update_layout(height=240, margin=dict(t=20, b=10))
                fig_gauge = style_chart(fig_gauge)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_sim:
                st.markdown("##### Retention Intervention Simulator")
                sim_comp = st.slider("Proposed Compensation Increase (%)", 0, 40, 10, 5, key="dir_sim_comp")
                sim_wlb = st.slider("Target Work-Life Balance", 1.0, 5.0, min(5.0, float(matched_emp['WorkLifeBalanceScore']) + 1.0), 0.5, key="dir_sim_wlb")
                
                reduction = (sim_comp / 100.0) * 0.4 + max(0.0, (sim_wlb - float(matched_emp['WorkLifeBalanceScore'])) / 5.0) * 0.4
                sim_prob = max(0.05, prob_val * (1.0 - reduction))
                sim_risk = assign_risk_category(sim_prob)
                
                m1, m2 = st.columns(2)
                with m1: st.metric("Baseline Probability", f"{prob_val:.1%}", risk_val)
                with m2: st.metric("Projected Probability", f"{sim_prob:.1%}", f"-{(prob_val - sim_prob):.1%} ({sim_risk})")


# ════════════════════════════════════════════════
# VIEW 5: ML RISK PREDICTION (REAL XGBOOST)
# ════════════════════════════════════════════════
elif nav == "ML Risk Prediction":
    st.markdown('<div class="section-heading">Real-Time Machine Learning Attrition Inference</div>', unsafe_allow_html=True)
    st.caption("Execute direct inference on the production-trained XGBoost pipeline.")

    with st.form("ml_prediction_input_form"):
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        with r1_c1: in_age = st.number_input("Age", 18, 65, 32)
        with r1_c2: in_sal = st.number_input("Annual Base Salary ($)", 20000, 250000, 68000, 5000)
        with r1_c3: in_ot = st.number_input("Overtime Hours/Month", 0.0, 80.0, 18.0, 2.0)

        r2_c1, r2_c2, r2_c3 = st.columns(3)
        with r2_c1: in_dept = st.selectbox("Department", ["Sales", "IT", "Finance", "HR", "Marketing", "Support"])
        with r2_c2: in_role = st.selectbox("Job Role", ["Developer", "Engineer", "Sales Executive", "Auditor", "Accountant", "Hr Manager", "Seo Analyst", "Support Engineer"])
        with r2_c3: in_ten = st.number_input("Years at Company", 0.0, 30.0, 3.5, 0.5)

        r3_c1, r3_c2, r3_c3 = st.columns(3)
        with r3_c1: in_wlb = st.slider("Work-Life Balance Rating (1-5)", 1.0, 5.0, 2.5, 0.5)
        with r3_c2: in_perf = st.slider("Performance Rating (1-5)", 1.0, 5.0, 3.5, 0.5)
        with r3_c3: in_sat = st.slider("Customer Satisfaction (1-5)", 1.0, 5.0, 3.0, 0.5)

        submit_inference = st.form_submit_button("Run Model Prediction", type="primary", use_container_width=True)

    if submit_inference:
        if ml_model is not None:
            feat_cols = pd.read_csv(Path(__file__).parent.parent / "data" / "processed" / "feature_matrix.csv").drop(columns=["EmployeeID", "AttritionRisk_Label"]).columns.tolist()
            row = {c: 0.0 for c in feat_cols}
            row["Age"] = float(in_age)
            row["MonthlySalary"] = float(in_sal)
            row["OvertimeHoursPerMonth"] = float(in_ot)
            row["LeavesTaken"] = 8.0
            row["ProjectsHandled"] = 4.0
            row["TrainingHours"] = 15.0
            row["CustomerSatisfaction"] = float(in_sat)
            row["LastPromotionYear"] = 2021.0
            row["YearsAtCompany"] = float(in_ten)
            row["WorkLifeBalanceScore"] = float(in_wlb)
            row["PerformanceRating"] = float(in_perf)
            row["tenure_adjusted_salary"] = float(in_sal) / (float(in_ten) + 1.0)
            row["years_since_last_promotion"] = 3.0
            row["overtime_to_projects_ratio"] = float(in_ot) / 5.0
            row["days_since_last_leave"] = 96.0
            row["engagement_score"] = (float(in_wlb) + float(in_perf) + float(in_sat)) / 3.0

            d_fmt = in_dept.capitalize() if in_dept not in ["IT", "HR"] else in_dept
            d_col = f"Department_{d_fmt}"
            if d_col in row: row[d_col] = 1.0
            elif f"Department_{in_dept.title()}" in row: row[f"Department_{in_dept.title()}"] = 1.0

            r_col = f"JobRole_{in_role.title()}"
            if r_col in row: row[r_col] = 1.0

            X_in = pd.DataFrame([row])[feat_cols]
            pred_probability = float(ml_model.predict_proba(X_in)[0, 1])
            pred_risk_level = assign_risk_category(pred_probability)

            st.markdown("---")
            out_c1, out_c2 = st.columns([1, 2])
            with out_c1:
                st.metric("Predicted Flight Probability", f"{pred_probability:.1%}")
                st.markdown(render_risk_badge(pred_risk_level), unsafe_allow_html=True)
            with out_c2:
                st.markdown(f"**Recommended Action:** {'Schedule immediate 1-on-1 retention review.' if pred_risk_level == 'HIGH' else 'Monitor workload and maintain growth trajectory.'}")


# ════════════════════════════════════════════════
# VIEW 6: WORKFORCE AI ASSISTANT
# ════════════════════════════════════════════════
elif nav == "Workforce AI Assistant":
    st.markdown('<div class="section-heading">Workforce Intelligence Natural Language Assistant</div>', unsafe_allow_html=True)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello. I have access to the workforce analytics dataset and trained XGBoost model. How can I assist with your retention analysis today?"}
        ]

    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_query = st.chat_input("Ask a question regarding workforce turnover, departments, or skill deficits...")
    if user_query:
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"): st.markdown(user_query)

        rag_context = build_system_context(intel_df, org_gap_df, dept_scores_df, user_query)
        active_key = ai_key if 'ai_key' in locals() and ai_key else os.environ.get("GEMINI_API_KEY", "")
        active_provider = ai_provider if 'ai_provider' in locals() and ai_provider else ("Google Gemini" if active_key else "Built-in Synthesizer")

        with st.chat_message("assistant"):
            with st.spinner("Analyzing workforce telemetry..."):
                reply = generate_llm_response(
                    st.session_state.chat_messages, rag_context,
                    api_key=active_key,
                    provider=active_provider
                )
                st.markdown(reply)
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
