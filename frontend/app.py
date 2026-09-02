"""
Enterprise HR AI — Workforce Intelligence & Retention Platform
Production-Grade Enterprise SaaS Dashboard
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
    page_title="Workforce Intelligence | HR Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Clean, Grounded Enterprise CSS ──
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  :root {
    --bg-page: #0f1117;
    --bg-card: #181b24;
    --bg-card-hover: #1e222d;
    --border-color: #272c38;
    --border-subtle: #1e2330;
    
    --text-main: #f3f4f6;
    --text-muted: #9ca3af;
    --text-dim: #6b7280;
    
    --accent: #3b82f6;
    --accent-hover: #2563eb;
    
    --risk-high: #ef4444;
    --risk-med: #f59e0b;
    --risk-low: #10b981;
    
    --radius-card: 8px;
    --radius-sm: 6px;
  }

  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: var(--text-main);
  }

  .stApp {
    background-color: var(--bg-page);
  }

  [data-testid="stSidebar"] {
    background-color: #13161f !important;
    border-right: 1px solid var(--border-color) !important;
  }

  /* Header Bar */
  .app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-card);
    margin-bottom: 24px;
  }
  .app-header-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-main);
    letter-spacing: -0.01em;
  }
  .app-header-subtitle {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 2px;
  }

  /* Metric KPI Cards */
  .metrics-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }
  .metric-box {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-card);
    padding: 18px 20px;
  }
  .metric-box-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .metric-box-val {
    font-size: 1.75rem;
    font-weight: 700;
    color: #ffffff;
    margin: 6px 0 2px 0;
    letter-spacing: -0.02em;
  }
  .metric-box-desc {
    font-size: 0.75rem;
    color: var(--text-dim);
  }

  /* Content Cards */
  .content-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-card);
    padding: 20px;
    margin-bottom: 20px;
  }

  .card-heading {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-main);
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
  }

  /* Badges */
  .risk-tag {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  .risk-tag-high { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
  .risk-tag-med { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
  .risk-tag-low { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }

  /* Forms and Controls */
  .stButton button {
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
  }

  [data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-card) !important;
    margin-bottom: 12px !important;
  }

  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── 3. Data Loaders & Real Model Integration ──
DEPT_MAP = {"It": "IT", "Hr": "HR", "it": "IT", "hr": "HR"}

def clean_dept(name):
    return DEPT_MAP.get(str(name), str(name))

@st.cache_data(ttl=300)
def load_datasets():
    base = Path(__file__).parent.parent / "data" / "processed"
    intel = pd.read_csv(base / "employee_intelligence.csv") if (base / "employee_intelligence.csv").exists() else None
    org_gap = pd.read_csv(base / "org_skill_gap.csv") if (base / "org_skill_gap.csv").exists() else None
    dept_s = pd.read_csv(base / "department_composite_scores.csv") if (base / "department_composite_scores.csv").exists() else None

    if intel is not None and "Department" in intel.columns:
        intel["Department"] = intel["Department"].apply(clean_dept)
    if dept_s is not None and "Department" in dept_s.columns:
        dept_s["Department"] = dept_s["Department"].apply(clean_dept)

    return intel, org_gap, dept_s

@st.cache_resource
def load_trained_model():
    p = Path(__file__).parent.parent / "models" / "v1" / "attrition_pipeline.joblib"
    if not p.exists(): p = Path(__file__).parent.parent / "models" / "attrition_pipeline.joblib"
    return joblib.load(p) if p.exists() else None

intel_df, org_gap_df, dept_scores_df = load_datasets()
trained_model = load_trained_model()

# ── 4. Sidebar Navigation & Filter Controls ──
if "high_thresh" not in st.session_state: st.session_state.high_thresh = 65
if "med_thresh" not in st.session_state: st.session_state.med_thresh = 40

with st.sidebar:
    st.markdown("#### Workforce Analytics")
    st.caption("Human Resources Decision Platform")
    st.markdown("---")
    
    PAGES = [
        "Executive Summary",
        "Attrition Risk Analysis",
        "Skills & Competencies",
        "Employee Directory",
        "Retention Simulator",
        "AI Assistant"
    ]
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    
    st.markdown("---")
    with st.expander("Risk Threshold Settings", expanded=False):
        st.session_state.high_thresh = st.slider("High Risk Cutoff (%)", 45, 90, st.session_state.high_thresh, 1)
        st.session_state.med_thresh = st.slider("Medium Risk Cutoff (%)", 15, 60, st.session_state.med_thresh, 1)
        st.caption(f"High: ≥{st.session_state.high_thresh}% | Medium: {st.session_state.med_thresh}%–{st.session_state.high_thresh}% | Low: <{st.session_state.med_thresh}%")
        if st.button("Reset to Defaults", use_container_width=True):
            st.session_state.high_thresh = 65
            st.session_state.med_thresh = 40
            st.rerun()

# Apply thresholds to dataset
h_cutoff = st.session_state.high_thresh / 100.0
m_cutoff = st.session_state.med_thresh / 100.0

def categorize_risk(p):
    if p >= h_cutoff: return "HIGH"
    elif p >= m_cutoff: return "MEDIUM"
    return "LOW"

if intel_df is not None:
    intel_df["Risk_Level"] = intel_df["Attrition_Prob"].apply(categorize_risk)

CHART_PALETTE = {
    "HIGH": "#ef4444",
    "MEDIUM": "#f59e0b",
    "LOW": "#10b981",
    "accent": "#3b82f6"
}

def style_chart(fig):
    fig.update_layout(
        plot_bgcolor="#181b24",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d1d5db", family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor="#272c38", linecolor="#272c38", zerolinecolor="#272c38"),
        yaxis=dict(gridcolor="#272c38", linecolor="#272c38", zerolinecolor="#272c38"),
        margin=dict(t=20, b=20, l=15, r=15),
    )
    return fig

def render_risk_tag(risk: str) -> str:
    css = {"HIGH": "risk-tag-high", "MEDIUM": "risk-tag-med", "LOW": "risk-tag-low"}.get(risk, "risk-tag-low")
    return f'<span class="risk-tag {css}">{risk} RISK</span>'


# ── 5. Main Content Header ──
st.markdown(f"""
<div class="app-header">
  <div>
    <div class="app-header-title">{page}</div>
    <div class="app-header-subtitle">Enterprise workforce intelligence and predictive retention modeling</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
# 1. EXECUTIVE SUMMARY
# ════════════════════════════════════════════════
if page == "Executive Summary":
    if intel_df is not None:
        total = len(intel_df)
        high = (intel_df["Risk_Level"] == "HIGH").sum()
        med = (intel_df["Risk_Level"] == "MEDIUM").sum()
        avg_prob = intel_df["Attrition_Prob"].mean()
        avg_sal = intel_df["MonthlySalary"].mean()

        st.markdown(f"""
        <div class="metrics-row">
          <div class="metric-box">
            <div class="metric-box-title">Total Headcount</div>
            <div class="metric-box-val">{total:,}</div>
            <div class="metric-box-desc">Active employees monitored</div>
          </div>
          <div class="metric-box">
            <div class="metric-box-title">High Flight Risk</div>
            <div class="metric-box-val" style="color:#ef4444;">{high}</div>
            <div class="metric-box-desc">{high/total:.1%} of workforce (≥{st.session_state.high_thresh}%)</div>
          </div>
          <div class="metric-box">
            <div class="metric-box-title">Medium Risk (Watchlist)</div>
            <div class="metric-box-val" style="color:#f59e0b;">{med}</div>
            <div class="metric-box-desc">{f'{med} in {st.session_state.med_thresh}%–{st.session_state.high_thresh}% range' if med > 0 else 'No employees in this band'}</div>
          </div>
          <div class="metric-box">
            <div class="metric-box-title">Average Attrition Risk</div>
            <div class="metric-box-val">{avg_prob:.1%}</div>
            <div class="metric-box-desc">Across all departments</div>
          </div>
          <div class="metric-box">
            <div class="metric-box-title">Average Base Salary</div>
            <div class="metric-box-val">${avg_sal:,.0f}</div>
            <div class="metric-box-desc">Annual compensation average</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1.4, 1])
        with col1:
            st.markdown('<div class="card-heading">High-Risk Concentration by Department</div>', unsafe_allow_html=True)
            dept_summary = intel_df.groupby("Department").agg(
                Total=("EmployeeID", "count"),
                High=("Risk_Level", lambda x: (x == "HIGH").sum()),
            ).reset_index()
            dept_summary["High_Pct"] = (dept_summary["High"] / dept_summary["Total"] * 100).round(1)
            dept_summary = dept_summary.sort_values("High_Pct", ascending=True)

            fig_bar = go.Figure(go.Bar(
                y=dept_summary["Department"],
                x=dept_summary["High_Pct"],
                orientation="h",
                marker=dict(
                    color=dept_summary["High_Pct"],
                    colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1.0, "#ef4444"]],
                    showscale=False
                ),
                text=[f"{v:.1f}%" for v in dept_summary["High_Pct"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>High-Risk Proportion: %{x:.1f}%<br>High-Risk Employees: %{customdata[0]}<br>Total Department Size: %{customdata[1]}<extra></extra>",
                customdata=dept_summary[["High", "Total"]].values,
            ))
            fig_bar.update_layout(height=280, xaxis_title="% of Department at High Risk", yaxis_title="")
            fig_bar = style_chart(fig_bar)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.markdown('<div class="card-heading">Workforce Risk Distribution</div>', unsafe_allow_html=True)
            counts = intel_df["Risk_Level"].value_counts().to_dict()
            labels = ["HIGH", "MEDIUM", "LOW"]
            values = [counts.get(k, 0) for k in labels]
            colors = [CHART_PALETTE[k] for k in labels]

            fig_pie = go.Figure(go.Pie(
                labels=[f"{k} RISK" for k in labels],
                values=values,
                hole=0.6,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Count: %{value} (%{percent})<extra></extra>",
                sort=False
            ))
            fig_pie.update_layout(
                height=280,
                showlegend=True,
                legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center", font=dict(size=11)),
                annotations=[dict(text=f"<b>{total}</b><br>Staff", x=0.5, y=0.5, font_size=15, showarrow=False, font=dict(color="#f3f4f6"))]
            )
            fig_pie = style_chart(fig_pie)
            st.plotly_chart(fig_pie, use_container_width=True)


# ════════════════════════════════════════════════
# 2. ATTRITION RISK ANALYSIS
# ════════════════════════════════════════════════
elif page == "Attrition Risk Analysis":
    st.markdown('<div class="card-heading">Predictive Risk Breakdown & Identified Roster</div>', unsafe_allow_html=True)

    if intel_df is not None:
        c1, c2, c3 = st.columns([1.5, 1.5, 1])
        with c1: f_dept = st.selectbox("Filter by Department", ["All Departments"] + sorted(intel_df["Department"].dropna().unique().tolist()))
        with c2: f_risk = st.selectbox("Filter by Risk Level", ["All Levels", "HIGH", "MEDIUM", "LOW"])
        with c3: limit = st.slider("Display Limit", 10, 100, 25)

        filtered = intel_df.copy()
        if f_dept != "All Departments": filtered = filtered[filtered["Department"] == f_dept]
        if f_risk != "All Levels": filtered = filtered[filtered["Risk_Level"] == f_risk]

        chart_l, chart_r = st.columns(2)
        with chart_l:
            fig_hist = px.histogram(
                filtered, x="Attrition_Prob", nbins=24, title="<b>Attrition Probability Distribution</b>",
                color_discrete_sequence=["#3b82f6"],
                labels={"Attrition_Prob": "Predicted Probability"}
            )
            fig_hist.add_vline(x=h_cutoff, line_dash="dash", line_color="#ef4444", annotation_text=f"High Risk (≥{st.session_state.high_thresh}%)")
            fig_hist.add_vline(x=m_cutoff, line_dash="dash", line_color="#f59e0b", annotation_text=f"Medium Risk (≥{st.session_state.med_thresh}%)")
            fig_hist = style_chart(fig_hist)
            fig_hist.update_layout(height=280)
            st.plotly_chart(fig_hist, use_container_width=True)

        with chart_r:
            fig_box = px.box(
                intel_df, x="Department", y="Attrition_Prob", color="Department",
                title="<b>Risk Dispersion by Department</b>",
                labels={"Attrition_Prob": "Flight Risk Probability"}
            )
            fig_box = style_chart(fig_box)
            fig_box.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown(f"**Identified At-Risk Employees ({len(filtered)} matching)**")
        cols = ["EmployeeID", "Name", "Department", "JobRole", "Attrition_Prob", "Risk_Level", "YearsAtCompany", "Top_Skill_Gap"]
        roster_df = filtered.nlargest(limit, "Attrition_Prob")[cols].copy()
        roster_df["Attrition_Prob"] = roster_df["Attrition_Prob"].apply(lambda p: f"{p:.1%}")
        roster_df["Risk_Level"] = roster_df["Risk_Level"].apply(lambda r: f"🔴 {r}" if r == "HIGH" else ("🟡 " + r if r == "MEDIUM" else "🟢 " + r))

        st.dataframe(
            roster_df, use_container_width=True, hide_index=True,
            column_config={
                "EmployeeID": st.column_config.TextColumn("ID"),
                "Attrition_Prob": st.column_config.TextColumn("Probability"),
                "Risk_Level": st.column_config.TextColumn("Risk Rating"),
                "YearsAtCompany": st.column_config.NumberColumn("Tenure (Years)"),
                "Top_Skill_Gap": st.column_config.TextColumn("Primary Skill Deficit"),
            }
        )


# ════════════════════════════════════════════════
# 3. SKILLS & COMPETENCIES
# ════════════════════════════════════════════════
elif page == "Skills & Competencies":
    st.markdown('<div class="card-heading">Organizational Competency Architecture & Skill Deficits</div>', unsafe_allow_html=True)

    if org_gap_df is not None:
        c1, c2 = st.columns([1.5, 1])
        with c1: f_sev = st.selectbox("Severity Classification", ["All Deficits", "HIGH", "MEDIUM", "LOW"])
        with c2: top_count = st.slider("Number of Skills", 10, 40, 15)

        gaps = org_gap_df if f_sev == "All Deficits" else org_gap_df[org_gap_df["severity"] == f_sev]
        top_skills = gaps.nlargest(top_count, "total_gap_weight")

        fig_skills = px.bar(
            top_skills, x="total_gap_weight", y="skill", orientation="h", color="severity",
            color_discrete_map=CHART_PALETTE, title="<b>Top Deficits by Weighted O*NET Importance</b>",
            labels={"total_gap_weight": "Total Gap Score", "skill": "Skill Name"}
        )
        fig_skills = style_chart(fig_skills)
        fig_skills.update_layout(height=max(360, top_count * 20), yaxis_title="", xaxis_title="Total Gap Score (Headcount × Importance)")
        st.plotly_chart(fig_skills, use_container_width=True)

        st.dataframe(
            top_skills[["skill", "severity", "pct_employees_lacking", "avg_importance", "total_gap_weight", "employees_lacking"]].rename(columns={
                "skill": "Skill", "severity": "Severity", "pct_employees_lacking": "% Employees Lacking",
                "avg_importance": "O*NET Importance", "total_gap_weight": "Weighted Score", "employees_lacking": "Affected Headcount"
            }),
            use_container_width=True, hide_index=True
        )


# ════════════════════════════════════════════════
# 4. EMPLOYEE DIRECTORY
# ════════════════════════════════════════════════
elif page == "Employee Directory":
    st.markdown('<div class="card-heading">Employee Profile & Record Lookup</div>', unsafe_allow_html=True)

    if intel_df is not None:
        query = st.text_input("Search by Employee ID (1–500) or Name", value="1")
        record = None
        if query:
            match_id = intel_df[intel_df["EmployeeID"].astype(str) == str(query.strip())]
            if not match_id.empty: record = match_id.iloc[0]
            else:
                match_name = intel_df[intel_df["Name"].str.contains(query.strip(), case=False, na=False)]
                if not match_name.empty: record = match_name.iloc[0]

        if record is not None:
            prob = float(record["Attrition_Prob"])
            risk = record["Risk_Level"]

            st.markdown(f"""
            <div class="content-card">
              <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                  <div style="font-size:1.35rem; font-weight:700; color:#ffffff;">{record['Name']}</div>
                  <div style="font-size:0.85rem; color:#9ca3af; margin-top:2px;">{record['JobRole']} · {record['Department']} Department</div>
                </div>
                <div>{render_risk_tag(risk)}</div>
              </div>
              <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:12px; margin-top:16px; padding-top:12px; border-top:1px solid #272c38;">
                <div><span style="font-size:0.75rem; color:#6b7280;">Employee ID</span><br><b style="font-size:0.95rem; color:#f3f4f6;">#{record['EmployeeID']}</b></div>
                <div><span style="font-size:0.75rem; color:#6b7280;">Annual Salary</span><br><b style="font-size:0.95rem; color:#f3f4f6;">${record['MonthlySalary']:,.0f}</b></div>
                <div><span style="font-size:0.75rem; color:#6b7280;">Tenure</span><br><b style="font-size:0.95rem; color:#f3f4f6;">{record['YearsAtCompany']:.1f} Years</b></div>
                <div><span style="font-size:0.75rem; color:#6b7280;">Work-Life Balance</span><br><b style="font-size:0.95rem; color:#f3f4f6;">{record['WorkLifeBalanceScore']:.1f} / 5.0</b></div>
                <div><span style="font-size:0.75rem; color:#6b7280;">Primary Skill Deficit</span><br><b style="font-size:0.95rem; color:#f3f4f6;">{record.get('Top_Skill_Gap', 'None')}</b></div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
# 5. RETENTION SIMULATOR (WHAT-IF LAB)
# ════════════════════════════════════════════════
elif page == "Retention Simulator":
    st.markdown('<div class="card-heading">Intervention & Retention What-If Modeling</div>', unsafe_allow_html=True)

    if intel_df is not None:
        sel_emp = st.selectbox("Select Employee to Model", [f"#{r['EmployeeID']} — {r['Name']} ({r['JobRole']})" for _, r in intel_df.iterrows()])
        emp_id = sel_emp.split(" — ")[0].replace("#", "")
        emp_data = intel_df[intel_df["EmployeeID"].astype(str) == str(emp_id)].iloc[0]
        base_p = float(emp_data["Attrition_Prob"])
        base_r = emp_data["Risk_Level"]

        col1, col2 = st.columns([1.2, 1.8])
        with col1:
            st.markdown(f"**Baseline Probability: {base_p:.1%}** ({base_r} Risk)")
            sim_sal = st.slider("Proposed Salary Adjustment (%)", 0, 40, 10, 5)
            sim_wlb = st.slider("Target Work-Life Balance Rating", 1.0, 5.0, min(5.0, float(emp_data["WorkLifeBalanceScore"]) + 1.0), 0.5)

            reduction = (sim_sal / 100.0) * 0.4 + max(0.0, (sim_wlb - float(emp_data["WorkLifeBalanceScore"])) / 5.0) * 0.4
            new_p = max(0.05, base_p * (1.0 - reduction))
            new_r = categorize_risk(new_p)

        with col2:
            st.markdown("**Simulated Retention Impact:**")
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Current Risk", f"{base_p:.1%}", base_r)
            with m2: st.metric("Projected Risk", f"{new_p:.1%}", new_r)
            with m3: st.metric("Estimated Reduction", f"-{(base_p - new_p):.1%}")


# ════════════════════════════════════════════════
# 6. AI ASSISTANT
# ════════════════════════════════════════════════
elif page == "AI Assistant":
    st.markdown('<div class="card-heading">Workforce Analytics Query Assistant</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello. Ask any question regarding workforce retention, department diagnostics, or skill gaps."}
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a question about workforce data...")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"): st.markdown(user_q)

        ctx = build_system_context(intel_df, org_gap_df, dept_scores_df, user_q)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = generate_llm_response(
                    st.session_state.chat_history, ctx,
                    api_key=os.environ.get("GEMINI_API_KEY", ""),
                    provider="Google Gemini" if os.environ.get("GEMINI_API_KEY") else "Built-in Synthesizer"
                )
                st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
