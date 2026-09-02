"""
TalentPulse AI — Enterprise Workforce Intelligence Platform
Minimalist Executive Edition — Clean, Focused & High-Impact
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
    page_title="TalentPulse AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Minimalist CSS Design System ──
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg-base: #0b0f17;
    --bg-surface: #121826;
    --bg-surface-hover: #182236;
    --border-subtle: rgba(255, 255, 255, 0.07);
    --border-accent: rgba(99, 102, 241, 0.4);
    
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    
    --color-primary: #6366f1;
    --color-success: #10b981;
    --color-warning: #f59e0b;
    --color-danger: #ef4444;
    
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-full: 9999px;
  }

  html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
  }

  .stApp {
    background-color: var(--bg-base);
  }

  [data-testid="stSidebar"] {
    background-color: #0d121e !important;
    border-right: 1px solid var(--border-subtle) !important;
  }

  /* Minimal Top Bar */
  .brand-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    margin-bottom: 20px;
  }
  .brand-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
  }
  .brand-title span { color: var(--color-primary); font-weight: 400; }
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: var(--radius-full);
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--color-success);
    text-transform: uppercase;
  }

  /* Minimal KPI Cards */
  .kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
  }
  .kpi-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 18px 20px;
    transition: transform 0.2s ease, border-color 0.2s ease;
  }
  .kpi-card:hover {
    transform: translateY(-2px);
    border-color: var(--border-accent);
  }
  .kpi-label { font-size: 0.74rem; color: var(--text-secondary); font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
  .kpi-value { font-size: 1.9rem; font-weight: 800; margin: 4px 0 2px 0; color: #ffffff; letter-spacing: -0.02em; }
  .kpi-sub { font-size: 0.72rem; color: var(--text-muted); }

  /* Section Title Bar */
  .section-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #ffffff;
    font-size: 0.92rem;
    font-weight: 700;
    margin: 24px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-subtle);
  }
  .section-bar span { color: var(--color-primary); }

  /* Minimal Risk Badges */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: var(--radius-full);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
  }
  .badge-high { background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }
  .badge-medium { background: rgba(245, 158, 11, 0.15); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.4); }
  .badge-low { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }

  /* Minimal Cards */
  .ui-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 20px;
    margin-bottom: 16px;
  }

  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid var(--border-subtle);
  }
  .stat-box {
    background: rgba(255, 255, 255, 0.02);
    padding: 10px 14px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-subtle);
  }
  .stat-box-lbl { font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; }
  .stat-box-val { font-size: 0.95rem; color: #ffffff; font-weight: 700; margin-top: 2px; }

  [data-testid="stChatMessage"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    margin-bottom: 10px !important;
  }

  [data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px !important;
  }

  .stButton button {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
  }

  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── 3. Data Loaders & Model Loading ──
@st.cache_data(ttl=300)
def load_data():
    base = Path(__file__).parent.parent / "data" / "processed"
    intel = pd.read_csv(base / "employee_intelligence.csv") if (base / "employee_intelligence.csv").exists() else None
    org_gap = pd.read_csv(base / "org_skill_gap.csv") if (base / "org_skill_gap.csv").exists() else None
    dept_s = pd.read_csv(base / "department_composite_scores.csv") if (base / "department_composite_scores.csv").exists() else None
    return intel, org_gap, dept_s

@st.cache_resource
def load_model():
    p = Path(__file__).parent.parent / "models" / "v1" / "attrition_pipeline.joblib"
    if not p.exists(): p = Path(__file__).parent.parent / "models" / "attrition_pipeline.joblib"
    return joblib.load(p) if p.exists() else None

intel_df, org_gap_df, dept_scores_df = load_data()
ml_model = load_model()

# ── 4. Sidebar Controls ──
if "high_thresh" not in st.session_state: st.session_state.high_thresh = 65
if "med_thresh" not in st.session_state: st.session_state.med_thresh = 40

with st.sidebar:
    st.markdown("### ⚡ TalentPulse AI")
    st.caption("Workforce Intelligence Platform")
    st.markdown("---")
    
    NAV_ITEMS = ["📊 Overview", "⚠️ Attrition Risk", "🎯 Skill Gaps", "👤 Employee Lookup", "🔮 ML Prediction", "🤖 AI Copilot"]
    nav = st.radio("Navigation", NAV_ITEMS, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("##### Risk Thresholds")
    st.session_state.high_thresh = st.slider("High Risk (≥ %)", 45, 90, st.session_state.high_thresh, 1)
    st.session_state.med_thresh = st.slider("Medium Risk (≥ %)", 15, 60, st.session_state.med_thresh, 1)
    
    st.caption(f"🔴 ≥{st.session_state.high_thresh}% | 🟡 {st.session_state.med_thresh}–{st.session_state.high_thresh}% | 🟢 <{st.session_state.med_thresh}%")

# Apply dynamic risk cutoffs
h_p = st.session_state.high_thresh / 100.0
m_p = st.session_state.med_thresh / 100.0

def assign_user_risk(p):
    if p >= h_p: return "HIGH"
    elif p >= m_p: return "MEDIUM"
    return "LOW"

if intel_df is not None:
    intel_df["Risk_Level"] = intel_df["Attrition_Prob"].apply(assign_user_risk)

PALETTE = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981", "primary": "#6366f1"}

def apply_chart_theme(fig):
    fig.update_layout(
        plot_bgcolor="rgba(18, 24, 38, 0.6)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc", family="Plus Jakarta Sans, sans-serif"),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", linecolor="rgba(255, 255, 255, 0.06)"),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", linecolor="rgba(255, 255, 255, 0.06)"),
        margin=dict(t=30, b=30, l=15, r=15),
    )
    return fig

def risk_badge(risk: str) -> str:
    css = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(risk, "badge-low")
    icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk, "🟢")
    return f'<span class="badge {css}">{icon} {risk}</span>'


# ── 5. Minimal Top Header ──
st.markdown("""
<div class="brand-bar">
  <div class="brand-title">⚡ TalentPulse <span>AI</span></div>
  <span class="status-pill">● System Active</span>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
# 1. OVERVIEW
# ════════════════════════════════════════════════
if nav == "📊 Overview":
    if intel_df is not None:
        total = len(intel_df)
        high = (intel_df["Risk_Level"] == "HIGH").sum()
        med = (intel_df["Risk_Level"] == "MEDIUM").sum()
        avg_prob = intel_df["Attrition_Prob"].mean()
        avg_sal = intel_df["MonthlySalary"].mean()

        st.markdown(f"""
        <div class="kpi-container">
          <div class="kpi-card"><div class="kpi-label">Employees</div><div class="kpi-value">{total:,}</div><div class="kpi-sub">Total Monitored</div></div>
          <div class="kpi-card"><div class="kpi-label">High Risk Staff</div><div class="kpi-value" style="color:#ef4444;">{high}</div><div class="kpi-sub">{high/total:.1%} of workforce</div></div>
          <div class="kpi-card"><div class="kpi-label">Watchlist</div><div class="kpi-value" style="color:#f59e0b;">{med}</div><div class="kpi-sub">Medium Risk Category</div></div>
          <div class="kpi-card"><div class="kpi-label">Avg Flight Risk</div><div class="kpi-value">{avg_prob:.1%}</div><div class="kpi-sub">Company Baseline</div></div>
          <div class="kpi-card"><div class="kpi-label">Avg Salary</div><div class="kpi-value">${avg_sal:,.0f}</div><div class="kpi-sub">Monthly Compensation</div></div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.markdown('<div class="section-bar">🏢 Department Flight-Risk Rate</div>', unsafe_allow_html=True)
            dept_r = intel_df.groupby("Department").agg(
                Total=("EmployeeID", "count"),
                High=("Risk_Level", lambda x: (x == "HIGH").sum()),
            ).reset_index()
            dept_r["High_Pct"] = (dept_r["High"] / dept_r["Total"] * 100).round(1)
            dept_r = dept_r.sort_values("High_Pct", ascending=True)

            fig = go.Figure(go.Bar(
                y=dept_r["Department"], x=dept_r["High_Pct"], orientation="h",
                marker=dict(color=dept_r["High_Pct"], colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#ef4444"]], showscale=False),
                text=[f"{v:.1f}%" for v in dept_r["High_Pct"]], textposition="outside"
            ))
            fig.update_layout(height=300, xaxis_title="% High Risk", yaxis_title="")
            fig = apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-bar">📊 Risk Breakdown</div>', unsafe_allow_html=True)
            rc = intel_df["Risk_Level"].value_counts().reset_index()
            rc.columns = ["Risk", "Count"]
            fig2 = go.Figure(go.Pie(
                labels=rc["Risk"], values=rc["Count"], hole=0.6,
                marker=dict(colors=[PALETTE.get(r, "#6366f1") for r in rc["Risk"]]),
                textinfo="label+percent"
            ))
            fig2.update_layout(height=300, showlegend=False)
            fig2 = apply_chart_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════
# 2. ATTRITION RISK
# ════════════════════════════════════════════════
elif nav == "⚠️ Attrition Risk":
    st.markdown('<div class="section-bar">⚠️ Predictive Attrition Analytics</div>', unsafe_allow_html=True)

    if intel_df is not None:
        f1, f2, f3 = st.columns([1.5, 1.5, 1])
        with f1: f_dept = st.selectbox("Department", ["All"] + sorted(intel_df["Department"].dropna().unique().tolist()))
        with f2: f_risk = st.selectbox("Risk Level", ["All", "HIGH", "MEDIUM", "LOW"])
        with f3: max_rows = st.slider("Max Rows", 10, 100, 25)

        filtered = intel_df.copy()
        if f_dept != "All": filtered = filtered[filtered["Department"] == f_dept]
        if f_risk != "All": filtered = filtered[filtered["Risk_Level"] == f_risk]

        c_l, c_r = st.columns(2)
        with c_l:
            fig = px.histogram(filtered, x="Attrition_Prob", nbins=25, title="<b>Probability Distribution</b>", color_discrete_sequence=["#6366f1"])
            fig.add_vline(x=h_p, line_dash="dash", line_color="#ef4444", annotation_text=f"HIGH (≥{st.session_state.high_thresh}%)")
            fig = apply_chart_theme(fig)
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

        with c_r:
            fig2 = px.box(intel_df, x="Department", y="Attrition_Prob", color="Department", title="<b>Departmental Risk Box Plot</b>")
            fig2 = apply_chart_theme(fig2)
            fig2.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"**Identified At-Risk Roster ({len(filtered)} Employees)**")
        d_cols = ["EmployeeID", "Name", "Department", "JobRole", "Attrition_Prob", "Risk_Level", "YearsAtCompany", "Top_Skill_Gap"]
        roster = filtered.nlargest(max_rows, "Attrition_Prob")[d_cols].copy()
        roster["Attrition_Prob"] = roster["Attrition_Prob"].apply(lambda p: f"{p:.1%}")
        roster["Risk_Level"] = roster["Risk_Level"].apply(lambda r: f"🔴 {r}" if r == "HIGH" else ("🟡 " + r if r == "MEDIUM" else "🟢 " + r))

        st.dataframe(roster, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════
# 3. SKILL GAPS
# ════════════════════════════════════════════════
elif nav == "🎯 Skill Gaps":
    st.markdown('<div class="section-bar">🎯 Organizational Competency & Skill Gaps</div>', unsafe_allow_html=True)

    if org_gap_df is not None:
        f_sev = st.selectbox("Filter Severity", ["All", "HIGH", "MEDIUM", "LOW"])
        top_n = st.slider("Top N Skills", 10, 40, 15)

        gaps = org_gap_df if f_sev == "All" else org_gap_df[org_gap_df["severity"] == f_sev]
        top_gaps = gaps.nlargest(top_n, "total_gap_weight")

        fig_g = px.bar(
            top_gaps, x="total_gap_weight", y="skill", orientation="h", color="severity",
            color_discrete_map=PALETTE, title="<b>Top Deficits by Weighted Importance</b>"
        )
        fig_g = apply_chart_theme(fig_g)
        fig_g.update_layout(height=max(360, top_n * 20), yaxis_title="", xaxis_title="Total Gap Weight")
        st.plotly_chart(fig_g, use_container_width=True)

        st.dataframe(
            top_gaps[["skill", "severity", "pct_employees_lacking", "avg_importance", "total_gap_weight", "employees_lacking"]].rename(columns={
                "skill": "Skill", "severity": "Severity", "pct_employees_lacking": "% Lacking",
                "avg_importance": "Importance", "total_gap_weight": "Gap Score", "employees_lacking": "Affected Staff"
            }),
            use_container_width=True, hide_index=True
        )


# ════════════════════════════════════════════════
# 4. EMPLOYEE LOOKUP
# ════════════════════════════════════════════════
elif nav == "👤 Employee Lookup":
    st.markdown('<div class="section-bar">👤 Employee Dossier & Intervention Lab</div>', unsafe_allow_html=True)

    if intel_df is not None:
        search_q = st.text_input("Lookup Employee ID (1–500) or Name", value="1")
        found = None
        if search_q:
            id_m = intel_df[intel_df["EmployeeID"].astype(str) == str(search_q.strip())]
            if not id_m.empty: found = id_m.iloc[0]
            else:
                nm_m = intel_df[intel_df["Name"].str.contains(search_q.strip(), case=False, na=False)]
                if not nm_m.empty: found = nm_m.iloc[0]

        if found is not None:
            prob = float(found["Attrition_Prob"])
            risk = found["Risk_Level"]

            st.markdown(f"""
            <div class="ui-card">
              <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                  <div style="font-size:1.4rem; font-weight:800; color:#ffffff;">{found['Name']}</div>
                  <div style="font-size:0.88rem; color:#6366f1; font-weight:600;">{found['JobRole']} · {found['Department']}</div>
                </div>
                <div>{risk_badge(risk)}</div>
              </div>
              <div class="stat-grid">
                <div class="stat-box"><div class="stat-box-lbl">Employee ID</div><div class="stat-box-val">#{found['EmployeeID']}</div></div>
                <div class="stat-box"><div class="stat-box-lbl">Monthly Salary</div><div class="stat-box-val">${found['MonthlySalary']:,.0f}</div></div>
                <div class="stat-box"><div class="stat-box-lbl">Tenure</div><div class="stat-box-val">{found['YearsAtCompany']:.1f} Yrs</div></div>
                <div class="stat-box"><div class="stat-box-lbl">Work-Life Balance</div><div class="stat-box-val">{found['WorkLifeBalanceScore']:.1f}/5.0</div></div>
                <div class="stat-box"><div class="stat-box-lbl">Primary Skill Gap</div><div class="stat-box-val">{found.get('Top_Skill_Gap', 'None')}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            c_g, c_sim = st.columns([1, 1.4])
            with c_g:
                gauge_color = PALETTE.get(risk, "#6366f1")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100, number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [0, st.session_state.med_thresh], "color": "rgba(16, 185, 129, 0.15)"},
                            {"range": [st.session_state.med_thresh, st.session_state.high_thresh], "color": "rgba(245, 158, 11, 0.15)"},
                            {"range": [st.session_state.high_thresh, 100], "color": "rgba(239, 68, 68, 0.15)"},
                        ],
                    },
                    title={"text": f"<b>Flight Risk Meter ({risk})</b>", "font": {"color": gauge_color}}
                ))
                fig_gauge.update_layout(height=240, margin=dict(t=20, b=10))
                fig_gauge = apply_chart_theme(fig_gauge)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with c_sim:
                st.markdown("##### 🧪 Retention What-If Simulator")
                s_sal = st.slider("💰 Proposed Salary Increase (%)", 0, 40, 10, 5, key="min_sim_sal")
                s_wlb = st.slider("⚖️ Target WLB Rating", 1.0, 5.0, 4.0, 0.5, key="min_sim_wlb")
                
                red = (s_sal / 100.0) * 0.4 + max(0.0, (s_wlb - float(found['WorkLifeBalanceScore'])) / 5.0) * 0.4
                sim_prob = max(0.05, prob * (1.0 - red))
                
                m1, m2 = st.columns(2)
                with m1: st.metric("Baseline Risk", f"{prob:.1%}")
                with m2: st.metric("Simulated Risk", f"{sim_prob:.1%}", f"🔻 -{(prob - sim_prob):.1%}")


# ════════════════════════════════════════════════
# 5. ML PREDICTION FORM
# ════════════════════════════════════════════════
elif nav == "🔮 ML Prediction":
    st.markdown('<div class="section-bar">🔮 Real-Time Attrition Model Inference</div>', unsafe_allow_html=True)

    with st.form("min_pred_form"):
        p1, p2, p3 = st.columns(3)
        with p1: in_age = st.number_input("Age", 18, 65, 32)
        with p2: in_sal = st.number_input("Monthly Salary ($)", 20000, 250000, 68000, 5000)
        with p3: in_ot = st.number_input("Overtime Hours/Mo", 0.0, 80.0, 18.0, 2.0)

        p4, p5, p6 = st.columns(3)
        with p4: in_dept = st.selectbox("Department", ["Sales", "It", "Finance", "Hr", "Marketing", "Support"])
        with p5: in_role = st.selectbox("Job Role", ["Developer", "Engineer", "Sales Executive", "Auditor", "Accountant", "Hr Manager", "Seo Analyst", "Support Engineer"])
        with p6: in_ten = st.number_input("Years at Company", 0.0, 30.0, 3.5, 0.5)

        p7, p8, p9 = st.columns(3)
        with p7: in_wlb = st.slider("Work-Life Balance (1-5)", 1.0, 5.0, 2.5, 0.5)
        with p8: in_perf = st.slider("Performance Rating (1-5)", 1.0, 5.0, 3.5, 0.5)
        with p9: in_sat = st.slider("Customer Satisfaction (1-5)", 1.0, 5.0, 3.0, 0.5)

        btn_pred = st.form_submit_button("⚡ Predict Flight Risk", type="primary", use_container_width=True)

    if btn_pred:
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

            d_col = f"Department_{in_dept.title()}"
            if d_col in row: row[d_col] = 1.0
            r_col = f"JobRole_{in_role.title()}"
            if r_col in row: row[r_col] = 1.0

            X_in = pd.DataFrame([row])[feat_cols]
            pred_prob = float(ml_model.predict_proba(X_in)[0, 1])
            pred_r = assign_user_risk(pred_prob)

            st.markdown("---")
            k1, k2 = st.columns([1, 2])
            with k1:
                st.metric("Predicted Flight Probability", f"{pred_prob:.1%}")
                st.markdown(risk_badge(pred_r), unsafe_allow_html=True)
            with k2:
                st.markdown(f"**Action Recommendation:** {'Schedule immediate 1-on-1 retention review.' if pred_r == 'HIGH' else 'Monitor workload and growth trajectory.'}")


# ════════════════════════════════════════════════
# 6. AI COPILOT
# ════════════════════════════════════════════════
elif nav == "🤖 AI Copilot":
    st.markdown('<div class="section-bar">🤖 Generative AI Workforce Copilot</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 **Hello!** I am your TalentPulse AI Copilot. Ask me about workforce risk, department telemetry, or upskilling."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.chat_input("Ask a question about workforce retention or skills...")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"): st.markdown(q)

        ctx = build_system_context(intel_df, org_gap_df, dept_scores_df, q)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                ans = generate_llm_response(st.session_state.messages, ctx, api_key=os.environ.get("GEMINI_API_KEY", ""), provider="Google Gemini" if os.environ.get("GEMINI_API_KEY") else "Built-in Synthesizer")
                st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
