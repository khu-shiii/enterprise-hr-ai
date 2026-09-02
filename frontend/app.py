"""
Enterprise HR AI — Streamlit Dashboard
A polished workforce intelligence dashboard with custom CSS, Plotly charts, and tabbed navigation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import requests
import sys
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.services.llm_chat_service import build_system_context, generate_llm_response

# ── Page config ──
st.set_page_config(
    page_title="Enterprise HR AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark professional theme ──
st.markdown("""
<style>
  /* Google Font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Page background */
  .stApp {
    background-color: #0f1117;
    color: #e8eaf0;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1f2e 0%, #12161f 100%);
    border-right: 1px solid #2d3748;
  }
  [data-testid="stSidebar"] .stMarkdown h2 {
    color: #7c8cf8;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  /* KPI Cards */
  .kpi-card {
    background: linear-gradient(135deg, #1e2533 0%, #252d3d 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
  }
  .kpi-icon { font-size: 2rem; margin-bottom: 8px; }
  .kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 4px 0;
  }
  .kpi-label {
    font-size: 0.8rem;
    color: #8892a4;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
  }
  .kpi-high   { color: #f56565; }
  .kpi-medium { color: #ed8936; }
  .kpi-green  { color: #48bb78; }
  .kpi-blue   { color: #7c8cf8; }
  .kpi-purple { color: #b794f4; }

  /* Risk badges */
  .badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .badge-high   { background: rgba(245,101,101,0.15); color: #f56565; border: 1px solid #f56565; }
  .badge-medium { background: rgba(237,137,54,0.15);  color: #ed8936; border: 1px solid #ed8936; }
  .badge-low    { background: rgba(72,187,120,0.15);  color: #48bb78; border: 1px solid #48bb78; }

  /* Section headers */
  .section-header {
    color: #7c8cf8;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 24px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #2d3748;
  }

  /* Employee profile card */
  .profile-card {
    background: linear-gradient(135deg, #1e2533 0%, #252d3d 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  }
  .profile-name {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8eaf0;
    margin-bottom: 4px;
  }
  .profile-role {
    font-size: 0.95rem;
    color: #7c8cf8;
    margin-bottom: 16px;
    font-weight: 500;
  }
  .profile-stat {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #2d3748;
    font-size: 0.9rem;
  }
  .profile-stat-label { color: #8892a4; }
  .profile-stat-value { color: #e8eaf0; font-weight: 500; }

  /* Tabs */
  [data-testid="stTab"] {
    background: transparent;
    color: #8892a4;
    font-weight: 500;
  }
  [aria-selected="true"][data-testid="stTab"] {
    color: #7c8cf8 !important;
    border-bottom: 2px solid #7c8cf8;
  }

  /* Skill gap severity badge */
  .sev-high   { color: #f56565; font-weight: 600; }
  .sev-medium { color: #ed8936; font-weight: 600; }
  .sev-low    { color: #48bb78; font-weight: 600; }

  /* Hide default Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Metric styling override */
  [data-testid="stMetric"] {
    background: #1e2533;
    border-radius: 8px;
    padding: 12px;
    border: 1px solid #2d3748;
  }
</style>
""", unsafe_allow_html=True)

# ── Configuration ──
API_BASE = "http://localhost:8000"

# Chart theme
CHART_THEME = {
    "plot_bgcolor": "#1e2533",
    "paper_bgcolor": "#1e2533",
    "font_color": "#e8eaf0",
    "grid_color": "#2d3748",
}

PALETTE = {
    "HIGH": "#f56565",
    "MEDIUM": "#ed8936",
    "LOW": "#48bb78",
    "primary": "#7c8cf8",
    "secondary": "#b794f4",
    "accent": "#63b3ed",
}


# ── Data loading helpers ──
@st.cache_data(ttl=300)
def fetch_api(endpoint: str, params: dict = None):
    """Fetch data from the FastAPI backend."""
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Start the API with: uvicorn app.main:app --reload"
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=300)
def load_local_intelligence():
    """Load employee intelligence table directly (fallback if API is down)."""
    path = Path(__file__).parent.parent / "data" / "processed" / "employee_intelligence.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(ttl=300)
def load_org_skill_gap():
    path = Path(__file__).parent.parent / "data" / "processed" / "org_skill_gap.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(ttl=300)
def load_dept_scores():
    path = Path(__file__).parent.parent / "data" / "processed" / "department_composite_scores.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🧠 HR Intelligence")
    st.markdown("---")
    
    nav = st.radio(
        "Navigation",
        ["📊 Overview", "⚠️ Attrition Risk", "🎯 Skill Gaps", "👤 Employee Lookup", "🤖 AI HR Chatbot"],
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    st.markdown("### Data Source")
    
    # Check API status
    health, err = fetch_api("/health")
    if health:
        st.success("✅ API Connected")
        st.caption(f"Model: {health.get('model_name', 'N/A')}")
    else:
        st.warning("⚡ Direct Mode (API offline)")
        st.caption("Using local data files")
    
    st.markdown("---")
    st.markdown("### 🤖 LLM Settings")
    llm_provider = st.selectbox(
        "Provider",
        ["Google Gemini", "OpenAI / Compatible", "Built-in Synthesizer"],
        index=0,
        help="Select your generative AI model provider"
    )
    
    llm_api_key = ""
    llm_model = ""
    llm_base_url = ""
    
    if llm_provider == "Google Gemini":
        env_key = os.environ.get("GEMINI_API_KEY", "")
        llm_api_key = st.text_input(
            "Gemini API Key",
            value=env_key,
            type="password",
            placeholder="AIzaSy...",
            help="Get your free key at aistudio.google.com"
        )
        llm_model = st.selectbox(
            "Model",
            ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
            index=0
        )
    elif llm_provider == "OpenAI / Compatible":
        env_key = os.environ.get("OPENAI_API_KEY", "")
        llm_api_key = st.text_input(
            "API Key",
            value=env_key,
            type="password",
            placeholder="sk-... or gsk_...",
            help="OpenAI / Groq / DeepSeek API key"
        )
        llm_model = st.text_input("Model Name", value="gpt-4o-mini", placeholder="e.g. gpt-4o-mini, llama3-70b-8192")
        llm_base_url = st.text_input(
            "Base URL (Optional)",
            value="",
            placeholder="e.g. https://api.groq.com/openai/v1 or http://localhost:11434/v1"
        )

    st.markdown("---")
    st.markdown("### ⚙️ Attrition Thresholds")
    
    # Initialize session state for default thresholds if not present
    if "high_thresh" not in st.session_state:
        st.session_state.high_thresh = 65
    if "med_thresh" not in st.session_state:
        st.session_state.med_thresh = 40

    high_risk_thresh = st.slider(
        "🔴 High Risk Cutoff (%)",
        min_value=45,
        max_value=90,
        value=st.session_state.high_thresh,
        step=1,
        help="Employees with attrition probability at or above this value are classified as HIGH risk."
    )
    
    med_risk_thresh = st.slider(
        "🟡 Medium Risk Cutoff (%)",
        min_value=15,
        max_value=min(60, high_risk_thresh - 5),
        value=min(st.session_state.med_thresh, high_risk_thresh - 5),
        step=1,
        help="Employees with attrition probability between this value and High Risk Cutoff are classified as MEDIUM risk."
    )
    
    st.session_state.high_thresh = high_risk_thresh
    st.session_state.med_thresh = med_risk_thresh

    st.caption(f"🔴 **HIGH**: ≥ {high_risk_thresh}% | 🟡 **MED**: {med_risk_thresh}–{high_risk_thresh}% | 🟢 **LOW**: < {med_risk_thresh}%")

    if st.button("↺ Reset Thresholds (65% / 40%)", use_container_width=True):
        st.session_state.high_thresh = 65
        st.session_state.med_thresh = 40
        st.rerun()

# ── Load primary data & dynamically re-bucket risk based on user thresholds ──
intel_df = load_local_intelligence()
org_gap_df = load_org_skill_gap()
dept_scores_df = load_dept_scores()

if intel_df is not None:
    # Dynamically apply user's custom threshold values
    h_p = high_risk_thresh / 100.0
    m_p = med_risk_thresh / 100.0
    
    def assign_user_risk(p):
        if p >= h_p:
            return "HIGH"
        elif p >= m_p:
            return "MEDIUM"
        return "LOW"
        
    intel_df["Risk_Level"] = intel_df["Attrition_Prob"].apply(assign_user_risk)


def make_kpi_card(icon, value, label, css_class):
    return f"""
    <div class="kpi-card">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-value {css_class}">{value}</div>
      <div class="kpi-label">{label}</div>
    </div>"""


def risk_badge(risk: str) -> str:
    css = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(risk, "badge-low")
    return f'<span class="badge {css}">{risk}</span>'


def apply_chart_theme(fig):
    fig.update_layout(
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        font=dict(color=CHART_THEME["font_color"], family="Inter"),
        xaxis=dict(gridcolor=CHART_THEME["grid_color"], linecolor=CHART_THEME["grid_color"]),
        yaxis=dict(gridcolor=CHART_THEME["grid_color"], linecolor=CHART_THEME["grid_color"]),
        margin=dict(t=40, b=40, l=20, r=20),
    )
    return fig


# ════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════
if nav == "📊 Overview":
    st.markdown("# 📊 Workforce Intelligence Overview")
    st.markdown("*Real-time insights powered by ML attrition model + O*NET skill gap engine*")
    
    if intel_df is not None:
        total = len(intel_df)
        high = (intel_df["Risk_Level"] == "HIGH").sum()
        medium = (intel_df["Risk_Level"] == "MEDIUM").sum()
        avg_prob = intel_df["Attrition_Prob"].mean()
        avg_gap = intel_df["gaps_count"].mean()
        
        # ── KPI Row ──
        st.markdown('<div class="section-header">KEY METRICS</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        
        with c1:
            st.markdown(make_kpi_card("👥", f"{total:,}", "Total Employees", "kpi-blue"), unsafe_allow_html=True)
        with c2:
            st.markdown(make_kpi_card("🔴", f"{high}", "High Attrition Risk", "kpi-high"), unsafe_allow_html=True)
        with c3:
            st.markdown(make_kpi_card("🟡", f"{medium}", "Medium Risk", "kpi-medium"), unsafe_allow_html=True)
        with c4:
            st.markdown(make_kpi_card("📈", f"{avg_prob:.1%}", "Avg Attrition Prob", "kpi-purple"), unsafe_allow_html=True)
        with c5:
            st.markdown(make_kpi_card("🎯", f"{avg_gap:.1f}", "Avg Skill Gaps", "kpi-green"), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1.2, 1])
        
        with col_left:
            st.markdown('<div class="section-header">ATTRITION RISK BY DEPARTMENT</div>', unsafe_allow_html=True)
            dept_risk = intel_df.groupby("Department").agg(
                Total=("EmployeeID", "count"),
                High=("Risk_Level", lambda x: (x == "HIGH").sum()),
                Avg_Prob=("Attrition_Prob", "mean"),
            ).reset_index()
            dept_risk["High_Pct"] = dept_risk["High"] / dept_risk["Total"] * 100
            dept_risk = dept_risk.sort_values("High_Pct", ascending=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=dept_risk["Department"],
                x=dept_risk["High_Pct"],
                orientation="h",
                marker=dict(
                    color=dept_risk["High_Pct"],
                    colorscale=[[0, "#48bb78"], [0.5, "#ed8936"], [1.0, "#f56565"]],
                    showscale=True,
                    colorbar=dict(title="% High Risk", tickfont=dict(color="#e8eaf0")),
                ),
                text=[f"{v:.1f}%" for v in dept_risk["High_Pct"]],
                textposition="outside",
                textfont=dict(color="#e8eaf0"),
                hovertemplate=(
                    "<b>%{y}</b><br>High Risk: %{x:.1f}%<br>"
                    "Count: %{customdata[0]}<extra></extra>"
                ),
                customdata=dept_risk[["High"]].values,
            ))
            fig.update_layout(
                title="% High-Risk Employees by Department",
                height=350,
                xaxis_title="% High Attrition Risk",
                yaxis_title="",
            )
            fig = apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown('<div class="section-header">RISK DISTRIBUTION</div>', unsafe_allow_html=True)
            risk_counts = intel_df["Risk_Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            
            fig2 = go.Figure(go.Pie(
                labels=risk_counts["Risk Level"],
                values=risk_counts["Count"],
                hole=0.55,
                marker=dict(colors=[
                    PALETTE.get(r, "#7c8cf8") for r in risk_counts["Risk Level"]
                ]),
                textinfo="label+percent",
                textfont=dict(color="#e8eaf0"),
                hovertemplate="<b>%{label}</b><br>%{value} employees (%{percent})<extra></extra>",
            ))
            fig2.update_layout(
                title="Attrition Risk Distribution",
                height=350,
                showlegend=False,
                annotations=[dict(
                    text=f"{total}<br>Employees",
                    x=0.5, y=0.5, font_size=18, showarrow=False,
                    font=dict(color="#e8eaf0", family="Inter"),
                )],
            )
            fig2 = apply_chart_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)
        
        # ── Department engagement scores ──
        if dept_scores_df is not None and not dept_scores_df.empty:
            st.markdown('<div class="section-header">DEPARTMENT ENGAGEMENT SCORES</div>', unsafe_allow_html=True)
            dept_s = dept_scores_df.sort_values("Composite_Engagement_Score", ascending=False)
            
            fig3 = go.Figure(go.Bar(
                x=dept_s["Department"],
                y=dept_s["Composite_Engagement_Score"] * 100,
                marker=dict(
                    color=dept_s["Composite_Engagement_Score"],
                    colorscale=[[0, "#f56565"], [0.5, "#ed8936"], [1, "#48bb78"]],
                ),
                text=[f"{v*100:.1f}%" for v in dept_s["Composite_Engagement_Score"]],
                textposition="outside",
                textfont=dict(color="#e8eaf0"),
                hovertemplate="<b>%{x}</b><br>Engagement: %{y:.1f}%<extra></extra>",
            ))
            fig3.update_layout(title="Composite Engagement Score by Department (%)", height=320)
            fig3 = apply_chart_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)
    
    else:
        st.error("Employee intelligence data not found. Please run notebooks 01–16 first.")


# ════════════════════════════════════════════════
# PAGE 2 — ATTRITION RISK
# ════════════════════════════════════════════════
elif nav == "⚠️ Attrition Risk":
    st.markdown("# ⚠️ Attrition Risk Analysis")
    
    if intel_df is not None:
        # ── Filters ──
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            dept_options = ["All"] + sorted(intel_df["Department"].dropna().unique().tolist())
            dept_filter = st.selectbox("Department", dept_options)
        with col_f2:
            risk_options = ["All", "HIGH", "MEDIUM", "LOW"]
            risk_filter = st.selectbox("Risk Level", risk_options)
        with col_f3:
            n_rows = st.slider("Rows to show", 10, 100, 25)
        
        filtered = intel_df.copy()
        if dept_filter != "All":
            filtered = filtered[filtered["Department"] == dept_filter]
        if risk_filter != "All":
            filtered = filtered[filtered["Risk_Level"] == risk_filter]
        
        st.markdown(f"**Showing {len(filtered)} employees**")
        
        # ── Attrition probability histogram ──
        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.histogram(
                filtered, x="Attrition_Prob", nbins=30,
                color_discrete_sequence=[PALETTE["primary"]],
                title="Attrition Probability Distribution",
                labels={"Attrition_Prob": "Attrition Probability"},
            )
            fig.add_vline(x=h_p, line_dash="dash", line_color=PALETTE["HIGH"],
                         annotation_text=f"HIGH (≥{high_risk_thresh}%)", annotation_font_color=PALETTE["HIGH"])
            fig.add_vline(x=m_p, line_dash="dash", line_color=PALETTE["MEDIUM"],
                         annotation_text=f"MEDIUM (≥{med_risk_thresh}%)", annotation_font_color=PALETTE["MEDIUM"])
            fig = apply_chart_theme(fig)
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_r:
            # Box plot by department
            fig2 = px.box(
                intel_df, x="Department", y="Attrition_Prob",
                color="Department",
                title="Attrition Probability by Department",
                labels={"Attrition_Prob": "Probability"},
            )
            fig2 = apply_chart_theme(fig2)
            fig2.update_layout(height=320, showlegend=False)
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)
        
        col_tbl_hdr, col_tbl_exp = st.columns([4, 1])
        with col_tbl_hdr:
            st.markdown('<div class="section-header">AT-RISK EMPLOYEE ROSTER</div>', unsafe_allow_html=True)
        with col_tbl_exp:
            csv_data = filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export CSV",
                data=csv_data,
                file_name=f"at_risk_employees_{dept_filter.lower()}_{risk_filter.lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        display_cols = ["EmployeeID", "Name", "Department", "JobRole", "Attrition_Prob", "Risk_Level",
                        "YearsAtCompany", "Top_Skill_Gap"]
        display_cols = [c for c in display_cols if c in filtered.columns]
        
        top_risk = filtered.nlargest(n_rows, "Attrition_Prob")[display_cols].copy()
        top_risk["Attrition_Prob"] = top_risk["Attrition_Prob"].apply(lambda x: f"{x:.1%}")
        top_risk["Risk_Level"] = top_risk["Risk_Level"].apply(
            lambda r: f"🔴 {r}" if r == "HIGH" else ("🟡 " + r if r == "MEDIUM" else "🟢 " + r)
        )
        
        st.dataframe(
            top_risk,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Attrition_Prob": st.column_config.TextColumn("Risk Probability"),
                "Risk_Level": st.column_config.TextColumn("Risk Level"),
                "YearsAtCompany": st.column_config.NumberColumn("Tenure (yrs)"),
            },
        )
    else:
        st.error("Data not available. Run notebooks 01–16 first.")


# ════════════════════════════════════════════════
# PAGE 3 — SKILL GAPS
# ════════════════════════════════════════════════
elif nav == "🎯 Skill Gaps":
    st.markdown("# 🎯 Organizational Skill Gap Analysis")
    st.markdown("*Based on O\\*NET essential skills (IM scale) vs employee current skills*")
    
    if org_gap_df is not None:
        # ── Summary KPIs ──
        high_gaps = (org_gap_df["severity"] == "HIGH").sum()
        medium_gaps = (org_gap_df["severity"] == "MEDIUM").sum()
        low_gaps = (org_gap_df["severity"] == "LOW").sum()
        top_skill = org_gap_df.nlargest(1, "total_gap_weight")["skill"].iloc[0]
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(make_kpi_card("🔴", high_gaps, "High Severity Gaps", "kpi-high"), unsafe_allow_html=True)
        with c2:
            st.markdown(make_kpi_card("🟡", medium_gaps, "Medium Severity", "kpi-medium"), unsafe_allow_html=True)
        with c3:
            st.markdown(make_kpi_card("🟢", low_gaps, "Low Severity", "kpi-green"), unsafe_allow_html=True)
        with c4:
            st.markdown(make_kpi_card("⚡", "—", f"Top Gap: {top_skill[:20]}...", "kpi-blue"), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── Severity filter ──
        sev_filter = st.selectbox("Filter by Severity", ["All", "HIGH", "MEDIUM", "LOW"])
        top_n_gaps = st.slider("Show top N gaps", 10, 50, 20)
        
        if sev_filter != "All":
            filtered_gaps = org_gap_df[org_gap_df["severity"] == sev_filter]
        else:
            filtered_gaps = org_gap_df
        
        top_gaps = filtered_gaps.nlargest(top_n_gaps, "total_gap_weight")
        
        # ── Horizontal bar chart ──
        color_map = {"HIGH": PALETTE["HIGH"], "MEDIUM": PALETTE["MEDIUM"], "LOW": PALETTE["LOW"]}
        
        fig = go.Figure()
        for severity in ["HIGH", "MEDIUM", "LOW"]:
            seg = top_gaps[top_gaps["severity"] == severity]
            if not seg.empty:
                fig.add_trace(go.Bar(
                    y=seg["skill"],
                    x=seg["total_gap_weight"],
                    orientation="h",
                    name=severity,
                    marker=dict(color=color_map[severity]),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Gap Weight: %{x:.1f}<br>"
                        "Employees Lacking: %{customdata[0]}<br>"
                        "Avg Importance: %{customdata[1]:.2f}/5.0<extra></extra>"
                    ),
                    customdata=seg[["employees_lacking", "avg_importance"]].values,
                ))
        
        fig.update_layout(
            title=f"Top {top_n_gaps} Skill Gaps by Weighted Importance",
            barmode="stack",
            height=max(400, top_n_gaps * 22),
            xaxis_title="Total Gap Weight (employees × importance)",
            yaxis_title="",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01,
                xanchor="right", x=1,
                font=dict(color="#e8eaf0"),
            ),
        )
        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        # ── Data table ──
        col_sg_hdr, col_sg_exp = st.columns([4, 1])
        with col_sg_hdr:
            st.markdown('<div class="section-header">SKILL GAP DETAIL TABLE</div>', unsafe_allow_html=True)
        with col_sg_exp:
            sg_csv = top_gaps.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Gaps CSV",
                data=sg_csv,
                file_name=f"org_skill_gaps_{sev_filter.lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        display_df = top_gaps[["skill", "severity", "pct_employees_lacking",
                                 "avg_importance", "total_gap_weight", "employees_lacking"]].copy()
        display_df["severity"] = display_df["severity"].apply(
            lambda s: f"🔴 {s}" if s == "HIGH" else ("🟡 " + s if s == "MEDIUM" else "🟢 " + s)
        )
        display_df["pct_employees_lacking"] = display_df["pct_employees_lacking"].apply(lambda x: f"{x:.1f}%")
        display_df["avg_importance"] = display_df["avg_importance"].apply(lambda x: f"{x:.2f}/5.0")
        display_df["total_gap_weight"] = display_df["total_gap_weight"].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(
            display_df.rename(columns={
                "skill": "Skill", "severity": "Severity",
                "pct_employees_lacking": "% Employees Lacking",
                "avg_importance": "Avg Importance",
                "total_gap_weight": "Gap Weight",
                "employees_lacking": "# Employees",
            }),
            use_container_width=True,
            hide_index=True,
        )
        
        # ── Severity threshold note ──
        st.info(
            "**Severity Thresholds (scaled to 500-person org):**  \n"
            "🔴 **HIGH** — ≥30% employees lacking + O*NET importance ≥ 3.5/5  \n"
            "🟡 **MEDIUM** — ≥15% employees lacking + importance ≥ 3.0/5  \n"
            "🟢 **LOW** — everything else"
        )
    else:
        st.error("Skill gap data not found. Run notebooks 13–14 first.")


# ════════════════════════════════════════════════
# PAGE 4 — EMPLOYEE LOOKUP
# ════════════════════════════════════════════════
elif nav == "👤 Employee Lookup":
    st.markdown("# 👤 Employee Intelligence Drill-Down")
    
    if intel_df is not None:
        # ── Search input ──
        col_search, col_btn = st.columns([3, 1])
        with col_search:
            search_input = st.text_input(
                "Search by Employee ID or Name",
                placeholder="Enter Employee ID (e.g., 1, 42) or name...",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
        
        # ── Find employee ──
        found_emp = None
        if search_input:
            # Try ID match first
            id_match = intel_df[intel_df["EmployeeID"].astype(str) == str(search_input.strip())]
            if not id_match.empty:
                found_emp = id_match.iloc[0]
            else:
                # Try name match (case-insensitive)
                name_match = intel_df[intel_df["Name"].str.contains(search_input.strip(), case=False, na=False)]
                if not name_match.empty:
                    found_emp = name_match.iloc[0]
        
        if found_emp is not None:
            emp_id = str(found_emp["EmployeeID"])
            risk = str(found_emp.get("Risk_Level", "LOW"))
            prob = float(found_emp.get("Attrition_Prob", 0))
            
            # ── Profile Card ──
            badge_html = risk_badge(risk)
            st.markdown(f"""
            <div class="profile-card">
              <div class="profile-name">{found_emp.get('Name', 'Unknown')}</div>
              <div class="profile-role">{found_emp.get('JobRole', 'N/A')} · {found_emp.get('Department', 'N/A')}</div>
              <div style="margin-bottom: 16px;">{badge_html}</div>
              <div class="profile-stat">
                <span class="profile-stat-label">Employee ID</span>
                <span class="profile-stat-value">{emp_id}</span>
              </div>
              <div class="profile-stat">
                <span class="profile-stat-label">Age</span>
                <span class="profile-stat-value">{int(found_emp.get('Age', 0))}</span>
              </div>
              <div class="profile-stat">
                <span class="profile-stat-label">Gender</span>
                <span class="profile-stat-value">{found_emp.get('Gender', 'N/A')}</span>
              </div>
              <div class="profile-stat">
                <span class="profile-stat-label">Tenure</span>
                <span class="profile-stat-value">{found_emp.get('YearsAtCompany', 0):.1f} years</span>
              </div>
              <div class="profile-stat">
                <span class="profile-stat-label">Monthly Salary</span>
                <span class="profile-stat-value">${found_emp.get('MonthlySalary', 0):,.0f}</span>
              </div>
              <div class="profile-stat">
                <span class="profile-stat-label">Performance Rating</span>
                <span class="profile-stat-value">{found_emp.get('PerformanceRating', 0):.1f}/5.0</span>
              </div>
              <div class="profile-stat">
                <span class="profile-stat-label">Work-Life Balance</span>
                <span class="profile-stat-value">{found_emp.get('WorkLifeBalanceScore', 0):.1f}/5.0</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            
            # ── Attrition gauge + skill gap ──
            col_gauge, col_skill = st.columns(2)
            
            with col_gauge:
                st.markdown('<div class="section-header">ATTRITION PROBABILITY</div>', unsafe_allow_html=True)
                
                gauge_color = PALETTE["HIGH"] if risk == "HIGH" else (PALETTE["MEDIUM"] if risk == "MEDIUM" else PALETTE["LOW"])
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"color": "#e8eaf0", "size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#8892a4",
                                 "tickfont": {"color": "#8892a4"}},
                        "bar": {"color": gauge_color, "thickness": 0.3},
                        "bgcolor": "#1e2533",
                        "steps": [
                            {"range": [0, med_risk_thresh], "color": "rgba(72,187,120,0.1)"},
                            {"range": [med_risk_thresh, high_risk_thresh], "color": "rgba(237,137,54,0.1)"},
                            {"range": [high_risk_thresh, 100], "color": "rgba(245,101,101,0.1)"},
                        ],
                        "threshold": {
                            "line": {"color": gauge_color, "width": 4},
                            "thickness": 0.75,
                            "value": prob * 100,
                        },
                    },
                    title={"text": f"Attrition Risk: {risk}", "font": {"color": gauge_color, "size": 14}},
                ))
                fig_gauge.update_layout(
                    height=280,
                    paper_bgcolor="#1e2533",
                    font=dict(color="#e8eaf0", family="Inter"),
                    margin=dict(t=30, b=10),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col_skill:
                st.markdown('<div class="section-header">SKILL GAP SUMMARY</div>', unsafe_allow_html=True)
                
                gaps_count = int(found_emp.get("gaps_count", 0))
                gap_score = float(found_emp.get("weighted_gap_score", 0))
                coverage = float(found_emp.get("coverage_pct", 0))
                top_skill_gap = str(found_emp.get("Top_Skill_Gap", "N/A"))
                rec_priority = str(found_emp.get("Rec_Priority", "N/A"))
                
                st.metric("Skill Gaps Count", gaps_count)
                st.metric("Skill Coverage", f"{coverage:.0%}")
                st.metric("Weighted Gap Score", f"{gap_score:.1f}")
                
                priority_color = {"Critical": "🔴", "Growth": "🟡", "Development": "🟢"}.get(rec_priority, "⚪")
                st.markdown(f"**Top Gap:** `{top_skill_gap}`")
                st.markdown(f"**Priority:** {priority_color} {rec_priority}")
            
            # ── Upskilling recommendation ──
            st.markdown('<div class="section-header">UPSKILLING RECOMMENDATION</div>', unsafe_allow_html=True)
            
            rec_tools = str(found_emp.get("Recommended_Tools", "N/A"))
            all_gaps = str(found_emp.get("all_gaps", top_skill_gap))
            
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(f"""
                <div class="profile-card" style="padding: 20px;">
                  <div style="color: #7c8cf8; font-weight: 600; margin-bottom: 12px;">📚 Priority Skills to Develop</div>
                  {"".join(f'<div class="profile-stat"><span class="profile-stat-value">• {s.strip()}</span></div>' for s in all_gaps.split(';') if s.strip())}
                </div>
                """, unsafe_allow_html=True)
            with col_r2:
                st.markdown(f"""
                <div class="profile-card" style="padding: 20px;">
                  <div style="color: #48bb78; font-weight: 600; margin-bottom: 12px;">🛠️ Recommended Tools</div>
                  <div class="profile-stat-value">{rec_tools}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Interactive Retention What-If Simulator ──
            st.markdown('<div class="section-header">🧪 RETENTION WHAT-IF SIMULATOR (INTERVENTION LAB)</div>', unsafe_allow_html=True)
            st.caption("Simulate managerial interventions and calculate real-time estimated reduction in employee flight risk.")

            sim_col1, sim_col2, sim_col3 = st.columns(3)
            with sim_col1:
                sim_raise = st.slider("💰 Proposed Salary Increase (%)", min_value=0, max_value=40, value=10, step=5, key=f"sim_sal_{emp_id}")
            with sim_col2:
                cur_wlb = float(found_emp.get("WorkLifeBalanceScore", 3.0))
                sim_wlb = st.slider("⚖️ Target Work-Life Balance (1-5)", min_value=1.0, max_value=5.0, value=min(5.0, max(1.0, cur_wlb + 1.0)), step=0.5, key=f"sim_wlb_{emp_id}")
            with sim_col3:
                sim_training = st.slider("🎓 Additional Training Hours", min_value=0, max_value=40, value=15, step=5, key=f"sim_trn_{emp_id}")

            # Calculate simulated probability reduction
            salary_factor = (sim_raise / 100.0) * 0.35
            wlb_factor = max(0.0, (sim_wlb - cur_wlb) / 5.0) * 0.40
            training_factor = (sim_training / 40.0) * 0.15
            total_reduction = salary_factor + wlb_factor + training_factor
            
            sim_prob = max(0.05, prob * (1.0 - total_reduction))
            sim_risk = assign_user_risk(sim_prob)
            delta_prob = prob - sim_prob

            sim_card_c1, sim_card_c2, sim_card_c3 = st.columns(3)
            with sim_card_c1:
                st.metric("Baseline Attrition Risk", f"{prob:.1%}", risk)
            with sim_card_c2:
                sim_badge = "🔴 HIGH" if sim_risk == "HIGH" else ("🟡 MEDIUM" if sim_risk == "MEDIUM" else "🟢 LOW")
                st.metric("Simulated New Risk", f"{sim_prob:.1%}", sim_badge)
            with sim_card_c3:
                st.metric("Estimated Risk Reduction", f"🔻 -{delta_prob:.1%}", f"{total_reduction*100:.0f}% Intervention Power")

            # ── Download Employee Dossier ──
            dossier_json = json.dumps({k: v for k, v in found_emp.to_dict().items() if not pd.isna(v)}, indent=2)
            st.download_button(
                label=f"📥 Download {found_emp.get('Name', 'Employee')}'s Dossier (JSON)",
                data=dossier_json,
                file_name=f"employee_{emp_id}_intelligence_dossier.json",
                mime="application/json",
                use_container_width=True
            )
        
        elif search_input:
            st.warning(f"No employee found for '{search_input}'. Try an Employee ID number (1–500) or partial name.")
        
        else:
            # ── Show top 10 highest risk as default ──
            st.markdown('<div class="section-header">HIGHEST RISK EMPLOYEES (TOP 10)</div>', unsafe_allow_html=True)
            top10 = intel_df.nlargest(10, "Attrition_Prob")[
                ["EmployeeID", "Name", "Department", "JobRole", "Attrition_Prob", "Risk_Level", "gaps_count"]
            ].copy()
            top10["Attrition_Prob"] = top10["Attrition_Prob"].apply(lambda x: f"{x:.1%}")
            top10["Risk_Level"] = top10["Risk_Level"].apply(
                lambda r: f"🔴 {r}" if r == "HIGH" else ("🟡 " + r if r == "MEDIUM" else "🟢 " + r)
            )
            st.dataframe(top10, use_container_width=True, hide_index=True)
            st.caption("Search for a specific employee using their ID or name above.")
    
    else:
        st.error("Employee data not found. Run notebooks 01–16 first.")


# ════════════════════════════════════════════════
# PAGE 5 — AI HR CHATBOT / COPILOT
# ════════════════════════════════════════════════
elif nav == "🤖 AI HR Chatbot":
    st.markdown("# 🤖 AI Workforce Intelligence Copilot")
    st.markdown("*Ask questions in natural language about attrition risk, skill gaps, employee profiles, and upskilling strategies.*")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hello! I am your **Enterprise HR AI Copilot**. "
                    "I have real-time access to the **Attrition ML model**, **O*NET skill gap matrix**, and **employee intelligence data**.\n\n"
                    "💡 **Try asking me:**\n"
                    "- *\"Who are the top 5 highest flight-risk employees?\"*\n"
                    "- *\"Which department has the worst attrition risk?\"*\n"
                    "- *\"What are our top critical skill gaps across the company?\"*\n"
                    "- *\"Give me a report on employee 1 (or search by name)\"*\n"
                    "- *\"How many employees are in high attrition risk?\"*\n"
                    "- *\"What upskilling is recommended for Sales or IT?\"*"
                ),
            }
        ]

    # Quick prompt buttons & Clear Chat row
    col_q, col_clear = st.columns([5, 1])
    quick_query = None
    with col_q:
        qc1, qc2, qc3, qc4 = st.columns(4)
        with qc1:
            if st.button("🚨 Top Flight Risks", use_container_width=True):
                quick_query = "Who are the highest flight-risk employees in the company?"
        with qc2:
            if st.button("🏢 Dept Risk Breakdown", use_container_width=True):
                quick_query = "Which department has the worst attrition risk and why?"
        with qc3:
            if st.button("🎯 Top Skill Gaps", use_container_width=True):
                quick_query = "What are the most critical organization-wide skill gaps and how should we close them?"
        with qc4:
            if st.button("📊 Org Health Briefing", use_container_width=True):
                quick_query = "Give me an executive briefing on overall workforce health and 3 actionable recommendations."
    
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True, help="Clear chat history"):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "👋 Chat history cleared. How can I assist you with your workforce intelligence today?"
                }
            ]
            st.rerun()

    # Provider status banner
    status_label = f"🤖 Engine: **{llm_provider}** ({llm_model if llm_model else 'Default'})"
    st.caption(status_label)

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle chat input or button click
    user_input = st.chat_input("Ask any question about employees, risk, skill gaps, or HR retention strategy...") or quick_query
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build dynamic RAG context from dataset for this specific prompt
        system_context = build_system_context(intel_df, org_gap_df, dept_scores_df, user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("🧠 LLM is analyzing workforce telemetry & generating response..."):
                response = generate_llm_response(
                    messages=st.session_state.messages,
                    system_context=system_context,
                    api_key=llm_api_key,
                    provider=llm_provider,
                    model_name=llm_model,
                    base_url=llm_base_url if 'llm_base_url' in locals() else None
                )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

