"""
Configuration and constants for the Enterprise HR AI application.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PROC_DIR = DATA_DIR / "processed"
PRED_DIR = DATA_DIR / "predictions"

# Model paths
MODEL_PATH = MODELS_DIR / "attrition_pipeline.joblib"
MODEL_V1_PATH = MODELS_DIR / "v1" / "attrition_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "v1" / "metadata.json"

# Data paths
EMPLOYEE_INTELLIGENCE_PATH = PROC_DIR / "employee_intelligence.csv"
FEATURE_MATRIX_PATH = PROC_DIR / "feature_matrix.csv"
ENGAGEMENT_PATH = PROC_DIR / "engagement_processed.csv"
DEPT_ENGAGEMENT_PATH = PROC_DIR / "department_engagement_summary.csv"
DEPT_SCORES_PATH = PROC_DIR / "department_composite_scores.csv"
ORG_SKILL_GAP_PATH = PROC_DIR / "org_skill_gap.csv"
ROLE_SKILL_GAP_PATH = PROC_DIR / "role_skill_gap.csv"
RECOMMENDATIONS_PATH = PROC_DIR / "upskilling_recommendations.csv"
EMPLOYEE_GAP_SUMMARY_PATH = PROC_DIR / "employee_skill_gap_summary.csv"
ATTRITION_PROCESSED_PATH = PROC_DIR / "employee_attrition_processed.csv"

# Risk thresholds
HIGH_RISK_THRESHOLD = 0.65
MEDIUM_RISK_THRESHOLD = 0.40

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"

# API
API_TITLE = "Enterprise HR AI — Workforce Intelligence API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
## Enterprise HR AI Backend

Provides:
- **Attrition Prediction** — ML-powered employee flight-risk scoring
- **Dashboard Analytics** — Department-level engagement & attrition summaries  
- **Skill Gap Analysis** — O*NET-based skill gap scoring and severity
- **Employee Intelligence** — Per-employee unified profile with risk + skills
"""
