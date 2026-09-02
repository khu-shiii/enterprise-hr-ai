"""
Skill gap service — org-wide and per-employee skill gap data.
"""
import pandas as pd
from functools import lru_cache
from app.utils.config import ORG_SKILL_GAP_PATH, ROLE_SKILL_GAP_PATH, EMPLOYEE_GAP_SUMMARY_PATH
from app.utils.logger import ml_logger


@lru_cache(maxsize=1)
def _load_org_gap() -> pd.DataFrame:
    if not ORG_SKILL_GAP_PATH.exists():
        ml_logger.warning("Org skill gap file not found")
        return pd.DataFrame()
    return pd.read_csv(ORG_SKILL_GAP_PATH)


@lru_cache(maxsize=1)
def _load_emp_gap() -> pd.DataFrame:
    if not EMPLOYEE_GAP_SUMMARY_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(EMPLOYEE_GAP_SUMMARY_PATH)


def get_skill_gaps(severity: str | None = None, top_n: int = 20) -> list[dict]:
    """Return org-wide skill gaps, optionally filtered by severity."""
    df = _load_org_gap()
    if df.empty:
        return []
    if severity:
        df = df[df["severity"].str.upper() == severity.upper()]
    return df.nlargest(top_n, "total_gap_weight").to_dict(orient="records")


def get_skill_gap_summary() -> dict:
    """Return summary counts of skill gap severity."""
    df = _load_org_gap()
    if df.empty:
        return {}
    counts = df["severity"].value_counts().to_dict()
    return {
        "total_skills_with_gaps": len(df),
        "high_severity_count": int(counts.get("HIGH", 0)),
        "medium_severity_count": int(counts.get("MEDIUM", 0)),
        "low_severity_count": int(counts.get("LOW", 0)),
        "top_gap_skill": df.nlargest(1, "total_gap_weight")["skill"].iloc[0] if not df.empty else "N/A",
    }


def get_employee_skill_gap(employee_id: str) -> dict | None:
    """Return skill gap data for a specific employee."""
    df = _load_emp_gap()
    if df.empty:
        return None
    row = df[df["employee_id"].astype(str) == str(employee_id)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()
