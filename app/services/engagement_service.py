"""
Engagement service — department-level analytics from the 5000-employee dataset.
"""
import pandas as pd
from functools import lru_cache
from app.utils.config import DEPT_ENGAGEMENT_PATH, DEPT_SCORES_PATH
from app.utils.logger import ml_logger


@lru_cache(maxsize=1)
def _load_dept_engagement() -> pd.DataFrame:
    if DEPT_ENGAGEMENT_PATH.exists():
        return pd.read_csv(DEPT_ENGAGEMENT_PATH)
    ml_logger.warning("Department engagement summary not found")
    return pd.DataFrame()


@lru_cache(maxsize=1)
def _load_dept_scores() -> pd.DataFrame:
    if DEPT_SCORES_PATH.exists():
        return pd.read_csv(DEPT_SCORES_PATH)
    return pd.DataFrame()


def get_engagement_summary() -> dict:
    """Return org-level engagement summary."""
    scores = _load_dept_scores()
    if scores.empty:
        return {"error": "Engagement data not available"}
    
    return {
        "departments_analyzed": len(scores),
        "avg_composite_score": round(float(scores["Composite_Engagement_Score"].mean()), 4),
        "top_department": scores.nlargest(1, "Composite_Engagement_Score")["Department"].iloc[0],
        "bottom_department": scores.nsmallest(1, "Composite_Engagement_Score")["Department"].iloc[0],
    }


def get_department_scores() -> list[dict]:
    """Return composite engagement score per department."""
    scores = _load_dept_scores()
    if scores.empty:
        return []
    return scores.sort_values("Composite_Engagement_Score", ascending=False).to_dict(orient="records")


def get_avg_engagement() -> float:
    """Return overall average engagement score."""
    scores = _load_dept_scores()
    if scores.empty:
        return 0.0
    return round(float(scores["Composite_Engagement_Score"].mean()) * 100, 1)
