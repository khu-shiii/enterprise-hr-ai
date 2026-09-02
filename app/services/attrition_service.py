"""
Attrition service — loads and queries the employee intelligence table for attrition data.
"""
import pandas as pd
import numpy as np
from functools import lru_cache
from app.utils.config import EMPLOYEE_INTELLIGENCE_PATH, FEATURE_MATRIX_PATH
from app.utils.logger import ml_logger


@lru_cache(maxsize=1)
def _load_intelligence() -> pd.DataFrame:
    if not EMPLOYEE_INTELLIGENCE_PATH.exists():
        raise FileNotFoundError(f"Employee intelligence table not found at {EMPLOYEE_INTELLIGENCE_PATH}")
    df = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH)
    ml_logger.info(f"Loaded employee intelligence: {df.shape}")
    return df


@lru_cache(maxsize=1)
def _load_features() -> pd.DataFrame:
    if not FEATURE_MATRIX_PATH.exists():
        raise FileNotFoundError(f"Feature matrix not found at {FEATURE_MATRIX_PATH}")
    return pd.read_csv(FEATURE_MATRIX_PATH)


def get_attrition_summary() -> dict:
    """Return org-level attrition statistics."""
    df = _load_intelligence()
    risk_counts = df["Risk_Level"].value_counts().to_dict()
    return {
        "total_employees": len(df),
        "high_risk_count": int(risk_counts.get("HIGH", 0)),
        "medium_risk_count": int(risk_counts.get("MEDIUM", 0)),
        "low_risk_count": int(risk_counts.get("LOW", 0)),
        "avg_attrition_probability": round(float(df["Attrition_Prob"].mean()), 4),
        "high_risk_pct": round(float(risk_counts.get("HIGH", 0)) / len(df) * 100, 2),
    }


def get_attrition_by_department() -> list[dict]:
    """Return attrition stats grouped by department."""
    df = _load_intelligence()
    dept_stats = (
        df.groupby("Department")
        .agg(
            total_employees=("EmployeeID", "count"),
            high_risk=("Risk_Level", lambda x: (x == "HIGH").sum()),
            medium_risk=("Risk_Level", lambda x: (x == "MEDIUM").sum()),
            avg_prob=("Attrition_Prob", "mean"),
        )
        .reset_index()
    )
    dept_stats["high_risk_pct"] = (dept_stats["high_risk"] / dept_stats["total_employees"] * 100).round(1)
    dept_stats["avg_prob"] = dept_stats["avg_prob"].round(4)
    return dept_stats.to_dict(orient="records")


def get_employee_attrition(employee_id: str) -> dict | None:
    """Return attrition data for a specific employee."""
    df = _load_intelligence()
    row = df[df["EmployeeID"].astype(str) == str(employee_id)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()
