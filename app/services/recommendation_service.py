"""
Recommendation service — upskilling recommendations per employee.
"""
import pandas as pd
from functools import lru_cache
from app.utils.config import RECOMMENDATIONS_PATH
from app.utils.logger import ml_logger


@lru_cache(maxsize=1)
def _load_recs() -> pd.DataFrame:
    if not RECOMMENDATIONS_PATH.exists():
        ml_logger.warning("Recommendations file not found")
        return pd.DataFrame()
    return pd.read_csv(RECOMMENDATIONS_PATH)


def get_recommendations(employee_id: str | None = None, priority: str | None = None,
                        top_n: int = 10) -> list[dict]:
    """Return upskilling recommendations, optionally filtered."""
    df = _load_recs()
    if df.empty:
        return []
    if employee_id:
        df = df[df["employee_id"].astype(str) == str(employee_id)]
    if priority:
        df = df[df["priority"].str.lower() == priority.lower()]
    return df.head(top_n).to_dict(orient="records")


def get_employee_recommendation(employee_id: str) -> dict | None:
    """Return recommendation for a specific employee."""
    results = get_recommendations(employee_id=employee_id)
    return results[0] if results else None
