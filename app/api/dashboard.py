"""
Dashboard API endpoints — summary KPIs and analytics.
GET /dashboard/summary
GET /dashboard/attrition-by-department
GET /dashboard/skill-gaps
GET /dashboard/recommendations
"""
from fastapi import APIRouter, Query
from app.services.attrition_service import get_attrition_summary, get_attrition_by_department
from app.services.engagement_service import get_engagement_summary, get_department_scores, get_avg_engagement
from app.services.skill_gap_service import get_skill_gaps, get_skill_gap_summary
from app.services.recommendation_service import get_recommendations
from app.utils.logger import api_logger

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/summary", summary="Top-level workforce KPIs")
async def dashboard_summary():
    """
    Returns key workforce intelligence metrics:
    - Total employees, high/medium/low risk counts
    - Avg attrition probability
    - Avg engagement score
    - Skill gap severity breakdown
    """
    attrition = get_attrition_summary()
    engagement = get_engagement_summary()
    skill_gaps = get_skill_gap_summary()
    avg_engagement = get_avg_engagement()
    
    api_logger.info("Dashboard summary requested")
    return {
        "attrition": attrition,
        "engagement": {**engagement, "avg_score_pct": avg_engagement},
        "skill_gaps": skill_gaps,
    }


@router.get("/attrition-by-department", summary="Attrition risk breakdown by department")
async def attrition_by_department():
    """Returns attrition statistics (count + % high risk + avg probability) for each department."""
    data = get_attrition_by_department()
    api_logger.info(f"Attrition by dept: {len(data)} departments")
    return {"departments": data, "count": len(data)}


@router.get("/skill-gaps", summary="Org-wide skill gaps by severity")
async def skill_gaps(
    severity: str | None = Query(None, description="Filter by severity: HIGH, MEDIUM, LOW"),
    top_n: int = Query(20, ge=1, le=100, description="Number of top gaps to return"),
):
    """Returns organizational skill gaps ranked by weighted importance score."""
    data = get_skill_gaps(severity=severity, top_n=top_n)
    return {"skill_gaps": data, "count": len(data), "severity_filter": severity}


@router.get("/recommendations", summary="Top upskilling recommendations")
async def recommendations(
    priority: str | None = Query(None, description="Filter by priority: Critical, Growth, Development"),
    top_n: int = Query(10, ge=1, le=100),
):
    """Returns top upskilling recommendations across all employees."""
    data = get_recommendations(priority=priority, top_n=top_n)
    return {"recommendations": data, "count": len(data)}


@router.get("/engagement", summary="Department engagement scores")
async def engagement_scores():
    """Returns composite engagement scores per department (from 5000-employee dataset)."""
    data = get_department_scores()
    return {"departments": data, "count": len(data)}
