"""
Skills API endpoints.
GET /employees/{employee_id} — full employee profile
GET /employees/{employee_id}/skills — skill gaps for one employee
"""
from fastapi import APIRouter, HTTPException
from app.services.attrition_service import get_employee_attrition
from app.services.skill_gap_service import get_employee_skill_gap
from app.services.recommendation_service import get_employee_recommendation
from app.utils.logger import api_logger

router = APIRouter(prefix="/employees", tags=["Employee Intelligence"])


@router.get("/{employee_id}", summary="Full employee intelligence profile")
async def get_employee(employee_id: str):
    """
    Returns the complete intelligence profile for a single employee:
    - Demographics and role
    - Attrition probability and risk level
    - Skill gap count and weighted gap score  
    - Top skill gap and upskilling recommendation
    """
    attrition_data = get_employee_attrition(employee_id)
    if attrition_data is None:
        raise HTTPException(status_code=404, detail=f"Employee '{employee_id}' not found")

    skill_gap = get_employee_skill_gap(employee_id)
    recommendation = get_employee_recommendation(employee_id)

    profile = {
        "employee_id": employee_id,
        "profile": {k: v for k, v in attrition_data.items() 
                    if not isinstance(v, float) or not __import__('math').isnan(v)},
        "skill_gap": skill_gap or {"message": "No skill gap data available"},
        "recommendation": recommendation or {"message": "No recommendation available"},
    }

    api_logger.info(f"Employee profile requested: {employee_id}")
    return profile


@router.get("/{employee_id}/skills", summary="Skill gap detail for one employee")
async def get_employee_skills(employee_id: str):
    """Returns skill gap summary for a specific employee."""
    data = get_employee_skill_gap(employee_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No skill gap data for employee '{employee_id}'")
    return data


@router.get("/{employee_id}/recommendation", summary="Upskilling recommendation for one employee")
async def get_employee_rec(employee_id: str):
    """Returns the upskilling recommendation for a specific employee."""
    data = get_employee_recommendation(employee_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No recommendation for employee '{employee_id}'")
    return data
