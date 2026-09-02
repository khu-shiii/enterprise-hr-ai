"""
Pydantic validation schemas for engagement data.
"""
from pydantic import BaseModel, Field
from typing import Optional


class EngagementRecord(BaseModel):
    """Schema for an employee engagement record from the 5000-person dataset."""
    employee_id: str = Field(..., description="Employee identifier")
    department: str = Field(..., description="Department name")
    job_role: str = Field(..., description="Job role title")
    performance_score: float = Field(..., ge=0, le=100)
    kpi_score: float = Field(..., ge=0, le=100)
    attendance_pct: float = Field(..., ge=0, le=100)
    peer_rating: float = Field(..., ge=1, le=5)
    task_completion_pct: float = Field(..., ge=0, le=100)
    work_hours_logged: float = Field(..., ge=0)
    training_hours: float = Field(..., ge=0)
    promotion_eligibility: Optional[str] = None
