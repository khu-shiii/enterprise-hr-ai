"""
Pydantic validation schemas for employee input.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class EmployeeFeatures(BaseModel):
    """Input schema for a single employee attrition prediction."""
    EmployeeID: Optional[str] = Field(None, description="Employee identifier")
    Age: float = Field(..., ge=18, le=70, description="Employee age (18-70)")
    MonthlySalary: float = Field(..., gt=0, description="Monthly salary in USD")
    OvertimeHoursPerMonth: float = Field(..., ge=0, description="Overtime hours per month")
    LeavesTaken: float = Field(..., ge=0, description="Leaves taken in the period")
    ProjectsHandled: float = Field(..., ge=0, description="Number of projects handled")
    TrainingHours: float = Field(..., ge=0, description="Training hours completed")
    CustomerSatisfaction: float = Field(..., ge=1, le=5, description="Customer satisfaction score (1-5)")
    LastPromotionYear: int = Field(..., ge=2000, le=2025, description="Year of last promotion")
    YearsAtCompany: float = Field(..., ge=0, description="Years at company")
    WorkLifeBalanceScore: float = Field(..., ge=1, le=5, description="Work-life balance score (1-5)")
    PerformanceRating: float = Field(..., ge=1, le=5, description="Performance rating (1-5)")

    @field_validator("MonthlySalary")
    @classmethod
    def salary_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("MonthlySalary must be positive")
        return v

    @field_validator("Age")
    @classmethod
    def age_range(cls, v):
        if not (18 <= v <= 70):
            raise ValueError("Age must be between 18 and 70")
        return v


class BatchPredictionRequest(BaseModel):
    """Batch prediction request — list of employee feature sets."""
    employees: list[EmployeeFeatures] = Field(..., min_length=1, max_length=500)
