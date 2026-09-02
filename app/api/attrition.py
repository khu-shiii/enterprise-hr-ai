"""
Attrition API endpoints.
POST /predict/attrition — predict attrition for a single employee
"""
from fastapi import APIRouter, HTTPException
from app.validation.employee_schema import EmployeeFeatures, BatchPredictionRequest
from app.ml.predictor import predict_single, predict_batch
from app.services.attrition_service import get_employee_attrition
from app.utils.logger import api_logger

router = APIRouter(prefix="/predict", tags=["Attrition Prediction"])


@router.post("/attrition", summary="Predict attrition risk for one employee")
async def predict_attrition(employee: EmployeeFeatures):
    """
    Predict attrition probability for a single employee.
    
    Returns attrition probability (0–1), risk level (HIGH/MEDIUM/LOW), 
    and a prediction ID for audit logging.
    
    **Bad input (age out of range, negative salary, etc.) returns HTTP 400.**
    """
    try:
        features = employee.model_dump()
        result = predict_single(features)
        api_logger.info(f"Attrition prediction: emp={features.get('EmployeeID')}, risk={result['risk_level']}")
        return result
    except Exception as e:
        api_logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/attrition/batch", summary="Batch predict attrition for multiple employees")
async def predict_attrition_batch(request: BatchPredictionRequest):
    """Predict attrition for multiple employees at once (max 500)."""
    try:
        records = [emp.model_dump() for emp in request.employees]
        results = predict_batch(records)
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        api_logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
