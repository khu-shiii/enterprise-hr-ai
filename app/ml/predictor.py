"""
ML predictor — wraps the pipeline to return structured attrition predictions.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import json
import uuid
from pathlib import Path

from app.ml.model_loader import get_model
from app.utils.config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, PRED_DIR
from app.utils.logger import ml_logger


def _assign_risk_level(prob: float) -> str:
    if prob >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    elif prob >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def predict_single(features: dict) -> dict:
    """
    Predict attrition probability for a single employee.
    features: dict of feature_name -> value matching the training feature matrix columns.
    Returns: {employee_id, attrition_probability, risk_level, predicted_at}
    """
    model = get_model()
    emp_id = features.get("EmployeeID", features.get("employee_id", "unknown"))

    # Build DataFrame row (drop id cols)
    id_cols = {"EmployeeID", "employee_id", "AttritionRisk_Label"}
    feat_dict = {k: v for k, v in features.items() if k not in id_cols}
    X = pd.DataFrame([feat_dict])

    # Align to training columns
    try:
        X = X.astype(float)
    except Exception as e:
        ml_logger.warning(f"Feature coercion warning: {e}")
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    prob = float(model.predict_proba(X)[0, 1])
    risk = _assign_risk_level(prob)

    result = {
        "employee_id": str(emp_id),
        "attrition_probability": round(prob, 4),
        "risk_level": risk,
        "predicted_at": datetime.utcnow().isoformat(),
        "prediction_id": str(uuid.uuid4()),
    }

    _log_prediction(result, features)
    ml_logger.info(f"Predicted: emp={emp_id}, prob={prob:.4f}, risk={risk}")
    return result


def predict_batch(records: list[dict]) -> list[dict]:
    """Predict for a list of employee feature dicts."""
    return [predict_single(r) for r in records]


def _log_prediction(result: dict, features: dict) -> None:
    """Append prediction to the prediction log file."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    log_path = PRED_DIR / f"predictions_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    log_entry = {**result, "input_features": features}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
