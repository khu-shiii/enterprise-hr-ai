"""
Model Performance Monitor — recomputes classification metrics when true outcomes are available.

Usage:
    python monitoring/model_monitor.py \
        --predictions data/predictions/predictions_20241201.jsonl \
        --actuals data/processed/employee_attrition_processed.csv
"""
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

try:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
except ImportError:
    print("sklearn not available — install requirements.txt")
    raise

REPORT_DIR = Path(__file__).parent.parent / "docs"
F1_FLOOR = 0.60
RECALL_FLOOR = 0.55


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NpEncoder, self).default(obj)


def load_predictions(pred_path: str) -> pd.DataFrame:
    """Load JSONL prediction log into a DataFrame."""
    records = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    df = pd.DataFrame(records)
    if "employee_id" in df.columns:
        df["employee_id"] = df["employee_id"].astype(str)
    return df


def compute_metrics(y_true, y_pred_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_pred_prob >= threshold).astype(int)
    metrics = {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_pred_prob)), 4) if len(set(y_true)) > 1 else None,
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "threshold": float(threshold),
    }
    return metrics


def run_model_monitor(pred_path: str, actuals_path: str) -> dict:
    now_utc = datetime.now(timezone.utc)
    print(f"\n{'='*60}")
    print("ENTERPRISE HR AI -- MODEL PERFORMANCE MONITOR")
    print(f"{'='*60}")
    print(f"Predictions: {pred_path}")
    print(f"Actuals    : {actuals_path}")
    print(f"Run time   : {now_utc.isoformat()}\n")

    preds_df = load_predictions(pred_path)
    actuals_df = pd.read_csv(actuals_path)
    actuals_df["EmployeeID"] = actuals_df["EmployeeID"].astype(str)

    # Join on employee_id
    merged = preds_df.merge(
        actuals_df[["EmployeeID", "AttritionRisk_Label"]],
        left_on="employee_id", right_on="EmployeeID",
        how="inner"
    )
    print(f"Matched predictions with actuals: {len(merged)} records")

    if merged.empty:
        print("[!] No matching records found -- cannot compute metrics")
        return {}

    y_true = merged["AttritionRisk_Label"]
    y_prob = merged["attrition_probability"]

    metrics = compute_metrics(y_true, y_prob)
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']}")

    # Retraining check
    retrain_needed = metrics["f1"] < F1_FLOOR or metrics["recall"] < RECALL_FLOOR
    metrics["retrain_recommended"] = bool(retrain_needed)
    if retrain_needed:
        print(f"\n[!] RETRAINING RECOMMENDED")
        if metrics["f1"] < F1_FLOOR:
            print(f"    F1={metrics['f1']:.4f} < floor={F1_FLOOR}")
        if metrics["recall"] < RECALL_FLOOR:
            print(f"    Recall={metrics['recall']:.4f} < floor={RECALL_FLOOR}")
    else:
        print("\n[OK] Model performance is within acceptable bounds")

    # Save
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"model_monitor_{now_utc.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"run_at": now_utc.isoformat(), "metrics": metrics}, f, indent=2, cls=NpEncoder)
    print(f"\nReport saved: {report_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HR AI Model Performance Monitor")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--actuals", default="data/processed/employee_attrition_processed.csv")
    args = parser.parse_args()
    run_model_monitor(args.predictions, args.actuals)
