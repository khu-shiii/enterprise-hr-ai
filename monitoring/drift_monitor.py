"""
Data Drift Monitoring — compares production distribution vs training baseline.
Monitors: Age, MonthlySalary, YearsAtCompany, WorkLifeBalanceScore,
          OvertimeHoursPerMonth, AttritionRisk (Attrition_Prob).

Usage:
    python monitoring/drift_monitor.py --prod data/processed/employee_intelligence.csv
"""
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from scipy import stats

# ── Config ──
MONITOR_COLS = [
    "Age", "MonthlySalary", "YearsAtCompany",
    "WorkLifeBalanceScore", "OvertimeHoursPerMonth",
]
DRIFT_THRESHOLD = 0.05   # p-value threshold for KS test (p < 0.05 → drift detected)
F1_FLOOR = 0.60          # retrain if F1 drops below this
RETRAIN_MONTHS = 6       # retrain after this many months of new data

BASELINE_PATH = Path(__file__).parent.parent / "data" / "processed" / "employee_attrition_processed.csv"
REPORT_DIR = Path(__file__).parent.parent / "docs"


def ks_drift(baseline_series: pd.Series, prod_series: pd.Series, col: str) -> dict:
    """Kolmogorov–Smirnov test between baseline and production distributions."""
    baseline_clean = baseline_series.dropna()
    prod_clean = prod_series.dropna()
    stat, pvalue = stats.ks_2samp(baseline_clean, prod_clean)
    drifted = pvalue < DRIFT_THRESHOLD
    return {
        "feature": col,
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(pvalue), 4),
        "drift_detected": drifted,
        "baseline_mean": round(float(baseline_clean.mean()), 3),
        "prod_mean": round(float(prod_clean.mean()), 3),
        "baseline_std": round(float(baseline_clean.std()), 3),
        "prod_std": round(float(prod_clean.std()), 3),
        "mean_shift_pct": round(
            abs(prod_clean.mean() - baseline_clean.mean()) / (baseline_clean.mean() + 1e-9) * 100, 2
        ),
    }


def run_drift_report(prod_path: str) -> dict:
    """Run full drift analysis between baseline and production data."""
    print(f"\n{'='*60}")
    print("ENTERPRISE HR AI — DATA DRIFT MONITOR")
    print(f"{'='*60}")
    print(f"Baseline : {BASELINE_PATH}")
    print(f"Production: {prod_path}")
    print(f"Run time  : {datetime.utcnow().isoformat()}\n")

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline not found: {BASELINE_PATH}")
    if not Path(prod_path).exists():
        raise FileNotFoundError(f"Production data not found: {prod_path}")

    baseline = pd.read_csv(BASELINE_PATH)
    prod = pd.read_csv(prod_path)

    results = []
    for col in MONITOR_COLS:
        if col in baseline.columns and col in prod.columns:
            result = ks_drift(baseline[col], prod[col], col)
            results.append(result)
            status = "⚠  DRIFT" if result["drift_detected"] else "✓  OK"
            print(
                f"{status:<12} {col:<30} "
                f"p={result['p_value']:.4f}  "
                f"baseline_mean={result['baseline_mean']:.2f}  "
                f"prod_mean={result['prod_mean']:.2f}  "
                f"shift={result['mean_shift_pct']:.1f}%"
            )
        else:
            print(f"SKIP     {col:<30} (not in both datasets)")

    # Attrition probability drift (if prod has it)
    prob_col = "Attrition_Prob"
    if prob_col in prod.columns:
        if "AttritionRisk_Label" in baseline.columns:
            result = ks_drift(baseline["AttritionRisk_Label"].astype(float),
                              prod[prob_col], prob_col)
            results.append(result)
            status = "⚠  DRIFT" if result["drift_detected"] else "✓  OK"
            print(f"{status:<12} {prob_col:<30} p={result['p_value']:.4f}")

    n_drifted = sum(r["drift_detected"] for r in results)
    overall_drift = n_drifted > 0

    summary = {
        "run_at": datetime.utcnow().isoformat(),
        "baseline_rows": len(baseline),
        "production_rows": len(prod),
        "features_checked": len(results),
        "features_drifted": n_drifted,
        "overall_drift_detected": overall_drift,
        "drift_threshold": DRIFT_THRESHOLD,
        "feature_results": results,
    }

    print(f"\n{'='*60}")
    print(f"SUMMARY: {n_drifted}/{len(results)} features drifted")
    if overall_drift:
        print("⚠  DRIFT DETECTED — Retraining recommended")
        print("\nRetraining triggers:")
        print(f"  1. Drift p-value < {DRIFT_THRESHOLD} on any monitored feature (TRIGGERED)")
        print(f"  2. F1 drops below {F1_FLOOR}")
        print(f"  3. {RETRAIN_MONTHS} months of new data collected")
    else:
        print("✓  No significant drift. Model is stable.")
    print(f"{'='*60}\n")

    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"drift_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Report saved: {report_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HR AI Data Drift Monitor")
    parser.add_argument(
        "--prod", default="data/processed/employee_intelligence.csv",
        help="Path to production data CSV"
    )
    args = parser.parse_args()
    run_drift_report(args.prod)
