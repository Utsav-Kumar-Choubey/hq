"""
audit.py
--------
Orchestrator: ties every module together into one call. The FastAPI /audit
endpoint and the standalone test script both use run_full_audit().
"""

from __future__ import annotations

import pandas as pd

from . import fairness as fairness_mod
from . import drift as drift_mod
from . import explain as explain_mod
from . import report as report_mod
from .data_loader import feature_columns


# Guidance table: which fairness metric fits which situation.
METRIC_ADVICE = {
    "hiring": {
        "metric": "Demographic parity (80% rule)",
        "why": "Groups should be selected at similar rates when screening.",
        "watch": "Ignores whether candidates were actually qualified.",
        "backup": "Equal opportunity",
    },
    "lending": {
        "metric": "Equal opportunity",
        "why": "Qualified applicants must not be missed across groups.",
        "watch": "Needs reliable ground-truth outcomes.",
        "backup": "Equalized odds",
    },
    "triage": {
        "metric": "Equalized odds",
        "why": "Both false alarms and misses are costly in medical/risk use.",
        "watch": "Hard to satisfy fully; expect trade-offs.",
        "backup": "Calibration",
    },
    "scoring": {
        "metric": "Predictive parity",
        "why": "A given score should mean the same thing for every group.",
        "watch": "Can conflict with equal opportunity.",
        "backup": "Calibration by group",
    },
}


def advise_metric(usecase: str) -> dict:
    """Return the recommended metric for a use case (powers the UI advisor)."""
    return METRIC_ADVICE.get(usecase, METRIC_ADVICE["hiring"])


def run_full_audit(df: pd.DataFrame, sensitive: list[str], pred_col: str,
                   outcome_col: str | None = None,
                   intersectional: bool = True,
                   reference: pd.DataFrame | None = None,
                   model_name: str = "Uploaded model") -> dict:
    """Run fairness + drift + explanations + report in one shot."""
    # 1. Fairness
    fairness = fairness_mod.audit_fairness(
        df, sensitive, pred_col, outcome_col, intersectional)

    # 2. Drift (only if a reference dataset was supplied)
    if reference is not None:
        feats = feature_columns(df, sensitive + [pred_col, outcome_col])
        drift = drift_mod.audit_drift(reference, df, feats)
    else:
        drift = {"features": [], "overall_band": "not_tested",
                 "worst_psi": 0.0, "n_drifted": 0}

    # 3. Explanations for a couple of sample individuals
    feature_cols = feature_columns(df, sensitive + [pred_col, outcome_col])
    explanations = []
    if feature_cols:
        try:
            ex = explain_mod.build_explainer(df, feature_cols, pred_col)
            # explain the first rejected and first approved person we find
            preds = ex["y"]
            for target in (0, 1):
                idx = next((i for i, v in enumerate(preds) if v == target), None)
                if idx is not None:
                    explanations.append(explain_mod.explain_row(ex, idx))
        except Exception as e:
            explanations = [{"error": f"Explanation unavailable: {e}"}]

    # 4. Recommendations + report
    recommendations = report_mod.build_recommendations(fairness, drift)
    dataset_info = {"rows": int(len(df)), "columns": list(df.columns),
                    "sensitive_attributes": sensitive}
    report = report_mod.build_report(
        model_name, dataset_info, fairness, drift, recommendations)

    return {
        "model_name": model_name,
        "dataset": dataset_info,
        "fairness": fairness,
        "drift": drift,
        "explanations": explanations,
        "recommendations": recommendations,
        "report": report,
    }
