"""
audit.py
--------
Orchestrator: combines fairness, drift, explanations and reporting into one
call, and provides the metric-selection guidance shown in the UI.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from . import drift as drift_mod
from . import explain as explain_mod
from . import fairness as fairness_mod
from . import report as report_mod
from .data_loader import feature_columns

# Which fairness metric fits which situation. Each entry links to a metric id
# in fairness.METRICS so the advisor's choice drives the audit directly.
METRIC_ADVICE = {
    "hiring": {
        "metric_id": "demographic_parity",
        "metric": "Demographic parity (80% rule)",
        "best_for": "Hiring, screening and shortlisting",
        "why": ("Candidates from every group should be shortlisted at similar "
                "rates. This mirrors the four-fifths rule used in employment law."),
        "watch": "Ignores whether candidates were actually qualified.",
        "backup": "Equal opportunity",
    },
    "lending": {
        "metric_id": "equal_opportunity",
        "metric": "Equal opportunity",
        "best_for": "Lending, admissions and benefits",
        "why": ("Applicants who would have repaid or succeeded must be approved "
                "at similar rates, whatever their group."),
        "watch": "Requires reliable historical outcomes, which can themselves be biased.",
        "backup": "Demographic parity",
    },
    "triage": {
        "metric_id": "equalized_odds",
        "metric": "Equalized odds",
        "best_for": "Medical triage, fraud and risk flags",
        "why": ("Both missed cases and false alarms cause harm, so both error "
                "rates should be similar across groups."),
        "watch": "Strict; usually requires a trade-off with overall accuracy.",
        "backup": "Equal opportunity",
    },
    "scoring": {
        "metric_id": "predictive_parity",
        "metric": "Predictive parity",
        "best_for": "Credit scores, risk scores and rankings",
        "why": ("A positive decision or high score should be equally reliable "
                "for every group."),
        "watch": ("Cannot generally hold together with equal opportunity when "
                  "base rates differ between groups."),
        "backup": "Equalized odds",
    },
}

METRIC_GUIDE = [
    {"question": "Do you have the real outcome (e.g. repaid, hired, diagnosed)?",
     "no": "Use demographic parity: it only needs the model's decisions."},
    {"question": "Is missing a qualified person the main harm?",
     "yes": "Use equal opportunity."},
    {"question": "Are false alarms harmful as well as misses?",
     "yes": "Use equalized odds."},
    {"question": "Will people act on the score as a probability or ranking?",
     "yes": "Use predictive parity."},
    {"note": ("Metrics can conflict mathematically when groups have different "
              "base rates. Choose the one that matches the harm you most need "
              "to prevent and report the others for context.")},
]


def advise_metric(usecase: str) -> dict:
    """Recommended metric for a use case (powers the UI advisor)."""
    advice = METRIC_ADVICE.get(usecase, METRIC_ADVICE["hiring"])
    definition = fairness_mod.METRICS[advice["metric_id"]]
    return {**advice, "use_case": usecase if usecase in METRIC_ADVICE else "hiring",
            "question": definition["question"], "threshold": definition["threshold"]}


def metric_catalogue() -> dict:
    return {"metrics": [{"id": k, **v} for k, v in fairness_mod.METRICS.items()],
            "use_cases": {k: advise_metric(k) for k in METRIC_ADVICE},
            "guide": METRIC_GUIDE}


def run_full_audit(df: pd.DataFrame, sensitive: List[str], pred_col: str,
                   outcome_col: Optional[str] = None,
                   intersectional: bool = True,
                   reference: Optional[pd.DataFrame] = None,
                   model_name: str = "Uploaded model",
                   metric: Optional[str] = None,
                   use_case: Optional[str] = None) -> dict:
    """Run fairness, drift, explanations and the report in one call."""
    if not metric and use_case in METRIC_ADVICE:
        metric = METRIC_ADVICE[use_case]["metric_id"]

    fairness = fairness_mod.audit_fairness(
        df, sensitive, pred_col, outcome_col, intersectional, metric)

    features = feature_columns(df, sensitive + [pred_col, outcome_col])
    if reference is not None:
        drift = drift_mod.audit_drift(reference, df, features)
    else:
        drift = {"features": [], "overall_band": "not_tested",
                 "worst_psi": 0.0, "n_drifted": 0}

    explanations = []
    if features:
        try:
            ex = explain_mod.build_explainer(df, features, pred_col)
            preds = ex["y"]
            for target in (0, 1):
                idx = next((i for i, v in enumerate(preds) if v == target), None)
                if idx is not None:
                    explanations.append(explain_mod.explain_row(ex, idx))
        except Exception as exc:  # explanations must never break the audit
            explanations = [{"error": f"Explanation unavailable: {exc}"}]

    recommendations = report_mod.build_recommendations(fairness, drift)
    dataset_info = {"rows": int(len(df)), "columns": list(df.columns),
                    "sensitive_attributes": sensitive,
                    "prediction_column": pred_col, "outcome_column": outcome_col,
                    "use_case": use_case}
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
