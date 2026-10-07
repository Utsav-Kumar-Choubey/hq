import numpy as np
import pandas as pd

from engine import data_loader
from engine.audit import advise_metric, run_full_audit
from engine.drift import audit_drift
from engine.fairness import audit_fairness


def test_to_binary_handles_common_encodings():
    assert data_loader.to_binary(pd.Series([0, 1, 1])).tolist() == [0, 1, 1]
    assert data_loader.to_binary(pd.Series([0.1, 0.49, 0.5, 0.9])).tolist() == [0, 0, 1, 1]
    assert data_loader.to_binary(pd.Series(["Yes", "no", "APPROVED"])).tolist() == [1, 0, 1]
    assert data_loader.to_binary(pd.Series([True, False])).tolist() == [1, 0]


def test_suggest_roles_prefers_binary_outcome(loans):
    roles = data_loader.suggest_roles(loans)
    assert roles["suggested_prediction"] == "predicted_approval"
    assert roles["suggested_outcome"] == "loan_repaid"
    assert set(roles["suggested_sensitive"]) == {"gender", "age_band"}


def test_continuous_sensitive_attribute_is_binned():
    df = pd.DataFrame({"age": np.arange(18, 98), "pred": [0, 1] * 40})
    result = audit_fairness(df, ["age"], "pred", None, intersectional=False)
    assert result["summary"]["groups_tested"] <= 4


def test_detects_injected_intersectional_bias(loans):
    result = audit_fairness(loans, ["gender", "age_band"], "predicted_approval",
                            "loan_repaid", intersectional=True)
    worst = min(result["intersectional"], key=lambda r: r["dp_ratio"])
    assert worst["group"] == "Female, 45plus"
    assert worst["status"] == "fail"
    assert result["summary"]["groups_flagged"] >= 1


def test_fair_model_passes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"group": rng.choice(["a", "b"], 4000),
                       "pred": rng.integers(0, 2, 4000)})
    result = audit_fairness(df, ["group"], "pred", None, intersectional=False)
    assert result["summary"]["groups_flagged"] == 0


def test_drift_detects_shift(loans):
    reference = loans.copy()
    reference["income"] = reference["income"] * 0.7
    drift = audit_drift(reference, loans, ["income", "loan_amount"])
    by_feature = {d["feature"]: d for d in drift["features"]}
    assert by_feature["income"]["band"] == "significant"
    assert by_feature["loan_amount"]["band"] == "stable"


def test_full_audit_survives_missing_values(loans):
    loans.loc[:100, "income"] = np.nan
    result = run_full_audit(loans, ["gender"], "predicted_approval", "loan_repaid")
    assert result["explanations"] and "error" not in result["explanations"][0]


def test_full_audit_report_contains_html(loans):
    result = run_full_audit(loans, ["gender", "age_band"], "predicted_approval",
                            "loan_repaid", model_name="Test model")
    assert "<html" in result["report"]["html"].lower()
    assert result["recommendations"]


def test_advisor_covers_all_use_cases():
    for case in ["hiring", "lending", "triage", "scoring"]:
        assert advise_metric(case)["metric"]
