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


# ---- metric suite ---------------------------------------------------------
from engine.fairness import METRICS, wilson_interval  # noqa: E402


def test_all_metrics_computed_per_group(loans):
    result = audit_fairness(loans, ["gender"], "predicted_approval", "loan_repaid",
                            intersectional=False, metric="equal_opportunity")
    assert result["metric"]["id"] == "equal_opportunity"
    for row in result["single"]["gender"]:
        assert set(row["metrics"]) == set(METRICS)
        assert row["true_positive_rate"] is not None
        assert row["false_positive_rate"] is not None
        assert row["primary_value"] == row["eo_ratio"]


def test_metric_falls_back_without_outcome(loans):
    result = audit_fairness(loans, ["gender"], "predicted_approval", None,
                            intersectional=False, metric="equalized_odds")
    assert result["metric"]["id"] == "demographic_parity"
    assert "requires the actual outcome" in result["metric"]["note"]


def test_equalized_odds_flags_error_rate_gap(loans):
    result = audit_fairness(loans, ["gender"], "predicted_approval", "loan_repaid",
                            intersectional=False, metric="equalized_odds")
    female = next(r for r in result["single"]["gender"] if r["group"] == "Female")
    assert female["odds_gap"] > 0.10 and female["status"] == "fail"


def test_intersectional_covers_every_combination():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({a: rng.choice(["x", "y"], 2000) for a in ["a", "b", "c"]})
    df["pred"] = rng.integers(0, 2, 2000)
    result = audit_fairness(df, ["a", "b", "c"], "pred", None, intersectional=True)
    assert {r["grouping"] for r in result["intersectional"]} == {
        "a x b", "a x c", "b x c", "a x b x c"}


def test_small_failing_groups_marked_for_review():
    df = pd.DataFrame({"g": ["big"] * 500 + ["tiny"] * 20,
                       "pred": [1] * 400 + [0] * 100 + [0] * 18 + [1] * 2})
    result = audit_fairness(df, ["g"], "pred", None, intersectional=False)
    tiny = next(r for r in result["single"]["g"] if r["group"] == "tiny")
    assert tiny["status"] == "review" and tiny["confidence"] == "low"


def test_wilson_interval_bounds():
    low, high = wilson_interval(50, 100)
    assert 0.39 < low < 0.5 < high < 0.61
    assert wilson_interval(0, 0) == [0.0, 0.0]


def test_use_case_selects_metric(loans):
    result = run_full_audit(loans, ["gender"], "predicted_approval", "loan_repaid",
                            use_case="triage")
    assert result["fairness"]["metric"]["id"] == "equalized_odds"


def test_failing_audit_never_scores_as_healthy(loans):
    result = audit_fairness(loans, ["gender", "age_band"], "predicted_approval",
                            "loan_repaid", intersectional=True)
    assert result["summary"]["groups_flagged"] >= 1
    assert result["summary"]["fairness_score"] < 70
