"""
test_engine.py
--------------
Standalone test of the audit engine -- NO web server, NO front end.
This is Step 1 of the plan: prove the engine works on its own.

Run:  uv run python test_engine.py
"""

import json
import numpy as np
import pandas as pd

from engine import data_loader, drift as drift_mod
from engine.audit import run_full_audit, advise_metric


def make_biased_loan_data(n=4000, seed=42):
    """Create a synthetic loan dataset with KNOWN gender/age bias baked in,
    so we can confirm the engine actually detects it."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
    age_band = rng.choice(["under25", "25-44", "45plus"], size=n, p=[0.2, 0.5, 0.3])
    income = rng.normal(45000, 15000, n).clip(12000, 150000)
    loan_amount = rng.normal(16000, 6000, n).clip(2000, 60000)
    emp_years = rng.integers(0, 25, n)

    # True creditworthiness depends on real factors only.
    qualified_score = (income / 50000) - (loan_amount / 40000) + emp_years / 30
    repaid = (qualified_score + rng.normal(0, 0.3, n) > 0.4).astype(int)

    # The model UNFAIRLY penalises women and the 45+ band (injected bias).
    bias = np.where(gender == "Female", -0.25, 0) + np.where(age_band == "45plus", -0.2, 0)
    model_score = qualified_score + bias + rng.normal(0, 0.25, n)
    predicted_approval = (model_score > 0.4).astype(int)

    return pd.DataFrame({
        "gender": gender, "age_band": age_band, "income": income.round(0),
        "loan_amount": loan_amount.round(0), "employment_years": emp_years,
        "loan_repaid": repaid, "predicted_approval": predicted_approval,
    })


def main():
    print("=" * 60)
    print("STEP 1 TEST: Audit engine, standalone")
    print("=" * 60)

    df = make_biased_loan_data()
    df.to_csv("demo/sample_loan_predictions.csv", index=False)
    print(f"\nGenerated {len(df)} rows -> demo/sample_loan_predictions.csv")

    # --- data_loader ---
    roles = data_loader.suggest_roles(df)
    print("\n[data_loader] suggested roles:")
    print(json.dumps(roles, indent=2))

    # --- reference data for drift (shift income upward to force drift) ---
    reference = df.copy()
    reference["income"] = reference["income"] * 0.75  # training data had lower income

    # --- full audit ---
    result = run_full_audit(
        df, sensitive=["gender", "age_band"], pred_col="predicted_approval",
        outcome_col="loan_repaid", intersectional=True,
        reference=reference, model_name="Loan Approval v2 (synthetic)")

    s = result["fairness"]["summary"]
    print("\n[fairness] summary:")
    print(json.dumps(s, indent=2))

    print("\n[fairness] intersectional groups (should flag Female/45plus):")
    for r in result["fairness"]["intersectional"]:
        print(f"  {r['group']:<22} size={r['size']:<5} "
              f"rate={r['selection_rate']:.0%} ratio={r['dp_ratio']} -> {r['status']}")

    print("\n[drift] features (income should show drift):")
    for d in result["drift"]["features"]:
        print(f"  {d['feature']:<18} PSI={d['psi']} -> {d['band']}")

    print("\n[explain] sample decisions:")
    for e in result["explanations"]:
        if "error" in e:
            print("  ", e["error"]); continue
        print(f"  {e['decision'].upper()}: {e['plain_language']}")

    print("\n[advisor] lending use case ->")
    print(json.dumps(advise_metric("lending"), indent=2))

    print("\n[recommendations]:")
    for r in result["recommendations"]:
        print(f"  [{r['priority']:<6}] {r['text']}")

    # Save the HTML report
    with open("demo/sample_report.html", "w") as f:
        f.write(result["report"]["html"])
    print("\nSaved report -> demo/sample_report.html")

    # Sanity assertions: the known bias MUST be detected.
    assert s["groups_flagged"] >= 1, "Engine failed to flag injected bias!"
    assert result["drift"]["worst_psi"] > 0.1, "Engine failed to detect drift!"
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED - engine detects the injected bias and drift.")
    print("=" * 60)


if __name__ == "__main__":
    main()
