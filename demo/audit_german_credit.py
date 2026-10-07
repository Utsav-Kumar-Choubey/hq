"""
audit_german_credit.py
----------------------
DEMO DELIVERABLE #2: audit a real public model on the Statlog German Credit
dataset (credit risk / lending). We train a logistic-regression classifier to
predict good vs bad credit, then audit it for age and sex-related bias.

Run:  uv run python demo/audit_german_credit.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from engine.audit import run_full_audit

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "statlog/german/german.data")

# The german.data file is space-separated with coded attributes (A11, A34...).
COLS = ["status", "duration", "credit_history", "purpose", "credit_amount",
        "savings", "employment", "installment_rate", "personal_status_sex",
        "other_debtors", "residence_since", "property", "age",
        "other_installment", "housing", "existing_credits", "job",
        "num_dependents", "telephone", "foreign_worker", "credit_risk"]

# Decode the personal_status_sex codes into a readable sex column.
SEX_MAP = {"A91": "male", "A92": "female", "A93": "male",
           "A94": "male", "A95": "female"}


def main():
    print("Downloading Statlog German Credit dataset...")
    df = pd.read_csv(URL, sep=r"\s+", header=None, names=COLS)
    print(f"Loaded {len(df)} rows.")

    # credit_risk: 1 = good, 2 = bad -> turn into "good credit" (1/0)
    df["good_credit"] = (df["credit_risk"] == 1).astype(int)
    df["sex"] = df["personal_status_sex"].map(SEX_MAP).fillna("unknown")
    df["age_band"] = pd.cut(df["age"], [0, 25, 45, 200],
                            labels=["under25", "25-44", "45plus"])
    df = df.drop(columns=["credit_risk", "personal_status_sex"])

    # Encode for the model.
    model_df = pd.get_dummies(df.drop(columns=["good_credit"]), drop_first=True)
    X = StandardScaler().fit_transform(model_df.astype(float))
    y = df["good_credit"]
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    print(f"Model trained. Test accuracy: {clf.score(X_te, y_te):.1%}")

    audit_df = df.loc[idx_te].copy()
    audit_df["predicted_good_credit"] = clf.predict(X_te)
    audit_df.to_csv("demo/german_credit_predictions.csv", index=False)
    print("Saved predictions -> demo/german_credit_predictions.csv")

    result = run_full_audit(
        audit_df, sensitive=["sex", "age_band"],
        pred_col="predicted_good_credit", outcome_col="good_credit",
        intersectional=True, model_name="German Credit - LogisticRegression")

    s = result["fairness"]["summary"]
    print(f"\nFairness score: {s['fairness_score']}/100  verdict: {s['verdict']}")
    print("By sex:")
    for r in result["fairness"]["single"]["sex"]:
        print(f"  {r['group']:<10} rate={r['selection_rate']:.0%} "
              f"ratio={r['dp_ratio']} -> {r['status']}")
    print("By age band:")
    for r in result["fairness"]["single"]["age_band"]:
        print(f"  {r['group']:<10} rate={r['selection_rate']:.0%} "
              f"ratio={r['dp_ratio']} -> {r['status']}")
    print("\nTop recommendations:")
    for r in result["recommendations"][:3]:
        print(f"  [{r['priority']}] {r['text']}")

    with open("demo/german_credit_report.html", "w") as f:
        f.write(result["report"]["html"])
    print("\nSaved report -> demo/german_credit_report.html")


if __name__ == "__main__":
    main()
