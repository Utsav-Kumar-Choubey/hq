"""
audit_adult_income.py
---------------------
DEMO DELIVERABLE #1: audit a real public model on the UCI Adult Income dataset.

We download the dataset, train a RandomForest to predict income >50K,
generate its predictions, then run the FairLens engine to audit it for
gender/race bias. This is one of the two "public model audits" judges require.

Run:  uv run python demo/audit_adult_income.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from engine.audit import run_full_audit

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "adult/adult.data")
COLS = ["age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"]


def main():
    print("Downloading UCI Adult Income dataset...")
    df = pd.read_csv(URL, header=None, names=COLS, skipinitialspace=True,
                     na_values="?").dropna()
    print(f"Loaded {len(df)} rows.")

    df["income_over_50k"] = (df["income"] == ">50K").astype(int)
    df = df.drop(columns=["income", "fnlwgt"])

    # Encode categoricals for the model (keep readable copies of sensitive cols).
    model_df = df.copy()
    encoders = {}
    for col in model_df.select_dtypes(include="object").columns:
        encoders[col] = LabelEncoder()
        model_df[col] = encoders[col].fit_transform(model_df[col])

    X = model_df.drop(columns=["income_over_50k"])
    y = model_df["income_over_50k"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42)
    clf.fit(X_tr, y_tr)
    print(f"Model trained. Test accuracy: {clf.score(X_te, y_te):.1%}")

    # Build the audit table: original readable columns + the model's prediction.
    audit_df = df.loc[X_te.index].copy()
    audit_df["predicted_high_income"] = clf.predict(X_te)
    # age bands make the intersectional view readable
    audit_df["age_band"] = pd.cut(audit_df["age"], [0, 25, 45, 200],
                                  labels=["under25", "25-44", "45plus"])
    audit_df.to_csv("demo/adult_income_predictions.csv", index=False)
    print("Saved predictions -> demo/adult_income_predictions.csv")

    result = run_full_audit(
        audit_df, sensitive=["sex", "race"], pred_col="predicted_high_income",
        outcome_col="income_over_50k", intersectional=True,
        model_name="UCI Adult Income - RandomForest")

    s = result["fairness"]["summary"]
    print(f"\nFairness score: {s['fairness_score']}/100  verdict: {s['verdict']}")
    print(f"Groups tested: {s['groups_tested']}  flagged: {s['groups_flagged']}")
    print("\nBy sex:")
    for r in result["fairness"]["single"]["sex"]:
        print(f"  {r['group']:<10} rate={r['selection_rate']:.0%} "
              f"ratio={r['dp_ratio']} -> {r['status']}")
    print("\nTop recommendations:")
    for r in result["recommendations"][:3]:
        print(f"  [{r['priority']}] {r['text']}")

    with open("demo/adult_income_report.html", "w") as f:
        f.write(result["report"]["html"])
    print("\nSaved report -> demo/adult_income_report.html")


if __name__ == "__main__":
    main()
