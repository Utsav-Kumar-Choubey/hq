"""
audit_german_credit.py
----------------------
Public model audit #2: Statlog German Credit, predicting good vs bad credit.

Steps
  1. Download the Statlog German Credit data and decode the coded columns.
  2. Train a scikit-learn pipeline (one-hot encoding + logistic regression).
  3. Save the model and the data files used by the web tool.
  4. Run the FairLens audit and write the sample report.

Outputs (in demo/)
  models/german_credit.joblib     the trained model (upload it in field 2)
  german_credit_test.csv          test records without predictions (field 1)
  german_credit_predictions.csv   the same records with model decisions
  german_credit_reference.csv     training data, for the drift check (field 8)
  german_credit_report.html       the generated compliance report

Run:  python3 demo/audit_german_credit.py
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from _pipeline import make_pipeline, print_summary, save_model, write_report
from engine.audit import run_full_audit

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "statlog/german/german.data")
COLS = ["status", "duration", "credit_history", "purpose", "credit_amount",
        "savings", "employment", "installment_rate", "personal_status_sex",
        "other_debtors", "residence_since", "property", "age",
        "other_installment", "housing", "existing_credits", "job",
        "num_dependents", "telephone", "foreign_worker", "credit_risk"]
SEX = {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"}
DECODE = {
    "status": {"A11": "below 0 DM", "A12": "0-200 DM", "A13": "200+ DM", "A14": "no account"},
    "credit_history": {"A30": "no credits", "A31": "all paid here", "A32": "paid duly",
                       "A33": "past delays", "A34": "critical account"},
    "savings": {"A61": "below 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
                "A64": "1000+ DM", "A65": "unknown"},
    "employment": {"A71": "unemployed", "A72": "under 1 year", "A73": "1-4 years",
                   "A74": "4-7 years", "A75": "7+ years"},
    "purpose": {"A40": "new car", "A41": "used car", "A42": "furniture", "A43": "radio/TV",
                "A44": "appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
                "A48": "retraining", "A49": "business", "A410": "other"},
    "property": {"A121": "real estate", "A122": "savings or insurance", "A123": "car or other",
                 "A124": "none known"},
    "job": {"A171": "unskilled non-resident", "A172": "unskilled resident",
            "A173": "skilled", "A174": "highly skilled"},
    "telephone": {"A191": "none", "A192": "yes"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "other_installment": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "free"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}
SENSITIVE = ["sex", "age_band"]


def main():
    print("Downloading the Statlog German Credit dataset...")
    df = pd.read_csv(URL, sep=r"\s+", header=None, names=COLS)
    df["good_credit"] = (df["credit_risk"] == 1).astype(int)
    df["sex"] = df["personal_status_sex"].map(SEX).fillna("unknown")
    df["age_band"] = pd.cut(df["age"], [0, 25, 45, 200],
                            labels=["under25", "25-44", "45plus"]).astype(str)
    for col, mapping in DECODE.items():
        df[col] = df[col].map(mapping).fillna(df[col])
    df = df.drop(columns=["credit_risk", "personal_status_sex"])
    print(f"Loaded {len(df):,} records.")

    features = [c for c in df.columns if c not in SENSITIVE + ["good_credit"]]
    train, test = train_test_split(df, test_size=0.4, random_state=42,
                                   stratify=df["good_credit"])
    model = make_pipeline(train[features], LogisticRegression(max_iter=2000))
    model.fit(train[features], train["good_credit"])
    print(f"Test accuracy: {model.score(test[features], test['good_credit']):.1%}")
    print("Model saved ->", save_model(model, "german_credit"))

    test = test.reset_index(drop=True)
    test.to_csv("demo/german_credit_test.csv", index=False)
    train.to_csv("demo/german_credit_reference.csv", index=False)
    audited = test.copy()
    audited["predicted_good_credit"] = model.predict(test[features])
    audited.to_csv("demo/german_credit_predictions.csv", index=False)

    result = run_full_audit(
        audited, sensitive=SENSITIVE, pred_col="predicted_good_credit",
        outcome_col="good_credit", intersectional=True, reference=train,
        use_case="lending", model_name="German Credit - Logistic Regression")
    print_summary(result, SENSITIVE)
    print("Report ->", write_report(result, "german_credit"))


if __name__ == "__main__":
    main()
