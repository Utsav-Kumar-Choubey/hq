"""
audit_adult_income.py
---------------------
Public model audit #1: UCI Adult Income (census), predicting income > 50K.

Steps
  1. Download the UCI Adult training data.
  2. Train a scikit-learn pipeline (one-hot encoding + random forest).
     Sensitive attributes (sex, race) are deliberately excluded from the
     model's inputs, as many organisations do, to show that bias persists
     through proxy features.
  3. Save the model and the data files used by the web tool.
  4. Run the FairLens audit and write the sample report.

Outputs (in demo/)
  models/adult_income.joblib      the trained model (upload it in field 2)
  adult_income_test.csv           test records without predictions (field 1)
  adult_income_predictions.csv    the same records with model decisions
  adult_income_reference.csv      training data, for the drift check (field 8)
  adult_income_report.html        the generated compliance report

Run:  python3 demo/audit_adult_income.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from _pipeline import make_pipeline, print_summary, save_model, write_report
from engine.audit import run_full_audit

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLS = ["age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"]
SENSITIVE = ["sex", "race"]


def main():
    print("Downloading the UCI Adult Income dataset...")
    df = pd.read_csv(URL, header=None, names=COLS, skipinitialspace=True,
                     na_values="?").dropna()
    df["income_over_50k"] = (df["income"] == ">50K").astype(int)
    df = df.drop(columns=["income", "fnlwgt"]).reset_index(drop=True)
    df["age_band"] = pd.cut(df["age"], [0, 25, 45, 200],
                            labels=["under25", "25-44", "45plus"]).astype(str)
    print(f"Loaded {len(df):,} records.")

    features = [c for c in df.columns
                if c not in SENSITIVE + ["income_over_50k", "age_band"]]
    train, test = train_test_split(df, test_size=0.3, random_state=42,
                                   stratify=df["income_over_50k"])
    model = make_pipeline(train[features], RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
    model.fit(train[features], train["income_over_50k"])
    print(f"Test accuracy: {model.score(test[features], test['income_over_50k']):.1%}")
    print("Model saved ->", save_model(model, "adult_income"))

    test = test.reset_index(drop=True)
    test.to_csv("demo/adult_income_test.csv", index=False)
    train.to_csv("demo/adult_income_reference.csv", index=False)
    audited = test.copy()
    audited["predicted_high_income"] = model.predict(test[features])
    audited.to_csv("demo/adult_income_predictions.csv", index=False)

    result = run_full_audit(
        audited, sensitive=SENSITIVE, pred_col="predicted_high_income",
        outcome_col="income_over_50k", intersectional=True,
        reference=train, use_case="lending",
        model_name="UCI Adult Income - Random Forest")
    print_summary(result, SENSITIVE)
    print("Report ->", write_report(result, "adult_income"))


if __name__ == "__main__":
    main()
