"""
make_sample.py
--------------
Creates an easy, beginner-friendly sample CSV for a live demo. Column names
are chosen to match the default chips on the page (gender, age_band) and to be
self-explanatory. A clear hiring bias against women and the 45+ age band is
baked in, so the audit visibly flags it.

Run:  python3 demo/make_sample.py
Produces: demo/sample_hiring.csv
"""

import csv
import random

random.seed(7)
rows = []
genders = ["Male", "Female"]
age_bands = ["under25", "25-44", "45plus"]
races = ["White", "Black", "Asian", "Hispanic"]

for i in range(600):
    gender = random.choice(genders)
    age_band = random.choice(age_bands)
    race = random.choice(races)
    experience = random.randint(0, 20)
    test_score = random.randint(40, 100)

    # "truly_qualified" depends ONLY on fair factors (experience + score).
    qualified = 1 if (experience / 20 + test_score / 100) > 1.0 else 0

    # The model is UNFAIR: it penalises women and the 45+ age band on top of
    # the fair factors, creating measurable bias for the tool to detect.
    bias = 0.0
    if gender == "Female":
        bias -= 0.30
    if age_band == "45plus":
        bias -= 0.25
    model_signal = (experience / 20 + test_score / 100) + bias + random.uniform(-0.15, 0.15)
    predicted_hire = 1 if model_signal > 1.0 else 0

    rows.append({
        "applicant_id": 1000 + i,
        "gender": gender,
        "age_band": age_band,
        "race": race,
        "years_experience": experience,
        "test_score": test_score,
        "truly_qualified": qualified,     # the real outcome
        "predicted_hire": predicted_hire,  # what the model decided
    })

with open("demo/sample_hiring.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote demo/sample_hiring.csv with {len(rows)} rows.")


# ---------------------------------------------------------------------------
# Drift demo: a "training-time" reference for sample_loan_predictions.csv.
# Applicant incomes were about 25% lower and loans about 10% smaller when the
# model was trained, so the drift check should flag both features.
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402

loans = pd.read_csv("demo/sample_loan_predictions.csv")
reference = loans.drop(columns=["predicted_approval"]).copy()
reference["income"] = (reference["income"] * 0.75).round(0)
reference["loan_amount"] = (reference["loan_amount"] * 0.9).round(0)
reference.to_csv("demo/sample_loan_reference.csv", index=False)
print(f"Wrote demo/sample_loan_reference.csv with {len(reference)} rows.")
