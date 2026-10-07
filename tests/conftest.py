import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_biased_loans(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Synthetic lending data with known bias against women aged 45+."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
    age_band = rng.choice(["under25", "25-44", "45plus"], size=n, p=[0.2, 0.5, 0.3])
    income = rng.normal(45000, 15000, n).clip(12000, 150000)
    loan_amount = rng.normal(16000, 6000, n).clip(2000, 60000)
    emp_years = rng.integers(0, 25, n)
    merit = income / 50000 - loan_amount / 40000 + emp_years / 30
    repaid = (merit + rng.normal(0, 0.3, n) > 0.4).astype(int)
    bias = np.where(gender == "Female", -0.25, 0) + np.where(age_band == "45plus", -0.2, 0)
    approved = (merit + bias + rng.normal(0, 0.25, n) > 0.4).astype(int)
    return pd.DataFrame({
        "applicant_id": np.arange(n), "gender": gender, "age_band": age_band,
        "income": income.round(0), "loan_amount": loan_amount.round(0),
        "employment_years": emp_years, "loan_repaid": repaid,
        "predicted_approval": approved,
    })


@pytest.fixture
def loans() -> pd.DataFrame:
    return make_biased_loans()


@pytest.fixture
def root() -> Path:
    return ROOT
