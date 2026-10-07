"""
data_loader.py
--------------
Step 1 of the audit engine: read a CSV and understand its columns.

This module does NOT compute any fairness math. Its only job is to:
  - load a CSV into a pandas DataFrame safely
  - describe each column (name, type, how many unique values, example values)
  - guess which columns are "sensitive attributes" (gender, race, age...),
    which could be the "true outcome", and which could be the "prediction"

The front end calls this (via /upload) to auto-fill the dropdowns.
"""

from __future__ import annotations

import pandas as pd


# Words we commonly see in column names for each role. Used only for *guessing*
# sensible defaults in the dropdowns; the user can always override.
SENSITIVE_HINTS = [
    "gender", "sex", "race", "ethnic", "age", "disab",
    "religion", "nationality", "marital", "citizen",
]
OUTCOME_HINTS = ["outcome", "label", "target", "approved", "repaid",
                 "hired", "default", "income", "churn", "y_true", "actual"]
PREDICTION_HINTS = ["pred", "predicted", "score", "proba", "y_pred", "decision"]


def load_csv(path_or_buffer) -> pd.DataFrame:
    """Read a CSV into a DataFrame. Accepts a file path or a file-like object."""
    df = pd.read_csv(path_or_buffer)
    # Strip whitespace from column names so "age " and "age" don't differ.
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError("The uploaded CSV has no rows.")
    return df


def _looks_binary(series: pd.Series) -> bool:
    """True if the column has exactly two unique non-null values (0/1, yes/no)."""
    return series.dropna().nunique() == 2


def describe_columns(df: pd.DataFrame) -> list[dict]:
    """Return a plain-language description of every column for the UI."""
    out = []
    for col in df.columns:
        s = df[col]
        n_unique = int(s.dropna().nunique())
        is_numeric = pd.api.types.is_numeric_dtype(s)
        out.append({
            "name": col,
            "dtype": "number" if is_numeric else "category",
            "unique_values": n_unique,
            "is_binary": _looks_binary(s),
            "examples": [str(v) for v in s.dropna().unique()[:4]],
        })
    return out


def _match(col: str, hints: list[str]) -> bool:
    low = col.lower()
    return any(h in low for h in hints)


def suggest_roles(df: pd.DataFrame) -> dict:
    """
    Guess good default picks for the dropdowns. Everything here is a *hint*;
    the user confirms the real choice in the form.
    """
    cols = list(df.columns)
    sensitive = [c for c in cols if _match(c, SENSITIVE_HINTS)]

    # Prediction column: prefer name hints, else a binary column late in the file.
    prediction = next((c for c in cols if _match(c, PREDICTION_HINTS)), None)

    # Outcome column: prefer name hints, but don't reuse the prediction column.
    outcome = next((c for c in cols
                    if _match(c, OUTCOME_HINTS) and c != prediction), None)

    return {
        "all_columns": cols,
        "suggested_sensitive": sensitive,
        "suggested_outcome": outcome,
        "suggested_prediction": prediction,
    }
