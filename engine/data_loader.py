"""
data_loader.py
--------------
Reads an uploaded CSV and describes its columns so the front end can offer
sensible defaults. This module performs no fairness calculations.

Responsibilities:
  - load a CSV safely and normalise column names
  - describe every column (type, cardinality, examples)
  - suggest which columns are sensitive attributes, the true outcome and the
    model decision
  - convert yes/no style columns to 0/1 and bin continuous sensitive
    attributes so that groups remain interpretable
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

SENSITIVE_HINTS = [
    "gender", "sex", "race", "ethnic", "age", "disab",
    "religion", "nationality", "marital", "citizen",
]
OUTCOME_HINTS = ["outcome", "label", "target", "actual", "true", "qualified",
                 "repaid", "default", "churn", "over_50k", "good_credit",
                 "y_true"]
PREDICTION_HINTS = ["pred", "score", "proba", "y_pred", "decision"]

POSITIVE_LABELS = {"yes", "y", "true", "1", "1.0", "approved", "approve",
                   "hired", "hire", "repaid", "positive", "good", "accept",
                   "accepted", ">50k", "pass"}

MAX_GROUPS_PER_ATTRIBUTE = 12   # above this a numeric attribute is binned
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def load_csv(path_or_buffer) -> pd.DataFrame:
    """Read a CSV into a DataFrame. Accepts a file path or a file-like object."""
    try:
        df = pd.read_csv(path_or_buffer)
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty.")
    except Exception as exc:  # malformed CSV, wrong encoding, binary file...
        raise ValueError(f"The file could not be parsed as CSV ({exc}).")
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError("The uploaded CSV has a header but no rows.")
    if len(df.columns) < 2:
        raise ValueError("The CSV must contain at least two columns.")
    return df


def to_binary(series: pd.Series) -> pd.Series:
    """
    Convert a decision/outcome column to 0/1.

    Rules, in order:
      - numeric columns that are already 0/1 are kept as-is
      - numeric columns within [0, 1] are treated as probabilities (>= 0.5)
      - other numeric columns with exactly two values: larger value = 1
      - other numeric columns: > 0 means positive
      - text columns: matched against a list of positive labels
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        s = series.astype(float)
        values = set(s.dropna().unique())
        if values <= {0.0, 1.0}:
            return s.fillna(0).astype(int)
        if s.min() >= 0 and s.max() <= 1:
            return (s >= 0.5).astype(int)
        if len(values) == 2:
            return (s == max(values)).astype(int)
        return (s > 0).astype(int)
    return (series.astype(str).str.strip().str.lower()
            .isin(POSITIVE_LABELS).astype(int))


def is_binary_like(series: pd.Series) -> bool:
    """True if the column can be interpreted as a yes/no decision."""
    s = series.dropna()
    if s.nunique() == 2:
        return True
    return (pd.api.types.is_numeric_dtype(s) and len(s) > 0
            and s.min() >= 0 and s.max() <= 1)


def prepare_sensitive(df: pd.DataFrame, sensitive: List[str]) -> pd.DataFrame:
    """
    Return a copy where each sensitive attribute is a readable categorical.
    Continuous attributes with many values (e.g. raw age) are binned into
    quartile bands so the audit produces a small number of meaningful groups.
    """
    out = df.copy()
    for col in sensitive:
        s = out[col]
        if (pd.api.types.is_numeric_dtype(s)
                and s.nunique() > MAX_GROUPS_PER_ATTRIBUTE):
            try:
                binned = pd.qcut(s, q=4, duplicates="drop")
                out[col] = binned.apply(
                    lambda iv: "missing" if pd.isna(iv)
                    else f"{iv.left:g}-{iv.right:g}").astype(str)
            except ValueError:
                out[col] = s.astype(str)
        else:
            out[col] = s.astype(str).fillna("missing")
        out[col] = out[col].replace({"nan": "missing"})
    return out


def describe_columns(df: pd.DataFrame) -> List[dict]:
    """Return a plain-language description of every column for the UI."""
    out = []
    for col in df.columns:
        s = df[col]
        out.append({
            "name": col,
            "dtype": "number" if pd.api.types.is_numeric_dtype(s) else "category",
            "unique_values": int(s.dropna().nunique()),
            "missing": int(s.isna().sum()),
            "is_binary": bool(is_binary_like(s)),
            "examples": [str(v) for v in s.dropna().unique()[:4]],
        })
    return out


def _match(col: str, hints: List[str]) -> bool:
    low = col.lower()
    return any(h in low for h in hints)


def suggest_roles(df: pd.DataFrame) -> dict:
    """Suggest defaults for the form. The user always confirms the choice."""
    cols = list(df.columns)
    binary_cols = [c for c in cols if is_binary_like(df[c])]

    prediction: Optional[str] = next(
        (c for c in binary_cols if _match(c, PREDICTION_HINTS)), None)
    outcome: Optional[str] = next(
        (c for c in binary_cols
         if c != prediction and _match(c, OUTCOME_HINTS)), None)
    if outcome is None:  # fall back to any other binary column
        outcome = next((c for c in binary_cols
                        if c != prediction
                        and not _match(c, SENSITIVE_HINTS)), None)

    reserved = {prediction, outcome}
    sensitive = [c for c in cols
                 if c not in reserved and _match(c, SENSITIVE_HINTS)]

    return {
        "all_columns": cols,
        "suggested_sensitive": sensitive,
        "suggested_outcome": outcome,
        "suggested_prediction": prediction,
    }


def feature_columns(df: pd.DataFrame, exclude: List[Optional[str]]) -> List[str]:
    """Columns usable as model features (everything except excluded roles and
    obvious identifiers)."""
    skip = {c for c in exclude if c}
    feats = []
    for c in df.columns:
        if c in skip:
            continue
        low = c.lower()
        if low in {"id", "index"} or low.endswith("_id"):
            continue
        feats.append(c)
    return feats


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numbers, 'missing' for categories."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            median = out[c].median()
            out[c] = out[c].fillna(0 if np.isnan(median) else median)
        else:
            out[c] = out[c].astype(str).replace({"nan": "missing"})
    return out
