"""Shared helpers for the public-model demo audits."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def make_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    """One-hot encode text columns, scale numbers, then fit the estimator."""
    categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in X.columns if c not in categorical]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric),
    ])
    return Pipeline([("prep", pre), ("model", estimator)])


def save_model(pipeline: Pipeline, name: str) -> Path:
    path = DEMO / "models" / f"{name}.joblib"
    path.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def write_report(result: dict, name: str) -> Path:
    path = DEMO / f"{name}_report.html"
    path.write_text(result["report"]["html"], encoding="utf-8")
    return path


def print_summary(result: dict, attributes) -> None:
    s = result["fairness"]["summary"]
    m = result["fairness"]["metric"]
    print(f"\nPrimary metric: {m['name']}")
    print(f"Fairness score: {s['fairness_score']}/100  verdict: {s['verdict']}  "
          f"flagged: {s['groups_flagged']}  review: {s['groups_review']}")
    for attr in attributes:
        print(f"By {attr}:")
        for r in result["fairness"]["single"][attr]:
            print(f"  {r['group']:<22} n={r['size']:<6} rate={r['selection_rate']:.0%} "
                  f"{m['name']}={r['primary_value']} -> {r['status']}")
    print("Drift:", result["drift"]["overall_band"])
    print("Explanation fidelity:", result["explainability"].get("fidelity"))
    print("Top recommendations:")
    for r in result["recommendations"][:4]:
        print(f"  [{r['priority']}] {r['text']}")
