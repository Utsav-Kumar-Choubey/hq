"""
explain.py
----------
Per-decision explanations in plain language.

Full SHAP is powerful but heavy and slow for a live demo. For a tabular model
we get a very usable explanation from a transparent surrogate: fit a quick
logistic regression on the features and read off each feature's contribution
for one individual (coefficient * that person's standardized value). Positive
contributions pushed toward "yes", negative toward "no".

We then turn the top contributions into a sentence a non-expert can read, and
produce a simple counterfactual ("lower the loan amount to flip the decision").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data_loader import to_binary, clean_features


def _prep_features(df: pd.DataFrame, feature_cols: list[str]):
    """One-hot encode categoricals and standardize; return X, names, scaler."""
    X = pd.get_dummies(clean_features(df[feature_cols]), drop_first=True)
    names = list(X.columns)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.astype(float))
    return Xs, names, X


def build_explainer(df: pd.DataFrame, feature_cols: list[str], pred_col: str):
    """Fit the surrogate model once; reuse it to explain many individuals."""
    y = to_binary(df[pred_col])
    Xs, names, X_raw = _prep_features(df, feature_cols)
    model = LogisticRegression(max_iter=1000)
    model.fit(Xs, y)
    return {"model": model, "names": names, "Xs": Xs,
            "X_raw": X_raw, "y": np.asarray(y)}


def explain_row(explainer: dict, index: int, top_k: int = 4) -> dict:
    """Explain a single individual's decision."""
    model, names = explainer["model"], explainer["names"]
    row = explainer["Xs"][index]
    contributions = model.coef_[0] * row       # per-feature push
    decision = int(model.predict([row])[0])

    order = np.argsort(np.abs(contributions))[::-1][:top_k]
    factors = []
    for i in order:
        factors.append({
            "feature": names[i],
            "effect": "helped" if contributions[i] > 0 else "hurt",
            "weight": round(float(abs(contributions[i])), 3),
        })

    helped = [f["feature"] for f in factors if f["effect"] == "helped"]
    hurt = [f["feature"] for f in factors if f["effect"] == "hurt"]
    verb = "approved" if decision == 1 else "rejected"
    if decision == 1:
        reason = f"mainly because {helped[0]} was favourable" if helped else ""
    else:
        reason = f"mainly because {hurt[0]} counted against it" if hurt else ""
    plain = f"This decision was {verb} {reason}.".replace("  ", " ").strip()

    return {
        "index": int(index),
        "decision": verb,
        "plain_language": plain,
        "factors": factors,
        "counterfactual": _counterfactual(explainer, index, decision),
    }


def _counterfactual(explainer: dict, index: int, decision: int) -> str:
    """A simple 'what would flip it' hint based on the strongest factor."""
    model, names = explainer["model"], explainer["names"]
    row = explainer["Xs"][index].copy()
    contributions = model.coef_[0] * row
    # The feature most responsible for the current decision.
    key = int(np.argmax(np.abs(contributions)))
    direction = "increase" if decision == 0 else "reduce"
    return (f"To change this outcome, {direction} "
            f"'{names[key]}' - it was the strongest driver of the decision.")
