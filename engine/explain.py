"""
explain.py
----------
Per-decision explanations in plain language.

Approach
  A transparent surrogate (standardised logistic regression) is trained to
  reproduce the audited model's decisions from the non-sensitive features.
  For one record, each feature's contribution is its coefficient times the
  record's standardised value, i.e. how far that feature pushed the decision
  compared with a typical record. One-hot columns are folded back into the
  original feature so explanations read "workclass = Private", not
  "workclass_Private".

  Fidelity (how often the surrogate agrees with the model) is always reported,
  so users know how far to trust the explanation.

  Counterfactuals search for the smallest single-feature change (numeric
  values within the observed range, or another category) that flips the
  surrogate's decision. Sensitive attributes are never used as levers.

  A proxy check flags features strongly associated with a sensitive attribute
  (for example "relationship" with "sex"), because a model can discriminate
  through proxies even when the sensitive column is excluded.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data_loader import clean_features, to_binary

MAX_CATEGORIES = 30          # rarer categories are folded into "other"
MAX_TRAIN_ROWS = 20000       # cap surrogate training size for speed
PROXY_THRESHOLD = 0.5


def _label(feature: str) -> str:
    return feature.replace("_", " ").strip()


def _fmt(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.2f}".rstrip("0").rstrip(".")
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


def _limit_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            top = out[c].value_counts().index[:MAX_CATEGORIES]
            out[c] = out[c].where(out[c].isin(top), "other")
    return out


class Explainer:
    """Surrogate model fitted once per audit and reused for any record."""

    def __init__(self, df: pd.DataFrame, feature_cols: List[str], pred_col: str):
        self.feature_cols = list(feature_cols)
        self.raw = _limit_categories(clean_features(df[self.feature_cols])).reset_index(drop=True)
        self.y = to_binary(df[pred_col]).to_numpy()
        if len(set(self.y)) < 2:
            raise ValueError("the model gave the same decision to every record")

        self.numeric = [c for c in self.feature_cols
                        if pd.api.types.is_numeric_dtype(self.raw[c])]
        encoded = pd.get_dummies(self.raw, drop_first=False).astype(float)
        self.columns = list(encoded.columns)
        self.owner = {}  # encoded column -> original feature
        for col in self.columns:
            self.owner[col] = col if col in self.numeric else max(
                (f for f in self.feature_cols
                 if f not in self.numeric and col.startswith(f + "_")), key=len)

        self.scaler = StandardScaler().fit(encoded.to_numpy())
        X = self.scaler.transform(encoded.to_numpy())
        rng = np.random.default_rng(0)
        idx = (rng.choice(len(X), MAX_TRAIN_ROWS, replace=False)
               if len(X) > MAX_TRAIN_ROWS else np.arange(len(X)))
        self.model = LogisticRegression(max_iter=2000, C=1.0).fit(X[idx], self.y[idx])
        self.X = X
        self.surrogate_pred = self.model.predict(X)
        self.fidelity = float((self.surrogate_pred == self.y).mean())

    # ------------------------------------------------------------------
    def _encode(self, row: pd.DataFrame) -> np.ndarray:
        enc = pd.get_dummies(row, drop_first=False).astype(float)
        enc = enc.reindex(columns=self.columns, fill_value=0.0)
        return self.scaler.transform(enc.to_numpy())

    def _contributions(self, x: np.ndarray) -> Dict[str, float]:
        per_col = self.model.coef_[0] * x
        out: Dict[str, float] = {}
        for col, value in zip(self.columns, per_col):
            out[self.owner[col]] = out.get(self.owner[col], 0.0) + float(value)
        return out

    def global_importance(self, top_k: int = 8) -> List[dict]:
        per_col = np.abs(self.model.coef_[0] * self.X).mean(axis=0)
        agg: Dict[str, float] = {}
        for col, value in zip(self.columns, per_col):
            agg[self.owner[col]] = agg.get(self.owner[col], 0.0) + float(value)
        total = sum(agg.values()) or 1.0
        ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [{"feature": f, "label": _label(f), "share": round(v / total, 3)}
                for f, v in ranked]

    # ------------------------------------------------------------------
    def explain(self, index: int, top_k: int = 4) -> dict:
        if not 0 <= index < len(self.raw):
            raise IndexError(f"record {index} does not exist (0 to {len(self.raw) - 1})")
        row = self.raw.iloc[[index]]
        x = self.X[index]
        decision = int(self.y[index])
        verb = "approved" if decision == 1 else "rejected"
        contrib = self._contributions(x)
        ranked = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top = [kv for kv in ranked if abs(kv[1]) > 1e-9][:top_k]

        factors = [{
            "feature": f, "label": _label(f), "value": _fmt(row.iloc[0][f]),
            "effect": "helped" if v > 0 else "hurt", "weight": round(abs(v), 3),
        } for f, v in top]

        drivers = [f for f in factors if (f["effect"] == "helped") == (decision == 1)]
        against = [f for f in factors if f not in drivers]
        reason = " and ".join(f"{f['label']} ({f['value']})" for f in drivers[:2])
        sentence = f"The model {verb} this record"
        sentence += (f", mainly because of {reason}, compared with a typical record."
                     if reason else ".")
        if against:
            effect = "counted in its favour" if decision == 0 else "counted against it"
            sentence += (f" {against[0]['label'].capitalize()} ({against[0]['value']}) "
                         f"{effect}, but not enough to change the outcome.")

        agrees = bool(self.surrogate_pred[index] == decision)
        return {
            "index": int(index),
            "decision": verb,
            "plain_language": sentence,
            "factors": factors,
            "counterfactuals": self.counterfactuals(index, decision) if agrees else [],
            "surrogate_agrees": agrees,
            "fidelity": round(self.fidelity, 3),
            "caveat": None if agrees else (
                "The simplified explanation model disagrees with the audited model "
                "for this record, so the reasons above are less reliable."),
        }

    def counterfactuals(self, index: int, decision: int, limit: int = 3) -> List[dict]:
        """Smallest single-feature changes that flip the surrogate decision."""
        target = 1 - decision
        base = self.raw.iloc[[index]].copy()
        options = []
        for f in self.feature_cols:
            current = base.iloc[0][f]
            if f in self.numeric:
                col = self.raw[f]
                lo, hi = float(col.quantile(0.01)), float(col.quantile(0.99))
                spread = (hi - lo) or 1.0
                for direction in (1, -1):
                    for step in np.linspace(0.05, 1.0, 20):
                        new = float(current) + direction * step * spread
                        if new < lo or new > hi:
                            break
                        if pd.api.types.is_integer_dtype(col):
                            new = int(round(new))
                        trial = base.copy()
                        trial[f] = new
                        if int(self.model.predict(self._encode(trial))[0]) == target:
                            options.append((step, f, current, new))
                            break
            else:
                for value in self.raw[f].value_counts().index[:MAX_CATEGORIES]:
                    if value == current:
                        continue
                    trial = base.copy()
                    trial[f] = value
                    if int(self.model.predict(self._encode(trial))[0]) == target:
                        options.append((0.5, f, current, value))  # rank after small numeric moves
                        break
        options.sort(key=lambda o: o[0])
        outcome = "approved" if target == 1 else "rejected"
        result = []
        for _, f, old, new in options[:limit]:
            if f in self.numeric:
                verb = "increased" if float(new) > float(old) else "reduced"
                text = (f"If {_label(f)} were {verb} from {_fmt(old)} to {_fmt(new)}, "
                        f"the decision would likely be {outcome}.")
            else:
                text = (f"If {_label(f)} were '{new}' instead of '{old}', "
                        f"the decision would likely be {outcome}.")
            result.append({"feature": f, "from": _fmt(old), "to": _fmt(new), "text": text})
        return result

    def sample_records(self, limit: int = 3) -> dict:
        approved = [int(i) for i in np.flatnonzero(self.y == 1)[:limit]]
        rejected = [int(i) for i in np.flatnonzero(self.y == 0)[:limit]]
        return {"approved": approved, "rejected": rejected}

    def summary(self) -> dict:
        return {"fidelity": round(self.fidelity, 3),
                "records": int(len(self.raw)),
                "features": [_label(f) for f in self.feature_cols],
                "global_importance": self.global_importance(),
                "samples": self.sample_records()}


# --------------------------------------------------------------------------
# proxy detection
# --------------------------------------------------------------------------
def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b).to_numpy().astype(float)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    n = table.sum()
    expected = table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / n
    chi2 = ((table - expected) ** 2 / np.where(expected == 0, 1, expected)).sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def find_proxies(df: pd.DataFrame, sensitive: List[str], features: List[str],
                 threshold: float = PROXY_THRESHOLD) -> List[dict]:
    """Features strongly associated (Cramer's V) with a sensitive attribute."""
    sample = df.sample(min(len(df), 20000), random_state=0) if len(df) > 20000 else df
    found = []
    for s in sensitive:
        sens = sample[s].astype(str)
        for f in features:
            col = sample[f]
            if pd.api.types.is_numeric_dtype(col) and col.nunique() > 10:
                col = pd.qcut(col, q=5, duplicates="drop").astype(str)
            else:
                col = col.astype(str)
            v = _cramers_v(sens, col)
            if v >= threshold:
                found.append({"feature": f, "sensitive": s, "strength": round(v, 2)})
    return sorted(found, key=lambda p: p["strength"], reverse=True)


def build_explainer(df: pd.DataFrame, feature_cols: List[str], pred_col: str) -> Explainer:
    return Explainer(df, feature_cols, pred_col)


def explain_row(explainer: Explainer, index: int) -> dict:
    return explainer.explain(index)
