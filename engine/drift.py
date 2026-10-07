"""
drift.py
--------
Detects whether the live data has "drifted" away from the training/reference
data. If the world changes, a model trained on old data can quietly go stale.

Two complementary tests:
  - PSI (Population Stability Index): bins each feature and measures how much
    the distribution shifted. Industry rule of thumb:
        < 0.10  stable, 0.10-0.25 moderate, > 0.25 significant.
  - KS test (Kolmogorov-Smirnov): a statistical test for numeric features;
    a small p-value means the two distributions differ significantly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _psi_for_feature(reference: pd.Series, current: pd.Series,
                     bins: int = 10) -> float:
    ref = reference.dropna()
    cur = current.dropna()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    if pd.api.types.is_numeric_dtype(ref):
        # Bin by quantiles of the reference so each bin has ~equal mass.
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(ref, quantiles))
        if len(edges) < 2:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts = np.histogram(ref, bins=edges)[0]
        cur_counts = np.histogram(cur, bins=edges)[0]
    else:
        # Categorical: one bin per category seen in either dataset.
        cats = sorted(set(ref.unique()) | set(cur.unique()))
        ref_counts = np.array([(ref == c).sum() for c in cats])
        cur_counts = np.array([(cur == c).sum() for c in cats])

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    # Avoid log(0) / divide-by-zero.
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(round(psi, 4))


def _psi_band(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


def audit_drift(reference: pd.DataFrame, current: pd.DataFrame,
                features: list[str] | None = None) -> dict:
    """Compare reference vs current data across shared features."""
    if features is None:
        features = [c for c in reference.columns if c in current.columns]

    per_feature = []
    for col in features:
        if col not in reference.columns or col not in current.columns:
            continue
        psi = _psi_for_feature(reference[col], current[col])
        row = {"feature": col, "psi": psi, "band": _psi_band(psi)}

        # Add a KS p-value for numeric features as a second opinion.
        if pd.api.types.is_numeric_dtype(reference[col]):
            try:
                ks = stats.ks_2samp(reference[col].dropna(),
                                    current[col].dropna())
                row["ks_pvalue"] = float(round(ks.pvalue, 4))
                row["ks_significant"] = bool(ks.pvalue < 0.05)
            except Exception:
                row["ks_pvalue"] = None
        per_feature.append(row)

    per_feature.sort(key=lambda r: r["psi"], reverse=True)
    worst = per_feature[0]["psi"] if per_feature else 0.0
    overall = _psi_band(worst)
    return {
        "features": per_feature,
        "overall_band": overall,
        "worst_psi": worst,
        "n_drifted": sum(1 for r in per_feature if r["band"] != "stable"),
    }
