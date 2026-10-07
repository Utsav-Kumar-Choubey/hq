"""
fairness.py
-----------
The core bias math (single-group and intersectional).

Key idea in plain words:
  We split people into GROUPS by their sensitive attribute(s) -- e.g. "women"
  or, intersectionally, "women aged 45+". For each group we measure how often
  the model said "yes" (selection rate) and, when we know the true outcome,
  how good it was at finding the people who truly qualified (true positive rate).

Metrics implemented:
  - Demographic parity (the "80% / four-fifths rule"):
      Is each group selected at a similar rate to the best-off group?
  - Equal opportunity:
      Among people who truly qualified, are groups approved at similar rates?

Both reduce to a simple ratio against the top group. A ratio below 0.8 is the
classic legal "adverse impact" flag.
"""

from __future__ import annotations

import itertools
import pandas as pd

FOUR_FIFTHS = 0.8          # legal "adverse impact" threshold
MIN_GROUP_SIZE = 100       # below this, we flag "small sample / low confidence"


def _to_binary(series: pd.Series) -> pd.Series:
    """
    Coerce a yes/no-ish column into 1/0. Handles 'yes'/'no', 'true'/'false',
    'approved'/'denied', and already-numeric 1/0 columns.
    """
    if pd.api.types.is_numeric_dtype(series):
        return (series > 0).astype(int)
    positives = {"yes", "y", "true", "1", "approved", "hired",
                 "repaid", "positive", "good", "accept", "accepted"}
    return series.astype(str).str.strip().str.lower().isin(positives).astype(int)


def _group_rate(frame: pd.DataFrame, pred_col: str) -> float:
    """Selection rate = share of the group the model said 'yes' to."""
    return float(frame[pred_col].mean())


def _true_positive_rate(frame: pd.DataFrame, pred_col: str,
                        outcome_col: str) -> float | None:
    """Among people who truly qualified (outcome==1), share the model approved."""
    qualified = frame[frame[outcome_col] == 1]
    if len(qualified) == 0:
        return None
    return float(qualified[pred_col].mean())


def _status(ratio: float, size: int) -> str:
    if size < MIN_GROUP_SIZE:
        return "small_sample"
    return "pass" if ratio >= FOUR_FIFTHS else "fail"


def _analyze_grouping(df: pd.DataFrame, group_cols: list[str],
                      pred_col: str, outcome_col: str | None) -> list[dict]:
    """Compute per-group stats for one grouping (single or intersectional)."""
    rows = []
    for key, frame in df.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        label = ", ".join(f"{v}" for v in key)
        rows.append({
            "group": label,
            "attributes": dict(zip(group_cols, [str(k) for k in key])),
            "size": int(len(frame)),
            "selection_rate": round(_group_rate(frame, pred_col), 4),
            "true_positive_rate": (
                None if outcome_col is None
                else _tpr_rounded(frame, pred_col, outcome_col)
            ),
        })

    # Compare every group to the best-performing group (the reference).
    if rows:
        top_sel = max(r["selection_rate"] for r in rows) or 1e-9
        tprs = [r["true_positive_rate"] for r in rows
                if r["true_positive_rate"] is not None]
        top_tpr = max(tprs) if tprs else None
        for r in rows:
            r["dp_ratio"] = round(r["selection_rate"] / top_sel, 3)
            if top_tpr and r["true_positive_rate"] is not None and top_tpr > 0:
                r["eo_ratio"] = round(r["true_positive_rate"] / top_tpr, 3)
            else:
                r["eo_ratio"] = None
            r["status"] = _status(r["dp_ratio"], r["size"])
    return sorted(rows, key=lambda r: r["selection_rate"], reverse=True)


def _tpr_rounded(frame, pred_col, outcome_col):
    v = _true_positive_rate(frame, pred_col, outcome_col)
    return None if v is None else round(v, 4)


def audit_fairness(df: pd.DataFrame, sensitive: list[str], pred_col: str,
                   outcome_col: str | None = None,
                   intersectional: bool = True) -> dict:
    """
    Run the full fairness audit.

    Returns a dict with:
      - single: {attribute -> [group rows]}
      - intersectional: [group rows] for the combined attributes (if >=2)
      - summary: counts of groups tested / flagged, and an overall 0-100 score
    """
    work = df.copy()
    work[pred_col] = _to_binary(work[pred_col])
    if outcome_col:
        work[outcome_col] = _to_binary(work[outcome_col])

    single = {attr: _analyze_grouping(work, [attr], pred_col, outcome_col)
              for attr in sensitive}

    inter = []
    if intersectional and len(sensitive) >= 2:
        inter = _analyze_grouping(work, sensitive, pred_col, outcome_col)

    # Build the summary / overall score.
    all_rows = [r for rows in single.values() for r in rows] + inter
    tested = len(all_rows)
    flagged = sum(1 for r in all_rows if r["status"] == "fail")
    small = sum(1 for r in all_rows if r["status"] == "small_sample")
    worst = min((r["dp_ratio"] for r in all_rows), default=1.0)

    # Score: start at 100, lose points for each failing group and for the
    # severity of the worst gap. Clamped to 0-100.
    score = 100 - flagged * 8 - max(0, (FOUR_FIFTHS - worst)) * 100
    score = int(max(0, min(100, round(score))))

    return {
        "single": single,
        "intersectional": inter,
        "summary": {
            "groups_tested": tested,
            "groups_flagged": flagged,
            "small_sample_groups": small,
            "worst_ratio": round(worst, 3),
            "fairness_score": score,
            "verdict": ("pass" if flagged == 0 else
                        "needs_attention" if flagged <= 2 else "fail"),
        },
    }
