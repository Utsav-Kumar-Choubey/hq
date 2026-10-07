"""
fairness.py
-----------
Group fairness metrics for single attributes and intersectional subgroups.

People are split into groups by one or more sensitive attributes (for example
"Female" or, intersectionally, "Female, 45plus"). For each group we compute:

  selection_rate      share of the group that received the positive decision
  true_positive_rate  share of truly qualified people who were approved (TPR)
  false_positive_rate share of unqualified people who were approved (FPR)
  precision           share of approvals that were truly qualified (PPV)

Each group is then compared with a reference group (the best-off group of
adequate size) using the selected primary metric:

  demographic_parity  selection-rate ratio        pass if >= 0.80 (four-fifths rule)
  equal_opportunity   TPR ratio                   pass if >= 0.80
  equalized_odds      max(|TPR gap|, |FPR gap|)   pass if <= 0.10
  predictive_parity   precision ratio             pass if >= 0.80

Groups smaller than MIN_GROUP_SIZE are reported with low confidence; a failing
low-confidence group is marked "review" rather than counted as a firm failure.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional

import pandas as pd

from .data_loader import prepare_sensitive, to_binary

RATIO_THRESHOLD = 0.80
GAP_THRESHOLD = 0.10
MIN_GROUP_SIZE = 100
MIN_REFERENCE_SIZE = MIN_GROUP_SIZE   # tiny groups make unreliable benchmarks

METRICS: Dict[str, dict] = {
    "demographic_parity": {
        "name": "Demographic parity",
        "kind": "ratio", "field": "selection_rate", "needs_outcome": False,
        "question": "Are groups selected at similar rates?",
        "threshold": "Ratio to the reference group of at least 0.80 (four-fifths rule)",
    },
    "equal_opportunity": {
        "name": "Equal opportunity",
        "kind": "ratio", "field": "true_positive_rate", "needs_outcome": True,
        "question": "Are truly qualified people approved at similar rates?",
        "threshold": "True-positive-rate ratio of at least 0.80",
    },
    "equalized_odds": {
        "name": "Equalized odds",
        "kind": "gap", "field": None, "needs_outcome": True,
        "question": "Are both error types (misses and false alarms) similar?",
        "threshold": "Largest TPR or FPR gap of at most 0.10",
    },
    "predictive_parity": {
        "name": "Predictive parity",
        "kind": "ratio", "field": "precision", "needs_outcome": True,
        "question": "Does a positive decision mean the same for every group?",
        "threshold": "Precision ratio of at least 0.80",
    },
}


# --------------------------------------------------------------------------
# per-group statistics
# --------------------------------------------------------------------------
def _rate(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator == 0 else round(numerator / denominator, 4)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> List[float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _group_stats(frame: pd.DataFrame, pred: str, outcome: Optional[str]) -> dict:
    n = len(frame)
    selected = int(frame[pred].sum())
    stats = {
        "size": int(n),
        "selection_rate": _rate(selected, n),
        "selection_ci": wilson_interval(selected, n),
        "true_positive_rate": None, "false_positive_rate": None,
        "precision": None, "accuracy": None, "base_rate": None,
    }
    if outcome:
        y, p = frame[outcome], frame[pred]
        tp = int(((y == 1) & (p == 1)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        tn = int(((y == 0) & (p == 0)).sum())
        stats.update({
            "true_positive_rate": _rate(tp, tp + fn),
            "false_positive_rate": _rate(fp, fp + tn),
            "precision": _rate(tp, tp + fp),
            "accuracy": _rate(tp + tn, n),
            "base_rate": _rate(tp + fn, n),
        })
    return stats


def _reference(rows: List[dict], field: str) -> Optional[dict]:
    """Best-off group on `field`, preferring groups of adequate size."""
    candidates = [r for r in rows if r[field] is not None]
    sized = [r for r in candidates if r["size"] >= MIN_REFERENCE_SIZE]
    pool = sized or candidates
    return max(pool, key=lambda r: r[field]) if pool else None


def _ratio(value: Optional[float], ref: Optional[float]) -> Optional[float]:
    if value is None or not ref:
        return None
    return round(value / ref, 3)


def _compliance(metric_id: str, value: Optional[float]) -> Optional[float]:
    """Map a metric value to 0..1, where 1 means the threshold is met."""
    if value is None:
        return None
    if METRICS[metric_id]["kind"] == "ratio":
        return round(min(1.0, value / RATIO_THRESHOLD), 4)
    if value <= GAP_THRESHOLD:
        return 1.0
    return round(max(0.0, 1 - (value - GAP_THRESHOLD) / 0.30), 4)


def _passes(metric_id: str, value: Optional[float]) -> Optional[bool]:
    if value is None:
        return None
    if METRICS[metric_id]["kind"] == "ratio":
        return value >= RATIO_THRESHOLD
    return value <= GAP_THRESHOLD


def _analyze(df: pd.DataFrame, group_cols: List[str], pred: str,
             outcome: Optional[str], primary: str) -> List[dict]:
    rows = []
    for key, frame in df.groupby(group_cols, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        row = {
            "group": ", ".join(str(k) for k in key),
            "grouping": " x ".join(group_cols),
            "attributes": dict(zip(group_cols, [str(k) for k in key])),
        }
        row.update(_group_stats(frame, pred, outcome))
        rows.append(row)
    if not rows:
        return rows

    refs = {f: _reference(rows, f) for f in
            ("selection_rate", "true_positive_rate", "precision")}
    eo_ref = refs["true_positive_rate"]

    for r in rows:
        dp = _ratio(r["selection_rate"], refs["selection_rate"]["selection_rate"])
        eo = (_ratio(r["true_positive_rate"], eo_ref["true_positive_rate"])
              if eo_ref else None)
        pp = (_ratio(r["precision"], refs["precision"]["precision"])
              if refs["precision"] else None)
        odds = None
        if eo_ref and r["true_positive_rate"] is not None \
                and r["false_positive_rate"] is not None \
                and eo_ref["false_positive_rate"] is not None:
            odds = round(max(abs(r["true_positive_rate"] - eo_ref["true_positive_rate"]),
                             abs(r["false_positive_rate"] - eo_ref["false_positive_rate"])), 3)
        values = {"demographic_parity": dp, "equal_opportunity": eo,
                  "equalized_odds": odds, "predictive_parity": pp}

        r["dp_ratio"] = dp
        r["eo_ratio"] = eo
        r["pp_ratio"] = pp
        r["odds_gap"] = odds
        r["metrics"] = {m: {"value": v, "passes": _passes(m, v)}
                        for m, v in values.items()}
        r["primary_value"] = values[primary]
        r["compliance"] = _compliance(primary, values[primary])
        r["confidence"] = "high" if r["size"] >= MIN_GROUP_SIZE else "low"
        ref_field = METRICS[primary]["field"] or "true_positive_rate"
        r["is_reference"] = r is refs[ref_field]

        ok = _passes(primary, values[primary])
        if ok is None:
            r["status"] = "not_applicable"
        elif ok:
            r["status"] = "pass"
        else:
            r["status"] = "fail" if r["confidence"] == "high" else "review"
    return sorted(rows, key=lambda r: (r["selection_rate"] or 0), reverse=True)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def resolve_metric(requested: Optional[str], has_outcome: bool) -> dict:
    """Pick the primary metric, falling back when ground truth is missing."""
    metric = requested if requested in METRICS else "demographic_parity"
    note = None
    if METRICS[metric]["needs_outcome"] and not has_outcome:
        note = (f"{METRICS[metric]['name']} requires the actual outcome column; "
                "demographic parity was used instead.")
        metric = "demographic_parity"
    return {"id": metric, **METRICS[metric], "note": note}


def audit_fairness(df: pd.DataFrame, sensitive: List[str], pred_col: str,
                   outcome_col: Optional[str] = None, intersectional: bool = True,
                   metric: Optional[str] = None) -> dict:
    """
    Run the fairness audit.

    Returns:
      metric          the primary metric used (and any fallback note)
      single          {attribute: [group rows]}
      intersectional  group rows for every combination of 2+ attributes
      summary         counts, worst group and an overall 0-100 score
    """
    chosen = resolve_metric(metric, bool(outcome_col))
    work = prepare_sensitive(df, sensitive)
    work[pred_col] = to_binary(work[pred_col])
    if outcome_col:
        work[outcome_col] = to_binary(work[outcome_col])

    single = {a: _analyze(work, [a], pred_col, outcome_col, chosen["id"])
              for a in sensitive}

    inter: List[dict] = []
    if intersectional and len(sensitive) >= 2:
        for size in range(2, len(sensitive) + 1):
            for combo in itertools.combinations(sensitive, size):
                inter.extend(_analyze(work, list(combo), pred_col,
                                      outcome_col, chosen["id"]))

    all_rows = [r for rows in single.values() for r in rows] + inter
    scored = [r for r in all_rows
              if r["compliance"] is not None and r["confidence"] == "high"]
    scored = scored or [r for r in all_rows if r["compliance"] is not None]
    if scored:
        mean_c = sum(r["compliance"] for r in scored) / len(scored)
        worst_c = min(r["compliance"] for r in scored)
        score = int(round(100 * (0.5 * mean_c + 0.5 * worst_c)))
    else:
        score = 100

    flagged = [r for r in all_rows if r["status"] == "fail"]
    review = [r for r in all_rows if r["status"] == "review"]
    worst = min(scored, key=lambda r: r["compliance"]) if scored else None
    dp_values = [r["dp_ratio"] for r in all_rows if r["dp_ratio"] is not None]

    overall = {"selection_rate": round(float(work[pred_col].mean()), 4)}
    if outcome_col:
        overall["accuracy"] = round(float((work[pred_col] == work[outcome_col]).mean()), 4)
        overall["base_rate"] = round(float(work[outcome_col].mean()), 4)

    return {
        "metric": chosen,
        "single": single,
        "intersectional": inter,
        "overall": overall,
        "summary": {
            "groups_tested": len(all_rows),
            "groups_flagged": len(flagged),
            "groups_review": len(review),
            "small_sample_groups": sum(1 for r in all_rows if r["confidence"] == "low"),
            "worst_ratio": round(min(dp_values), 3) if dp_values else None,
            "worst_group": worst["group"] if worst else None,
            "worst_value": worst["primary_value"] if worst else None,
            "fairness_score": score,
            "verdict": ("pass" if not flagged and not review else
                        "needs_attention" if len(flagged) <= 2 else "fail"),
        },
    }
