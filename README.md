# FairLens — Bias Auditing & Explainability Toolkit

**Problem 17 · AI / Machine Learning · Track 02: Model Governance**

A web tool where non-experts upload a dataset with model predictions and get a
**bias, drift and explainability audit** plus an auto-generated,
compliance-style report with recommendations.

---

## What it does (maps to the 4 required deliverables)

| # | Requirement | Where it lives |
|---|-------------|----------------|
| 1 | Upload data + predictions → bias, drift & explainability audit | `engine/`, `server.py`, web UI |
| 2 | Intersectional fairness metrics + guidance on which metric to use | `engine/fairness.py`, metric advisor (`engine/audit.py`) |
| 3 | Per-decision explanations in plain language | `engine/explain.py` |
| 4 | Auto-generated compliance report with recommendations | `engine/report.py`, "Download report" button |

**Public model audits (required deliverable):**
- **UCI Adult Income** — RandomForest. Result: severe gender bias (women
  selected at 0.35× the male rate). `demo/audit_adult_income.py`
- **Statlog German Credit** — LogisticRegression. Result: mostly fair (84/100),
  flags small-sample groups. `demo/audit_german_credit.py`

Sample reports: `demo/adult_income_report.html`, `demo/german_credit_report.html`.

---

## Architecture

```
Browser (index.html + style.css + app.js)
        │  fetch()  ▲ JSON
        ▼           │
FastAPI server (server.py)
        │
Audit engine (engine/)
  data_loader.py  → read CSV, detect columns
  fairness.py     → single + intersectional metrics, 80% rule, equal opportunity
  drift.py        → PSI + KS test
  explain.py      → plain-language per-decision explanations + counterfactuals
  report.py       → recommendations + downloadable HTML report
  audit.py        → orchestrates all of the above; metric advisor
```

- **Front end:** plain HTML/CSS/JavaScript (no framework, no build step).
- **Back end:** Python + FastAPI.
- **ML/stats:** pandas, scikit-learn, scipy, numpy.

---

## Run it locally

```bash
# 1. install dependencies (uv is pre-installed; or use pip)
uv sync

# 2. start the server
bash run_server.sh         # serves http://127.0.0.1:8000

# 3. open the page
#    http://127.0.0.1:8000  → upload a CSV from demo/ and click "Run audit"
```

### Try it with the demo data
Upload `demo/adult_income_predictions.csv`, pick:
- sensitive attributes: `sex`, `race`
- decision column: `predicted_high_income`
- outcome column: `income_over_50k`

Click **Run audit** → you'll see the real bias found in that dataset.

---

## Reproduce the public-model audits

```bash
uv run python demo/audit_adult_income.py     # downloads UCI Adult, trains, audits
uv run python demo/audit_german_credit.py    # downloads German Credit, trains, audits
```

---

## Tests (prove the engine works)

```bash
uv run python test_engine.py    # engine on synthetic data with known bias
uv run python test_server.py    # all FastAPI endpoints end-to-end
```

---

## How the fairness metrics work (for the demo Q&A)

- **Demographic parity / 80% rule:** each group's selection rate ÷ the top
  group's rate. Below **0.80** → flagged (the legal "four-fifths" rule).
- **Equal opportunity:** among people who *truly* qualified, are groups
  approved at similar rates? (needs the true-outcome column)
- **Intersectional:** the same math applied to *combinations* of attributes
  (e.g. "Female, Black"), which often reveals bias that single-axis checks miss.
- **Drift (PSI):** `< 0.10` stable, `0.10–0.25` moderate, `> 0.25` significant.
- **Metric advisor:** recommends the metric that fits the use case (hiring →
  demographic parity, lending → equal opportunity, triage → equalized odds,
  scoring → predictive parity).

> Explanations describe how the model *behaved*; they do not by themselves
> prove a decision was fair — always read the group results too.
