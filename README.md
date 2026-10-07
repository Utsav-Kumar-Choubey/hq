# FairLens: Bias Auditing and Explainability Toolkit

**Problem 17 · AI / Machine Learning · Track 02: Model Governance**

FairLens lets non-experts upload a dataset together with a trained model or its
predictions, and receive a **bias, drift and explainability audit** with a
compliance-style report and prioritised recommendations.

---

## Requirements coverage

| Problem statement requirement | How FairLens meets it |
|---|---|
| **1.** Upload a model or predictions and a dataset; receive a bias, drift and explainability audit | Dataset CSV plus either a predictions column, a separate predictions CSV, or a model file (`.pkl`, `.joblib`, `.onnx`). Drift is measured against an optional training-time reference (PSI and KS test). |
| **2.** Intersectional fairness metrics and guidance on which metric fits which situation | Demographic parity, equal opportunity, equalized odds and predictive parity for every attribute and every combination of two or more attributes, with 95% confidence intervals. A metric advisor, a decision guide and a use-case selector choose the primary metric. |
| **3.** Per-decision explanations in plain language | Pick any record to see a plain-English reason, the main factors, and counterfactuals ("if X were Y, the decision would likely change"). Fidelity of the explanation model is stated, and proxy features are flagged. |
| **4.** Auto-generated compliance-style report with recommendations | Ten-section report (executive summary, scope, method, single and intersectional results, drift, explainability, limitations, recommendations, audit trail and sign-off) as HTML, PDF (print) and JSON evidence. |

| Expected deliverable | Location |
|---|---|
| Working web tool | `index.html`, `app.js`, `style.css`, `server.py`, `engine/` |
| Audit of at least two public models | UCI Adult Income (random forest) and Statlog German Credit (logistic regression): `demo/audit_adult_income.py`, `demo/audit_german_credit.py` |
| Sample report | `demo/adult_income_report.{html,pdf}`, `demo/german_credit_report.{html,pdf}`, `demo/sample_loan_report.{html,pdf}` |

### Public model audit results

| Model | Primary metric | Score | Key finding |
|---|---|---|---|
| UCI Adult Income, random forest (sex and race excluded from inputs) | Equal opportunity | 51/100, fail | 5 groups fail; qualified Black women are approved at 0.61 of the reference rate. `relationship` acts as a proxy for sex (Cramér's V 0.64), so excluding the sensitive columns did not remove the bias. |
| German Credit, logistic regression | Equal opportunity | 100/100, pass | No group with adequate sample size fails; several groups have fewer than 100 records and are reported as low confidence. |

The contrast shows that the tool distinguishes a biased model from a fair one.

---

## Quick start

Requires Python 3.9 or newer.

```bash
git clone https://github.com/Utsav-Kumar-Choubey/hq.git
cd hq
python3 -m pip install -r requirements.txt
python3 -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Keep the terminal open while you use the tool;
press `Ctrl+C` to stop the server.

---

## Using the tool

1. **Dataset (field 1):** upload a CSV. The column controls fill in automatically.
2. **Model or predictions (field 2, optional):** a model file, or a predictions CSV
   with the same number of rows. Leave empty if the dataset already contains the
   model's decisions.
3. **Actual outcome (field 3):** the real result (for example repaid or hired).
   Optional, but required for equal opportunity, equalized odds and predictive parity.
4. **Model decision (field 4):** the model's yes/no decision or probability.
5. **Sensitive attributes (field 5):** tick the attributes to audit. Continuous
   attributes such as age are grouped into quartiles automatically.
6. **Use case and metric (fields 6 and 7):** choose the use case and let FairLens
   pick the metric, or choose it yourself. The metric advisor explains the choice.
7. **Reference dataset (field 8, optional):** the training data, for the drift check.
8. Click **Run audit**, then review the results, explanations and report.

### Demo files

| Scenario | Field 1 | Field 2 | Field 3 | Field 4 | Field 5 | Field 6 | Field 8 |
|---|---|---|---|---|---|---|---|
| Biased public model (predictions) | `demo/adult_income_predictions.csv` | | `income_over_50k` | `predicted_high_income` | `sex`, `race` | Lending | `demo/adult_income_reference.csv` |
| Fair public model (model upload) | `demo/german_credit_test.csv` | `demo/models/german_credit.joblib`* | `good_credit` | `model_prediction` | `sex`, `age_band` | Lending | `demo/german_credit_reference.csv` |
| Drift demonstration | `demo/sample_loan_predictions.csv` | | `loan_repaid` | `predicted_approval` | `gender`, `age_band` | Lending | `demo/sample_loan_reference.csv` |
| Simple hiring example | `demo/sample_hiring.csv` | | `truly_qualified` | `predicted_hire` | `gender`, `age_band`, `race` | Screening or hiring | |

The scores in the results table above use the *Lending* use case (equal
opportunity). With *Screening or hiring* (demographic parity), the Adult model
scores far lower (about 9/100) because its selection rates differ even more
than its true-positive rates.

Columns you do not tick as sensitive are treated as ordinary model features and
can appear in explanations, so tick every protected attribute you want excluded.

\*Model files are created on your machine, so they always match your installed
scikit-learn version. Run `python3 demo/audit_german_credit.py` (or
`demo/audit_adult_income.py`) once; it downloads the public data, trains the
model, saves it to `demo/models/`, and regenerates the CSVs and sample report.

---

## How the audit works

**Fairness.** Records are grouped by each sensitive attribute and by every
combination of two or more attributes. Each group is compared with a reference
group: the best-off group with at least 100 records.

| Metric | Question | Passes when | Recommended for |
|---|---|---|---|
| Demographic parity | Are groups selected at similar rates? | Ratio ≥ 0.80 (four-fifths rule) | Hiring, screening |
| Equal opportunity | Are truly qualified people approved at similar rates? | TPR ratio ≥ 0.80 | Lending, admissions |
| Equalized odds | Are misses and false alarms similar? | TPR and FPR gaps ≤ 0.10 | Triage, risk flags |
| Predictive parity | Does a positive decision mean the same for every group? | Precision ratio ≥ 0.80 | Scores, rankings |

Groups with fewer than 100 records are low confidence; if they fail they are
marked **Review** rather than **Fail**. The fairness score (0 to 100) weights the
worst group most heavily and is capped below 70 whenever a group fails.

**Drift.** Population Stability Index per feature (below 0.10 stable, 0.10 to 0.25
moderate, above 0.25 significant), with a Kolmogorov-Smirnov test for numeric features.

**Explanations.** A transparent surrogate model (standardised logistic
regression) is trained to reproduce the audited model's decisions, and its
agreement rate (fidelity) is reported. Each feature's contribution to a decision
is shown in plain language. Counterfactuals search for the smallest realistic
single-feature change that would flip the decision; sensitive attributes are
never suggested. Features strongly associated with a sensitive attribute
(Cramér's V ≥ 0.5) are flagged as possible proxies.

**Limitations.** Metrics can conflict when base rates differ; outcome-based
metrics inherit any bias in historical outcomes; explanations approximate the
model; proxy detection shows association, not reliance. The report states these
explicitly and supports, but does not replace, legal and domain review.

---

## Architecture

```
Browser: index.html + style.css + app.js (plain JavaScript)
   │ fetch()
FastAPI: server.py
   │
engine/
  data_loader.py   CSV parsing, role suggestions, binarisation, binning
  model_loader.py  load and apply .pkl / .joblib / .onnx models
  fairness.py      group metrics, intersectional analysis, score
  drift.py         PSI and KS test
  explain.py       surrogate explanations, counterfactuals, proxies
  report.py        recommendations and the compliance report
  audit.py         orchestration and metric guidance
```

### API

| Method and path | Purpose |
|---|---|
| `POST /upload` | Read the dataset (and optional model or predictions) and suggest column roles |
| `POST /audit` | Run the full audit and return JSON |
| `GET /explain/{index}` | Explain one record of the last audit |
| `GET /advisor/{use_case}`, `GET /metrics` | Metric recommendation and catalogue |
| `GET /report/view`, `GET /report/download`, `GET /report.json` | Report in the browser (print to PDF), as HTML, or as JSON evidence |
| `GET /health` | Liveness check |

---

## Development

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
```

Continuous integration (GitHub Actions) runs the test suite on Python 3.9, 3.11
and 3.12 and checks the JavaScript syntax on every pull request.

## Security and privacy

- Files are processed in memory by the local server and are not stored or sent elsewhere.
- Loading a pickled model executes code contained in the file. Only upload
  models from a trusted source.
- All values from uploaded files are HTML-escaped before display.
