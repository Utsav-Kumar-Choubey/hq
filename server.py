"""
server.py
---------
FastAPI server that exposes the FairLens audit engine to the web page.

Endpoints:
  GET  /                -> the FairLens front end (index.html)
  GET  /health          -> liveness check
  POST /upload          -> read a CSV, return its columns and suggested roles
  POST /audit           -> run the full audit and return JSON results
  GET  /explain/{row}   -> plain-language explanation for one record
  GET  /advisor/{case}  -> recommended fairness metric for a use case
  GET  /metrics         -> metric definitions and selection guide
  POST /report          -> the latest compliance report as an HTML file
  GET  /report/view     -> the report in the browser (print to PDF)
  GET  /report.json     -> full audit results as JSON evidence

Run:  python3 -m uvicorn server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from engine import data_loader, model_loader
from engine.audit import advise_metric, metric_catalogue, run_full_audit

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="FairLens Bias Audit API", version="1.1.0")

# Latest generated report. Adequate for a single-user local tool; a multi-user
# deployment would key reports by session.
_LAST_REPORT_HTML = {"html": None, "json": None}
_LAST_EXPLAINER = {"explainer": None}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _read_csv_upload(file: UploadFile, label: str = "dataset",
                     min_columns: int = 2) -> pd.DataFrame:
    name = file.filename or label
    if not name.lower().endswith(".csv"):
        raise HTTPException(400, f"The {label} must be a .csv file (received '{name}').")
    raw = file.file.read()
    if len(raw) > data_loader.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"The {label} exceeds the 50 MB limit.")
    try:
        return data_loader.load_csv(io.BytesIO(raw), min_columns)
    except ValueError as exc:
        raise HTTPException(400, f"Could not read '{name}': {exc}")


def _attach_model_or_predictions(df: pd.DataFrame, upload: Optional[UploadFile],
                                 exclude: List[Optional[str]]) -> pd.DataFrame:
    """Apply an uploaded model, or join an uploaded predictions CSV, to df."""
    if upload is None or not upload.filename:
        return df
    name = upload.filename
    if name.lower().endswith(".csv"):
        preds = _read_csv_upload(upload, "predictions file", min_columns=1)
        try:
            return model_loader.merge_predictions(df, preds)
        except model_loader.ModelError as exc:
            raise HTTPException(400, str(exc))
    if not model_loader.is_model_file(name):
        raise HTTPException(400, "The model must be a .pkl, .joblib or .onnx file, "
                                 f"or a predictions .csv (received '{name}').")
    raw = upload.file.read()
    if len(raw) > data_loader.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The model file exceeds the 50 MB limit.")
    try:
        model = model_loader.load_model(raw, name)
        return model_loader.apply_model(model, df, exclude)
    except model_loader.ModelError as exc:
        raise HTTPException(400, str(exc))


def _parse_list(value: str) -> List[str]:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if str(v).strip()]
    except (TypeError, ValueError):
        pass
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _validate_roles(df: pd.DataFrame, sensitive: List[str], prediction_col: str,
                    outcome_col: Optional[str]) -> None:
    if not prediction_col:
        raise HTTPException(400, "Select the column that holds the model decision.")
    if not sensitive:
        raise HTTPException(400, "Select at least one sensitive attribute to audit.")
    requested = sensitive + [prediction_col] + ([outcome_col] if outcome_col else [])
    missing = [c for c in requested if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found in the CSV: {missing}")
    if prediction_col in sensitive or (outcome_col and outcome_col in sensitive):
        raise HTTPException(400, "A sensitive attribute cannot also be the "
                                 "decision or outcome column.")
    if outcome_col and outcome_col == prediction_col:
        raise HTTPException(400, "The outcome and decision columns must differ.")
    if not data_loader.is_binary_like(df[prediction_col]):
        raise HTTPException(400, f"'{prediction_col}' must be a yes/no decision or a "
                                 "probability between 0 and 1.")
    if outcome_col and not data_loader.is_binary_like(df[outcome_col]):
        raise HTTPException(400, f"'{outcome_col}' must be a yes/no outcome.")


# --------------------------------------------------------------------------
# static front end
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse(BASE_DIR / "index.html", media_type="text/html")


@app.get("/style.css")
def style():
    return FileResponse(BASE_DIR / "style.css", media_type="text/css")


@app.get("/app.js")
def appjs():
    return FileResponse(BASE_DIR / "app.js", media_type="application/javascript")


@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.post("/upload")
async def upload(dataset: UploadFile = File(...),
                 model: Optional[UploadFile] = File(None)):
    """Read a CSV (plus an optional model or predictions file) and return the
    columns and suggested roles for the form."""
    df = _read_csv_upload(dataset)
    hints = data_loader.suggest_roles(df)
    df = _attach_model_or_predictions(df, model, [hints["suggested_outcome"],
                                                  hints["suggested_prediction"]])
    suggestions = data_loader.suggest_roles(df)
    if model_loader.PRED_COL in df.columns:
        suggestions["suggested_prediction"] = model_loader.PRED_COL
    return JSONResponse({
        "filename": dataset.filename,
        "rows": int(len(df)),
        "columns": data_loader.describe_columns(df),
        "suggestions": suggestions,
        "model_applied": model_loader.PRED_COL in df.columns,
    })


@app.post("/audit")
async def audit(
    dataset: UploadFile = File(...),
    sensitive: str = Form(...),
    prediction_col: str = Form(""),
    outcome_col: str = Form(""),
    intersectional: bool = Form(True),
    model_name: str = Form("Uploaded model"),
    use_case: str = Form(""),
    metric: str = Form(""),
    reference: Optional[UploadFile] = File(None),
    model: Optional[UploadFile] = File(None),
):
    """Run the bias, drift and explainability audit and return JSON."""
    df = _read_csv_upload(dataset)
    sensitive_list = _parse_list(sensitive)
    outcome = outcome_col or None
    df = _attach_model_or_predictions(df, model, [outcome, prediction_col])
    if not prediction_col and model_loader.PRED_COL in df.columns:
        prediction_col = model_loader.PRED_COL
    _validate_roles(df, sensitive_list, prediction_col, outcome)

    ref_df = None
    if reference is not None and reference.filename:
        ref_df = _read_csv_upload(reference, "reference dataset")

    result = run_full_audit(
        df, sensitive=sensitive_list, pred_col=prediction_col,
        outcome_col=outcome, intersectional=intersectional,
        reference=ref_df, model_name=model_name or "Uploaded model",
        metric=metric or None, use_case=use_case or None, return_explainer=True)
    result, explainer = result

    _LAST_EXPLAINER["explainer"] = explainer
    _LAST_REPORT_HTML["html"] = result["report"].pop("html", None)
    _LAST_REPORT_HTML["json"] = result
    return JSONResponse(result)


@app.get("/explain/{index}")
def explain(index: int):
    """Plain-language explanation for one record of the last audited dataset."""
    explainer = _LAST_EXPLAINER["explainer"]
    if explainer is None:
        raise HTTPException(404, "No explanations yet. Run an audit first.")
    try:
        return JSONResponse(explainer.explain(index))
    except IndexError as exc:
        raise HTTPException(404, str(exc))


@app.get("/advisor/{usecase}")
def advisor(usecase: str):
    """Return the recommended fairness metric for a use case."""
    return JSONResponse(advise_metric(usecase))


@app.get("/metrics")
def metrics():
    """All supported fairness metrics, use-case advice and the decision guide."""
    return JSONResponse(metric_catalogue())


def _require_report() -> str:
    if not _LAST_REPORT_HTML["html"]:
        raise HTTPException(404, "No report yet. Run an audit first.")
    return _LAST_REPORT_HTML["html"]


@app.post("/report")
@app.get("/report/download")
def report():
    """The latest report as a downloadable HTML file."""
    return Response(
        content=_require_report(), media_type="text/html",
        headers={"Content-Disposition": "attachment; filename=fairlens_report.html"})


@app.get("/report/view", response_class=HTMLResponse)
def report_view(print_dialog: bool = False):
    """The latest report shown in the browser; print_dialog=true opens the
    print dialog so it can be saved as a PDF."""
    page = _require_report()
    if print_dialog:
        page = page.replace("</body>", "<script>window.addEventListener('load', "
                                       "() => setTimeout(() => window.print(), 300));"
                                       "</script></body>")
    return HTMLResponse(page)


@app.get("/report.json")
def report_json():
    """Full audit results as JSON evidence (for audit files or regulators)."""
    _require_report()
    return Response(
        content=json.dumps(_LAST_REPORT_HTML["json"], indent=2, default=str),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=fairlens_audit.json"})
